// Python sources shown in the side panel (ring-flash-attention). A trailing
// "#@tag tag" on a line is stripped before display and drives highlighting:
// the view lights every line whose tag is active for the current phase.

function parse(src) {
  return src
    .replace(/^\n/, "")
    .replace(/\n\s*$/, "")
    .split("\n")
    .map((line) => {
      const m = line.match(/^(.*?)\s*#@(.*)$/);
      return m ? { text: m[1], tags: m[2].trim().split(/\s+/) } : { text: line, tags: [] };
    });
}

export const RING_SRC = parse(String.raw`
def ring_flash_attn_forward(
    process_group, q, k, v, softmax_scale,
    dropout_p=0, causal=True, window_size=(-1, -1),
    alibi_slopes=None, deterministic=False,
):
    comm = RingComm(process_group)  #@init

    out = None  #@init
    lse = None  #@init
    next_k, next_v = None, None  #@init

    for step in range(comm.world_size):  #@loop
        if step + 1 != comm.world_size:  #@send
            next_k, next_v = comm.send_recv_kv(k, v)  #@send

        if not causal or step <= comm.rank:  #@cond
            params = get_default_args(_flash_attn_forward).copy()  #@comp
            params.update(  #@comp
                {  #@comp
                    "q": q,  #@comp
                    "k": k,  #@comp
                    "v": v,  #@comp
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,  #@comp
                    "causal": causal and step == 0,  #@comp causal
                    "window_size": window_size,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }  #@comp
            )  #@comp
            # (output unpacking differs across flash-attn versions)
            block_out, block_lse = _flash_attn_forward(**params)  #@comp
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)  #@merge

        if step + 1 != comm.world_size:  #@wait
            comm.wait()  #@wait
            k, v = next_k, next_v  #@wait

    out = out.to(q.dtype)  #@final
    lse = lse.squeeze(dim=-1).transpose(1, 2)  #@final
    return out, lse  #@final
`);

export const ZIGZAG_SRC = parse(String.raw`
def zigzag_ring_flash_attn_forward(
    process_group, q, k, v, softmax_scale,
    dropout_p=0, causal=True, window_size=(-1, -1),
    alibi_slopes=None, deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)  #@init

    block_seq_len = q.shape[1] // 2  #@init
    q1 = q[:, block_seq_len:]  #@init

    out = None  #@init
    lse = None  #@init
    next_k, next_v = None, None  #@init

    def forward(q, k, v, causal):  #@fwd
        params = get_default_args(_flash_attn_forward).copy()  #@fwd
        params.update(
            {
                "q": q,  #@fwd
                "k": k,  #@fwd
                "v": v,  #@fwd
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,  #@fwd
                "window_size": window_size,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        block_out, block_lse = _flash_attn_forward(**params)  #@fwd
        return block_out, block_lse  #@fwd

    for step in range(comm.world_size):  #@loop
        if step + 1 != comm.world_size:  #@send
            next_k, next_v = comm.send_recv_kv(k, v)  #@send

        if step == 0:  #@cond:0
            block_out, block_lse = forward(q, k, v, causal=True)  #@comp:0
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)  #@merge:0
        elif step <= comm.rank:  #@cond:lo
            k0 = k[:, :block_seq_len]  #@comp:lo
            v0 = v[:, :block_seq_len]  #@comp:lo
            block_out, block_lse = forward(q, k0, v0, causal=False)  #@comp:lo
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)  #@merge:lo
        else:  #@cond:hi
            block_out, block_lse = forward(q1, k, v, causal=False)  #@comp:hi
            out, lse = update_out_and_lse(  #@merge:hi
                out,  #@merge:hi
                lse,  #@merge:hi
                block_out,  #@merge:hi
                block_lse,  #@merge:hi
                slice_=(slice(None), slice(block_seq_len, None)),  #@merge:hi
            )  #@merge:hi

        if step + 1 != comm.world_size:  #@wait
            comm.wait()  #@wait
            k, v = next_k, next_v  #@wait

    out = out.to(q.dtype)  #@final
    lse = lse.squeeze(dim=-1).transpose(1, 2)  #@final
    return out, lse  #@final
`);

// From the repo's test utils: how each rank slices its zigzag shard.
const EXTRACT_LOCAL = String.raw`
def extract_local(value, rank, world_size, dim=1):  #@layout
    value_chunks = value.chunk(2 * world_size, dim=dim)  #@layout
    local_value = torch.cat(  #@layout
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim  #@layout
    )  #@layout
    return local_value.contiguous()  #@layout

`;

const UTILS = String.raw`
@torch.jit.script
def _update_out_and_lse(  #@core
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)  #@core
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  #@core

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)  #@core
    lse = lse - F.logsigmoid(lse - block_lse)  #@core

    return out, lse  #@core


def update_out_and_lse(  #@upd
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:  #@first
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)  #@first
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  #@first
    elif slice_ is not None:  #@slice
        slice_out, slice_lse = out[slice_], lse[slice_]  #@slice
        slice_out, slice_lse = _update_out_and_lse(  #@slice
            slice_out, slice_lse, block_out, block_lse  #@slice
        )  #@slice
        out[slice_], lse[slice_] = slice_out, slice_lse  #@slice
    else:  #@else
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)  #@else
    return out, lse  #@upd


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):  #@init
        self._process_group = process_group
        self._ops = []  #@init
        self.rank = dist.get_rank(self._process_group)  #@init
        self.world_size = dist.get_world_size(self._process_group)  #@init
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size  #@init
        self.recv_rank = (self.rank - 1) % self.world_size  #@init

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(  #@send
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:  #@send
            res = torch.empty_like(to_send)  #@send
        else:
            res = recv_tensor

        send_op = dist.P2POp(  #@send
            dist.isend, to_send, self.send_rank, group=self._process_group  #@send
        )  #@send
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)  #@send
        self._ops.append(send_op)  #@send
        self._ops.append(recv_op)  #@send
        return res  #@send

    def commit(self):  #@send
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)  #@send pending

    def wait(self):  #@wait
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:  #@wait
            req.wait()  #@wait
        self._reqs = None  #@wait
        self._ops = []  #@wait

    def send_recv_kv(  #@send
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)  #@send
        self.commit()  #@send
        return next_k, next_v  #@send
`;

export const UTILS_SRC = parse(UTILS);
export const ZIGZAG_UTILS_SRC = parse("# test utils: zigzag sharding of the full sequence\n" + EXTRACT_LOCAL.replace(/^\n/, "") + "# utils.py" + UTILS);

export const STRIPE_SRC = parse(String.raw`
def stripe_flash_attn_forward(
    process_group, q, k, v, softmax_scale,
    dropout_p=0, causal=True, window_size=(-1, -1),
    alibi_slopes=None, deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention"
    comm = RingComm(process_group)  #@init

    out = None  #@init
    lse = None  #@init
    next_k, next_v = None, None  #@init

    for step in range(comm.world_size):  #@loop
        if step + 1 != comm.world_size:  #@send
            next_k, next_v = comm.send_recv_kv(k, v)  #@send

        params = get_default_args(_flash_attn_forward).copy()
        if step <= comm.rank:  #@cond:lo
            params.update(  #@comp:lo
                {  #@comp:lo
                    "q": q,  #@comp:lo
                    "k": k,  #@comp:lo
                    "v": v,  #@comp:lo
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,  #@comp:lo
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }  #@comp:lo
            )  #@comp:lo
            # (window_size / output unpacking differ across flash-attn versions)
            block_out, block_lse = _flash_attn_forward(**params)  #@comp:lo
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)  #@merge:lo
        else:  #@cond:hi
            params.update(  #@comp:hi
                {  #@comp:hi
                    "q": q[:, 1:],  #@comp:hi
                    "k": k[:, :-1],  #@comp:hi
                    "v": v[:, :-1],  #@comp:hi
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,  #@comp:hi
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }  #@comp:hi
            )  #@comp:hi
            block_out, block_lse = _flash_attn_forward(**params)  #@comp:hi
            out, lse = update_out_and_lse(  #@merge:hi
                out, lse, block_out, block_lse, slice_=(slice(None), slice(1, None))  #@merge:hi
            )  #@merge:hi

        if step + 1 != comm.world_size:  #@wait
            comm.wait()  #@wait
            k, v = next_k, next_v  #@wait

    out = out.to(q.dtype)  #@final
    lse = lse.squeeze(dim=-1).transpose(1, 2)  #@final
    return out, lse  #@final
`);

// Illustrative: how a rank takes its stripe of the full sequence.
const STRIPE_LOCAL = String.raw`
def extract_local(value, rank, world_size, dim=1):  #@layout
    # token t -> rank t % world_size; local position p = t // world_size  #@layout
    return value[:, rank::world_size].contiguous()  #@layout

`;

export const STRIPE_UTILS_SRC = parse("# stripe sharding (illustrative)\n" + STRIPE_LOCAL.replace(/^\n/, "") + "# utils.py" + UTILS);

// Tree attention decode (Shyam et al. 2024), written with torch.distributed.
// The paper's JAX version is lax.pmax + lax.psum; tree_allreduce spells out
// the binomial tree those collectives run on.
export const TREE_SRC = parse(String.raw`
def tree_attn_decode(q, k, v, group=None):
    # q: [B, H, 1, d] (same new token on every rank)
    # k, v: this rank's shard of the KV cache, [B, H, S/N, d]
    local_out, local_lse = flash_attn_decode(q, k, v)  #@local

    # 1) global max of the local log-sum-exps: a shared shift for exp()
    lse_max = local_lse.clone()  #@max
    tree_allreduce(lse_max, op=torch.maximum, group=group)  #@max

    # 2) rescale this shard's numerator / denominator to that shift
    w = torch.exp(local_lse - lse_max).unsqueeze(-1)  #@rescale
    num = local_out * w  #@rescale
    den = w  #@rescale

    # 3) one fused sum-allreduce for (num, den)
    buf = torch.cat([num, den], dim=-1)  #@sum
    tree_allreduce(buf, op=torch.add, group=group)  #@sum
    num, den = buf[..., :-1], buf[..., -1:]  #@final
    return num / den  #@final
`);

export const TREE_AR_SRC = parse(String.raw`
def tree_allreduce(x, op, group=None):
    # in production: dist.all_reduce(x, op=ReduceOp.MAX / SUM)
    rank = dist.get_rank(group)  #@ar
    N = dist.get_world_size(group)  #@ar
    buf = torch.empty_like(x)  #@ar

    # reduce: log2(N) levels, partial results flow up to rank 0
    step = 1  #@up
    while step < N:  #@up
        if rank % (2 * step) == 0:  #@up_recv
            dist.recv(buf, src=rank + step, group=group)  #@up_recv
            x.copy_(op(x, buf))  #@up_recv
        elif rank % (2 * step) == step:  #@up_send
            dist.send(x, dst=rank - step, group=group)  #@up_send
        step *= 2  #@up

    # broadcast: walk back down the same tree
    step //= 2  #@down
    while step >= 1:  #@down
        if rank % (2 * step) == 0:  #@down_send
            dist.send(x, dst=rank + step, group=group)  #@down_send
        elif rank % (2 * step) == step:  #@down_recv
            dist.recv(x, src=rank - step, group=group)  #@down_recv
        step //= 2  #@down
    return x
`);
