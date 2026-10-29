import { T } from "./theme";
import { PHASE, STRIPE_L, deviceStep, inFlight, mod, covered, tokensOf } from "./ringModel";
import { RING_SRC, ZIGZAG_SRC, STRIPE_SRC, UTILS_SRC, ZIGZAG_UTILS_SRC, STRIPE_UTILS_SRC } from "./codeSources";

// Per-variant copy, code highlighting and live code comments for the ring family
// (ring / zigzag / stripe). Shared by RingView and CompareView.

export const COPY = {
  ring: {
    eyebrow: "SEQUENCE PARALLELISM · RING ATTENTION",
    title: "Ring attention: Q stays, KV goes around",
    intro: "Each GPU keeps its query block and passes its key/value block to the next GPU, $N$ times. The send is posted <i>before</i> the kernel, so it hides behind compute. The causal mask makes the load uneven.",
    src: ["ring_flash_attn.py", RING_SRC, 430],
    utils: ["utils.py", UTILS_SRC],
    matrixNote: (N) => `Token resolution, $S = ${STRIPE_L * N}$ tokens. GPU 0's rows see almost nothing; GPU ${N - 1}'s see everything. White outline = the GPU you follow.`,
  },
  zigzag: {
    eyebrow: "SEQUENCE PARALLELISM · ZIG-ZAG RING ATTENTION",
    title: "Zig-zag ring attention: balance the causal triangle",
    intro: "Same ring and hops, but the sequence is cut into $2N$ chunks and GPU $r$ takes chunks $r$ and $2N-1-r$: one early, one late. Every GPU now does the same work at each step.",
    src: ["zigzag_ring_flash_attn.py", ZIGZAG_SRC, 470],
    utils: ["extract_local + utils.py", ZIGZAG_UTILS_SRC],
    matrixNote: (N) => `Token resolution, $S = ${STRIPE_L * N}$ tokens. Each GPU has one chunk near the top (few keys) and one near the bottom (many keys).`,
  },
  stripe: {
    eyebrow: "SEQUENCE PARALLELISM · STRIPE ATTENTION",
    title: "Stripe attention: deal tokens round-robin",
    intro: "Same ring and hops, but tokens are dealt round-robin: token $t$ goes to GPU $t \\bmod N$. Each GPU holds a stripe spread over the whole sequence, so every KV block is about half visible. A block from a later rank needs the causal mask shifted by one token.",
    src: ["stripe_flash_attn.py", STRIPE_SRC, 470],
    utils: ["extract_local + utils.py", STRIPE_UTILS_SRC],
    matrixNote: (N) => `Token resolution: $S = ${STRIPE_L * N}$ tokens, row $t$ belongs to GPU $t \\bmod ${N}$. Every GPU's rows are spread evenly down the triangle.`,
  },
};

// tag -> color for the main function and utils listings at this phase.
export function codeFocus(variant, N, phase, r) {
  const { type, s } = phase;
  const C = (t) => PHASE[t].color;
  const main = {}, utils = {};
  if (type === "init") {
    main.init = C("init");
    utils.init = C("init");
    utils.layout = C("init");
    return { main, utils };
  }
  if (type === "final") {
    main.final = C("final");
    return { main, utils };
  }
  main.loop = `dim:${T.soft}`;
  const d = deviceStep(variant, N, r, s);
  if (type === "send") {
    main.send = C("send");
    utils.send = C("send");
  } else if (type === "wait") {
    main.wait = C("wait");
    utils.wait = C("wait");
  } else if (type === "compute") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (variant === "ring") {
      main.cond = d.active ? C("compute") : T.bad;
      if (d.active) main.comp = C("compute");
    } else {
      main[`cond:${d.branch}`] = C("compute");
      main[`comp:${d.branch}`] = C("compute");
      main.fwd = `dim:${C("compute")}`;
    }
  } else if (type === "merge") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (d.active) {
      if (variant === "ring") main.merge = C("merge");
      else main[`merge:${d.branch}`] = C("merge");
      utils.upd = C("merge");
      if (s === 0) utils.first = C("merge");
      else {
        utils[d.branch === "hi" ? "slice" : "else"] = C("merge");
        utils.core = C("merge");
      }
    } else if (variant === "ring") main.cond = T.bad;
  }
  return { main, utils };
}

// Live comments for the attention code at this phase, for GPU r: what each
// executed line evaluates to, with explicit token ids. Returns notes for the
// main forward function and for utils.py (see CodeBlock `notes`).
export function codeNotes(variant, N, phase, r) {
  const { main, utils } = phaseNotes(variant, N, phase, r);
  const sh = shapeNotes(variant, tokensOf(variant, N, r).length);
  return { main: [...main, ...sh.main], utils: [...utils, ...sh.utils] };
}

// Permanent comments on every squeeze / unsqueeze / transpose of the lse,
// spelled out as einops. Layouts follow flash-attn: out is [B, L, H, D],
// the kernel's lse is [B, H, L]; the running lse is kept as [B, L, H, 1] so it
// broadcasts against out.
function shapeNotes(variant, L) {
  const n = (m, t, after) => ({ m, t, after });
  // what the flash-attn kernel computes, written as einsum (S = keys in the block)
  const kernel = (m, after, qn = "q", kn = "k", vn = "v") => [
    n(m, `scores = einsum("blhd,bshd->bhls", ${qn}, ${kn}) * softmax_scale   # q·kᵀ → [B, H, L, S]`, after),
    n(m, `p = softmax(scores + mask, dim=-1);  block_lse = logsumexp(scores + mask, dim=-1)   # [B, H, L]`, after),
    n(m, `block_out = einsum("bhls,bshd->blhd", p, ${vn})   # p·v → [B, L, H, D]`, after),
  ];
  const kernels = variant === "stripe"
    ? [...kernel("block_out, block_lse = _flash_attn_forward(**params)"), ...kernel("block_out, block_lse = _flash_attn_forward(**params)", "else:", "q[:, 1:]", "k[:, :-1]", "v[:, :-1]")]
    : kernel("block_out, block_lse = _flash_attn_forward");
  return {
    main: [
      ...kernels,
      n("lse = lse.squeeze(dim=-1).transpose(1, 2)", `[B, L=${L}, H, 1] → squeeze(-1) drops the size-1 last dim → [B, ${L}, H] → transpose(1, 2) → [B, H, ${L}]  ≡  rearrange(lse, "b l h 1 -> b h l")`),
    ],
    utils: [
      n("block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)", `kernel lse [B, H, L] → transpose(-2, -1) → [B, L, H] → unsqueeze(-1) adds a size-1 dim → [B, L, H, 1]  ≡  rearrange(block_lse, "b h l -> b l h 1")`),
      n("out = out - F.sigmoid(block_lse - lse) * (out - block_out)", `σ is [B, L, H, 1]: the size-1 dim broadcasts over D  ≡  out − einsum("blh,blhd->blhd", σ[..., 0], out − block_out)`),
      n("lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)", `first block: [B, H, L] → [B, L, H, 1]  ≡  rearrange(block_lse, "b h l -> b l h 1")`, "if out is None:"),
    ],
  };
}

function phaseNotes(variant, N, phase, r) {
  const { type, s } = phase;
  const n = (m, t, after) => ({ m, t, after });
  const nxt = mod(r + 1, N), prv = mod(r - 1, N);
  const tk = (j) => tokensOf(variant, N, j);
  const T = (ts) => `t[${ts.join(",")}]`;
  const qT = tk(r);
  const L = qT.length;
  const main = [n("comm = RingComm(process_group)", `rank ${r} of ${N}: sends to ${nxt}, receives from ${prv}`)];
  const utils = [
    n("self.send_rank = (self.rank + 1)", `(${r} + 1) % ${N} = ${nxt}`),
    n("self.recv_rank = (self.rank - 1)", `(${r} − 1) % ${N} = ${prv}`),
  ];
  if (variant === "zigzag") {
    main.push(n("block_seq_len = q.shape[1] // 2", `${L} // 2 = ${L / 2}`));
    main.push(n("q1 = q[:, block_seq_len:]", `late half of q = ${T(qT.slice(L / 2))}`));
  }
  if (type === "init") {
    main.push(n("out = None", `q = ${T(qT)}, k, v = ${T(qT)}; nothing merged yet`));
    return { main, utils };
  }
  if (type === "final") {
    const js = covered(variant, N, r, phase);
    main.push(n("out = out.to(q.dtype)", `[${L}, H, D] for q ${T(qT)}, merged over KV from GPU${js.length > 1 ? "s" : ""} ${js.join(", ")}`));
    return { main, utils };
  }

  const d = deviceStep(variant, N, r, s);
  const j = d.j, jn = mod(r - s - 1, N);
  const kT = tk(j);
  const last = s === N - 1;
  main.push(n("for step in range(comm.world_size)", `step = ${s}  (k, v held: ${T(kT)} from GPU ${j})`));
  main.push(n("if step + 1 != comm.world_size", `${s} + 1 != ${N} → ${!last}${last ? ": last step, nothing to send" : ""}`));
  main.push(n("if step + 1 != comm.world_size", `${s} + 1 != ${N} → ${!last}${last ? ": nothing to wait for" : ""}`, "comm.send_recv_kv"));
  if (!last) {
    main.push(n("next_k, next_v = comm.send_recv_kv(k, v)", `send ${T(kT)} → GPU ${nxt}; receive ${T(tk(jn))} ← GPU ${prv}`));
    utils.push(n("send_op = dist.P2POp(", `isend ${T(kT)} → GPU ${nxt}`));
    utils.push(n("recv_op = dist.P2POp", `irecv ${T(tk(jn))} ← GPU ${prv}`));
  }
  const js = covered(variant, N, r, { type: "merge", s });
  const done = type === "merge" || type === "wait";
  const mergeText = s === 0 ? "first block: out, lse = block_out, block_lse" : `${done ? "out now covers" : "after this merge, out covers"} KV from GPUs ${js.join(", ")}`;

  if (variant === "ring") {
    main.push(n("if not causal or step <= comm.rank", `${s} <= ${r} → ${d.active}${d.active ? "" : `: keys ${T(kT)} are all after queries ${T(qT)}, skip`}`));
    if (d.active) {
      main.push(n('"causal": causal and step == 0', `step == 0 → ${s === 0}${s === 0 ? " (diagonal block)" : " (all keys earlier, no mask)"}`));
      main.push(n("block_out, block_lse = _flash_attn_forward", `q ${T(qT)} × k ${T(kT)}`));
      main.push(n("out, lse = update_out_and_lse", mergeText));
    }
  } else if (variant === "zigzag") {
    main.push(n("if step == 0:", `${s} == 0 → ${s === 0}`));
    if (d.branch === "0") {
      main.push(n("forward(q, k, v, causal=True)", `q ${T(qT)} × k ${T(kT)}, causal`));
      main.push(n("out, lse = update_out_and_lse", mergeText, "if step == 0:"));
    } else {
      main.push(n("elif step <= comm.rank", `${s} <= ${r} → ${d.branch === "lo"}`));
      if (d.branch === "lo") {
        main.push(n("k0 = k[:, :block_seq_len]", `early half of k = ${T(kT.slice(0, L / 2))}`));
        main.push(n("forward(q, k0, v0, causal=False)", `q ${T(qT)} × k0 ${T(kT.slice(0, L / 2))}, no mask`));
        main.push(n("out, lse = update_out_and_lse", mergeText, "elif step <= comm.rank"));
      } else {
        main.push(n("forward(q1, k, v, causal=False)", `q1 ${T(qT.slice(L / 2))} × k ${T(kT)}, no mask`));
        main.push(n("out, lse = update_out_and_lse(", `${mergeText} (q1 rows only)`, "else:"));
        main.push(n("slice_=(slice(None), slice(block_seq_len, None))", `only rows of q1 ${T(qT.slice(L / 2))} change`));
        utils.push(n("slice_out, slice_lse = out[slice_]", `rows ${L / 2}: (q1)`));
      }
    }
  } else {
    main.push(n("if step <= comm.rank:", `${s} <= ${r} → ${d.branch === "lo"}`));
    if (d.branch === "lo") {
      main.push(n('"q": q,', T(qT)));
      main.push(n('"k": k,', T(kT)));
      main.push(n("block_out, block_lse = _flash_attn_forward(**params)", "causal: key p′ ≤ query p"));
      main.push(n("out, lse = update_out_and_lse(out, lse, block_out, block_lse)", mergeText));
    } else {
      main.push(n('"q": q[:, 1:],', `${T(qT.slice(1))} (drop ${qT[0]})`));
      main.push(n('"k": k[:, :-1],', `${T(kT.slice(0, -1))} (drop ${kT[kT.length - 1]})`));
      main.push(n("block_out, block_lse = _flash_attn_forward(**params)", "causal, shifted by one: key p′ ≤ p − 1", "else:"));
      main.push(n("slice(1, None)", `rows 1: (all but ${qT[0]})`, "else:"));
      utils.push(n("slice_out, slice_lse = out[slice_]", "rows 1:"));
    }
  }
  if (type === "merge" && d.active) {
    if (s === 0) utils.push(n("out = block_out.to(torch.float32)", "first block, just store it", "if out is None:"));
    else utils.push(n("out = out - F.sigmoid(block_lse - lse)", "reweight old out vs new block by their lse"));
  }
  if (type === "wait") {
    main.push(n("comm.wait()", `block until ${T(tk(jn))} has arrived from GPU ${prv}`));
    main.push(n("k, v = next_k, next_v", `k, v = ${T(tk(jn))}`));
    utils.push(n("req.wait()", "the isend and irecv for k and for v"));
  }
  return { main, utils };
}
