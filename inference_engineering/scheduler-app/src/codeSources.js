// Python from tinyserve.py shown in the side panel (long comments trimmed).
// A trailing "#@tag tag" is stripped before display and drives highlighting:
// each trace event lights the lines whose tags it names.

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

export const KV_SRC = parse(String.raw`
class KVCacheManager:
    def _update_hashes(self, req: Request):
        bs = self.block_size
        while len(req.block_hashes) < req.num_tokens // bs:  #@uh_loop
            i = len(req.block_hashes)  #@uh_body
            parent = req.block_hashes[-1] if req.block_hashes else None  #@uh_body
            req.block_hashes.append(hash((parent, tuple(req.token_ids[i * bs:(i + 1) * bs]))))  #@uh_body

    def find_cached_prefix(self, req: Request):
        if not self.enable_caching:
            return []
        self._update_hashes(req)  #@fcp_hash
        max_blocks = (req.num_tokens - 1) // self.block_size  #@fcp_max
        hits = []  #@fcp_max
        for h in req.block_hashes[:max_blocks]:  #@fcp_loop
            b = self.hash_to_block.get(h)  #@fcp_loop
            if b is None:  #@fcp_miss fcp_hit
                break  #@fcp_miss
            hits.append(b)  #@fcp_hit
        return hits  #@fcp_ret

    def allocate(self, req, num_new_tokens, cached=()):
        total = (  #@al_need
            req.num_computed_tokens  #@al_need
            + len(cached) * self.block_size  #@al_need
            + num_new_tokens  #@al_need
        )  #@al_need
        total_needed_blocks = -(-total // self.block_size)  #@al_need al_tnb
        need = total_needed_blocks - len(req.block_ids) - len(cached)  #@al_need al_needv
        reclaim = sum(  #@al_need al_rec
            1 for b in cached if self.ref_cnt[b] == 0  #@al_need
        )  #@al_need
        if need + reclaim > self.num_free:  #@al_check
            return False  #@al_fail
        for b in cached:  #@al_claim
            if self.ref_cnt[b] == 0:  #@al_claim
                del self.free[b]  #@al_claim
            self.ref_cnt[b] += 1  #@al_claim
            req.block_ids.append(b)  #@al_claim
        for _ in range(max(need, 0)):  #@al_new
            b, _ = self.free.popitem(last=False)  #@al_new
            if self.block_hash[b] is not None:  #@al_evict
                self.hash_to_block.pop(self.block_hash[b], None)  #@al_evict
                self.block_hash[b] = None  #@al_evict
            self.ref_cnt[b] = 1  #@al_new
            req.block_ids.append(b)  #@al_new
        self._register_full_blocks(req, total)  #@al_reg
        return True  #@al_ok

    def _register_full_blocks(self, req, num_tokens):
        if not self.enable_caching:
            return
        self._update_hashes(req)  #@reg_hash
        for i in range(min(num_tokens, req.num_tokens) // self.block_size):  #@reg_loop
            b, h = req.block_ids[i], req.block_hashes[i]  #@reg_loop
            if self.block_hash[b] is None:  #@reg_set
                self.block_hash[b] = h  #@reg_set
                self.hash_to_block.setdefault(h, b)  #@reg_set

    def free_request(self, req: Request):
        for b in reversed(req.block_ids):  #@fr_loop
            self.ref_cnt[b] -= 1  #@fr_loop
            if self.ref_cnt[b] == 0:  #@fr_free
                self.free[b] = None  #@fr_free
        req.block_ids = []  #@fr_done
`);

export const SCHED_SRC = parse(String.raw`
class Scheduler:
    def schedule(self) -> list[ScheduledRequest]:
        budget = self.cfg.max_num_batched_tokens  #@sc_init
        plan: dict[str, int] = {}  #@sc_init
        preempted = False  #@sc_init

        # 1. running requests
        i = 0  #@sc_init
        while i < len(self.running) and budget > 0:  #@p1_loop
            req = self.running[i]  #@p1_req
            n = min(req.num_tokens - req.num_computed_tokens, budget)  #@p1_req p1_n
            while not self.kv.allocate(req, n):  #@p1_alloc
                victim = self.running.pop()  #@pre
                self.kv.free_request(victim)  #@pre_free
                victim.num_computed_tokens = 0  #@pre_after
                self.waiting.appendleft(victim)  #@pre_after
                budget += plan.pop(victim.request_id, 0)  #@pre_after
                preempted = True  #@pre_after
                if victim is req:  #@pre_self
                    break  #@pre_self
            if req not in self.running:  #@p1_gone
                break  #@p1_gone
            plan[req.request_id] = n  #@p1_plan
            budget -= n  #@p1_plan
            i += 1  #@p1_plan

        # 2. waiting requests
        while (not preempted and self.waiting and budget > 0  #@p2_loop
               and len(self.running) < self.cfg.max_num_seqs):  #@p2_loop
            req = self.waiting[0]  #@p2_req
            cached = self.kv.find_cached_prefix(req)  #@p2_cache
            n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)  #@p2_n
            if not self.kv.allocate(req, n, cached):  #@p2_alloc
                break  #@p2_fail
            self.waiting.popleft()  #@p2_admit
            req.num_computed_tokens = len(cached) * self.cfg.block_size  #@p2_admit
            req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)  #@p2_admit
            self.running.append(req)  #@p2_admit
            plan[req.request_id] = n  #@p2_admit
            budget -= n  #@p2_admit

        # 3. emit the batch in running order
        batch = []  #@emit
        for req in self.running:  #@emit
            if req.request_id in plan:  #@emit
                batch.append(ScheduledRequest(req, req.num_computed_tokens, plan[req.request_id]))  #@emit
                req.num_computed_tokens += plan[req.request_id]  #@emit
        return batch  #@emit

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason  #@fin
        if req in self.running:  #@fin
            self.running.remove(req)  #@fin
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)  #@fin_free

class Engine:
    def add_request(self, prompt, params, request_id=None):
        ...
        self.scheduler.add(req)  # waiting.append(req)  #@add

    def step(self):
        batch = self.scheduler.schedule()  #@st_sched
        if not batch:  #@st_empty
            return []  #@st_empty
        tokens, logits = self.runner.run(batch)  # writes K/V into pages  #@st_run
        for i, (s, tok) in enumerate(zip(batch, tokens)):  #@st_loop
            req = s.req  #@st_loop
            if not s.completes_prompt:  #@st_skip
                continue  #@st_skip
            req.token_ids.append(tok)  #@st_append
            n_out = len(req.output_ids)  #@st_append
            if tok in self.stop_ids and not req.params.ignore_eos:
                self.scheduler.finish(req, "stop")
            elif n_out >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:  #@st_len
                self.scheduler.finish(req, "length")  #@st_len
`);

export const demoSrc = (bs) => parse(String.raw`
def kv_cache_demo():
    B = ${bs}  #@d0
    mgr = KVCacheManager(num_blocks=8, block_size=B, enable_caching=True)  #@d0

    # 1. chunked prefill
    a = Request("A", [10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=4))  #@d1
    print("  hits", mgr.find_cached_prefix(a), mgr.allocate(a, 4))  #@d1a
    a.num_computed_tokens = 4  #@d1b
    print("  allocate +2", mgr.allocate(a, 2))  #@d1c
    a.num_computed_tokens = 6  #@d1d

    # 2. decode
    for t in (16, 17):  #@d2
        a.token_ids.append(t)  #@d2
        mgr.allocate(a, 1)  #@d2a
        a.num_computed_tokens += 1  #@d2b

    # 3. B shares first block only
    b = Request("B", [10, 11, 12, 13, 77, 88], SamplingParams(max_tokens=4))  #@d3
    hits = mgr.find_cached_prefix(b)  #@d3a
    print("  hits", hits, mgr.allocate(b, 2, cached=hits))  #@d3b
    b.num_computed_tokens = len(hits) * B + 2  #@d3c

    # 4. why max_blocks uses num_tokens-1
    c = Request("C", a.token_ids[:8], SamplingParams(max_tokens=4))  #@d4
    print("  hits", mgr.find_cached_prefix(c), ", one block, not two.")  #@d4a

    # 5. free A
    mgr.free_request(a)  #@d5

    # 6. eviction
    d = Request("D", list(range(200, 220)), SamplingParams(max_tokens=4))  #@d6
    print("  allocate D +20", mgr.allocate(d, 20))  #@d6a
    d.num_computed_tokens = 20  #@d6b

    # 7. OOM on pages
    e = Request("E", [1, 2, 3, 4], SamplingParams(max_tokens=4))  #@d7
    print("  allocate E +4", mgr.allocate(e, 4), ", no mutation")  #@d7a
`);
