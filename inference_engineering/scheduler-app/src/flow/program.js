import { def, S, IF, WHILE, FOR, CALL, RET, BRK, CONT, AWAIT } from "./dsl.js";
import { blockHash, EOS, MAX_TOKENS } from "./world.js";
import { fmt } from "./interp.js";

// tinyserve.py as executable flowcharts, end to end from main(). Node text is
// the real Python (comments trimmed); the closure next to it does what the line
// does on the simulated world G. GPU work is simulated at the level of "which
// slot gets which token" and tensor shapes, never numbers.

const Tn = (...shape) => ({ _t: shape });
const X = (s) => ({ _x: s });
const tk = (t) => (t === -1 ? "?" : t === EOS ? "⏎" : String(t));
const toks = (a) => a.map(tk).join(" ");
const ids = (rs) => rs.map((r) => r.id).join(", ");
const short = (h) => (h ? `#${h.slice(0, 4)}` : "None");
const nTok = (r) => r.token_ids.length;
const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
const SR = (req, nc, nn) => ({ _sr: true, req, num_computed: nc, num_new: nn });
const completes = (s) => s.num_computed + s.num_new >= nTok(s.req);
const hl = (G, o) => {
  G.hl = { ...(G.hl || {}), ...o };
};
const hasWork = (G) =>
  G.sched.waiting.length > 0 || G.sched.running.length > 0 || G.engine.next_step !== null || G.engine.in_flight !== null;
const workStr = (G) =>
  [`waiting ${G.sched.waiting.length}`, `running ${G.sched.running.length}`, G.engine.next_step && "next_step", G.engine.in_flight && "in_flight"]
    .filter(Boolean).join(" · ");
const output = (req, text, delta) => ({
  _x: `Output(${req.id}, '${text}'${req.finished ? `, ${req.finish_reason}` : ""})`,
  request_id: req.id, text, delta, finished: req.finished, finish_reason: req.finish_reason,
});
export const samplingParams = (rid) => ({
  _x: `SamplingParams(temperature=0.0, max_tokens=${MAX_TOKENS})`, _rid: rid,
  temperature: 0, max_tokens: MAX_TOKENS, repetition_penalty: 1.1, repeat_last_n: 64,
  stop: null, forced_token_ids: null, return_logprobs: false, ignore_eos: false,
});

// ============================================================== main

def("main", {
  mod: "main", sig: "def main():", body: [
    S("ap = argparse.ArgumentParser()\nap.add_argument(\"command\", choices=[\"serve\", \"generate\"])\n...\na = ap.parse_args()",
      (L, G) => { L.a = X(`Namespace(command='${G.cfg.mode}')`); return `command = '${G.cfg.mode}'`; }),
    S("cfg = Config(model=a.model, attention_backend=a.attention_backend,\n             max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,\n             max_num_batched_tokens=a.max_num_batched_tokens, ...)",
      (L, G) => { L.cfg = X("Config(...)"); const c = G.cfg; return `block_size=${c.bs} · max_num_batched_tokens=${c.budget} · max_num_seqs=${c.maxSeqs} · max_model_len=${c.maxModelLen}`; }),
    CALL("engine = Engine(cfg)", "Engine.__init__", (L) => [L.cfg], (L) => { L.engine = X("Engine"); return "← engine ready"; }),
    IF('if a.command == "generate":', (L, G) => G.cfg.mode === "generate", [
      CALL("outs = engine.generate(prompts, SamplingParams(temperature=a.temperature,\n                                                max_tokens=a.max_tokens, ...))\n# tinyserve passes [a.prompt]: 4 prompts here so the scheduler has work",
        "Engine.generate", (L, G) => [G.clients.map((c) => [...c.prompt]), samplingParams(null)],
        (L, r) => { L.outs = r; return `← ${r.length} outputs`; }),
      S("print(a.prompt + out.text)", (L) => L.outs.map((o) => `${o.request_id}: '${o.text}'`).join("  ")),
      RET("return"),
    ]),
    S("import uvicorn"),
    CALL("async_engine = AsyncEngine(engine, a.log_stats_interval)\n# inline in tinyserve: uvicorn.run(build_app(AsyncEngine(engine, ...), a.model), ...)",
      "AsyncEngine.__init__", (L) => [L.engine, 10.0], (L) => { L.async_engine = X("AsyncEngine"); return "← engine thread started"; }),
    CALL('uvicorn.run(build_app(async_engine, a.model), host="0.0.0.0", port=a.port)', "uvicorn.run", (L) => [L.async_engine]),
  ],
});

// ============================================================== startup

def("Engine.__init__", {
  mod: "engine", sig: "def __init__(self, cfg: Config):", params: ["cfg"], body: [
    S("self.cfg = cfg"),
    S("self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)", () => "toy tokenizer: one letter = one token"),
    S("self.stop_ids = {self.tokenizer.eos_token_id}\n...  # + generation_config eos ids\nself.stop_ids.discard(None)",
      (L, G) => { G.engine = { requests: {}, next_step: null, in_flight: null, stop_ids: [EOS] }; return "{⏎}"; }),
    IF("if cfg.tensor_parallel_size > 1:", () => false, [S("port = spawn_tp_workers(cfg)\ninit_tensor_parallel(0, cfg.tensor_parallel_size, port)")]),
    CALL("self.runner, num_blocks = build_runner(cfg)", "build_runner", (L) => [L.cfg], (L, r) => { L.num_blocks = r[1]; return `num_blocks = ${r[1]}`; }),
    IF("if cfg.tensor_parallel_size > 1:", () => false, [S("atexit.register(stop_tensor_parallel, self.runner)")]),
    CALL("self.kv = KVCacheManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)", "KVCacheManager.__init__",
      (L, G) => [L.num_blocks, G.cfg.bs, true], () => "← pages ready"),
    S("self.scheduler = Scheduler(cfg, self.kv)", (L, G) => { G.sched = { waiting: [], running: [] }; return "waiting = deque(), running = []"; }),
    S("self.detok: dict[str, Detokenizer] = {}\nself.requests: dict[str, Request] = {}"),
    S("self.next_step = None\nself.in_flight = None", () => "the async pipeline starts empty"),
  ],
});

def("build_runner", {
  mod: "runner", sig: "def build_runner(cfg: Config) -> tuple[ModelRunner, int]:", params: ["cfg"], body: [
    CALL("runner = ModelRunner(cfg)", "ModelRunner.__init__", (L) => [L.cfg], (L) => { L.runner = X("ModelRunner"); return "← weights + persistent buffers"; }),
    CALL("num_blocks = runner.allocate_kv_cache()", "ModelRunner.allocate_kv_cache", () => [], (L, r) => { L.num_blocks = r; return `← ${r}`; }),
    IF("if cfg.use_cuda_graphs:", () => true, [CALL("runner.capture_cuda_graphs()", "ModelRunner.capture_cuda_graphs", () => [], () => "← graphs captured")]),
    RET("return runner, num_blocks", (L) => [L.runner, L.num_blocks]),
  ],
});

def("ModelRunner.__init__", {
  mod: "runner", sig: "def __init__(self, cfg):", params: ["cfg"], body: [
    S("self.model = load_model(cfg.model, cfg.dtype, cfg.max_model_len, TP_RANK, TP_SIZE, ...)",
      (L, G) => `toy model: ${G.cfg.layers} decoder layers, hidden ${G.cfg.hidden}`),
    S("mc = self.model.cfg\nself.H, self.Hkv = mc.num_attention_heads, mc.num_key_value_heads\nself.D = getattr(mc, \"head_dim\", None) or mc.hidden_size // self.H\nself.num_layers = mc.num_hidden_layers",
      (L, G) => `H=${G.cfg.H}, Hkv=${G.cfg.Hkv}, D=${G.cfg.D}, num_layers=${G.cfg.layers}`),
    S("ATTN_FN = ATTENTION_BACKENDS[cfg.attention_backend]\nTRITON_KERNELS = cfg.use_triton_kernels", () => "triton_attention, fused kernels on"),
    S("self.max_blocks_per_seq = -(-cfg.max_model_len // cfg.block_size)", (L, G) => {
      const c = G.cfg;
      const gs = [1, 2, 4, ...range(1, Math.floor(c.maxSeqs / 8) + 1).map((k) => 8 * k)].filter((s) => s <= c.maxSeqs);
      const ms = range(1, 33).map((k) => 16 * k).filter((s) => s <= c.budget);
      G.runner = {
        max_blocks_per_seq: Math.ceil(c.maxModelLen / c.bs), graphs: [], graph_sizes: gs, mixed_sizes: ms,
        prev_tokens: Array(c.maxSeqs).fill(0), tokens_host: Array(c.maxSeqs).fill(0), hostPending: false, buf: null, fwd: null,
      };
      return `ceil(${c.maxModelLen} / ${c.bs}) = ${G.runner.max_blocks_per_seq}`;
    }),
    S("N, S = cfg.max_num_batched_tokens, cfg.max_num_seqs\nself.input_ids, self.positions = z(N, dt=torch.int64), z(N, dt=torch.int64)\nself.slot_mapping = z(N, dt=torch.int64)\nself.query_start_loc, self.cu_seqlens_k = z(S + 1), z(S + 1)\nself.block_table = z(S, self.max_blocks_per_seq)",
      (L, G) => `fixed-address GPU buffers: input_ids[${G.cfg.budget}], block_table[${G.cfg.maxSeqs}, ${G.runner.max_blocks_per_seq}]`),
    S("self.graphs = {}"),
    S("self.prev_tokens = z(S, dt=torch.int64)\nself.tokens_host = torch.zeros(S, dtype=torch.int64, pin_memory=True)",
      (L, G) => `prev_tokens[${G.cfg.maxSeqs}] on GPU, tokens_host[${G.cfg.maxSeqs}] pinned CPU`),
  ],
});

def("ModelRunner.allocate_kv_cache", {
  mod: "runner", sig: "def allocate_kv_cache(self):", body: [
    S("ATTN_META = None\nself.autotune_triton()", () => "benchmarks kernel configs once"),
    S("torch.cuda.reset_peak_memory_stats()\nh = self.model(self.input_ids[:n], self.positions[:n])\nself.model.logits(h[: self.cfg.max_num_seqs]).float().softmax(-1)",
      () => "profiling run: measure peak activation memory"),
    S("peak = torch.cuda.max_memory_allocated()\nfree, total = torch.cuda.mem_get_info()\nother = total - free - torch.cuda.memory_reserved()\nbudget = total * self.cfg.gpu_memory_utilization - other - peak"),
    S("bytes_per_block = 2 * self.num_layers * self.cfg.block_size * self.Hkv * self.D * 2", (L, G) => {
      const c = G.cfg;
      L.bytes_per_block = 2 * c.layers * c.bs * c.Hkv * c.D * 2;
      return `2·${c.layers}·${c.bs}·${c.Hkv}·${c.D}·2 = ${L.bytes_per_block} B`;
    }),
    S("num_blocks = min(int(budget // bytes_per_block),\n                 self.cfg.max_num_seqs * self.max_blocks_per_seq + 1)",
      (L, G) => { L.num_blocks = G.cfg.numBlocks; return `= ${L.num_blocks} (toy GPU)`; }),
    FOR("for layer in self.model.layers:", (L, G) => range(0, G.cfg.layers), (L, k) => { L.layer = X(`DecoderLayer(${k})`); }, [
      S("layer.self_attn.kv_cache = torch.zeros(\n    2, num_blocks, self.cfg.block_size, self.Hkv, self.D)",
        (L, G) => `[2, ${L.num_blocks}, ${G.cfg.bs}, ${G.cfg.Hkv}, ${G.cfg.D}]  (K|V, page, slot, head, dim)`),
    ]),
    RET("return num_blocks", (L) => L.num_blocks),
  ],
});

def("ModelRunner.capture_cuda_graphs", {
  mod: "runner", sig: "def capture_cuda_graphs(self):", body: [
    S("pool = torch.cuda.graph_pool_handle()"),
    S("captures = [(\"mixed\", T) for T in reversed(self.mixed_sizes())] + \\\n           [(\"decode\", B) for B in reversed(self.graph_sizes())]", (L, G) => {
      const r = G.runner;
      L.captures = [...[...r.mixed_sizes].reverse().map((t) => ["mixed", t]), ...[...r.graph_sizes].reverse().map((b) => ["decode", b])];
      return `mixed_sizes() = [${r.mixed_sizes}] (needs ≥16 tokens), graph_sizes() = [${r.graph_sizes}]`;
    }),
    FOR("for kind, size in captures:", (L) => L.captures, (L, [kind, size]) => { L.kind = kind; L.size = size; }, [
      S("packed = self._pack([], pad_to=size) if kind == \"decode\" else self._pack([], pad_tokens=size)\nT, ATTN_META = self._upload(packed, for_graph=True)",
        (L) => `${L.size} dummy rows, all pointing at the null page 0`),
      S("self.model(self.input_ids[:T], self.positions[:T])", () => "warm-up: lazy workspace allocs"),
      S("g = torch.cuda.CUDAGraph()\nwith torch.cuda.graph(g, pool=pool):\n    out = self.model(self.input_ids[:T], self.positions[:T])", () => "records every kernel launch + buffer address"),
      S("self.graphs[(kind, size)] = (g, out)", (L, G) => { G.runner.graphs.push([L.kind, L.size]); return `graphs = {${G.runner.graphs.map((g) => `(${g[0]}, ${g[1]})`).join(", ")}}`; }),
    ], { label: (L) => `('${L.kind}', ${L.size})` }),
  ],
});

def("KVCacheManager.__init__", {
  mod: "kv", sig: "def __init__(self, num_blocks, block_size, enable_caching):", params: ["num_blocks", "block_size", "enable_caching"], body: [
    S("self.block_size = block_size\nself.enable_caching = enable_caching\nself.num_blocks = num_blocks",
      (L, G) => { G.kv = { bs: L.block_size, caching: L.enable_caching, num_blocks: L.num_blocks, ref: [], bh: [], free: [], h2b: new Map() }; }),
    S("self.ref_cnt = [0] * num_blocks", (L, G) => { G.kv.ref = Array(L.num_blocks).fill(0); G.pages = Array.from({ length: L.num_blocks }, () => Array(L.block_size).fill(null)); }),
    S("self.block_hash = [None] * num_blocks", (L, G) => { G.kv.bh = Array(L.num_blocks).fill(null); }),
    S("self.free = OrderedDict((i, None) for i in range(1, num_blocks))", (L, G) => { G.kv.free = range(1, L.num_blocks); return `[${G.kv.free.join(", ")}]  (page 0 = null page)`; }),
    S("self.hash_to_block: dict[int, int] = {}"),
  ],
});

// ============================================================== offline API

def("Engine.generate", {
  mod: "engine", sig: "def generate(self, prompts: list, params):", params: ["prompts", "params"], body: [
    S("params = params if isinstance(params, list) else [params] * len(prompts)", (L) => {
      L.params = L.prompts.map((_, k) => samplingParams(null, k));
      L.rids = [];
      return `${L.params.length} × SamplingParams`;
    }),
    FOR("rids = [self.add_request(p, sp) for p, sp in zip(prompts, params)]", (L) => L.prompts.map((p, k) => [p, L.params[k]]),
      (L, [p, sp]) => { L.p = p; L.sp = sp; }, [
        CALL("self.add_request(p, sp)", "Engine.add_request", (L) => [L.p, L.sp, null], (L, r) => { L.rids.push(r); return `← '${r}'`; }),
      ], { label: (L) => `p = ${L.p.join("")}` }),
    S("done: dict[str, Output] = {}", (L) => { L.done = {}; }),
    WHILE("while self.has_work():", (L, G) => hasWork(G), [
      FOR("for o in self.step():", (L, G, ret) => ret, (L, o) => { L.o = o; }, [
        IF("if o.finished:", (L) => L.o.finished, [
          S("done[o.request_id] = o", (L) => { L.done[L.o.request_id] = L.o; return `${L.o.request_id}: '${L.o.text}'`; }),
        ]),
      ], { call: { fn: "Engine.step", args: () => [] }, label: (L) => `o = ${fmt(L.o)}` }),
    ], { label: (L, G, v) => (v ? `True · ${workStr(G)}` : "False · nothing left") }),
    RET("return [done[r] for r in rids]", (L) => L.rids.map((r) => L.done[r])),
  ],
});

def("Engine.add_request", {
  mod: "engine", sig: "def add_request(self, prompt, params, request_id=None):", params: ["prompt", "params", "request_id"], body: [
    S("ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)", (L) => { L.ids = [...L.prompt]; return toks(L.ids); }),
    S('assert 0 < len(ids) < self.cfg.max_model_len, "prompt too long"', (L, G) => `0 < ${L.ids.length} < ${G.cfg.maxModelLen} ✓`),
    S("rid = request_id or uuid.uuid4().hex", (L, G) => {
      L.rid = L.request_id ?? G.clients[G.reqs.length].id;
      return L.request_id ? `'${L.rid}'` : `uuid4().hex, drawn as '${L.rid}'`;
    }),
    S("self.requests[rid] = req = Request(rid, ids, params)", (L, G) => {
      L.req = G.newReq(L.rid, L.ids, L.params);
      G.engine.requests[L.rid] = L.req;
      hl(G, { req: L.rid });
      return "num_computed_tokens = 0, block_ids = []";
    }),
    S("self.scheduler.add(req)          # waiting.append(req)", (L, G) => { G.sched.waiting.push(L.req); hl(G, { req: L.rid }); return `waiting = [${ids(G.sched.waiting)}]`; }),
    S("self.detok[rid] = Detokenizer(self.tokenizer)", (L, G) => { G.detok[L.rid] = { text: "", prefix: 0, read: 0 }; G.outputs[L.rid] = ""; }),
    RET("return rid", (L) => L.rid),
  ],
});

// ============================================================== Engine.step (async pipeline)

def("Engine.step", {
  mod: "engine", sig: "def step(self):", body: [
    S("batch, logits = None, None", (L, G) => { G.stepNo += 1; L.batch = null; L.logits = null; return `engine step ${G.stepNo}`; }),
    IF("if self.next_step is not None:", (L, G) => G.engine.next_step !== null, [
      S("batch, prepared = self.next_step", (L, G) => {
        [L.batch, L.prepared] = G.engine.next_step;
        G.runner.fwd = L.batch;
        return `prepared last step: ${L.batch.map((s) => `${s.req.id}+${s.num_new}`).join(" ")} · ${L.prepared[1]}`;
      }),
      CALL("logits = self.runner.launch_forward(prepared)", "ModelRunner.launch_forward", (L) => [L.prepared], (L, r) => { L.logits = r; return "← logits queued on the GPU"; }),
      S("self.next_step = None", (L, G) => { G.engine.next_step = null; }),
    ]),
    CALL("outputs = self._finish_step(*self.in_flight) if self.in_flight is not None else []", "Engine._finish_step",
      (L, G) => G.engine.in_flight, (L, r) => { L.outputs = r; return `← ${r.length} outputs`; },
      { when: (L, G) => G.engine.in_flight !== null, otherwise: (L) => { L.outputs = []; return "in_flight is None → []"; } }),
    S("self.in_flight = None", (L, G) => { G.engine.in_flight = null; }),
    IF("if batch:", (L) => !!(L.batch && L.batch.length), [
      CALL("done = self.runner.launch_sample(batch, logits)", "ModelRunner.launch_sample", (L) => [L.batch, L.logits], (L, r) => { L.done = r; return "← cuda.Event"; }),
      S("self.num_prompt_tokens_run += sum(s.num_new for s in batch if s.num_computed < s.req.prompt_len)"),
      S("live = []", (L) => { L.live = []; }),
      FOR("for i, s in enumerate(batch):", (L) => L.batch, (L, s, i) => { L.i = i; L.s = s; L.req = undefined; }, [
        S("req = s.req", (L, G) => { L.req = L.s.req; hl(G, { req: L.req.id }); }),
        IF("if not s.completes_prompt:", (L) => !completes(L.s), [
          S("req.was_chunked_prefill = True", (L) => { L.req.was_chunked_prefill = true; return `${L.s.num_computed}+${L.s.num_new} < ${nTok(L.req)}: mid-prompt chunk, sample is junk`; }),
          CONT(),
        ]),
        IF("if req.finished:", (L) => L.req.finished, [CONT()]),
        S("live.append(i)", (L) => { L.live.push(L.i); return `[${L.live}]`; }),
        S("forced = req.params.forced_token_ids", (L) => { L.forced = null; }),
        IF("if forced is not None:", () => false, [S("req.token_ids.append(forced[len(req.output_ids)])")], [
          S("req.token_ids.append(-1)    # placeholder, filled in by _finish_step", (L, G) => {
            L.req.token_ids.push(-1);
            hl(G, { req: L.req.id, sampled: nTok(L.req) - 1 });
            return "the sampled id is still on the GPU";
          }),
          S("req.pending, req.pending_row = 1, i", (L) => { L.req.pending = 1; L.req.pending_row = L.i; return `pending_row = ${L.i}`; }),
        ]),
        IF("if len(req.output_ids) >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:",
          (L, G) => nTok(L.req) - L.req.prompt_len >= L.req.params.max_tokens || nTok(L.req) >= G.cfg.maxModelLen, [
            CALL('self.scheduler.finish(req, "length")', "Scheduler.finish", (L) => [L.req, "length"]),
          ], null, { label: (L, G, v) => `${nTok(L.req) - L.req.prompt_len} >= ${L.req.params.max_tokens} → ${v ? "True" : "False"}` }),
      ], { label: (L) => `i = ${L.i}, s = ${fmt(L.s)}` }),
      S("self.in_flight = (batch, live, logits, done)", (L, G) => { G.engine.in_flight = [L.batch, L.live, L.logits, L.done]; return `live rows [${L.live}]`; }),
      S("self.num_generated_tokens += len(live)"),
    ]),
    CALL("nxt = self.scheduler.schedule()", "Scheduler.schedule", () => [], (L, r) => {
      L.nxt = r;
      return r.length ? `← ${r.map((s) => `${s.req.id}+${s.num_new}`).join(" ")}` : "← []";
    }),
    S("self.tokens_last_step = sum(s.num_new for s in nxt) if nxt else 0", (L) => `${L.nxt.reduce((a, s) => a + s.num_new, 0)} tokens`),
    IF("if nxt:", (L) => L.nxt.length > 0, [
      CALL("self.next_step = (nxt, self.runner.prepare(nxt))", "ModelRunner.prepare", (L) => [L.nxt],
        (L, r, G) => { G.engine.next_step = [L.nxt, r]; return `← mode '${r[1]}'${r[2] ? `, size ${r[2]}` : ""}`; }),
    ]),
    RET("return outputs", (L) => L.outputs),
  ],
});

def("Engine._finish_step", {
  mod: "engine", sig: "def _finish_step(self, batch, live, logits, done):", params: ["batch", "live", "logits", "done"], body: [
    S("done.synchronize()", (L, G) => { G.runner.hostPending = false; return "wait until the GPU → CPU copy has landed"; }),
    S("tokens = self.runner.tokens_host[:len(batch)].tolist()", (L, G) => { L.tokens = G.runner.tokens_host.slice(0, L.batch.length); return `[${L.tokens.map(tk)}]`; }),
    S("outputs = []", (L) => { L.outputs = []; }),
    FOR("for i in live:", (L) => L.live, (L, i) => { L.i = i; L.req = undefined; }, [
      S("req = batch[i].req", (L, G) => { L.req = L.batch[L.i].req; hl(G, { req: L.req.id }); }),
      IF("if req.request_id not in self.requests:", (L, G) => !G.engine.requests[L.req.id], [CONT()]),
      IF("if req.pending:", (L) => !!L.req.pending, [
        S("req.token_ids[-1] = tokens[i]\nreq.pending = 0", (L, G) => {
          L.req.token_ids[nTok(L.req) - 1] = L.tokens[L.i];
          L.req.pending = 0;
          hl(G, { req: L.req.id, sampled: nTok(L.req) - 1 });
          return `? → '${tk(L.tokens[L.i])}'`;
        }),
      ]),
      S("tok = req.token_ids[-1]", (L) => { L.tok = L.req.token_ids[nTok(L.req) - 1]; }),
      IF("if req.params.return_logprobs:", () => false, [S("req.logprobs.append(logits[i].float().log_softmax(-1).cpu())")]),
      IF("if tok in self.stop_ids and not req.params.ignore_eos:", (L) => L.tok === EOS, [
        IF("if req.finished:", (L) => L.req.finished, [S('req.finish_reason = "stop"', (L) => { L.req.finish_reason = "stop"; })], [
          CALL('self.scheduler.finish(req, "stop")', "Scheduler.finish", (L) => [L.req, "stop"]),
        ]),
      ]),
      S("d = self.detok[req.request_id]", (L) => { L.d = X(`Detokenizer(${L.req.id})`); }),
      CALL("delta = d.update(req.output_ids, req.finished)", "Detokenizer.update",
        (L, G) => [L.req.token_ids.slice(L.req.prompt_len), L.req.finished, G.detok[L.req.id]], (L, r) => { L.delta = r; return `← '${r}'`; }),
      S("text = d.text", (L, G) => { L.text = G.detok[L.req.id].text; G.outputs[L.req.id] = L.text; }),
      IF("if req.params.stop:", () => false, [S("text, hit = apply_stop_strings(d.text, req.params.stop)\n...")]),
      S("outputs.append(Output(req.request_id, text, delta, req.output_ids, req.finished,\n                      req.finish_reason, req.prompt_len, req.num_cached_tokens, ...))",
        (L) => { L.outputs.push(output(L.req, L.text, L.delta)); return fmt(L.outputs.at(-1)); }),
      IF("if req.finished:", (L) => L.req.finished, [
        S("self.total_prompt_tokens += req.prompt_len\nself.total_cached_tokens += req.num_cached_tokens\ndel self.detok[req.request_id]\ndel self.requests[req.request_id]",
          (L, G) => { delete G.engine.requests[L.req.id]; return `${L.req.id} done (${L.req.finish_reason})`; }),
      ]),
    ], { label: (L) => `i = ${L.i}` }),
    RET("return outputs", (L) => L.outputs),
  ],
});

def("Detokenizer.update", {
  mod: "detok", sig: "def update(self, ids, final) -> str:", params: ["ids", "final", "_d"], body: [
    S("old = self.tok.decode(ids[self.prefix:self.read], skip_special_tokens=True)", (L) => {
      const dec = (a) => a.filter((t) => t !== EOS && t !== -1).join("");
      L.old = dec(L.ids.slice(L._d.prefix, L._d.read));
      return `'${L.old}'`;
    }),
    S("new = self.tok.decode(ids[self.prefix:], skip_special_tokens=True)", (L) => {
      L.new = L.ids.slice(L._d.prefix).filter((t) => t !== EOS && t !== -1).join("");
      return `'${L.new}'`;
    }),
    IF('if len(new) <= len(old) or (new.endswith("�") and not final):', (L) => L.new.length <= L.old.length, [RET('return ""', () => "")]),
    S("delta = new[len(old):]", (L) => { L.delta = L.new.slice(L.old.length); return `'${L.delta}'`; }),
    S("self.text += delta", (L) => { L._d.text += L.delta; return `self.text = '${L._d.text}'`; }),
    S("self.prefix, self.read = self.read, len(ids)", (L) => { L._d.prefix = L._d.read; L._d.read = L.ids.length; return `${L._d.prefix}, ${L._d.read}`; }),
    RET("return delta", (L) => L.delta),
  ],
});

// ============================================================== Scheduler

def("Scheduler.schedule", {
  mod: "sched", sig: "def schedule(self) -> list[ScheduledRequest]:", body: [
    S("budget = self.cfg.max_num_batched_tokens", (L, G) => { L.budget = G.cfg.budget; return `= ${L.budget}`; }),
    S("plan: dict[str, int] = {}", (L) => { L.plan = {}; }),
    S("preempted = False", (L) => { L.preempted = false; }),
    S("i = 0", (L) => { L.i = 0; }),
    WHILE("while i < len(self.running) and budget > 0:", (L, G) => L.i < G.sched.running.length && L.budget > 0, [
      S("req = self.running[i]", (L, G) => { L.req = G.sched.running[L.i]; L.victim = undefined; hl(G, { req: L.req.id }); return L.req.id; }),
      S("n = min(req.num_tokens - req.num_computed_tokens, budget)", (L) => {
        L.n = Math.min(nTok(L.req) - L.req.num_computed_tokens, L.budget);
        return `min(${nTok(L.req)} - ${L.req.num_computed_tokens}, ${L.budget}) = ${L.n}`;
      }),
      WHILE("while not self.kv.allocate(req, n):", (L, G, ok) => !ok, [
        S("victim = self.running.pop()", (L, G) => { L.victim = G.sched.running.pop(); hl(G, { req: L.victim.id }); return `${L.victim.id} (newest running)`; }),
        S("victim.was_preempted = True", (L) => { L.victim.was_preempted = true; }),
        CALL("self.kv.free_request(victim)", "KVCacheManager.free_request", (L) => [L.victim]),
        S("victim.num_computed_tokens = 0", (L, G) => { Object.assign(L.victim, { num_computed_tokens: 0, kvWritten: 0, hit: 0 }); hl(G, { req: L.victim.id }); return "its K/V is gone: recompute later"; }),
        S("self.waiting.appendleft(victim)", (L, G) => { G.sched.waiting.unshift(L.victim); return `waiting = [${ids(G.sched.waiting)}]`; }),
        S("budget += plan.pop(victim.request_id, 0)", (L) => { L.budget += L.plan[L.victim.id] ?? 0; delete L.plan[L.victim.id]; return `= ${L.budget}`; }),
        S("preempted = True", (L) => { L.preempted = true; }),
        IF("if victim is req:", (L) => L.victim === L.req, [BRK()]),
      ], { call: { fn: "KVCacheManager.allocate", args: (L) => [L.req, L.n, []] }, label: (L, G, v) => (v ? "allocate → False · preempt" : "allocate → True · fits") }),
      IF("if req not in self.running:", (L, G) => !G.sched.running.includes(L.req), [BRK()]),
      S("plan[req.request_id] = n", (L) => { L.plan[L.req.id] = L.n; return fmt(L.plan); }),
      S("budget -= n", (L) => { L.budget -= L.n; return `= ${L.budget}`; }),
      S("i += 1", (L) => { L.i += 1; }),
    ], { label: (L, G, v) => `${L.i} < ${G.sched.running.length} and ${L.budget} > 0 → ${v ? "True" : "False"}` }),
    WHILE("while (not preempted and self.waiting and budget > 0\n       and len(self.running) < self.cfg.max_num_seqs):",
      (L, G) => !L.preempted && G.sched.waiting.length > 0 && L.budget > 0 && G.sched.running.length < G.cfg.maxSeqs, [
        S("req = self.waiting[0]", (L, G) => { L.req = G.sched.waiting[0]; L.victim = undefined; hl(G, { req: L.req.id }); return `head = ${L.req.id}`; }),
        CALL("cached = self.kv.find_cached_prefix(req)", "KVCacheManager.find_cached_prefix", (L) => [L.req], (L, r) => { L.cached = r; return `← [${r.join(", ")}]`; }),
        S("n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)", (L, G) => {
          L.n = Math.min(nTok(L.req) - L.cached.length * G.cfg.bs, L.budget);
          return `min(${nTok(L.req)} - ${L.cached.length}·${G.cfg.bs}, ${L.budget}) = ${L.n}`;
        }),
        IF("if not self.kv.allocate(req, n, cached):", (L, G, ok) => !ok, [BRK()], null, {
          call: { fn: "KVCacheManager.allocate", args: (L) => [L.req, L.n, L.cached] },
          label: (L, G, v) => (v ? "allocate → False · stop admitting (FIFO)" : "allocate → True"),
        }),
        S("self.waiting.popleft()", (L, G) => { G.sched.waiting.shift(); return `waiting = [${ids(G.sched.waiting)}]`; }),
        S("req.num_computed_tokens = len(cached) * self.cfg.block_size", (L, G) => {
          const v = L.cached.length * G.cfg.bs;
          Object.assign(L.req, { num_computed_tokens: v, kvWritten: v, hit: v });
          hl(G, { req: L.req.id });
          return `= ${v} (from the prefix cache)`;
        }),
        S("req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)", (L) => { L.req.num_cached_tokens = Math.max(L.req.num_cached_tokens, L.req.num_computed_tokens); }),
        S("self.running.append(req)", (L, G) => { G.sched.running.push(L.req); return `running = [${ids(G.sched.running)}]`; }),
        S("plan[req.request_id] = n", (L) => { L.plan[L.req.id] = L.n; return fmt(L.plan); }),
        S("budget -= n", (L) => { L.budget -= L.n; return `= ${L.budget}`; }),
      ], {
        label: (L, G, v) => (v ? `True · head ${G.sched.waiting[0].id}`
          : `False · ${L.preempted ? "preempted" : !G.sched.waiting.length ? "waiting empty" : L.budget <= 0 ? "budget 0" : "max_num_seqs"}`),
      }),
    S("batch = []", (L) => { L.batch = []; L.req = undefined; }),
    FOR("for req in self.running:", (L, G) => [...G.sched.running], (L, r) => { L.req = r; }, [
      IF("if req.request_id in plan:", (L) => L.req.id in L.plan, [
        S("batch.append(ScheduledRequest(req, req.num_computed_tokens, plan[req.request_id]))", (L) => {
          L.batch.push(SR(L.req, L.req.num_computed_tokens, L.plan[L.req.id]));
          return fmt(L.batch.at(-1));
        }),
        S("req.num_computed_tokens += plan[req.request_id]", (L, G) => {
          L.req.num_computed_tokens += L.plan[L.req.id];
          hl(G, { req: L.req.id });
          return `= ${L.req.num_computed_tokens} (before the model runs)`;
        }),
      ]),
    ], { label: (L) => `req = ${L.req.id}` }),
    RET("return batch", (L) => L.batch),
  ],
});

def("Scheduler.finish", {
  mod: "sched", sig: "def finish(self, req: Request, reason: str) -> None:", params: ["req", "reason"], body: [
    S("req.finished, req.finish_reason = True, reason", (L, G) => { L.req.finished = true; L.req.finish_reason = L.reason; hl(G, { req: L.req.id }); return `${L.req.id}: ${L.reason}`; }),
    IF("if req in self.running:", (L, G) => G.sched.running.includes(L.req), [
      S("self.running.remove(req)", (L, G) => { G.sched.running = G.sched.running.filter((r) => r !== L.req); return `running = [${ids(G.sched.running)}]`; }),
    ], [
      IF("elif req in self.waiting:", (L, G) => G.sched.waiting.includes(L.req), [
        S("self.waiting.remove(req)", (L, G) => { G.sched.waiting = G.sched.waiting.filter((r) => r !== L.req); }),
      ]),
    ]),
    CALL("self.kv.free_request(req)", "KVCacheManager.free_request", (L) => [L.req]),
  ],
});

// ============================================================== KVCacheManager

def("KVCacheManager._update_hashes", {
  mod: "kv", sig: "def _update_hashes(self, req: Request):", params: ["req"], body: [
    S("bs = self.block_size", (L, G) => { L.bs = G.kv.bs; }),
    WHILE("while len(req.block_hashes) < (req.num_tokens - req.pending) // bs:",
      (L) => L.req.block_hashes.length < Math.floor((nTok(L.req) - L.req.pending) / L.bs), [
        S("i = len(req.block_hashes)", (L) => { L.i = L.req.block_hashes.length; }),
        S("parent = req.block_hashes[-1] if req.block_hashes else None", (L) => { L.parent = L.i ? L.req.block_hashes[L.i - 1] : null; return short(L.parent); }),
        S("req.block_hashes.append(hash((parent, tuple(req.token_ids[i * bs:(i + 1) * bs]))))", (L, G) => {
          const t = L.req.token_ids.slice(L.i * L.bs, (L.i + 1) * L.bs);
          L.h = blockHash(G, L.parent, t);
          L.req.block_hashes.push(L.h);
          hl(G, { req: L.req.id, hashes: [L.h], hashBlock: { rid: L.req.id, i: L.i } });
          return `hash(${short(L.parent)}, ${toks(t)}) = ${short(L.h)}`;
        }),
      ], { label: (L, G, v) => `${L.req.block_hashes.length} < (${nTok(L.req)} - ${L.req.pending}) // ${L.bs} → ${v ? "True" : "False"}` }),
  ],
});

def("KVCacheManager.find_cached_prefix", {
  mod: "kv", sig: "def find_cached_prefix(self, req: Request):", params: ["req"], body: [
    IF("if not self.enable_caching:", (L, G) => !G.kv.caching, [RET("return []", () => [])]),
    CALL("self._update_hashes(req)", "KVCacheManager._update_hashes", (L) => [L.req], (L) => `${L.req.block_hashes.length} hashes`),
    S("max_blocks = (req.num_tokens - 1) // self.block_size", (L, G) => { L.max_blocks = Math.floor((nTok(L.req) - 1) / G.kv.bs); return `(${nTok(L.req)} - 1) // ${G.kv.bs} = ${L.max_blocks}`; }),
    S("hits = []", (L) => { L.hits = []; }),
    FOR("for h in req.block_hashes[:max_blocks]:", (L) => L.req.block_hashes.slice(0, L.max_blocks), (L, h) => { L.h = h; L.b = undefined; }, [
      S("b = self.hash_to_block.get(h)", (L, G) => {
        const b = G.kv.h2b.get(L.h);
        L.b = b ?? null;
        hl(G, { req: L.req.id, lookup: { h: L.h, hit: b !== undefined }, hashBlock: { rid: L.req.id, i: L.hits.length }, ...(b !== undefined ? { pages: { [b]: "hit" } } : {}) });
        return b === undefined ? `${short(L.h)} → None (miss)` : `${short(L.h)} → page ${b}`;
      }),
      IF("if b is None:", (L) => L.b === null, [BRK()]),
      S("hits.append(b)", (L) => { L.hits.push(L.b); return `[${L.hits}]`; }),
    ], { label: (L) => `h = ${short(L.h)}` }),
    RET("return hits", (L) => L.hits),
  ],
});

def("KVCacheManager.allocate", {
  mod: "kv", sig: "def allocate(self, req, num_new_tokens, cached=()):", params: ["req", "num_new_tokens", "cached"], body: [
    S("total = (req.num_computed_tokens\n         + len(cached) * self.block_size\n         + num_new_tokens)", (L, G) => {
      L.total = L.req.num_computed_tokens + L.cached.length * G.kv.bs + L.num_new_tokens;
      return `${L.req.num_computed_tokens} + ${L.cached.length}·${G.kv.bs} + ${L.num_new_tokens} = ${L.total}`;
    }),
    S("total_needed_blocks = -(-total // self.block_size)", (L, G) => { L.total_needed_blocks = Math.ceil(L.total / G.kv.bs); return `ceil(${L.total} / ${G.kv.bs}) = ${L.total_needed_blocks}`; }),
    S("need = total_needed_blocks - len(req.block_ids) - len(cached)", (L) => {
      L.need = L.total_needed_blocks - L.req.block_ids.length - L.cached.length;
      return `${L.total_needed_blocks} - ${L.req.block_ids.length} - ${L.cached.length} = ${L.need}`;
    }),
    S("reclaim = sum(1 for b in cached if self.ref_cnt[b] == 0)", (L, G) => { L.reclaim = L.cached.filter((b) => G.kv.ref[b] === 0).length; return `= ${L.reclaim}`; }),
    IF("if need + reclaim > self.num_free:", (L, G) => L.need + L.reclaim > G.kv.free.length, [RET("return False", () => false)], null,
      { label: (L, G, v) => `${L.need} + ${L.reclaim} > ${G.kv.free.length} free → ${v ? "True" : "False"}` }),
    FOR("for b in cached:", (L) => L.cached, (L, b) => { L.b = b; }, [
      IF("if self.ref_cnt[b] == 0:", (L, G) => G.kv.ref[L.b] === 0, [
        S("del self.free[b]", (L, G) => { G.kv.free = G.kv.free.filter((x) => x !== L.b); hl(G, { req: L.req.id, pages: { [L.b]: "claim" } }); return "revived off the free list"; }),
      ]),
      S("self.ref_cnt[b] += 1", (L, G) => { G.kv.ref[L.b] += 1; hl(G, { req: L.req.id, pages: { [L.b]: "claim" } }); return `ref_cnt[${L.b}] = ${G.kv.ref[L.b]}`; }),
      S("req.block_ids.append(b)", (L) => { L.req.block_ids.push(L.b); return `[${L.req.block_ids}]`; }),
    ], { label: (L) => `b = ${L.b}` }),
    FOR("for _ in range(max(need, 0)):", (L) => range(0, L.need), () => {}, [
      S("b, _ = self.free.popitem(last=False)", (L, G) => {
        L.b = G.kv.free.shift();
        G.pages[L.b] = G.pages[L.b].map((s) => s && { ...s, stale: true });
        hl(G, { req: L.req.id, pages: { [L.b]: "new" } });
        return `page ${L.b} (LRU front)`;
      }),
      IF("if self.block_hash[b] is not None:", (L, G) => G.kv.bh[L.b] !== null, [
        S("self.hash_to_block.pop(self.block_hash[b], None)\nself.block_hash[b] = None", (L, G) => {
          const old = G.kv.bh[L.b];
          G.kv.h2b.delete(old);
          G.kv.bh[L.b] = null;
          hl(G, { req: L.req.id, pages: { [L.b]: "evict" }, evicted: [old] });
          return `evict ${short(old)}: that prefix must be recomputed`;
        }),
      ]),
      S("self.ref_cnt[b] = 1", (L, G) => { G.kv.ref[L.b] = 1; }),
      S("req.block_ids.append(b)", (L, G) => { L.req.block_ids.push(L.b); hl(G, { req: L.req.id, pages: { [L.b]: "new" } }); return `[${L.req.block_ids}]`; }),
    ]),
    CALL("self._register_full_blocks(req, total)", "KVCacheManager._register_full_blocks", (L) => [L.req, L.total], () => "← done"),
    RET("return True", () => true),
  ],
});

def("KVCacheManager._register_full_blocks", {
  mod: "kv", sig: "def _register_full_blocks(self, req, num_tokens):", params: ["req", "num_tokens"], body: [
    IF("if not self.enable_caching:", (L, G) => !G.kv.caching, [RET("return")]),
    CALL("self._update_hashes(req)", "KVCacheManager._update_hashes", (L) => [L.req], (L) => `${L.req.block_hashes.length} hashes`),
    FOR("for i in range(min(num_tokens, req.num_tokens - req.pending) // self.block_size):",
      (L, G) => range(0, Math.floor(Math.min(L.num_tokens, nTok(L.req) - L.req.pending) / G.kv.bs)), (L, i) => { L.i = i; }, [
        S("b, h = req.block_ids[i], req.block_hashes[i]", (L) => { L.b = L.req.block_ids[L.i]; L.h = L.req.block_hashes[L.i]; return `page ${L.b}, ${short(L.h)}`; }),
        IF("if self.block_hash[b] is None:", (L, G) => G.kv.bh[L.b] === null, [
          S("self.block_hash[b] = h\nself.hash_to_block.setdefault(h, b)", (L, G) => {
            G.kv.bh[L.b] = L.h;
            if (!G.kv.h2b.has(L.h)) G.kv.h2b.set(L.h, L.b);
            hl(G, { req: L.req.id, pages: { [L.b]: "hash" }, hashes: [L.h], hashBlock: { rid: L.req.id, i: L.i } });
            return `page ${L.b} ↔ ${short(L.h)}: now findable`;
          }),
        ]),
      ], { label: (L) => `i = ${L.i}` }),
  ],
});

def("KVCacheManager.free_request", {
  mod: "kv", sig: "def free_request(self, req: Request):", params: ["req"], body: [
    FOR("for b in reversed(req.block_ids):", (L) => [...L.req.block_ids].reverse(), (L, b) => { L.b = b; }, [
      S("self.ref_cnt[b] -= 1", (L, G) => { G.kv.ref[L.b] -= 1; hl(G, { req: L.req.id, pages: { [L.b]: "deref" } }); return `ref_cnt[${L.b}] = ${G.kv.ref[L.b]}`; }),
      IF("if self.ref_cnt[b] == 0:", (L, G) => G.kv.ref[L.b] === 0, [
        S("self.free[b] = None", (L, G) => {
          G.kv.free.push(L.b);
          hl(G, { req: L.req.id, pages: { [L.b]: "free" } });
          return `free tail ← ${L.b}${G.kv.bh[L.b] ? " (hash kept: still a cache hit)" : ""}`;
        }),
      ]),
    ], { label: (L) => `b = ${L.b}` }),
    S("req.block_ids = []", (L) => { L.req.block_ids = []; }),
  ],
});

// ============================================================== ModelRunner (CPU half)

def("ModelRunner.prepare", {
  mod: "runner", sig: "def prepare(self, batch: list[ScheduledRequest]):", params: ["batch"], body: [
    S("is_decode = all(s.num_new == 1 for s in batch)", (L) => { L.is_decode = L.batch.every((s) => s.num_new === 1); }),
    S("size = next((s for s in self.graph_sizes() if s >= len(batch)), None)", (L, G) => {
      L.size = G.runner.graph_sizes.find((s) => s >= L.batch.length) ?? null;
      return `graph_sizes() = [${G.runner.graph_sizes}], len(batch) = ${L.batch.length} → ${fmt(L.size)}`;
    }),
    IF("if self.graphs and is_decode and size is not None:", (L, G) => G.runner.graphs.length > 0 && L.is_decode && L.size !== null, [
      CALL('return self._pack(batch, pad_to=size), "decode", size', "ModelRunner._pack", (L) => [L.batch, L.size, 0],
        (L) => `← decode graph padded to ${L.size}`, { returns: (L, p) => [p, "decode", L.size] }),
    ]),
    S("T = sum(s.num_new for s in batch)", (L) => { L.T = L.batch.reduce((a, s) => a + s.num_new, 0); }),
    S("size = next((s for s in self.mixed_sizes() if s >= T), None)", (L, G) => {
      L.size = G.runner.mixed_sizes.find((s) => s >= L.T) ?? null;
      return `mixed_sizes() = [${G.runner.mixed_sizes}] → ${fmt(L.size)}`;
    }),
    IF("if self.graphs and size is not None:", (L, G) => G.runner.graphs.length > 0 && L.size !== null, [
      CALL('return self._pack(batch, pad_tokens=size), "mixed", size', "ModelRunner._pack", (L) => [L.batch, 0, L.size],
        () => "← mixed graph", { returns: (L, p) => [p, "mixed", L.size] }),
    ]),
    CALL('return self._pack(batch), "eager", None', "ModelRunner._pack", (L) => [L.batch, 0, 0], () => "← eager (no graph fits)",
      { returns: (L, p) => [p, "eager", null] }),
  ],
});

def("ModelRunner._pack", {
  mod: "runner", sig: "def _pack(self, batch, pad_to: int = 0, pad_tokens: int = 0) -> PackedBatch:", params: ["batch", "pad_to", "pad_tokens"], body: [
    S("bs = self.cfg.block_size\nnum_seqs = max(len(batch), pad_to, self.mixed_num_seqs(pad_tokens) if pad_tokens else 0)",
      (L, G) => { L.bs = G.cfg.bs; L.num_seqs = Math.max(L.batch.length, L.pad_to, L.pad_tokens ? Math.min(L.pad_tokens, G.cfg.maxSeqs) : 0); return `num_seqs = ${L.num_seqs}`; }),
    S("ids, pos, slots, logit_rows = [], [], [], []\nfill_rows, fill_src = [], []\nqsl, seq_lens = [0], []",
      (L) => Object.assign(L, { ids: [], pos: [], slots: [], logit_rows: [], fill_rows: [], fill_src: [], qsl: [0], seq_lens: [], _owner: [] }) && undefined),
    S("bt = np.zeros((num_seqs, self.max_blocks_per_seq), dtype=np.int32)",
      (L, G) => { L.bt = Array.from({ length: L.num_seqs }, () => Array(G.runner.max_blocks_per_seq).fill(0)); return `[${L.num_seqs}, ${G.runner.max_blocks_per_seq}] zeros`; }),
    FOR("for i, s in enumerate(batch):", (L) => L.batch, (L, s, i) => { L.i = i; L.s = s; }, [
      S("start, n = s.num_computed, s.num_new", (L) => { L.start = L.s.num_computed; L.n = L.s.num_new; }),
      S("p = np.arange(start, start + n)", (L) => { L.p = range(L.start, L.start + L.n); return `[${L.p}]`; }),
      S("blocks = np.asarray(s.req.block_ids)", (L) => { L.blocks = [...L.s.req.block_ids]; return `[${L.blocks}]`; }),
      S("ids.extend(s.req.token_ids[start:start + n])", (L, G) => {
        const t = L.s.req.token_ids.slice(L.start, L.start + L.n);
        L.ids.push(...t);
        L._owner.push(...t.map(() => L.s.req.id));
        hl(G, { req: L.s.req.id });
        return toks(t);
      }),
      IF("if s.req.pending and start + n == s.req.num_tokens:", (L) => !!L.s.req.pending && L.start + L.n === nTok(L.s.req), [
        S("ids[-1] = 0\nfill_rows.append(len(ids) - 1)\nfill_src.append(s.req.pending_row)", (L) => {
          L.ids[L.ids.length - 1] = 0;
          L.fill_rows.push(L.ids.length - 1);
          L.fill_src.push(L.s.req.pending_row);
          return `row ${L.ids.length - 1} gets prev_tokens[${L.s.req.pending_row}] on the GPU`;
        }),
      ]),
      S("pos.extend(p.tolist())", (L) => { L.pos.push(...L.p); }),
      S("slots.extend((blocks[p // bs] * bs + p % bs).tolist())", (L) => {
        const sl = L.p.map((q) => L.blocks[Math.floor(q / L.bs)] * L.bs + (q % L.bs));
        L.slots.push(...sl);
        return sl.map((x) => `${x}=p${Math.floor(x / L.bs)}:${x % L.bs}`).join(" ");
      }),
      S("qsl.append(qsl[-1] + n)", (L) => { L.qsl.push(L.qsl.at(-1) + L.n); return `[${L.qsl}]`; }),
      S("seq_lens.append(start + n)", (L) => { L.seq_lens.push(L.start + L.n); return `[${L.seq_lens}]`; }),
      S("bt[i, : len(blocks)] = blocks", (L) => { L.blocks.forEach((b, k) => (L.bt[L.i][k] = b)); return `row ${L.i} = [${L.bt[L.i]}]`; }),
      S("logit_rows.append(qsl[-1] - 1)", (L) => { L.logit_rows.push(L.qsl.at(-1) - 1); return `[${L.logit_rows}]`; }),
    ], { label: (L) => `i = ${L.i}, s = ${fmt(L.s)}` }),
    FOR("for _ in range(len(batch), pad_to):", (L) => range(L.batch.length, L.pad_to), () => {}, [
      S("ids.append(0); pos.append(0); slots.append(0)\nqsl.append(qsl[-1] + 1); seq_lens.append(1)", (L) => {
        L.ids.push(0); L.pos.push(0); L.slots.push(0); L._owner.push(null);
        L.qsl.push(L.qsl.at(-1) + 1); L.seq_lens.push(1);
        return "dummy row: writes the null page 0, never sampled";
      }),
    ]),
    S("max_query_len = max(s.num_new for s in batch) if batch else 1", (L) => { L.max_query_len = L.batch.length ? Math.max(...L.batch.map((s) => s.num_new)) : 1; }),
    IF("if pad_tokens:", (L) => !!L.pad_tokens, [S("ids += [0] * (pad_tokens - len(ids))\n...  # rows of no request, up to the mixed graph's size")]),
    RET("return PackedBatch(ids, pos, slots, qsl, seq_lens, bt, logit_rows,\n                   max_query_len, fill_rows, fill_src)", (L) => ({
      _x: `PackedBatch(T=${L.ids.length}, B=${L.seq_lens.length})`, _batch: L.batch,
      ids: L.ids, pos: L.pos, slots: L.slots, qsl: L.qsl, seq_lens: L.seq_lens, bt: L.bt, logit_rows: L.logit_rows,
      max_query_len: L.max_query_len, fill_rows: L.fill_rows, fill_src: L.fill_src, owner: L._owner,
    })),
  ],
});

// ============================================================== ModelRunner (GPU half)

def("ModelRunner.launch_forward", {
  mod: "runner", sig: "def launch_forward(self, prepared) -> torch.Tensor:", params: ["prepared"], body: [
    S("packed, mode, size = prepared", (L) => { [L.packed, L.mode, L.size] = L.prepared; return `mode = '${L.mode}'`; }),
    IF("if TP_SIZE > 1:", () => false, [S("tp_broadcast(prepared)")]),
    CALL("hidden = self.forward_step(packed, mode, size)", "ModelRunner.forward_step", (L) => [L.packed, L.mode, L.size], (L, r) => { L.hidden = r; return `← ${fmt(r)}`; }),
    RET("return self.model.logits(hidden[to_gpu(packed.logit_rows, torch.int64)])", (L) => ({ _t: [L.packed.logit_rows.length, "V"], _batch: L.packed._batch })),
  ],
});

def("ModelRunner.forward_step", {
  mod: "runner", sig: "def forward_step(self, packed, mode: str, size: int | None) -> torch.Tensor:", params: ["packed", "mode", "size"], body: [
    IF('if mode != "eager":', (L) => L.mode !== "eager", [
      CALL("T, ATTN_META = self._upload(packed, for_graph=True)", "ModelRunner._upload", (L) => [L.packed, true], (L, r) => { L.T = r[0]; return `← T = ${L.T}`; }),
      S("graph, out = self.graphs[(mode, size)]", (L) => { L.graph = X(`CUDAGraph(${L.mode}, ${L.size})`); }),
      CALL("graph.replay()                 # one launch, no allocs, same addresses", "Model.forward", (L, G) => [Tn(L.T), Tn(L.T)],
        (L, r) => { L.out = r; return "← out (the captured output buffer)"; }, { label: "replay of the captured graph" }),
      RET("return out", (L) => L.out),
    ]),
    CALL("T, ATTN_META = self._upload(packed, for_graph=False)", "ModelRunner._upload", (L) => [L.packed, false], (L, r) => { L.T = r[0]; return `← T = ${L.T}`; }),
    CALL("return self.model(self.input_ids[:T], self.positions[:T])", "Model.forward", (L) => [Tn(L.T), Tn(L.T)], undefined, { returns: true }),
  ],
});

def("ModelRunner._upload", {
  mod: "runner", sig: "def _upload(self, p: PackedBatch, for_graph: bool):", params: ["p", "for_graph"], body: [
    S("T, B = len(p.ids), len(p.seq_lens)", (L) => { L.T = L.p.ids.length; L.B = L.p.seq_lens.length; }),
    S("self.input_ids[:T].copy_(to_gpu(p.ids, torch.int64))\nself.positions[:T].copy_(to_gpu(p.pos, torch.int64))\nself.slot_mapping[:T].copy_(to_gpu(p.slots, torch.int64))\nself.query_start_loc[: B + 1].copy_(to_gpu(p.qsl, torch.int32))\nself.cu_seqlens_k[: B + 1].copy_(to_gpu([0] + np.cumsum(p.seq_lens).tolist(), torch.int32))\nself.block_table[:B].copy_(to_gpu(p.bt))",
      (L, G) => {
        const p = L.p;
        const cu = [0];
        for (const s of p.seq_lens) cu.push(cu.at(-1) + s);
        G.runner.buf = {
          input_ids: [...p.ids], positions: [...p.pos], slot_mapping: [...p.slots], qsl: [...p.qsl], seq_lens: [...p.seq_lens],
          cu_seqlens_k: cu, bt: p.bt.map((r) => [...r]), owner: [...p.owner], fill: [], logit_rows: [...p.logit_rows],
        };
        return "host lists → the persistent GPU buffers";
      }),
    IF("if p.fill_rows:", (L) => L.p.fill_rows.length > 0, [
      IF("if TP_SIZE > 1:", () => false, [S("dist.broadcast(self.prev_tokens, src=0)")]),
      S("rows, src = to_gpu(p.fill_rows, torch.int64), to_gpu(p.fill_src, torch.int64)", (L) => `rows [${L.p.fill_rows}], src [${L.p.fill_src}]`),
      S("self.input_ids[rows] = self.prev_tokens[src]", (L, G) => {
        const b = G.runner.buf;
        b.fill = L.p.fill_rows.map((r, k) => [r, L.p.fill_src[k]]);
        for (const [r, s] of b.fill) b.input_ids[r] = G.runner.prev_tokens[s];
        return b.fill.map(([r, s]) => `ids[${r}] ← prev_tokens[${s}] = '${tk(G.runner.prev_tokens[s])}'`).join(", ") + "  (GPU only, the CPU still has '?')";
      }),
    ]),
    S("max_seq = self.cfg.max_model_len if for_graph else max(p.seq_lens)", (L, G) => { L.max_seq = L.for_graph ? G.cfg.maxModelLen : Math.max(...L.p.seq_lens); }),
    S("meta = AttnMetadata(self.slot_mapping[:T], self.query_start_loc[: B + 1],\n                    self.cu_seqlens_k[: B + 1], self.block_table[:B], B,\n                    p.max_query_len, max_seq, p.qsl, p.seq_lens)",
      (L) => { L.meta = X("AttnMetadata"); return `qsl [${L.p.qsl}], seq_lens [${L.p.seq_lens}]`; }),
    RET("return T, meta", (L) => [L.T, L.meta]),
  ],
});

// ============================================================== Model

def("Model.forward", {
  mod: "model", sig: "def forward(self, input_ids, positions):", params: ["input_ids", "positions"], body: [
    S("cos, sin = self.cos[positions], self.sin[positions]", (L, G) => { const T = L.input_ids._t[0]; L.cos = Tn(T, G.cfg.D / 2); L.sin = Tn(T, G.cfg.D / 2); return "RoPE rows for each token's absolute position"; }),
    S("x, residual = self.embed_tokens(input_ids), None", (L, G) => { L.x = Tn(L.input_ids._t[0], G.cfg.hidden); L.residual = null; }),
    FOR("for layer in self.layers:", (L, G) => range(0, G.cfg.layers), (L, k, _, G) => { L.layer = X(`DecoderLayer(${k})`); G.layer = k; }, [
      CALL("x, residual = layer(x, residual, cos, sin)", "DecoderLayer.forward", (L) => [L.x, L.residual, L.cos, L.sin], (L, r) => { [L.x, L.residual] = r; return "← (x, residual)"; }),
    ], { label: (L, G, k) => `layer ${k} of ${G.cfg.layers}` }),
    RET("return add_rmsnorm(self.norm, x, residual)[0]     # final residual add + norm", (L, G) => { G.layer = null; return L.x; }),
  ],
});

def("DecoderLayer.forward", {
  mod: "model", sig: "def forward(self, x, residual, cos, sin):", params: ["x", "residual", "cos", "sin"], body: [
    S("h, residual = add_rmsnorm(self.input_layernorm, x, residual)", (L) => { L.h = L.x; const first = L.residual === null; L.residual = L.x; return first ? "first layer: residual = x" : "residual += x, then norm"; }),
    CALL("h, residual = add_rmsnorm(self.post_attention_layernorm,\n                          self.self_attn(h, cos, sin), residual)", "Attention.forward",
      (L) => [L.h, L.cos, L.sin], (L, r) => { L.h = r; return "← attn out, + residual, norm"; }),
    CALL("return self.mlp(h), residual", "MLP.forward", (L) => [L.h], undefined, { returns: (L, r) => [r, L.residual] }),
  ],
});

def("Attention.forward", {
  mod: "model", sig: "def forward(self, x, cos, sin):\n    # return self.output(self.attend(*self.project(x, cos, sin)))", params: ["x", "cos", "sin"], body: [
    CALL("q, k, v = self.project(x, cos, sin)", "Attention.project", (L) => [L.x, L.cos, L.sin], (L, r) => { [L.q, L.k, L.v] = r; return "← q, k, v"; }),
    CALL("out = self.attend(q, k, v)", "Attention.attend", (L) => [L.q, L.k, L.v], (L, r) => { L.out = r; return `← ${fmt(r)}`; }),
    CALL("return self.output(out)", "Attention.output", (L) => [L.out], undefined, { returns: true }),
  ],
});

def("Attention.project", {
  mod: "model", sig: "def project(self, x, cos, sin):", params: ["x", "cos", "sin"], body: [
    S("T = x.shape[0]", (L) => { L.T = L.x._t[0]; }),
    S("q, k, v = self.qkv_proj(x).split(self.qkv_sizes, dim=-1)\nq = q.view(T, self.H, self.D)\nk = k.view(T, self.Hkv, self.D)\nv = v.view(T, self.Hkv, self.D)",
      (L, G) => { const c = G.cfg; L.q = Tn(L.T, c.H, c.D); L.k = Tn(L.T, c.Hkv, c.D); L.v = Tn(L.T, c.Hkv, c.D); return `one GEMM, then views (GQA: ${c.H} q heads share ${c.Hkv} kv heads)`; }),
    IF("if TRITON_KERNELS:", () => true, [S("tk.qk_norm_rope(q, k, cos, sin, ...)       # q/k norm + RoPE, in place", () => "rotates q, k by each token's position")],
      [S("q, k = self.q_norm(q), self.k_norm(k)\nq, k = rotary(q, cos, sin), rotary(k, cos, sin)")]),
    RET("return q, k, v", (L) => [L.q, L.k, L.v]),
  ],
});

function buildAttn(G) {
  const b = G.runner.buf, bs = G.cfg.bs;
  const reqs = [];
  for (let i = 0; i + 1 < b.qsl.length; i++) {
    const s = b.qsl[i], e = b.qsl[i + 1], kv = b.seq_lens[i];
    if (kv === 0 || e === s) continue;
    const owner = b.owner[s];
    const ktoks = [];
    for (let t = 0; t < kv; t++) {
      const slot = b.bt[i][Math.floor(t / bs)] * bs + (t % bs);
      const c = G.pages[Math.floor(slot / bs)][slot % bs];
      ktoks.push(owner ? (c ? c.tok : "·") : "0");
    }
    reqs.push({ rid: owner, qtoks: b.input_ids.slice(s, e), ktoks, q0: kv - (e - s), kv, pages: b.bt[i].slice(0, Math.ceil(kv / bs)) });
  }
  return { layer: G.layer, reqs };
}

def("Attention.attend", {
  mod: "model", sig: "def attend(self, q, k, v):", params: ["q", "k", "v"], body: [
    S("meta = ATTN_META", (L) => { L.meta = X("AttnMetadata"); }),
    IF("if meta is None:", () => false, [RET("return q * 0")]),
    S("k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]", (L, G) => `layer ${G.layer}'s [${G.cfg.numBlocks}, ${G.cfg.bs}, ${G.cfg.Hkv}, ${G.cfg.D}] pools`),
    S("k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k)\nv_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v)", (L, G) => {
      const b = G.runner.buf, bs = G.cfg.bs, pages = {};
      b.slot_mapping.forEach((slot, row) => {
        const pg = Math.floor(slot / bs), o = slot % bs, rid = b.owner[row];
        pages[pg] = "write";
        if (G.layer !== 0) return;
        if (!rid) {
          G.pages[0][0] = { tok: "0", rid: null, gen: false };
          return;
        }
        const r = G.reqById(rid), pos = b.positions[row];
        G.pages[pg][o] = { tok: b.input_ids[row], rid, gen: pos >= r.prompt_len, stale: false };
        r.kvWritten = Math.max(r.kvWritten, pos + 1);
      });
      hl(G, { pages });
      return `${b.slot_mapping.length} rows → slots [${b.slot_mapping}]`;
    }),
    RET("return ATTN_FN(q, k_cache, v_cache, meta, self.D ** -0.5)", (L, G) => { G.attn = buildAttn(G); return L.q; }),
  ],
});

def("Attention.output", {
  mod: "model", sig: "def output(self, out):", params: ["out"], body: [
    RET("return tp_all_reduce(self.o_proj(out.reshape(out.shape[0], -1)))", (L, G) => Tn(L.out._t[0], G.cfg.hidden)),
  ],
});

def("MLP.forward", {
  mod: "model", sig: "def forward(self, x):", params: ["x"], body: [
    S("gate, up = self.gate_up_proj(x).chunk(2, dim=-1)", (L) => { L.gate = Tn(L.x._t[0], "I"); L.up = Tn(L.x._t[0], "I"); }),
    IF("if TRITON_KERNELS:", () => true, [RET("return tp_all_reduce(self.down_proj(tk.silu_and_mul(gate, up)))", (L, G) => Tn(L.x._t[0], G.cfg.hidden))]),
    RET("return tp_all_reduce(self.down_proj(F.silu(gate) * up))", (L, G) => Tn(L.x._t[0], G.cfg.hidden)),
  ],
});

// ============================================================== sampling

def("ModelRunner.launch_sample", {
  mod: "sample", sig: "def launch_sample(self, batch, logits) -> torch.cuda.Event:", params: ["batch", "logits"], body: [
    CALL("tokens = sample(logits, [s.req.params for s in batch], [s.req.token_ids for s in batch])", "sample",
      (L) => [L.logits, L.batch.map((s) => s.req.params), L.batch.map((s) => [...s.req.token_ids])],
      (L, r) => { L.tokens = r; return `← [${r.map(tk)}]`; }),
    S("B = len(batch)", (L) => { L.B = L.batch.length; }),
    S("self.prev_tokens[:B].copy_(tokens)", (L, G) => { L.tokens.forEach((t, i) => (G.runner.prev_tokens[i] = t)); return "kept on the GPU: next step's input ids come from here"; }),
    S("self.tokens_host[:B].copy_(tokens, non_blocking=True)", (L, G) => {
      L.tokens.forEach((t, i) => (G.runner.tokens_host[i] = t));
      G.runner.hostPending = true;
      return "async GPU → CPU copy started";
    }),
    S("done = torch.cuda.Event()\ndone.record()", (L) => { L.done = X("cuda.Event"); return "fires when that copy lands"; }),
    RET("return done", (L) => L.done),
  ],
});

def("sample", {
  mod: "sample", sig: "def sample(logits, params: list[SamplingParams], token_ids=None):", params: ["logits", "params", "token_ids"], body: [
    IF("if token_ids is not None:", (L) => L.token_ids != null, [
      IF("if any(p.return_logprobs for p in params):", () => false, [S("logits = logits.clone()")]),
      CALL("logits = apply_repetition_penalty(logits, params, token_ids)", "apply_repetition_penalty", (L) => [L.logits, L.params, L.token_ids], (L, r) => { L.logits = r; return "← logits (edited in place)"; }),
    ]),
    IF("if all(p.temperature == 0 for p in params):", (L) => L.params.every((p) => p.temperature === 0), [
      RET("return logits.argmax(-1)", (L, G) => L.logits._batch.map((s) => (completes(s) ? G.next(s.req, s.num_computed + s.num_new - 1) : "~"))),
    ]),
    S("ks = [1 if p.temperature == 0 else (p.top_k if p.top_k > 0 else V) for p in params]\n...  # top-k, temperature, top-p, Gumbel-max\n(not taken here: every request is greedy)"),
    RET("return torch.where(is_greedy, idx[:, 0], sampled)"),
  ],
});

def("apply_repetition_penalty", {
  mod: "sample", sig: "def apply_repetition_penalty(logits, params, token_ids):", params: ["logits", "params", "token_ids"], body: [
    S("rows, windows = [], []", (L) => { L.rows = []; L.windows = []; }),
    FOR("for i, (p, ids) in enumerate(zip(params, token_ids)):", (L) => L.params.map((p, i) => [p, L.token_ids[i]]), (L, [p, idsv], i) => { L.i = i; L.p = p; L.ids = idsv; }, [
      IF("if p.repetition_penalty != 1.0 and p.repeat_last_n != 0:", (L) => L.p.repetition_penalty !== 1 && L.p.repeat_last_n !== 0, [
        S("rows.append(i)", (L) => { L.rows.push(L.i); }),
        S("windows.append(ids if p.repeat_last_n < 0 else ids[-p.repeat_last_n:])", (L) => { L.windows.push(L.ids.slice(-L.p.repeat_last_n)); return toks(L.windows.at(-1)); }),
      ]),
    ], { label: (L) => `i = ${L.i}` }),
    IF("if not rows:", (L) => !L.rows.length, [RET("return logits", (L) => L.logits)]),
    S("n = max(len(w) for w in windows)", (L) => { L.n = Math.max(...L.windows.map((w) => w.length)); }),
    S("ids = to_gpu(np.array([w + [w[0]] * (n - len(w)) for w in windows]))   # [R, n]", (L) => { L.ids = Tn(L.rows.length, L.n); return "pad every window to n with its first id"; }),
    S("penalty = to_gpu([params[i].repetition_penalty for i in rows])[:, None]", (L) => { L.penalty = Tn(L.rows.length, 1); return "1.1 per row"; }),
    S("rows = to_gpu(rows, torch.long)[:, None]", (L) => { L.rows = Tn(L.rows.length, 1); }),
    S("seen = logits[rows, ids]", (L) => { L.seen = Tn(L.rows._t[0], L.n); return "the logits of recently seen tokens only"; }),
    S("logits[rows, ids] = torch.where(seen > 0, seen / penalty, seen * penalty)", () => "push them toward less likely"),
    RET("return logits", (L) => L.logits),
  ],
});

// ============================================================== serve: AsyncEngine + HTTP

def("AsyncEngine.__init__", {
  mod: "async", sig: "def __init__(self, engine: Engine, log_interval: float = 10.0):", params: ["engine", "log_interval"], body: [
    S("self.engine = engine"),
    S("self.log_interval = log_interval"),
    S("self.inbox = queue.Queue()", (L, G) => { G.async = { inbox: [], streams: {} }; return "web server → engine thread"; }),
    S("self.streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}", () => "engine thread → each client"),
    S("threading.Thread(target=self._loop, daemon=True).start()", (L, G) => { G.driver.spawn("engine thread", "engine", "AsyncEngine._loop", []); return "engine thread starts AsyncEngine._loop()"; }),
  ],
});

def("uvicorn.run", {
  mod: "async", sig: 'uvicorn.run(build_app(async_engine, a.model), host="0.0.0.0", port=a.port)', params: ["async_engine"], body: [
    S("app = build_app(async_engine, model_name)\n# POST /v1/completions → respond() → async_engine.generate(prompt, params)", () => "routes registered"),
    AWAIT("serve forever: each HTTP request runs AsyncEngine.generate on this event loop", { ready: () => false, take: () => {}, waitNote: "⏸ serving HTTP" }),
  ],
});

def("AsyncEngine._loop", {
  mod: "async", sig: "def _loop(self):", body: [
    WHILE("while True:", (L, G) => { G.arrivals(); return true; }, [
      S("block = not self.engine.has_work()", (L, G) => { L.block = !hasWork(G); return L.block ? "True: idle, sleep on the inbox" : `False: busy (${workStr(G)})`; }),
      WHILE("while True:", () => true, [
        AWAIT("kind, rid, payload = self.inbox.get(block=block, timeout=...)", {
          ready: (L, G) => G.async.inbox.length > 0,
          take: (L, G) => { const [kind, rid, payload] = G.async.inbox.shift(); Object.assign(L, { kind, rid, payload }); return `('${kind}', '${rid}', …)`; },
          empty: (L) => !L.block,
          waitNote: "⏸ inbox empty: waiting for a client",
        }),
        S("block = False", (L) => { L.block = false; }),
        IF('if kind == "add":', (L) => L.kind === "add", [
          CALL("self.engine.add_request(*payload, request_id=rid)", "Engine.add_request", (L) => [L.payload[0], L.payload[1], L.rid]),
        ], [S("self.engine.abort(rid)")]),
      ], { label: () => "True" }),
      CALL("outputs = self.engine.step()", "Engine.step", () => [], (L, r) => { L.outputs = r; return `← ${r.length} outputs`; }),
      FOR("for o in outputs:", (L) => L.outputs, (L, o) => { L.o = o; }, [
        S("stream = self.streams.get(o.request_id)", (L, G) => { L.stream = G.async.streams[L.o.request_id] ? X("(loop, q)") : null; }),
        IF("if stream is not None:", (L) => L.stream !== null, [
          S("loop, q = stream"),
          S("loop.call_soon_threadsafe(q.put_nowait, o)", (L, G) => {
            G.async.streams[L.o.request_id].queue.push(L.o);
            G.driver.wake(G.driver.thread(`client ${L.o.request_id}`), true);
            return `→ client ${L.o.request_id}: '${L.o.delta}'`;
          }),
        ]),
      ], { label: (L) => `o = ${fmt(L.o)}` }),
      IF("if self.log_interval:", () => true, [S("self.engine.maybe_log_stats(self.log_interval)")]),
    ], { label: (L, G, v, k) => `iter ${k}` }),
  ],
});

def("AsyncEngine.generate", {
  mod: "async", sig: "async def generate(self, prompt, params: SamplingParams):", params: ["prompt", "params"], body: [
    S("rid = uuid.uuid4().hex", (L) => { L.rid = L.params._rid; return `drawn as '${L.rid}'`; }),
    S("q: asyncio.Queue = asyncio.Queue()", (L) => { L.q = X("asyncio.Queue()"); }),
    S("self.streams[rid] = (asyncio.get_running_loop(), q)", (L, G) => { G.async.streams[L.rid] = { queue: [], received: "", finished: false }; }),
    S('self.inbox.put(("add", rid, (prompt, params)))', (L, G) => {
      G.async.inbox.push(["add", L.rid, [L.prompt, L.params]]);
      G.driver.wake(G.driver.thread("engine thread"));
      return `inbox = [${G.async.inbox.map((m) => m[1]).join(", ")}]`;
    }),
    S("finished = False", (L) => { L.finished = false; }),
    WHILE("while not finished:", (L) => !L.finished, [
      AWAIT("o = await q.get()", {
        ready: (L, G) => G.async.streams[L.rid].queue.length > 0,
        take: (L, G) => { L.o = G.async.streams[L.rid].queue.shift(); return `delta '${L.o.delta}'`; },
        waitNote: "⏸ awaiting the engine thread",
      }),
      S("finished = o.finished", (L) => { L.finished = L.o.finished; }),
      S("yield o # stream the token to the user", (L, G) => {
        const s = G.async.streams[L.rid];
        s.received += L.o.delta;
        s.finished = L.o.finished;
        return `client has '${s.received}'`;
      }),
    ]),
    S("self.streams.pop(rid, None)        # finally:", () => "stream closed"),
    IF("if not finished:", (L) => !L.finished, [S('self.inbox.put(("abort", rid, None))')]),
  ],
});
