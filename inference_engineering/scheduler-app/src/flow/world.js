// The simulated world the flowcharts act on (G), and the plain snapshot of it
// that every event stores for the side panels.

export const MODS = {
  main: { color: "#94A1B2", label: "main" },
  engine: { color: "#6EF3DD", label: "Engine" },
  sched: { color: "#FFC94D", label: "Scheduler" },
  kv: { color: "#FF8AC6", label: "KVCacheManager" },
  runner: { color: "#C49CFF", label: "ModelRunner" },
  model: { color: "#8AB4FF", label: "Model" },
  sample: { color: "#FFB86B", label: "sampler" },
  detok: { color: "#A5D6A7", label: "Detokenizer" },
  async: { color: "#5FD3F3", label: "AsyncEngine / HTTP" },
};

export const REQ_COLORS = ["#5FD3F3", "#FF8AC6", "#FFC94D", "#C49CFF", "#A5D6A7", "#FFB86B"];
export const EOS = "<e>";

export const DEFAULTS = { mode: "generate", bs: 4, numBlocks: 8, budget: 6, maxSeqs: 4, maxModelLen: 32, layers: 2, maxSteps: 40 };

// Letter tokens. Prompts share the prefix ABCDEFGH (two full blocks at block size 4).
// `out` is what the toy model "predicts" (greedy), <e> is EOS. `arrive` is the
// engine step a client sends the request in serve mode.
export const REQUESTS = [
  { id: "R1", prompt: "ABCDEFGHIJ", out: "pqrs", arrive: 0 },
  { id: "R2", prompt: "ABCDEFGHKL", out: ["s", "t", EOS], arrive: 1 },
  { id: "R3", prompt: "MNOPQRSTU", out: "uvwx", arrive: 1 },
  { id: "R4", prompt: "ABCDEFGHIJY", out: "yzab", arrive: 6 },
];
export const MAX_TOKENS = 4;

export function makeWorld(cfg) {
  const G = {
    cfg: { ...cfg, H: 4, Hkv: 2, D: 8, hidden: 32 },
    stepNo: 0,
    hl: null,
    hashInfo: {},
    reqs: [], // every Request object created so far
    clients: REQUESTS.map((r, k) => ({ ...r, color: REQ_COLORS[k], started: false })),
    engine: null,
    sched: null,
    kv: null,
    runner: null,
    pages: [],
    detok: {},
    attn: null,
    layer: null,
    meta: null,
    async: null,
    outputs: {},
  };

  // the toy model: the logit row at position pos predicts token pos + 1
  G.next = (req, pos) => {
    const k = pos + 1 - req.prompt_len;
    return k >= 0 ? req.out[k] ?? EOS : "~";
  };
  G.reqById = (id) => G.reqs.find((r) => r.id === id);
  G.newReq = (rid, ids, params) => {
    const c = G.clients.find((x) => x.id === rid);
    const r = {
      _req: true, id: rid, color: c.color, out: [...c.out], token_ids: [...ids], prompt_len: ids.length, params,
      num_computed_tokens: 0, pending: 0, pending_row: -1, block_ids: [], block_hashes: [], num_cached_tokens: 0,
      finished: false, finish_reason: null, was_preempted: false, was_chunked_prefill: false,
      kvWritten: 0, hit: 0,
    };
    G.reqs.push(r);
    return r;
  };
  return G;
}

// Python's hash((parent, tuple(tokens))), made deterministic: FNV-1a, 8 hex chars
export function blockHash(G, parent, toks) {
  const s = `${parent ?? "None"}|${toks.join(",")}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const hex = h.toString(16).padStart(8, "0");
  G.hashInfo[hex] = { parent, toks: [...toks] };
  return hex;
}

const srView = (s) => ({ rid: s.req.id, num_computed: s.num_computed, num_new: s.num_new });

export function snapshotWorld(G) {
  const kv = G.kv, rn = G.runner, e = G.engine;
  return {
    step: G.stepNo,
    reqs: G.reqs.map((r) => ({
      id: r.id, color: r.color, token_ids: [...r.token_ids], prompt_len: r.prompt_len, num_computed: r.num_computed_tokens,
      block_ids: [...r.block_ids], block_hashes: [...r.block_hashes], num_cached: r.num_cached_tokens, kvWritten: r.kvWritten,
      hit: r.hit, pending: r.pending, finished: r.finished, finish_reason: r.finish_reason, was_preempted: r.was_preempted,
    })),
    clients: G.clients.map((c) => ({ id: c.id, color: c.color, started: c.started })),
    waiting: G.sched ? G.sched.waiting.map((r) => r.id) : [],
    running: G.sched ? G.sched.running.map((r) => r.id) : [],
    done: G.reqs.filter((r) => r.finished).map((r) => r.id),
    kv: kv && { ref: [...kv.ref], bh: [...kv.bh], free: [...kv.free], h2b: [...kv.h2b.entries()] },
    pages: G.pages.map((p) => p.map((s) => s && { ...s })),
    next_step: e && e.next_step && { batch: e.next_step[0].map(srView), mode: e.next_step[1][1], size: e.next_step[1][2] },
    in_flight: e && e.in_flight && { batch: e.in_flight[0].map(srView), live: [...e.in_flight[1]] },
    runner: rn && {
      graphs: rn.graphs.map((g) => `${g[0]}:${g[1]}`), prev_tokens: [...rn.prev_tokens], tokens_host: [...rn.tokens_host],
      hostPending: rn.hostPending, buf: rn.buf && JSON.parse(JSON.stringify(rn.buf)), fwd: rn.fwd && rn.fwd.map(srView),
    },
    attn: G.attn && JSON.parse(JSON.stringify(G.attn)),
    layer: G.layer,
    outputs: { ...G.outputs },
    async: G.async && {
      inbox: G.async.inbox.map((m) => `${m[0]} ${m[1]}`),
      streams: Object.fromEntries(Object.entries(G.async.streams).map(([k, s]) => [k, { queued: s.queue.length, received: s.received, finished: s.finished }])),
    },
  };
}
