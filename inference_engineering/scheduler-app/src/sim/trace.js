// Event recorder shared by the KV manager, the scheduler and the demo script.
// Every emit() stores a deep snapshot of the whole world plus which code lines
// to light up, so the UI can step backwards and forwards freely.

export const KIND = {
  init: { color: "#94A1B2", label: "setup" },
  arrive: { color: "#8AB4FF", label: "arrive" },
  p1: { color: "#6EF3DD", label: "1 · running first" },
  p2: { color: "#FFC94D", label: "2 · admit waiting" },
  preempt: { color: "#FF8A8A", label: "preempt" },
  kv: { color: "#FF8AC6", label: "allocate" },
  hash: { color: "#C49CFF", label: "hash" },
  hit: { color: "#A5D6A7", label: "cache hit" },
  miss: { color: "#FF8A8A", label: "cache miss" },
  evict: { color: "#FFB86B", label: "evict" },
  free: { color: "#5FD3F3", label: "free" },
  fail: { color: "#FF8A8A", label: "no room" },
  emit: { color: "#9FE8D9", label: "3 · emit batch" },
  run: { color: "#E3E9F0", label: "forward" },
  finish: { color: "#A5D6A7", label: "finish" },
};

export const REQ_COLORS = ["#5FD3F3", "#FF8AC6", "#FFC94D", "#C49CFF", "#A5D6A7", "#FFB86B"];

export const short = (h) => (h == null ? "None" : h.slice(0, 4));

export function makeWorld(bs) {
  return {
    bs, kv: null, reqs: [], waiting: [], running: [], done: [], pages: [],
    frames: [], ctx: [], batch: null, step: -1, group: 0, events: [], hashInfo: {},
  };
}

export function newReq(w, id, tokens, extra = {}) {
  const r = {
    id, color: REQ_COLORS[w.reqs.length % REQ_COLORS.length], token_ids: [...tokens], prompt_len: tokens.length,
    num_computed: 0, block_ids: [], block_hashes: [], num_cached: 0,
    kvWritten: 0, hit: 0, finish_reason: null, arrival: 0, outputs: [], maxTokens: 0, ...extra,
  };
  w.reqs.push(r);
  return r;
}

// Python's hash((parent, tuple(tokens))), made deterministic: FNV-1a, 8 hex chars.
export function blockHash(w, parent, toks) {
  const s = `${parent ?? "None"}|${toks.join(",")}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const hex = h.toString(16).padStart(8, "0");
  w.hashInfo[hex] = { parent, toks: [...toks] };
  return hex;
}

// call stack of Python frames, shown as the "variables" panel
export function enter(w, fn, vars = {}) {
  const f = { fn, vars };
  w.frames.push(f);
  return f;
}
export const leave = (w) => w.frames.pop();

// run fn while `tag` in `code` is the calling line (drawn dimmed, with its note)
export function call(w, code, tag, fn, note) {
  w.ctx.push({ code, tag, note });
  try {
    return fn();
  } finally {
    w.ctx.pop();
  }
}

// write K/V for positions [from, to) of req into its pages
export function writeKV(w, req, from, to) {
  const touched = {};
  for (let p = from; p < to; p++) {
    const b = req.block_ids[Math.floor(p / w.bs)];
    w.pages[b][p % w.bs] = { tok: req.token_ids[p], rid: req.id, gen: p >= req.prompt_len, stale: false };
    touched[b] = "write";
  }
  req.kvWritten = Math.max(req.kvWritten, to);
  return touched;
}

function snapshot(w) {
  const kv = w.kv;
  return {
    ref: [...kv.ref],
    bh: [...kv.bh],
    free: [...kv.free],
    h2b: [...kv.h2b.entries()],
    reqs: w.reqs.map((r) => ({
      ...r, token_ids: [...r.token_ids], block_ids: [...r.block_ids], block_hashes: [...r.block_hashes], outputs: undefined,
    })),
    waiting: w.waiting.map((r) => r.id),
    running: w.running.map((r) => r.id),
    done: w.done.map((r) => r.id),
    pages: w.pages.map((p) => p.map((s) => s && { ...s })),
    frames: structuredClone(w.frames),
    batch: w.batch && w.batch.map((s) => ({ ...s })),
    step: w.step,
  };
}

// ev: { code, tags, kind, notes: [{tag, t}], req, pages: {b: label}, hashes, lookup, evicted, title }
export function emit(w, ev) {
  const color = KIND[ev.kind].color;
  const focus = {};
  const notes = {};
  for (const c of w.ctx) {
    (focus[c.code] ??= {})[c.tag] = `dim:${color}`;
    if (c.note) (notes[c.code] ??= []).push({ tag: c.tag, t: c.note });
  }
  for (const t of ev.tags || []) (focus[ev.code] ??= {})[t] = color;
  (notes[ev.code] ??= []).push(...(ev.notes || []));
  w.events.push({
    ...ev, color, focus, notes, group: w.group,
    title: ev.title ?? (ev.notes && ev.notes[0] ? ev.notes[0].t : KIND[ev.kind].label),
    snap: snapshot(w),
  });
}
