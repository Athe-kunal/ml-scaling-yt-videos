import { KV } from "./kv.js";
import { call, emit, makeWorld, newReq, writeKV } from "./trace.js";

// Replays tinyserve.kv_cache_demo() against the JS KVCacheManager.
// Token ids in the Python are drawn as letters: see TOKEN_LEGEND.

const L = (t) => {
  if (t >= 10 && t <= 17) return "ABCDEFGH"[t - 10];
  if (t === 77) return "X";
  if (t === 88) return "Y";
  if (t >= 200 && t < 220) return "abcdefghijklmnopqrst"[t - 200];
  if (t >= 1 && t <= 4) return "PQRS"[t - 1];
  return String(t);
};
const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

export const TOKEN_LEGEND = [
  ["10…17", "A…H"], ["77, 88", "X Y"], ["200…219", "a…t"], ["1…4", "P…S"],
];

export function runDemo(bs) {
  const w = makeWorld(bs);
  const mgr = new KV(w, 8, bs);
  const D = (group, tag, kind, title, extra = {}) => {
    w.group = group;
    emit(w, { code: "demo", tags: [tag], kind, title, ...extra });
  };
  // a.num_computed_tokens = v: the demo's stand-in for "the forward pass wrote K/V"
  const compute = (group, tag, r, v, title) => {
    w.group = group;
    r.num_computed = v;
    const pages = writeKV(w, r, Math.max(r.kvWritten, r.hit), v);
    emit(w, { code: "demo", tags: [tag], kind: "run", req: r.id, pages, title });
  };

  D(0, "d0", "init", `8 pages × ${bs} slots · page 0 = null`);

  // 1. chunked prefill
  const a = newReq(w, "A", [10, 11, 12, 13, 14, 15].map(L));
  D(1, "d1", "arrive", "A: 6-token prompt", { req: "A" });
  w.group = 1;
  call(w, "demo", "d1a", () => mgr.findCachedPrefix(a), "lookup");
  call(w, "demo", "d1a", () => mgr.allocate(a, 4), "allocate(a, 4)");
  compute(1, "d1b", a, 4, "A: 4 tokens computed");
  call(w, "demo", "d1c", () => mgr.allocate(a, 2), "allocate(a, 2)");
  compute(1, "d1d", a, 6, "A: prompt done");

  // 2. decode
  for (const t of [16, 17]) {
    a.token_ids.push(L(t));
    D(2, "d2", "run", `A appends '${L(t)}'`, { req: "A", sampled: a.token_ids.length - 1 });
    call(w, "demo", "d2a", () => mgr.allocate(a, 1), "allocate(a, 1)");
    compute(2, "d2b", a, a.num_computed + 1, `A: ${a.num_computed + 1} computed`);
  }

  // 3. B shares first block only
  const b = newReq(w, "B", [10, 11, 12, 13, 77, 88].map(L));
  D(3, "d3", "arrive", "B: same first block as A", { req: "B" });
  w.group = 3;
  const hits = call(w, "demo", "d3a", () => mgr.findCachedPrefix(b));
  call(w, "demo", "d3b", () => mgr.allocate(b, 2, hits), `allocate(b, 2, [${hits.join(", ")}])`);
  b.hit = hits.length * bs;
  b.kvWritten = b.hit;
  compute(3, "d3c", b, hits.length * bs + 2, `B: ${hits.length * bs} cached + 2 computed`);

  // 4. why max_blocks uses num_tokens-1
  const c = newReq(w, "C", a.token_ids.slice(0, 8));
  D(4, "d4", "arrive", "C: exactly A's 8 tokens", { req: "C" });
  w.group = 4;
  call(w, "demo", "d4a", () => mgr.findCachedPrefix(c));

  // 5. free A
  w.group = 5;
  call(w, "demo", "d5", () => mgr.freeRequest(a));
  w.done.push(a);

  // 6. eviction
  const d = newReq(w, "D", range(200, 220).map(L));
  D(6, "d6", "arrive", `D: 20 tokens = ${Math.ceil(20 / bs)} pages`, { req: "D" });
  w.group = 6;
  const okD = call(w, "demo", "d6a", () => mgr.allocate(d, 20), "allocate(d, 20)");
  if (okD) compute(6, "d6b", d, 20, "D: 20 computed");
  else D(6, "d6b", "fail", "D does not fit: nothing written");

  // 7. OOM on pages
  const e = newReq(w, "E", [1, 2, 3, 4].map(L));
  D(7, "d7", "arrive", "E: 4 tokens", { req: "E" });
  w.group = 7;
  call(w, "demo", "d7a", () => mgr.allocate(e, 4), "allocate(e, 4)");
  return { events: w.events, hashInfo: w.hashInfo };
}
