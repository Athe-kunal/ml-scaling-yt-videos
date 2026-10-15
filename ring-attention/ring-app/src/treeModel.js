import { T } from "./theme";

// Tree attention for decoding: one query token, KV cache sharded over N GPUs
// (GPU i holds tokens [i*PER, (i+1)*PER)). Each GPU attends to its shard,
// then two tree allreduces combine the partial softmaxes:
//   1) max of the local lse     -> shared shift m (keeps exp() <= 1)
//   2) sum of (num, den) where num_i = o_i e^{lse_i - m}, den_i = e^{lse_i - m}
// out = num / den. Each allreduce = log2(N) reduce levels up to rank 0, then
// log2(N) broadcast levels back down (binomial tree).

export const D = 4; // head dim
export const PER = 3; // keys per GPU

export const TREE_PHASE = {
  init: { label: "setup", color: T.soft },
  local: { label: "local attention", color: T.accent },
  up: { label: "reduce ↑", color: T.wire },
  down: { label: "broadcast ↓", color: "#5FD3F3" },
  rescale: { label: "rescale", color: T.accent2 },
  final: { label: "divide", color: T.soft },
};

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const lse = (xs) => {
  const m = Math.max(...xs);
  return m + Math.log(xs.reduce((a, x) => a + Math.exp(x - m), 0));
};

export const log2 = (N) => Math.round(Math.log2(N));

export function makeData(N) {
  const R = mulberry32(20241);
  const g = (s) => (R() * 2 - 1) * s;
  const q = Array.from({ length: D }, () => g(2));
  const gpus = Array.from({ length: N }, (_, i) => {
    const keys = Array.from({ length: PER }, () => Array.from({ length: D }, () => g(1.5)));
    const vals = Array.from({ length: PER }, () => Array.from({ length: D }, () => g(1)));
    const scores = keys.map((k) => k.reduce((a, x, t) => a + x * q[t], 0) / Math.sqrt(D));
    const l = lse(scores);
    const o = Array.from({ length: D }, (_, t) => scores.reduce((a, s, j) => a + Math.exp(s - l) * vals[j][t], 0));
    return { i, keys, vals, scores, lse: l, o };
  });
  const all = gpus.flatMap((gp) => gp.scores.map((s, j) => ({ s, v: gp.vals[j] })));
  const L = lse(all.map((x) => x.s));
  const exact = Array.from({ length: D }, (_, t) => all.reduce((a, x) => a + Math.exp(x.s - L) * x.v[t], 0));
  return { N, q, gpus, exact, lseAll: L };
}

export function buildPhases(N) {
  const L = log2(N);
  const ph = [{ type: "init" }, { type: "local" }];
  for (const op of ["max", "sum"]) {
    if (op === "sum") ph.push({ type: "rescale" });
    for (let lvl = 1; lvl <= L; lvl++) ph.push({ type: "up", op, lvl, step: 2 ** (lvl - 1) });
    for (let lvl = L; lvl >= 1; lvl--) ph.push({ type: "down", op, lvl, step: 2 ** (lvl - 1) });
  }
  ph.push({ type: "final" });
  return ph;
}

// Role of `rank` at one tree level (mirrors tree_allreduce in the code panel).
export function role(rank, phase) {
  const { step, type } = phase;
  const m = rank % (2 * step);
  if (type === "up") return m === 0 ? { kind: "recv", peer: rank + step } : m === step ? { kind: "send", peer: rank - step } : { kind: "idle" };
  return m === 0 ? { kind: "send", peer: rank + step } : m === step ? { kind: "recv", peer: rank - step } : { kind: "idle" };
}

// Replay phases [0..idx] and return the state after phase idx.
//   gpus[i]: { mx, num, den, out } buffers (null until set)
//   nodes:   `${op}:${lvl}:${k}` -> reduced value over GPUs [k 2^lvl, (k+1) 2^lvl)
//   msgs:    messages sent during phase idx
export function stateAt(data, phases, idx) {
  const { N, gpus: G } = data;
  const st = G.map(() => ({ local: false, mx: null, num: null, den: null, out: null, got: false }));
  const nodes = {};
  let msgs = [];
  const add = (a, b) => a.map((x, t) => x + b[t]);
  for (let p = 0; p <= idx; p++) {
    const ph = phases[p];
    msgs = [];
    if (ph.type === "local") st.forEach((s, i) => { s.local = true; s.mx = G[i].lse; });
    if (ph.type === "rescale") st.forEach((s, i) => {
      const w = Math.exp(G[i].lse - s.mx);
      s.w = w; s.num = G[i].o.map((x) => x * w); s.den = w; s.got = false;
    });
    if (ph.type === "up" || ph.type === "down") {
      const snap = st.map((s) => ({ mx: s.mx, num: s.num, den: s.den }));
      for (let r = 0; r < N; r++) {
        const ro = role(r, ph);
        if (ro.kind !== "recv") continue;
        const src = snap[ro.peer], me = st[r];
        if (ph.op === "max") me.mx = ph.type === "up" ? Math.max(me.mx, src.mx) : src.mx;
        else if (ph.type === "up") { me.num = add(me.num, src.num); me.den = me.den + src.den; }
        else { me.num = src.num; me.den = src.den; }
        if (ph.type === "down") me.got = true;
        msgs.push({ from: ro.peer, to: r, val: ph.op === "max" ? src.mx : src.den, vec: ph.op === "sum" ? src.num : null });
      }
      if (ph.type === "up") {
        for (let k = 0; k < N / 2 ** ph.lvl; k++) {
          const owner = k * 2 ** ph.lvl;
          nodes[`${ph.op}:${ph.lvl}:${k}`] = ph.op === "max" ? st[owner].mx : st[owner].den;
        }
      }
    }
    if (ph.type === "final") st.forEach((s) => { s.out = s.num.map((x) => x / s.den); });
  }
  return { gpus: st, nodes, msgs };
}

export const fmt = (x, d = 2) => (x == null ? "—" : (Math.abs(x) < 0.005 && x !== 0 ? x.toExponential(1) : x.toFixed(d)));
export const vec = (v, d = 2) => (v ? `[${v.map((x) => fmt(x, d)).join(", ")}]` : "—");
