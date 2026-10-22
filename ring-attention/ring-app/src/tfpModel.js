// Toy numbers for "Tree attention from first principles": one query, 8 keys,
// 2-D vectors so every number fits on screen. Scores are q·k (no 1/√d, as in
// the paper's equations).

export const Q = [1.2, -0.4];
export const K = [[0.9, 0.2], [-0.5, 1.0], [1.4, -0.3], [0.1, 0.8], [-1.1, -0.6], [0.7, 0.9], [1.6, 0.4], [-0.3, -1.2]];
export const V = [[1, 0], [0, 1], [2, 1], [1, 2], [-1, 0], [0, -1], [3, 1], [1, -2]];
export const NTOK = K.length;

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const scores = (alpha = 1) => K.map((k) => alpha * dot(Q, k));
export const lse = (xs) => {
  const m = Math.max(...xs);
  return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0));
};

// exact attention for the toy query
export function attention(alpha = 1) {
  const s = scores(alpha);
  const L = lse(s);
  const p = s.map((x) => Math.exp(x - L));
  return [0, 1].map((c) => p.reduce((acc, w, a) => acc + w * V[a][c], 0));
}

// which keys GPU i holds when the 8 keys are split over P GPUs
export const shard = (P, i) => Array.from({ length: NTOK / P }, (_, j) => i * (NTOK / P) + j);

// per-GPU partials of Algorithm 3: local (o_i, lse_i), then the global max m,
// then n_i = o_i e^{lse_i - m}, d_i = e^{lse_i - m}
export function partials(P, alpha = 1) {
  const s = scores(alpha);
  const gpus = Array.from({ length: P }, (_, i) => {
    const idx = shard(P, i);
    const li = lse(idx.map((a) => s[a]));
    const o = [0, 1].map((c) => idx.reduce((acc, a) => acc + Math.exp(s[a] - li) * V[a][c], 0));
    return { i, idx, lse: li, o };
  });
  const m = Math.max(...gpus.map((g) => g.lse));
  gpus.forEach((g) => {
    g.w = Math.exp(g.lse - m);
    g.n = g.o.map((x) => x * g.w);
    g.d = g.w;
  });
  const n = [0, 1].map((c) => gpus.reduce((acc, g) => acc + g.n[c], 0));
  const d = gpus.reduce((acc, g) => acc + g.d, 0);
  return { gpus, m, n, d, z: n.map((x) => x / d) };
}

// the energy F(ζ) = log Σ_a exp(q·k_a + ζ·v_a); its gradient at ζ = 0 is attention
export const energy = (zeta, alpha = 1) => lse(scores(alpha).map((x, a) => x + dot(zeta, V[a])));

// float16 can hold at most 65504, so e^x overflows once x > ln(65504) ≈ 11.09
export const FP16_MAX = 65504;
export const fp16exp = (x) => (Math.exp(x) > FP16_MAX ? Infinity : Math.exp(x));

export const f = (x, d = 2) => {
  if (x === Infinity) return "inf";
  if (Number.isNaN(x)) return "NaN";
  if (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
  return x.toFixed(d);
};
export const fv = (v, d = 2) => `[${v.map((x) => f(x, d)).join(", ")}]`;
