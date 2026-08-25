// Per-op FLOPs/params formulas from the JAX scaling book
// (https://jax-ml.github.io/scaling-book/transformers/). FLOPs = 6 x params
// for every weighted matmul; the two attention matmuls have no params but
// still follow the "6x the multiply-count" convention. Reshapes, norm,
// mask, softmax, gate-multiply and residual adds are elementwise/shape ops
// with ~0 params and ~0 flops, called out honestly rather than faked.
//
// Every weighted step is expressed as an "expr" — a coefficient times a
// product of dimension symbols (e.g. { coef: 6, symbols: ["B","T","D","N","H"] }
// for 6BTDNH) — instead of a bare compute function, so the UI can show the
// derivation (symbols -> substituted numbers -> result), not just the
// final number.

export const DEFAULT_DIMS = {
  B: 1,
  T: 8192,
  S: 8192,
  D: 4096,
  F: 16384,
  N: 32,
  K: 32,
  H: 128,
};

export const DIM_STEP = 32;

export const DIM_RANGES = {
  B: { min: 1, max: 512 },
  T: { min: 1, max: 131072 },
  S: { min: 1, max: 131072 },
  D: { min: 128, max: 16384 },
  F: { min: 512, max: 65536 },
  N: { min: 1, max: 128 },
  K: { min: 1, max: 128 },
  H: { min: 16, max: 256 },
};

export const DIM_INFO = [
  { key: "B", label: "batch" },
  { key: "T", label: "sequence length (query)" },
  { key: "S", label: "sequence length (key/value)" },
  { key: "D", label: "d_model" },
  { key: "F", label: "MLP hidden dim" },
  { key: "N", label: "query heads" },
  { key: "K", label: "kv heads" },
  { key: "H", label: "head dim" },
];

// A weight matrix of shape (symbols...) has that many params; running it
// over B*T tokens costs 2x that many FLOPs forward + 4x backward = 6x, so
// its flopsExpr is always the same symbols prefixed with B,T and coef 6.
function weightExpr(symbols) {
  return {
    paramsExpr: { coef: 1, symbols },
    flopsExpr: { coef: 6, symbols: ["B", "T", ...symbols] },
  };
}

// The two attention matmuls (scores, weighted-sum) have no weights of
// their own — both operands are activations — so there's no paramsExpr,
// but they still cost 6x the multiply-count.
function bilinearExpr(symbols) {
  return {
    paramsExpr: null,
    flopsExpr: { coef: 6, symbols },
  };
}

const noCompute = { paramsExpr: null, flopsExpr: null };

// Assumed storage dtype for the KV cache — bf16, 2 bytes/element, matching
// the convention used elsewhere in this codebase (video_3's sharding
// visualizer) for "bytes on the wire" style accounting.
const KV_CACHE_BYTES_PER_ELEM = 2;

// The KV cache holds one K tensor and one V tensor per token, each shaped
// [B,S,K,H] (post-reshape, per-kv-head) — a memory footprint, not a
// compute cost, so it gets its own memExpr rather than params/flops.
function kvCacheExpr() {
  return {
    memExpr: { coef: 2 * KV_CACHE_BYTES_PER_ELEM, symbols: ["B", "S", "K", "H"] },
  };
}

// RMSNorm/LayerNorm has a real, computable cost — a per-channel scale
// weight (params = D) and an elementwise pass over every activation to
// compute the sum-of-squares, rescale, and apply that weight
// (flops ≈ 4·B·T·D: square, reduce, rsqrt-multiply, scale). This coefficient
// is our own elementwise-op estimate, forward-pass only — NOT the book's
// number, since the scaling book explicitly declines to derive one
// ("layernorms are comparatively cheap and can be ignored for first-order
// cost estimates") and doesn't use the ×6 train-step convention here. It's
// shown rather than hand-waved to zero, but it's O(D), next to the O(D²)
// projections around it, so it's excluded from the running totals below —
// same call the scaling book makes.
function normExpr() {
  return {
    paramsExpr: { coef: 1, symbols: ["D"] },
    flopsExpr: { coef: 4, symbols: ["B", "T", "D"] },
    excludedFromTotals: true,
  };
}

// Question 6 (scaling book): if activation checkpointing only saves the
// output of the 7 big matmuls (Q,K,V,O,in1,in2,out), everything else must
// be recomputed ("rematerialized") in the backward pass. The two attention
// matmuls (QKᵀ and softmax(QKᵀ)·V) are the expensive part to redo — their
// *inputs* (Q,K,V) were saved, but their own outputs weren't, and ∂L/∂W_O
// needs softmax(QKᵀ)·V back. Recomputing one matmul costs its forward-only
// FLOPs (coef 2, half of the usual 6x train coefficient — no backward
// needed for a value we're only using as an input, not differentiating
// through here); two such matmuls (S=T) sum to the book's 4·B·T²·N·H.
function rematExpr(symbols) {
  return { rematExpr: { coef: 2, symbols } };
}

// Metadata for the four activation-checkpointing categories (Question 6):
// which of the 7 big matmul outputs are saved, and what has to be
// recomputed from them in the backward pass. Colors are theme keys,
// resolved where used so this file stays free of the theme import.
export const CHECKPOINT_META = {
  saved: { label: "saved", detail: "checkpointed — one of the 7 matmul outputs", colorKey: "accent" },
  "remat-heavy": { label: "recomputed", detail: "rematerialized — real FLOPs cost", colorKey: "bad" },
  "remat-cheap": { label: "recomputed", detail: "rematerialized — elementwise, ~0 FLOPs", colorKey: "wire" },
  free: { label: "free", detail: "input/residual boundary — nothing to redo", colorKey: "soft" },
};

export const STEPS = [
  {
    id: 1,
    phase: "attn",
    node: "x-input",
    title: "Input X",
    body: "The residual stream enters the attention block. Nothing has been computed yet — this is just the tensor <b>X[B,T,D]</b> flowing in.",
    checkpoint: "free",
    ...noCompute,
  },
  {
    id: 2,
    phase: "attn",
    node: "norm1",
    title: "Norm",
    body: "A normalization layer (commonly RMSNorm in gated-MLP architectures like this one) rescales <b>X</b> before the attention projections. Per token it's a square + sum-reduce (mean of squares) + rsqrt-scale + weight-multiply pass over <b>D</b> elements — 4 elementwise ops, so flops ≈ 4·B·T·D (forward-pass only, not the ×6 train convention used for matmuls below). The book itself doesn't pin a coefficient here — it calls norm cost negligible and skips it — so this is shown for completeness, not hand-waved to zero, then excluded from the running totals because it's O(D) next to the O(D²) projections around it.",
    checkpoint: "remat-cheap",
    ...normExpr(),
  },
  {
    id: 3,
    phase: "attn",
    node: "q-proj",
    title: "Q projection",
    body: "Project the normalized input onto the query heads: <b>X[B,T,D] · W_Q[D,N,H] → Q[B,T,N,H]</b>.",
    checkpoint: "saved",
    ...weightExpr(["D", "N", "H"]),
  },
  {
    id: 4,
    phase: "attn",
    node: "k-proj",
    title: "K projection",
    body: "Project onto the key heads: <b>X[B,T,D] · W_K[D,K,H] → K[B,T,K,H]</b>. It's T-shaped, same as the input — this is only the newly-computed slice; it becomes S-long once appended to the KV cache below. Fewer KV heads than query heads (K ≤ N) is the multi-query/grouped-query trick.",
    checkpoint: "saved",
    ...weightExpr(["D", "K", "H"]),
  },
  {
    id: 5,
    phase: "attn",
    node: "v-proj",
    title: "V projection",
    body: "Project onto the value heads: <b>X[B,T,D] · W_V[D,K,H] → V[B,T,K,H]</b>. Same T-shaped, newly-computed slice as the K projection — same shape and cost.",
    checkpoint: "saved",
    ...weightExpr(["D", "K", "H"]),
  },
  {
    id: 6,
    phase: "attn",
    node: "reshape-qkv",
    title: "Reshape into heads",
    body: "Q's N heads are viewed as K groups of G = N/K heads each (<b>Q: BTNH → BTKGH</b>), so every group shares one KV head. K and V are already K-headed ([B,T,K,H]) and need no reshape of their own — they broadcast against Q's group axis G when scored. A pure layout change on Q — no compute, no params.",
    checkpoint: "remat-cheap",
    ...noCompute,
  },
  {
    id: 7,
    phase: "attn",
    node: "kv-cache",
    title: "KV cache (K, V)",
    body: "For autoregressive decoding, the newly-computed <b>K[B,T,K,H]</b> and <b>V[B,T,K,H]</b> slices are appended to the running cache so future tokens can reuse them instead of recomputing attention over the whole prefix — the full cache is <b>[B,S,K,H]</b>, S growing by T each step. It's a <b>memory</b> cost, not a FLOPs or params cost, and it's what actually limits how many sequences fit in memory at once. (Per layer, in bf16; multiply by the number of layers L for the model-wide cache.)",
    ...kvCacheExpr(),
  },
  {
    id: 8,
    phase: "attn",
    node: "attn-scores",
    title: "Attention scores + mask",
    body: "Compute <b>Q · Kᵀ → [B,T,S,N]</b> and add the causal mask — H is the contracted (summed-over) axis here, not part of the output shape; it shows up in the flops count below but not in the tensor's shape. No parameters live here — the cost scales with sequence length squared, which is why long context is expensive independent of model size.",
    checkpoint: "remat-heavy",
    ...bilinearExpr(["B", "T", "S", "N", "H"]),
    ...rematExpr(["B", "T", "S", "N", "H"]),
  },
  {
    id: 9,
    phase: "attn",
    node: "softmax",
    title: "Softmax",
    body: "Normalize the masked scores into attention weights along S. Elementwise + a reduction — negligible FLOPs next to the matmuls either side of it.",
    checkpoint: "remat-cheap",
    ...noCompute,
  },
  {
    id: 10,
    phase: "attn",
    node: "weighted-sum",
    title: "Weighted value sum",
    body: "Mix the values by the attention weights: <b>softmax(QKᵀ) · V → [B,T,N,H]</b>. Same shape/cost as the score matmul, and again no parameters.",
    checkpoint: "remat-heavy",
    ...bilinearExpr(["B", "T", "S", "N", "H"]),
    ...rematExpr(["B", "T", "S", "N", "H"]),
  },
  {
    id: 11,
    phase: "attn",
    node: "out-proj",
    title: "Output projection",
    body: "Reshape the per-head outputs back to <b>D</b> and project: <b>[B,T,N,H] · W_O[N,H,D] → [B,T,D]</b>.",
    checkpoint: "saved",
    ...weightExpr(["D", "N", "H"]),
  },
  {
    id: 12,
    phase: "attn",
    node: "residual1",
    title: "Residual add",
    body: "Add the attention block's output back onto the original stream: <b>X + Attention(norm(X))</b>. An elementwise add — free in params and flops.",
    checkpoint: "free",
    ...noCompute,
  },
  {
    id: 13,
    phase: "mlp",
    node: "norm2",
    title: "Norm",
    body: "A second normalization layer (same RMSNorm, ≈4·B·T·D flops as derived above) prepares the stream for the MLP block, and is likewise excluded from the running totals below.",
    checkpoint: "remat-cheap",
    ...normExpr(),
  },
  {
    id: 14,
    phase: "mlp",
    node: "in1-proj",
    title: "W_in1 projection",
    body: "First of the two gated MLP input projections: <b>X[B,T,D] · W_in1[D,F] → [B,T,F]</b>, later passed through GELU.",
    checkpoint: "saved",
    ...weightExpr(["D", "F"]),
  },
  {
    id: 15,
    phase: "mlp",
    node: "in2-proj",
    title: "W_in2 projection",
    body: "The parallel gating projection: <b>X[B,T,D] · W_in2[D,F] → [B,T,F]</b>, same shape and cost as W_in1.",
    checkpoint: "saved",
    ...weightExpr(["D", "F"]),
  },
  {
    id: 16,
    phase: "mlp",
    node: "gelu-gate",
    title: "GELU + gate multiply",
    body: "Apply GELU to one branch and multiply elementwise with the other: <b>gelu(W_in1(x)) * W_in2(x)</b>. Elementwise — no parameters, negligible flops next to the projections.",
    checkpoint: "remat-cheap",
    ...noCompute,
  },
  {
    id: 17,
    phase: "mlp",
    node: "out-proj-mlp",
    title: "W_out projection",
    body: "Project back down to model dimension: <b>[B,T,F] · W_out[F,D] → [B,T,D]</b>.",
    checkpoint: "saved",
    ...weightExpr(["D", "F"]),
  },
  {
    id: 18,
    phase: "mlp",
    node: "residual2",
    title: "Residual add",
    body: "Add the MLP block's output back onto the stream: <b>X + MLP(norm(X))</b>. One transformer layer complete.",
    checkpoint: "free",
    ...noCompute,
  },
];

export function fmtNum(n) {
  if (!Number.isFinite(n) || n === 0) return "0";
  const units = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [v, suffix] of units) {
    if (Math.abs(n) >= v) return `${(n / v).toFixed(2)}${suffix}`;
  }
  return String(n);
}

// Plain, comma-grouped formatting for the (comparatively small) dimension
// values substituted into a derivation — as opposed to fmtNum's K/M/B/T
// abbreviation, which is reserved for the large aggregate results.
export function fmtPlain(n) {
  return Math.round(n).toLocaleString("en-US");
}

export function exprValue(expr, dims) {
  if (!expr) return 0;
  return expr.coef * expr.symbols.reduce((acc, sym) => acc * dims[sym], 1);
}

export function stepParams(step, dims) {
  return exprValue(step.paramsExpr, dims);
}

export function stepFlops(step, dims) {
  return exprValue(step.flopsExpr, dims);
}

export function stepMem(step, dims) {
  return exprValue(step.memExpr, dims);
}

export function stepRemat(step, dims) {
  return exprValue(step.rematExpr, dims);
}

// Total extra FLOPs needed to rematerialize the backward pass if only the
// 7 main matmul outputs are checkpointed (scaling book Question 6) — the
// two attention matmuls recomputed to reconstruct softmax(QKᵀ)·V for
// ∂L/∂W_O. Sums to the book's 4·B·T·S·N·H (S=T gives 4BT²NH).
export function totalRemat(dims) {
  return STEPS.reduce((acc, s) => acc + stepRemat(s, dims), 0);
}

// Byte-oriented formatting for memory quantities (the KV cache), kept
// separate from fmtNum's K/M/B/T count abbreviation so a "1.00B" reading
// never gets mistaken for a billion when it means a byte.
export function fmtBytes(n) {
  if (!Number.isFinite(n) || n === 0) return "0 B";
  const units = [
    [1e12, "TB"],
    [1e9, "GB"],
    [1e6, "MB"],
    [1e3, "KB"],
  ];
  for (const [v, suffix] of units) {
    if (Math.abs(n) >= v) return `${(n / v).toFixed(2)} ${suffix}`;
  }
  return `${n} B`;
}

export function cumulativeTotals(dims, uptoStepIndex) {
  let params = 0;
  let flops = 0;
  let attnParams = 0;
  let attnFlops = 0;
  let mlpParams = 0;
  let mlpFlops = 0;
  for (let i = 0; i <= uptoStepIndex; i++) {
    const s = STEPS[i];
    if (s.excludedFromTotals) continue;
    const p = stepParams(s, dims);
    const f = stepFlops(s, dims);
    params += p;
    flops += f;
    if (s.phase === "attn") {
      attnParams += p;
      attnFlops += f;
    } else {
      mlpParams += p;
      mlpFlops += f;
    }
  }
  return { params, flops, attnParams, attnFlops, mlpParams, mlpFlops };
}

// Full-layer totals (attention + MLP), independent of which step is
// currently revealed in the stepper — used for the params/activations
// breakdown and the attention-vs-MLP crossover exploration.
export function layerTotals(dims) {
  return cumulativeTotals(dims, STEPS.length - 1);
}

// Assuming S = T (the query attends over a KV cache that has grown to the
// same length, the book's simplifying assumption), attention FLOPs grow
// with T^2 while MLP FLOPs grow linearly with T. Solving
// attnFlops(T) = mlpFlops(T) for T gives the crossover sequence length
// where attention overtakes the MLP as the FLOPs-dominant block:
//
//   12*D*(N*H+K*H) + 12*T*N*H = 18*D*F
//   T* = (3*D*F) / (2*N*H)  -  D*(1 + K/N)
//
// Returns null if N*H is 0 (degenerate) or if the crossover is never
// reached for positive T (attention already dominates at any T > 0).
export function crossoverT(dims) {
  const { D, F, N, K, H } = dims;
  const NH = N * H;
  if (NH <= 0) return null;
  const t = (3 * D * F) / (2 * NH) - D * (1 + K / N);
  return t > 0 ? t : null;
}
