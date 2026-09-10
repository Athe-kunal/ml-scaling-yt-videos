// Shared builder for the per-device diagram rows used by DP/ZeRO-1/2/3's
// STEPS arrays. Each device has 3 rows (weights, grads, optimizer state),
// one entry per weight — "in" and "out", matching the book's 2-matmul MLP
// (Win[D,F], Wout[F,D]) that fsdp.js/tp.js also use, rather than a toy
// numbered W_1..W_4 chain. Each entry is a { state } (or { state, label }
// for weights/grads, whose label changes across steps e.g. "Win[D,F]" ->
// "Win'[D,F]").
//
// state is one of:
//   "hidden" - not materialized on this device at all (ZeRO-3 unowned shard)
//   "ghost"  - present but borrowed/not-yet-synced (dashed, e.g. a
//              temporarily all-gathered weight, or a not-yet-reduced grad)
//   "solid"  - present and settled
//   "active" - present and is what this step is highlighting (accent)

export const LAYER_IDS = ["in", "out"];

export function weightsRow(states, labelFn) {
  return LAYER_IDS.map((k) => ({ i: k, label: labelFn(k), state: states[k] || "hidden" }));
}

export function gradsRow(states, labelFn) {
  return LAYER_IDS.map((k) => ({ i: k, label: labelFn(k), state: states[k] || "hidden" }));
}

export function optimRow(states) {
  return LAYER_IDS.map((k) => ({ i: k, state: states[k] || "hidden" }));
}

export const ALL = { in: "solid", out: "solid" };
export const NONE = {};
