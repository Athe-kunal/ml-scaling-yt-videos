import { T } from "./theme";

// Shared model for both views. N GPUs in a ring; GPU r sends to next =
// (r+1)%N and receives from prev = (r-1)%N. At step s, GPU r holds the KV
// block that started on GPU j = (r-s)%N.
//
// Everything is drawn on a 2N-chunk grid (chunk = S/2N tokens) so the two
// views share one attention matrix:
//   ring:   GPU r owns chunks [2r, 2r+1]            (one contiguous block)
//   zigzag: GPU r owns chunks [r, 2N-1-r]           (one early, one late)
// Work is counted in chunk×chunk units (a causal diagonal chunk = 0.5).

export const DEV_COLORS = ["#8AB4FF", "#FF8AC6", "#7EE787", "#FFB86B", "#C49CFF", "#5FD3F3", "#F2E366", "#FF7B7B"];

export const PHASE = {
  init: { label: "setup", color: T.soft },
  send: { label: "send / recv", color: T.wire },
  compute: { label: "compute", color: T.accent },
  merge: { label: "merge out, lse", color: T.accent2 },
  wait: { label: "wait + swap", color: T.wire },
  final: { label: "finalize", color: T.soft },
};

export const mod = (a, n) => ((a % n) + n) % n;

const SUB = "₀₁₂₃₄₅₆₇₈₉";
export const sub = (n) => String(n).split("").map((c) => SUB[+c]).join("");

// One phase per highlighted region of the code. The last step posts no
// send (step + 1 == world_size), so it has no send / wait phases.
export function buildPhases(N) {
  const ph = [{ type: "init", s: -1 }];
  for (let s = 0; s < N; s++) {
    const last = s === N - 1;
    if (!last) ph.push({ type: "send", s });
    ph.push({ type: "compute", s });
    ph.push({ type: "merge", s });
    if (!last) ph.push({ type: "wait", s });
  }
  ph.push({ type: "final", s: N - 1 });
  return ph;
}

export const STEP_PHASES = ["send", "compute", "merge", "wait"];

// Chunk ids a GPU's Q (or a KV block from GPU j) covers.
export function ownChunks(variant, N, r) {
  return variant === "ring" ? [2 * r, 2 * r + 1] : [r, 2 * N - 1 - r];
}

// What GPU r does at step s. `branch` names the code path taken, `cells`
// the attention-matrix rectangles it fills (in chunk coordinates), `useQ` /
// `useKV` which halves of the local Q / held KV the kernel reads.
export function deviceStep(variant, N, r, s) {
  const j = mod(r - s, N);
  if (variant === "ring") {
    const blk = (q, k, tri = false) => ({ q0: 2 * q, qn: 2, k0: 2 * k, kn: 2, tri });
    if (s === 0) return { j, branch: "0", active: true, cells: [blk(r, r, true)], work: 2, useQ: [0, 1], useKV: [0, 1] };
    if (s <= r) return { j, branch: "lo", active: true, cells: [blk(r, j)], work: 4, useQ: [0, 1], useKV: [0, 1] };
    return { j, branch: "skip", active: false, cells: [], work: 0, useQ: [], useKV: [] };
  }
  const a = r, b = 2 * N - 1 - r, jb = 2 * N - 1 - j;
  const c = (q, k, tri = false) => ({ q0: q, qn: 1, k0: k, kn: 1, tri });
  if (s === 0) return { j, branch: "0", active: true, cells: [c(a, a, true), c(b, a), c(b, b, true)], work: 2, useQ: [0, 1], useKV: [0, 1] };
  if (s <= r) return { j, branch: "lo", active: true, cells: [c(a, j), c(b, j)], work: 2, useQ: [0, 1], useKV: [0] };
  return { j, branch: "hi", active: true, cells: [c(b, j), c(b, jb)], work: 2, useQ: [1], useKV: [0, 1] };
}

// Which KV block GPU r holds in `k, v` during this phase.
export function heldKV(N, r, phase) {
  const { type, s } = phase;
  if (type === "init") return r;
  if (type === "wait") return mod(r - s - 1, N);
  if (type === "final") return mod(r - (N - 1), N);
  return mod(r - s, N);
}

export const inFlight = (N, phase) => ["send", "compute", "merge"].includes(phase.type) && phase.s < N - 1;

// Steps whose blocks are finished / currently running at this phase.
export function progress(phase) {
  const { type, s } = phase;
  if (type === "init") return { doneBelow: 0, current: -1 };
  if (type === "send") return { doneBelow: s, current: -1 };
  if (type === "compute" || type === "merge") return { doneBelow: s, current: s };
  return { doneBelow: s + 1, current: -1 }; // wait, final
}

// KV blocks GPU r has merged into `out` by this phase.
export function covered(variant, N, r, phase) {
  const { doneBelow } = progress(phase);
  const upto = phase.type === "merge" ? phase.s + 1 : doneBelow;
  const js = [];
  for (let t = 0; t < upto; t++) {
    const d = deviceStep(variant, N, r, t);
    if (d.active) js.push(d.j);
  }
  return js;
}

export function stepTimes(variant, N) {
  return Array.from({ length: N }, (_, s) => Math.max(...Array.from({ length: N }, (_, r) => deviceStep(variant, N, r, s).work)));
}
