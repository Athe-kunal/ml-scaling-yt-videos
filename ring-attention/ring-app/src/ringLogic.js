import { T } from "./theme";
import { PHASE, STEP_PHASES, shardLabel, STRIPE_L, deviceStep, heldKV, inFlight, mod, covered } from "./ringModel";
import { RING_SRC, ZIGZAG_SRC, STRIPE_SRC, UTILS_SRC, ZIGZAG_UTILS_SRC, STRIPE_UTILS_SRC } from "./codeSources";

// Per-variant copy, code highlighting and live variables for the ring family
// (ring / zigzag / stripe). Shared by RingView and CompareView.

export const COPY = {
  ring: {
    eyebrow: "SEQUENCE PARALLELISM · RING ATTENTION",
    title: "Ring attention: Q stays, KV goes around",
    intro: "Each GPU keeps its query block and passes its key/value block to the next GPU, $N$ times. Every hop is posted <i>before</i> the attention kernel, so the transfer hides behind compute. With a causal mask, half the blocks it receives are all future tokens, and that is where the imbalance comes from.",
    src: ["ring_flash_attn.py", RING_SRC, 430],
    utils: ["utils.py", UTILS_SRC],
    matrixNote: () => "GPU $r$ owns two neighbouring chunk rows ($c_{2r}, c_{2r+1}$). The top pair has 3 visible cells (2 chunk²), the bottom pair $4N-1$ cells ($4N-2$ chunk²). White outline = the GPU you follow.",
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$). A step lasts as long as its slowest GPU. Step 0 costs 2, every later step 4 (a full block on GPU ${N - 1}), so wall time is $4N-2 = ${4 * N - 2}$ against an ideal $2N = ${2 * N}$.`,
  },
  zigzag: {
    eyebrow: "SEQUENCE PARALLELISM · ZIG-ZAG RING ATTENTION",
    title: "Zig-zag ring attention: balance the causal triangle",
    intro: "Same ring and same hops, but the sequence is cut into $2N$ chunks and GPU $r$ takes chunks $r$ and $2N-1-r$: one early, one late. Every off-diagonal step now costs exactly half a block on every GPU, picked by one of two slices: <code>k[:, :L/2]</code> or <code>q[:, L/2:]</code>.",
    src: ["zigzag_ring_flash_attn.py", ZIGZAG_SRC, 470],
    utils: ["extract_local + utils.py", ZIGZAG_UTILS_SRC],
    matrixNote: () => "Each GPU owns two rows of chunks, one near the top (few keys) and one near the bottom (many keys), so every GPU's total share of the triangle is the same.",
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$). Every step costs 2 on every GPU: wall time $2N = ${2 * N}$, no idle time. Same bar scale as the ring view.`,
  },
  stripe: {
    eyebrow: "SEQUENCE PARALLELISM · STRIPE ATTENTION",
    title: "Stripe attention: deal tokens round-robin",
    intro: "Same ring and same hops, but tokens are dealt like cards: token $t$ goes to GPU $t \\bmod N$. Every GPU then holds a thin stripe spread over the whole sequence, so each KV block it receives is about half visible, at every step. If the block came from an earlier rank the plain causal mask is right; from a later rank the mask shifts by one token: <code>q[:, 1:]</code> against <code>k[:, :-1]</code>.",
    src: ["stripe_flash_attn.py", STRIPE_SRC, 470],
    utils: ["extract_local + utils.py", STRIPE_UTILS_SRC],
    matrixNote: (N) => `Token resolution: $S = ${STRIPE_L}N = ${STRIPE_L * N}$ tokens, row $t$ belongs to GPU $t \\bmod ${N}$. Every GPU's rows are spread evenly down the triangle. A step fills $\\tfrac{L(L+1)}{2}$ cells (from an earlier rank) or $\\tfrac{L(L-1)}{2}$ (later rank): with $L=${STRIPE_L}$ that is ${(STRIPE_L * (STRIPE_L + 1)) / 2}$ vs ${(STRIPE_L * (STRIPE_L - 1)) / 2}$, but both tend to $L^2/2$ as $L$ grows.`,
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$), large-$L$ limit: every step costs $L^2/2 = 2$ on every GPU, wall time $2N = ${2 * N}$. The leftover imbalance is one token's diagonal, $L$ cells per step, negligible for long sequences.`,
  },
};

// tag -> color for the main function and utils listings at this phase.
export function codeFocus(variant, N, phase, r) {
  const { type, s } = phase;
  const C = (t) => PHASE[t].color;
  const main = {}, utils = {};
  if (type === "init") {
    main.init = C("init");
    utils.init = C("init");
    utils.layout = C("init");
    return { main, utils };
  }
  if (type === "final") {
    main.final = C("final");
    return { main, utils };
  }
  main.loop = `dim:${T.soft}`;
  const d = deviceStep(variant, N, r, s);
  if (type === "send") {
    main.send = C("send");
    utils.send = C("send");
  } else if (type === "wait") {
    main.wait = C("wait");
    utils.wait = C("wait");
  } else if (type === "compute") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (variant === "ring") {
      main.cond = d.active ? C("compute") : T.bad;
      if (d.active) main.comp = C("compute");
    } else {
      main[`cond:${d.branch}`] = C("compute");
      main[`comp:${d.branch}`] = C("compute");
      main.fwd = `dim:${C("compute")}`;
    }
  } else if (type === "merge") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (d.active) {
      if (variant === "ring") main.merge = C("merge");
      else main[`merge:${d.branch}`] = C("merge");
      utils.upd = C("merge");
      if (s === 0) utils.first = C("merge");
      else {
        utils[d.branch === "hi" ? "slice" : "else"] = C("merge");
        utils.core = C("merge");
      }
    } else if (variant === "ring") main.cond = T.bad;
  }
  return { main, utils };
}

export function variables(variant, N, phase, r) {
  const { type, s } = phase;
  const kvLbl = (j) => shardLabel(variant, N, j, "KV");
  const rows = [
    ["comm.rank", r],
    ["comm.send_rank", `${mod(r + 1, N)}  (next)`],
    ["comm.recv_rank", `${mod(r - 1, N)}  (prev)`],
    ["step", type === "init" ? "—" : type === "final" ? `${N - 1} (done)` : s],
    ["k, v", kvLbl(heldKV(N, r, phase))],
  ];
  if (inFlight(N, phase)) rows.push(["next_k, next_v", `← ${kvLbl(mod(r - s - 1, N))} (in flight)`]);
  else if (type === "wait") rows.push(["next_k, next_v", `${kvLbl(mod(r - s - 1, N))} (landed)`]);
  else rows.push(["next_k, next_v", type === "init" ? "None" : "—"]);
  if (STEP_PHASES.includes(type)) {
    const d = deviceStep(variant, N, r, s);
    if (variant === "ring") {
      rows.push(["step <= comm.rank", `${s} <= ${r} → ${s <= r}`]);
      rows.push(["causal and step == 0", `${s === 0}`]);
    } else if (variant === "stripe") {
      rows.push(["step <= comm.rank", `${s} <= ${r} → ${s <= r}`]);
      rows.push(["kernel inputs", d.branch === "lo" ? "q, k, v  causal" : "q[:,1:], k[:,:-1], v[:,:-1]  causal"]);
      rows.push(["slice_", d.branch === "hi" ? "(:, 1:)" : "None"]);
    } else {
      rows.push(["branch", d.branch === "0" ? "step == 0" : d.branch === "lo" ? `step <= rank (${s} <= ${r})` : `else (${s} > ${r})`]);
      rows.push(["kernel inputs", d.branch === "0" ? "q, k, v  causal" : d.branch === "lo" ? "q, k0, v0" : "q1, k, v"]);
      rows.push(["slice_", d.branch === "hi" ? "(:, L/2:)" : "None"]);
    }
  }
  const js = covered(variant, N, r, phase);
  rows.push(["out covers", js.length ? js.map(kvLbl).join("  ") : "None"]);
  return rows;
}
