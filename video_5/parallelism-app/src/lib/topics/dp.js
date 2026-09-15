// Plain data parallel — deck pages 5-6, using the book's own 2-matmul MLP
// weights (Win[D,F], Wout[F,D]) instead of a toy numbered chain, matching
// fsdp.js/tp.js's naming. Weights and optimizer state are fully replicated
// on both devices; only the gradients need to be synchronized.
//
// Step granularity follows the book's exact per-line pseudocode order
// rather than collapsing both weights' backward pass into one step:
// dWout is computed locally (unreduced, tagged {U_X}) and AllReduced
// FIRST, entirely independently of dWin — dWin's own local computation
// needs an intervening dTmp = dOut *_D Wout step (using the WEIGHT, not
// its gradient, so it never has to wait on dWout's AllReduce), and only
// then is dWin computed locally and AllReduced in turn. Both AllReduces
// are off the critical path/async per the book, which is exactly why
// dTmp and dIn can run without waiting on them.
//
// notation/formula/note/body strings use $...$ to mark math spans — see
// lib/latex.js's withInlineMath, which renders those spans with real
// KaTeX (fractions, subscripts, \cdot, etc.) and leaves everything else
// (including plain English and any <b>/<i> tags) as normal text.
import { weightsRow, gradsRow, optimRow, ALL } from "./helpers";

function device(mb, input, weightStates, weightLabelFn, gradStates, gradLabelFn) {
  return {
    mb,
    input,
    layers: weightsRow(weightStates, weightLabelFn),
    grads: gradsRow(gradStates, gradLabelFn),
    optim: optimRow(ALL),
  };
}

const W = (k) => (k === "in" ? "Win[D,F]" : "Wout[F,D]");
const WPRIME = (k) => (k === "in" ? "Win'[D,F]" : "Wout'[F,D]");
const GUNREDUCED = (k) => (k === "in" ? "dWin[D,F]^{\\{U_X\\}}" : "dWout[F,D]^{\\{U_X\\}}");
const GSYNC = (k) => (k === "in" ? "dWin[D,F]" : "dWout[F,D]");

export const STEPS = [
  {
    id: 1,
    title: "Setup — replicated model, sharded batch",
    notation: "$In[B_X,D]$, $X \\in \\{0,1\\}$ — weights $Win[D,F]$, $Wout[F,D]$ and optimizer state $(m,v)$ replicated on every $X$",
    body:
      "Both devices hold a full copy of the model — weights $Win[D,F]$, $Wout[F,D]$ and optimizer state $(m_{in},v_{in}),(m_{out},v_{out})$ — and get their own slice of the global batch: $In[B_X,D]$, device 0 from mb1, device 1 from mb2. " +
      "$(m,v)$ is Adam's per-parameter optimizer state for a weight: $m$ is the running mean of the gradient (first moment), $v$ is the running mean of the squared gradient (second moment) — both have to be stored somewhere for every parameter, which is exactly what ZeRO-1 shards.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, {}, GUNREDUCED),
        device("mb2", "In[B₁,D]", ALL, W, {}, GUNREDUCED),
      ],
      comm: null,
    },
  },
  {
    id: 2,
    title: "Forward pass — no communication",
    notation: "$Tmp[B_X,F] = In[B_X,D] \\cdot_D Win[D,F] \\to Out[B_X,D] = Tmp[B_X,F] \\cdot_F Wout[F,D] \\to Loss[B_X]$",
    body: "Each device runs the full forward chain independently over its own batch slice — no partial sums, no collectives; every device is on its own until the backward pass. $Tmp$ is kept around afterward, since $dWout$'s backward computation needs it.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active", out: "active" }, W, {}, GUNREDUCED),
        device("mb2", "In[B₁,D]", { in: "active", out: "active" }, W, {}, GUNREDUCED),
      ],
      comm: null,
    },
  },
  {
    id: 3,
    title: "dOut, then local dWout — unreduced",
    notation: "$dOut[B_X,D] = \\ldots \\quad\\Rightarrow\\quad dWout[F,D]^{\\{U_X\\}} = Tmp[B_X,F] \\cdot_B dOut[B_X,D]$",
    body:
      "Backward starts from the loss gradient $dOut[B_X,D]$. Contracting it against $Tmp$ (kept from the forward pass) over the batch $B$ gives each device its own local, not-yet-reduced contribution toward $Wout$'s gradient — tagged $\\{U_X\\}$ for “unreduced along $X$.” $Win$'s gradient isn't computed yet: it needs $dTmp$ first, which comes after this.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, { out: "partial" }, GUNREDUCED),
        device("mb2", "In[B₁,D]", ALL, W, { out: "partial" }, GUNREDUCED),
      ],
      comm: null,
    },
  },
  {
    id: 4,
    title: "AllReduce — sync dWout (async)",
    notation: "$dWout[F,D] = \\text{AllReduce}_X\\!\\left(dWout[F,D]^{\\{U_X\\}}\\right)$",
    body: "Not on the critical path — this AllReduce can run in the background while the next step (computing $dTmp$) proceeds, since $dTmp$ only needs the <i>weight</i> $Wout$, not its gradient.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, { out: "solid" }, GSYNC),
        device("mb2", "In[B₁,D]", ALL, W, { out: "solid" }, GSYNC),
      ],
      comm: {
        type: "allreduce",
        label: "$dWout[F,D] = \\text{AllReduce}_X(dWout[F,D]^{\\{U_X\\}})$ — async, not on the critical path",
      },
    },
  },
  {
    id: 5,
    title: "dTmp — local, doesn't wait on the AllReduce above",
    notation: "$dTmp[B_X,F] = dOut[B_X,D] \\cdot_D Wout[F,D]$",
    body: "A pure local matmul using the replicated <i>weight</i> $Wout$ — it doesn't need $Wout$'s gradient to have finished reducing, so it can run immediately, overlapped with step 4's async AllReduce.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, { out: "solid" }, GSYNC),
        device("mb2", "In[B₁,D]", ALL, W, { out: "solid" }, GSYNC),
      ],
      comm: null,
    },
  },
  {
    id: 6,
    title: "Local dWin — unreduced",
    notation: "$dWin[D,F]^{\\{U_X\\}} = In[B_X,D] \\cdot_B dTmp[B_X,F]$",
    body: "Mirrors step 3, for the other weight: contract $In$ against $dTmp$ over the batch $B$ to get each device's own local, not-yet-reduced contribution toward $Win$'s gradient.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, { in: "partial", out: "solid" }, GUNREDUCED),
        device("mb2", "In[B₁,D]", ALL, W, { in: "partial", out: "solid" }, GUNREDUCED),
      ],
      comm: null,
    },
  },
  {
    id: 7,
    title: "dIn — needed for previous layers",
    notation: "$dIn[B_X,D] = dTmp[B_X,F] \\cdot_F Win[D,F]$",
    body: "Also a pure local matmul on the replicated weight $Win$ — independent of $Win$'s own gradient, so it can run any time after $dTmp$, in parallel with $Win$'s AllReduce below. This is exactly what the <i>previous</i> layer's backward pass needs next.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, { in: "partial", out: "solid" }, GUNREDUCED),
        device("mb2", "In[B₁,D]", ALL, W, { in: "partial", out: "solid" }, GUNREDUCED),
      ],
      comm: null,
    },
  },
  {
    id: 8,
    title: "AllReduce — sync dWin (async)",
    notation: "$dWin[D,F] = \\text{AllReduce}_X\\!\\left(dWin[D,F]^{\\{U_X\\}}\\right)$",
    body: "Same as step 4, now for the other weight — not on the critical path, async. Both devices end this step holding the <i>same</i> synced gradient for $Win$ and for $Wout$.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, ALL, GSYNC),
        device("mb2", "In[B₁,D]", ALL, W, ALL, GSYNC),
      ],
      comm: {
        type: "allreduce",
        label: "$dWin[D,F] = \\text{AllReduce}_X(dWin[D,F]^{\\{U_X\\}})$ — async, not on the critical path",
      },
    },
  },
  {
    id: 9,
    title: "Optimizer step — update weights",
    notation: "$Win' = \\text{update}(Win, dWin; m_{in},v_{in})$,  $Wout' = \\text{update}(Wout, dWout; m_{out},v_{out})$",
    body: "Each device applies the update rule locally, using the now-identical synced gradients against its own (also identical) optimizer state. No communication needed here — same inputs in, same weights out.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active", out: "active" }, WPRIME, ALL, GSYNC),
        device("mb2", "In[B₁,D]", { in: "active", out: "active" }, WPRIME, ALL, GSYNC),
      ],
      comm: null,
    },
  },
  {
    id: 10,
    title: "Compute vs. communication",
    notation: "$T_{\\text{comms}} = \\dfrac{8DF}{W_{ici}} \\qquad T_{\\text{math}} = \\dfrac{8BDF}{XC}$",
    body: "Both devices now hold identical, updated weights $Win'$, $Wout'$ — replicated data parallelism has come full circle. The two AllReduces (steps 4 and 8) are the whole communication cost per step; whether that's hidden behind compute depends on batch size.",
    note:
      "Compute-bound when $\\dfrac{B}{X} > \\dfrac{C}{W_{ici}}$ — i.e. the per-device batch has to be big enough to keep both AllReduces off the critical path.",
    formula: "$\\text{compute-bound} \\iff \\dfrac{B}{X} > \\dfrac{C}{W_{ici}}$",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, WPRIME, ALL, GSYNC),
        device("mb2", "In[B₁,D]", ALL, WPRIME, ALL, GSYNC),
      ],
      comm: null,
    },
  },
];
