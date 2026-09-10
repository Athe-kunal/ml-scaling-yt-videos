// Plain data parallel — deck pages 5-6, using the book's own 2-matmul MLP
// weights (Win[D,F], Wout[F,D]) instead of a toy numbered chain, matching
// fsdp.js/tp.js's naming. Weights and optimizer state are fully replicated
// on both devices; only the gradients need to be synchronized (AllReduce)
// once per step, then each device applies the same synced gradient to its
// own local copy of the weights.
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
const G0 = (k) => (k === "in" ? "dWin[D,F]^{(0)}" : "dWout[F,D]^{(0)}");
const G1 = (k) => (k === "in" ? "dWin[D,F]^{(1)}" : "dWout[F,D]^{(1)}");
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
        device("mb1", "In[B₀,D]", ALL, W, {}, G0),
        device("mb2", "In[B₁,D]", ALL, W, {}, G1),
      ],
      comm: null,
    },
  },
  {
    id: 2,
    title: "Forward pass — no communication",
    notation: "$In[B_X,D] \\cdot_D Win[D,F] \\to Tmp[B_X,F] \\cdot_F Wout[F,D] \\to Out[B_X,D]$",
    body: "Each device runs the full forward chain independently over its own batch slice. <i>“The forward pass involves no communication”</i> — every device is on its own until the backward pass.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active", out: "active" }, W, {}, G0),
        device("mb2", "In[B₁,D]", { in: "active", out: "active" }, W, {}, G1),
      ],
      comm: null,
    },
  },
  {
    id: 3,
    title: "Backward pass — local gradients",
    notation: "$dWin^{(X)}[D,F] = \\dfrac{\\partial L}{\\partial Win}$,  $dWout^{(X)}[F,D] = \\dfrac{\\partial L}{\\partial Wout}$,  computed locally on device $X$",
    body: "Backprop produces both weight gradients on every device, but the two devices' gradients differ — they were computed from different batch slices and haven't been synced yet.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, ALL, G0),
        device("mb2", "In[B₁,D]", ALL, W, ALL, G1),
      ],
      comm: null,
    },
  },
  {
    id: 4,
    title: "AllReduce — sync gradients",
    notation: "$dW = \\text{AllReduce}_X\\!\\left(dW^{(0)}, dW^{(1)}\\right) = \\dfrac{dW^{(0)} + dW^{(1)}}{X}$, for $W \\in \\{Win, Wout\\}$",
    body: "Every device All-Reduces its local gradients with the other's — averaging over the world size — so both devices end this step holding the <i>same</i> synced gradient for $Win$ and for $Wout$.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, ALL, GSYNC),
        device("mb2", "In[B₁,D]", ALL, W, ALL, GSYNC),
      ],
      comm: {
        type: "allreduce",
        label: "$dW = \\text{AllReduce}_X(dW^{(0)}, dW^{(1)}) = \\dfrac{dW^{(0)}+dW^{(1)}}{X}$, for $W \\in \\{Win, Wout\\}$",
      },
    },
  },
  {
    id: 5,
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
    id: 6,
    title: "Compute vs. communication",
    notation: "$T_{\\text{comms}} = \\dfrac{8DF}{W_{ici}} \\qquad T_{\\text{math}} = \\dfrac{8BDF}{XC}$",
    body: "Both devices now hold identical, updated weights $Win'$, $Wout'$ — replicated data parallelism has come full circle. The AllReduce in step 4 is the whole communication cost per step; whether that's hidden behind compute depends on batch size.",
    note:
      "Compute-bound when $\\dfrac{B}{X} > \\dfrac{C}{W_{ici}}$ — i.e. the per-device batch has to be big enough to keep the AllReduce off the critical path.",
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
