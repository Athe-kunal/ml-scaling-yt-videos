// ZeRO-2 — deck pages 10-12, using the book's Win/Wout naming. Same
// optimizer-state sharding as ZeRO-1 (device 0 owns Win, device 1 owns
// Wout), plus gradients are now sharded too: a ReduceScatter (not an
// AllReduce) leaves each device holding the synced gradient only for the
// weight it owns. No more redundant full-gradient copies; the final
// AllGather of weights is unchanged from ZeRO-1.
import { weightsRow, gradsRow, optimRow, ALL } from "./helpers";

const D0_OWNS = { in: "solid", out: "ghost" };
const D1_OWNS = { in: "ghost", out: "solid" };
const D0_GRAD_OWNED = { in: "solid" };
const D1_GRAD_OWNED = { out: "solid" };

function device(mb, input, weightStates, weightLabelFn, gradStates, gradLabelFn, optimStates) {
  return {
    mb,
    input,
    layers: weightsRow(weightStates, weightLabelFn),
    grads: gradsRow(gradStates, gradLabelFn),
    optim: optimRow(optimStates),
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
    title: "Setup — shard optimizer state and gradients",
    notation: "device 0 owns $Win$'s $(m,v)$ and gradient; device 1 owns $Wout$'s",
    body: "Weights are still fully replicated. Optimizer state is sharded exactly like ZeRO-1 — and this time gradient <i>ownership</i> is sharded the same way, which changes how the backward-pass sync works.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, {}, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, W, {}, G1, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 2,
    title: "Forward pass — no communication",
    notation: "$In[B_X,D] \\cdot_D Win[D,F] \\to Tmp[B_X,F] \\cdot_F Wout[F,D] \\to Out[B_X,D]$",
    body: "Unchanged — every device still has every weight, so the forward pass is identical to plain DP and ZeRO-1.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active", out: "active" }, W, {}, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", { in: "active", out: "active" }, W, {}, G1, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 3,
    title: "Backward pass — still fully local",
    notation: "$dWin^{(X)}[D,F]$ and $dWout^{(X)}[F,D]$ both computed on every $X$",
    body: "Backprop still has to touch both weights locally — they're replicated, so there's no way around computing a local gradient for each one. The saving happens in the next step, not this one.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, ALL, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, W, ALL, G1, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 4,
    title: "ReduceScatter — sync gradients, keep only your shard",
    notation: "$dW = \\text{ReduceScatter}_X(dW^{(0)}, dW^{(1)})$,  materializes only on $\\text{owner}(W)$",
    body: "Instead of an AllReduce, a ReduceScatter combines the gradients but only <i>materializes</i> the result on the device that owns that weight's optimizer state. Device 0 ends up with synced $dWin$ only; device 1 with $dWout$ only.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, W, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: {
        type: "reducescatter",
        label: "$dW = \\text{ReduceScatter}_X(dW^{(0)}, dW^{(1)}),\\ \\text{owner}(W)$ only",
      },
    },
  },
  {
    id: 5,
    title: "Local optimizer update — owned weight only",
    notation: "$Win' = \\text{update}(Win, dWin; m_{in},v_{in})$  only on device 0;  $Wout'$ only on device 1",
    body: "Same update pattern as ZeRO-1 — except this time neither device ever held a synced gradient for the weight it doesn't own, so there was nothing wasted.",
    diagram: {
      devices: [
        device(
          "mb1",
          "In[B₀,D]",
          { in: "active", out: "solid" },
          (k) => (k === "in" ? WPRIME(k) : W(k)),
          D0_GRAD_OWNED,
          GSYNC,
          D0_OWNS
        ),
        device(
          "mb2",
          "In[B₁,D]",
          { in: "solid", out: "active" },
          (k) => (k === "out" ? WPRIME(k) : W(k)),
          D1_GRAD_OWNED,
          GSYNC,
          D1_OWNS
        ),
      ],
      comm: null,
    },
  },
  {
    id: 6,
    title: "AllGather — re-sync the weights",
    notation: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$",
    body: "Same weight re-sync as ZeRO-1: each device broadcasts the weight it just updated. Both end up with the full updated pair again.",
    note:
      "No more redundant gradients, and ReduceScatter has less communication overhead than AllReduce — the FSDP framing: each backward-pass AllReduce becomes an AllGather + a ReduceScatter, at equivalent total bytes moved but without the wasted local copies.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, WPRIME, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, WPRIME, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: { type: "allgather", label: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$" },
    },
  },
];
