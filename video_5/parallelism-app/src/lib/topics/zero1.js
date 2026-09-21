// ZeRO-1 — deck pages 7-9, using the book's Win/Wout naming. Shard only
// the optimizer state: device 0 owns Win's (m,v), device 1 owns Wout's.
// Weights are still fully replicated, and gradients are still fully
// AllReduced (redundantly — each device ends up with the other weight's
// gradient too, even though it doesn't own that weight's optimizer
// state). Only the post-update weight sync changes: an AllGather instead
// of nothing, since each device only just updated its own shard.
import { weightsRow, gradsRow, optimRow, ALL } from "./helpers";

const D0_OWNS = { in: "solid", out: "ghost" };
const D1_OWNS = { in: "ghost", out: "solid" };

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
    title: "Setup — shard the optimizer state",
    notation: "device 0 owns $(m_{in},v_{in})$; device 1 owns $(m_{out},v_{out})$",
    body: "Weights $Win[D,F]$, $Wout[F,D]$ are still fully replicated on both devices, exactly like plain DP. The only change: each device now keeps optimizer state for only one weight — the greyed pair isn't stored here at all.",
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
    flops: "$2B_XDF + 2B_XFD = 4B_XDF$",
    body: "Unchanged from plain DP — sharding the optimizer state doesn't touch the forward pass at all, since every device still has every weight.",
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
    title: "Backward pass — still fully local, still redundant",
    notation: "$dWin^{(X)}[D,F]$ and $dWout^{(X)}[F,D]$ both computed on every $X$",
    flops: "4 matmuls ($dWout,dTmp,dWin,dIn$) $= 2\\!\\times\\!2B_XDF + 2\\!\\times\\!2B_XFD = 8B_XDF$",
    body: "Both devices still compute <b>both</b> weight gradients — even the one they don't own optimizer state for. That redundant compute (and the redundant AllReduce next) is exactly what ZeRO-2 removes.",
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
    title: "AllReduce — same full sync as plain DP",
    notation: "$dW = \\text{AllReduce}_X(dW^{(0)}, dW^{(1)})$ for both $Win$ and $Wout$, on every $X$",
    comms: "$4DF + 4FD = 8DF$ bytes — both gradients AllReduced in full, same total as plain DP",
    body: "Same full AllReduce as plain DP: every device ends up holding both synced gradients — but each will only use the one that matches the optimizer state it actually owns.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, W, ALL, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, W, ALL, GSYNC, D1_OWNS),
      ],
      comm: {
        type: "allreduce",
        label: "$dW = \\text{AllReduce}_X(dW^{(0)}, dW^{(1)}) = \\dfrac{dW^{(0)}+dW^{(1)}}{X}$",
      },
    },
  },
  {
    id: 5,
    title: "Local optimizer update — owned weight only",
    notation: "$Win' = \\text{update}(Win, dWin; m_{in},v_{in})$  only on device 0;  $Wout'$ only on device 1",
    body: "Device 0 updates $Win$ (it holds $Win$'s optimizer state); device 1 updates $Wout$. The other weight on each device stays untouched this step — that device has no $(m,v)$ for it.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active", out: "solid" }, (k) => (k === "in" ? WPRIME(k) : W(k)), ALL, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", { in: "solid", out: "active" }, (k) => (k === "out" ? WPRIME(k) : W(k)), ALL, GSYNC, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 6,
    title: "AllGather — re-sync the weights",
    notation: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$",
    comms: "$2DF + 2FD = 4DF$ bytes — an AllGather costs $\\approx 1\\times$ each array's bytes",
    body: "Device 0 sends its freshly-updated $Win$ to device 1; device 1 sends $Wout$ back. Both devices end up with the full, updated pair again — replicated, just like DP, but only after paying for an AllGather that plain DP never needed.",
    note: "The redundancy ZeRO-2 removes: the earlier full AllReduce carried $Wout$'s gradient all the way to device 0, which never used it.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, WPRIME, ALL, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, WPRIME, ALL, GSYNC, D1_OWNS),
      ],
      comm: { type: "allgather", label: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$" },
    },
  },
  {
    id: 7,
    title: "Compute vs. communication",
    notation: "$T_{\\text{math}} = \\dfrac{8BDF}{XC} \\qquad T_{\\text{comms}} = \\dfrac{8DF + 4DF}{W_{ici}} = \\dfrac{12DF}{W_{ici}}$",
    flops: "backward pass only, same as plain DP: $8B_XDF$ (4 matmuls) — sharding the optimizer state doesn't touch a single matmul",
    comms: "the gradient AllReduce ($8DF$) $+$ the post-update weight AllGather ($4DF$) $= 12DF$ bytes — <i>more</i> than DP's $8DF$",
    body:
      "ZeRO-1 does the exact same compute as plain DP. What it costs is <i>extra</i> communication beyond DP's own AllReduce: a whole new weight AllGather DP never needed, just to get the two devices back in sync after each only updated its own optimizer-state shard.",
    note:
      "Compute-bound ($T_{\\text{math}} > T_{\\text{comms}}$) when $\\dfrac{B}{X} > \\dfrac{3}{2}\\cdot\\dfrac{C}{W_{ici}}$ — a harder bar than plain DP's $\\dfrac{C}{W_{ici}}$ (ratio 1), since ZeRO-1 pays $1.5\\times$ DP's bytes for the same compute. The book's own text glosses ZeRO-{1,2,3} as “the same communication cost” when making the AllReduce=AllGather+ReduceScatter argument for ZeRO-2/3 — that argument doesn't cover ZeRO-1's extra post-update weight AllGather, which is genuinely additional.",
    formula: "$\\text{compute-bound} \\iff \\dfrac{B}{X} > \\dfrac{3}{2}\\cdot\\dfrac{C}{W_{ici}}$",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", ALL, WPRIME, ALL, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", ALL, WPRIME, ALL, GSYNC, D1_OWNS),
      ],
      comm: null,
    },
  },
];
