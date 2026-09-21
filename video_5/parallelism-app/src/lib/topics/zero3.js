// ZeRO-3 — deck pages 13-16, using the book's Win/Wout naming. Weights
// are sharded too, not just optimizer state and gradients: device 0 only
// ever holds Win at rest, device 1 only ever holds Wout. Before compute,
// the missing weight is temporarily AllGathered (a new comm cost unique
// to this stage); after the backward pass, gradients are ReduceScattered
// like ZeRO-2, and the borrowed weight is dropped again rather than
// kept — so there's no final AllGather at all.
import { weightsRow, gradsRow, optimRow } from "./helpers";

const D0_OWNS = { in: "solid" };
const D1_OWNS = { out: "solid" };
const D0_BORROWED = { in: "solid", out: "ghost" };
const D1_BORROWED = { in: "ghost", out: "solid" };
const BOTH_ACTIVE = { in: "active", out: "active" };
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
    title: "Setup — shard everything, including the weights",
    notation: "$Win, (m_{in},v_{in})$ live only on device 0;  $Wout, (m_{out},v_{out})$ live only on device 1",
    body: "For the first time, weights aren't replicated at rest either — device 0 only physically holds $Win[D,F]$; device 1 only holds $Wout[F,D]$. There is no full copy of the model on any single device.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", D0_OWNS, W, {}, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", D1_OWNS, W, {}, G1, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 2,
    title: "AllGather — borrow the missing weight",
    notation: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$   (temporary, before compute)",
    comms: "$2DF + 2FD = 4DF$ bytes",
    body: "Before it can run the forward pass, each device has to temporarily materialize the weight it doesn't own — the dashed box is borrowed, not permanent. This is a comm cost unique to ZeRO-3: weights, not just gradients, now have to move.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", D0_BORROWED, W, {}, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", D1_BORROWED, W, {}, G1, D1_OWNS),
      ],
      comm: { type: "allgather", label: "$Win, Wout = \\text{AllGather}_X\\!\\left(W^{\\text{owner}(W)}\\right)$" },
    },
  },
  {
    id: 3,
    title: "Forward + backward — using the borrowed pair",
    notation: "$In \\cdot_D Win \\to Tmp \\cdot_F Wout \\to Out \\ \\Rightarrow\\ dWin^{(X)}, dWout^{(X)}$  computed on every $X$",
    flops: "forward $4B_XDF$ + 4 backward matmuls $8B_XDF$ $= 12B_XDF$ — same total as DP, just using the borrowed weights",
    body: "With both weights temporarily in hand, both devices run forward and backward exactly like plain DP — computing both gradients, owned or not.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", BOTH_ACTIVE, W, { in: "solid", out: "solid" }, G0, D0_OWNS),
        device("mb2", "In[B₁,D]", BOTH_ACTIVE, W, { in: "solid", out: "solid" }, G1, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 4,
    title: "ReduceScatter — sync gradients, keep only your shard",
    notation: "$dW = \\text{ReduceScatter}_X(dW^{(0)}, dW^{(1)})$,  materializes only on $\\text{owner}(W)$",
    comms: "$2DF + 2FD = 4DF$ bytes",
    body: "Same as ZeRO-2: the gradient sync scatters the result so each device only ends up holding the synced gradient for the weight it actually owns.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", D0_BORROWED, W, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", D1_BORROWED, W, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: {
        type: "reducescatter",
        label: "$dW = \\text{ReduceScatter}_X(dW^{(0)}, dW^{(1)}),\\ \\text{owner}(W)$ only",
      },
    },
  },
  {
    id: 5,
    title: "Update, then discard the borrowed shard",
    notation: "$W' = \\text{update}(W, dW; m,v)$  where $\\text{owner}(W)=X$,  then drop the rest",
    body: "Each device updates only the weight it owns, then <b>frees</b> the temporarily-borrowed weight it gathered via the earlier AllGather — back down to holding only its own shard.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "active" }, WPRIME, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", { out: "active" }, WPRIME, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 6,
    title: "No final AllGather",
    notation: "state at rest:  $W$ on $\\text{owner}(W)$ only — nothing re-replicated",
    body: "Unlike ZeRO-1/2, there's no step here to re-sync a full replicated copy — each device simply stays at its own sharded resting state. The next forward pass will re-gather whatever it needs, on demand, via the same borrow-weight AllGather used earlier.",
    note: "Trade-off: ZeRO-3 communicates more often (every weight, every step) than ZeRO-1/2, in exchange for the smallest possible peak memory footprint — no device ever holds more than its own shard plus whatever it's currently borrowing.",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "solid" }, WPRIME, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", { out: "solid" }, WPRIME, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: null,
    },
  },
  {
    id: 7,
    title: "Compute vs. communication",
    notation: "$T_{\\text{math}} = \\dfrac{12B_XDF}{C} \\qquad T_{\\text{comms}} = \\dfrac{4DF + 4DF}{W_{ici}} = \\dfrac{8DF}{W_{ici}}$",
    flops: "$12B_XDF$ total per device (this walkthrough's forward+backward combined into one step)",
    comms: "the borrow-weight AllGather ($4DF$) $+$ the gradient ReduceScatter ($4DF$) $= 8DF$ bytes",
    body:
      "This walkthrough's simplification — gather <i>both</i> weights once, hold them through forward <i>and</i> backward, then discard — moves the same $8DF$ bytes as plain DP's two AllReduces, giving it an easier compute-bound bar than the book's actual FSDP below.",
    note:
      "Compute-bound here when $\\dfrac{B}{X} > \\dfrac{2}{3}\\cdot\\dfrac{C}{W_{ici}}$ — but that comm saving is a mirage: holding <i>both</i> full weights through backward needs the same peak memory as no sharding at all, giving up ZeRO-3's actual point. The book's real FSDP (see the <b>FSDP</b> tab) discards each weight right after its forward use and re-gathers it for backward — <i>more</i> communication ($12DF$, not $8DF$) in exchange for genuinely minimal peak memory, landing on the harder, book-verified $\\dfrac{B}{X} > \\dfrac{C}{W_{ici}}$ bar instead.",
    formula: "$\\text{compute-bound (this simplification)} \\iff \\dfrac{B}{X} > \\dfrac{2}{3}\\cdot\\dfrac{C}{W_{ici}}$",
    diagram: {
      devices: [
        device("mb1", "In[B₀,D]", { in: "solid" }, WPRIME, D0_GRAD_OWNED, GSYNC, D0_OWNS),
        device("mb2", "In[B₁,D]", { out: "solid" }, WPRIME, D1_GRAD_OWNED, GSYNC, D1_OWNS),
      ],
      comm: null,
    },
  },
];
