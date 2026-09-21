// Tensor Parallelism — one step per line of the exact pseudocode from
// https://jax-ml.github.io/scaling-book/training/, "Tensor Parallelism".
// Same 2-matmul MLP block as fsdp.js (In -> Win -> Tmp -> Wout -> Out ->
// Loss, then dOut -> dWout -> dTmp -> dWin -> dIn), but the sharding is
// inverted: here the WEIGHTS never move — Win[D,F_Y] and Wout[F_Y,D] sit
// fixed on their device the whole block — and it's the ACTIVATION that
// cycles between a sharded resting state (In[B,D_Y]) and a temporarily
// gathered, replicated form (In[B,D]), via an AllGather at each block's
// entry and a ReduceScatter at its exit. Both collectives sit on the
// critical path (unlike FSDP's prefetchable weight AllGathers), since
// compute can't start until the activation they produce is in hand.
//
// Diagram nodes are pure LaTeX (rendered via MathLabel, no $ needed).
// notation/body/note/comm.label are prose+math strings using $...$ spans
// (see lib/latex.js). Reuses FSDPDiagram (kind: "tp") since both topics
// trace the identical named-tensor set through a forward and backward row.

function node(label, state) {
  return { label, state };
}

const HIDDEN = () => node("", "hidden");

function snapshot(overrides) {
  const base = {
    In: HIDDEN(),
    Win: HIDDEN(),
    Tmp: HIDDEN(),
    Wout: HIDDEN(),
    Out: HIDDEN(),
    Loss: HIDDEN(),
    dOut: HIDDEN(),
    dWout: HIDDEN(),
    dTmp: HIDDEN(),
    dWin: HIDDEN(),
    dIn: HIDDEN(),
  };
  return { ...base, ...overrides };
}

function diagramFrom(snap, comm) {
  const forward = ["In", "Win", "Tmp", "Wout", "Out", "Loss"].map((id) => ({ id, ...snap[id] }));
  const backward = ["dOut", "dWout", "dTmp", "dWin", "dIn"].map((id) => ({ id, ...snap[id] }));
  return { kind: "tp", forward, backward, comm };
}

// Per-line "shapes & sharding" row (rendered by MatrixShapes, a separate
// panel from the tensor-flow diagram above): the 1-3 concrete matrices
// this pseudocode line actually operates on, each annotated with which
// axis (if any) is sharded. mat() mirrors node()'s tensor ids/labels but
// carries shape + sharding instead of a flow state; op() is the connecting
// "×" / "=" / "→" between two matrices, optionally tinted to a comm type.
function mat(id, rows, cols, opts = {}) {
  return { id, rows, cols, shardAxis: null, state: "solid", tone: "act", ...opts };
}
function op(symbol, opts = {}) {
  return { op: symbol, ...opts };
}

// Once the forward pass has finished (the dOut step onward), none of Win/Tmp/
// Wout/Out/Loss ever change state again — unlike FSDP, TP's weights never
// move, so nothing needs to drop back to "ghost". Folded into every
// backward-step snapshot below so stepping through the backward pass
// doesn't make the completed forward blocks vanish.
const FWD_DONE = {
  In: node("In[B,D]", "solid"),
  Win: node("Win[D,F_Y]", "solid"),
  Tmp: node("Tmp[B,F_Y]", "solid"),
  Wout: node("Wout[F_Y,D]", "solid"),
  Out: node("Out[B,D_Y]", "solid"),
  Loss: node("Loss[B]", "solid"),
};

export const STEPS = [
  {
    id: 1,
    title: "Setup — weights fixed, activations sharded at rest",
    notation: "$In[B,D_Y]$ (sharded along $D$)$\\quad Win[D,F_Y]$ (column-sharded)$\\quad Wout[F_Y,D]$ (row-sharded)",
    body:
      "Two GPUs share mesh axis $Y$. Unlike FSDP, where the weights themselves get gathered and scattered, here the weight shards $Win[D,F_Y]$ and $Wout[F_Y,D]$ never move — they sit fixed on their device for this entire block. What moves instead is the <b>activation</b>: the residual stream is kept sharded along the feature dimension $D$ between blocks (cheaper in memory than keeping it fully replicated), and gets temporarily gathered and re-scattered as it passes through.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B,D_Y]", "ghost"),
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
      }),
      null
    ),
    matrices: [
      mat("In", "B", "D_Y", { shardAxis: "cols", tone: "act" }),
      mat("Win", "D", "F_Y", { shardAxis: "cols", tone: "weight" }),
      mat("Wout", "F_Y", "D", { shardAxis: "rows", tone: "weight" }),
    ],
  },
  {
    id: 2,
    title: "Line 1 — AllGather In",
    notation: "$In[B,D] = \\text{AllGather}_Y\\!\\left(In[B,D_Y]\\right)$",
    comms: "$2BD$ bytes — an AllGather costs $\\approx 1\\times$ the gathered array's bytes",
    body: "On the critical path — unlike FSDP's weight AllGathers, this can't be prefetched during a previous step: the block's first matmul can't start until the full activation is in hand. Every device ends this step holding the same, fully replicated $In[B,D]$.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B,D]", "active"),
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
      }),
      { type: "allgather", targetId: "In", label: "$In[B,D] = \\text{AllGather}_Y(In[B,D_Y])$" }
    ),
    matrices: [
      mat("In", "B", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→", { comm: "allgather" }),
      mat("In", "B", "D", { tone: "act", state: "active" }),
    ],
  },
  {
    id: 3,
    title: "Line 2 — column matmul",
    notation: "$Tmp[B,F_Y] = In[B,D] \\cdot_D Win[D,F_Y]$",
    flops: "$2BDF_Y$ — the mesh axis $Y$ already divides $F$ down, so no extra $/Y$ needed here",
    body: "The contracting dimension $D$ isn't sharded — only $F$ is, via $Win$ — so this matmul needs zero communication. Each device produces its own $F_Y$ shard of $Tmp$.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B,D]", "solid"),
        Win: node("Win[D,F_Y]", "solid"),
        Tmp: node("Tmp[B,F_Y]", "active"),
        Wout: node("Wout[F_Y,D]", "solid"),
      }),
      null
    ),
    matrices: [
      mat("In", "B", "D", { tone: "act" }),
      op("×"),
      mat("Win", "D", "F_Y", { shardAxis: "cols", tone: "weight" }),
      op("="),
      mat("Tmp", "B", "F_Y", { shardAxis: "cols", tone: "act", state: "active" }),
    ],
  },
  {
    id: 4,
    title: "Line 3 — row matmul, unreduced",
    notation: "$Out[B,D]^{\\{U_Y\\}} = Tmp[B,F_Y] \\cdot_F Wout[F_Y,D]$",
    flops: "$2BF_YD$",
    body: "Now $F$ is the contracting dimension and it <i>is</i> sharded, so each device only computes a partial contribution — tagged $\\{U_Y\\}$ for “unreduced along $Y$”: the right shape, but the wrong (incomplete) value until every device's partial is combined.",
    diagram: diagramFrom(
      snapshot({
        Tmp: node("Tmp[B,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        Win: node("Win[D,F_Y]", "solid"),
        Out: node("Out[B,D]^{\\{U_Y\\}}", "partial"),
      }),
      null
    ),
    matrices: [
      mat("Tmp", "B", "F_Y", { shardAxis: "cols", tone: "act" }),
      op("×"),
      mat("Wout", "F_Y", "D", { shardAxis: "rows", tone: "weight" }),
      op("="),
      mat("Out", "B", "D", { tone: "act", state: "partial" }),
    ],
  },
  {
    id: 5,
    title: "Line 4 — ReduceScatter Out",
    notation: "$Out[B,D_Y] = \\text{ReduceScatter}_Y\\!\\left(Out[B,D]^{\\{U_Y\\}}\\right)$",
    comms: "$2BD$ bytes — a ReduceScatter costs $\\approx 1\\times$ the pre-scatter array's bytes",
    body: "On the critical path — this single collective both sums the partial output across $Y$ and re-shards it along $D$ in the same step, so the block ends exactly like it started: sharded, never fully materialized on one device. An AllReduce is exactly an AllGather followed by a ReduceScatter — splitting the block's two ends like this is what lets each half overlap with a neighboring block's compute.",
    diagram: diagramFrom(
      snapshot({
        Tmp: node("Tmp[B,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        Win: node("Win[D,F_Y]", "solid"),
        Out: node("Out[B,D_Y]", "active"),
      }),
      { type: "reducescatter", targetId: "Out", label: "$Out[B,D_Y] = \\text{ReduceScatter}_Y(Out[B,D]^{\\{U_Y\\}})$" }
    ),
    matrices: [
      mat("Out", "B", "D", { tone: "act", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("Out", "B", "D_Y", { shardAxis: "cols", tone: "act", state: "active" }),
    ],
  },
  {
    id: 6,
    title: "Line 5 — loss",
    notation: "$Loss[B] = \\ldots$",
    body:
      "Forward pass complete — $Out[B,D_Y]$ is exactly the sharded resting state the next block (or the loss) expects as its own input. The book leaves this line as “…” deliberately: whatever the loss function actually does with $Out$ (e.g. gather it first, or reduce it as-is) is outside the scope of the parallelism strategy being illustrated here, so the shapes panel below shows $Loss[B]$ as an unspecified reduction rather than claiming a concrete shape. Backward now needs to produce $dWout[F_Y,D]$ and $dWin[D,F_Y]$.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Loss: node("Loss[B]", "active"),
      }),
      null
    ),
    matrices: [
      mat("Out", "B", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→"),
      mat("Loss", "B", "\\ldots", { tone: "act", state: "active" }),
    ],
  },
  {
    id: 7,
    title: "Line 6 — dOut",
    notation: "$dOut[B,D_Y] = \\ldots$",
    body: "Backward starts from the next layer (or the loss) handing back a gradient in exactly the same sharded layout the forward pass produced its output in. $Tmp$ is kept around from the forward pass — it's needed again in a few lines.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        dOut: node("dOut[B,D_Y]", "active"),
      }),
      null
    ),
    matrices: [mat("dOut", "B", "D_Y", { shardAxis: "cols", tone: "grad", state: "active" })],
  },
  {
    id: 8,
    title: "Line 7 — AllGather dOut",
    notation: "$dOut[B,D] = \\text{AllGather}_Y\\!\\left(dOut[B,D_Y]\\right)$",
    comms: "$2BD$ bytes",
    body: "On the critical path — mirrors the $In$ AllGather exactly, but for the gradient: $dOut$ must be fully replicated along $D$ before it can be contracted against $Wout$'s replicated $D$ side.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        dOut: node("dOut[B,D]", "active"),
      }),
      { type: "allgather", targetId: "dOut", label: "$dOut[B,D] = \\text{AllGather}_Y(dOut[B,D_Y])$" }
    ),
    matrices: [
      mat("dOut", "B", "D_Y", { shardAxis: "cols", tone: "grad" }),
      op("→", { comm: "allgather" }),
      mat("dOut", "B", "D", { tone: "grad", state: "active" }),
    ],
  },
  {
    id: 9,
    title: "Line 8 — dWout, local",
    notation: "$dWout[F_Y,D] = Tmp[B,F_Y] \\cdot_B dOut[B,D]$",
    flops: "$2BF_YD$",
    body: "Contract over the batch dimension $B$. $Tmp$'s shard and the now-gathered $dOut$ are both already local — this weight gradient is a pure local matmul, no communication needed.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "active"),
      }),
      null
    ),
    matrices: [
      mat("Tmp", "B", "F_Y", { shardAxis: "cols", tone: "act" }),
      op("×"),
      mat("dOut", "B", "D", { tone: "grad" }),
      op("="),
      mat("dWout", "F_Y", "D", { shardAxis: "rows", tone: "grad", state: "active" }),
    ],
  },
  {
    id: 10,
    title: "Line 9 — dTmp, local",
    notation: "$dTmp[B,F_Y] = dOut[B,D] \\cdot_D Wout[F_Y,D]$",
    flops: "$2BDF_Y$",
    body: "Contract over $D$, fully replicated on both operands — local, no communication. $dOut[B,D]$ can be thrown away right after this; it isn't needed again.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Win: node("Win[D,F_Y]", "solid"),
        Wout: node("Wout[F_Y,D]", "solid"),
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "active"),
      }),
      null
    ),
    matrices: [
      mat("dOut", "B", "D", { tone: "grad" }),
      op("×"),
      mat("Wout", "F_Y", "D", { shardAxis: "rows", tone: "weight" }),
      op("="),
      mat("dTmp", "B", "F_Y", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
  {
    id: 11,
    title: "Line 10 — AllGather In (reused)",
    notation: "$In[B,D] = \\text{AllGather}_Y\\!\\left(In[B,D_Y]\\right)$",
    comms: "$2BD$ bytes if actually re-paid — but see note",
    body: "The column-parallel weight's gradient needs the same fully-gathered $In[B,D]$ the forward pass produced back at the $In$ AllGather.",
    note: "This can be skipped entirely by simply keeping (or re-checkpointing) that earlier $In$ AllGather's result instead of paying for the same AllGather twice — the book calls this out as a direct saving. That's the $2BD$ this topic's final Tcomms total assumes is <i>not</i> repaid.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        In: node("In[B,D]", "active"),
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "solid"),
      }),
      { type: "allgather", targetId: "In", label: "$In[B,D] = \\text{AllGather}_Y(In[B,D_Y])$" }
    ),
    matrices: [
      mat("In", "B", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→", { comm: "allgather" }),
      mat("In", "B", "D", { tone: "act", state: "active" }),
    ],
  },
  {
    id: 12,
    title: "Line 11 — dWin, local",
    notation: "$dWin[D,F_Y] = In[B,D] \\cdot_B dTmp[B,F_Y]$",
    flops: "$2BDF_Y$",
    body: "Contract over batch $B$ again — both operands local, no communication, exactly mirroring $dWout$'s local derivation.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "solid"),
        dWin: node("dWin[D,F_Y]", "active"),
      }),
      null
    ),
    matrices: [
      mat("In", "B", "D", { tone: "act" }),
      op("×"),
      mat("dTmp", "B", "F_Y", { shardAxis: "cols", tone: "grad" }),
      op("="),
      mat("dWin", "D", "F_Y", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
  {
    id: 13,
    title: "Line 12 — dIn, unreduced",
    notation: "$dIn[B,D]^{\\{U_Y\\}} = dTmp[B,F_Y] \\cdot_F Win[D,F_Y]$",
    flops: "$2BF_YD$",
    body: "Contract over the sharded $F_Y$ — each device produces only a partial contribution, tagged $\\{U_Y\\}$, mirroring the row matmul's unreduced output exactly. This is the gradient the <i>previous</i> layer needs.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "solid"),
        dWin: node("dWin[D,F_Y]", "solid"),
        dIn: node("dIn[B,D]^{\\{U_Y\\}}", "partial"),
      }),
      null
    ),
    matrices: [
      mat("dTmp", "B", "F_Y", { shardAxis: "cols", tone: "grad" }),
      op("×"),
      mat("Win", "D", "F_Y", { shardAxis: "cols", tone: "weight" }),
      op("="),
      mat("dIn", "B", "D", { tone: "grad", state: "partial" }),
    ],
  },
  {
    id: 14,
    title: "Line 13 — ReduceScatter dIn",
    notation: "$dIn[B,D_Y] = \\text{ReduceScatter}_Y\\!\\left(dIn[B,D]^{\\{U_Y\\}}\\right)$",
    comms: "$2BD$ bytes",
    body: "On the critical path — closes the block exactly like the $Out$ ReduceScatter did for the forward pass: sum the partial gradient across $Y$ and re-shard along $D$, handing the previous layer a $dIn[B,D_Y]$ in the same sharded resting state its own forward output was in.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "solid"),
        dWin: node("dWin[D,F_Y]", "solid"),
        dIn: node("dIn[B,D_Y]", "active"),
      }),
      { type: "reducescatter", targetId: "dIn", label: "$dIn[B,D_Y] = \\text{ReduceScatter}_Y(dIn[B,D]^{\\{U_Y\\}})$" }
    ),
    matrices: [
      mat("dIn", "B", "D", { tone: "grad", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("dIn", "B", "D_Y", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
  {
    id: 15,
    title: "Compute vs. communication",
    notation: "$T_{\\text{math}} = \\dfrac{4BDF}{YC} \\qquad T_{\\text{comms}} = \\dfrac{2\\cdot2\\cdot BD}{W_{ici}} = \\dfrac{4BD}{W_{ici}}$",
    flops: "the book models the forward pass only (2 matmuls, $4BDF_Y=\\dfrac{4BDF}{Y}$) — “the backwards pass is just the transpose of each operation here”",
    comms: "forward pass only: the $In$ AllGather ($2BD$) $+$ the $Out$ ReduceScatter ($2BD$) $= 4BD$ bytes",
    body:
      "Matches the book's own derivation exactly, and it's the mirror image of DP's: DP shards the <b>batch</b> and divides FLOPs by $X$, moving <b>weight-sized</b> data ($DF$); TP shards the <b>feature width</b> $F$ and divides FLOPs by $Y$, moving <b>activation-sized</b> data ($BD$) instead — same $4\\times$/$4\\times$ shape, just which axis gets divided and which tensor moves has swapped.",
    note:
      "Compute-bound ($T_{\\text{math}} > T_{\\text{comms}}$) when $\\dfrac{F}{Y} > \\dfrac{C}{W_{ici}}$, i.e. $F > Y\\cdot\\dfrac{C}{W_{ici}}$ — the book's own stated result verbatim. Summed over the <i>whole</i> step (both passes, and assuming the second $In$ AllGather is reused rather than repaid) the total is $12BDF/Y$ FLOPs against $8BD$ comms bytes — same $4\\times$-per-pass shape doubled, so the ratio (and threshold) comes out identical either way.",
    formula: "$\\text{compute-bound} \\iff F > Y\\cdot\\dfrac{C}{W_{ici}}$",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        dOut: node("dOut[B,D]", "solid"),
        dWout: node("dWout[F_Y,D]", "solid"),
        dTmp: node("dTmp[B,F_Y]", "solid"),
        dWin: node("dWin[D,F_Y]", "solid"),
        dIn: node("dIn[B,D_Y]", "solid"),
      }),
      null
    ),
  },
];
