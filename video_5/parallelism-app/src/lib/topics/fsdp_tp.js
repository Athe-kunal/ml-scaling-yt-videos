// FSDP + TP combined — a 2D device mesh with axes X (the FSDP/data axis)
// and Y (the tensor-parallel axis), one step per line of the exact
// pseudocode given for this walkthrough. Same 2-matmul MLP block as
// fsdp.js/tp.js, but every tensor is now sharded along BOTH axes at once:
//   - batch B is always split by X (never gathered — the DP/FSDP axis)
//   - Win's/Wout's D dimension is split by X at rest, AllGathered/
//     ReduceScattered across X exactly like plain FSDP (fsdp.js)
//   - Win's F / Wout's F is ALWAYS split by Y, never gathered — the
//     permanent TP column/row split (matches tp.js's fixed weights)
//   - the activations' feature dim (In/Tmp/Out/dOut/dTmp/dIn's D or F)
//     cycles between full and Y-sharded via AllGather/ReduceScatter
//     across Y, exactly like plain TP (tp.js)
// So this is literally FSDP's X-axis mechanics and TP's Y-axis mechanics
// layered on the same two matmuls — nothing new, just both at once.
//
// Diagram nodes are pure LaTeX (rendered via MathLabel, no $ needed).
// notation/body/note/comm.label are prose+math strings using $...$ spans
// (see lib/latex.js). Reuses FSDPDiagram (kind: "fsdp_tp") since this is
// still the same forward/backward named-tensor shape as FSDP/TP.
//
// State is accumulated across steps rather than hand-repeated in every
// entry: each RAW_STEP only lists its `delta` (what this line changes),
// and the reduce below carries every other box's last-known state
// forward automatically. Hand-repeating full snapshots per step (the
// previous approach) is exactly how a box silently drops to "hidden" the
// moment one step's author forgets to re-list it — this structure makes
// that bug impossible instead of relying on remembering to avoid it.

function node(label, state) {
  return { label, state };
}

const HIDDEN = () => node("", "hidden");

const INITIAL_STATE = {
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

function diagramFrom(state, comm) {
  const forward = ["In", "Win", "Tmp", "Wout", "Out", "Loss"].map((id) => ({ id, ...state[id] }));
  const backward = ["dOut", "dWout", "dTmp", "dWin", "dIn"].map((id) => ({ id, ...state[id] }));
  return { kind: "fsdp_tp", forward, backward, comm };
}

// Shapes & sharding row builders — see MatrixShapes.jsx / tp.js for the
// full contract. Only the axis this line is actually operating on gets
// drawn sharded; the tensor's OTHER, permanently-sharded axis (e.g.
// Win/Wout's F, always split by Y) is called out in prose instead of
// double-encoded in the icon.
function mat(id, rows, cols, opts = {}) {
  return { id, rows, cols, shardAxis: null, state: "solid", tone: "act", ...opts };
}
function op(symbol, opts = {}) {
  return { op: symbol, ...opts };
}

const RAW_STEPS = [
  {
    title: "Setup — sharded along both X and Y",
    notation: "at rest: $In[B_X,D_Y]$, $Win[D_X,F_Y]$, $Wout[F_Y,D_X]$",
    body:
      "Two mesh axes at once. $X$ is the FSDP/data axis: the batch is always split by $X$, and the weights $Win,Wout$ are $X$-sharded at rest, just like plain FSDP. $Y$ is the tensor-parallel axis: $Win$'s $F$ and $Wout$'s $F$ are <i>permanently</i> split by $Y$ and never gathered — and the activations ($In$, $Tmp$, $Out$) get their feature dimension split by $Y$ too, gathered and re-scattered as they flow through, just like plain TP. Combining both cuts memory further than either alone.",
    delta: {
      In: node("In[B_X,D_Y]", "ghost"),
      Win: node("Win[D_X,F_Y]", "ghost"),
      Wout: node("Wout[F_Y,D_X]", "ghost"),
    },
    matrices: [
      mat("In", "B_X", "D_Y", { shardAxis: "cols", tone: "act" }),
      mat("Win", "D_X", "F_Y", { shardAxis: "rows", tone: "weight" }),
      mat("Wout", "F_Y", "D_X", { shardAxis: "cols", tone: "weight" }),
    ],
  },
  {
    title: "Line 1 — AllGather In over Y",
    notation: "$In[B_X,D] = \\text{AllGather}_Y\\!\\left(In[B_X,D_Y]\\right)$",
    body: "On the critical path — the block's first matmul can't start until $In$ is fully gathered along $Y$. ($B$ stays $X$-sharded throughout; only $Y$-sharding is touched here.)",
    delta: { In: node("In[B_X,D]", "active") },
    comm: { type: "allgather", targetId: "In", label: "$In[B_X,D] = \\text{AllGather}_Y(In[B_X,D_Y])$" },
    matrices: [
      mat("In", "B_X", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→", { comm: "allgather" }),
      mat("In", "B_X", "D", { tone: "act", state: "active" }),
    ],
  },
  {
    title: "Line 2 — AllGather Win over X",
    notation: "$Win[D,F_Y] = \\text{AllGather}_X\\!\\left(Win[D_X,F_Y]\\right)$",
    body: "Can be done ahead of time — prefetched during the previous layer's compute, exactly like plain FSDP's weight AllGathers. $Win$'s $F$ stays $Y$-sharded; only its $X$-sharded $D$ gets gathered.",
    delta: { In: node("In[B_X,D]", "solid"), Win: node("Win[D,F_Y]", "active") },
    comm: { type: "allgather", targetId: "Win", label: "$Win[D,F_Y] = \\text{AllGather}_X(Win[D_X,F_Y])$" },
    matrices: [
      mat("Win", "D_X", "F_Y", { shardAxis: "rows", tone: "weight" }),
      op("→", { comm: "allgather" }),
      mat("Win", "D", "F_Y", { tone: "weight", state: "active" }),
    ],
  },
  {
    title: "Line 3 — column matmul",
    notation: "$Tmp[B_X,F_Y] = In[B_X,D] \\cdot_D Win[D,F_Y]$",
    body: "Local — the contracted dimension $D$ is fully materialized on both operands, so this needs no communication. $In[B_X,D]$ can be thrown away right after; it won't be needed again until the backward pass re-gathers it.",
    delta: { Win: node("Win[D,F_Y]", "solid"), Tmp: node("Tmp[B_X,F_Y]", "active") },
    matrices: [
      mat("In", "B_X", "D", { tone: "act" }),
      op("×"),
      mat("Win", "D", "F_Y", { shardAxis: "cols", tone: "weight" }),
      op("="),
      mat("Tmp", "B_X", "F_Y", { shardAxis: "cols", tone: "act", state: "active" }),
    ],
  },
  {
    title: "Line 4 — AllGather Wout over X",
    notation: "$Wout[F_Y,D] = \\text{AllGather}_X\\!\\left(Wout[F_Y,D_X]\\right)$",
    body: "Can be done ahead of time, same as $Win$'s gather. $Win$ has already dropped back to its $X$-sharded resting state, and $In$ has been thrown away.",
    delta: {
      In: node("In[B_X,D_Y]", "ghost"),
      Win: node("Win[D_X,F_Y]", "ghost"),
      Tmp: node("Tmp[B_X,F_Y]", "solid"),
      Wout: node("Wout[F_Y,D]", "active"),
    },
    comm: { type: "allgather", targetId: "Wout", label: "$Wout[F_Y,D] = \\text{AllGather}_X(Wout[F_Y,D_X])$" },
    matrices: [
      mat("Wout", "F_Y", "D_X", { shardAxis: "cols", tone: "weight" }),
      op("→", { comm: "allgather" }),
      mat("Wout", "F_Y", "D", { tone: "weight", state: "active" }),
    ],
  },
  {
    title: "Line 5 — row matmul, unreduced over Y",
    notation: "$Out[B_X,D]^{\\{U_Y\\}} = Tmp[B_X,F_Y] \\cdot_F Wout[F_Y,D]$",
    body: "Now $F$ is the contracted dimension and it's $Y$-sharded, so each device only produces a partial sum — tagged $\\{U_Y\\}$ for “unreduced along $Y$.” $Wout$ can be thrown away right after.",
    delta: { Wout: node("Wout[F_Y,D_X]", "ghost"), Out: node("Out[B_X,D]^{\\{U_Y\\}}", "partial") },
    matrices: [
      mat("Tmp", "B_X", "F_Y", { shardAxis: "cols", tone: "act" }),
      op("×"),
      mat("Wout", "F_Y", "D", { shardAxis: "rows", tone: "weight" }),
      op("="),
      mat("Out", "B_X", "D", { tone: "act", state: "partial" }),
    ],
  },
  {
    title: "Line 6 — ReduceScatter Out over Y",
    notation: "$Out[B_X,D_Y] = \\text{ReduceScatter}_Y\\!\\left(Out[B_X,D]^{\\{U_Y\\}}\\right)$",
    body: "On the critical path — sums the partial output across $Y$ and re-shards it in the same step, handing the next block (or the loss) an $Out[B_X,D_Y]$ in the same $Y$-sharded resting state $In$ started this block in.",
    delta: { Out: node("Out[B_X,D_Y]", "active") },
    comm: { type: "reducescatter", targetId: "Out", label: "$Out[B_X,D_Y] = \\text{ReduceScatter}_Y(Out[B_X,D]^{\\{U_Y\\}})$" },
    matrices: [
      mat("Out", "B_X", "D", { tone: "act", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("Out", "B_X", "D_Y", { shardAxis: "cols", tone: "act", state: "active" }),
    ],
  },
  {
    title: "Line 7 — loss",
    notation: "$Loss[B_X] = \\ldots$",
    body: "Forward pass complete. Backward now needs to produce $dWout[F_Y,D_X]$ and $dWin[D_X,F_Y]$ — the doubly-sharded gradients this device will actually use to update its own weight shard.",
    delta: { Out: node("Out[B_X,D_Y]", "solid"), Loss: node("Loss[B_X]", "active") },
    matrices: [
      mat("Out", "B_X", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→"),
      mat("Loss", "B_X", "\\ldots", { tone: "act", state: "active" }),
    ],
  },
  {
    title: "Line 8 — dOut",
    notation: "$dOut[B_X,D_Y] = \\ldots$",
    body: "Backward starts from the next layer (or the loss) handing back a gradient in the same $Y$-sharded layout $Out$ ended the forward pass in.",
    delta: { Loss: node("Loss[B_X]", "solid"), dOut: node("dOut[B_X,D_Y]", "active") },
    matrices: [mat("dOut", "B_X", "D_Y", { shardAxis: "cols", tone: "grad", state: "active" })],
  },
  {
    title: "Line 9 — AllGather dOut over Y",
    notation: "$dOut[B_X,D] = \\text{AllGather}_Y\\!\\left(dOut[B_X,D_Y]\\right)$",
    body: "On the critical path — mirrors line 1 exactly, but for the gradient: $dOut$ must be fully replicated along $D$ before it can be contracted against anything.",
    delta: { dOut: node("dOut[B_X,D]", "active") },
    comm: { type: "allgather", targetId: "dOut", label: "$dOut[B_X,D] = \\text{AllGather}_Y(dOut[B_X,D_Y])$" },
    matrices: [
      mat("dOut", "B_X", "D_Y", { shardAxis: "cols", tone: "grad" }),
      op("→", { comm: "allgather" }),
      mat("dOut", "B_X", "D", { tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 10 — dWout, unreduced over X",
    notation: "$dWout[F_Y,D]^{\\{U_X\\}} = Tmp[B_X,F_Y] \\cdot_B dOut[B_X,D]$",
    body: "Contract over the batch dimension $B$ — but $B$ is $X$-sharded, so each device only holds its own batch slice, and the result is only a partial sum, tagged $\\{U_X\\}$ for “unreduced along $X$.” This is exactly FSDP's gradient accumulation across the data-parallel replicas.",
    delta: { dOut: node("dOut[B_X,D]", "solid"), dWout: node("dWout[F_Y,D]^{\\{U_X\\}}", "partial") },
    matrices: [
      mat("Tmp", "B_X", "F_Y", { shardAxis: "cols", tone: "act" }),
      op("×"),
      mat("dOut", "B_X", "D", { tone: "grad" }),
      op("="),
      mat("dWout", "F_Y", "D", { tone: "grad", state: "partial" }),
    ],
  },
  {
    title: "Line 11 — ReduceScatter dWout over X",
    notation: "$dWout[F_Y,D_X] = \\text{ReduceScatter}_X\\!\\left(dWout[F_Y,D]^{\\{U_X\\}}\\right)$",
    body: "Sums the partial gradient across the data-parallel axis $X$ and re-shards it in the same step — the doubly-sharded gradient this device will use to update its own $Wout$ shard.",
    delta: { dWout: node("dWout[F_Y,D_X]", "active") },
    comm: { type: "reducescatter", targetId: "dWout", label: "$dWout[F_Y,D_X] = \\text{ReduceScatter}_X(dWout[F_Y,D]^{\\{U_X\\}})$" },
    matrices: [
      mat("dWout", "F_Y", "D", { tone: "grad", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("dWout", "F_Y", "D_X", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 12 — AllGather Wout over X (again)",
    notation: "$Wout[F_Y,D] = \\text{AllGather}_X\\!\\left(Wout[F_Y,D_X]\\right)$",
    body: "Can be done ahead of time. Same weight as the forward pass, gathered again — the backward pass needs it for the very next line.",
    delta: { Wout: node("Wout[F_Y,D]", "active"), dWout: node("dWout[F_Y,D_X]", "solid") },
    comm: { type: "allgather", targetId: "Wout", label: "$Wout[F_Y,D] = \\text{AllGather}_X(Wout[F_Y,D_X])$" },
    matrices: [
      mat("Wout", "F_Y", "D_X", { shardAxis: "cols", tone: "weight" }),
      op("→", { comm: "allgather" }),
      mat("Wout", "F_Y", "D", { tone: "weight", state: "active" }),
    ],
  },
  {
    title: "Line 13 — dTmp",
    notation: "$dTmp[B_X,F_Y] = dOut[B_X,D] \\cdot_D Wout[F_Y,D]$",
    body: "Local — contract over the now-fully-gathered $D$, no communication needed. $dOut[B_X,D]$ isn't needed again after this.",
    delta: { Wout: node("Wout[F_Y,D]", "solid"), dTmp: node("dTmp[B_X,F_Y]", "active") },
    matrices: [
      mat("dOut", "B_X", "D", { tone: "grad" }),
      op("×"),
      mat("Wout", "F_Y", "D", { shardAxis: "rows", tone: "weight" }),
      op("="),
      mat("dTmp", "B_X", "F_Y", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 14 — AllGather In over Y (reused)",
    notation: "$In[B_X,D] = \\text{AllGather}_Y\\!\\left(In[B_X,D_Y]\\right)$",
    body: "Not on the critical path this time — and this gather can be shared with the <i>previous</i> layer's own forward-pass AllGather of $In$, so it only ever gets paid for once. $Wout$ has dropped back to sharded.",
    delta: { Wout: node("Wout[F_Y,D_X]", "ghost"), In: node("In[B_X,D]", "active"), dTmp: node("dTmp[B_X,F_Y]", "solid") },
    comm: { type: "allgather", targetId: "In", label: "$In[B_X,D] = \\text{AllGather}_Y(In[B_X,D_Y])$" },
    matrices: [
      mat("In", "B_X", "D_Y", { shardAxis: "cols", tone: "act" }),
      op("→", { comm: "allgather" }),
      mat("In", "B_X", "D", { tone: "act", state: "active" }),
    ],
  },
  {
    title: "Line 15 — dWin, unreduced over X",
    notation: "$dWin[D,F_Y]^{\\{U_X\\}} = In[B_X,D] \\cdot_B dTmp[B_X,F_Y]$",
    body: "Contract over batch $B$ again — the same FSDP-style accumulation as line 10's $dWout$, unreduced along $X$.",
    delta: { In: node("In[B_X,D]", "solid"), dWin: node("dWin[D,F_Y]^{\\{U_X\\}}", "partial") },
    matrices: [
      mat("In", "B_X", "D", { tone: "act" }),
      op("×"),
      mat("dTmp", "B_X", "F_Y", { shardAxis: "cols", tone: "grad" }),
      op("="),
      mat("dWin", "D", "F_Y", { tone: "grad", state: "partial" }),
    ],
  },
  {
    title: "Line 16 — ReduceScatter dWin over X",
    notation: "$dWin[D_X,F_Y] = \\text{ReduceScatter}_X\\!\\left(dWin[D,F_Y]^{\\{U_X\\}}\\right)$",
    body: "Sums across $X$ and re-shards — the doubly-sharded gradient this device will use to update its own $Win$ shard. $In$ has dropped back to sharded.",
    delta: { In: node("In[B_X,D_Y]", "ghost"), dWin: node("dWin[D_X,F_Y]", "active") },
    comm: { type: "reducescatter", targetId: "dWin", label: "$dWin[D_X,F_Y] = \\text{ReduceScatter}_X(dWin[D,F_Y]^{\\{U_X\\}})$" },
    matrices: [
      mat("dWin", "D", "F_Y", { tone: "grad", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("dWin", "D_X", "F_Y", { shardAxis: "rows", tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 17 — AllGather Win over X (again)",
    notation: "$Win[D,F_Y] = \\text{AllGather}_X\\!\\left(Win[D_X,F_Y]\\right)$",
    body: "Can be done ahead of time — one last full gather of $Win$, needed for the final line, which passes the gradient on to the previous layer.",
    delta: { Win: node("Win[D,F_Y]", "active"), dWin: node("dWin[D_X,F_Y]", "solid") },
    comm: { type: "allgather", targetId: "Win", label: "$Win[D,F_Y] = \\text{AllGather}_X(Win[D_X,F_Y])$" },
    matrices: [
      mat("Win", "D_X", "F_Y", { shardAxis: "rows", tone: "weight" }),
      op("→", { comm: "allgather" }),
      mat("Win", "D", "F_Y", { tone: "weight", state: "active" }),
    ],
  },
  {
    title: "Line 18 — dIn, unreduced over Y",
    notation: "$dIn[B_X,D]^{\\{U_Y\\}} = dTmp[B_X,F_Y] \\cdot_F Win[D,F_Y]$",
    body: "Contract over the $Y$-sharded $F$ — each device only produces a partial contribution, tagged $\\{U_Y\\}$, mirroring line 6 exactly. This is the gradient the <i>previous</i> layer needs.",
    delta: { Win: node("Win[D,F_Y]", "solid"), dIn: node("dIn[B_X,D]^{\\{U_Y\\}}", "partial") },
    matrices: [
      mat("dTmp", "B_X", "F_Y", { shardAxis: "cols", tone: "grad" }),
      op("×"),
      mat("Win", "D", "F_Y", { shardAxis: "cols", tone: "weight" }),
      op("="),
      mat("dIn", "B_X", "D", { tone: "grad", state: "partial" }),
    ],
  },
  {
    title: "Line 19 — ReduceScatter dIn over Y",
    notation: "$dIn[B_X,D_Y] = \\text{ReduceScatter}_Y\\!\\left(dIn[B_X,D]^{\\{U_Y\\}}\\right)$",
    body: "On the critical path — closes the block exactly like line 6 did for the forward pass: sum across $Y$ and re-shard, handing the previous layer a $dIn[B_X,D_Y]$ in the same resting state its own forward input was in. $Win[D,F_Y]$ can be thrown away here too, dropping back to its $X$-sharded resting state.",
    delta: { Win: node("Win[D_X,F_Y]", "ghost"), dIn: node("dIn[B_X,D_Y]", "active") },
    comm: { type: "reducescatter", targetId: "dIn", label: "$dIn[B_X,D_Y] = \\text{ReduceScatter}_Y(dIn[B_X,D]^{\\{U_Y\\}})$" },
    matrices: [
      mat("dIn", "B_X", "D", { tone: "grad", state: "partial" }),
      op("→", { comm: "reducescatter" }),
      mat("dIn", "B_X", "D_Y", { shardAxis: "cols", tone: "grad", state: "active" }),
    ],
  },
];

let runningState = INITIAL_STATE;
export const STEPS = RAW_STEPS.map((s, i) => {
  runningState = { ...runningState, ...s.delta };
  return {
    id: i + 1,
    title: s.title,
    notation: s.notation,
    body: s.body,
    note: s.note,
    diagram: diagramFrom(runningState, s.comm || null),
    matrices: s.matrices,
  };
});
