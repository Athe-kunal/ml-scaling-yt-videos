// No parallelism — the baseline. A single device, no mesh axes, no
// sharding, no AllGather/ReduceScatter: every tensor in this 2-matmul MLP
// block (In -> Win -> Tmp -> Wout -> Out -> Loss, then the mirrored
// backward pass) is simply fully materialized, all the time. This is the
// "before" picture that DP/ZeRO/FSDP/TP all shard pieces away from —
// meant to be walked through first, so every later topic reads as "now
// shard this one thing" against a plain reference point.
//
// Same per-line pseudocode convention as fsdp.js/tp.js/fsdp_tp.js (one
// step per line, reusing FSDPDiagram's forward/backward-row layout via
// kind: "no_parallelism"), and the same accumulate-by-delta state
// pattern as fsdp_tp.js: each RAW_STEP only lists what that line
// changes, so a box already drawn never has to be re-listed just to
// keep it from vanishing.

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

function diagramFrom(state) {
  const forward = ["In", "Win", "Tmp", "Wout", "Out", "Loss"].map((id) => ({ id, ...state[id] }));
  const backward = ["dOut", "dWout", "dTmp", "dWin", "dIn"].map((id) => ({ id, ...state[id] }));
  return { kind: "no_parallelism", forward, backward, comm: null };
}

// Shapes row — see MatrixShapes.jsx. Nothing is ever sharded here, so
// every mat() below omits shardAxis/state entirely (both default to
// "fully materialized" / "solid") except the one box each line produces,
// which gets state:"active" to highlight it.
function mat(id, rows, cols, opts = {}) {
  return { id, rows, cols, shardAxis: null, state: "solid", tone: "act", ...opts };
}
function op(symbol, opts = {}) {
  return { op: symbol, ...opts };
}

const RAW_STEPS = [
  {
    title: "Setup — one device, nothing sharded",
    notation: "$In[B,D]$, $Win[D,F]$, $Wout[F,D]$ — all fully materialized, one copy each",
    body:
      "No mesh, no axes, no communication anywhere in this walkthrough. Every tensor — the input $In[B,D]$ and the two weights $Win[D,F]$, $Wout[F,D]$ — simply exists in full on the one device doing the work. This is the reference point every other topic in this app shards a piece away from: DP/ZeRO/FSDP shard along a data axis $X$, TP shards along a model axis $Y$ — starting from exactly this picture.",
    delta: {
      In: node("In[B,D]", "solid"),
      Win: node("Win[D,F]", "solid"),
      Wout: node("Wout[F,D]", "solid"),
    },
    matrices: [
      mat("In", "B", "D"),
      mat("Win", "D", "F", { tone: "weight" }),
      mat("Wout", "F", "D", { tone: "weight" }),
    ],
  },
  {
    title: "Line 1 — column matmul",
    notation: "$Tmp[B,F] = In[B,D] \\cdot_D Win[D,F]$",
    body: "A single local matmul — no gathering, no scattering, nothing to synchronize. Contract over $D$ to produce $Tmp[B,F]$.",
    delta: { Tmp: node("Tmp[B,F]", "active") },
    matrices: [
      mat("In", "B", "D"),
      op("×"),
      mat("Win", "D", "F", { tone: "weight" }),
      op("="),
      mat("Tmp", "B", "F", { state: "active" }),
    ],
  },
  {
    title: "Line 2 — row matmul",
    notation: "$Out[B,D] = Tmp[B,F] \\cdot_F Wout[F,D]$",
    body: "Another single local matmul, contracting over $F$ this time. Unlike TP's row-parallel matmul, there's no partial sum here — $F$ isn't split across anything, so the result is already the complete, correct $Out[B,D]$.",
    delta: { Tmp: node("Tmp[B,F]", "solid"), Out: node("Out[B,D]", "active") },
    matrices: [
      mat("Tmp", "B", "F"),
      op("×"),
      mat("Wout", "F", "D", { tone: "weight" }),
      op("="),
      mat("Out", "B", "D", { state: "active" }),
    ],
  },
  {
    title: "Line 3 — loss",
    notation: "$Loss[B] = \\ldots$",
    body: "Forward pass complete. Backward now needs to produce $dWout[F,D]$ and $dWin[D,F]$ — full, unsharded gradients, since nothing was ever split in the first place.",
    delta: { Out: node("Out[B,D]", "solid"), Loss: node("Loss[B]", "active") },
    matrices: [mat("Out", "B", "D"), op("→"), mat("Loss", "B", "\\ldots", { state: "active" })],
  },
  {
    title: "Line 4 — dOut",
    notation: "$dOut[B,D] = \\ldots$",
    body: "Backward starts from the loss gradient, same shape as $Out$. $Tmp$ was kept around from the forward pass — it's needed again on the very next line.",
    delta: { Loss: node("Loss[B]", "solid"), dOut: node("dOut[B,D]", "active") },
    matrices: [mat("dOut", "B", "D", { tone: "grad", state: "active" })],
  },
  {
    title: "Line 5 — dWout",
    notation: "$dWout[F,D] = Tmp[B,F] \\cdot_B dOut[B,D]$",
    body: "Contract over the batch dimension $B$ to get the full weight gradient in one step — no partial sum, no reduction, because $B$ was never split across devices in the first place.",
    delta: { dOut: node("dOut[B,D]", "solid"), dWout: node("dWout[F,D]", "active") },
    matrices: [
      mat("Tmp", "B", "F"),
      op("×"),
      mat("dOut", "B", "D", { tone: "grad" }),
      op("="),
      mat("dWout", "F", "D", { tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 6 — dTmp",
    notation: "$dTmp[B,F] = dOut[B,D] \\cdot_D Wout[F,D]$",
    body: "Contract over $D$ to propagate the gradient back through $Wout$. $dOut[B,D]$ isn't needed again after this.",
    delta: { dWout: node("dWout[F,D]", "solid"), dTmp: node("dTmp[B,F]", "active") },
    matrices: [
      mat("dOut", "B", "D", { tone: "grad" }),
      op("×"),
      mat("Wout", "F", "D", { tone: "weight" }),
      op("="),
      mat("dTmp", "B", "F", { tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 7 — dWin",
    notation: "$dWin[D,F] = In[B,D] \\cdot_B dTmp[B,F]$",
    body: "Contract over $B$ again, mirroring line 5, to get the other weight's full gradient — again, no partial sum needed.",
    delta: { dTmp: node("dTmp[B,F]", "solid"), dWin: node("dWin[D,F]", "active") },
    matrices: [
      mat("In", "B", "D"),
      op("×"),
      mat("dTmp", "B", "F", { tone: "grad" }),
      op("="),
      mat("dWin", "D", "F", { tone: "grad", state: "active" }),
    ],
  },
  {
    title: "Line 8 — dIn",
    notation: "$dIn[B,D] = dTmp[B,F] \\cdot_F Win[D,F]$",
    body: "Contract over $F$ to close out the block — a single local matmul, no partial sum, no ReduceScatter. This is exactly what the <i>previous</i> layer's backward pass needs. Compare this whole walkthrough to TP: there, this same line only produces a partial $dIn^{\\{U_Y\\}}$ that then needs a ReduceScatter — here, it's just the answer.",
    delta: { dIn: node("dIn[B,D]", "active") },
    matrices: [
      mat("dTmp", "B", "F", { tone: "grad" }),
      op("×"),
      mat("Win", "D", "F", { tone: "weight" }),
      op("="),
      mat("dIn", "B", "D", { tone: "grad", state: "active" }),
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
    diagram: diagramFrom(runningState),
    matrices: s.matrices,
  };
});
