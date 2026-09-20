// FSDP (= ZeRO-3, in the book's own notation) — one step per line of the
// exact pseudocode from https://jax-ml.github.io/scaling-book/training/,
// "Fully-Sharded Data Parallelism (FSDP)". Two 2-layer-MLP weights,
// Win[D,F] and Wout[F,D], sharded along mesh axis X at rest; each is
// temporarily AllGathered right before it's needed and thrown away right
// after, and every gradient is ReduceScattered back down to its shard —
// same mechanism as zero3.js, but walked through as the book's own
// forward/backward derivation instead of the deck's toy 4-weight chain.
//
// Diagram nodes are pure LaTeX (rendered via MathLabel, no $ needed).
// notation/body/note/comm.label are prose+math strings using $...$ spans
// (see lib/latex.js).

function node(label, state) {
  return { label, state };
}

const HIDDEN = () => node("", "hidden");

// A step's diagram = a full snapshot of every named tensor's box. Each
// step below is built by cloning the previous step's snapshot and
// overriding just what that pseudocode line changes — the same
// "cumulative reveal" every other topic uses, just written explicitly
// per box since FSDP's tensors get reused (Win/Wout toggle gathered vs.
// sharded multiple times) rather than only ever accumulating.

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

// Once the forward pass has finished (step 6 onward), In/Tmp/Out/Loss
// never change state again — folded into every backward-step snapshot
// below so stepping through the backward pass doesn't make the completed
// forward blocks vanish. (Win/Wout are intentionally excluded: they
// genuinely cycle gathered/sharded again during backward, which is the
// whole point of this walkthrough.)
const FWD_DONE = {
  In: node("In[B_X,D]", "solid"),
  Tmp: node("Tmp[B_X,F]", "solid"),
  Out: node("Out[B_X,D]", "solid"),
  Loss: node("Loss[B_X]", "solid"),
};

function diagramFrom(snap, comm) {
  const forward = ["In", "Win", "Tmp", "Wout", "Out", "Loss"].map((id) => ({ id, ...snap[id] }));
  const backward = ["dOut", "dWout", "dTmp", "dWin", "dIn"].map((id) => ({ id, ...snap[id] }));
  return { kind: "fsdp", forward, backward, comm };
}

export const STEPS = [
  {
    id: 1,
    title: "Setup — weights sharded at rest",
    notation: "at rest: $Win[D_X,F]$, $Wout[F,D_X]$ — each weight sharded along mesh axis $X$",
    body:
      "Same idea as ZeRO-3, in the book's own notation for a single 2-matmul MLP block. $Win[D,F]$ and $Wout[F,D]$ are never fully materialized on one device at rest — only their $X$-sharded slices, $Win[D_X,F]$ and $Wout[F,D_X]$, are stored. $In[B_X,D]$ (this device's batch shard) is already local and replicated in full.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D_X,F]", "ghost"),
        Wout: node("Wout[F,D_X]", "ghost"),
      }),
      null
    ),
  },
  {
    id: 2,
    title: "Line 1 — AllGather Win",
    notation: "$Win[D,F] = \\text{AllGather}_X\\!\\left(Win[D_X,F]\\right)$",
    body: "Not on the critical path — this AllGather can be prefetched during the <i>previous</i> layer's compute, so by the time this layer needs $Win$, it's already there.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D,F]", "active"),
        Wout: node("Wout[F,D_X]", "ghost"),
      }),
      { type: "allgather", targetId: "Win", label: "$Win[D,F] = \\text{AllGather}_X(Win[D_X,F])$" }
    ),
  },
  {
    id: 3,
    title: "Line 2 — column matmul",
    notation: "$Tmp[B_X,F] = In[B_X,D] \\cdot_D Win[D,F]$",
    body: "Both operands are already local — $In$ was replicated, $Win$ was just gathered — so this matmul needs no communication. $Win[D,F]$ can be thrown away right after; it isn't needed again until the backward pass.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D,F]", "solid"),
        Tmp: node("Tmp[B_X,F]", "active"),
        Wout: node("Wout[F,D_X]", "ghost"),
      }),
      null
    ),
  },
  {
    id: 4,
    title: "Line 3 — AllGather Wout",
    notation: "$Wout[F,D] = \\text{AllGather}_X\\!\\left(Wout[F,D_X]\\right)$",
    body: "Same as step 2, one layer later: not on the critical path, prefetchable during the previous layer. $Win$ has already been dropped back to its sharded resting state.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D_X,F]", "ghost"),
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D]", "active"),
      }),
      { type: "allgather", targetId: "Wout", label: "$Wout[F,D] = \\text{AllGather}_X(Wout[F,D_X])$" }
    ),
  },
  {
    id: 5,
    title: "Line 4 — row matmul",
    notation: "$Out[B_X,D] = Tmp[B_X,F] \\cdot_F Wout[F,D]$",
    body: "Again both operands are local, so no communication. This finishes the block's output.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D_X,F]", "ghost"),
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D]", "solid"),
        Out: node("Out[B_X,D]", "active"),
      }),
      null
    ),
  },
  {
    id: 6,
    title: "Line 5 — loss",
    notation: "$Loss[B_X] = \\ldots$",
    body: "Forward pass complete. Backward now needs to produce $dWout[F,D_X]$ and $dWin[D_X,F]$ — the sharded gradients this device will actually use to update its own weight shard.",
    diagram: diagramFrom(
      snapshot({
        In: node("In[B_X,D]", "solid"),
        Win: node("Win[D_X,F]", "ghost"),
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D_X]", "ghost"),
        Out: node("Out[B_X,D]", "solid"),
        Loss: node("Loss[B_X]", "active"),
      }),
      null
    ),
  },
  {
    id: 7,
    title: "Line 6 — dOut",
    notation: "$dOut[B_X,D] = \\ldots$",
    body: "Backward starts from the loss gradient, same shape as $Out$. $Tmp$ was kept around from the forward pass — it's needed again on the very next line.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "active"),
      }),
      null
    ),
  },
  {
    id: 8,
    title: "Line 7 — local, unreduced dWout",
    notation: "$dWout[F,D]^{\\{U_X\\}} = Tmp[B_X,F] \\cdot_B dOut[B_X,D]$",
    body: "Each device computes its own local contribution toward $dWout$ — but it's only a partial sum over this device's batch shard, tagged $\\{U_X\\}$ for “unreduced along $X$.” It isn't the real gradient until it's combined with every other device's partial.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D]^{\\{U_X\\}}", "partial"),
      }),
      null
    ),
  },
  {
    id: 9,
    title: "Line 8 — ReduceScatter dWout",
    notation:
      "$dWout[F,D_X] = \\text{ReduceScatter}_X\\!\\left(dWout[F,D]^{\\{U_X\\}}\\right) = \\dfrac{dWout^{(0)}[F,D] + dWout^{(1)}[F,D]}{X}$",
    body:
      "Not on the critical path — this reduction can happen asynchronously, overlapped with the compute that follows. Concretely, \"reduce\" here means each device's local partial is <i>summed</i> across all $X$ devices and divided by the world size $X$ — the same sum-then-average an AllReduce would compute. The \"scatter\" half is what's different: instead of handing that averaged result to <i>every</i> device, each device keeps only its own $D_X$ shard of it — exactly like ZeRO-2/3's gradient sync.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "active"),
      }),
      {
        type: "reducescatter",
        targetId: "dWout",
        label: "$dWout[F,D_X] = \\text{ReduceScatter}_X(dWout[F,D]^{\\{U_X\\}}) = \\dfrac{dWout^{(0)}+dWout^{(1)}}{X}\\Big|_{D_X\\text{ shard}}$",
      }
    ),
  },
  {
    id: 10,
    title: "Line 9 — AllGather Wout (again)",
    notation: "$Wout[F,D] = \\text{AllGather}_X\\!\\left(Wout[F,D_X]\\right)$",
    body: "Can be done ahead of time. Same weight as the forward pass — gathered a second time, because the backward pass needs it too, for the very next line.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Tmp: node("Tmp[B_X,F]", "solid"),
        Wout: node("Wout[F,D]", "active"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
      }),
      { type: "allgather", targetId: "Wout", label: "$Wout[F,D] = \\text{AllGather}_X(Wout[F,D_X])$" }
    ),
  },
  {
    id: 11,
    title: "Line 10 — dTmp",
    notation: "$dTmp[B_X,F] = dOut[B_X,D] \\cdot_D Wout[F,D]$",
    body: "Local matmul, no communication. $Wout[F,D]$ can be thrown away right after — it won't be needed again until the next layer's backward pass.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Wout: node("Wout[F,D]", "solid"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
        dTmp: node("dTmp[B_X,F]", "active"),
      }),
      null
    ),
  },
  {
    id: 12,
    title: "Line 11 — local, unreduced dWin",
    notation: "$dWin[D,F]^{\\{U_X\\}} = In[B_X,D] \\cdot_B dTmp[B_X,F]$",
    body: "Mirrors line 7: each device's own local, not-yet-reduced contribution toward $dWin$, tagged $\\{U_X\\}$. $Wout$ has already been dropped back to sharded.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
        dTmp: node("dTmp[B_X,F]", "solid"),
        dWin: node("dWin[D,F]^{\\{U_X\\}}", "partial"),
      }),
      null
    ),
  },
  {
    id: 13,
    title: "Line 12 — ReduceScatter dWin",
    notation:
      "$dWin[D_X,F] = \\text{ReduceScatter}_X\\!\\left(dWin[D,F]^{\\{U_X\\}}\\right) = \\dfrac{dWin^{(0)}[D,F] + dWin^{(1)}[D,F]}{X}$",
    body:
      "Same async, off-critical-path reduction as line 8, now for the other weight — sum each device's local partial across all $X$ devices, divide by $X$ to average, then keep only this device's $D_X$ shard of the result. This is the gradient this device will actually use to update its own $Win$ shard.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D_X,F]", "ghost"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
        dTmp: node("dTmp[B_X,F]", "solid"),
        dWin: node("dWin[D_X,F]", "active"),
      }),
      {
        type: "reducescatter",
        targetId: "dWin",
        label: "$dWin[D_X,F] = \\text{ReduceScatter}_X(dWin[D,F]^{\\{U_X\\}}) = \\dfrac{dWin^{(0)}+dWin^{(1)}}{X}\\Big|_{D_X\\text{ shard}}$",
      }
    ),
  },
  {
    id: 14,
    title: "Line 13 — AllGather Win (again)",
    notation: "$Win[D,F] = \\text{AllGather}_X\\!\\left(Win[D_X,F]\\right)$",
    body: "Can be done ahead of time. One last full gather of $Win$ — needed for the final line, which passes the gradient on to the previous layer.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Wout: node("Wout[F,D_X]", "ghost"),
        Win: node("Win[D,F]", "active"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
        dTmp: node("dTmp[B_X,F]", "solid"),
        dWin: node("dWin[D_X,F]", "solid"),
      }),
      { type: "allgather", targetId: "Win", label: "$Win[D,F] = \\text{AllGather}_X(Win[D_X,F])$" }
    ),
  },
  {
    id: 15,
    title: "Line 14 — dIn",
    notation: "$dIn[B_X,D] = dTmp[B_X,F] \\cdot_F Win[D,F]$",
    body: "Local matmul closes out the block. $dIn$ is what the <i>previous</i> layer's backward pass needs — this is exactly why $Win$ had to be re-gathered one more time. $Win[D,F]$ can be thrown away here too, dropping back to its sharded resting state.",
    diagram: diagramFrom(
      snapshot({
        ...FWD_DONE,
        Win: node("Win[D,F]", "solid"),
        dOut: node("dOut[B_X,D]", "solid"),
        dWout: node("dWout[F,D_X]", "solid"),
        dTmp: node("dTmp[B_X,F]", "solid"),
        dWin: node("dWin[D_X,F]", "solid"),
        dIn: node("dIn[B_X,D]", "active"),
      }),
      null
    ),
  },
];
