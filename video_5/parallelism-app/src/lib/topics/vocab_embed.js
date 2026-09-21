// Vocab-parallel embedding — Scaling_Lectures/Tensor Parallelism.pdf,
// pages 23-26. The book walks this one as a concrete worked numeric
// example rather than shape-level pseudocode, so this topic mirrors that:
// a 6-row embedding table (vocab size 6), split by contiguous ROW RANGE
// across 2 GPUs — a different sharding shape than the MLP/attention
// blocks (which shard a matmul weight's column/row dimension). Each GPU
// masks which of the 4 requested token indices it owns, looks them up
// locally, re-masks to zero out the ones it doesn't own, and a single
// AllReduce combines the two masked partials into the real answer.
//
// Rendered by EmbedLookupDiagram (kind: "vpe"). Rows are NOT
// cumulative/auto-carried like FSDPDiagram's forward/backward chain —
// each step lists exactly the rows relevant to it (same convention as
// tp.js's `matrices` field), since scratch rows like "local index" or
// "gathered" are genuinely thrown away once GPU1's masked result lands.

const EMBEDDINGS = [10, 20, 30, 40, 50, 60];
const X = [1, 4, 2, 5];
const RANGES = [
  { gpu: 1, lo: 0, hi: 2 },
  { gpu: 2, lo: 3, hi: 5 },
];

function cell(value, tone = null, dim = false) {
  return { value, tone, dim };
}
function row(label, cells) {
  return { label, cells };
}
function vpe(rows, opts = {}) {
  return { kind: "vpe", embeddings: EMBEDDINGS, index: X, ranges: opts.ranges ?? null, rows, comm: opts.comm ?? null };
}

export const STEPS = [
  {
    id: 1,
    title: "Setup — the lookup task",
    notation: "$Emb(X)[t] = Embeddings[X[t]]$, for $X = [1,4,2,5]$",
    body:
      "A concrete numeric example: a 6-row embedding table (vocab size 6) and 4 token indices $X=[1,4,2,5]$ to look up. This is the model's input (or output/unembedding) layer, and it's sharded differently from the MLP and attention blocks — those split a matmul weight by column or row; this splits a <b>lookup table</b> by row range over the vocabulary.",
    diagram: vpe([row("Emb(X) = ?", X.map(() => cell("?", null, true)))]),
  },
  {
    id: 2,
    title: "Shard the table by row range across 2 GPUs",
    notation: "$GPU1$ owns rows $0..2 = [10,20,30]$$\\quad$ $GPU2$ owns rows $3..5 = [40,50,60]$",
    body:
      "Each GPU physically stores only its own contiguous slice of the table — GPU1 holds rows 0-2, GPU2 holds rows 3-5. Neither GPU has the full table, and unlike a replicated activation, there's no AllGather that would even make sense here: the table is too large to ever fully materialize on one device (that's the whole point of sharding vocab embeddings for huge vocabularies).",
    diagram: vpe([], { ranges: RANGES }),
  },
  {
    id: 3,
    title: "GPU1 — which tokens does it own?",
    notation: "$mask_1 = (0 \\le X \\le 2) = (0 \\le [1,4,2,5] \\le 2) = [T,F,T,F]$",
    body:
      "Every GPU checks, independently, which of the 4 requested tokens fall inside the row range it owns. For GPU1 (rows 0-2): token 1 → owned, token 4 → not, token 2 → owned, token 5 → not.",
    diagram: vpe([row("mask (GPU1)", [cell("T", "gpu1"), cell("F", "gpu1", true), cell("T", "gpu1"), cell("F", "gpu1", true)])], { ranges: RANGES }),
  },
  {
    id: 4,
    title: "GPU1 — mask, then localize the index",
    notation:
      "$X_{GPU1} = mask_1 \\cdot X = [T,F,T,F] \\cdot [1,4,2,5] = [1,0,2,0]$<br/>$local_1 = [\\max(1{-}0,0),\\max(0{-}0,0),\\max(2{-}0,0),\\max(0{-}0,0)] = [1,0,2,0]$",
    body:
      "Not-owned positions are zeroed to a dummy index (0) first. Then the range's <i>lower bound</i> is subtracted to turn a global vocab index into a local row index inside GPU1's own 3-row shard. For GPU1 the lower bound happens to be 0, so this subtraction is a no-op here — GPU2 will show why it matters in a moment.",
    diagram: vpe([row("local (GPU1)", [cell(1, "gpu1"), cell(0, "gpu1", true), cell(2, "gpu1"), cell(0, "gpu1", true)])], { ranges: RANGES }),
  },
  {
    id: 5,
    title: "GPU1 — gather from its own shard",
    notation: "$gathered_1 = Embeddings_1[local_1] = [10,20,30][1,0,2,0] = [20,10,30,10]$",
    body:
      "GPU1 looks up values from its own 3-row shard $[10,20,30]$ using the local indices. Positions 1 and 3 come back as $10$ — a real value, but a <i>bogus</i> one, since those two tokens don't actually belong to GPU1 (their local index of 0 was just a placeholder). The next step cleans that up.",
    diagram: vpe([row("gathered (GPU1)", [cell(20, "gpu1"), cell(10, "gpu1", true), cell(30, "gpu1"), cell(10, "gpu1", true)])], { ranges: RANGES }),
  },
  {
    id: 6,
    title: "GPU1 — mask again to zero the bogus entries",
    notation: "$masked_1 = gathered_1 \\cdot mask_1 = [20,10,30,10] \\cdot [T,F,T,F] = [20,0,30,0]$",
    body:
      "Multiplying by the <i>same</i> mask a second time zeroes out the bogus lookups. What's left is GPU1's correct partial contribution: nonzero exactly where it owns a token, zero everywhere else. This is what GPU1 hands to the AllReduce.",
    diagram: vpe([row("masked (GPU1)", [cell(20, "gpu1"), cell(0, "gpu1", true), cell(30, "gpu1"), cell(0, "gpu1", true)])], { ranges: RANGES }),
  },
  {
    id: 7,
    title: "GPU2 mirrors the same steps",
    notation:
      "$mask_2 = (3 \\le X \\le 5) = (3 \\le [1,4,2,5] \\le 5) = [F,T,F,T]$<br/>" +
      "$X_{GPU2} = mask_2 \\cdot X = [F,T,F,T] \\cdot [1,4,2,5] = [0,4,0,5]$<br/>" +
      "$local_2 = [\\max(0{-}3,0),\\max(4{-}3,0),\\max(0{-}3,0),\\max(5{-}3,0)] = [0,1,0,2]$<br/>" +
      "$gathered_2 = Embeddings_2[local_2] = [40,50,60][0,1,0,2] = [40,50,40,60]$<br/>" +
      "$masked_2 = gathered_2 \\cdot mask_2 = [40,50,40,60] \\cdot [F,T,F,T] = [0,50,0,60]$",
    body:
      "GPU2 runs the identical procedure against its own range (3-5) and its own shard $[40,50,60]$. This time the lower-bound subtraction actually does something: raw indices 4 and 5 shift down to local rows 1 and 2 before gathering — unlike GPU1, where the lower bound of 0 made that step a no-op. The result is GPU2's masked partial — again zero everywhere it doesn't own a token.",
    diagram: vpe(
      [
        row("masked (GPU1)", [cell(20, "gpu1"), cell(0, "gpu1", true), cell(30, "gpu1"), cell(0, "gpu1", true)]),
        row("mask (GPU2)", [cell("F", "gpu2", true), cell("T", "gpu2"), cell("F", "gpu2", true), cell("T", "gpu2")]),
        row("local (GPU2)", [cell(0, "gpu2", true), cell(1, "gpu2"), cell(0, "gpu2", true), cell(2, "gpu2")]),
        row("gathered (GPU2)", [cell(40, "gpu2", true), cell(50, "gpu2"), cell(40, "gpu2", true), cell(60, "gpu2")]),
        row("masked (GPU2)", [cell(0, "gpu2", true), cell(50, "gpu2"), cell(0, "gpu2", true), cell(60, "gpu2")]),
      ],
      { ranges: RANGES }
    ),
  },
  {
    id: 8,
    title: "AllReduce combines the two masked partials",
    notation: "$Emb(X) = \\text{AllReduce}([20,0,30,0],\\,[0,50,0,60]) = [20,50,30,60]$",
    comms:
      "in the real (non-toy) setting each looked-up row is a $D$-wide embedding vector, not a scalar: $Emb(X) \\in [B,D]$, so this AllReduce costs $4BD$ bytes — same $\\approx\\!2\\times$-array-size rule as any other AllReduce",
    body:
      "Because the two masked partials are zero exactly where the other is nonzero, summing them elementwise reconstructs the correct lookup for every token — matching the target $Emb(X) = [20,50,30,60]$ that the setup step posed as the goal (for $X=[1,4,2,5]$).",
    note:
      "Unlike the MLP and attention blocks (which need an AllGather <i>and</i> a ReduceScatter on the activation), a vocab-parallel embedding needs just one AllReduce, on the output of the lookup — and its backward pass is free: every table row belongs to exactly one GPU, so the gradient update is a local scatter-add with no communication at all. There's no FLOPs line for this topic at all: a lookup is a gather + a mask, not a matmul — so $T_{\\text{math}}\\approx 0$ and this is comms-bound by construction; the interesting cost here is entirely $T_{\\text{comms}} = \\dfrac{4BD}{W_{ici}}$.",
    diagram: vpe(
      [
        row("masked (GPU1)", [cell(20, "gpu1"), cell(0, "gpu1", true), cell(30, "gpu1"), cell(0, "gpu1", true)]),
        row("masked (GPU2)", [cell(0, "gpu2", true), cell(50, "gpu2"), cell(0, "gpu2", true), cell(60, "gpu2")]),
        row("Emb(X)", [cell(20, "gpu1"), cell(50, "gpu2"), cell(30, "gpu1"), cell(60, "gpu2")]),
      ],
      { ranges: RANGES, comm: { type: "allreduce", label: "$Emb(X) = \\text{AllReduce}([20,0,30,0], [0,50,0,60]) = [20,50,30,60]$" } }
    ),
  },
];
