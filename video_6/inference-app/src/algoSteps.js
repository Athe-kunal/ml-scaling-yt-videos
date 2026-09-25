// Full sharded-attention decode algorithm from the scaling-book Inference
// chapter ("Here's the full algorithm!"): model parallelism over both Y and Z,
// M = N/K. One step per pseudocode line.
//
// `code` uses a tiny markup (see AlgorithmSteps.jsx): X_YZ -> subscript,
// _{..} -> subscript, **..** -> bold. `row` is the shapes-&-sharding strip:
// each tensor is "B_Z,S,K_Y,H" (dim, optional shard axes after "_"), ops sit
// between tensors. body/note/perChip use $...$ KaTeX spans.

const T = (name, dims) => ({ t: name, d: dims });
const OP = (o) => ({ op: o });

export const HEADER = {
  intro:
    "Full attention algorithm with model parallelism over both $Y$ and $Z$. $K$ is used for both the key tensor and the KV head dimension. Let $M = N/K$.",
  outro:
    "The new comms are modestly expensive since they operate on our small activations, while in return we save a huge amount of memory bandwidth loading the KVs (which are stationary).",
};

export const STEPS = [
  {
    code: "X[B, D] = … (existing activations, unsharded from previous layer)",
    kind: "load",
    title: "Activations arrive unsharded",
    row: [T("X", "B,D")],
    body: "The residual stream $X[B,D]$ is fully replicated: every chip in the $Y\\times Z$ mesh holds the whole (small) decode activation, one token per sequence.",
    perChip: "$B\\cdot D$ elements: tiny next to weights or KV cache",
  },
  {
    code: "K[B_Z, S, K_Y, H], V[B_Z, S, K_Y, H] = … (existing KV cache, batch sharded)",
    kind: "load",
    title: "KV cache is sharded, and it stays put",
    row: [T("K", "B_Z,S,K_Y,H"), T("V", "B_Z,S,K_Y,H")],
    body: "The cache is split over both axes: batch over $Z$, KV heads over $Y$. This is the whole trick. The big tensor never moves; we will move the small activations to it instead.",
    perChip: "each chip stores $\\frac{B\\,S\\,K\\,H}{YZ}$ of K (and the same of V)",
    note: "Sharding the cache this way divides the KV bytes each chip loads by $YZ$, which pushes $B^*$ up, the same lever as GQA in the roofline page.",
  },
  {
    code: "Q[B, N_YZ, H] = X[B, D] * W_Q[D, N_YZ, H]",
    kind: "compute",
    title: "Project to Q: heads sharded over Y and Z",
    row: [T("Q", "B,N_YZ,H"), OP("="), T("X", "B,D"), OP("×"), T("W_Q", "D,N_YZ,H")],
    body: "$W_Q$ is sharded over heads across both axes, so each chip computes Q for its $N/(YZ)$ heads for the <b>entire</b> batch. No comms: $X$ was already replicated.",
    perChip: "weights loaded per chip: $\\frac{D\\,N\\,H}{YZ}$",
  },
  {
    code: "Q[B_Z, N_Y, H] = **AllToAll**_{Z→B}(Q[B, N_YZ, H])",
    kind: "comm",
    comm: "AllToAll  Z: heads → batch",
    title: "AllToAll: line Q up with the KV cache",
    row: [T("Q", "B_Z,N_Y,H"), OP("←"), T("Q", "B,N_YZ,H")],
    body: "K and V are sharded over batch on $Z$, but Q is sharded over heads on $Z$. An AllToAll trades the $Z$ shard from the $N$ axis to the $B$ axis, so each chip now holds Q for its slice of the batch, next to that batch slice's KV cache.",
    perChip: "moves activations only, $O(B\\,N\\,H/YZ)$ per chip",
    note: "This is the first of the two new collectives. It runs on Q, one token per sequence, not on the KV cache.",
  },
  {
    code: "Q[B_Z, K_Y, M, H] = **Reshape**(Q[B_Z, N_Y, H])",
    kind: "reshape",
    title: "Reshape N into (K, M)",
    row: [T("Q", "B_Z,K_Y,M,H"), OP("←"), T("Q", "B_Z,N_Y,H")],
    body: "With $N = K\\cdot M$, split the head axis into KV heads $K$ and the group size $M$ of query heads per KV head. The $Y$ shard rides along on $K$. Purely local, no comms.",
  },
  {
    code: "O[B_Z, S, K_Y, M] = Q[B_Z, K_Y, M, H] *_H K[B_Z, S, K_Y, H]",
    kind: "compute",
    title: "Scores: contract Q with K over H",
    row: [T("O", "B_Z,S,K_Y,M"), OP("="), T("Q", "B_Z,K_Y,M,H"), OP("×ᴴ"), T("K", "B_Z,S,K_Y,H")],
    body: "Every batch/KV-head block of Q meets exactly the KV cache chunk on the same chip. This is where the KV cache is read from HBM, the bandwidth-bound part of decode, and each chip only reads its own $1/(YZ)$ share.",
    perChip: "KV bytes read per chip: $\\frac{B\\,S\\,K\\,H}{YZ}$",
    note: "No comms: all three tensors are already aligned on $B_Z$ and $K_Y$.",
  },
  {
    code: "O[B_Z, S, K_Y, M] = **Softmax**_S(O[B_Z, S, K_Y, M])",
    kind: "compute",
    title: "Softmax over the sequence axis",
    row: [T("O", "B_Z,S,K_Y,M"), OP("←"), T("O", "B_Z,S,K_Y,M")],
    body: "$S$ is not sharded anywhere, so the softmax over $S$ is entirely local.",
  },
  {
    code: "O[B_Z, K_Y, M, H] = O[B_Z, S, K_Y, M] *_S V[B_Z, S, K_Y, H]",
    kind: "compute",
    title: "Weighted sum: contract with V over S",
    row: [T("O", "B_Z,K_Y,M,H"), OP("="), T("O", "B_Z,S,K_Y,M"), OP("×ˢ"), T("V", "B_Z,S,K_Y,H")],
    body: "The second and last read of the KV cache: $V$, again the local shard. Contracting over $S$ leaves the per-head output.",
    perChip: "V bytes read per chip: $\\frac{B\\,S\\,K\\,H}{YZ}$",
  },
  {
    code: "O[B, K_Y, M_Z, H] = **AllToAll**_{Z→M}(O[B_Z, K_Y, M, H])",
    kind: "comm",
    comm: "AllToAll  Z: batch → M",
    title: "AllToAll: undo the batch sharding",
    row: [T("O", "B,K_Y,M_Z,H"), OP("←"), T("O", "B_Z,K_Y,M,H")],
    body: "Attention is done, so we no longer need to be aligned with the cache. Swap the $Z$ shard back off the batch and onto the query-group axis $M$, giving every chip the full batch again.",
    note: "The second new collective, again on small activations.",
  },
  {
    code: "O[B, N_YZ, H] = **Reshape**(O[B, K_Y, M_Z, H])",
    kind: "reshape",
    title: "Reshape (K, M) back into N",
    row: [T("O", "B,N_YZ,H"), OP("←"), T("O", "B,K_Y,M_Z,H")],
    body: "Merge $K$ and $M$ back into the head axis $N$, now sharded over $Y$ and $Z$ together, matching how $W_O$ is sharded. Local.",
  },
  {
    code: "X[B, D] {U_YZ} = W_O[N_YZ, H, D] *_{N,H} O[B, N_YZ, H]",
    kind: "compute",
    title: "Output projection: partial sums",
    row: [T("X", "B,D {U_YZ}"), OP("="), T("W_O", "N_YZ,H,D"), OP("×ᴺᴴ"), T("O", "B,N_YZ,H")],
    body: "Each chip contracts over only its heads, so it holds an <b>unreduced partial sum</b> of $X$, written $\\{U_{YZ}\\}$: the true value is the sum across all $YZ$ chips.",
    perChip: "weights loaded per chip: $\\frac{N\\,H\\,D}{YZ}$",
  },
  {
    code: "X[B, D] = **AllReduce**(X[B, D] {U_YZ})",
    kind: "comm",
    comm: "AllReduce  over Y, Z",
    title: "AllReduce: back to replicated X",
    row: [T("X", "B,D"), OP("←"), T("X", "B,D {U_YZ}")],
    body: "Sum the partials over all $YZ$ chips so every chip again holds the full $X[B,D]$, ready for the next layer.",
    note: "This is pretty complicated, but you can see generally how it works. The new comms (two AllToAlls and the AllReduce) operate on our small activations, while in return we save a huge amount of memory bandwidth loading the KVs, which are stationary.",
  },
];
