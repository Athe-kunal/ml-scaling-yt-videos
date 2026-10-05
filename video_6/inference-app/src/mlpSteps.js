// MLP weight-stationary inference layouts (1D vs 2D) from the scaling-book
// Inference chapter's "Here's the 1D weight-stationary layout" / "2D
// weight-stationary layout" figure. Three ICI axes x, y, z. 1D merges all
// three into a single sharding axis for both weights and the resident
// activation; 2D splits the mesh into x (shards the weight's E dimension)
// vs y,z together (shards F), halving each collective's size at the cost
// of needing two AllGather/ReduceScatter pairs instead of one.
//
// `diagram.nodes` drives MLPFlowDiagram (one row of boxes: Act -> Win ->
// Tmp -> Wout -> Out -> Final, cycling hidden/ghost/solid/active/partial).
// `matrices` drives MatrixShapes (reused from the sharded-attention view):
// each tensor is "Name,dim_shard,dim_shard {U_axis}" — see notation.js.

const T_ = (name, dims) => ({ t: name, d: dims });
const OP = (o) => ({ op: o });

function node(label, state) {
  return { label, state };
}
const HIDDEN = () => node("", "hidden");

function snapshot(overrides) {
  const base = { Act: HIDDEN(), Win: HIDDEN(), Tmp: HIDDEN(), Wout: HIDDEN(), Out: HIDDEN(), Final: HIDDEN() };
  return { ...base, ...overrides };
}
function diagram(nodes, comm) {
  return { nodes: snapshot(nodes), comm: comm || null };
}

export const HEADER_1D = {
  title: "1D weight-stationary",
  intro:
    "All three ICI axes $x,y,z$ merge into one mesh for sharding both $Win$/$Wout$ and the resident activation. One AllGather opens the block, one ReduceScatter closes it — each moves the full activation across the whole mesh.",
};

export const STEPS_1D = [
  {
    id: 1,
    title: "Setup — weights and activations sharded over the whole mesh",
    notation: "$Act[B,L,E_{xyz}]$ — $Win[E,F_{xyz}]$ (column-sharded) — $Wout[F_{xyz},E]$ (row-sharded)",
    body:
      "$Win$ is column-sharded on $F$ across all three axes merged into one; $Wout$ is row-sharded on $F$ the same way. The activation rests between blocks sharded along $E$, not fully replicated — cheaper to hold, but it must be gathered before this block's first matmul can start.",
    diagram: diagram({
      Act: node("Act[B,L,E_{xyz}]", "ghost"),
      Win: node("Win[E,F_{xyz}]", "solid"),
      Wout: node("Wout[F_{xyz},E]", "solid"),
    }),
    matrices: [T_("Act", "B,L,E_xyz"), T_("Win", "E,F_xyz"), T_("Wout", "F_xyz,E")],
  },
  {
    id: 2,
    title: "AllGather(xyz) — gather the activation",
    notation: "$Act[B,L,E] = \\text{AllGather}_{xyz}\\!\\left(Act[B,L,E_{xyz}]\\right)$",
    comms: "$\\approx BLE$ elements moved — an AllGather costs $\\approx 1\\times$ the gathered array's bytes",
    body:
      "On the critical path: the block's first matmul can't start until every chip holds the complete $E$ dimension. After this, $Act$ is fully replicated across the whole mesh.",
    diagram: diagram(
      { Act: node("Act[B,L,E]", "active"), Win: node("Win[E,F_{xyz}]", "solid"), Wout: node("Wout[F_{xyz},E]", "solid") },
      { type: "allgather", targetId: "Act", label: "$Act[B,L,E] = \\text{AllGather}_{xyz}(Act[B,L,E_{xyz}])$" }
    ),
    matrices: [T_("Act", "B,L,E_xyz"), OP("→"), T_("Act", "B,L,E")],
  },
  {
    id: 3,
    title: "Matmul 1 — local, no comms",
    notation: "$Tmp[B,L,F_{xyz}] = Act[B,L,E] \\cdot_E Win[E,F_{xyz}]$",
    body:
      "The contracting dimension $E$ isn't sharded, so this is a fully local matmul — each chip just computes its own $F_{xyz}$ slice of the hidden activation.",
    diagram: diagram({
      Act: node("Act[B,L,E]", "solid"),
      Win: node("Win[E,F_{xyz}]", "solid"),
      Tmp: node("Tmp[B,L,F_{xyz}]", "active"),
      Wout: node("Wout[F_{xyz},E]", "solid"),
    }),
    matrices: [T_("Act", "B,L,E"), OP("×"), T_("Win", "E,F_xyz"), OP("="), T_("Tmp", "B,L,F_xyz")],
  },
  {
    id: 4,
    title: "gelu — elementwise, local",
    notation: "$Tmp[B,L,F_{xyz}] = \\text{gelu}\\!\\left(Tmp[B,L,F_{xyz}]\\right)$",
    body: "Elementwise, so purely local — no communication, no shard change.",
    diagram: diagram({
      Act: node("Act[B,L,E]", "solid"),
      Win: node("Win[E,F_{xyz}]", "solid"),
      Tmp: node("Tmp[B,L,F_{xyz}]", "active"),
      Wout: node("Wout[F_{xyz},E]", "solid"),
    }),
    matrices: [T_("Tmp", "B,L,F_xyz")],
  },
  {
    id: 5,
    title: "Matmul 2 — unreduced over xyz",
    notation: "$Out[B,L,E]^{\\{U_{xyz}\\}} = Tmp[B,L,F_{xyz}] \\cdot_F Wout[F_{xyz},E]$",
    body:
      "Now $F$ is the contracting dimension and it <i>is</i> sharded across all three axes, so each chip only computes a partial contribution, tagged $\\{U_{xyz}\\}$: the right shape, but the wrong (incomplete) value until every chip's partial is summed.",
    diagram: diagram({
      Tmp: node("Tmp[B,L,F_{xyz}]", "solid"),
      Wout: node("Wout[F_{xyz},E]", "solid"),
      Win: node("Win[E,F_{xyz}]", "solid"),
      Out: node("Out[B,L,E] {U_xyz}", "partial"),
    }),
    matrices: [T_("Tmp", "B,L,F_xyz"), OP("×"), T_("Wout", "F_xyz,E"), OP("="), T_("Out", "B,L,E {U_xyz}")],
  },
  {
    id: 6,
    title: "ReduceScatter(xyz) — sum and re-shard the output",
    notation: "$Out[B,L,E_{xyz}] = \\text{ReduceScatter}_{xyz}\\!\\left(Out[B,L,E]^{\\{U_{xyz}\\}}\\right)$",
    comms: "$\\approx BLE$ elements moved — a ReduceScatter costs $\\approx 1\\times$ the pre-scatter array's bytes",
    body:
      "A single collective both sums the partial output across all three axes and re-shards it along $E$ — the block ends exactly like it started: sharded, never fully materialized on one chip.",
    diagram: diagram(
      { Tmp: node("Tmp[B,L,F_{xyz}]", "solid"), Wout: node("Wout[F_{xyz},E]", "solid"), Win: node("Win[E,F_{xyz}]", "solid"), Out: node("Out[B,L,E_{xyz}]", "active") },
      { type: "reducescatter", targetId: "Out", label: "$Out[B,L,E_{xyz}] = \\text{ReduceScatter}_{xyz}(Out[B,L,E]^{\\{U_{xyz}\\}})$" }
    ),
    matrices: [T_("Out", "B,L,E {U_xyz}"), OP("→"), T_("Out", "B,L,E_xyz")],
  },
  {
    id: 7,
    title: "Residual add — local",
    notation: "$Act'[B,L,E_{xyz}] = Act[B,L,E_{xyz}] + Out[B,L,E_{xyz}]$",
    body:
      "Both operands are sharded the same way, so the residual add is local — the block hands the next layer the same sharded resting state it started in. Total new comms for this block: one AllGather + one ReduceScatter, both $\\approx BLE$ elements, on activations rather than weights.",
    diagram: diagram({
      Act: node("Act[B,L,E_{xyz}]", "solid"),
      Win: node("Win[E,F_{xyz}]", "solid"),
      Wout: node("Wout[F_{xyz},E]", "solid"),
      Out: node("Out[B,L,E_{xyz}]", "solid"),
      Final: node("Act'[B,L,E_{xyz}]", "active"),
    }),
    matrices: [T_("Act", "B,L,E_xyz"), OP("+"), T_("Out", "B,L,E_xyz"), OP("="), T_("Final", "B,L,E_xyz")],
  },
];

export const HEADER_2D = {
  title: "2D weight-stationary",
  intro:
    "The mesh splits into two logical groups: axis $x$ shards the weight's input dimension $E$, axes $y,z$ together shard the hidden dimension $F$. Each collective now only spans part of the mesh — half the elements per AllGather/ReduceScatter compared to 1D — at the cost of needing two pairs of them instead of one.",
};

export const STEPS_2D = [
  {
    id: 1,
    title: "Setup — mesh split: x vs (y, z)",
    notation: "$Act[B,L,E_{xyz}]$ — $Win[E_x,F_{yz}]$ — $Wout[F_{yz},E_x]$",
    body:
      "$Win$'s input dimension $E$ is sharded only on $x$; its output dimension $F$ is sharded on $y,z$. $Wout$ mirrors that. The activation still rests sharded over the full $xyz$ mesh between blocks, same as 1D.",
    diagram: diagram({
      Act: node("Act[B,L,E_{xyz}]", "ghost"),
      Win: node("Win[E_x,F_{yz}]", "solid"),
      Wout: node("Wout[F_{yz},E_x]", "solid"),
    }),
    matrices: [T_("Act", "B,L,E_xyz"), T_("Win", "E_x,F_yz"), T_("Wout", "F_yz,E_x")],
  },
  {
    id: 2,
    title: "AllGather(yz) — gather only the y,z shard of E",
    notation: "$Act[B,L,E_x] = \\text{AllGather}_{yz}\\!\\left(Act[B,L,E_{xyz}]\\right)$",
    comms: "$\\approx BLE/x$ elements moved — only the $y,z$ shard of $E$ needs gathering; the $x$ shard stays split",
    body:
      "The $x$ shard of $E$ stays split the whole time, since $Win$'s input dimension is itself sharded on $x$. This AllGather spans only the $y,z$ sub-mesh, so it's cheaper than 1D's full-mesh gather.",
    diagram: diagram(
      { Act: node("Act[B,L,E_x]", "active"), Win: node("Win[E_x,F_{yz}]", "solid"), Wout: node("Wout[F_{yz},E_x]", "solid") },
      { type: "allgather", targetId: "Act", label: "$Act[B,L,E_x] = \\text{AllGather}_{yz}(Act[B,L,E_{xyz}])$" }
    ),
    matrices: [T_("Act", "B,L,E_xyz"), OP("→"), T_("Act", "B,L,E_x")],
  },
  {
    id: 3,
    title: "Matmul 1 — unreduced over x",
    notation: "$Tmp[B,L,F_{yz}]^{\\{U_x\\}} = Act[B,L,E_x] \\cdot_E Win[E_x,F_{yz}]$",
    body:
      "$E$ is still sharded, on $x$, so each chip only contracts its own slice — the result is a partial sum tagged $\\{U_x\\}$, not yet the true value.",
    diagram: diagram({
      Act: node("Act[B,L,E_x]", "solid"),
      Win: node("Win[E_x,F_{yz}]", "solid"),
      Tmp: node("Tmp[B,L,F_yz] {U_x}", "partial"),
      Wout: node("Wout[F_{yz},E_x]", "solid"),
    }),
    matrices: [T_("Act", "B,L,E_x"), OP("×"), T_("Win", "E_x,F_yz"), OP("="), T_("Tmp", "B,L,F_yz {U_x}")],
  },
  {
    id: 4,
    title: "ReduceScatter(x) — sum and re-shard onto x",
    notation: "$Tmp[B,L,F_{xyz}] = \\text{ReduceScatter}_x\\!\\left(Tmp[B,L,F_{yz}]^{\\{U_x\\}}\\right)$",
    comms: "$\\approx BLF/(yz)$ elements moved",
    body:
      "Sums the partial over $x$ while simultaneously re-sharding onto $x$ — $Tmp$ ends up fully sharded over all three axes, ready for an elementwise op.",
    diagram: diagram(
      { Act: node("Act[B,L,E_x]", "solid"), Win: node("Win[E_x,F_{yz}]", "solid"), Wout: node("Wout[F_{yz},E_x]", "solid"), Tmp: node("Tmp[B,L,F_{xyz}]", "active") },
      { type: "reducescatter", targetId: "Tmp", label: "$Tmp[B,L,F_{xyz}] = \\text{ReduceScatter}_x(Tmp[B,L,F_{yz}]^{\\{U_x\\}})$" }
    ),
    matrices: [T_("Tmp", "B,L,F_yz {U_x}"), OP("→"), T_("Tmp", "B,L,F_xyz")],
  },
  {
    id: 5,
    title: "gelu — elementwise, local",
    notation: "$Tmp[B,L,F_{xyz}] = \\text{gelu}\\!\\left(Tmp[B,L,F_{xyz}]\\right)$",
    body: "Elementwise and fully local, exactly like the 1D layout — gelu never needs communication regardless of how the hidden dim is sharded.",
    diagram: diagram({
      Act: node("Act[B,L,E_x]", "solid"),
      Win: node("Win[E_x,F_{yz}]", "solid"),
      Wout: node("Wout[F_{yz},E_x]", "solid"),
      Tmp: node("Tmp[B,L,F_{xyz}]", "active"),
    }),
    matrices: [T_("Tmp", "B,L,F_xyz")],
  },
  {
    id: 6,
    title: "AllGather(x) — regather F onto y,z",
    notation: "$Tmp[B,L,F_{yz}] = \\text{AllGather}_x\\!\\left(Tmp[B,L,F_{xyz}]\\right)$",
    comms: "$\\approx BLF/(yz)$ elements moved",
    body: "Before the second matmul, $Tmp$ needs to be sharded to match $Wout$'s contracting layout ($F_{yz}$), so the $x$ shard gets gathered back off.",
    diagram: diagram(
      { Act: node("Act[B,L,E_x]", "solid"), Win: node("Win[E_x,F_{yz}]", "solid"), Wout: node("Wout[F_{yz},E_x]", "solid"), Tmp: node("Tmp[B,L,F_{yz}]", "active") },
      { type: "allgather", targetId: "Tmp", label: "$Tmp[B,L,F_{yz}] = \\text{AllGather}_x(Tmp[B,L,F_{xyz}])$" }
    ),
    matrices: [T_("Tmp", "B,L,F_xyz"), OP("→"), T_("Tmp", "B,L,F_yz")],
  },
  {
    id: 7,
    title: "Matmul 2 — unreduced over y,z",
    notation: "$Out[B,L,E_x]^{\\{U_{yz}\\}} = Tmp[B,L,F_{yz}] \\cdot_F Wout[F_{yz},E_x]$",
    body:
      "$F$ is the contracting dimension and it's sharded on $y,z$, so each chip again only holds a partial contribution — tagged $\\{U_{yz}\\}$ — while $E$ inherits $Wout$'s $x$-sharding directly, no extra comms needed for that part.",
    diagram: diagram({
      Tmp: node("Tmp[B,L,F_{yz}]", "solid"),
      Wout: node("Wout[F_{yz},E_x]", "solid"),
      Win: node("Win[E_x,F_{yz}]", "solid"),
      Out: node("Out[B,L,E_x] {U_yz}", "partial"),
    }),
    matrices: [T_("Tmp", "B,L,F_yz"), OP("×"), T_("Wout", "F_yz,E_x"), OP("="), T_("Out", "B,L,E_x {U_yz}")],
  },
  {
    id: 8,
    title: "ReduceScatter(yz) — sum and re-shard the output",
    notation: "$Out[B,L,E_{xyz}] = \\text{ReduceScatter}_{yz}\\!\\left(Out[B,L,E_x]^{\\{U_{yz}\\}}\\right)$",
    comms: "$\\approx BLE/x$ elements moved",
    body:
      "Sums the partial across $y,z$ while re-sharding onto them, landing back at the fully 3-axis-sharded resting state — mirroring the AllGather that opened this block.",
    diagram: diagram(
      { Tmp: node("Tmp[B,L,F_{yz}]", "solid"), Wout: node("Wout[F_{yz},E_x]", "solid"), Win: node("Win[E_x,F_{yz}]", "solid"), Out: node("Out[B,L,E_{xyz}]", "active") },
      { type: "reducescatter", targetId: "Out", label: "$Out[B,L,E_{xyz}] = \\text{ReduceScatter}_{yz}(Out[B,L,E_x]^{\\{U_{yz}\\}})$" }
    ),
    matrices: [T_("Out", "B,L,E_x {U_yz}"), OP("→"), T_("Out", "B,L,E_xyz")],
  },
  {
    id: 9,
    title: "Residual add — local",
    notation: "$Act'[B,L,E_{xyz}] = Act[B,L,E_{xyz}] + Out[B,L,E_{xyz}]$",
    body:
      "Same shape as 1D's residual step, but 2D got here using four half-sized collectives (two gathers, two scatters) instead of 1D's two full-sized ones — the same total bytes moved overall, but each individual collective is cheaper and easier to overlap with compute.",
    diagram: diagram({
      Act: node("Act[B,L,E_{xyz}]", "solid"),
      Win: node("Win[E_x,F_{yz}]", "solid"),
      Wout: node("Wout[F_{yz},E_x]", "solid"),
      Out: node("Out[B,L,E_{xyz}]", "solid"),
      Final: node("Act'[B,L,E_{xyz}]", "active"),
    }),
    matrices: [T_("Act", "B,L,E_xyz"), OP("+"), T_("Out", "B,L,E_xyz"), OP("="), T_("Final", "B,L,E_xyz")],
  },
];
