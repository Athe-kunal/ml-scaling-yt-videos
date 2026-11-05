import { useState } from "react";
import { T, mono, btn } from "./theme";
import { XC } from "./lbModel";
import Stepper from "./Stepper";
import { Tex } from "./tex";
import { useProgress, ease, lerp, clamp01 } from "./anim";

// EPLB (Expert Parallelism Load Balancer). It never touches routing: given measured
// load per logical expert it decides WHERE experts live and how many copies each gets.
// Everything below is computed by a JS port of the algorithm, so every number is real.

// ---------------------------------------------------------------- algorithm port
const sum = (a) => a.reduce((x, y) => x + y, 0);

// rows of weights -> per row: pack[i] (which pack item i goes to), rank[i] (slot inside the pack)
function balancedPacking(rows, m) {
  return rows.map((w) => {
    const n = w.length, cap = n / m;
    const pack = Array(n).fill(-1), rank = Array(n).fill(-1);
    if (cap === 1) {
      w.forEach((_, i) => { pack[i] = i; rank[i] = 0; });
      return { pack, rank, order: [...w.keys()], events: [] };
    }
    const order = [...w.keys()].sort((a, b) => w[b] - w[a] || a - b);
    const pw = Array(m).fill(0), pn = Array(m).fill(0), events = [];
    for (const g of order) {
      let best = -1;
      for (let q = 0; q < m; q++) if (pn[q] < cap && (best < 0 || pw[q] < pw[best])) best = q;
      const ev = { item: g, pack: best, rank: pn[best], pwBefore: [...pw], pnBefore: [...pn] };
      pack[g] = best; rank[g] = pn[best];
      pw[best] += w[g]; pn[best] += 1;
      ev.pwAfter = [...pw]; ev.pnAfter = [...pn];
      events.push(ev);
    }
    return { pack, rank, order, events };
  });
}

// rows of weights -> per row: phy2log, rank, cnt, and one event per extra slot
function replicate(rows, numPhy) {
  return rows.map((w) => {
    const numLog = w.length;
    const phy2log = Array.from({ length: numPhy }, (_, i) => (i < numLog ? i : 0));
    const rank = Array(numPhy).fill(0), cnt = Array(numLog).fill(1), events = [];
    for (let i = numLog; i < numPhy; i++) {
      const cand = w.map((x, j) => x / cnt[j]);
      let pick = 0;
      cand.forEach((v, j) => { if (v > cand[pick]) pick = j; });
      events.push({ cand, pick, cntBefore: [...cnt], slot: i });
      phy2log[i] = pick; rank[i] = cnt[pick]; cnt[pick] += 1;
    }
    return { phy2log, rank, cnt: [...cnt], events };
  });
}

function hierarchical(weight, P, G, N, Q) {
  const E = weight.length, gs = E / G, gpn = G / N, ppg = P / Q, perNode = E / N;
  const tpg = Array.from({ length: G }, (_, g) => sum(weight.slice(g * gs, (g + 1) * gs)));
  const [gp] = balancedPacking([tpg], N);
  const log2mlog = Array(E);
  for (let g = 0; g < G; g++) for (let j = 0; j < gs; j++) log2mlog[g * gs + j] = (gp.pack[g] * gpn + gp.rank[g]) * gs + j;
  const mlog2log = Array(E);
  log2mlog.forEach((m, l) => { mlog2log[m] = l; });
  const tpm = Array.from({ length: N }, (_, n) => Array.from({ length: perNode }, (_, j) => weight[mlog2log[n * perNode + j]]));
  const rep = replicate(tpm, P / N);
  const tpp = rep.map((r, n) => r.phy2log.map((m) => tpm[n][m] / r.cnt[m]));
  const gpk = balancedPacking(tpp, Q / N);
  const phy2log = [], phyrank = [];
  for (let n = 0; n < N; n++) {
    const phy2pphy = gpk[n].pack.map((pk, i) => pk * ppg + gpk[n].rank[i]);
    const pphy2phy = Array(P / N);
    phy2pphy.forEach((s, i) => { pphy2phy[s] = i; });
    for (let s = 0; s < P / N; s++) {
      const ph = pphy2phy[s];
      phy2log.push(mlog2log[rep[n].phy2log[ph] + n * perNode]);
      phyrank.push(rep[n].rank[ph]);
    }
  }
  const mlogcnt = rep.flatMap((r) => r.cnt);
  const logcnt = log2mlog.map((m) => mlogcnt[m]);
  return { tpg, gp, log2mlog, mlog2log, tpm, rep, tpp, gpk, phy2log, phyrank, logcnt };
}

// ---------------------------------------------------------------- the toy instance
const WL = [90, 20, 15, 10, 60, 12, 8, 40]; // tokens per logical expert
const E = 8, G = 4, N = 2, Q = 4, P = 12, PPG = P / Q, PERN = E / N;
const R = hierarchical(WL, P, G, N, Q); // num_groups % num_nodes == 0
const MEAN = sum(WL) / Q;
const NAIVE = Array.from({ length: Q }, (_, g) => WL[2 * g] + WL[2 * g + 1]); // experts in order, 2 per GPU, no copies
const AFTER = Array.from({ length: Q }, (_, g) => sum(R.phy2log.slice(g * PPG, (g + 1) * PPG).map((l) => WL[l] / R.logcnt[l])));
const LOG2PHY = Array.from({ length: E }, () => []);
R.phy2log.forEach((l, s) => { LOG2PHY[l][R.phyrank[s]] = s; });
const GC = ["#E6B450", "#59C2FF", "#D2A6FF", "#AAD94C"]; // one colour per expert group
const fm = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const f2 = (v) => v.toFixed(2);
const col = (c, s) => `{\\color{${c}}\\mathbf{${s}}}`;
const W = 860, H = 430;

// pack-scene rows: where the items queue up, where the packs stand, and the real greedy trace
const NODE_ROW = {
  title: "GROUP LOADS → NODES", x0: 40, itemW: 64, gap: 18, packX: 480, packW: 110, packGap: 40, baseY: 300, scale: 1.7, m: N, cap: G / N,
  items: R.tpg.map((w, g) => ({ w, label: `g${g}`, color: GC[g], has: [2 * g, 2 * g + 1] })),
  order: R.gp.order, events: R.gp.events, packName: (q) => `node ${q}`, wname: (q) => `n${q}`, texY: 322,
};
const GPU_ROWS = [0, 1].map((n) => ({
  title: `NODE ${n} · COPIES → GPUs`, x0: 20 + n * 430, itemW: 28, gap: 6, packX: 20 + n * 430 + 235, packW: 70, packGap: 20, baseY: 290, scale: 2.2, m: Q / N, cap: PPG,
  items: R.tpp[n].map((w, j) => { const l = R.rep[n].phy2log[j] + n * PERN; const lg = R.mlog2log[l]; return { w, label: `E${lg}`, color: XC[lg], has: [lg] }; }),
  order: R.gpk[n].order, events: R.gpk[n].events, packName: (q) => `GPU ${n * 2 + q}`, wname: (q) => `g${n * 2 + q}`, texY: 318,
}));

// replicate state of node n after r rounds: 6 columns
function repCols(n, r) {
  const rep = R.rep[n], cnt = rep.events[r > 0 ? r - 1 : 0].cntBefore.slice();
  const c = r === 0 ? Array(PERN).fill(1) : (() => { const a = cnt.slice(); a[rep.events[r - 1].pick] += 1; return a; })();
  return Array.from({ length: P / N }, (_, j) => {
    const on = j < PERN + r, ml = rep.phy2log[j];
    return { on, ml, lg: R.mlog2log[n * PERN + ml], c: c[ml], v: R.tpm[n][ml] / c[ml] };
  });
}

// ---------------------------------------------------------------- code
const bp = (n, m, what) => [
  `order = weight.sort(-1, descending=True).indices   # [L,n] → item ids, heaviest first · n=${n} ${what}`,
  "for g in order[i]:   # i = row · g = next heaviest item",
  `    p = argmin(pack_w[q] for q in range(m) if pack_n[q] < n//m)   # lightest of m=${m} packs that still has room`,
  "    pack[i,g] = p; pack_w[p] += weight[i,g]; pack_n[p] += 1   # pack_w [m] running load · pack_n [m] item count",
];

const BLOCKS = [
  { id: "load", label: "measure expert load", sub: "weight from live traffic", code: ["weight = load_stats()   # [L,E] · weight[l,e] = tokens sent to logical expert e (L layers; one shown)"] },
  { id: "entry", label: "rebalance_experts", sub: "pick the policy", code: [
    "# E=8 logical experts · P=12 physical slots · G=4 groups · N=2 nodes · Q=4 GPUs",
    "if num_groups % num_nodes == 0:   # G % N = 0 → hierarchical: a group's experts stay on one node",
    "    return rebalance_experts_hierarchical(weight, P, G, N, Q)   # → phy2log [L,P], rank [L,P], logcnt [L,E]",
    "# else: same call with 1 group, 1 node (global policy)"] },
  {
    group: "rebalance_experts_hierarchical", items: [
      { id: "groups", label: "1 · groups → nodes", sub: "keep groups together", code: [
        "tokens_per_group = weight.unflatten(-1, (G, E//G)).sum(-1)   # [L,E] → [L,G,E/G] → sum over E/G → [L,G]",
        "pack, rank = balanced_packing(tokens_per_group, N)   # [L,G] → node of each group [L,G], slot in node [L,G]"] },
      { id: "bpn", label: "balanced_packing", sub: "greedy, heaviest first", loop: true, code: bp(4, 2, "groups") },
      { id: "rcall", label: "2 · replicate inside nodes", sub: "extra copies of hot experts", code: [
        "tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, E//N)   # [L,E] → regroup by node → [L·N, E/N]",
        "phy2mlog, rank, mlogcnt = replicate_experts(tokens_per_mlog, P//N)   # → phy2mlog [L·N, P/N], mlogcnt [L·N, E/N]"] },
      { id: "rloop", label: "replicate_experts", sub: "hottest per copy gets one more", loop: true, code: [
        "logcnt = ones(n, num_log)   # [n, E/N] · copies per expert, all start at 1",
        "for i in range(num_log, num_phy):   # one extra slot per round (P/N − E/N = 2 rounds)",
        "    e = (weight / logcnt).max(dim=-1).indices   # [n,E/N] ÷ [n,E/N] → max over E/N → [n] · hottest expert per copy",
        "    phy2log[:, i] = e; logcnt[:, e] += 1   # slot i = a copy of e · its load now splits one more way"] },
      { id: "gpus", label: "3 · copies → GPUs", sub: "balance inside each node", code: [
        "tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)   # [L·N,E/N] ÷ [L·N,E/N] → gather → [L·N,P/N] · load of one copy",
        "pack, rank = balanced_packing(tokens_per_phy, Q//N)   # [L·N,P/N] → GPU inside the node, slot on that GPU"] },
      { id: "bpg", label: "balanced_packing", sub: "same greedy, per node", loop: true, code: bp(6, 2, "copies per node") },
    ],
  },
  { id: "result", label: "placement table", sub: "who lives where", code: [
    "pphy2log = mlog2log.gather(-1, pphy2mlog)   # [L,P] · slot s (GPU-major) → logical expert it hosts",
    "logcnt = mlogcnt.view(L, -1).gather(-1, log2mlog)   # [L,E] · copies per logical expert",
    "log2phy[l, phy2log[l,s], rank[l,s]] = s   # [L,E,R] · slots holding each expert's copies, R = max copies"] },
  { id: "note", label: "next to V3 balancing", sub: "complementary", code: [
    "# V3 bias b:  changes WHICH experts a token picks (routing)",
    "# EPLB:       changes WHERE experts live and how many copies exist (placement)"] },
];

const nodeEv = R.gp.events, GPU_EV = R.gpk.map((g) => g.events);
const STEPS = [
  { id: "load", block: "load", lines: [0], scene: "load", caption: "a few experts get most of the tokens", vars: () => [["E", `${E}`], ["total", `${sum(WL)}`], ["hottest", `E0 = ${WL[0]}`, "bad"]] },
  { id: "topo", block: "entry", lines: [0, 1, 2], visAt: 0, scene: "topo", caption: "12 slots to fill: 2 nodes × 2 GPUs × 3 slots, 4 of them spare", vars: () => [["G % N", `${G} % ${N} = ${G % N}`], ["slots / GPU", `${PPG}`]] },
  { id: "gsum", block: "groups", lines: [0], scene: "gsum", caption: "add up each group's experts: a group is the unit moved between nodes", vars: () => [["tokens_per_group", `[${R.tpg.join(" ")}]`]] },
  { id: "gcall", block: "groups", lines: [1], scene: "pack", stage: "node", k: 0, caption: "now spread the 4 groups over 2 nodes, 2 per node", vars: () => [["items", `${G}`], ["packs", `${N}`], ["per pack", `${G / N}`]] },
  { id: "psort", block: "bpn", lines: [0], scene: "pack", stage: "node", k: 0, sortAnim: true, caption: "sort groups heaviest first", vars: () => [["order", R.gp.order.map((g) => `g${g}`).join(" ")]] },
  ...nodeEv.map((ev, i) => ({
    id: `pn${i + 1}`, block: "bpn", lines: [1, 2, 3], visAt: 1, scene: "pack", stage: "node", k: i + 1, fly: true,
    caption: `g${ev.item} (${fm(NODE_ROW.items[ev.item].w)}) goes to the lightest node with room: node ${ev.pack}`,
    vars: () => [["g", `g${ev.item}`], ["pack_w", `[${ev.pwAfter.map(fm).join(" ")}]`], ["pack_n", `[${ev.pnAfter.join(" ")}]`]],
  })),
  { id: "mlog", block: "rcall", lines: [0], scene: "rep", phase: "mlog", r: 0, caption: "regroup experts by node: each node now owns 4 logical experts", vars: () => [["node 0", R.mlog2log.slice(0, PERN).map((l) => `E${l}`).join(" ")], ["node 1", R.mlog2log.slice(PERN).map((l) => `E${l}`).join(" ")]] },
  { id: "rcall", block: "rcall", lines: [1], scene: "rep", phase: "init", r: 0, caption: "each node has 2 spare slots to give to its hot experts", vars: () => [["slots / node", `${P / N}`], ["experts / node", `${PERN}`], ["spare", `${P / N - PERN}`]] },
  { id: "ones", block: "rloop", lines: [0], scene: "rep", phase: "init", r: 0, ones: true, caption: "every expert starts with one copy: load per copy = load", vars: () => [["logcnt", "[1 1 1 1]"]] },
  ...[1, 2].flatMap((r) => [
    { id: `rp${r}`, block: "rloop", lines: [1, 2], visAt: 1, scene: "rep", phase: "pick", r, caption: `round ${r}: the expert with the most load per copy wins`, vars: () => [["round", `${r}`], ["winner node 0", `E${R.mlog2log[R.rep[0].events[r - 1].pick]}`], ["winner node 1", `E${R.mlog2log[PERN + R.rep[1].events[r - 1].pick]}`]] },
    { id: `rs${r}`, block: "rloop", lines: [3], scene: "rep", phase: "split", r, caption: `round ${r}: the winner gets a copy and its load splits`, vars: () => [["slot", `${PERN + r - 1}`], ["logcnt", `+1`]] },
  ]),
  { id: "tpp", block: "gpus", lines: [0], scene: "rep", phase: "final", r: 2, caption: "load of one copy = expert load ÷ copies", vars: () => [["tokens_per_phy", `[${R.tpp.flat().map(fm).join(" ")}]`]] },
  { id: "gcall2", block: "gpus", lines: [1], scene: "pack", stage: "gpu", k: 0, caption: "in each node, spread the 6 copies over its 2 GPUs, 3 per GPU", vars: () => [["items / node", `${P / N}`], ["packs", `${Q / N}`], ["per pack", `${PPG}`]] },
  { id: "gsort", block: "bpg", lines: [0], scene: "pack", stage: "gpu", k: 0, sortAnim: true, caption: "sort the copies heaviest first (both nodes at once)", vars: () => [["node 0", R.gpk[0].order.map((j) => GPU_ROWS[0].items[j].label).join(" ")], ["node 1", R.gpk[1].order.map((j) => GPU_ROWS[1].items[j].label).join(" ")]] },
  ...GPU_EV[0].map((_, i) => ({
    id: `pg${i + 1}`, block: "bpg", lines: [1, 2, 3], visAt: 1, scene: "pack", stage: "gpu", k: i + 1, fly: true,
    caption: `item ${i + 1} of 6 goes to the lighter GPU of its node`,
    vars: () => [0, 1].map((n) => [`node ${n}`, `${GPU_ROWS[n].items[GPU_EV[n][i].item].label} → GPU ${2 * n + GPU_EV[n][i].pack}`]),
  })),
  { id: "gpuload", block: "result", lines: [0], scene: "result", stage: 0, caption: "every GPU now carries about the same load", vars: () => [["phy2log", `[${R.phy2log.map((l) => `E${l}`).join(" ")}]`]] },
  { id: "log2phy", block: "result", lines: [1, 2], visAt: 0, scene: "result", stage: 1, caption: "routing looks up where each expert's copies live", vars: () => [["logcnt", `[${R.logcnt.join(" ")}]`]] },
  { id: "note", block: "note", lines: [0, 1], visAt: 0, scene: "note", caption: "bias picks the experts; EPLB decides where they live", vars: () => [["routing", "unchanged by EPLB"]] },
];

// ---------------------------------------------------------------- shapes
const row = (name, dims, what, io, note) => ({ name, dims, what, io, note });
const WROW = row("weight", ["L", "E"], "tokens sent to each logical expert over a window of traffic", "in");
const PACK_NODE = [
  row("tokens_per_group", ["L", "G"], "weight of each item to pack", "in"),
  row("pack", ["L", "G"], "node chosen for each group", "out"),
  row("rank", ["L", "G"], "slot of the group inside its node", "out"),
];
const PACK_GPU = [
  row("tokens_per_phy", ["L·N", "P/N"], "weight of each copy; one row per node", "in"),
  row("pack", ["L·N", "P/N"], "GPU (inside the node) chosen for each copy", "out"),
  row("rank", ["L·N", "P/N"], "slot of the copy on that GPU", "out"),
];
const REP = [
  row("tokens_per_mlog", ["L·N", "E/N"], "expert loads, one row per node", "in"),
  row("logcnt", ["L·N", "E/N"], "copies of each expert so far", "out"),
  row("phy2log", ["L·N", "P/N"], "expert behind each slot; first E/N slots are the originals", "out"),
];
const SHAPES = {
  load: [WROW],
  topo: [row("phy2log", ["L", "P"], "expert each physical slot will hold (filled in below)", "out")],
  gsum: [WROW, row("tokens_per_group", ["L", "G"], "sum over the E/G experts of a group", "out")],
  gcall: PACK_NODE, psort: PACK_NODE,
  mlog: [row("tokens_per_mlog", ["L·N", "E/N"], "regrouped view: row = node, column = expert local to that node", "out")],
  rcall: [row("tokens_per_mlog", ["L·N", "E/N"], "in", "in"), row("phy2mlog", ["L·N", "P/N"], "local expert behind each slot", "out"), row("mlogcnt", ["L·N", "E/N"], "copies per expert", "out")],
  ones: REP, rp1: REP, rs1: REP, rp2: REP, rs2: REP,
  tpp: [row("tokens_per_phy", ["L·N", "P/N"], "load of one copy = tokens_per_mlog ÷ mlogcnt, laid out by slot", "out")],
  gcall2: PACK_GPU, gsort: PACK_GPU,
  gpuload: [row("phy2log", ["L", "P"], "slot s (GPU-major, 3 per GPU) → logical expert it hosts", "out")],
  log2phy: [row("logcnt", ["L", "E"], "copies per logical expert", "out"), row("log2phy", ["L", "E", "R"], "slots holding each expert's copies; the router picks one of them", "out")],
};
nodeEv.forEach((_, i) => { SHAPES[`pn${i + 1}`] = PACK_NODE; });
GPU_EV[0].forEach((_, i) => { SHAPES[`pg${i + 1}`] = PACK_GPU; });
const GLOSSARY = {
  L: ["layers (one shown)", 1], E: ["logical experts", E], G: ["expert groups", G], N: ["nodes", N], Q: ["GPUs", Q],
  P: ["physical slots = E + 4 spare", P], R: ["max copies of one expert", Math.max(...R.logcnt)],
};

// ---------------------------------------------------------------- scenes
const Title = ({ x, y, children }) => <text x={x} y={y} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{children}</text>;
const dimOf = (sel, has) => (sel == null || has.includes(sel) ? 1 : 0.28);

function LoadScene({ p, sel }) {
  const e = ease(p), s = 1.7, base = 270;
  return (
    <g>
      <Title x={30} y={34}>TOKENS PER LOGICAL EXPERT (weight)</Title>
      <Title x={560} y={34}>NO EPLB · 2 EXPERTS PER GPU</Title>
      <line x1={20} x2={500} y1={base} y2={base} stroke={T.rule} />
      {WL.map((w, i) => (
        <g key={i} opacity={dimOf(sel, [i])}>
          <rect x={30 + i * 58} y={base - w * s * e} width={50} height={w * s * e} rx={3} fill={XC[i]} />
          <text x={55 + i * 58} y={base - w * s * e - 5} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.soft}>{Math.round(w * e)}</text>
          <text x={55 + i * 58} y={base + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={XC[i]}>E{i}</text>
        </g>
      ))}
      <line x1={550} x2={830} y1={base} y2={base} stroke={T.rule} />
      {NAIVE.map((tot, g) => {
        let cum = 0;
        const x = 570 + g * 66;
        return (
          <g key={g}>
            {[2 * g, 2 * g + 1].map((i) => {
              const h = WL[i] * s * e, y = base - cum * s * e - h;
              cum += WL[i];
              return <rect key={i} x={x} y={y} width={50} height={h} fill={XC[i]} opacity={0.85 * dimOf(sel, [i])} stroke={T.bg} />;
            })}
            <text x={x + 25} y={base - tot * s * e - 6} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={tot > MEAN * 1.3 ? T.bad : T.ink}>{Math.round(tot * e)}</text>
            <text x={x + 25} y={base + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.soft}>GPU {g}</text>
          </g>
        );
      })}
      <line x1={550} x2={830} y1={base - MEAN * s} y2={base - MEAN * s} stroke={T.wire} strokeDasharray="5 4" />
      <text x={830} y={base - MEAN * s - 4} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.wire}>even {fm(MEAN)}</text>
      <Tex x={30} y={318} w={800} h={34} size={15} color={T.ink} tex={`\\dfrac{\\max_g \\text{load}_g}{\\text{mean}}=\\dfrac{${col(T.bad, Math.max(...NAIVE))}}{${fm(MEAN)}}=${col(T.bad, f2(Math.max(...NAIVE) / MEAN))}\\quad\\text{the busiest GPU sets the step time}`} opacity={e} />
      <Tex x={30} y={366} w={800} h={26} size={12} color={T.dim} tex={`\\text{EPLB reads these loads and decides where the experts live. It never changes which expert a token picks.}`} opacity={e} />
    </g>
  );
}

function TopoScene({ p }) {
  const e = ease(p);
  return (
    <g>
      <text x={30} y={32} fontFamily={mono} fontSize="12" fill={T.soft}>8 experts in {G} groups of 2 · {P} slots = 8 + <tspan fill={T.wire} fontWeight="700">4 spare</tspan></text>
      {Array.from({ length: N }, (_, n) => (
        <g key={n} opacity={clamp01(e * 2 - n * 0.4)}>
          <rect x={30 + n * 410} y={56} width={390} height={210} rx={12} fill={T.panel} stroke={T.rule} />
          <text x={44 + n * 410} y={78} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.ink}>node {n}</text>
          {[0, 1].map((gi) => {
            const gx = 45 + n * 410 + gi * 185;
            return (
              <g key={gi}>
                <rect x={gx} y={96} width={175} height={156} rx={8} fill={T.well} stroke={T.rule} />
                <text x={gx + 10} y={116} fontFamily={mono} fontSize="11" fill={T.soft}>GPU {n * 2 + gi}</text>
                {[0, 1, 2].map((s) => (
                  <g key={s}>
                    <rect x={gx + 10 + s * 54} y={130} width={50} height={90} rx={5} fill="none" stroke={T.rule} strokeDasharray="3 3" />
                    <text x={gx + 35 + s * 54} y={236} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>s{(n * 2 + gi) * PPG + s}</text>
                  </g>
                ))}
              </g>
            );
          })}
        </g>
      ))}
      <Tex x={30} y={290} w={800} h={30} size={15} color={T.ink} tex={`G \\bmod N=${G}\\bmod ${N}=${col(T.accent, 0)}\\;\\Rightarrow\\;\\text{hierarchical: a group never straddles two nodes}`} opacity={e} />
      <Tex x={30} y={332} w={800} h={30} size={13} color={T.soft} tex={`\\text{slots per GPU}=\\dfrac{P}{Q}=\\dfrac{${P}}{${Q}}=${col(T.wire, PPG)}\\qquad \\text{spare slots}=P-E=${P}-${E}=${col(T.wire, P - E)}`} opacity={e} />
      <Tex x={30} y={372} w={800} h={26} size={12} color={T.dim} tex={`\\text{three decisions: which node owns a group, which experts get spare copies, which GPU holds each copy}`} opacity={e} />
    </g>
  );
}

function GroupSumScene({ p, sel }) {
  const e = ease(p), s = 1.7, base = 270;
  const cum = Array(G).fill(0);
  return (
    <g>
      <Title x={30} y={34}>EXPERT LOAD</Title>
      <Title x={560} y={34}>GROUP LOAD = SUM OF ITS EXPERTS</Title>
      <line x1={20} x2={500} y1={base} y2={base} stroke={T.rule} />
      <line x1={550} x2={830} y1={base} y2={base} stroke={T.rule} />
      {WL.map((w, i) => {
        const g = Math.floor(i / 2), sx = 30 + i * 58, sy = base - w * s;
        const dx = 570 + g * 66, dy = base - (cum[g] + w) * s;
        cum[g] += w;
        return (
          <g key={i} opacity={dimOf(sel, [i])}>
            <rect x={sx} y={sy} width={50} height={w * s} rx={3} fill="none" stroke={XC[i]} strokeDasharray="3 3" opacity={0.5} />
            <rect x={lerp(sx, dx, e)} y={lerp(sy, dy, e)} width={50} height={w * s} rx={3} fill={XC[i]} stroke={T.bg} />
            <text x={55 + i * 58} y={base + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={XC[i]}>E{i}</text>
          </g>
        );
      })}
      {R.tpg.map((t, g) => (
        <g key={g}>
          <text x={595 + g * 66} y={base - t * s * e - 6} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={GC[g]} opacity={e}>{t}</text>
          <text x={595 + g * 66} y={base + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={GC[g]}>g{g}</text>
        </g>
      ))}
      {R.tpg.map((t, g) => (
        <Tex key={g} x={30 + (g % 2) * 420} y={312 + Math.floor(g / 2) * 36} w={400} h={30} size={14} color={T.ink} opacity={e}
          tex={`w_{g${g}}=w_{E${2 * g}}+w_{E${2 * g + 1}}=${WL[2 * g]}+${WL[2 * g + 1]}=${col(GC[g], t)}`} />
      ))}
    </g>
  );
}

function PackScene({ step, p, sel }) {
  const e = ease(p);
  const rows = step.stage === "node" ? [NODE_ROW] : GPU_ROWS;
  const k = step.k;
  return (
    <g>
      {rows.map((rw, ri) => {
        const pos = (id) => rw.order.indexOf(id);
        const qx = (id) => {
          const o = rw.x0 + id * (rw.itemW + rw.gap), so = rw.x0 + pos(id) * (rw.itemW + rw.gap);
          return k === 0 ? (step.sortAnim ? lerp(o, so, e) : o) : so;
        };
        const evs = rw.events;
        const cur = k > 0 ? evs[k - 1] : null;
        const prev = k > 1 ? evs[k - 2] : null;
        const fe = step.fly ? e : 1;
        const tot = Array.from({ length: rw.m }, (_, q) => lerp(prev ? prev.pwAfter[q] : 0, cur ? cur.pwAfter[q] : 0, fe));
        const cntN = Array.from({ length: rw.m }, (_, q) => (cur ? cur.pnAfter[q] : 0));
        const px = (q) => rw.packX + q * (rw.packW + rw.packGap);
        const total = sum(rw.items.map((it) => it.w));
        const ideal = total / rw.m;
        return (
          <g key={ri}>
            <Title x={rw.x0} y={30}>{rw.title}</Title>
            <line x1={rw.x0 - 10} x2={rw.x0 + rw.items.length * (rw.itemW + rw.gap)} y1={rw.baseY} y2={rw.baseY} stroke={T.rule} />
            {Array.from({ length: rw.m }, (_, q) => {
              const chosen = cur && step.fly && cur.pack === q;
              return (
                <g key={q}>
                  <rect x={px(q) - 4} y={rw.baseY - (ideal * 1.45) * rw.scale} width={rw.packW + 8} height={ideal * 1.45 * rw.scale} rx={6} fill="none" stroke={chosen ? T.accent : T.rule} strokeWidth={chosen ? 2 : 1} strokeDasharray="4 4" />
                  <text x={px(q) + rw.packW / 2} y={rw.baseY + 16} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={chosen ? T.accent : T.soft}>{rw.packName(q)}</text>
                  <text x={px(q) + rw.packW / 2} y={rw.baseY - tot[q] * rw.scale - 9} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={T.wire}>{tot[q] > 0 ? fm(Math.round(tot[q] * 10) / 10) : ""}</text>
                  <text x={px(q) + rw.packW / 2} y={rw.baseY + 30} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>{cntN[q]}/{rw.cap} items</text>
                </g>
              );
            })}
            <line x1={rw.packX - 10} x2={px(rw.m - 1) + rw.packW + 10} y1={rw.baseY - ideal * rw.scale} y2={rw.baseY - ideal * rw.scale} stroke={T.wire} strokeDasharray="5 4" opacity={0.7} />
            <text x={px(rw.m - 1) + rw.packW + 10} y={rw.baseY - ideal * rw.scale - 4} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.wire}>even {fm(ideal)}</text>
            {rw.items.map((it, id) => {
              const ei = pos(id), h = it.w * rw.scale;
              const sx = qx(id), sy = rw.baseY - h;
              const op = dimOf(sel, it.has);
              const placed = ei < k;
              const flying = placed && ei === k - 1 && step.fly;
              const ev = evs[ei];
              const dx = ev ? px(ev.pack) : sx, dy = ev ? rw.baseY - (ev.pwBefore[ev.pack] + it.w) * rw.scale : sy;
              const small = rw.itemW < 40;
              const lab = (x, y, w2, fs) => (
                <text x={x + w2 / 2} y={y + h / 2 + 4} textAnchor="middle" fontFamily={mono} fontSize={fs} fontWeight="700" fill={T.bg}>{h > 15 ? (w2 > 50 ? `${it.label} ${fm(Math.round(it.w * 10) / 10)}` : it.label) : ""}</text>
              );
              if (!placed) {
                return (
                  <g key={id} opacity={op}>
                    <rect x={sx} y={sy} width={rw.itemW} height={h} rx={3} fill={it.color} stroke={T.bg} />
                    {lab(sx, sy, rw.itemW, small ? 8.5 : 11)}
                    <text x={sx + rw.itemW / 2} y={sy - 5} textAnchor="middle" fontFamily={mono} fontSize={small ? 9.5 : 11} fill={T.soft}>{fm(Math.round(it.w * 10) / 10)}</text>
                    <text x={sx + rw.itemW / 2} y={rw.baseY + 14} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={it.color}>{it.label}</text>
                  </g>
                );
              }
              const t = flying ? e : 1;
              const x = lerp(rw.x0 + pos(id) * (rw.itemW + rw.gap), dx, t), y = lerp(sy, dy, t) - Math.sin(Math.PI * t) * (flying ? 34 : 0), w2 = lerp(rw.itemW, rw.packW, t);
              const sx2 = rw.x0 + pos(id) * (rw.itemW + rw.gap);
              return (
                <g key={id} opacity={op}>
                  <rect x={sx2} y={sy} width={rw.itemW} height={h} rx={3} fill="none" stroke={it.color} strokeDasharray="3 3" opacity={0.4} />
                  <text x={sx2 + rw.itemW / 2} y={rw.baseY + 14} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>{it.label}</text>
                  <rect x={x} y={y} width={w2} height={h} rx={3} fill={it.color} stroke={flying ? T.ink : T.bg} strokeWidth={flying ? 2 : 1} />
                  {lab(x, y, w2, small ? 9.5 : 11)}
                </g>
              );
            })}
            {cur && step.fly && (
              <g>
                <Tex x={rw.x0} y={rw.texY} w={rw.items.length > 4 ? 410 : 800} h={30} size={rw.items.length > 4 ? 11 : 14} color={T.ink} opacity={clamp01(e * 3)}
                  tex={`${rw.items[cur.item].label}\\ (${fm(rw.items[cur.item].w)})\\to\\arg\\min\\!\\big(${Array.from({ length: rw.m }, (_, q) => {
                    const room = cur.pnBefore[q] < rw.cap;
                    const lightest = room && cur.pack === q;
                    return `w_{${rw.wname(q)}}{=}${room ? (lightest ? col(T.accent, fm(cur.pwBefore[q])) : fm(cur.pwBefore[q])) : "\\text{full}"}`;
                  }).join(",\\,")}\\big)=\\text{${rw.packName(cur.pack)}}`} />
                <Tex x={rw.x0} y={rw.texY + 34} w={rw.items.length > 4 ? 410 : 800} h={30} size={rw.items.length > 4 ? 11 : 14} color={T.soft} opacity={clamp01(e * 3 - 1.6)}
                  tex={`w_{${rw.wname(cur.pack)}}\\leftarrow ${fm(cur.pwBefore[cur.pack])}+${fm(rw.items[cur.item].w)}=${col(T.wire, fm(cur.pwAfter[cur.pack]))}`} />
              </g>
            )}
            {k === 0 && (
              <Tex x={rw.x0} y={rw.texY} w={800} h={26} size={12} color={T.dim} tex={step.sortAnim ? `\\text{heaviest first: the big items are placed while every pack still has room to balance them}` : `\\text{${rw.items.length} items, ${rw.m} packs, exactly ${rw.cap} per pack; ${step.stage === "node" ? "one row" : "one row per node"}}`} />
            )}
          </g>
        );
      })}
    </g>
  );
}

function RepScene({ step, p, sel }) {
  const e = ease(p), s = 1.6, base = 262, ph = step.phase, r = step.r;
  const colX = (n, j) => 20 + n * 430 + 8 + j * 66;
  const origX = (l) => 20 + l * 52;
  const nodes = [0, 1];
  return (
    <g>
      {nodes.map((n) => {
        const evt = r > 0 && (ph === "pick" || ph === "split") ? R.rep[n].events[r - 1] : null;
        const before = repCols(n, ph === "split" ? r - 1 : ph === "pick" ? r - 1 : ph === "final" ? 2 : 0);
        const after = ph === "split" ? repCols(n, r) : before;
        const winLg = evt ? R.mlog2log[n * PERN + evt.pick] : -1;
        const ox = 20 + n * 430;
        return (
          <g key={n}>
            <rect x={ox} y={44} width={410} height={base - 44 + 52} rx={10} fill={T.panel} stroke={T.rule} opacity={ph === "mlog" ? e : 1} />
            <Title x={ox + 10} y={36}>NODE {n} · load ÷ copies</Title>
            <line x1={ox + 6} x2={ox + 404} y1={base} y2={base} stroke={T.rule} />
            {after.map((cA, j) => {
              const cB = before[j];
              const x0 = colX(n, j);
              const slotLab = <text x={x0 + 25} y={base + 34} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>slot {j}</text>;
              if (!cA.on && !cB.on) {
                return (
                  <g key={j}>
                    <rect x={x0} y={base - 60} width={50} height={60} rx={3} fill="none" stroke={T.rule} strokeDasharray="3 3" />
                    {slotLab}
                  </g>
                );
              }
              let x = x0, v = cA.v, c = cA.c, lg = cA.lg;
              if (ph === "split" && !cB.on) {
                const src = evt.pick; // new copy appears at its source column and slides to its slot
                x = lerp(colX(n, src), x0, e);
                v = lerp(before[src].v, cA.v, e);
                c = cA.c;
              } else if (ph === "split") {
                v = lerp(cB.v, cA.v, e);
                c = e > 0.5 ? cA.c : cB.c;
              }
              if (ph === "mlog") {
                const ml = n * PERN + j;
                if (j < PERN) { x = lerp(origX(R.mlog2log[ml]), x0, e); }
              }
              const h = v * s;
              const win = evt && ph === "pick" && cA.lg === winLg && j === evt.pick;
              const op = dimOf(sel, [lg]);
              return (
                <g key={j} opacity={op}>
                  <rect x={x} y={base - h} width={50} height={h} rx={3} fill={XC[lg]} stroke={win ? T.ink : T.bg} strokeWidth={win ? 2.4 : 1} />
                  <text x={x + 25} y={base - h - 5} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight={win || ph === "final" ? 800 : 400} fill={win ? T.ink : T.soft}>{fm(Math.round(v * 10) / 10)}</text>
                  <text x={x + 25} y={base + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={XC[lg]}>E{lg}</text>
                  <text x={x + 25} y={base + 26} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>{ph === "mlog" ? "" : `÷${c}`}</text>
                  {slotLab}
                </g>
              );
            })}
            {ph === "pick" && (() => {
              const ev = R.rep[n].events[r - 1];
              const terms = ev.cand.map((v, j) => {
                const t = `\\tfrac{${WL[R.mlog2log[n * PERN + j]]}}{${ev.cntBefore[j]}}`;
                return j === ev.pick ? col(XC[R.mlog2log[n * PERN + j]], t) : t;
              });
              return (
                <Tex x={ox} y={base + 50} w={410} h={34} size={12} color={T.ink} opacity={clamp01(e * 3)}
                  tex={`\\max\\!\\Big(${terms.join(",\\,")}\\Big)\\Rightarrow\\ \\text{E${winLg}}`} />
              );
            })()}
            {ph === "split" && (() => {
              const ev = R.rep[n].events[r - 1];
              const lg = R.mlog2log[n * PERN + ev.pick], w = WL[lg], c0 = ev.cntBefore[ev.pick];
              return (
                <Tex x={ox} y={base + 50} w={410} h={34} size={12} color={T.ink} opacity={clamp01(e * 2)}
                  tex={`c_{E${lg}}\\!:\\ ${c0}\\to ${col(XC[lg], c0 + 1)}\\qquad \\tfrac{${w}}{${c0}}\\to ${col(XC[lg], `\\tfrac{${w}}{${c0 + 1}}=${fm(w / (c0 + 1))}`)}`} />
              );
            })()}
            {ph === "mlog" && (
              <Tex x={ox} y={base + 50} w={410} h={30} size={12} color={T.dim} opacity={e} tex={`\\text{groups ${n === 0 ? R.gp.pack.map((q, g) => (q === 0 ? g : null)).filter((g) => g != null).map((g) => `g${g}`).join(",") : R.gp.pack.map((q, g) => (q === 1 ? g : null)).filter((g) => g != null).map((g) => `g${g}`).join(",")} live here}`} />
            )}
            {(ph === "init" || ph === "final") && (
              <Tex x={ox} y={base + 50} w={410} h={30} size={12} color={T.dim} tex={ph === "final" ? `\\text{one row of tokens\\_per\\_phy per node}` : step.ones ? `c_e=1\\ \\forall e\\ \\Rightarrow\\ \\tfrac{w}{c}=w;\\ \\text{slots }${PERN},${PERN + 1}\\text{ wait for copies}` : `\\text{${PERN} experts, ${P / N} slots: ${P / N - PERN} spare}`} />
            )}
          </g>
        );
      })}
      {ph === "mlog" && (
        <g opacity={1 - e}>
          {WL.map((w, l) => <text key={l} x={origX(l) + 22} y={base + 70} textAnchor="middle" fontFamily={mono} fontSize="9" fill={XC[l]}>E{l}</text>)}
        </g>
      )}
    </g>
  );
}

function ResultScene({ step, p, sel }) {
  const e = ease(p), s = 2.0, base = 290;
  const bx = (g) => 40 + g * 92;
  return (
    <g>
      <Title x={30} y={34}>LOAD PER GPU · BEFORE (dashed) → AFTER</Title>
      <line x1={20} x2={410} y1={base} y2={base} stroke={T.rule} />
      {Array.from({ length: Q }, (_, g) => {
        const f = lerp(NAIVE[g], AFTER[g], e) / AFTER[g];
        let cum = 0;
        return (
          <g key={g}>
            <rect x={bx(g)} y={base - NAIVE[g] * s} width={70} height={NAIVE[g] * s} fill="none" stroke={T.bad} strokeDasharray="4 3" opacity={0.7} />
            {R.phy2log.slice(g * PPG, (g + 1) * PPG).map((l, i) => {
              const v = (WL[l] / R.logcnt[l]) * f, y = base - (cum + v) * s;
              cum += v;
              return <rect key={i} x={bx(g)} y={y} width={70} height={v * s} fill={XC[l]} stroke={T.bg} opacity={dimOf(sel, [l])} />;
            })}
            <text x={bx(g) + 35} y={base - lerp(NAIVE[g], AFTER[g], e) * s - 6} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={T.ink}>{Math.round(lerp(NAIVE[g], AFTER[g], e))}</text>
            <text x={bx(g) + 35} y={base + 15} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.soft}>GPU {g}</text>
          </g>
        );
      })}
      <line x1={20} x2={410} y1={base - MEAN * s} y2={base - MEAN * s} stroke={T.wire} strokeDasharray="5 4" />
      <text x={410} y={base - MEAN * s - 4} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.wire}>even {fm(MEAN)}</text>

      {step.stage === 0 ? (
        <g>
          <Title x={450} y={34}>phy2log · EXPERT IN EACH SLOT</Title>
          {Array.from({ length: Q }, (_, g) => (
            <g key={g} opacity={clamp01(e * 2.5 - g * 0.3)}>
              <text x={450} y={72 + g * 52} fontFamily={mono} fontSize="11" fill={T.soft}>GPU {g}</text>
              {R.phy2log.slice(g * PPG, (g + 1) * PPG).map((l, i) => {
                const sl = g * PPG + i;
                return (
                  <g key={i} opacity={dimOf(sel, [l])}>
                    <rect x={506 + i * 108} y={50 + g * 52} width={98} height={38} rx={6} fill={XC[l]} fillOpacity={0.9} />
                    <text x={555 + i * 108} y={68 + g * 52} textAnchor="middle" fontFamily={mono} fontSize="13" fontWeight="800" fill={T.bg}>E{l}</text>
                    <text x={555 + i * 108} y={82 + g * 52} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.bg}>slot {sl} · copy {R.phyrank[sl]}</text>
                  </g>
                );
              })}
            </g>
          ))}
        </g>
      ) : (
        <g>
          <Title x={450} y={34}>log2phy · WHERE EACH EXPERT'S COPIES LIVE</Title>
          {Array.from({ length: E }, (_, l) => (
            <g key={l} opacity={dimOf(sel, [l]) * clamp01(e * 3 - l * 0.2)}>
              <rect x={450} y={46 + l * 30} width={44} height={24} rx={5} fill={XC[l]} />
              <text x={472} y={63 + l * 30} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="800" fill={T.bg}>E{l}</text>
              <text x={506} y={63 + l * 30} fontFamily={mono} fontSize="11.5" fill={T.soft}>×{R.logcnt[l]}</text>
              {LOG2PHY[l].map((sl, i) => (
                <g key={i}>
                  <rect x={548 + i * 74} y={46 + l * 30} width={66} height={24} rx={5} fill="none" stroke={XC[l]} />
                  <text x={581 + i * 74} y={63 + l * 30} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.ink}>s{sl}</text>
                </g>
              ))}
            </g>
          ))}
        </g>
      )}
      <Tex x={30} y={336} w={400} h={34} size={14} color={T.ink} tex={`\\dfrac{\\max}{\\text{mean}}:\\ ${col(T.bad, f2(Math.max(...NAIVE) / MEAN))}\\ \\to\\ ${col(T.accent, f2(Math.max(...AFTER) / MEAN))}`} opacity={e} />
      <Tex x={30} y={376} w={400} h={26} size={12} color={T.dim} tex={`\\text{${E} experts became ${P} slots; hot experts have copies, every GPU holds ${PPG} slots}`} opacity={e} />
    </g>
  );
}

function NoteScene({ p }) {
  const e = ease(p);
  const card = (x, c, a, b, d, f) => (
    <g opacity={e}>
      <rect x={x} y={60} width={390} height={200} rx={12} fill={T.well} stroke={c} strokeWidth={1.6} />
      <text x={x + 20} y={96} fontFamily={mono} fontSize="15" fontWeight="700" fill={c}>{a}</text>
      <text x={x + 20} y={134} fontFamily={mono} fontSize="12.5" fill={T.ink}>{b}</text>
      <text x={x + 20} y={160} fontFamily={mono} fontSize="12" fill={T.soft}>{d}</text>
      <text x={x + 20} y={186} fontFamily={mono} fontSize="12" fill={T.soft}>{f}</text>
    </g>
  );
  return (
    <g>
      {card(30, T.accent, "V3 bias b", "changes WHICH experts a token picks", "acts every training step", "no gradient, no copies")}
      {card(440, T.wire, "EPLB", "changes WHERE experts live", "run every so often on measured loads", "hot experts get copies; routing untouched")}
      <text x={30} y={310} fontFamily={mono} fontSize="12.5" fill={T.soft}>use both: the bias evens out what is chosen, EPLB evens out what is left</text>
    </g>
  );
}

function Scene({ step, sel }) {
  const p = useProgress(step.id);
  const body = {
    load: <LoadScene p={p} sel={sel} />,
    topo: <TopoScene p={p} />,
    gsum: <GroupSumScene p={p} sel={sel} />,
    pack: <PackScene step={step} p={p} sel={sel} />,
    rep: <RepScene step={step} p={p} sel={sel} />,
    result: <ResultScene step={step} p={p} sel={sel} />,
    note: <NoteScene p={p} />,
  }[step.scene];
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

export { Scene as EplbScene, STEPS as EPLB_STEPS, BLOCKS as EPLB_BLOCKS };

export default function EplbView() {
  const [sel, setSel] = useState(null); // logical expert to follow through every picture
  return (
    <Stepper
      title="EPLB · expert placement"
      subtitle={`${E} experts, skewed load · ${N} nodes × ${Q / N} GPUs · ${P - E} spare slots (${P} physical)`}
      blocks={BLOCKS}
      steps={STEPS}
      scene={(step) => <Scene step={step} sel={sel} />}
      header={() => (
        <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow expert</span>
          {Array.from({ length: E }, (_, j) => (
            <button key={j} onClick={() => setSel(sel === j ? null : j)} style={{ ...btn(j === sel, false), height: 28, padding: "0 9px", color: j === sel ? undefined : XC[j] }}>E{j}</button>
          ))}
        </div>
      )}
      glossary={GLOSSARY}
      shapes={(step) => SHAPES[step.sid]}
      legend={<span>source: deepseek-ai/EPLB · load is split evenly between an expert's copies · <span style={{ color: T.wire }}>dashed</span> = even split</span>}
    />
  );
}
