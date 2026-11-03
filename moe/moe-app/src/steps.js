import { CFG, MODEL } from "./moeModel";

// One source of truth for the walkthrough: each step names the flowchart
// block it belongs to, which canvas scene/stage to draw, the paper's code
// lines it covers, the tensors it touches, and the live variables to show.

const { G, S, E, C, N } = CFG;

const A1 = {
  1: "c_E = 0   # [E]  one counter per expert",
  2: "g_SE = softmax(wg · x_S)   # [S,M]·[M,E] → [S,E]",
  3: "m_E = mean(g_SE, over S)   # [S,E] → [E]",
  4: "for s in 1..S:",
  5: "  g1,e1,g2,e2 = top_2(g_SE[s])   # g_SE[s] is [E] → 2 weights, 2 ids",
  6: "  g1 = g1 / (g1 + g2)   # scalars; g1+g2 = 1",
  7: "  c = c_E[e1]   # scalar: how full bucket e1 is",
  8: "  if c < C:",
  9: "    G[s,e1] = g1   # combine_weights [S,E] (+ slot c)",
  11: "  c_E[e1] = c + 1",
  13: "l_aux = 1/E · Σ_e (c_E[e]/S) · m_E[e]   # [E]·[E] → scalar",
  14: "for s in 1..S:",
  15: "  g1,e1,g2,e2 = top_2(g_SE[s])   # g_SE[s] is [E] → 2 weights, 2 ids",
  16: "  g2 = g2 / (g1 + g2)",
  17: "  rnd = uniform(0, 1)   # scalar per token",
  18: "  c = c_E[e2]",
  19: "  if c < C and 2·g2 > rnd:",
  20: "    G[s,e2] = g2",
  22: "  c_E[e2] = c + 1",
};

const A2 = {
  1: 'gates = softmax(einsum("GSM,ME->GSE", inputs, wg))   # [G,S,M]·[M,E] → [G,S,E]',
  2: "combine_weights, dispatch_mask = Top2Gating(gates)   # [G,S,E] → two [G,S,E,C]",
  3: 'dispatched = einsum("GSEC,GSM->EGCM", dispatch_mask, inputs)   # → [E,G,C,M]',
  5: 'h = einsum("EGCM,EMH->EGCH", dispatched, wi)   # → [E,G,C,H]',
  6: "h = relu(h)   # [E,G,C,H] unchanged",
  7: 'expert_out = einsum("EGCH,EHM->GECM", h, wo)   # → [G,E,C,M]',
  8: 'outputs = einsum("GSEC,GECM->GSM", combine_weights, expert_out)   # → [G,S,M]',
  s1: "inputs = split(inputs, 0, D)     # [G,S,M] → [1,S,M] per device",
  s2: "wg = replicate(wg)   # [M,E] full copy everywhere",
  s3: "dispatched = split(dispatched, 0, D)  # [E,G,C,M]: split G → split E",
};

const t = (name, dims, extra = {}) => ({ name, dims, ...extra });
const ids = (arr) => arr.map((x) => `x${x}`).join(", ") || "none";

export const FLOW = [
  { id: "split", label: "split into groups", sub: "x[N,M] → [G,S,M]", code: [A2.s1, A2.s2] },
  {
    group: "Top2Gating · Algorithm 1",
    items: [
      { id: "softmax", label: "softmax gates + mean", sub: "g_SE, m_E", code: [A1[2], A1[3]] },
      { id: "top2", label: "pick top-2", sub: "e1, e2 per token", code: [A1[5], A1[6]] },
      { id: "pass1", label: "for s: fill 1st choice", sub: "counters c_E vs C", loop: "↻ ×S", code: [A1[1], A1[4], A1[5], A1[6], A1[7], A1[8], A1[9], A1[11]] },
      { id: "aux", label: "balance loss", sub: "ℓ_aux", code: [A1[13]] },
      { id: "pass2", label: "for s: random 2nd choice", sub: "dice + capacity", loop: "↻ ×S", code: [A1[14], A1[15], A1[16], A1[17], A1[18], A1[19], A1[20], A1[22]] },
      { id: "weights", label: "combine_weights", sub: "[G,S,E,C]", code: [A2[2]] },
    ],
  },
  { id: "dispatch", label: "dispatch einsum", sub: "all-to-all  G → E", code: [A2[3], A2.s3] },
  { id: "ffn", label: "expert FFN", sub: "each device, own tokens", code: [A2[5], A2[6], A2[7]] },
  { id: "combine", label: "all-to-all back", sub: "E → G", code: ["expert_out = split(expert_out, 0, D)   # [G,E,C,M]: reshard E → G, the second all-to-all"] },
  { id: "output", label: "weighted sum", sub: "y = g1·FFN + g2·FFN", code: [A2[8]] },
];

export const STEPS = [
  {
    id: "split", block: "split", scene: "devices", stage: 0,
    caption: `${N} tokens → ${G} groups of ${S}, one group per device`,
    lines: [0, 1], visAt: 0,
    tensors: {
      ins: [t("inputs", ["G", "S", "M"], { shard: "G" }), t("wg", ["M", "E"], { repl: true })],
      op: "split(inputs, 0, D)  ·  replicate(wg)",
    },
    vars: () => [["N", `${N}`], ["G", `${G}`], ["S", `${S}`], ["C = 2S/E", `${C}`]],
  },
  {
    id: "gates", block: "softmax", scene: "group", stage: 0,
    caption: "every token scores every expert",
    lines: [0],
    tensors: {
      ins: [t("inputs", ["G", "S", "M"], { shard: "G" }), t("wg", ["M", "E"], { repl: true })],
      op: 'einsum("GSM,ME->GSE")  →  softmax',
      outs: [t("gates", ["G", "S", "E"], { shard: "G" })],
    },
    vars: () => [["g_SE", `[${S}×${E}]`]],
  },
  {
    id: "mean", block: "softmax", scene: "group", stage: 1,
    caption: "m_e: how much the router likes each expert",
    lines: [1],
    tensors: {
      ins: [t("gates", ["G", "S", "E"], { shard: "G" })],
      op: "mean over S",
      outs: [t("m_E", ["G", "E"], { shard: "G" })],
    },
    vars: (gm) => [["m_E", `[${gm.m.map((x) => x.toFixed(2)).join(", ")}]`]],
  },
  {
    id: "top2", block: "top2", scene: "group", stage: 2,
    caption: "each token keeps its two best experts",
    lines: [0, 1], visAt: 0,
    tensors: {
      ins: [t("gates", ["G", "S", "E"], { shard: "G" })],
      op: "top_2  ·  g1 / (g1 + g2)",
      outs: [t("g1, e1, g2, e2", ["G", "S"], { shard: "G" })],
    },
    vars: (gm) => {
      const k = gm.toks[0];
      return [["x0", `E${k.e1} ${k.g1n.toFixed(2)} · E${k.e2} ${k.g2n.toFixed(2)}`]];
    },
  },
  {
    id: "pass1", block: "pass1", scene: "group", stage: 3,
    caption: `1st choices fill the buffers · capacity C = ${C}`,
    lines: [],
    tensors: {
      ins: [t("e1, g1", ["G", "S"], { shard: "G" })],
      op: "sequential per group: c_E[e1] < C ?",
      outs: [t("c_E", ["G", "E"], { shard: "G" }), t("combine_weights", ["G", "S", "E", "C"], { shard: "G" })],
    },
    vars: (gm) => {
      const over = gm.toks.filter((k) => !k.t1.ok);
      return [
        ["c_E", `[${gm.cPass1.join(", ")}]`],
        ["C", `${C}`],
        ["overflow", over.length ? over.map((k) => `x${k.s}→E${k.e1}`).join(", ") : "none", over.length ? "bad" : null],
      ];
    },
  },
  {
    id: "aux", block: "aux", scene: "group", stage: 4,
    caption: "ℓ_aux grows when the popular expert is also picked often",
    lines: [0],
    tensors: {
      ins: [t("c_E", ["G", "E"], { shard: "G" }), t("m_E", ["G", "E"], { shard: "G" })],
      op: "ℓ_aux = 1/E · Σ_e (c_e/S) · m_e   →   mean over groups",
      outs: [t("ℓ_aux", [], { note: "scalar, added to loss: L = ℓ_nll + k·ℓ_aux" })],
    },
    vars: (gm) => [["ℓ_aux", gm.aux.toFixed(3)], ["balanced", (1 / (E * E)).toFixed(4)]],
  },
  {
    id: "pass2", block: "pass2", scene: "group", stage: 5,
    caption: "2nd choice: roll the dice, then check capacity",
    lines: [],
    tensors: {
      ins: [t("e2, g2", ["G", "S"], { shard: "G" }), t("c_E", ["G", "E"], { shard: "G" })],
      op: "keep e2 if  c < C  and  2·g2 > rnd",
      outs: [t("combine_weights", ["G", "S", "E", "C"], { shard: "G" })],
    },
    vars: (gm) => {
      const gone = gm.toks.filter((k) => k.final.length === 0).map((k) => k.s);
      return [["c_E", `[${gm.cFinal.join(", ")}]`], ["dropped", ids(gone), gone.length ? "bad" : null]];
    },
  },
  {
    id: "weights", block: "weights", scene: "group", stage: 6,
    caption: "≤ 2 weights per token · dropped tokens ride the residual",
    lines: [0],
    tensors: {
      ins: [t("gates", ["G", "S", "E"], { shard: "G" })],
      op: "Top2Gating",
      outs: [
        t("combine_weights", ["G", "S", "E", "C"], { shard: "G" }),
        t("dispatch_mask", ["G", "S", "E", "C"], { shard: "G", note: "weights → 1" }),
      ],
    },
    vars: (gm) => [["kept", `${gm.toks.reduce((a, k) => a + k.final.length, 0)} / ${2 * S}`], ["aux", gm.aux.toFixed(3)]],
  },
  {
    id: "dispatch", block: "dispatch", scene: "devices", stage: 1,
    caption: "all-to-all: tokens fly to their experts",
    lines: [0, 1], visAt: 1,
    tensors: {
      ins: [t("dispatch_mask", ["G", "S", "E", "C"], { shard: "G" }), t("inputs", ["G", "S", "M"], { shard: "G" })],
      op: 'einsum("GSEC,GSM->EGCM")',
      outs: [
        t("dispatched", ["E", "G", "C", "M"], { shard: "G" }),
        t("dispatched", ["E", "G", "C", "M"], { shard: "E", comm: "ALL-TO-ALL  G → E" }),
      ],
    },
    vars: () => [["load / expert", MODEL.load.join(" · ")], ["buffer", `${G}×${C} = ${G * C}`]],
  },
  {
    id: "ffn", block: "ffn", scene: "devices", stage: 2,
    caption: "each expert runs only the tokens it received",
    lines: [0, 1, 2], visAt: 0,
    tensors: {
      ins: [t("dispatched", ["E", "G", "C", "M"], { shard: "E" }), t("wi, wo", ["E", "M", "H"], { shard: "E" })],
      op: "einsum  →  relu  →  einsum   (no communication)",
      outs: [t("expert_out", ["E", "G", "C", "M"], { shard: "E" })],
    },
    vars: () => [["busy slots", MODEL.load.join(" · ")], ["idle", MODEL.load.map((l) => G * C - l).join(" · ")]],
  },
  {
    id: "combine", block: "combine", scene: "devices", stage: 3,
    caption: "all-to-all back to each token's own device",
    lines: [0],
    tensors: {
      ins: [t("expert_out", ["E", "G", "C", "M"], { shard: "E" })],
      op: 'output sharding of  "EGCH,EHM->GECM"',
      outs: [t("expert_out", ["G", "E", "C", "M"], { shard: "G", comm: "ALL-TO-ALL  E → G" })],
    },
    vars: () => [],
  },
  {
    id: "output", block: "output", scene: "devices", stage: 4,
    caption: "y = g₁·FFN(e₁) + g₂·FFN(e₂)",
    lines: [0],
    tensors: {
      ins: [t("combine_weights", ["G", "S", "E", "C"], { shard: "G" }), t("expert_out", ["G", "E", "C", "M"], { shard: "G" })],
      op: 'einsum("GSEC,GECM->GSM")',
      outs: [t("outputs", ["G", "S", "M"], { shard: "G" })],
    },
    vars: () => {
      const gone = MODEL.groups.reduce((a, g) => a + g.toks.filter((k) => k.final.length === 0).length, 0);
      return [["tokens", `${N}`], ["residual only", `${gone}`, gone ? "bad" : null]];
    },
  },
];
