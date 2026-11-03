import { CFG, MODEL } from "./moeModel";

// One source of truth for the walkthrough: each step names the flowchart
// block it belongs to, which canvas scene/stage to draw, the paper's code
// lines it covers, the tensors it touches, and the live variables to show.

const { G, S, E, C, N } = CFG;

const A1 = {
  1: "c_E = 0",
  2: "g_SE = softmax(wg · x_S)",
  3: "m_E = mean(g_SE, over S)",
  4: "for s in 1..S:",
  5: "  g1,e1,g2,e2 = top_2(g_SE[s])",
  6: "  g1 = g1 / (g1 + g2)",
  7: "  c = c_E[e1]",
  8: "  if c < C:",
  9: "    G[s,e1] = g1",
  11: "  c_E[e1] = c + 1",
  13: "l_aux = 1/E · Σ_e (c_E[e]/S) · m_E[e]",
  14: "for s in 1..S:",
  15: "  g1,e1,g2,e2 = top_2(g_SE[s])",
  16: "  g2 = g2 / (g1 + g2)",
  17: "  rnd = uniform(0, 1)",
  18: "  c = c_E[e2]",
  19: "  if c < C and 2·g2 > rnd:",
  20: "    G[s,e2] = g2",
  22: "  c_E[e2] = c + 1",
};

const A2 = {
  1: 'gates = softmax(einsum("GSM,ME->GSE", inputs, wg))',
  2: "combine_weights, dispatch_mask = Top2Gating(gates)",
  3: 'dispatched = einsum("GSEC,GSM->EGCM", dispatch_mask, inputs)',
  5: 'h = einsum("EGCM,EMH->EGCH", dispatched, wi)',
  6: "h = relu(h)",
  7: 'expert_out = einsum("EGCH,EHM->GECM", h, wo)',
  8: 'outputs = einsum("GSEC,GECM->GSM", combine_weights, expert_out)',
  s1: "inputs = split(inputs, 0, D)     # shard on G",
  s2: "wg = replicate(wg)",
  s3: "dispatched = split(dispatched, 0, D)  # reshard G → E",
};

const pick = (src, keys) => keys.map((k) => ({ n: k, t: src[k] }));
const t = (name, dims, extra = {}) => ({ name, dims, ...extra });
const ids = (arr) => arr.map((x) => `x${x}`).join(", ") || "none";

export const FLOW = [
  { id: "split", label: "split into groups", sub: "x[N,M] → [G,S,M]" },
  {
    group: "Top2Gating · Algorithm 1",
    items: [
      { id: "softmax", label: "softmax gates + mean", sub: "g_SE, m_E" },
      { id: "top2", label: "pick top-2", sub: "e1, e2 per token" },
      { id: "pass1", label: "for s: fill 1st choice", sub: "counters c_E vs C", loop: true },
      { id: "aux", label: "balance loss", sub: "ℓ_aux" },
      { id: "pass2", label: "for s: random 2nd choice", sub: "dice + capacity", loop: true },
      { id: "weights", label: "combine_weights", sub: "[G,S,E,C]" },
    ],
  },
  { id: "dispatch", label: "dispatch einsum", sub: "all-to-all  G → E" },
  { id: "ffn", label: "expert FFN", sub: "each device, own tokens" },
  { id: "combine", label: "all-to-all back", sub: "E → G" },
  { id: "output", label: "weighted sum", sub: "y = g1·FFN + g2·FFN" },
];

export const STEPS = [
  {
    id: "split", block: "split", scene: "devices", stage: 0,
    caption: `${N} tokens → ${G} groups of ${S}, one group per device`,
    code: pick(A2, ["s1", "s2"]),
    tensors: {
      ins: [t("inputs", ["G", "S", "M"], { shard: "G" }), t("wg", ["M", "E"], { repl: true })],
      op: "split(inputs, 0, D)  ·  replicate(wg)",
    },
    vars: () => [["N", `${N}`], ["G", `${G}`], ["S", `${S}`], ["C = 2S/E", `${C}`]],
  },
  {
    id: "gates", block: "softmax", scene: "group", stage: 0,
    caption: "every token scores every expert",
    code: pick(A1, [2]),
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
    code: pick(A1, [3]),
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
    code: pick(A1, [5, 6]),
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
    code: pick(A1, [4, 5, 7, 8, 9, 11]),
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
    code: pick(A1, [13]),
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
    code: pick(A1, [14, 16, 17, 18, 19, 20, 22]),
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
    code: pick(A2, [2]),
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
    code: pick(A2, [3, "s3"]),
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
    code: pick(A2, [5, 6, 7]),
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
    code: pick(A2, [7]),
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
    code: pick(A2, [8]),
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
