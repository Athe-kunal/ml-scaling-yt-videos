import { LB } from "./lbModel";

const r = (name, dims, what, extra = {}) => ({ name, dims, what, ...extra });
const IN = { io: "in" }, OUT = { io: "out" };

// ---- DeepSeek-V3 -----------------------------------------------------------
export const DS_GLOSSARY = {
  T: ["tokens in the sequence", LB.T],
  N: ["routed experts", LB.n],
  K: ["experts kept per token (top-K)", LB.k],
  d: ["hidden size of a token / expert centroid", null],
};

const S_ = r("s", ["T", "N"], "s[t,i] = sigmoid(u_t · e_i): affinity of token t for expert i; each score is independent, they do not sum to 1");
const IDX = r("idx", ["T", "K"], "ids of the K experts each token keeps");
const G_ = r("g", ["T", "K"], "gate weights of the kept experts: s / Σ s over the K chosen");
const LOAD = r("load", ["N"], "tokens routed to each expert in the batch; fair share is T·K/N");
const B_ = r("b", ["N"], "one bias per expert; positive pulls tokens in, negative pushes them away");

export const DS_SHAPES = {
  scores: [r("u", ["T", "d"], "token vectors entering the MoE layer", IN), r("e", ["N", "d"], "one learned centroid per expert", IN), { ...S_, ...OUT }],
  topk: [{ ...S_, io: "in" }, { ...IDX, io: "out" }],
  gate: [{ ...S_, io: "in" }, { ...IDX, io: "in" }, { ...G_, io: "out" }],
  skew: [{ ...IDX, io: "in" }, { ...LOAD, io: "out", what: "count of routed tokens per expert: a few experts hog most of them" }],
  f: [{ ...LOAD, io: "in" }, r("f", ["N"], "f_i = N/(K·T) · count_i: 1.0 means exactly the fair share, 2.0 means double", OUT)],
  P: [{ ...S_, io: "in" }, r("P", ["N"], "P_i = mean over tokens of s_i / Σ_j s_j: the smooth, differentiable share of router affinity; sums to 1", OUT)],
  L: [r("f", ["N"], "load share (not differentiable, a constant to the optimiser)", IN), r("P", ["N"], "affinity share (carries the gradient)", IN), r("L_bal", [], "α · Σ f_i·P_i: smallest when both are uniform", OUT)],
  select: [{ ...B_, io: "in" }, r("s + b", ["T", "N"], "biased scores: used only to pick the top-K", IN), { ...IDX, io: "out", what: "chosen with s + b" }, { ...G_, io: "out", what: "weights come from the raw s, so b adds no gradient" }],
  monitor: [{ ...IDX, io: "in" }, { ...LOAD, io: "out", what: "counted over the whole batch every training step" }],
  err: [{ ...LOAD, io: "in" }, r("err", ["N"], "mean(load) − load: positive means the expert is starved", OUT)],
  update: [{ ...LOAD, io: "in" }, { ...B_, io: "in" }, r("b'", ["N"], "b + γ·sign(mean(load) − load): overloaded experts go down, underloaded go up", OUT)],
  loop: [{ ...B_, io: "in", what: "after t updates; watch the bars move" }, { ...LOAD, io: "in" }, r("CV", [], "std(load)/mean(load): 0 means perfectly flat", OUT)],
  recipe: [r("loss", [], "lm_loss + α·L_bal with a tiny α; the bias does the heavy lifting, the loss only guards each single sequence", OUT)],
};

// ---- Kimi K3 quantile balancing ----------------------------------------------
export const KIMI_GLOSSARY = {
  B: ["tokens in the batch", LB.m],
  N: ["routed experts", LB.n],
  k: ["experts kept per token", LB.k],
  R: ["data-parallel ranks", LB.ranks],
  "k+1": ["top-(k+1) scores kept per token", LB.k + 1],
  L: ["histogram bins per expert", 16],
};

const SB = r("s + b", ["B", "N"], "biased score of every token for every expert");
const AL = r("α", ["B"], "per token: its (k+1)-th largest biased score, the line an expert must beat");
const MG = r("margin", ["B", "N"], "margin[t,j] = s[t,j] − α[t]: above 0 means expert j is in token t's top-k");
const BH = r("b̂", ["N"], "raw bias for each expert before centring");
const KB = r("b", ["N"], "final bias: b̂ with its mean removed");

export const KIMI_SHAPES = {
  limit: [r("load", ["N"], "tokens per expert each step; with a fixed γ it either crawls or overshoots", IN), r("b", ["N"], "b += γ·sign(mean − load): the same step size for every expert", OUT)],
  topk1: [{ ...SB, io: "in" }, r("top-(k+1) scores", ["B", "k+1"], "each token's k+1 largest biased scores, sorted", OUT)],
  cutoff: [{ ...SB, io: "in" }, { ...AL, io: "out" }],
  margin: [{ ...SB, io: "in" }, { ...AL, io: "in" }, { ...MG, io: "out" }],
  q: [r("q", [], "= 1 − k/n: the quantile level; for K3 that is 1 − 16/896", OUT)],
  quantile: [r("margin[:, j]", ["B"], "one expert's margins over the whole batch", IN), r("b̂_j", [], "−quantile at 1 − k/n of that column: shifts the column until k/n of tokens clear the cutoff", OUT)],
  center: [{ ...BH, io: "in" }, { ...KB, io: "out", what: "subtracting a constant from every expert changes no top-k choice" }],
  hist: [{ ...MG, io: "in" }, r("hist (per rank)", ["R", "N", "L"], "each rank counts its own tokens' margins into bins", OUT)],
  reduce: [r("hist (per rank)", ["R", "N", "L"], "one copy per rank", IN), r("hist (pooled)", ["N", "L"], "sum over ranks by one all-reduce: traffic is N·L numbers, not B·N", OUT)],
  bhat: [r("hist (pooled)", ["N", "L"], "bin counts over the whole batch", IN), { ...BH, io: "out", what: "quantile read off the pooled histogram" }],
  apply: [{ ...KB, io: "in", what: "computed from batch t, used on batch t+1" }, r("load", ["N"], "tokens per expert on the next batch", OUT)],
  race: [{ ...KB, io: "in" }, r("load", ["N"], "tokens per expert", OUT), r("CV", [], "std(load)/mean(load); compare quantile with the ± γ rule", OUT)],
};

// ---- DeepSeek-V3 node-limited routing + dispatch shuffle --------------------
import { DEV } from "./devRoute";

export const DEV_GLOSSARY = {
  E: ["routed experts in this toy", DEV.E],
  D: ["devices (nodes in V3)", DEV.D],
  Ed: ["experts per device", DEV.Ed],
  K: ["routed experts per token", DEV.K],
  M: ["max devices a token may use", DEV.M],
  "K/M": ["experts summed into a device score", DEV.K / DEV.M],
  B: ["tokens in the batch", DEV.B],
  d: ["hidden size of a token", null],
  C: ["token copies one device receives", null],
};

const SD = r("s", ["E"], "affinity of one token for each routed expert (sigmoid scores)");
export const DEV_SHAPES = {
  dscore: [{ ...SD, io: "in" }, r("dev_score", ["D"], "per device: sum of the K/M highest affinities among its Ed experts", OUT)],
  dtop: [r("dev_score", ["D"], "score of each device", IN), r("keep", ["M"], "ids of the M best devices", OUT)],
  dmask: [{ ...SD, io: "in" }, r("keep", ["M"], "devices allowed for this token", IN), r("s_dev", ["E"], "s with every expert outside the kept devices set to −∞", OUT)],
  dpick: [r("s_dev", ["E"], "masked affinities", IN), r("idx", ["K"], "the K chosen experts: all live on at most M devices", OUT), r("g", ["K"], "gates s / Σ s over the chosen experts", OUT)],
  dest: [r("idx", ["B", "K"], "chosen experts of every token", IN), r("dest", ["B", "M"], "devices each token must visit: at most M of them", OUT)],
  a2a: [r("x", ["B", "d"], "tokens, still on their home devices", IN), r("recv", ["C", "d"], "token copies now sitting on the device that hosts their expert", OUT, "all-to-all")],
  dffn: [r("recv", ["C", "d"], "copies this device received", IN), r("out", ["C", "d"], "expert outputs for those copies; no communication", OUT)],
  dcomb: [r("out", ["C", "d"], "outputs on the experts' devices", IN), r("y", ["B", "d"], "back on the home device: Σ g_i · out_i, plus the shared expert and the residual", OUT, "all-to-all")],
};
