import { CFG } from "./moeModel";

export const GLOSSARY = {
  G: ["groups; one per device", CFG.G],
  S: ["tokens in one group", CFG.S],
  E: ["experts; one per device", CFG.E],
  C: ["expert capacity: buffer slots per group, 2S/E", CFG.C],
  M: ["model dimension of a token vector", null],
  H: ["hidden width inside an expert FFN", null],
};

// what each tensor holds, in words
const WHAT = {
  inputs: "token vectors; group g is the S tokens that live on device g",
  wg: "gate weights: one score direction per expert",
  gates: "gates[g,s,e] = how much token s of group g wants expert e (softmax, each row sums to 1)",
  m_E: "mean gate per expert in a group: a smooth stand-in for how much the router likes each expert",
  "g1, e1, g2, e2": "per token: weights and ids of its two best experts (g1+g2 = 1 after normalising)",
  "e1, g1": "each token's first choice: expert id and weight",
  "e2, g2": "each token's second choice: expert id and weight",
  c_E: "bucket fill: how many tokens each expert already took in this group (compared with C)",
  combine_weights: "[g,s,e,c] = gate weight if token s sits in slot c of expert e, else 0; ≤ 2 non-zeros per token",
  dispatch_mask: "same pattern as combine_weights but 0/1: which slot each token flies to",
  "ℓ_aux": "one number: larger when the busy experts are also the ones the router likes",
  dispatched: "expert e's input buffer: C slots from each of the G groups",
  "wi, wo": "expert weights: M → H and H → M, one set per expert (sharded on E)",
  expert_out: "what each expert computed for its slots",
  outputs: "layer output; same shape as inputs, dropped tokens just keep their residual",
};

const SHARD_TXT = (r) =>
  r.comm ? r.comm : r.repl ? "replicated" : r.shard ? `split on ${r.shard}` : null;

export function shapesOf(step) {
  const t = step.tensors;
  const row = (io) => (r) => ({
    io, name: r.name, dims: r.dims, what: r.note && r.name === "ℓ_aux" ? WHAT["ℓ_aux"] : WHAT[r.name],
    note: SHARD_TXT(r),
  });
  return [...(t.ins ?? []).map(row("in")), ...(t.outs ?? []).map(row("out"))];
}
