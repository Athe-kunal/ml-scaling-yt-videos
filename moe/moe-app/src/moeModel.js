// Deterministic toy MoE layer: G groups (one per device), S tokens per group,
// E experts (one per device). Algorithm 1 of GShard (2006.16668) is
// implemented literally so every counter the UI shows is the paper's c_E.

export const CFG = { G: 4, S: 6, E: 4 };
CFG.C = Math.round((2 * CFG.S) / CFG.E); // per-group expert capacity 2S/E
CFG.N = CFG.G * CFG.S;

export const EXPERT_COLORS = ["#6EF3DD", "#FF8AC6", "#FFC94D", "#8AB4FF"];
export const SEED = 66;
const BIAS = [1.0, 0.5, 0, -0.3]; // skewed router: experts 0 and 1 are popular

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const randn = (r) => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());

function softmax(v) {
  const m = Math.max(...v);
  const ex = v.map((x) => Math.exp(x - m));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / z);
}

// Algorithm 1: group-level top-2 gating with auxiliary loss.
export function top2Gating(gates, C, rnds) {
  const S = gates.length;
  const E = gates[0].length;
  const c = Array(E).fill(0); // line 1
  const m = Array.from({ length: E }, (_, e) => gates.reduce((a, row) => a + row[e], 0) / S); // line 3

  const toks = gates.map((row, s) => {
    const order = row.map((g, e) => [g, e]).sort((a, b) => b[0] - a[0]);
    const [g1, e1] = order[0];
    const [g2, e2] = order[1];
    return { s, g1, e1, g2, e2, g1n: g1 / (g1 + g2), g2n: g2 / (g1 + g2), rnd: rnds[s], t1: null, t2: null };
  });

  // lines 4-12: first choice. c_e1 is bumped even when the token overflows.
  const afterPass1 = [];
  for (const t of toks) {
    const pos = c[t.e1];
    t.t1 = { e: t.e1, pos, ok: pos < C, w: t.g1n };
    c[t.e1] = pos + 1;
  }
  afterPass1.push(...c);

  const aux = (1 / E) * c.reduce((a, ce, e) => a + (ce / S) * m[e], 0); // line 13
  const cPass1 = [...c];

  // lines 14-23: second choice with random routing.
  for (const t of toks) {
    const pos = c[t.e2];
    const capOk = pos < C;
    const diceOk = 2 * t.g2n > t.rnd;
    t.t2 = { e: t.e2, pos, ok: capOk && diceOk, capOk, diceOk, w: t.g2n, rnd: t.rnd };
    c[t.e2] = pos + 1;
  }

  toks.forEach((t) => {
    t.final = [t.t1, t.t2].filter((x) => x.ok).map((x) => ({ e: x.e, w: x.w, slot: x.pos, second: x === t.t2 }));
  });

  return { gates, toks, m, cPass1, cFinal: [...c], aux, C };
}

// E and C default to the GShard toy (4 experts); the all-to-all tab passes a
// larger E to put several experts on each device.
export function buildModel(seed = SEED, E = CFG.E, C = CFG.C) {
  const r = rng(seed);
  const groups = Array.from({ length: CFG.G }, () => {
    const gates = Array.from({ length: CFG.S }, () =>
      softmax(Array.from({ length: E }, (_, e) => BIAS[e % BIAS.length] + 1.1 * randn(r))),
    );
    const rnds = Array.from({ length: CFG.S }, () => r());
    return top2Gating(gates, C, rnds);
  });
  // tickets that really reach an expert buffer
  const load = Array(E).fill(0);
  groups.forEach((g) => g.toks.forEach((t) => t.final.forEach((f) => (load[f.e] += 1))));
  return { groups, load };
}

export const MODEL = buildModel();
