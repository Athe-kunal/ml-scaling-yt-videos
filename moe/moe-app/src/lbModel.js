// Toy router shared by the DeepSeek-V3 and Kimi K3 tabs.
// n experts, top-k routing, sigmoid affinities s = sigmoid(logit). Logits carry a
// fixed popularity skew so an untrained router overloads the first experts.

export const LB = { n: 4, k: 2, m: 64, T: 8, ranks: 4 }; // small on purpose: every number fits on screen
export const XC = ["#6EF3DD", "#FF8AC6", "#FFC94D", "#8AB4FF", "#B69CFF", "#7EE081", "#FF9E6E", "#C7D1DD"];
const POP = [1.0, 0.45, -0.1, -0.7]; // one popularity offset per expert (length n)

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randn = (r) => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sum = (a) => a.reduce((x, y) => x + y, 0);

// m tokens x n experts affinity matrix; one seed = one batch
export function scoresOf(seed, m = LB.m) {
  const r = rng(seed * 7919 + 13);
  return Array.from({ length: m }, () => POP.map((p) => sigmoid(p + 0.9 * randn(r))));
}

// Top-(k+1) on the biased score: first k are the routes, entry k+1 is the cutoff.
export function route(scores, b, k = LB.k) {
  return scores.map((s) => {
    const biased = s.map((x, j) => x + b[j]);
    const order = biased.map((v, j) => j).sort((x, y) => biased[y] - biased[x]);
    return { sel: order.slice(0, k), cutoff: biased[order[k]], biased };
  });
}

export function loadsOf(routes, n = LB.n) {
  const l = Array(n).fill(0);
  routes.forEach((r) => r.sel.forEach((j) => (l[j] += 1)));
  return l;
}

export const cvOf = (l) => {
  const mu = sum(l) / l.length;
  return Math.sqrt(sum(l.map((x) => (x - mu) ** 2)) / l.length) / mu;
};

// DeepSeek-V3: b_j += gamma * sign(mean load - load_j)
export function signStep(b, load, gamma) {
  const mu = sum(load) / load.length;
  return b.map((x, j) => x + gamma * Math.sign(mu - load[j]));
}

export function quantile(arr, q) {
  const s = [...arr].sort((x, y) => x - y);
  const pos = q * (s.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// Kimi K3 Quantile Balancing: b^_j = -quantile_{1-k/n}(s_:j - alpha), then mean-centre.
export function qbStep(scores, b, k = LB.k, n = LB.n) {
  const rt = route(scores, b, k);
  const margins = scores.map((s, i) => s.map((x) => x - rt[i].cutoff)); // m x n
  const q = 1 - k / n;
  const thr = Array.from({ length: n }, (_, j) => quantile(margins.map((row) => row[j]), q));
  const bhat = thr.map((t) => -t);
  const mu = sum(bhat) / n;
  return { rt, margins, thr, bhat, b: bhat.map((x) => x - mu) };
}

// training-loop simulation. record[t] = what batch t saw (bias used, loads) then the bias after the update.
export function simulate(kind, steps, gamma = 0.01) {
  let b = Array(LB.n).fill(0);
  const rec = [];
  for (let t = 0; t < steps; t++) {
    const sc = scoresOf(t + 1);
    const rt = route(sc, b);
    const loads = loadsOf(rt);
    const next = kind === "qb" ? qbStep(sc, b).b : signStep(b, loads, gamma);
    rec.push({ t, b: [...b], loads, cv: cvOf(loads), next });
    b = next;
  }
  return rec;
}

// DeepSeek-V3 sequence-wise balance loss (eq. 17-20) on one sequence of T tokens, no bias
export function auxTerms(scores, k = LB.k) {
  const T = scores.length, n = scores[0].length;
  const rt = route(scores, Array(n).fill(0), k);
  const cnt = loadsOf(rt, n);
  const f = cnt.map((c) => (n / (k * T)) * c);
  const P = Array.from({ length: n }, (_, j) => sum(scores.map((s) => s[j] / sum(s))) / T);
  const prod = f.map((x, j) => x * P[j]);
  return { rt, cnt, f, P, prod, loss: sum(prod) };
}
