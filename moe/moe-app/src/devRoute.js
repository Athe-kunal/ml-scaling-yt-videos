// Toy for DeepSeek-V3 node-limited routing (arXiv 2412.19437, §2.1.2 + §3.2.2):
// E routed experts live on D devices (Ed per device). Each token may use at most
// M devices, picked by the sum of the top K/M affinities of the experts on each
// device; the top-K experts are then taken among the experts of those devices.

export const DEV = { E: 8, D: 4, Ed: 2, K: 4, M: 2, B: 4, SEED: 4 };
export const devOf = (e) => Math.floor(e / DEV.Ed);

function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function devScores(seed) {
  const r = mulberry(seed);
  const randn = () => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
  return Array.from({ length: DEV.B }, () => Array.from({ length: DEV.E }, () => 1 / (1 + Math.exp(-1.3 * randn()))));
}

const topIdx = (vals, ids, n) => ids.map((e) => [vals[e], e]).sort((a, b) => b[0] - a[0]).slice(0, n).map(([, e]) => e);

export function routeDev(row, restricted = true) {
  const per = DEV.K / DEV.M;
  const devExperts = Array.from({ length: DEV.D }, (_, d) => Array.from({ length: DEV.Ed }, (_, j) => d * DEV.Ed + j));
  const top = devExperts.map((ids) => topIdx(row, ids, per)); // experts that feed each device score
  const devScore = top.map((ids) => ids.reduce((a, e) => a + row[e], 0));
  const keep = topIdx(devScore, devExperts.map((_, d) => d), DEV.M);
  const eligible = restricted ? keep.flatMap((d) => devExperts[d]) : Array.from({ length: DEV.E }, (_, e) => e);
  const idx = topIdx(row, eligible, DEV.K).sort((a, b) => a - b);
  const sum = idx.reduce((a, e) => a + row[e], 0);
  return { top, devScore, keep, eligible, idx, g: idx.map((e) => row[e] / sum), devs: [...new Set(idx.map(devOf))].sort() };
}
