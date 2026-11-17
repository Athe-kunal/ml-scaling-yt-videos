// Shared math for both tabs: procedural images, patch statistics, a toy
// "plausible" attention model on real pixels, and exact shape/param/MAC formulas.

export function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- images ----------
// an emoji subject on a two-tone gradient with a "ground" band: enough structure
// that patches differ and the toy attention has an object to find
export const SUBJECTS = [
  { id: "dog", emoji: "🐶", bg: ["#2a5d8f", "#9fd3f5"], ground: "#4f8a3a", caption: "a photo of a dog" },
  { id: "cat", emoji: "🐱", bg: ["#5b3b7a", "#e7b6f0"], ground: "#8a6a4a", caption: "a cat sitting on a rug" },
  { id: "car", emoji: "🚗", bg: ["#d97a3a", "#ffe0a8"], ground: "#555b66", caption: "a red car on the road" },
  { id: "pizza", emoji: "🍕", bg: ["#1f4f4a", "#8fe0c8"], ground: "#6b4a2f", caption: "a slice of pizza" },
  { id: "rocket", emoji: "🚀", bg: ["#0b1230", "#3a4a8f"], ground: "#2b2b3b", caption: "a rocket in space" },
  { id: "tree", emoji: "🌳", bg: ["#6fa8dc", "#d8ecff"], ground: "#6a9a3a", caption: "a tree in a park" },
  { id: "guitar", emoji: "🎸", bg: ["#3a1f1f", "#b0605a"], ground: "#2a1a10", caption: "a guitar on stage" },
  { id: "bird", emoji: "🐦", bg: ["#3f7f9f", "#bfe8f5"], ground: "#5a7f3a", caption: "a small bird" },
];

const cache = new Map();
export function renderSubject(subject, size) {
  const key = `${subject.id}@${size}`;
  if (cache.has(key)) return cache.get(key);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const x = c.getContext("2d");
  const g = x.createLinearGradient(0, 0, 0, size);
  g.addColorStop(0, subject.bg[0]);
  g.addColorStop(1, subject.bg[1]);
  x.fillStyle = g;
  x.fillRect(0, 0, size, size);
  x.fillStyle = subject.ground;
  x.fillRect(0, size * 0.78, size, size * 0.22);
  x.font = `${Math.round(size * 0.58)}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
  x.textAlign = "center";
  x.textBaseline = "middle";
  x.fillText(subject.emoji, size * 0.52, size * 0.5);
  const img = x.getImageData(0, 0, size, size);
  const out = { canvas: c, data: img.data, size };
  cache.set(key, out);
  return out;
}

// per-patch mean colour (0..1), row/col, and saliency = distance from the
// border colour (the "background"), normalised to 0..1
export function patchStats(img, P) {
  const g = Math.floor(img.size / P), N = g * g, d = img.data, S = img.size;
  const feats = [];
  for (let r = 0; r < g; r++)
    for (let c = 0; c < g; c++) {
      let R = 0, G = 0, B = 0;
      for (let y = r * P; y < (r + 1) * P; y++)
        for (let x = c * P; x < (c + 1) * P; x++) {
          const o = (y * S + x) * 4;
          R += d[o]; G += d[o + 1]; B += d[o + 2];
        }
      const k = P * P * 255;
      feats.push({ r: R / k, g: G / k, b: B / k, row: r, col: c });
    }
  const border = feats.filter((f) => f.row === 0 || f.col === 0 || f.row === g - 1 || f.col === g - 1);
  const bgc = ["r", "g", "b"].map((ch) => border.reduce((a, f) => a + f[ch], 0) / border.length);
  // compare to the patch directly above the edge as well, so the gradient sky is not "salient"
  feats.forEach((f) => {
    const sameRow = feats[f.row * g]; // left-edge patch of this row
    const dRow = Math.hypot(f.r - sameRow.r, f.g - sameRow.g, f.b - sameRow.b);
    const dAll = Math.hypot(f.r - bgc[0], f.g - bgc[1], f.b - bgc[2]);
    f.sal = Math.min(dRow, dAll * 0.6);
  });
  const mx = Math.max(1e-6, ...feats.map((f) => f.sal));
  feats.forEach((f) => (f.sal /= mx));
  return { g, N, feats };
}

// ---------- toy attention ----------
export const HEADS = [
  { id: "local", label: "head 1 · local", hint: "attends to nearby patches" },
  { id: "color", label: "head 2 · similar", hint: "attends to patches that look alike" },
  { id: "object", label: "head 3 · object", hint: "attends to the subject" },
  { id: "global", label: "head 4 · global", hint: "spreads out almost evenly" },
];

// raw score s(i, j) for token i (0 = CLS) attending to j; `depth` 0..1 sharpens
// the object head and blurs the local one, like deeper layers do
function scorer(head, feats, g, depth) {
  const sig = Math.max(0.8, g / 12) * (1 + depth * 2);
  const rnd = mulberry(7);
  const noise = head === "global" ? Float32Array.from({ length: feats.length + 1 }, () => rnd()) : null;
  return (i, j) => {
    if (j === 0) return i === 0 ? 1 : 0.5; // everyone glances at CLS a little
    const b = feats[j - 1];
    if (i === 0) {
      if (head === "object") return b.sal * (5 + depth * 6);
      if (head === "global") return noise[j] * 0.6;
      if (head === "color") return b.sal * 2.5;
      return b.sal * 1.2;
    }
    const a = feats[i - 1];
    if (head === "local") return -((a.row - b.row) ** 2 + (a.col - b.col) ** 2) / (2 * sig * sig);
    if (head === "color") return -((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2) * 18;
    if (head === "object") return b.sal * (3 + depth * 5);
    return noise[j] * 0.6;
  };
}

// full [T, T] softmaxed matrix for one head
export function attention(head, feats, g, depth = 0.5) {
  const Tn = feats.length + 1, s = scorer(head, feats, g, depth);
  const A = new Float32Array(Tn * Tn), row = new Float32Array(Tn);
  for (let i = 0; i < Tn; i++) {
    let m = -Infinity;
    for (let j = 0; j < Tn; j++) { row[j] = s(i, j); if (row[j] > m) m = row[j]; }
    let z = 0;
    for (let j = 0; j < Tn; j++) { row[j] = Math.exp(row[j] - m); z += row[j]; }
    for (let j = 0; j < Tn; j++) A[i * Tn + j] = row[j] / z;
  }
  return A;
}

// only the CLS row (cheap), as a length-N array over patches
export function clsRow(head, feats, g, depth = 0.5) {
  const s = scorer(head, feats, g, depth), Tn = feats.length + 1;
  const v = Array.from({ length: Tn }, (_, j) => s(0, j));
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
  return e.slice(1).map((x) => x / z);
}

// ---------- ViT config + exact counts ----------
export const VIT_B = { name: "ViT-B", D: 768, L: 12, h: 12, M: 3072, K: 1000 };

export function vitDims(res, P, cfg = VIT_B) {
  const g = Math.floor(res / P), N = g * g, Tn = N + 1, { D, L, h, M, K } = cfg, pc = P * P * 3;
  const perLayer = 4 * D + (3 * D * D + 3 * D) + (D * D + D) + (D * M + M) + (M * D + D);
  const params = pc * D + D + D + Tn * D + L * perLayer + 2 * D + D * K + K;
  const macsLayer = Tn * (4 * D * D + 2 * D * M) + 2 * Tn * Tn * D;
  const macs = N * pc * D + L * macsLayer + D * K;
  const attnFrac = (2 * Tn * Tn * D) / macsLayer;
  return { res, P, g, N, Tn, D, L, h, dh: D / h, M, K, pc, params, macs, attnFrac, attnEntries: Tn * Tn };
}

// ---------- CLIP ----------
export const CLIP_MODELS = {
  "ViT-B/32": { P: 32, Di: 768, Li: 12, hi: 12, Dt: 512, Lt: 12, ht: 8, E: 512 },
  "ViT-B/16": { P: 16, Di: 768, Li: 12, hi: 12, Dt: 512, Lt: 12, ht: 8, E: 512 },
  "ViT-L/14": { P: 14, Di: 1024, Li: 24, hi: 16, Dt: 768, Lt: 12, ht: 12, E: 768 },
};
export const CTX = 77, VOCAB = 49408, RES = 224;


const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

// toy embeddings: each class has a text direction; an image = mix of its text
// direction and its own noise, the mix weight grows with training progress
export function clipToy(n, train, d = 1024) {
  const r = mulberry(42);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
  const txt = Array.from({ length: SUBJECTS.length }, () => unit(Array.from({ length: d }, gauss)));
  const noise = Array.from({ length: SUBJECTS.length }, () => unit(Array.from({ length: d }, gauss)));
  const a = 0.003 + 0.16 * train ** 1.6;
  const img = txt.map((t, i) => unit(t.map((x, k) => a * x + (1 - a) * noise[i][k])));
  const scale = Math.exp(Math.log(1 / 0.07) + train ** 2 * (Math.log(100) - Math.log(1 / 0.07))); // reaches the 100 cap at train = 1
  const I = img.slice(0, n), Tt = txt.slice(0, n);
  const cos = I.map((u) => Tt.map((v) => dot(u, v)));
  const logits = cos.map((row) => row.map((x) => x * scale));
  const sm = (v) => { const m = Math.max(...v), e = v.map((x) => Math.exp(x - m)), z = e.reduce((p, q) => p + q, 0); return e.map((x) => x / z); };
  const pRow = logits.map(sm);
  const colsT = logits[0].map((_, j) => sm(logits.map((row) => row[j])));
  const lossRow = pRow.map((p, i) => -Math.log(p[i]));
  const lossCol = colsT.map((p, j) => -Math.log(p[j]));
  const mean = (v) => v.reduce((p, q) => p + q, 0) / v.length;
  const li = mean(lossRow), lt = mean(lossCol);
  return { img: I, txt: Tt, allTxt: txt, allImg: img, cos, logits, scale, pRow, pCol: colsT, lossRow, lossCol, li, lt, loss: (li + lt) / 2, d, sm, dot };
}

export const fmtN = (x) => (x >= 1e9 ? `${(x / 1e9).toFixed(1)}B` : x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(1)}K` : `${x}`);
export const fmtInt = (x) => x.toLocaleString("en-US");

// ---------- SigLIP ----------
// both towers have the same width; the image tower has no CLS (MAP head instead),
// text context is 64 SentencePiece tokens, vocab 32,000
export const SIGLIP_MODELS = {
  "B/16": { name: "SigLIP B/16 @224", res: 224, P: 16, D: 768, L: 12, h: 12, M: 3072 },
  "So400m/14": { name: "SigLIP So400m/14 @384", res: 384, P: 14, D: 1152, L: 27, h: 16, M: 4304 },
};
export const S_CTX = 64, S_VOCAB = 32000;

// the toy for SigLIP: the trained cosines of the CLIP toy, with toy t and b.
// cell loss ℓ_ij = −log σ(z_ij · L_ij), z = +1 on the diagonal, −1 elsewhere
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export function sigToy(n, t = 30, b = -4) {
  const base = clipToy(n, 1);
  const cos = base.cos;
  const logits = cos.map((r) => r.map((c) => t * c + b));
  const z = logits.map((r, i) => r.map((_, j) => (i === j ? 1 : -1)));
  const p = logits.map((r) => r.map(sigmoid)); // σ(L) = P(match)
  const ell = logits.map((r, i) => r.map((l, j) => -Math.log(sigmoid(z[i][j] * l))));
  const rowSum = ell.map((r) => r.reduce((a, c) => a + c, 0));
  const pos = ell.reduce((a, r, i) => a + r[i], 0);
  const total = rowSum.reduce((a, c) => a + c, 0);
  return { n, t, b, cos, logits, z, p, ell, rowSum, pos, neg: total - pos, total, loss: total / n, allImg: base.allImg, allTxt: base.allTxt, dot: base.dot };
}

// ---------- SigLIP 2 NaFlex ----------
// HF Siglip2ImageProcessor.get_image_size_for_max_num_patches: binary-search the
// largest scale whose (ceil(h·s/p)·p) × (ceil(w·s/p)·p) grid fits the token budget.
// Aspect ratio is kept (up to rounding to multiples of p).
export function naflexSize(h, w, p = 16, maxPatches = 256, eps = 1e-5) {
  const sz = (s, x) => Math.max(p, Math.ceil((x * s) / p) * p);
  let lo = eps / 10, hi = 100;
  while (hi - lo >= eps) {
    const s = (lo + hi) / 2;
    if ((sz(s, h) / p) * (sz(s, w) / p) <= maxPatches) lo = s; else hi = s;
  }
  const H = sz(lo, h), W = sz(lo, w);
  return { H, W, gh: H / p, gw: W / p, N: (H / p) * (W / p) };
}
export const NAFLEX_BUDGETS = [128, 256, 576, 784, 1024];
