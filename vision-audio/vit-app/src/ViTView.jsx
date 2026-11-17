import { useMemo, useState } from "react";
import { T, mono } from "./theme";
import { Tex } from "./tex";
import { ease, useProgress } from "./anim";
import StepShell, { Seg, Stat } from "./StepShell";
import { Canvas, Heatmap, PatchImage, Mat, Op, Row, Label, Bars, heatCss, MM, MMNote, MatMulCard, BlockMMs, DimCtx } from "./draw";
import { SUBJECTS, renderSubject, patchStats, attention, clsRow, HEADS, vitDims, mulberry, fmtN, fmtInt } from "./model";

// ViT-B walkthrough: image -> patches -> tokens (+CLS, +pos) -> encoder block
// internals -> ×L -> CLS -> head -> loss. Patch size / resolution / head are live.

const CODE = {
  image: ["x = load_image()   # [B, 3, H, W]"],
  patchify: ["p = x.unfold(2, P, P).unfold(3, P, P)   # [B, 3, g, g, P, P]", "p = p.permute(0, 2, 3, 4, 5, 1)   # [B, g, g, P, P, 3]"],
  flatten: ["p = p.reshape(B, N, P*P*3)   # [B, N, P²·3]  raster order"],
  embed: ["tok = p @ W_E + b_E   # [B,N,P²·3]·[P²·3,D] → [B,N,D]", "# equivalent to nn.Conv2d(3, D, kernel_size=P, stride=P): same weights, rearranged"],
  cls: ["cls = self.cls_token.expand(B, -1, -1)   # learnable [1,1,D] → [B,1,D]", "x = torch.cat([cls, tok], dim=1)   # [B, N+1, D]"],
  pos: ["x = x + self.pos_embed   # learnable [1, N+1, D], broadcast over B"],
  ln1: ["h = self.norm1(x)   # LayerNorm over D, per token  [B,T,D]"],
  qkv: ["qkv = h @ W_qkv + b_qkv   # [B,T,D]·[D,3D] → [B,T,3D]", "q, k, v = qkv.reshape(B,T,3,heads,d).permute(2,0,3,1,4)", "# each → [B, heads, T, d],  d = D / heads"],
  attn: ["A = (q @ k.transpose(-2, -1)) * d ** -0.5   # [B,h,T,d]·[B,h,d,T] → [B,h,T,T]", "A = A.softmax(dim=-1)   # each row sums to 1"],
  mix: ["o = A @ v   # [B,h,T,T]·[B,h,T,d] → [B,h,T,d]", "o = o.transpose(1, 2).reshape(B, T, D)   # concat heads", "x = x + o @ W_O + b_O   # [D,D] projection, residual add"],
  mlp: ["h = self.norm2(x)", "h = gelu(h @ W_1 + b_1)   # [B,T,D] → [B,T,4D]", "x = x + h @ W_2 + b_2   # [B,T,4D] → [B,T,D], residual"],
  stack: ["for blk in self.blocks:   # × L", "    x = blk(x)   # [B,T,D] → [B,T,D]  shape never changes"],
  clsout: ["x = self.norm(x)", "z = x[:, 0]   # [B, D]  keep only the CLS token"],
  head: ["logits = z @ W_head + b   # [B,D]·[D,K] → [B,K]", "probs = logits.softmax(dim=-1)"],
  loss: ["loss = F.cross_entropy(logits, y)   # mean over B of −log p[y]"],
};

const FLOW = [
  { id: "image", label: "input image", sub: "[B, 3, H, W]", code: CODE.image },
  {
    group: "patch embedding",
    items: [
      { id: "patchify", label: "cut into patches", sub: "g × g grid of P × P", code: CODE.patchify },
      { id: "flatten", label: "flatten to a sequence", sub: "N vectors of P²·3", code: CODE.flatten },
      { id: "embed", label: "linear projection", sub: "P²·3 → D", code: CODE.embed },
    ],
  },
  {
    group: "token sequence",
    items: [
      { id: "cls", label: "prepend [CLS]", sub: "N → N+1 tokens", code: CODE.cls },
      { id: "pos", label: "add position embeddings", sub: "one learned vector per slot", code: CODE.pos },
    ],
  },
  {
    group: "encoder block",
    loop: "↻ × L",
    items: [
      { id: "ln1", label: "LayerNorm", sub: "per token", code: CODE.ln1 },
      { id: "qkv", label: "Q, K, V + split heads", sub: "[T,D] → h × [T,d]", code: CODE.qkv },
      { id: "attn", label: "attention weights", sub: "softmax(QKᵀ/√d)  [T,T]", code: CODE.attn },
      { id: "mix", label: "mix values + residual", sub: "A·V → concat → W_O", code: CODE.mix },
      { id: "mlp", label: "MLP + residual", sub: "D → 4D → D", code: CODE.mlp },
    ],
  },
  { id: "stack", label: "repeat L blocks", sub: "[B,T,D] stays [B,T,D]", code: CODE.stack },
  {
    group: "classification",
    items: [
      { id: "clsout", label: "take the CLS token", sub: "[B,T,D] → [B,D]", code: CODE.clsout },
      { id: "head", label: "linear head + softmax", sub: "fine-tuning head · D → K", code: CODE.head },
      { id: "loss", label: "cross-entropy loss", sub: "−log p[true class]", code: CODE.loss },
    ],
  },
];

// illustrative logits over the 1000 ImageNet classes; the top-5 and the loss are
// computed from them with a real softmax, so every number on screen agrees
const TRUE_IDX = 207;
const TOP_NAMES = { 207: "golden retriever", 208: "Labrador retriever", 219: "cocker spaniel", 852: "tennis ball", 850: "teddy bear" };
const LOGITS = (() => {
  const r = mulberry(11);
  const v = Array.from({ length: 1000 }, () => (r() - 0.5) * 3);
  [[207, 9], [208, 7.9], [219, 7.1], [852, 6.3], [850, 5.9]].forEach(([k, x]) => (v[k] = x));
  return v;
})();
const PROBS = (() => { const m = Math.max(...LOGITS), e = LOGITS.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.map((x) => x / z); })();
const PRED = Object.keys(TOP_NAMES).map((k) => [TOP_NAMES[k], PROBS[k]]).sort((a, b) => b[1] - a[1]);
const LOSS = -Math.log(PROBS[TRUE_IDX]);

// ------------------------------------------------------------ pixel-level canvas
const CW = 880, CH = 400, IMG = { x: 24, y: 48, s: 300 }, SEQ = { x: 372, y: 40, w: 488, h: 340 };

// wrap at ~1.7·g columns (not g) so the sequence never looks like the image re-assembled
function seqLayout(Tn, g) {
  const cols = Math.max(Math.ceil(Math.sqrt((Tn * SEQ.w) / SEQ.h)), Math.round(1.7 * g)), rows = Math.ceil(Tn / cols);
  const tile = Math.floor(Math.min(40, SEQ.w / cols, SEQ.h / rows));
  const gap = tile > 14 ? 2 : tile > 6 ? 1 : 0;
  return { tile, gap, cols, rows, at: (i) => [SEQ.x + (i % cols) * tile, SEQ.y + Math.floor(i / cols) * tile] };
}

const txt = (x, s, X, Y, color = T.dim, size = 11, align = "left", weight = 400) => {
  x.fillStyle = color;
  x.font = `${weight} ${Math.round(size * 1.2)}px 'IBM Plex Mono', monospace`;
  x.textAlign = align;
  x.fillText(s, X, Y);
};

// pseudo-embedding look for token k: stripes whose brightness comes from a seeded rng
function drawEmbedChip(x, f, k, X, Y, s, alpha) {
  const r = mulberry(k * 97 + 13);
  x.globalAlpha = alpha;
  x.fillStyle = T.well;
  x.fillRect(X, Y, s, s);
  const n = s > 12 ? 5 : 3;
  for (let i = 0; i < n; i++) {
    const v = r();
    x.fillStyle = `hsl(${(f.r * 220 + f.b * 120 + i * 40) % 360} 65% ${30 + v * 45}%)`;
    x.fillRect(X, Y + (i * s) / n, s, s / n - (s > 12 ? 0.5 : 0));
  }
  x.globalAlpha = 1;
}

function PatchCanvas({ stage, img, d, stats, p }) {
  const { g, N, Tn, P } = d, u = IMG.s / g, L = seqLayout(Tn, g);
  const hi = Math.floor(g / 2) * g + Math.floor(g / 2);

  const draw = (x) => {
    const showImg = (dim) => {
      x.drawImage(img.canvas, 0, 0, g * P, g * P, IMG.x, IMG.y, IMG.s, IMG.s);
      if (dim) { x.fillStyle = `rgba(22,29,41,${dim})`; x.fillRect(IMG.x, IMG.y, IMG.s, IMG.s); }
    };
    const grid = (a) => {
      x.strokeStyle = `rgba(245,248,251,${0.6 * a})`;
      x.lineWidth = g > 40 ? 0.4 : 0.9;
      x.beginPath();
      for (let k = 0; k <= g; k++) {
        x.moveTo(IMG.x + k * u, IMG.y); x.lineTo(IMG.x + k * u, IMG.y + IMG.s);
        x.moveTo(IMG.x, IMG.y + k * u); x.lineTo(IMG.x + IMG.s, IMG.y + k * u);
      }
      x.stroke();
    };
    const patchSrc = (k) => [(k % g) * P, Math.floor(k / g) * P, P, P];

    if (stage === "image") {
      showImg(0);
      txt(x, `W = ${d.res} px`, IMG.x + IMG.s / 2, IMG.y - 12, T.soft, 12, "center");
      x.save(); x.translate(IMG.x - 10, IMG.y + IMG.s / 2); x.rotate(-Math.PI / 2); txt(x, `H = ${d.res} px`, 0, 0, T.soft, 12, "center"); x.restore();
      // three channel planes
      const off = document.createElement("canvas");
      off.width = off.height = img.size;
      const o = off.getContext("2d");
      ["R", "G", "B"].forEach((ch, c) => {
        const im = o.createImageData(img.size, img.size);
        for (let i = 0; i < img.data.length; i += 4) { im.data[i + c] = img.data[i + c]; im.data[i + 3] = 255; }
        o.putImageData(im, 0, 0);
        const t = ease(Math.min(1, p * 1.4)), s = 170, X = 420 + c * 120 * t, Y = 70 + c * 40 * t;
        x.drawImage(off, X, Y, s, s);
        x.strokeStyle = ["#ff6b6b", "#7ee08a", "#7aa8ff"][c];
        x.lineWidth = 1.5;
        x.strokeRect(X, Y, s, s);
        txt(x, `channel ${ch}`, X + 4, Y + s + 16, x.strokeStyle, 11);
      });
      txt(x, `3 × ${d.res} × ${d.res} = ${fmtInt(3 * d.res * d.res)} numbers`, 560, 370, T.ink, 13, "center", 700);
      return;
    }

    if (stage === "patchify") {
      showImg(0);
      grid(ease(p));
      x.strokeStyle = T.accent; x.lineWidth = 2.5;
      x.strokeRect(IMG.x + (hi % g) * u, IMG.y + Math.floor(hi / g) * u, u, u);
      txt(x, `${g} × ${g} = ${N} patches`, IMG.x + IMG.s / 2, IMG.y + IMG.s + 26, T.ink, 13, "center", 700);
      // zoom of one patch with its pixel grid
      const Z = { x: 470, y: 60, s: 250 };
      x.imageSmoothingEnabled = false;
      x.drawImage(img.canvas, ...patchSrc(hi), Z.x, Z.y, Z.s, Z.s);
      x.imageSmoothingEnabled = true;
      if (P <= 16) {
        x.strokeStyle = "rgba(10,14,20,.5)"; x.lineWidth = 0.6; x.beginPath();
        for (let k = 0; k <= P; k++) { const q = (k * Z.s) / P; x.moveTo(Z.x + q, Z.y); x.lineTo(Z.x + q, Z.y + Z.s); x.moveTo(Z.x, Z.y + q); x.lineTo(Z.x + Z.s, Z.y + q); }
        x.stroke();
      }
      x.strokeStyle = T.accent; x.lineWidth = 2; x.strokeRect(Z.x, Z.y, Z.s, Z.s);
      x.setLineDash([4, 4]); x.strokeStyle = `${T.accent}88`; x.lineWidth = 1;
      x.beginPath();
      x.moveTo(IMG.x + ((hi % g) + 1) * u, IMG.y + Math.floor(hi / g) * u); x.lineTo(Z.x, Z.y);
      x.moveTo(IMG.x + ((hi % g) + 1) * u, IMG.y + (Math.floor(hi / g) + 1) * u); x.lineTo(Z.x, Z.y + Z.s);
      x.stroke(); x.setLineDash([]);
      txt(x, `one patch: ${P} × ${P} × 3 = ${P * P * 3} numbers`, Z.x + Z.s / 2, Z.y + Z.s + 26, T.accent, 13, "center", 700);
      return;
    }

    // stages from here on lay patches out as a sequence; slot 0 is reserved for CLS
    const moving = stage === "flatten";
    showImg(moving ? 0.55 + 0.35 * p : 0.88);
    grid(moving ? 1 - p * 0.5 : 0.35);
    for (let k = 0; k < N; k++) {
      const [sx, sy] = L.at(k + 1);
      let X = sx, Y = sy, s = L.tile - L.gap;
      if (moving) {
        const t = ease(Math.max(0, Math.min(1, p * 1.6 - (k / N) * 0.6)));
        const gx = IMG.x + (k % g) * u, gy = IMG.y + Math.floor(k / g) * u;
        X = gx + (sx - gx) * t; Y = gy + (sy - gy) * t; s = u + (s - u) * t;
      }
      const embedA = stage === "embed" ? ease(Math.max(0, Math.min(1, p * 1.5 - (k / N) * 0.5))) : stage === "flatten" ? 0 : 1;
      if (embedA < 1) x.drawImage(img.canvas, ...patchSrc(k), X, Y, s, s);
      if (embedA > 0) drawEmbedChip(x, stats.feats[k], k, X, Y, s, embedA);
      if (stage === "pos") {
        const f = stats.feats[k];
        x.globalAlpha = 0.85 * ease(p);
        x.fillStyle = `hsl(${(f.col / g) * 300} 80% ${45 + (f.row / g) * 30}%)`;
        x.fillRect(X, Y + s * 0.72, s, s * 0.28);
        x.globalAlpha = 1;
      }
          }
    // index of the first token in each wrapped row
    // before CLS exists slot s holds patch s−1, so label rows with the patch index
    const preCls = stage === "flatten" || stage === "embed";
    if (L.tile >= 9) for (let r = 0; r < L.rows; r++) txt(x, `${preCls ? Math.max(0, r * L.cols - 1) : r * L.cols}`, SEQ.x - 6, SEQ.y + r * L.tile + L.tile * 0.65, T.dim, Math.min(10, L.tile * 0.8), "right");
    // CLS slot
    const [cx, cy] = L.at(0), cs = L.tile - L.gap;
    if (stage === "cls" || stage === "pos") {
      const t = stage === "cls" ? ease(p) : 1;
      x.globalAlpha = t;
      x.fillStyle = T.accent;
      x.fillRect(cx, cy - (1 - t) * 40, cs, cs);
      if (cs >= 16) txt(x, cs >= 26 ? "CLS" : "C", cx + cs / 2, cy - (1 - t) * 40 + cs / 2 + 4, T.bg, cs >= 26 ? 9 : 10, "center", 700);
      x.globalAlpha = 1;
      if (stage === "pos") { x.fillStyle = "hsl(0 0% 85%)"; x.fillRect(cx, cy + cs * 0.72, cs, cs * 0.28); }
    } else {
      x.setLineDash([3, 3]); x.strokeStyle = T.rule; x.strokeRect(cx + 0.5, cy + 0.5, cs - 1, cs - 1); x.setLineDash([]);
    }
    txt(x, stage === "flatten" ? `sequence of N = ${N} patch vectors` : stage === "embed" ? `N = ${N} tokens, each a D = ${d.D} vector` : `T = N + 1 = ${Tn} tokens`, SEQ.x, SEQ.y - 12, T.ink, 12, "left", 700);

    // left-side explanations drawn over the dimmed image
    const panel = (title) => { x.fillStyle = "rgba(16,21,31,.92)"; x.fillRect(IMG.x - 8, IMG.y - 8, IMG.s + 16, IMG.s + 16); txt(x, title, IMG.x, IMG.y + 8, T.dim, 10); };
    if (stage === "embed") {
      panel("ONE PATCH → ONE TOKEN");
      const k = hi, ox = IMG.x + 8, oy = IMG.y + 40;
      x.imageSmoothingEnabled = false;
      x.drawImage(img.canvas, ...patchSrc(k), ox, oy, 56, 56);
      x.imageSmoothingEnabled = true;
      txt(x, `${P}×${P}×3`, ox + 28, oy + 72, T.soft, 10, "center");
      // flattened strip (raster order of the patch's pixels)
      const strip = document.createElement("canvas"); strip.width = 1; strip.height = P * P;
      const so = strip.getContext("2d"), im = so.createImageData(1, P * P), [px, py] = patchSrc(k);
      for (let i = 0; i < P * P; i++) { const o = ((py + Math.floor(i / P)) * img.size + px + (i % P)) * 4; im.data.set([img.data[o], img.data[o + 1], img.data[o + 2], 255], i * 4); }
      so.putImageData(im, 0, 0);
      x.imageSmoothingEnabled = false; x.drawImage(strip, ox + 86, oy - 20, 14, 230); x.imageSmoothingEnabled = true;
      txt(x, `${P * P} px × 3 = ${P * P * 3}`, ox + 93, oy + 226, T.accent2, 10, "center");
      txt(x, "×", ox + 117, oy + 96, T.ink, 16, "center");
      x.fillStyle = `${T.accent2}22`; x.strokeStyle = T.accent2; x.lineWidth = 1.2;
      x.fillRect(ox + 132, oy + 30, 80, 120); x.strokeRect(ox + 132, oy + 30, 80, 120);
      txt(x, "W_E", ox + 172, oy + 88, T.accent2, 13, "center", 700);
      txt(x, `[${P * P * 3} × ${d.D}]`, ox + 172, oy + 166, T.soft, 10, "center");
      txt(x, "=", ox + 226, oy + 96, T.ink, 16, "center");
      drawEmbedChip(x, stats.feats[k], k, ox + 240, oy + 20, 28, 1);
      x.fillStyle = T.well; x.fillRect(ox + 240, oy + 48, 28, 130);
      const r = mulberry(k);
      for (let i = 0; i < 26; i++) { x.fillStyle = heatCss(r()); x.fillRect(ox + 242, oy + 50 + i * 5, 24, 4); }
      txt(x, `D=${d.D}`, ox + 254, oy + 196, T.accent, 10, "center");
      txt(x, "same W_E for every patch", IMG.x, IMG.y + IMG.s - 6, T.soft, 11);
    }
    if (stage === "cls") {
      panel("THE [CLS] TOKEN");
      x.fillStyle = T.accent; x.fillRect(IMG.x + 20, IMG.y + 50, 40, 40);
      txt(x, "CLS", IMG.x + 40, IMG.y + 75, T.bg, 11, "center", 700);
      ["a learnable vector [1, 1, D]", "not from any patch", "same start for every image", "", "attention lets it read all", "patches; its final state", "is the image summary"].forEach((s, i) =>
        txt(x, s, IMG.x + 20, IMG.y + 120 + i * 20, i < 3 ? T.ink : T.soft, 12));
    }
    if (stage === "pos") {
      panel("POSITION-EMBEDDING SIMILARITY · toy pattern");
      const picks = [0, hi, N - 1], ms = 88;
      picks.forEach((k, j) => {
        const a = stats.feats[k], X = IMG.x + 4 + j * (ms + 10), Y = IMG.y + 40, uu = ms / g;
        for (const b of stats.feats) {
          const sim = Math.exp(-((a.row - b.row) ** 2 + (a.col - b.col) ** 2) / (2 * (g / 5) ** 2)) * 0.85 + (a.row === b.row || a.col === b.col ? 0.15 : 0);
          x.fillStyle = heatCss(sim); x.fillRect(X + b.col * uu, Y + b.row * uu, Math.ceil(uu), Math.ceil(uu));
        }
        x.strokeStyle = T.accent; x.lineWidth = 1.5; x.strokeRect(X + a.col * uu, Y + a.row * uu, Math.max(uu, 2), Math.max(uu, 2));
        txt(x, `patch (${a.row},${a.col})`, X + ms / 2, Y + ms + 16, T.soft, 10, "center");
      });
      txt(x, "nearby patches → similar rows", IMG.x + 4, IMG.y + 182, T.soft, 12);
      txt(x, `E_pos  [1, ${Tn}, ${d.D}]`, IMG.x + 4, IMG.y + 214, T.ink, 13, "left", 700);
      x.fillStyle = "hsl(0 0% 85%)"; x.fillRect(IMG.x + 4, IMG.y + 230, 14, 10);
      txt(x, "row 0 → CLS", IMG.x + 26, IMG.y + 240, T.accent, 14, "left", 700);
      txt(x, `rows 1…${N} → patches`, IMG.x + 26, IMG.y + 262, T.soft, 12);
      if (d.res !== 224) txt(x, `at ${d.res}: the 224-px grid is 2-D interpolated to ${d.g}×${d.g}`, IMG.x + 4, IMG.y + 288, T.wire, 10.5);
    }
  };

  return <Canvas w={CW} h={CH} draw={draw} deps={[stage, img, d.P, d.res, p]} />;
}

// ------------------------------------------------------------ encoder-block scenes
const SUBS = [
  { k: "in", t: "x  [B,T,D]" }, { k: "ln1", t: "LayerNorm" }, { k: "mha", t: "Self-Attention" }, { k: "add1", t: "⊕" },
  { k: "ln2", t: "LayerNorm" }, { k: "mlp", t: "MLP  D→4D→D" }, { k: "add2", t: "⊕" }, { k: "out", t: "x  [B,T,D]" },
];
const ACTIVE = { ln1: ["ln1"], qkv: ["mha"], attn: ["mha"], mix: ["mha", "add1"], mlp: ["ln2", "mlp", "add2"], stack: SUBS.map((s) => s.k) };

function BlockDiagram({ id, d }) {
  const on = ACTIVE[id] ?? [];
  const y = (i) => 14 + i * 46;
  return (
    <svg width={232} height={y(SUBS.length - 1) + 30} style={{ flexShrink: 0 }}>
      {/* residual skips */}
      {[[0, 3], [3, 6]].map(([a, b], i) => (
        <path key={i} d={`M105 ${y(a) + 14} H18 V${y(b) + 14} H86`} fill="none" stroke={on.includes(i ? "add2" : "add1") ? T.wire : T.rule} strokeWidth="1.4" className={on.includes(i ? "add2" : "add1") ? "flow" : ""} />
      ))}
      {SUBS.map((s, i) => {
        const a = on.includes(s.k), round = s.t === "⊕";
        return (
          <g key={s.k}>
            {i > 0 && <line x1={105} x2={105} y1={y(i - 1) + 28} y2={y(i)} stroke={T.rule} />}
            {round ? (
              <circle cx={105} cy={y(i) + 14} r={13} fill={a ? `${T.wire}33` : T.well} stroke={a ? T.wire : T.rule} />
            ) : (
              <rect x={22} y={y(i)} width={190} height={28} rx={6} fill={a ? `${T.accent}22` : T.well} stroke={a ? T.accent : T.rule} strokeWidth={a ? 1.6 : 1} />
            )}
            <text x={105} y={y(i) + 18} textAnchor="middle" fontFamily={mono} fontSize={round ? 17 : 14} fill={a ? (round ? T.wire : T.accent) : T.soft} fontWeight={a ? 700 : 400}>{s.k === "in" || s.k === "out" ? `x [B, ${d.Tn}, ${d.D}]` : s.k === "mlp" ? `MLP ${d.D}→${d.M}→${d.D}` : s.t}</text>
          </g>
        );
      })}
    </svg>
  );
}

function LNDetail({ d }) {
  const r = mulberry(3);
  const raw = Array.from({ length: 40 }, () => 2.5 + (r() - 0.3) * 6);
  const mu = raw.reduce((a, b) => a + b, 0) / raw.length, sd = Math.sqrt(raw.reduce((a, b) => a + (b - mu) ** 2, 0) / raw.length);
  const norm = raw.map((v) => (v - mu) / sd);
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Label>a random example vector (first 40 of D = {d.D})</Label>
      <Row>
        <div><Bars v={raw.map((v) => v)} w={260} h={80} color={T.soft} /><div style={{ fontFamily: mono, fontSize: 14, color: T.dim }}>μ = {mu.toFixed(2)}  σ = {sd.toFixed(2)}</div></div>
        <Op>→</Op>
        <div><Bars v={norm} w={260} h={80} /><div style={{ fontFamily: mono, fontSize: 14, color: T.accent }}>μ = 0  σ = 1  (then ×γ + β)</div></div>
      </Row>
      <svg width={520} height={40}><Tex x={0} y={4} size={19} color={T.ink} tex={"\\mathrm{LN}(x)=\\gamma\\odot\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}+\\beta"} /></svg>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>each token on its own · shape unchanged</div>
    </div>
  );
}

function QKVDetail({ d }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Row>
        <Mat shape="B T D" name="LN(x)" rows={d.Tn} cols={d.D} />
        <Op>×</Op>
        <Mat shape="D 3D" name="W_qkv" rows={d.D} cols={3 * d.D} color={T.accent2} />
        <Op>=</Op>
        {["Q", "K", "V"].map((n, i) => <Mat shape="B T D" key={n} name={n} rows={d.Tn} cols={d.D} color={["#8DB4FF", "#FFB86B", "#9BE564"][i]} />)}
      </Row>
      <Label>split D = {d.D} into h = {d.h} heads of d = {d.dh}</Label>
      <Row>
        {["Q", "K", "V"].map((n, i) => <Mat shape="B T D" key={n} name={`${n} heads`} rows={d.Tn} cols={d.D} cl={`D = ${d.h} heads × ${d.dh}`} stripes={d.h} color={["#8DB4FF", "#FFB86B", "#9BE564"][i]} />)}
        <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, maxWidth: 220, lineHeight: 1.5 }}><b style={{ color: T.ink }}>[B, {d.Tn}, {d.D}] → [B, {d.h}, {d.Tn}, {d.dh}]</b><br />{d.h} heads, {d.dh} dims each</div>
      </Row>
    </div>
  );
}

function AttnDetail({ d, img, stats, head, setHead }) {
  const A = useMemo(() => attention(head, stats.feats, stats.g), [head, stats]);
  const cls = useMemo(() => Array.from({ length: d.N }, (_, j) => A[j + 1]), [A, d.N]);
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <Row gap={6}>
        {HEADS.map((h) => (
          <button key={h.id} onClick={() => setHead(h.id)} style={{ fontFamily: mono, fontSize: 14, padding: "4px 9px", borderRadius: 6, cursor: "pointer", border: `1px solid ${h.id === head ? T.accent2 : T.rule}`, background: h.id === head ? `${T.accent2}22` : T.well, color: h.id === head ? T.accent2 : T.soft }}>{h.label}</button>
        ))}
        <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>toy heads on real pixels · {HEADS.find((h) => h.id === head).hint}</span>
      </Row>
      <Row gap={14} wrap={false} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>A · [{d.Tn} × {d.Tn}]</Label>
          <Heatmap A={A} n={d.Tn} size={210} mark={0} />
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>row = query token · col = key token · top row = CLS</span>
        </div>
        <div style={{ display: "grid", gap: 6 }}>
          <Label color={T.accent}>CLS row on the image</Label>
          <PatchImage img={img} P={d.P} size={170} grid={0.3} overlay={cls} />
          <div style={{ display: "grid", gap: 3, fontFamily: mono, fontSize: 15, marginTop: 4 }}>
            <span style={{ color: T.soft }}>per head: <b style={{ color: T.ink }}>{fmtInt(d.attnEntries)}</b> scores</span>
            <span style={{ color: T.soft }}>× {d.h} heads = <b style={{ color: d.attnEntries > 1e5 ? T.bad : T.ink }}>{fmtN(d.attnEntries * d.h)}</b></span>
            <span style={{ color: T.dim, fontSize: 13 }}>grows as T² ∝ (H/P)⁴</span>
          </div>
        </div>
      </Row>
      <svg width={520} height={40}><Tex x={0} y={4} size={19} color={T.ink} tex={`A_h=\\mathrm{softmax}\\!\\left(\\frac{Q_h K_h^{\\top}}{\\sqrt{${d.dh}}}\\right)\\in\\mathbb{R}^{${d.Tn}\\times ${d.Tn}}`} /></svg>
    </div>
  );
}

function MixDetail({ d }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Row>
        <Mat shape="B h T T" name="A_h" rows={d.Tn} cols={d.Tn} color={T.accent2} fill={`linear-gradient(135deg, ${T.accent2}55, ${T.wire}22)`} />
        <Op>×</Op>
        <Mat shape="B h T d" name="V_h" rows={d.Tn} cols={d.dh} color="#9BE564" />
        <Op>=</Op>
        <Mat shape="B h T d" name="o_h" rows={d.Tn} cols={d.dh} color="#9BE564" />
        <Op color={T.dim}>× {d.h} heads</Op>
      </Row>
      <Row>
        <Mat shape="B T D" name="concat" rows={d.Tn} cols={d.D} cl={`${d.h}·${d.dh}=${d.D}`} stripes={d.h} color="#9BE564" />
        <Op>× W_O</Op>
        <Mat shape="D D" name="W_O" rows={d.D} cols={d.D} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="B T D" name="attn out" rows={d.Tn} cols={d.D} />
        <Op color={T.wire}>+ x</Op>
        <Mat shape="B T D" name="x" rows={d.Tn} cols={d.D} color={T.wire} glow />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>each token = weighted mix of all tokens</div>
    </div>
  );
}

function MLPDetail({ d }) {
  const w = 360, h1 = 70, h2 = 150;
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <svg width={w + 100} height={h2 + 50}>
        <polygon points={`20,${(h2 - h1) / 2 + 10} 170,10 170,${h2 + 10} 20,${(h2 + h1) / 2 + 10}`} fill={`${T.accent2}22`} stroke={T.accent2} />
        <polygon points={`210,10 360,${(h2 - h1) / 2 + 10} 360,${(h2 + h1) / 2 + 10} 210,${h2 + 10}`} fill={`${T.accent2}22`} stroke={T.accent2} />
        <rect x={174} y={10} width={32} height={h2} rx={4} fill={`${T.wire}33`} stroke={T.wire} />
        <text x={190} y={h2 / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize={13} fill={T.wire} transform={`rotate(-90 190 ${h2 / 2 + 10})`}>GELU</text>
        <text x={95} y={h2 / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize={15} fill={T.accent2} fontWeight={700}>W_1</text>
        <text x={285} y={h2 / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize={15} fill={T.accent2} fontWeight={700}>W_2</text>
        <text x={20} y={h2 + 34} fontFamily={mono} fontSize={14} fill={T.soft}>D = {d.D}</text>
        <text x={190} y={h2 + 34} textAnchor="middle" fontFamily={mono} fontSize={14} fill={T.wire}>4D = {d.M}</text>
        <text x={360} y={h2 + 34} textAnchor="end" fontFamily={mono} fontSize={14} fill={T.soft}>D = {d.D}</text>
        <text x={400} y={h2 / 2 + 14} fontFamily={mono} fontSize={19} fill={T.wire}>+ x</text>
      </svg>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
        each token separately<br />
        [B, {d.Tn}, {d.D}] → [B, {d.Tn}, {d.M}] → [B, {d.Tn}, {d.D}] · {fmtN(2 * d.D * d.M)} weights, ⅔ of the block
      </div>
    </div>
  );
}

function StackDetail({ d, img, stats, head }) {
  const depths = [0, 0.5, 1];
  const rows = useMemo(() => depths.map((dp) => clsRow(head === "local" ? "object" : head, stats.feats, stats.g, dp)), [stats, head]);
  const n = d.L;
  return (
    <Row gap={22} style={{ alignItems: "flex-start" }}>
      <div style={{ display: "grid", gap: 2 }}>
        {Array.from({ length: n }, (_, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 120, height: n > 12 ? 10 : 18, borderRadius: 3, border: `1px solid ${T.accent}88`, background: `${T.accent}${(10 + Math.round((i / n) * 30)).toString(16).padStart(2, "0")}` }} />
            {(i === 0 || i === n - 1 || i === n / 2) && <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>block {i + 1} · [B,{d.Tn},{d.D}]</span>}
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gap: 8 }}>
        <Label color={T.accent}>CLS attention as depth grows (toy)</Label>
        <Row gap={10}>
          {rows.map((r, i) => (
            <div key={i} style={{ display: "grid", gap: 4, justifyItems: "center" }}>
              <PatchImage img={img} P={d.P} size={150} grid={0} overlay={r} />
              <span style={{ fontFamily: mono, fontSize: 13, color: T.soft }}>{["early", "middle", "last"][i]} block</span>
            </div>
          ))}
        </Row>
        <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
          {d.L} identical-shape blocks · {fmtN(d.params)} params total<br />
          {(d.macs / 1e9).toFixed(1)} GMACs per image, {(d.attnFrac * 100).toFixed(0)}% of each block in QKᵀ / A·V
        </div>
      </div>
    </Row>
  );
}

// ------------------------------------------------------------ head scenes
function HeadDetail({ id, d }) {
  const p = useProgress(id, 900);
  const logits = LOGITS;
  if (id === "clsout") {
    const n = Math.min(d.Tn, 40);
    return (
      <div style={{ display: "grid", gap: 16 }}>
        <Label>final LayerNorm, then index 0 of the sequence</Label>
        <div style={{ display: "flex", gap: 2, flexWrap: "wrap", maxWidth: 560 }}>
          {Array.from({ length: n }, (_, i) => (
            <div key={i} style={{ width: 12, height: 40, borderRadius: 2, background: i === 0 ? T.accent : T.rule, opacity: i === 0 ? 1 : 1 - 0.8 * ease(p), transition: "opacity .3s" }} />
          ))}
          {d.Tn > n && <span style={{ fontFamily: mono, fontSize: 14, color: T.dim, alignSelf: "center" }}>… {d.Tn - n} more</span>}
        </div>
        <Row>
          <Mat shape="B T D" name="x" rows={d.Tn} cols={d.D} />
          <Op>x[:, 0]</Op>
          <Mat shape="B D" name="z (CLS)" rows={1} cols={d.D} glow />
        </Row>
        <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>the {d.N} patch tokens are dropped</div>
      </div>
    );
  }
  const top = PRED;
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Row>
        <Mat shape="B D" name="z" rows={1} cols={d.D} />
        <Op>×</Op>
        <Mat shape="D K" name="W_head" rows={d.D} cols={d.K} color={T.accent2} />
        <Op>=</Op>
        <div style={{ display: "grid", gap: 4 }}>
          <Label>logits [B, {d.K}] · illustrative</Label>
          <Canvas w={300} h={70} deps={[]} draw={(x) => {
            const m = Math.max(...logits);
            logits.forEach((v, i) => { x.fillStyle = i === TRUE_IDX ? T.accent : T.soft; const hh = Math.max(0, (v + 2) / (m + 2)) * 66; x.fillRect((i * 300) / 1000, 70 - hh, 0.6, hh); });
          }} />
        </div>
      </Row>
      <Label>softmax → top-5 of {d.K} (illustrative)</Label>
      <div style={{ display: "grid", gap: 5, maxWidth: 520 }}>
        {top.map(([name, pr], i) => (
          <div key={name} style={{ display: "grid", gridTemplateColumns: "170px 1fr 54px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 15 }}>
            <span style={{ color: i === 0 ? T.accent : T.soft }}>{name}</span>
            <div style={{ height: 14, borderRadius: 3, background: T.well }}><div style={{ height: "100%", width: `${pr * 100 * ease(p)}%`, borderRadius: 3, background: i === 0 ? T.accent : T.soft }} /></div>
            <span style={{ color: T.ink, textAlign: "right" }}>{(pr * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
      {id === "loss" && (
        <div style={{ display: "grid", gap: 8, borderTop: `1px solid ${T.rule}`, paddingTop: 12 }}>
          <svg width={560} height={44}><Tex x={0} y={4} size={20} color={T.ink} tex={`\\mathcal{L}=-\\log p_{y}=-\\log ${PROBS[TRUE_IDX].toFixed(3)}=${LOSS.toFixed(3)}`} /></svg>
          <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
            true class: golden retriever<br />
            random guess: −log(1/{d.K}) = {Math.log(d.K).toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}

const pixel = ["image", "patchify", "flatten", "embed", "cls", "pos"];
function VitScene({ st, img, d, stats, head, setHead }) {
  const P = d.P, res = d.res;
  const p = useProgress(`${st.id}-${P}-${res}`, 1500);
  if (pixel.includes(st.id)) return <PatchCanvas stage={st.id} img={img} d={d} stats={stats} p={p} />;
  const body = {
    ln1: <LNDetail d={d} />,
    qkv: <QKVDetail d={d} />,
    attn: <AttnDetail d={d} img={img} stats={stats} head={head} setHead={setHead} />,
    mix: <MixDetail d={d} />,
    mlp: <MLPDetail d={d} />,
    stack: <StackDetail d={d} img={img} stats={stats} head={head} />,
  }[st.id];
  if (body) return <Row gap={24} wrap={false} style={{ alignItems: "flex-start", padding: 8, minHeight: 400, overflowX: "auto" }}><BlockDiagram id={st.id} d={d} />{body}</Row>;
  return <div style={{ padding: 12, minHeight: 400 }}><HeadDetail id={st.id} d={d} /></div>;
}

// ------------------------------------------------------------ view
export default function ViTView() {
  const [P, setP] = useState(16);
  const [res, setRes] = useState(224);
  const [head, setHead] = useState("object");
  const d = vitDims(res, P);
  const img = renderSubject(SUBJECTS[0], res);
  const stats = useMemo(() => patchStats(img, P), [img, P]);

  const B = "B";
  const steps = [
    { id: "image", block: "image", caption: "an RGB image is just a [3, H, W] grid of numbers", vars: () => [["H×W", `${res}×${res}`], ["C", "3"]] },
    { id: "patchify", block: "patchify", caption: `cut it into non-overlapping ${P}×${P} patches`, vars: () => [["g = H/P", `${d.g}`], ["N = g²", `${d.N}`]] },
    { id: "flatten", block: "flatten", caption: "line the patches up, left→right, top→bottom", vars: () => [["N", `${d.N}`], ["P²·3", `${d.pc}`]] },
    { id: "embed", block: "embed", caption: "one shared linear layer turns every patch into a D-dim token", vars: () => [["W_E", `[${d.pc}, ${d.D}]`], ["params", fmtN(d.pc * d.D + d.D)]] },
    { id: "cls", block: "cls", caption: "a learnable [CLS] token goes in front", vars: () => [["T = N+1", `${d.Tn}`]] },
    { id: "pos", block: "pos", caption: "add position row k to slot k · row 0 goes to CLS", vars: () => [["E_pos", `[${d.Tn}, ${d.D}]`]] },
    { id: "ln1", block: "ln1", caption: "normalise each token before attention (pre-norm)", vars: () => [["shape", `[${B},${d.Tn},${d.D}]`]] },
    { id: "qkv", block: "qkv", caption: `project to queries, keys, values, split into ${d.h} heads`, vars: () => [["h", `${d.h}`], ["d = D/h", `${d.dh}`]] },
    { id: "attn", block: "attn", caption: "every token scores every token, itself included", vars: () => [["T²", fmtInt(d.attnEntries), d.attnEntries > 1e5 ? "bad" : null]] },
    { id: "mix", block: "mix", caption: "weighted sum of values, heads concatenated, residual add", vars: () => [["out", `[${B},${d.Tn},${d.D}]`]] },
    { id: "mlp", block: "mlp", caption: "per-token MLP widens to 4D and back, residual add", vars: () => [["hidden", `${d.M}`]] },
    { id: "stack", block: "stack", caption: `stack ${d.L} blocks: the shape never changes`, vars: () => [["L", `${d.L}`], ["GMACs", (d.macs / 1e9).toFixed(1)]] },
    { id: "clsout", block: "clsout", caption: "only the CLS token's final state is used", vars: () => [["z", `[${B}, ${d.D}]`]] },
    { id: "head", block: "head", caption: `fine-tuning head: one linear layer → ${d.K} class scores (pre-training used Linear→tanh→Linear)`, vars: () => [["W_head", `[${d.D}, ${d.K}]`]] },
    { id: "loss", block: "loss", caption: "cross-entropy on the true label", vars: () => [["loss", `${LOSS.toFixed(3)} (illustrative)`]] },
  ];

  const G = {
    B: ["batch size", "B"], g: ["patches per side H/P", d.g], C: ["channels (RGB)", 3], H: ["image height", res], W: ["image width", res], P: ["patch side", P],
    N: ["patches (H/P)²", d.N], T: ["tokens N+1", d.Tn], "P²C": ["values per patch", d.pc], D: ["hidden width", d.D],
    "3D": ["Q,K,V stacked", 3 * d.D], h: ["heads", d.h], d: ["head width D/h", d.dh], M: ["MLP width 4D", d.M], K: ["classes", d.K],
  };
  const S = (name, dims, what, io, note) => ({ name, dims, what, io, note });
  const tags = (st) => ({
    image: [S("x", ["B", "C", "H", "W"], "batch of RGB images", "out")],
    patchify: [S("x", ["B", "C", "H", "W"], null, "in"), S("patches", ["B", "g", "g", "P", "P", "C"], null, "out")],
    flatten: [S("patches", ["B", "g", "g", "P", "P", "C"], null, "in"), S("patches", ["B", "N", "P²C"], "each patch is one long vector", "out")],
    embed: [S("patches", ["B", "N", "P²C"], null, "in"), S("W_E", ["P²C", "D"], "shared by all patches"), S("tokens", ["B", "N", "D"], null, "out")],
    cls: [S("cls_token", ["D"], "learned parameter", "in"), S("x", ["B", "T", "D"], "CLS at index 0", "out")],
    pos: [S("pos_embed", ["T", "D"], "learned, added element-wise", "in"), S("x", ["B", "T", "D"], null, "out")],
    ln1: [S("x", ["B", "T", "D"], null, "in"), S("LN(x)", ["B", "T", "D"], "normalised over D", "out")],
    attn: [S("q", ["B", "h", "T", "d"], null, "in"), S("kᵀ", ["B", "h", "d", "T"], null, "in"), S("A", ["B", "h", "T", "T"], "softmax(q·kᵀ/√d)", "out")],
    mlp: [S("hidden", ["B", "T", "M"], "after W_1 + GELU"), S("x", ["B", "T", "D"], "after W_2 + residual", "out")],
    stack: [S("x", ["B", "T", "D"], `same shape after each of ${d.L} blocks`, "out")],
    loss: [S("logits", ["B", "K"], null, "in"), S("loss", [], "scalar, mean over the batch", "out")],
  })[st.id];

  const { N, Tn, D, h, dh, M, K, pc, L } = d;
  const matmuls = (st) => {
    const body = {
      image: <MMNote>input [B, 3, {res}, {res}] = {fmtInt(3 * res * res)} numbers</MMNote>,
      patchify: <MMNote>rearrange: [B, 3, {res}, {res}] → [B, {d.g}, {d.g}, {P}, {P}, 3]</MMNote>,
      flatten: <MMNote>reshape: [B, {d.g}, {d.g}, {P}, {P}, 3] → [B, {N}, {pc}]</MMNote>,
      embed: <>
        <MM a={["patches", ["B", N, pc]]} b={["W_E", [pc, D]]} c={["tokens", ["B", N, D]]} note="same W_E for every patch" />
        <MMNote>= Conv2d(3, {D}, kernel={P}, stride={P})</MMNote>
      </>,
      cls: <MMNote>concat: cls [B, 1, {D}] ‖ tokens [B, {N}, {D}] → x [B, {Tn}, {D}]</MMNote>,
      pos: <MMNote>add: x [B, {Tn}, {D}] + E_pos [1, {Tn}, {D}] → [B, {Tn}, {D}]</MMNote>,
      ln1: <MMNote>normalise: [B, {Tn}, {D}] → [B, {Tn}, {D}], then × γ [{D}] + β [{D}]</MMNote>,
      qkv: <>
        <MM a={["LN(x)", ["B", Tn, D]]} b={["W_qkv", [D, 3 * D]]} c={["qkv", ["B", Tn, 3 * D]]} />
        <MMNote>reshape [B, {Tn}, 3, {h}, {dh}] → permute → q, k, v each [B, {h}, {Tn}, {dh}]</MMNote>
      </>,
      attn: <>
        <MM a={["q", ["B", h, Tn, dh]]} b={["kᵀ", ["B", h, dh, Tn]]} c={["S", ["B", h, Tn, Tn]]} note="for every head" />
        <MMNote>A = softmax(S / √{dh}, dim = −1)   [B, {h}, {Tn}, {Tn}], each row sums to 1</MMNote>
      </>,
      mix: <>
        <MM a={["A", ["B", h, Tn, Tn]]} b={["v", ["B", h, Tn, dh]]} c={["o", ["B", h, Tn, dh]]} />
        <MMNote>transpose + reshape (concat heads): [B, {h}, {Tn}, {dh}] → [B, {Tn}, {D}]</MMNote>
        <MM a={["o", ["B", Tn, D]]} b={["W_O", [D, D]]} c={["attn", ["B", Tn, D]]} note="x = x + attn" />
      </>,
      mlp: <>
        <MM a={["LN(x)", ["B", Tn, D]]} b={["W_1", [D, M]]} c={["h", ["B", Tn, M]]} note="+ b_1, then GELU element-wise" />
        <MM a={["h", ["B", Tn, M]]} b={["W_2", [M, D]]} c={["mlp", ["B", Tn, D]]} note="+ b_2, x = x + mlp" />
      </>,
      stack: <BlockMMs B="B" T={Tn} D={D} h={h} d={dh} M={M} L={L} />,
      clsout: <MMNote>index: x[:, 0] of [B, {Tn}, {D}] → z [B, {D}]</MMNote>,
      head: <MM a={["z", ["B", D]]} b={["W_head", [D, K]]} c={["logits", ["B", K]]} note={`+ b [${K}], then softmax over ${K}`} />,
      loss: <MMNote>pick p[y] from [B, {K}] → −log → mean over B → scalar</MMNote>,
    }[st.id];
    return <MatMulCard footer={st.id === "stack" ? `whole forward ≈ ${(d.macs / 1e9).toFixed(1)} GMAC per image` : null}>{body}</MatMulCard>;
  };

  return (
    <DimCtx.Provider value={G}>
    <StepShell
      matmuls={matmuls}
      title="Vision Transformer (ViT-B)"
      subtitle="attention maps, predictions and loss are illustrative (toy model on the real pixels) · shapes, params and MACs are exact"
      blocks={FLOW}
      steps={steps}
      scene={(st) => <VitScene st={st} img={img} d={d} stats={stats} head={head} setHead={setHead} />}
      tags={tags}
      controls={
        <Row gap={18}>
          <Seg label="PATCH SIZE P" value={P} options={[8, 14, 16, 32]} onChange={setP} disabled={res % 14 ? [14] : []} hint="384 / 14 is not a whole number of patches" />
          <Seg label="RESOLUTION" value={res} options={[224, 384]} onChange={(r) => { setRes(r); if (r % P) setP(16); }} />
          <Stat k="tokens T" v={fmtInt(d.Tn)} tone={T.accent} />
          <Stat k="attn scores / head" v={fmtInt(d.attnEntries)} tone={d.attnEntries > 1e5 ? T.bad : T.ink} />
          <Stat k="params" v={fmtN(d.params)} />
          <Stat k="GMACs" v={(d.macs / 1e9).toFixed(1)} tone={T.wire} />
        </Row>
      }
    />
    </DimCtx.Provider>
  );
}
