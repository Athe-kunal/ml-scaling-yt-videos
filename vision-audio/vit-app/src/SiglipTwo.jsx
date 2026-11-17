import { T, mono, btn } from "./theme";
import { Canvas, Mat, Op, Row, Label } from "./draw";
import { Seg } from "./StepShell";
import { SUBJECTS, naflexSize, NAFLEX_BUDGETS } from "./model";

// SigLIP 2 steps, appended to the SigLIP tab:
//   S2Diff  — what changed vs SigLIP (one table)
//   S2Size  — NaFlex: keep the aspect ratio, fit a token budget (real resize rule)
//   S2Pos   — the learned 16×16 position grid, resized to each image's h × w
//   S2Mask  — a batch of different shapes, padded to the budget, padding masked

export const P2 = 16; // every NaFlex checkpoint uses 16×16 patches

// test images with real-world shapes (height × width)
export const SHAPES = [
  { id: "photo", label: "photo 4:3", h: 480, w: 640, s: 0 },
  { id: "video", label: "video 16:9", h: 1080, w: 1920, s: 2 },
  { id: "square", label: "square 1:1", h: 512, w: 512, s: 3 },
  { id: "banner", label: "banner 1:4", h: 300, w: 1200, s: 4 },
];

const cache = new Map();
function render(shape) {
  if (cache.has(shape.id)) return cache.get(shape.id);
  const sub = SUBJECTS[shape.s], k = 400 / Math.max(shape.h, shape.w);
  const h = Math.round(shape.h * k), w = Math.round(shape.w * k);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const x = c.getContext("2d"), g = x.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, sub.bg[0]); g.addColorStop(1, sub.bg[1]);
  x.fillStyle = g; x.fillRect(0, 0, w, h);
  x.fillStyle = sub.ground; x.fillRect(0, h * 0.78, w, h * 0.22);
  x.font = `${Math.round(Math.min(w, h) * 0.62)}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
  x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(sub.emoji, w / 2, h * 0.5);
  cache.set(shape.id, c);
  return c;
}

// the image drawn into a w×h box with a gh×gw patch grid
function GridImage({ shape, W, H, gh, gw, tone = "rgba(245,248,251,.55)" }) {
  return (
    <Canvas w={W} h={H} deps={[shape.id, W, H, gh, gw]} style={{ borderRadius: 4 }} draw={(x) => {
      x.drawImage(render(shape), 0, 0, W, H);
      x.strokeStyle = tone; x.lineWidth = gh * gw > 400 ? 0.5 : 0.9;
      x.beginPath();
      for (let k = 0; k <= gw; k++) { x.moveTo((k * W) / gw, 0); x.lineTo((k * W) / gw, H); }
      for (let k = 0; k <= gh; k++) { x.moveTo(0, (k * H) / gh); x.lineTo(W, (k * H) / gh); }
      x.stroke();
    }} />
  );
}

export function ShapePicker({ shape, setShape, budget, setBudget, noImage }) {
  return (
    <Row gap={22}>
      {!noImage && <div style={{ display: "grid", gap: 4 }}>
        <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, letterSpacing: "0.08em" }}>IMAGE</span>
        <Row gap={4}>
          {SHAPES.map((s) => (
            <button key={s.id} onClick={() => setShape(s)} style={{ ...btn(s.id === shape.id, false), height: 28, fontSize: 14, padding: "0 9px" }}>
              {SUBJECTS[s.s].emoji} {s.label}
            </button>
          ))}
        </Row>
      </div>}
      <Seg label="TOKEN BUDGET" value={budget} options={NAFLEX_BUDGETS} onChange={setBudget} />
    </Row>
  );
}

// ------------------------------------------------------------ 1. what changed
export function S2Diff() {
  const rows = [
    ["image input", "square, fixed size (224, 384…)", "any aspect ratio, a token budget", true],
    ["patch embedding", "Conv2d(kernel = stride = P)", "Linear(16·16·3 → D) on pre-cut patches", true],
    ["position embedding", "one per slot, fixed grid", "16×16 grid, resized to each image", true],
    ["padding", "none (every image the same)", "padded to the budget, masked", true],
    ["tokenizer", "SentencePiece 32k, English", "Gemma 256k, multilingual", false],
    ["training losses", "sigmoid only", "sigmoid + captioning decoder + self-distillation + masked prediction", false],
    ["still the same", "ViT · MAP head · sigmoid loss · 64 text tokens", "← all kept", false],
  ];
  const cell = { fontFamily: mono, fontSize: 15, padding: "8px 12px", borderBottom: `1px solid ${T.rule}`, verticalAlign: "top" };
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <table style={{ borderCollapse: "collapse", maxWidth: 900 }}>
        <thead>
          <tr>
            <th style={{ ...cell, color: T.dim, textAlign: "left", fontWeight: 400 }} />
            <th style={{ ...cell, color: T.soft, textAlign: "left" }}>SigLIP</th>
            <th style={{ ...cell, color: T.accent, textAlign: "left" }}>SigLIP 2 (NaFlex)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, b, nf]) => (
            <tr key={k}>
              <td style={{ ...cell, color: nf ? T.wire : T.dim, whiteSpace: "nowrap" }}>{k}</td>
              <td style={{ ...cell, color: T.soft }}>{a}</td>
              <td style={{ ...cell, color: k === "still the same" ? T.dim : T.ink, fontWeight: nf ? 700 : 400 }}>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.7 }}>
        <span style={{ color: T.wire }}>yellow rows</span> = NaFlex, the next 3 steps · extra losses are training only, dropped after
      </div>
    </div>
  );
}

// ------------------------------------------------------------ 2. resize to the budget
export function S2Size({ shape, budget }) {
  const r = naflexSize(shape.h, shape.w, P2, budget);
  const box = 300, k = box / Math.max(r.H, r.W);
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={40} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 8, justifyItems: "start" }}>
          <Label>SigLIP · squash to 224 × 224</Label>
          <GridImage shape={shape} W={170} H={170} gh={14} gw={14} />
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>14 × 14 = <b>196</b> patches, always</span>
          {shape.h !== shape.w && <span style={{ fontFamily: mono, fontSize: 15, color: T.bad }}>shape distorted</span>}
        </div>
        <div style={{ display: "grid", gap: 8, justifyItems: "start" }}>
          <Label color={T.accent}>SigLIP 2 · keep the shape, fit {budget} tokens</Label>
          <GridImage shape={shape} W={Math.round(r.W * k)} H={Math.round(r.H * k)} gh={r.gh} gw={r.gw} />
          <span style={{ fontFamily: mono, fontSize: 17, color: T.ink }}>
            {shape.h}×{shape.w} → <b style={{ color: T.accent }}>{r.H}×{r.W}</b> → {r.gh} × {r.gw} = <b style={{ color: T.accent }}>{r.N}</b> patches
          </span>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.dim }}>both sides multiples of 16 · {budget - r.N} slots left over</span>
        </div>
      </Row>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>more budget = more detail · same weights for every budget</div>
    </div>
  );
}

// ------------------------------------------------------------ 3. position grid resized
// toy picture of the learned [256, D] table: each of the 16×16 rows drawn as a
// colour (3 of its D numbers). Bilinear resize, align_corners = False, like
// F.interpolate (the real one also antialiases when shrinking).
const toyPos = (r, c) => [0.5 + 0.45 * Math.sin(c * 0.7 + 0.3), 0.5 + 0.45 * Math.sin(r * 0.55 + 1.1), 0.5 + 0.45 * Math.cos((r + c) * 0.35)];
function bilinear(gh, gw) {
  const src = (d, out) => Math.min(15, Math.max(0, ((d + 0.5) * 16) / out - 0.5));
  return (i, j) => {
    const y = src(i, gh), x = src(j, gw), y0 = Math.floor(y), x0 = Math.floor(x), y1 = Math.min(15, y0 + 1), x1 = Math.min(15, x0 + 1), fy = y - y0, fx = x - x0;
    const a = toyPos(y0, x0), b = toyPos(y0, x1), c = toyPos(y1, x0), d = toyPos(y1, x1);
    return a.map((_, k) => (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy);
  };
}
function PosGrid({ gh, gw, size, f }) {
  const k = size / Math.max(gh, gw), W = Math.round(gw * k), H = Math.round(gh * k);
  return (
    <Canvas w={W} h={H} deps={[gh, gw, size]} style={{ borderRadius: 4 }} draw={(x) => {
      for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++) {
        const [r, g, b] = f(i, j);
        x.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
        x.fillRect(j * k, i * k, k + 0.5, k + 0.5);
      }
      if (k >= 6) {
        x.strokeStyle = "rgba(10,14,20,.35)"; x.lineWidth = 0.6; x.beginPath();
        for (let j = 0; j <= gw; j++) { x.moveTo(j * k, 0); x.lineTo(j * k, H); }
        for (let i = 0; i <= gh; i++) { x.moveTo(0, i * k); x.lineTo(W, i * k); }
        x.stroke();
      }
    }} />
  );
}
export function S2Pos({ shape, budget, m }) {
  const r = naflexSize(shape.h, shape.w, P2, budget);
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={26} style={{ alignItems: "center" }}>
        <div style={{ display: "grid", gap: 6, justifyItems: "start" }}>
          <Label>learned · 16 × 16 = 256 rows</Label>
          <PosGrid gh={16} gw={16} size={180} f={toyPos} />
          <span style={{ fontFamily: mono, fontSize: 14, color: T.dim }}>colour = a few of the D numbers</span>
        </div>
        <Op>bilinear resize →</Op>
        <div style={{ display: "grid", gap: 6, justifyItems: "start" }}>
          <Label color={T.accent}>this image · {r.gh} × {r.gw}</Label>
          <PosGrid gh={r.gh} gw={r.gw} size={280} f={bilinear(r.gh, r.gw)} />
          <span style={{ fontFamily: mono, fontSize: 14, color: T.dim }}>same pattern, stretched to the image's grid</span>
        </div>
      </Row>
      <Row>
        <Mat shape="16 16 D" name="position_embedding" rows={16} cols={m.D} rl="16·16" color={T.accent2} />
        <Op>resize to {r.gh}×{r.gw}</Op>
        <Mat shape={`${r.N} D`} name="positions" rows={r.N} cols={m.D} rl={`${r.N}`} glow />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>SigLIP: one fixed row per slot · SigLIP 2: one grid, any shape</div>
    </div>
  );
}

// ------------------------------------------------------------ 4. padded batch + mask
export function S2Mask({ budget, m }) {
  const rows = SHAPES.map((s) => ({ s, r: naflexSize(s.h, s.w, P2, budget) }));
  const worst = rows.reduce((a, r) => (budget - r.r.N > budget - a.r.N ? r : a), rows[0]);
  const Z = 24, C = 9, j0 = budget - Z;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label>a batch of 4 shapes · each padded to {budget}</Label>
      <span style={{ fontFamily: mono, fontSize: 14, color: T.dim }}>padding is small: the resize already fills the budget</span>
      <div style={{ display: "grid", gap: 6 }}>
        {rows.map(({ s, r }) => (
          <Row key={s.id} gap={10} wrap={false}>
            <span style={{ fontFamily: mono, fontSize: 14, color: T.soft, width: 150, whiteSpace: "nowrap" }}>{SUBJECTS[s.s].emoji} {s.label}</span>
            <div style={{ display: "flex", flex: 1, height: 16, borderRadius: 3, overflow: "hidden", background: T.well, maxWidth: 560 }}>
              <div style={{ width: `${(r.N / budget) * 100}%`, background: T.accent, transition: "width .4s" }} />
              <div style={{ flex: 1, background: `repeating-linear-gradient(45deg, ${T.rule}, ${T.rule} 3px, ${T.well} 3px, ${T.well} 6px)` }} />
            </div>
            <span style={{ fontFamily: mono, fontSize: 14, color: T.ink, whiteSpace: "nowrap", width: 190 }}>{r.gh}×{r.gw} = {r.N} <span style={{ color: T.dim }}>+ {budget - r.N} pad</span></span>
          </Row>
        ))}
      </div>
      <Row gap={26} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>attention mask · {SUBJECTS[worst.s.s].emoji} {worst.s.label} · zoom on the last {Z} slots</Label>
          <div style={{ display: "grid", gridTemplateColumns: `34px repeat(${Z}, ${C}px)`, gap: 1, alignItems: "center" }}>
            <span />
            {Array.from({ length: Z }, (_, j) => <span key={j} style={{ fontFamily: mono, fontSize: 10, color: T.dim, textAlign: "center" }}>{(j0 + j) % 4 === 0 ? j0 + j : ""}</span>)}
            {Array.from({ length: Z }, (_, i) => [
              <span key={`r${i}`} style={{ fontFamily: mono, fontSize: 10, color: T.dim, textAlign: "right", paddingRight: 3 }}>{(j0 + i) % 4 === 0 ? j0 + i : ""}</span>,
              ...Array.from({ length: Z }, (_, j) => {
                const padK = j0 + j >= worst.r.N, padQ = j0 + i >= worst.r.N;
                return <span key={j} style={{ height: C, borderRadius: 1, background: padK ? "#5a2a35" : "#2f8f80", opacity: padQ ? 0.35 : 1 }} />;
              }),
            ])}
          </div>
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>key → · query ↓ · slots {j0}–{budget - 1}</span>
        </div>
        <div style={{ display: "grid", gap: 8, fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
          <span style={{ display: "flex", gap: 8, alignItems: "center" }}><span style={{ width: 14, height: 14, borderRadius: 2, background: "#2f8f80" }} />real patch: can be looked at</span>
          <span style={{ display: "flex", gap: 8, alignItems: "center" }}><span style={{ width: 14, height: 14, borderRadius: 2, background: "#5a2a35" }} />padding: −∞, never looked at</span>
          <span style={{ display: "flex", gap: 8, alignItems: "center" }}><span style={{ width: 14, height: 14, borderRadius: 2, background: "#1d2a2c" }} />padding rows: computed, thrown away</span>
          <span style={{ marginTop: 6 }}>same mask in every block <b style={{ color: T.ink }}>and</b> in the MAP head</span>
        </div>
      </Row>
      <Row>
        <Mat shape={`n ${budget} 768`} name="pixel_values" rows={budget} cols={768} rl={`${budget}`} cl="16·16·3" color={T.soft} />
        <Op>× W</Op>
        <Mat shape="768 D" name="patch_embedding" rows={768} cols={m.D} rl="768" color={T.accent2} />
        <Op>=</Op>
        <Mat shape={`n ${budget} D`} name="x" rows={budget} cols={m.D} rl={`${budget}`} glow />
      </Row>
    </div>
  );
}
