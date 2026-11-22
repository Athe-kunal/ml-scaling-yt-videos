import { createContext, useContext, useMemo, useState } from "react";
import { T, mono, btn } from "./theme";
import { useProgress, ease, clamp01, lerp } from "./anim";
import StepShell from "./StepShell";
import { MM, MMNote, MatMulCard } from "./draw";
import { SUBJECTS, renderSubject } from "./model";

// How four vision-language models feed ViT output into an LLM decoder, one at a
// time (sub-tabs), each with its key numbers beside the lane. Toy picture: a 4×4 patch grid; the labels
// carry the real shapes (checked in HF transformers + open_flamingo source).
//   LLaVA-1.5   every patch → MLP → one LLM input token
//   Flamingo    Perceiver squeezes to 64 latents; text reads them by gated cross-attn
//   Q-Former    32 learned queries read the image → linear → 32 LLM input tokens
//   DeepStack   2×2 merge → input tokens; ViT layers 8/16/24 are added into LLM layers 1–3

const METHODS = [
  {
    id: "llava", name: "LLaVA", model: "LLaVA-1.5-7B", col: "#6EF3DD",
    vit: "CLIP ViT-L/14 · 336", vitN: 576, vitD: 1024, vitL: 24, pp: 588, tap: "layer −2 · drop CLS",
    bridge: "2-layer MLP", out: 16, nVis: 576, outD: 4096, outNote: "1 token per patch",
    llm: "Vicuna-7B", D: 4096, L: 32, kv: 4096, vocab: 32000, text: 4,
  },
  {
    id: "flamingo", name: "Flamingo", model: "OpenFlamingo-9B", col: "#FF8AC6",
    vit: "CLIP ViT-L/14 · 224", vitN: 256, vitD: 1024, vitL: 24, pp: 588, tap: "last layer · no CLS",
    bridge: "Perceiver ×6", out: 4, nVis: 64, outD: 1024, outNote: "64 learned latents",
    llm: "MPT-7B", D: 4096, L: 32, kv: 4096, vocab: 50432, text: 5,
  },
  {
    id: "qformer", name: "Q-Former", model: "BLIP-2 · OPT-6.7B", col: "#FFC94D",
    vit: "EVA ViT-g/14 · 224", vitN: 257, vitD: 1408, vitL: 40, pp: 588, tap: "layer −2 · 256 + CLS",
    bridge: "Q-Former ×12", out: 2, nVis: 32, outD: 4096, outNote: "32 learned queries",
    llm: "OPT-6.7B", D: 4096, L: 32, kv: 4096, vocab: 50272, text: 4,
  },
  {
    id: "deepstack", name: "DeepStack", model: "Qwen3-VL-8B", col: "#8DB4FF",
    vit: "SigLIP 2 ViT · 448", vitN: 784, vitD: 1152, vitL: 27, pp: 1536, tap: "+ taps after 8 · 16 · 24",
    bridge: "2×2 merge + MLP", out: 4, nVis: 196, outD: 4096, outNote: "4 patches → 1 token",
    llm: "Qwen3-8B", D: 4096, L: 36, kv: 1024, vocab: 151936, text: 4,
  },
];

const G = 4, W = 190, H = 440;
const seg = (p, a, b) => ease(clamp01((p - a) / (b - a)));
const rgb = (c) => `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`;
const mix = (cs, ws) => { const s = ws.reduce((a, b) => a + b, 0); return [0, 1, 2].map((k) => cs.reduce((a, c, j) => a + c[k] * ws[j], 0) / s); };

// the dog picture, its 4×4 patch colours
function usePic() {
  return useMemo(() => {
    const img = renderSubject(SUBJECTS[0], 64), d = img.data, cols = [];
    for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
      const s = [0, 0, 0];
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const o = ((r * 16 + y) * 64 + c * 16 + x) * 4; s[0] += d[o]; s[1] += d[o + 1]; s[2] += d[o + 2]; }
      cols.push(s.map((v) => v / 256 / 255));
    }
    return { url: img.canvas.toDataURL(), cols };
  }, []);
}

// soft weights of a learned query k (of K) over the 16 patches: each prefers a region
function qWeights(k, K) {
  const cy = K === 2 ? (k === 0 ? 1 : 2.4) : k < 2 ? 1 : 2.2, cx = K === 2 ? 1.6 : k % 2 ? 2.2 : 0.8;
  return Array.from({ length: 16 }, (_, j) => Math.exp(-((Math.floor(j / 4) - cy) ** 2 + ((j % 4) - cx) ** 2) / 1.6));
}
function outCols(m, pc) {
  if (m.id === "llava") return pc;
  if (m.id === "deepstack") return [0, 1, 2, 3].map((g) => { const r0 = Math.floor(g / 2) * 2, c0 = (g % 2) * 2; return mix([0, 1, 4, 5].map((o) => pc[r0 * 4 + c0 + o]), [1, 1, 1, 1]); });
  return Array.from({ length: m.out }, (_, k) => mix(pc, qWeights(k, m.out)));
}

// ------------------------------------------------------------ lanes
const Txt = ({ x, y, s, size = 14, c = T.dim, a = "middle", w = 400, op = 1 }) => (
  <text x={x} y={y} textAnchor={a} fontFamily={mono} fontSize={size} fontWeight={w} fill={c} opacity={op}>{s}</text>
);

// key numbers of one method; the rows that matter for the current step are lit
const ROWS = [
  ["vision encoder", (m) => m.vit],
  ["ViT output", (m) => `[${m.vitN}, ${m.vitD}] · ${m.tap}`],
  ["bridge", (m) => m.bridge],
  ["image tokens", (m) => (m.id === "flamingo" ? "64 (aside)" : `${m.nVis}`)],
  ["enters at", (m) => ({ llava: "input", flamingo: "x-attn, every 4th", qformer: "input", deepstack: "input + layers 1–3" }[m.id])],
  ["new LLM layers", (m) => (m.id === "flamingo" ? "8 gated x-attn" : "none")],
  ["LLM", (m) => `${m.llm} · ${m.L} × ${m.D}`],
  ["K,V kept after prompt", (m) => fmtMB(kvMB(m, realKeys(m, 0)))],
  ["new token looks at", (m) => { const [i, t] = lookAt(m, 1); return `${i} img + ${t} txt`; }],
  ["LLM trained?", (m) => ({ llava: "stage 2", flamingo: "never", qformer: "never", deepstack: "stage 2+" }[m.id])],
];
const STEP_ROWS = {
  vit: ["vision encoder", "ViT output"], bridge: ["bridge", "image tokens"], inside: ["bridge"],
  enter: ["image tokens", "enters at"], layer: ["enters at", "new LLM layers", "new token looks at"],
  prefill: ["K,V kept after prompt"], dec1: ["new token looks at"], dec2: ["new token looks at"], frozen: ["LLM trained?"],
};

function Solo({ m, stepId, children }) {
  const lit = STEP_ROWS[stepId] ?? [];
  return (
    <div style={{ display: "flex", gap: 28, alignItems: "flex-start", flexWrap: "wrap" }}>
      <div style={{ width: 300, flex: "none", background: T.well, border: `1px solid ${m.col}55`, borderRadius: 10, padding: "8px 6px 2px" }}>
        <div style={{ fontFamily: mono, fontSize: 18, fontWeight: 700, color: m.col, textAlign: "center" }}>{m.name}</div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", marginTop: 4 }}>{children}</svg>
      </div>
      <div style={{ display: "grid", minWidth: 300, flex: "1 1 300px", maxWidth: 480, fontFamily: mono }}>
        <div style={{ fontSize: 15, color: T.dim, paddingBottom: 8 }}>{m.model}</div>
        {ROWS.map(([k, f]) => {
          const on = lit.includes(k);
          return (
            <div key={k} style={{ display: "grid", gridTemplateColumns: "190px 1fr", gap: 10, padding: "9px 8px", borderTop: `1px solid ${T.rule}`, background: on ? `${m.col}14` : "none", borderLeft: `3px solid ${on ? m.col : "transparent"}` }}>
              <span style={{ fontSize: 14, color: on ? m.col : T.dim }}>{k}</span>
              <span style={{ fontSize: 16, color: on ? T.ink : T.soft, fontWeight: on ? 700 : 400 }}>{f(m)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const Sq = ({ x, y, s, c, stroke, op = 1, dash }) => (
  <rect x={x - s / 2} y={y - s / 2} width={s} height={s} rx={2} fill={c ? rgb(c) : "none"} stroke={stroke ?? "none"} strokeWidth={1.6} strokeDasharray={dash} opacity={op} />
);
const gridPos = (k, x0, y0, s = 14, gap = 3) => [x0 + (k % 4) * (s + gap) + s / 2, y0 + Math.floor(k / 4) * (s + gap) + s / 2];

// ---- step 1: image → ViT → patch vectors
function VitLane({ m, p, pic }) {
  const a = seg(p, 0, 0.4), b = clamp01((p - 0.4) / 0.2), c = seg(p, 0.6, 1);
  const BX = 25, BY = 112, BW = 140, BH = 70, lines = 8;
  const hot = m.id === "deepstack" ? [2, 5, 7] : m.id === "flamingo" ? [7] : [6];
  return (
    <g>
      <image href={pic.url} x={55} y={6} width={80} height={80} opacity={1 - 0.6 * a} />
      {[1, 2, 3].map((k) => <g key={k} stroke={T.bg} strokeWidth={1}><line x1={55 + k * 20} x2={55 + k * 20} y1={6} y2={86} /><line x1={55} x2={135} y1={6 + k * 20} y2={6 + k * 20} /></g>)}
      <rect x={BX} y={BY} width={BW} height={BH} rx={8} fill={`${m.col}10`} stroke={m.col} strokeWidth={1.4} />
      {Array.from({ length: lines }, (_, i) => {
        const lit = b * lines > i, tap = hot.includes(i);
        return <rect key={i} x={BX + 10} y={BY + 6 + i * 7.7} width={BW - 20} height={4} rx={2} fill={tap && c > 0 ? m.col : lit ? `${m.col}88` : T.rule} />;
      })}
      {hot.map((i) => <circle key={i} cx={BX + BW - 4} cy={BY + 8 + i * 7.7} r={3} fill={c > 0 ? m.col : "none"} />)}
      <Txt x={95} y={BY + BH + 18} s={`${m.vitL} blocks`} c={T.soft} />
      {pic.cols.map((col, k) => {
        const [sx, sy] = [55 + (k % 4) * 20 + 10, 6 + Math.floor(k / 4) * 20 + 10];
        const [mx, my] = [BX + 6 + (k + 0.5) * ((BW - 12) / 16), BY];
        const [ex, ey] = gridPos(k, 61, 215);
        const x = c > 0 ? lerp(mx, ex, c) : lerp(sx, mx, a), y = c > 0 ? lerp(BY + BH, ey, c) : lerp(sy, my, a);
        const s = c > 0 ? lerp(7, 14, c) : lerp(18, 7, a);
        return <Sq key={k} x={x} y={y} s={s} c={col} op={b > 0 && b < 1 && c === 0 ? 0 : 1} />;
      })}
      {m.id === "deepstack" && c > 0.5 && [3, 2, 1].map((o) => <rect key={o} x={61 + o * 4} y={215 - o * 4} width={65} height={65} rx={3} fill="none" stroke={m.col} opacity={0.35 * c} />)}
      <g opacity={c}>
        <Txt x={95} y={312} s={`[${m.vitN}, ${m.vitD}]`} size={15} w={700} c={m.col} />
        <Txt x={95} y={332} s={m.tap} size={13} />
        <Txt x={95} y={360} s={m.vit} size={13} c={T.soft} />
      </g>
    </g>
  );
}

// ---- step 2: the bridge
function BridgeLane({ m, p, pic }) {
  const oc = outCols(m, pic.cols), BX = 18, BY = 104, BW = 154, BH = 108;
  const inGrid = (k) => gridPos(k, 61, 8);
  const queries = m.id === "flamingo" || m.id === "qformer";
  const look = seg(p, 0.08, 0.45), fill = seg(p, 0.45, 0.62), down = seg(p, 0.62, 1);
  const outPos = (k) => (m.id === "llava" ? gridPos(k, 61, 236) : m.id === "deepstack" ? [80 + (k % 2) * 30, 252 + Math.floor(k / 2) * 30] : [95 + (k - (m.out - 1) / 2) * 30, 262]);
  return (
    <g>
      <rect x={BX} y={BY} width={BW} height={BH} rx={9} fill={`${m.col}10`} stroke={m.col} strokeWidth={1.4} />
      <Txt x={95} y={BY + 20} s={m.bridge} size={14} w={700} c={m.col} />
      {pic.cols.map((col, k) => {
        let [x, y] = inGrid(k), s = 14, op = 1;
        if (m.id === "llava") { const [ex, ey] = outPos(k), t = seg(p, 0.1, 0.9); x = lerp(x, ex, t); y = lerp(y, ey, t); op = 1; }
        if (m.id === "deepstack") { const g = (Math.floor(k / 8)) * 2 + (Math.floor((k % 4) / 2)), [gx, gy] = [inGrid(0)[0] + 8.5 + (g % 2) * 34, inGrid(0)[1] + 8.5 + Math.floor(g / 2) * 34], t = seg(p, 0.05, 0.3); x = lerp(x, gx + ((k % 2) - 0.5) * 7, t); y = lerp(y, gy + ((Math.floor(k / 4) % 2) - 0.5) * 7, t); s = lerp(14, 7, t); op = p > 0.3 ? 0.25 : 1; }
        return <Sq key={k} x={x} y={y} s={s} c={col} op={op} stroke={m.id === "llava" && p > 0.5 ? m.col : undefined} />;
      })}
      {queries && oc.map((col, k) => {
        const [qx, qy] = [95 + (k - (m.out - 1) / 2) * 30, 170], w = qWeights(k, m.out), mx = Math.max(...w);
        const [ox, oy] = outPos(k), x = lerp(qx, ox, down), y = lerp(qy, oy, down);
        return (
          <g key={k}>
            {down === 0 && look > 0 && pic.cols.map((_, j) => { const [jx, jy] = inGrid(j); return <line key={j} x1={qx} y1={qy - 9} x2={jx} y2={jy + 7} stroke={m.col} strokeWidth={1.2} opacity={look * 0.85 * (w[j] / mx) ** 1.5 * (1 - 0.5 * fill)} />; })}
            <rect x={x - 9} y={y - 9} width={18} height={18} rx={3} fill={fill > 0 ? rgb(col) : T.bg} fillOpacity={fill > 0 ? fill : 1} stroke={m.col} strokeWidth={1.8} strokeDasharray={fill > 0.5 ? undefined : "3 2"} />
          </g>
        );
      })}
      {m.id === "deepstack" && oc.map((col, k) => {
        const t = seg(p, 0.3, 0.95), [gx, gy] = [inGrid(0)[0] + 8.5 + (k % 2) * 34, inGrid(0)[1] + 8.5 + Math.floor(k / 2) * 34], [ox, oy] = outPos(k);
        return p > 0.3 && <rect key={k} x={lerp(gx, ox, t) - 11} y={lerp(gy, oy, t) - 11} width={22} height={22} rx={3} fill={rgb(col)} stroke={m.col} strokeWidth={1.8} />;
      })}
      {m.id === "qformer" && <Txt x={95} y={BY + BH - 10} s="+ Linear 768 → 4096" size={13} c={T.soft} op={down} />}
      {m.id === "deepstack" && <Txt x={95} y={BY + BH - 10} s="4608 → 4096" size={13} c={T.soft} />}
      <g opacity={seg(p, 0.75, 1)}>
        <Txt x={95} y={340} s={`${m.nVis}`} size={28} w={700} c={m.col} />
        <Txt x={95} y={360} s="image tokens" size={13} />
        <Txt x={95} y={384} s={`[${m.nVis}, ${m.outD}]`} size={15} w={700} c={T.ink} />
        <Txt x={95} y={404} s={m.outNote} size={13} />
        {m.id === "deepstack" && <Txt x={95} y={424} s="+ 3 × [196, 4096]" size={13} c={m.col} />}
        {m.id === "flamingo" && <Txt x={95} y={424} s="still ViT width" size={13} c={m.col} />}
      </g>
    </g>
  );
}

// ---- the LLM picture shared by the decoder steps: rows = positions (bottom = newest),
// columns = decoder layers (toy 8 for 32/36). A cell = that token's K,V in that layer.
const PROMPT = ["What", "is", "this", "?"], ANS = ["a", "dog", "</s>"];
const NC = 8, CX = 76, CW = 12, CG = 14, Y0 = 64;
const colX = (c) => CX + c * CG;
const XATTN = [0, 4], DSL = [0, 1, 2];
function rowsOf(m, nAns) {
  const rows = [];
  let y = Y0;
  const push = (r) => { rows.push({ ...r, y }); y += r.h; };
  if (m.id === "flamingo") push({ kind: "tag", label: "<img>", h: 17 });
  else for (let k = 0; k < m.out; k++) push({ kind: "img", k, h: m.id === "llava" ? 8 : 14 });
  PROMPT.forEach((w) => push({ kind: "txt", label: w, h: 17 }));
  for (let a = 0; a < nAns; a++) push({ kind: "ans", label: ANS[a], h: 17 });
  return { rows, end: y };
}
function realKeys(m, nAns) { return (m.id === "flamingo" ? m.text : m.nVis + m.text) + nAns; }
const lookAt = (m, nAns) => [m.id === "flamingo" ? 64 : m.nVis, m.text + nAns];
function LookAt({ m, nAns, y, who }) {
  const [img, txt] = lookAt(m, nAns);
  return (
    <g>
      <Txt x={95} y={y} s={`${who} looks at`} size={13} c={T.dim} />
      <text x={95} y={y + 19} textAnchor="middle" fontFamily={mono} fontSize={15} fontWeight={700}>
        <tspan fill={m.col}>{img} img</tspan><tspan fill={T.dim}> + </tspan><tspan fill={T.ink}>{txt} txt</tspan>
      </text>
    </g>
  );
}
const kvMB = (m, toks) => (toks * m.L * 2 * m.kv * 2) / 2 ** 20 + (m.id === "flamingo" ? (8 * 64 * 2 * 512 * 2) / 2 ** 20 : 0);
const fmtMB = (v) => (v >= 10 ? `${Math.round(v)} MB` : `${v.toFixed(1)} MB`);

function RowLabels({ m, rows, pic, op = 1, enter }) {
  const oc = outCols(m, pic.cols), img = rows.filter((r) => r.kind === "img");
  return (
    <g opacity={op}>
      {img.length > 0 && (() => {
        const y0 = img[0].y, y1 = img[img.length - 1].y + img[img.length - 1].h;
        return <g><line x1={47} x2={47} y1={y0 + 1} y2={y1 - 1} stroke={m.col} strokeWidth={1.5} /><Txt x={42} y={(y0 + y1) / 2 - 1} s={`${m.nVis}`} size={14} w={700} c={m.col} a="end" /><Txt x={42} y={(y0 + y1) / 2 + 13} s="img" size={13} a="end" /></g>;
      })()}
      {rows.map((r, i) => {
        if (r.kind === "img") {
          const t = enter ?? 1, [sx, sy] = [95 + (r.k % 4 - 1.5) * 16, 18 + Math.floor(r.k / 4) * 8];
          return <rect key={i} x={lerp(sx, 52, t)} y={lerp(sy, r.y + 1, t)} width={lerp(12, 18, t)} height={r.h - 2} rx={1.5} fill={rgb(oc[r.k])} stroke={m.col} strokeWidth={0.8} />;
        }
        const j = i - img.length, t = enter == null ? 1 : seg(enter, 0.2 + j * 0.1, 0.6 + j * 0.1);
        return (
          <g key={i} opacity={t}>
            <rect x={52} y={r.y + 2} width={18} height={r.h - 4} rx={2} fill={r.kind === "ans" ? `${m.col}55` : r.kind === "tag" ? `${m.col}33` : T.rule} stroke={r.kind === "tag" ? m.col : "none"} />
            <Txt x={46} y={r.y + 13} s={r.label} size={r.label.length > 5 ? 13 : 14} c={r.kind === "ans" ? m.col : T.soft} a="end" />
          </g>
        );
      })}
    </g>
  );
}

// Flamingo's latents beside the LM, DeepStack's three taps above layers 1–3
function SideInputs({ m, pic, hotCol, op = 1 }) {
  const oc = outCols(m, pic.cols);
  if (m.id === "flamingo") return (
    <g opacity={op}>
      <rect x={CX - 2} y={6} width={NC * CG} height={30} rx={6} fill={`${m.col}12`} stroke={m.col} strokeDasharray="4 3" />
      {oc.map((c, k) => <rect key={k} x={CX + 18 + k * 20} y={13} width={16} height={16} rx={2} fill={rgb(c)} stroke={m.col} strokeWidth={XATTN.includes(hotCol) ? 2.2 : 1} />)}
      <Txt x={46} y={20} s="64" size={14} w={700} c={m.col} a="end" />
      <Txt x={46} y={34} s="K,V" size={13} a="end" />
      {XATTN.map((c) => <line key={c} x1={colX(c) + CW / 2} x2={colX(c) + CW / 2} y1={36} y2={Y0 - 2} stroke={m.col} strokeWidth={hotCol === c ? 2.4 : 1.2} strokeDasharray={hotCol === c ? undefined : "3 3"} />)}
    </g>
  );
  if (m.id === "deepstack") return (
    <g opacity={op}>
      <Txt x={46} y={26} s="taps" size={13} a="end" />
      {DSL.map((c) => <g key={c}>{[0, 1].map((r) => [0, 1].map((q) => <rect key={`${r}${q}`} x={colX(c) + q * 6} y={12 + r * 6} width={5.5} height={5.5} fill={rgb(oc[r * 2 + q])} stroke={hotCol === c ? m.col : "none"} strokeWidth={1} />))}</g>)}
      {DSL.map((c) => <line key={`l${c}`} x1={colX(c) + CW / 2} x2={colX(c) + CW / 2} y1={26} y2={Y0 - 2} stroke={m.col} strokeWidth={hotCol === c ? 2.4 : 1} strokeDasharray={hotCol === c ? undefined : "3 3"} />)}
    </g>
  );
  return null;
}

function Columns({ rows, cell }) {
  return (
    <g>
      {rows.map((r, i) => Array.from({ length: NC }, (_, c) => {
        const st = cell(r, i, c) ?? {};
        return <rect key={`${i}.${c}`} x={colX(c)} y={r.y + 1} width={CW} height={r.h - 2} rx={1.5}
          fill={st.fill ?? "none"} fillOpacity={st.fo ?? 1} stroke={st.stroke ?? `${T.rule}`} strokeWidth={st.sw ?? 0.8} />;
      }))}
    </g>
  );
}

// ---- step 2b: one layer inside each bridge. Boxes light up as the tokens pass.
// node: { x, y, w, h, t, s, k: "in" | "out" | "op" | "add" }   (add = ⊕, x/y = centre)
const FULL = { x: 8, w: 174 }, LEFT = { x: 8, w: 84 }, RIGHT = { x: 98, w: 84 };
const ARCH = {
  llava: {
    title: "MLP", params: "≈ 21M", note: "each token alone",
    nodes: [
      { ...FULL, y: 0, h: 36, t: "x", s: "[576, 1024]", k: "in" },
      { ...FULL, y: 62, h: 30, t: "Linear 1024 → 4096" },
      { ...FULL, y: 118, h: 30, t: "GELU" },
      { ...FULL, y: 174, h: 30, t: "Linear 4096 → 4096" },
      { ...FULL, y: 230, h: 36, t: "v", s: "[576, 4096]", k: "out" },
    ],
    edges: [[95, 36, 95, 62], [95, 92, 95, 118], [95, 148, 95, 174], [95, 204, 95, 230]],
    res: [],
  },
  flamingo: {
    title: "Perceiver layer", params: "≈ 10M × 6", note: "latents are in K,V too",
    nodes: [
      { ...LEFT, y: 0, h: 36, t: "latents", s: "[64, 1024]", k: "in" },
      { ...RIGHT, y: 0, h: 36, t: "patches", s: "[256, 1024]", k: "in" },
      { ...LEFT, y: 58, h: 36, t: "LN · W_q", s: "[64, 512]" },
      { ...RIGHT, y: 58, h: 36, t: "concat·LN", s: "W_kv [320]" },
      { ...FULL, y: 116, h: 36, t: "attention", s: "8 heads · softmax" },
      { ...FULL, y: 170, h: 28, t: "W_o 512 → 1024" },
      { k: "add", x: 95, y: 214 },
      { ...FULL, y: 232, h: 36, t: "LN · FFN", s: "1024 → 4096 → 1024" },
      { k: "add", x: 95, y: 286 },
      { ...FULL, y: 304, h: 36, t: "latents", s: "[64, 1024]", k: "out" },
    ],
    edges: [[50, 36, 50, 58], [140, 36, 140, 58], [92, 30, 104, 58, true], [50, 94, 50, 116], [140, 94, 140, 116], [95, 152, 95, 170], [95, 198, 95, 207], [95, 221, 95, 232], [95, 268, 95, 279], [95, 293, 95, 304]],
    res: [[8, 18, 214, 3], [88, 224, 286, 3]],
  },
  qformer: {
    title: "Q-Former layer", params: "≈ 188M (12)", note: "x-attn every 2nd layer",
    nodes: [
      { ...LEFT, y: 0, h: 36, t: "queries", s: "[32, 768]", k: "in" },
      { ...RIGHT, y: 0, h: 36, t: "image", s: "[257, 1408]", k: "in" },
      { x: 8, w: 132, y: 56, h: 30, t: "self-attn" },
      { k: "add", x: 74, y: 104, ln: true },
      { ...FULL, y: 124, h: 36, t: "cross-attn", s: "k, v ← image" },
      { k: "add", x: 95, y: 178, ln: true },
      { ...FULL, y: 198, h: 36, t: "FFN", s: "768 → 3072 → 768" },
      { k: "add", x: 95, y: 252, ln: true },
      { ...FULL, y: 272, h: 36, t: "queries", s: "[32, 768]", k: "out" },
    ],
    edges: [[50, 36, 50, 56], [74, 86, 74, 97], [74, 111, 74, 124], [162, 36, 162, 124], [95, 160, 95, 171], [95, 185, 95, 198], [95, 234, 95, 245], [95, 259, 95, 272]],
    res: [[8, 18, 104, 3, 67], [60, 116, 178, 3], [60, 190, 252, 3]],
  },
  deepstack: {
    title: "merger", params: "≈ 40M × 4", note: "no attention",
    nodes: [
      { ...FULL, y: 52, h: 28, t: "LN each 1152" },
      { ...FULL, y: 136, h: 28, t: "Linear 4608 → 4608" },
      { ...FULL, y: 184, h: 28, t: "GELU" },
      { ...FULL, y: 232, h: 28, t: "Linear 4608 → 4096" },
      { ...FULL, y: 284, h: 36, t: "1 token", s: "[1, 4096] · × 196", k: "out" },
    ],
    edges: [[95, 120, 95, 136], [95, 164, 95, 184], [95, 212, 95, 232], [95, 260, 95, 284]],
    res: [],
  },
};

function InsideLane({ m, p, pic }) {
  const A = ARCH[m.id], Y = 34, sy = lerp(-10, 350, clamp01(p / 0.9));
  const lit = (y) => sy >= y, oc = outCols(m, pic.cols);
  const four = [0, 1, 4, 5].map((o) => pic.cols[o]);
  const join = seg(p, 0.12, 0.3);
  return (
    <g>
      <Txt x={95} y={18} s={A.title} size={15} w={700} c={m.col} />
      <g transform={`translate(0, ${Y})`}>
        <defs><marker id={`ia-${m.id}`} markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={T.dim} /></marker></defs>
        {A.res.map(([x0, y0, y1, xl, xe], i) => (
          <path key={i} d={`M${x0},${y0} H${xl} V${y1} H${(xe ?? 95) - 8}`} fill="none" stroke={lit(y1) ? m.col : T.rule} strokeWidth={1.4} strokeDasharray="4 3" markerEnd={`url(#ia-${m.id})`} />
        ))}
        {A.edges.map(([x1, y1, x2, y2, dash], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2 - 2} stroke={lit(y2) ? m.col : T.rule} strokeWidth={1.4} strokeDasharray={dash ? "3 3" : undefined} markerEnd={`url(#ia-${m.id})`} />
        ))}
        {A.nodes.map((n, i) => {
          const on = lit(n.y), now = on && sy < n.y + 40;
          if (n.k === "add") return (
            <g key={i}>
              <circle cx={n.x} cy={n.y} r={7} fill={on ? `${m.col}33` : T.bg} stroke={on ? m.col : T.rule} strokeWidth={1.4} />
              <Txt x={n.x} y={n.y + 4.5} s="+" size={13} w={700} c={on ? m.col : T.dim} />
              {n.ln && <Txt x={n.x + 12} y={n.y + 4.5} s="LN" size={13} c={on ? m.col : T.dim} a="start" />}
            </g>
          );
          const io = n.k === "in" || n.k === "out";
          return (
            <g key={i}>
              <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={io ? 4 : 7} fill={io ? `${m.col}${on ? "33" : "12"}` : now ? `${m.col}30` : on ? `${m.col}14` : T.bg}
                stroke={on ? m.col : T.rule} strokeWidth={now ? 2.2 : 1.3} />
              <Txt x={n.x + n.w / 2} y={n.y + (n.s ? 15 : n.h / 2 + 5)} s={n.t} size={14} w={700} c={on ? T.ink : T.dim} />
              {n.s && <Txt x={n.x + n.w / 2} y={n.y + 30} s={n.s} size={13} c={on ? T.soft : T.dim} />}
            </g>
          );
        })}
        {m.id === "llava" && [0, 1, 2].map((j) => sy > 0 && sy < 250 && <Sq key={j} x={55 + j * 40} y={Math.min(sy, 246)} s={10} c={pic.cols[5 + j]} stroke={m.col} />)}
        {m.id === "deepstack" && <g>
          {four.map((c, j) => {
            const x0 = 18 + j * 42, x1 = 8 + j * 43.5;
            return (
              <g key={j}>
                <rect x={lerp(x0, x1, join)} y={lerp(0, 98, join)} width={lerp(34, 43.5, join)} height={lerp(30, 20, join)} rx={3} fill={rgb(c)} stroke={m.col} strokeWidth={1.2} />
                <Txt x={lerp(x0, x1, join) + lerp(17, 21.75, join)} y={lerp(19, 112, join)} s="1152" size={13} c={T.bg} w={700} />
              </g>
            );
          })}
          <Txt x={95} y={93} s="concat → 4608" size={13} c={m.col} op={join} />
        </g>}
        {m.id !== "llava" && m.id !== "deepstack" && sy > 0 && sy < 340 && <circle cx={m.id === "flamingo" && sy < 116 ? 50 : 95} cy={sy} r={4} fill={m.col} />}
      </g>
      <g opacity={seg(p, 0.75, 1)}>
        <Txt x={95} y={396} s={A.params} size={16} w={700} c={T.ink} />
        <Txt x={95} y={414} s="weights" size={13} />
        <Txt x={95} y={434} s={A.note} size={13} c={m.col} />
      </g>
    </g>
  );
}

// ---- step 3: where the image enters
function EnterLane({ m, p, pic }) {
  const { rows } = rowsOf(m, 0), t = seg(p, 0, 0.55);
  return (
    <g>
      <Columns m={m} rows={rows} cell={() => ({})} />
      <RowLabels m={m} rows={rows} pic={pic} enter={t} />
      <SideInputs m={m} pic={pic} hotCol={-1} op={seg(p, 0.2, 0.7)} />
      <g opacity={seg(p, 0.6, 1)}>
        <Txt x={95} y={340} s={{ llava: "image = input tokens", flamingo: "image = side memory", qformer: "image = input tokens", deepstack: "input tokens" }[m.id]} size={13} c={T.soft} />
        <Txt x={95} y={358} s={{ llava: "", flamingo: "x-attn every 4th layer", qformer: "", deepstack: "+ added in layers 1–3" }[m.id]} size={13} c={m.col} />
        <Txt x={95} y={392} s={`[${m.id === "flamingo" ? m.text : m.nVis + m.text}, ${m.D}]`} size={15} w={700} c={T.ink} />
        <Txt x={95} y={410} s="into the LLM" size={13} />
        <Txt x={CX + (NC * CG) / 2} y={Y0 + (rows.at(-1).y + 17 - Y0) + 16} s={`${m.L} layers →`} size={13} c={T.dim} />
      </g>
    </g>
  );
}

// toy attention of the last prompt token over earlier rows
const attnW = (i, c) => 0.15 + 0.85 * Math.abs(Math.sin(i * 1.7 + c * 2.3 + 0.4));

// ---- step 4: one pass up the decoder, seen from the last prompt token
function LayerLane({ m, p, pic }) {
  const { rows } = rowsOf(m, 0), q = rows.length - 1;
  const sw = clamp01((p - 0.05) / 0.9) * NC, cur = Math.min(NC - 1, Math.floor(sw));
  const cell = (r, i, c) => {
    if (c !== cur) return c < cur ? { stroke: `${m.col}44` } : {};
    if (i === q) return { fill: m.col, stroke: T.ink, sw: 1.6 };
    return { fill: m.col, fo: attnW(i, c) * 0.9, stroke: `${m.col}88` };
  };
  const xHot = m.id === "flamingo" && XATTN.includes(cur), dHot = m.id === "deepstack" && DSL.includes(cur);
  return (
    <g>
      <Columns m={m} rows={rows} cell={cell} />
      <RowLabels m={m} rows={rows} pic={pic} />
      <SideInputs m={m} pic={pic} hotCol={cur} />
      {dHot && rows.filter((r) => r.kind === "img").map((r, k) => <Txt key={k} x={colX(cur) + CW / 2} y={r.y + r.h / 2 + 4} s="+" size={13} w={700} c={T.bg} />)}
      <Txt x={colX(cur) + CW / 2} y={rows.at(-1).y + 34} s={`layer ${Math.round(((cur + 1) * m.L) / NC)}`} size={13} c={m.col} />
      <g>
        <Txt x={95} y={352} s={{ llava: "self-attn: image + text", flamingo: xHot ? "x-attn → 64 latents" : "self-attn: text only", qformer: "self-attn: 32 + text", deepstack: dHot ? `h[img] += tap ${[8, 16, 24][cur]}` : "self-attn: image + text" }[m.id]} size={13} c={xHot || dHot ? m.col : T.soft} />
        {m.id === "flamingo" && <Txt x={95} y={372} s="h + tanh(α) · xattn" size={13} c={xHot ? m.col : T.dim} />}
        <LookAt m={m} nAns={0} y={400} who="“?”" />
      </g>
    </g>
  );
}

// next-token bars
const VOC = ["a", "dog", "cat", "grass", "</s>"];
const PROBS = [[0.74, 0.09, 0.05, 0.02, 0.1], [0.03, 0.82, 0.09, 0.04, 0.02], [0.04, 0.03, 0.02, 0.07, 0.84]];
function probsFor(m, s) {
  const k = METHODS.indexOf(m), v = PROBS[s].map((x, i) => x * (1 + 0.18 * Math.sin(i * 3.1 + k * 1.9 + s)));
  const z = v.reduce((a, b) => a + b, 0);
  return v.map((x) => x / z);
}
function Logits({ m, s, t, y0 }) {
  const pr = probsFor(m, s), am = pr.indexOf(Math.max(...pr));
  return (
    <g opacity={t > 0 ? 1 : 0.25}>
      {VOC.map((w, i) => (
        <g key={w}>
          <Txt x={56} y={y0 + i * 17 + 12} s={w} size={13} c={i === am && t > 0.6 ? m.col : T.dim} a="end" w={i === am && t > 0.6 ? 700 : 400} />
          <rect x={60} y={y0 + i * 17 + 3} width={Math.max(1, pr[i] * 100 * t)} height={11} rx={2} fill={i === am ? m.col : T.rule} />
          {t > 0.6 && <Txt x={64 + pr[i] * 100} y={y0 + i * 17 + 12} s={pr[i].toFixed(2)} size={13} c={i === am ? m.col : T.dim} a="start" />}
        </g>
      ))}
    </g>
  );
}

// ---- steps 5–7: prefill, then one token at a time with the KV cache
function GenLane({ m, p, pic, s }) {
  const { rows } = rowsOf(m, s), last = rows.length - 1;
  const decode = s > 0, a = decode ? seg(p, 0, 0.18) : 1;
  const sw = clamp01((p - (decode ? 0.2 : 0.05)) / 0.6) * NC, cur = Math.floor(sw), done = p >= (decode ? 0.8 : 0.65);
  const lt = seg(p, decode ? 0.8 : 0.68, 1);
  const cell = (r, i, c) => {
    const isNew = !decode || i === last;
    if (!isNew) return c === cur && !done ? { fill: m.col, fo: 0.35, stroke: T.ink, sw: 1.2 } : { fill: m.col, fo: 0.3, stroke: `${m.col}55` };
    if (c < cur || done) return { fill: m.col, fo: decode ? 0.55 : 0.3, stroke: `${m.col}88` };
    if (c === cur) return { fill: m.col, fo: 1, stroke: T.ink, sw: 1.4 };
    return {};
  };
  const hotCol = done ? -1 : cur;
  const toks = realKeys(m, s);
  return (
    <g>
      <Columns m={m} rows={rows} cell={cell} />
      <g transform={decode ? `translate(0, ${(1 - a) * 40})` : undefined} opacity={1}>
        <RowLabels m={m} rows={rows} pic={pic} />
      </g>
      <SideInputs m={m} pic={pic} hotCol={m.id === "deepstack" && decode ? -1 : hotCol} />
      {m.id === "deepstack" && !decode && DSL.includes(cur) && !done && rows.filter((r) => r.kind === "img").map((r, k) => <Txt key={k} x={colX(cur) + CW / 2} y={r.y + r.h / 2 + 4} s="+" size={13} w={700} c={T.bg} />)}
      {decode ? <LookAt m={m} nAns={s} y={312} who={`“${ANS[s - 1]}”`} /> : <>
        <Txt x={95} y={312} s={`${toks} tokens, one pass`} size={13} c={T.dim} />
        <Txt x={95} y={331} s={`K,V kept: ${fmtMB(kvMB(m, toks))}`} size={15} w={700} c={T.ink} />
      </>}
      <Logits m={m} s={s} t={lt} y0={345} />
    </g>
  );
}

// ---- step 8: frozen / trained
const TRAIN = {
  llava: { st: ["1", "2"], rows: [["ViT", "❄", "❄"], ["MLP", "🔥", "🔥"], ["LLM", "❄", "🔥"]], note: ["1: align", "2: instruct"] },
  flamingo: { st: ["1"], rows: [["ViT", "❄"], ["Perceiver", "🔥"], ["x-attn", "🔥"], ["LLM", "❄"]], note: ["α = 0 at start", "tanh(0) = 0 → plain LLM"] },
  qformer: { st: ["1", "2"], rows: [["ViT", "❄", "❄"], ["Q-Former", "🔥", "🔥"], ["Linear", "–", "🔥"], ["LLM", "–", "❄"]], note: ["1: no LLM yet", "2: + frozen LLM"] },
  deepstack: { st: ["1", "2+"], rows: [["ViT", "❄", "🔥"], ["mergers", "🔥", "🔥"], ["LLM", "❄", "🔥"]], note: ["1: mergers only", "2+: everything"] },
};
function FrozenLane({ m, p }) {
  const t = TRAIN[m.id];
  return (
    <g>
      <Txt x={48} y={30} s="stage" size={13} c={T.dim} />
      {t.st.map((s, j) => <Txt key={s} x={120 + j * 40} y={30} s={s} size={15} w={700} c={T.soft} />)}
      {t.rows.map(([name, ...ic], i) => {
        const y = 50 + i * 50, op = seg(p, i * 0.15, i * 0.15 + 0.3);
        return (
          <g key={name} opacity={op}>
            <rect x={6} y={y} width={84} height={36} rx={7} fill={`${m.col}12`} stroke={m.col} />
            <Txt x={48} y={y + 23} s={name} size={14} c={T.ink} w={700} />
            {ic.map((c, j) => <text key={j} x={120 + j * 40} y={y + 26} textAnchor="middle" fontSize={22} fill={T.dim}>{c}</text>)}
          </g>
        );
      })}
      <g opacity={seg(p, 0.7, 1)}>
        {t.note.map((s, i) => <Txt key={i} x={95} y={290 + i * 22} s={s} size={13} c={i === 0 ? m.col : T.soft} />)}
      </g>
    </g>
  );
}

// ---- step 9: the comparison table
function Compare({ sel }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "190px repeat(4, minmax(0, 1fr))", gap: "2px 10px", fontFamily: mono, padding: 6 }}>
      <span />
      {METHODS.map((m) => <div key={m.id} style={{ fontSize: 17, fontWeight: 700, color: m.col, padding: "4px 6px 8px", opacity: m.id === sel ? 1 : 0.7 }}>{m.name}</div>)}
      {ROWS.map(([k, f]) => [
        <div key={k} style={{ fontSize: 14, color: T.dim, padding: "9px 0", borderTop: `1px solid ${T.rule}` }}>{k}</div>,
        ...METHODS.map((m) => <div key={m.id + k} style={{ fontSize: 15, padding: "9px 6px", borderTop: `1px solid ${T.rule}`,
          color: m.id === sel ? T.ink : T.soft, fontWeight: m.id === sel ? 700 : 400, background: m.id === sel ? `${m.col}14` : "none" }}>{f(m)}</div>),
      ])}
      <div style={{ gridColumn: "1 / -1", fontSize: 14, color: T.dim, marginTop: 10 }}>KV cache in fp16 · toy 4-word prompt · Qwen3-8B caches 8 KV heads, the others 32</div>
    </div>
  );
}

// ------------------------------------------------------------ the view
const CODE = {
  vit: [
    "llava:     f = clip_vit(img).hidden_states[-2][:, 1:]   # [576, 1024]",
    "flamingo:  f = clip_vit(img).tokens                      # [256, 1024]",
    "qformer:   f = eva_vit_g(img)                            # [257, 1408]",
    "deepstack: f, taps = vit(patches)   # [784, 1152] + layers 8, 16, 24",
  ],
  bridge: [
    "llava:     v = W2(gelu(W1(f)))                     # [576, 4096]",
    "flamingo:  v = perceiver(f, latents)   # 6 layers → [64, 1024]",
    "qformer:   v = proj(qformer(queries, f))  # [32, 768] → [32, 4096]",
    "deepstack: v = merger(f.view(196, 4608))          # [196, 4096]",
    "deepstack: ds = [merger_k(t.view(196, 4608)) for t in taps]",
  ],
  inside: [
    "llava:     v = W2(gelu(W1(x)))",
    "flamingo:  kv = LN(cat([x, lat])); lat = lat + attn(LN(lat), kv); lat = lat + ffn(lat)",
    "qformer:   q = LN(q + self_attn(q)); q = LN(q + cross_attn(q, img)); q = LN(q + ffn(q))",
    "deepstack: v = W2(gelu(W1(LN(x).view(-1, 4608))))",
  ],
  enter: [
    "x = embed(input_ids)                    # [T, 4096]",
    "llava:     x[image_slots] = v           # 576 slots",
    "qformer:   x = cat([v, x])              # 32 tokens in front",
    "deepstack: x[image_slots] = v           # 196 slots · ds waits",
    "flamingo:  # one <image> token, v stays aside",
  ],
  layer: [
    "for i, layer in enumerate(llm.layers):",
    "flamingo:      if i % 4 == 0: h = h + tanh(α_i) * xattn(h, v)",
    "    h = layer(h)                       # causal self-attn + MLP",
    "deepstack:     if i < 3: h[image_slots] += ds[i]",
  ],
  prefill: ["logits, cache = llm(x, use_cache=True)   # whole prompt in one pass", "tok = logits[-1].argmax()"],
  dec: ["while tok != eos:", "    logits, cache = llm(embed(tok), past_key_values=cache)   # 1 row in", "    tok = logits[-1].argmax()"],
  frozen: ["vit.requires_grad_(False)   # every method, at least at first", "flamingo:  x-attn gates α start at 0 → the LLM starts unchanged"],
};

const FLOW = [
  { group: "vision encoder", items: [{ id: "vit", label: "image → ViT", sub: "patch vectors", code: CODE.vit }] },
  { group: "bridge", items: [
    { id: "bridge", label: "squeeze + translate", sub: "MLP · Perceiver · Q-Former · merge", code: CODE.bridge },
    { id: "inside", label: "one layer inside", sub: "the architecture of each", code: CODE.inside },
  ] },
  {
    group: "LLM decoder",
    items: [
      { id: "enter", label: "where the image goes", sub: "input or side memory", code: CODE.enter },
      { id: "layer", label: "up the layers", sub: "how text meets image", code: CODE.layer },
    ],
  },
  {
    group: "generate", loop: "until </s>",
    items: [
      { id: "prefill", label: "prefill → “a”", sub: "whole prompt, fill KV cache", code: CODE.prefill },
      { id: "dec1", label: "decode → “dog”", sub: "1 token reads the cache", code: CODE.dec },
      { id: "dec2", label: "decode → “</s>”", sub: "stop", code: CODE.dec },
    ],
  },
  { group: "training", items: [{ id: "frozen", label: "frozen ❄ / trained 🔥", sub: "per stage", code: CODE.frozen }] },
  { id: "compare", label: "side by side", sub: "tokens · cache · cost" },
];

const STEPS = [
  { id: "vit", block: "vit", caption: "image → vision encoder → patch vectors" },
  { id: "bridge", block: "bridge", caption: "squeeze and translate for the LLM" },
  { id: "inside", block: "inside", caption: "one layer inside the bridge" },
  { id: "enter", block: "enter", caption: "where the image goes into the LLM" },
  { id: "layer", block: "layer", caption: "up the layers: what “?” looks at" },
  { id: "prefill", block: "prefill", caption: "prefill: the whole prompt at once → KV cache → “a”" },
  { id: "dec1", block: "dec1", caption: "decode: “a” goes in, reads the cache → “dog”" },
  { id: "dec2", block: "dec2", caption: "decode: “dog” goes in → “</s>”, stop" },
  { id: "frozen", block: "frozen", caption: "what is frozen ❄ and what is trained 🔥" },
  { id: "compare", block: "compare", caption: "side by side" },
];

function Scene({ st, m }) {
  const pic = usePic(), p = useProgress(st.id + m.id, st.id.startsWith("dec") || st.id === "prefill" || st.id === "layer" || st.id === "inside" ? 3200 : 2200);
  if (st.id === "compare") return <Compare sel={m.id} />;
  const L = { vit: VitLane, bridge: BridgeLane, inside: InsideLane, enter: EnterLane, layer: LayerLane, frozen: FrozenLane }[st.id];
  const s = { prefill: 0, dec1: 1, dec2: 2 }[st.id];
  return <Solo m={m} stepId={st.id}>{L ? <L m={m} p={p} pic={pic} /> : <GenLane m={m} p={p} pic={pic} s={s} />}</Solo>;
}

const Tag = ({ m }) => <span style={{ display: "inline-block", minWidth: 92, color: m.col, fontWeight: 700 }}>{m.name}</span>;
const SelCtx = createContext(null);       // the picked method: the matmul card shows only its lines
const ML = ({ m, children }) => useContext(SelCtx) !== m.id ? null : <div style={{ display: "grid", gridTemplateColumns: "96px minmax(0, 1fr)", alignItems: "baseline", fontFamily: mono, fontSize: 15 }}><Tag m={m} /><div style={{ display: "grid", gap: 2 }}>{children}</div></div>;
const [LV, FL, QF, DS] = METHODS;

function matmuls(st) {
  const body = {
    vit: <>
      <ML m={LV}><MM a={["patches", [576, 588]]} b={["W_patch", [588, 1024]]} c={["f", [576, 1024]]} note="14×14×3 = 588 · 24 blocks, keep the 23rd" /></ML>
      <ML m={FL}><MM a={["patches", [256, 588]]} b={["W_patch", [588, 1024]]} c={["f", [256, 1024]]} note="24 blocks · patch tokens of the last layer" /></ML>
      <ML m={QF}><MM a={["patches", [256, 588]]} b={["W_patch", [588, 1408]]} c={["f", [257, 1408]]} note="+ CLS · 40 blocks, keep the 39th" /></ML>
      <ML m={DS}><MM a={["patches", [784, 1536]]} b={["W_patch", [1536, 1152]]} c={["f", [784, 1152]]} note="2 frames × 16×16×3 = 1536 · 27 blocks · taps after 8, 16, 24" /></ML>
    </>,
    bridge: <>
      <ML m={LV}><MM a={["f", [576, 1024]]} b={["W1", [1024, 4096]]} c={["h", [576, 4096]]} note="GELU" /><MM a={["h", [576, 4096]]} b={["W2", [4096, 4096]]} c={["v", [576, 4096]]} /></ML>
      <ML m={FL}><MM a={["latents", [64, 1024]]} b={["W_q", [1024, 512]]} c={["q", [64, 512]]} times={6} /><MM a={["[f; latents]", [320, 1024]]} b={["W_kv", [1024, 1024]]} c={["k, v", [320, 1024]]} times={6} /><MM a={["q", [8, 64, 64]]} b={["kᵀ", [8, 64, 320]]} c={["S", [8, 64, 320]]} times={6} note="softmax → @ v → out + FFN → v [64, 1024]" /></ML>
      <ML m={QF}><MM a={["queries", [32, 768]]} b={["W_q", [768, 768]]} c={["q", [32, 768]]} times={6} note="cross-attn in every 2nd of 12 layers" /><MM a={["f", [257, 1408]]} b={["W_kv", [1408, 1536]]} c={["k, v", [257, 1536]]} times={6} /><MM a={["q", [12, 32, 64]]} b={["kᵀ", [12, 64, 257]]} c={["S", [12, 32, 257]]} times={6} /><MM a={["out", [32, 768]]} b={["W_proj", [768, 4096]]} c={["v", [32, 4096]]} /></ML>
      <ML m={DS}><MMNote>reshape: [784, 1152] → 2×2 neighbours → [196, 4608]</MMNote><MM a={["LN(x)", [196, 4608]]} b={["W1", [4608, 4608]]} c={["h", [196, 4608]]} times={4} note="GELU · × 4 mergers: final + 3 taps" /><MM a={["h", [196, 4608]]} b={["W2", [4608, 4096]]} c={["v", [196, 4096]]} times={4} /></ML>
    </>,
    inside: <>
      <ML m={LV}><MM a={["x", [576, 1024]]} b={["W1", [1024, 4096]]} c={["h", [576, 4096]]} note="GELU" /><MM a={["h", [576, 4096]]} b={["W2", [4096, 4096]]} c={["v", [576, 4096]]} /></ML>
      <ML m={FL}><MM a={["LN(lat)", [64, 1024]]} b={["W_q", [1024, 512]]} c={["q", [64, 512]]} /><MM a={["LN([x; lat])", [320, 1024]]} b={["W_kv", [1024, 1024]]} c={["k, v", [320, 1024]]} /><MM a={["q", [8, 64, 64]]} b={["kᵀ", [8, 64, 320]]} c={["S", [8, 64, 320]]} note="÷ 8, softmax over the 320" /><MM a={["A", [8, 64, 320]]} b={["v", [8, 320, 64]]} c={["o", [8, 64, 64]]} note="concat heads → [64, 512]" /><MM a={["o", [64, 512]]} b={["W_o", [512, 1024]]} c={["lat +", [64, 1024]]} /><MM a={["LN(lat)", [64, 1024]]} b={["W_1", [1024, 4096]]} c={["h", [64, 4096]]} note="GELU" /><MM a={["h", [64, 4096]]} b={["W_2", [4096, 1024]]} c={["lat +", [64, 1024]]} /></ML>
      <ML m={QF}><MM a={["q", [32, 768]]} b={["W_qkv", [768, 2304]]} c={["q, k, v", [32, 2304]]} note="self-attn: S [12, 32, 32] → + q → LN" /><MM a={["q", [32, 768]]} b={["W_q", [768, 768]]} c={["q'", [32, 768]]} /><MM a={["img", [257, 1408]]} b={["W_kv", [1408, 1536]]} c={["k, v", [257, 1536]]} /><MM a={["q'", [12, 32, 64]]} b={["kᵀ", [12, 64, 257]]} c={["S", [12, 32, 257]]} note="softmax → @ v → W_o → + q → LN" /><MM a={["q", [32, 768]]} b={["W_1", [768, 3072]]} c={["h", [32, 3072]]} note="GELU" /><MM a={["h", [32, 3072]]} b={["W_2", [3072, 768]]} c={["q +", [32, 768]]} note="→ LN" /></ML>
      <ML m={DS}><MMNote>LN on each [784, 1152] → view → [196, 4608]</MMNote><MM a={["x", [196, 4608]]} b={["W1", [4608, 4608]]} c={["h", [196, 4608]]} note="GELU" /><MM a={["h", [196, 4608]]} b={["W2", [4608, 4096]]} c={["v", [196, 4096]]} /></ML>
    </>,
    enter: <>
      <ML m={LV}><MMNote>embed(ids) [4, 4096] · v [576, 4096] into the image slots → x [580, 4096]</MMNote></ML>
      <ML m={FL}><MMNote>embed(ids) [5, 4096] (one is &lt;image&gt;) → x [5, 4096] · v [64, 1024] waits aside</MMNote></ML>
      <ML m={QF}><MMNote>v [32, 4096] in front of embed(ids) [4, 4096] → x [36, 4096]</MMNote></ML>
      <ML m={DS}><MMNote>v [196, 4096] into the image slots → x [200, 4096] · ds 3 × [196, 4096] wait</MMNote></ML>
    </>,
    layer: <>
      <ML m={LV}><MM a={["q", [32, 580, 128]]} b={["kᵀ", [32, 128, 580]]} c={["S", [32, 580, 580]]} times={32} note="causal mask: text sees all image tokens" /></ML>
      <ML m={FL}><MM a={["h", [5, 4096]]} b={["W_q", [4096, 512]]} c={["q", [5, 512]]} times={8} /><MM a={["v", [64, 1024]]} b={["W_kv", [1024, 1024]]} c={["k, v", [64, 1024]]} times={8} /><MM a={["q", [8, 5, 64]]} b={["kᵀ", [8, 64, 64]]} c={["S", [8, 5, 64]]} times={8} note="→ out [5, 4096] × tanh(α) + h · then the normal self-attn [32, 5, 5]" /></ML>
      <ML m={QF}><MM a={["q", [32, 36, 128]]} b={["kᵀ", [32, 128, 36]]} c={["S", [32, 36, 36]]} times={32} /></ML>
      <ML m={DS}><MM a={["q", [32, 200, 128]]} b={["kᵀ", [8, 128, 200]]} c={["S", [32, 200, 200]]} times={36} note="each K head shared by 4 Q heads · after layers 1–3: h[img] [196, 4096] += ds[i]" /></ML>
    </>,
    prefill: <>
      {METHODS.map((m) => <ML key={m.id} m={m}><MMNote>K, V: 2 × [{m.L} layers, {realKeys(m, 0)}, {m.kv}]{m.id === "flamingo" ? " + x-attn 2 × [8, 64, 512]" : ""} = {fmtMB(kvMB(m, realKeys(m, 0)))}</MMNote></ML>)}
    </>,
    dec1: null,
    dec2: null,
    frozen: <MMNote>no new math · gradients flow only into the 🔥 parts</MMNote>,
    compare: <MMNote>image tokens in the sequence cost cache memory and attention reads on every new token</MMNote>,
  };
  const dec = (s) => <>
    {METHODS.map((m) => {
      const k = realKeys(m, s), hq = m.kv === 4096 ? 32 : 8;
      return (
        <ML key={m.id} m={m}>
          <MM a={["q", [32, 1, 128]]} b={["kᵀ", [hq, 128, k]]} c={["S", [32, 1, k]]} times={m.L} note={m.id === "flamingo" ? "+ x-attn [8, 1, 64] in 8 layers" : undefined} />
          <MM a={["h", [1, m.D]]} b={["W_vocab", [m.D, m.vocab]]} c={["logits", [1, m.vocab]]} />
        </ML>
      );
    })}
  </>;
  return <MatMulCard>{body[st.id] ?? dec(st.id === "dec1" ? 1 : 2)}</MatMulCard>;
}

// keep only this method's code lines ("llava: …" lines of the others are dropped)
const flowFor = (id) => FLOW.map((b) => {
  const fix = (it) => (it.code ? { ...it, code: it.code.filter((l) => !METHODS.some((o) => o.id !== id && l.startsWith(`${o.id}:`))) } : it);
  return b.items ? { ...b, items: b.items.map(fix) } : fix(b);
});

export default function VlmView() {
  const [sel, setSel] = useState(() => {
    try { return METHODS.find((m) => m.id === localStorage.getItem("vlm-arch"))?.id ?? "llava"; } catch { return "llava"; }
  });
  const pick = (id) => { setSel(id); try { localStorage.setItem("vlm-arch", id); } catch { /* ignore */ } };
  const m = METHODS.find((x) => x.id === sel);
  // each sub-tab keeps its own step: first visit starts at step 1, later visits resume
  const [stepOf, setStepOf] = useState(() => {
    try { return JSON.parse(localStorage.getItem("vlm-arch-steps")) ?? {}; } catch { return {}; }
  });
  const setStep = (i) => setStepOf((o) => {
    const n = { ...o, [sel]: i };
    try { localStorage.setItem("vlm-arch-steps", JSON.stringify(n)); } catch { /* ignore */ }
    return n;
  });
  const flow = useMemo(() => flowFor(sel), [sel]);
  return (
    <SelCtx.Provider value={sel}>
      <StepShell
        title="Vision → LLM"
        subtitle="toy picture and probabilities · shapes are the real ones"
        blocks={flow}
        steps={STEPS}
        index={stepOf[sel] ?? 0}
        onIndex={setStep}
        scene={(st) => <div style={{ minHeight: 420 }}><Scene key={st.id + sel} st={st} m={m} /></div>}
        matmuls={matmuls}
        controls={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {METHODS.map((x) => (
              <button key={x.id} onClick={() => pick(x.id)} style={{
                ...btn(x.id === sel), fontSize: 15, padding: "6px 14px", fontWeight: x.id === sel ? 700 : 400,
                color: x.id === sel ? x.col : T.soft, borderColor: x.id === sel ? x.col : T.rule, background: x.id === sel ? `${x.col}18` : T.well,
              }}>{x.name}</button>
            ))}
          </div>
        }
      />
    </SelCtx.Provider>
  );
}
