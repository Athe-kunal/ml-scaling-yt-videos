import { createContext, useContext, useMemo, useState } from "react";
import { T, mono, btn } from "./theme";
import { useProgress, ease, clamp01, lerp } from "./anim";
import StepShell from "./StepShell";
import { MM, MMNote, MatMulCard } from "./draw";
import { SUBJECTS, renderSubject, mulberry } from "./model";

// The shared LLaVA-style path (ViT → 2×2 merge → MLP → sequence → LLM → text loss),
// shown for four current model families, one at a time (sub-tabs). One 448×448 image throughout.
// Toy picture: 4×4 patches; labels carry the real shapes (HF configs + reports).
//   Qwen3-VL-8B       SigLIP 2 ViT 27×1152, patch 16 · merger 4608→4096 · DeepStack taps 8/16/24
//   DeepSeek-VL2      SigLIP-SO400M 27×1152 on 384 tiles + thumbnail · pixel-shuffle 4608 → 2560 · ↵ + sep tokens
//   Kimi K3           MoonViT-V2 27×1024, patch 14, from scratch · 4096 → 7168 · NoPE, 3 KDA : 1 MLA
//   MiMo-V2.6-Flash   MiMo-ViT 28×1280 (4 global + 24 window) · 5120 → 4096 · SWA-128 / global backbone

const M4 = [
  {
    id: "qwen", name: "Qwen3-VL", model: "Qwen3-VL-8B", col: "#8DB4FF",
    vitL: 27, vitD: 1152, patch: 16, pp: 1536, nPatch: 784, side: 28, mergeD: 4608, D: 4096, L: 36, vocab: 151936,
    nVis: 196, seq: 202, heads: 32, init: "SigLIP 2", llmInit: "Qwen3",
    mlp: ["LN each 1152", "Linear 4608 → 4608", "GELU", "Linear 4608 → 4096"], w: "≈ 40M",
  },
  {
    id: "deepseek", name: "DeepSeek-VL2", model: "DeepSeek-VL2 · 27B MoE", col: "#6EF3DD",
    vitL: 27, vitD: 1152, patch: 14, pp: 588, nPatch: 3645, side: 27, mergeD: 4608, D: 2560, L: 30, vocab: 129280,
    nVis: 1023, seq: 1027, heads: 32, init: "SigLIP", llmInit: "DeepSeekMoE",
    mlp: ["(no norm)", "Linear 4608 → 2560", "GELU", "Linear 2560 → 2560"], w: "≈ 18M",
  },
  {
    id: "kimi", name: "Kimi K3", model: "Kimi K3 · 2.8T MoE", col: "#FFC94D",
    vitL: 27, vitD: 1024, patch: 14, pp: 588, nPatch: 1024, side: 32, mergeD: 4096, D: 7168, L: 93, vocab: 163840,
    nVis: 256, seq: 262, heads: 96, init: "random", llmInit: "random",
    mlp: ["norm each 1024", "Linear 4096 → 4096", "GELU", "Linear 4096 → 7168"], w: "≈ 46M",
  },
  {
    id: "mimo", name: "MiMo-V2.6", model: "MiMo-V2.6-Flash", col: "#FF8AC6",
    vitL: 28, vitD: 1280, patch: 16, pp: 1536, nPatch: 784, side: 28, mergeD: 5120, D: 4096, L: 48, vocab: 152576,
    nVis: 196, seq: 202, heads: 64, init: "random", llmInit: "MiMo",
    mlp: ["LN each 1280", "Linear 5120 → 5120", "GELU", "Linear 5120 → 4096"], w: "≈ 47M",
  },
];
const [QW, DS, KM, MI] = M4;

const W = 190, H = 440;
const seg = (p, a, b) => ease(clamp01((p - a) / (b - a)));
const rgb = (c) => `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`;
const GRAD = T.bad;

// dog picture: 4×4 and 8×8 patch colours
function usePic() {
  return useMemo(() => {
    const img = renderSubject(SUBJECTS[0], 64), d = img.data;
    const grid = (n) => {
      const s = 64 / n, out = [];
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const a = [0, 0, 0];
        for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) { const o = ((r * s + y) * 64 + c * s + x) * 4; a[0] += d[o]; a[1] += d[o + 1]; a[2] += d[o + 2]; }
        out.push(a.map((v) => v / (s * s) / 255));
      }
      return out;
    };
    const c4 = grid(4);
    const merged = [0, 1, 2, 3].map((g) => { const r0 = Math.floor(g / 2) * 2, c0 = (g % 2) * 2; return [0, 1, 4, 5].map((o) => c4[r0 * 4 + c0 + o]).reduce((a, c) => a.map((v, i) => v + c[i] / 4), [0, 0, 0]); });
    return { url: img.canvas.toDataURL(), c4, c8: grid(8), merged };
  }, []);
}

const Txt = ({ x, y, s, size = 14, c = T.dim, a = "middle", w = 400, op = 1 }) => (
  <text x={x} y={y} textAnchor={a} fontFamily={mono} fontSize={size} fontWeight={w} fill={c} opacity={op}>{s}</text>
);
const Sq = ({ x, y, w, h, c, stroke, op = 1, sw = 1.4 }) => (
  <rect x={x - w / 2} y={y - (h ?? w) / 2} width={w} height={h ?? w} rx={2} fill={c ? rgb(c) : "none"} stroke={stroke ?? "none"} strokeWidth={sw} opacity={op} />
);

// one model at a time: its lane, big, next to its key numbers (this step's rows lit)
const STEP_ROWS = {
  vit: ["resolution", "ViT", "ViT starts from"], merge: ["compress"], mlp: ["compress", "MLP → width", "448² image → tokens"],
  seq: ["448² image → tokens", "positions", "extra"], mask: ["LLM attention"], predict: ["LLM", "LLM attention", "extra"],
  loss: ["ViT starts from"], train: ["ViT starts from"],
};
function Solo({ m, stepId, render }) {
  const lit = STEP_ROWS[stepId] ?? [];
  return (
    <div style={{ display: "flex", gap: 28, alignItems: "flex-start", flexWrap: "wrap" }}>
      <div style={{ width: 300, flex: "none", background: T.well, border: `1px solid ${m.col}55`, borderRadius: 10, padding: "8px 6px 2px" }}>
        <div style={{ fontFamily: mono, fontSize: 18, fontWeight: 700, color: m.col, textAlign: "center" }}>{m.name}</div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", marginTop: 4 }}>{render(m)}</svg>
      </div>
      <div style={{ display: "grid", gap: 0, minWidth: 300, flex: "1 1 300px", maxWidth: 460, fontFamily: mono }}>
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

// a crop of the picture: tile (tr, tc) of an n×n split
const Crop = ({ url, x, y, size, n = 1, tr = 0, tc = 0, op = 1 }) => (
  <svg x={x} y={y} width={size} height={size} viewBox={`${(tc * 64) / n} ${(tr * 64) / n} ${64 / n} ${64 / n}`} opacity={op}>
    <image href={url} x={0} y={0} width={64} height={64} />
  </svg>
);

// ------------------------------------------------------------ 1 · image → patch vectors
function VitLane({ m, p, pic }) {
  const a = seg(p, 0, 0.3), b = clamp01((p - 0.3) / 0.3), c = seg(p, 0.6, 1);
  const ds = m.id === "deepseek", BX = 25, BY = 112, BW = 140, BH = 70, N = 8;
  const glob = m.id === "mimo" ? [0, 7] : [], taps = m.id === "qwen" ? [2, 5, 7] : [];
  return (
    <g>
      {ds ? <g>
        {[0, 1, 2, 3].map((t) => <Crop key={t} url={pic.url} n={2} tr={t >> 1} tc={t & 1} size={40} x={70 + (t & 1) * lerp(40, 46, a)} y={6 + (t >> 1) * lerp(40, 46, a)} op={1 - 0.6 * b} />)}
        <Crop url={pic.url} x={10} y={22} size={46} op={a * (1 - 0.6 * b)} />
        <Txt x={33} y={84} s="thumb" size={13} op={a} />
      </g> : <g>
        <image href={pic.url} x={55} y={6} width={80} height={80} opacity={1 - 0.6 * b} />
        {[1, 2, 3].map((k) => <g key={k} stroke={T.bg} strokeWidth={1} opacity={a}><line x1={55 + k * 20} x2={55 + k * 20} y1={6} y2={86} /><line x1={55} x2={135} y1={6 + k * 20} y2={6 + k * 20} /></g>)}
      </g>}
      <Txt x={95} y={104} s={ds ? "4 tiles + thumbnail" : "native 448 × 448"} size={13} c={T.soft} />
      <rect x={BX} y={BY} width={BW} height={BH} rx={8} fill={`${m.col}10`} stroke={m.col} strokeWidth={1.4} />
      {Array.from({ length: N }, (_, i) => {
        const lit = b * N > i, g = glob.includes(i), win = m.id === "mimo" && !g;
        const fill = lit ? (g ? m.col : `${m.col}88`) : T.rule;
        return win
          ? <g key={i}>{[0, 1, 2, 3].map((q) => <rect key={q} x={BX + 10 + q * 30.5} y={BY + 6 + i * 7.7} width={27} height={4} rx={2} fill={fill} />)}</g>
          : <rect key={i} x={BX + 10} y={BY + 6 + i * 7.7} width={BW - 20} height={4} rx={2} fill={taps.includes(i) && c > 0 ? m.col : fill} />;
      })}
      {taps.map((i) => <circle key={i} cx={BX + BW - 4} cy={BY + 8 + i * 7.7} r={3} fill={c > 0 ? m.col : "none"} />)}
      <Txt x={95} y={BY + BH + 18} s={`${m.vitL} blocks · ${m.vitD} wide`} c={T.soft} size={13} />
      {ds ? <g opacity={c}>
        {pic.c4.map((col, k) => <Sq key={k} x={16 + (k % 4) * 9} y={224 + Math.floor(k / 4) * 9} w={8} c={col} />)}
        <Txt x={30} y={272} s="thumb" size={13} />
        {pic.c8.map((col, k) => { const r = Math.floor(k / 8), cc = k % 8; return <Sq key={k} x={84 + cc * 9 + (cc >= 4 ? 5 : 0)} y={214 + r * 9 + (r >= 4 ? 5 : 0)} w={8} c={col} />; })}
        <Txt x={120} y={300} s="4 tiles" size={13} />
      </g> : pic.c4.map((col, k) => {
        const [sx, sy] = [55 + (k % 4) * 20 + 10, 6 + Math.floor(k / 4) * 20 + 10];
        const [mx, my] = [BX + 6 + (k + 0.5) * ((BW - 12) / 16), BY];
        const [ex, ey] = [68 + (k % 4) * 17, 222 + Math.floor(k / 4) * 17];
        const x = c > 0 ? lerp(mx, ex, c) : lerp(sx, mx, a), y = c > 0 ? lerp(BY + BH, ey, c) : lerp(sy, my, a);
        return <Sq key={k} x={x} y={y} w={c > 0 ? lerp(7, 14, c) : lerp(18, 7, a)} c={col} op={b > 0 && b < 1 && c === 0 ? 0 : 1} />;
      })}
      <g opacity={c}>
        <Txt x={95} y={330} s={ds ? "5 × [729, 1152]" : `[${m.nPatch}, ${m.vitD}]`} size={15} w={700} c={m.col} />
        <Txt x={95} y={350} s={ds ? "27 × 27 per tile" : `${m.side} × ${m.side} patches`} size={13} c={T.soft} />
        <Txt x={95} y={368} s={`patch ${m.patch} px`} size={13} />
        {m.id === "qwen" && <Txt x={95} y={396} s="+ taps 8 · 16 · 24" size={13} c={m.col} />}
        {m.id === "mimo" && <Txt x={95} y={396} s="4 global + 24 window" size={13} c={m.col} />}
        {ds && <Txt x={95} y={396} s="tile = 384 × 384" size={13} c={m.col} />}
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 1 · image → patch vectors, explicitly
// one real patch of the picture: its pixels → flatten → × W_patch → one vector,
// then all patches → [N, d] → ViT. DeepSeek first picks a view (tile or thumbnail).
const VIEW = { qwen: "2D RoPE", deepseek: "learned 27×27", kimi: "learned + RoPE", mimo: "2D RoPE" };
function PatchScene({ m }) {
  const ds = m.id === "deepseek", P = m.patch, frames = m.pp / (P * P * 3);
  const [view, setView] = useState(0); // deepseek: tiles 0..3, 4 = thumbnail
  const S = ds ? 384 : 448, g = Math.floor(S / P);
  const [cell, setCell] = useState(null);
  const p = useProgress(`patch-${m.id}-${view}`, 4200);
  const a = seg(p, 0, 0.18), b = seg(p, 0.18, 0.4), f = seg(p, 0.4, 0.6), d = seg(p, 0.6, 0.8), e = seg(p, 0.8, 1);
  // the model's input view as a canvas: 448 image, or (deepseek) a 384 tile of the 768 resize / the 384 thumbnail
  const input = useMemo(() => {
    const src = renderSubject(SUBJECTS[0], 448).canvas, cv = document.createElement("canvas");
    cv.width = cv.height = S;
    const x = cv.getContext("2d");
    if (!ds) x.drawImage(src, 0, 0);
    else if (view === 4) x.drawImage(src, 0, 0, 384, 384);
    else x.drawImage(src, -(view & 1) * 384, -(view >> 1) * 384, 768, 768);
    return { url: cv.toDataURL(), ctx: x };
  }, [ds, view, S]);
  // default patch: the most varied one (an eye or ear edge), so the zoom shows real structure
  const busiest = useMemo(() => {
    const all = input.ctx.getImageData(0, 0, g * P, g * P).data;
    let best = [0, 0], bv = -1;
    for (let rr = 0; rr < g; rr++) for (let cc = 0; cc < g; cc++) {
      let s1 = 0, s2 = 0, n = 0;
      for (let y = 0; y < P; y += 2) for (let x = 0; x < P; x += 2) {
        const o = ((rr * P + y) * g * P + cc * P + x) * 4, v = all[o] + all[o + 1] + all[o + 2];
        s1 += v; s2 += v * v; n++;
      }
      const vr = s2 / n - (s1 / n) ** 2;
      if (vr > bv) { bv = vr; best = [rr, cc]; }
    }
    return best;
  }, [input, g, P]);
  const [r, c] = cell ?? busiest;
  const px = useMemo(() => input.ctx.getImageData(c * P, r * P, P, P).data, [input, r, c, P]);
  const vals = useMemo(() => { const v = []; for (let i = 0; i < P * P; i++) v.push(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]); return v; }, [px, P]);
  const out = useMemo(() => {
    const rnd = mulberry(7), w = Array.from({ length: 40 * 24 }, () => rnd() - 0.5);
    return Array.from({ length: 40 }, (_, j) => vals.slice(0, 24 * 12).reduce((s, v, k) => s + (v / 255) * w[j * 24 + (k % 24)], 0) / 12);
  }, [vals]);
  const mx = Math.max(...out.map(Math.abs)) || 1;
  // layout (viewBox 900 × 450)
  const IX = 10, IY = 34, IS = 200, cs = IS / g, ZX = 248, ZY = 34, ZS = 170, zc = ZS / P;
  const hx = IX + c * cs, hy = IY + r * cs;
  const VX = 452, nCells = 64, vw = 3.6, VB = 700;
  const chColor = (k, v) => (k % 3 === 0 ? `rgb(${v},40,40)` : k % 3 === 1 ? `rgb(40,${v},40)` : `rgb(40,40,${v})`);
  const pick = (ev) => {
    const box = ev.currentTarget.getBoundingClientRect(), sc = VB / box.width;
    const x = (ev.clientX - box.left) * sc - IX, y = (ev.clientY - box.top) * sc - IY;
    if (x >= 0 && y >= 0 && x < g * cs && y < g * cs) setCell([Math.floor(y / cs), Math.floor(x / cs)]);
  };
  const chip = (x, w, label, sub, col = T.soft, op = 1) => (
    <g opacity={op}>
      <rect x={x} y={334} width={w} height={48} rx={7} fill={`${col}14`} stroke={col} strokeWidth={1.3} />
      <Txt x={x + w / 2} y={354} s={label} size={15} w={700} c={T.ink} />
      {sub && <Txt x={x + w / 2} y={373} s={sub} size={13} c={col} />}
    </g>
  );
  const arrow = (x1, x2, op) => <path d={`M${x1},358 H${x2 - 6} l-5,-4 v8 z`} stroke={T.dim} fill={T.dim} opacity={op} />;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {ds && (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", fontFamily: mono, fontSize: 14, color: T.soft }}>
          <span>view:</span>
          {["tile 1", "tile 2", "tile 3", "tile 4", "thumbnail"].map((t, i) => (
            <button key={t} onClick={() => { setView(i); setCell(null); }} style={{ ...btn(view === i), fontSize: 14, color: view === i ? m.col : T.soft, borderColor: view === i ? m.col : T.rule }}>{t}</button>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${VB} 392`} width="100%" style={{ maxWidth: 780, display: "block", cursor: "crosshair" }} onClick={pick}>
        {/* 1 · the image and its patch grid */}
        <Txt x={IX} y={22} a="start" s={ds ? (view === 4 ? "thumbnail 384 × 384" : `tile ${view + 1} · 384 × 384`) : "image 448 × 448"} size={15} w={700} c={T.ink} />
        <image href={input.url} x={IX} y={IY} width={IS} height={IS} />
        <g opacity={a} stroke={T.bg} strokeWidth={0.7}>
          {Array.from({ length: g + 1 }, (_, k) => <g key={k}><line x1={IX + k * cs} x2={IX + k * cs} y1={IY} y2={IY + g * cs} /><line x1={IX} x2={IX + g * cs} y1={IY + k * cs} y2={IY + k * cs} /></g>)}
        </g>
        <rect x={hx} y={hy} width={cs} height={cs} fill="none" stroke={m.col} strokeWidth={2.2} opacity={a} />
        <Txt x={IX} y={IY + IS + 24} a="start" s={`${g} × ${g} = ${g * g} patches`} size={15} w={700} c={m.col} op={a} />
        <Txt x={IX} y={IY + IS + 44} a="start" s={`each ${P} × ${P} px · click one`} size={14} c={T.soft} op={a} />
        {/* 2 · zoom: the patch's real pixels */}
        <g opacity={b}>
          <line x1={hx + cs} y1={hy} x2={ZX} y2={ZY} stroke={m.col} strokeOpacity={0.5} />
          <line x1={hx + cs} y1={hy + cs} x2={ZX} y2={ZY + ZS} stroke={m.col} strokeOpacity={0.5} />
          {Array.from({ length: P * P }, (_, k) => (
            <rect key={k} x={ZX + (k % P) * zc} y={ZY + Math.floor(k / P) * zc} width={zc - 0.6} height={zc - 0.6} fill={`rgb(${px[k * 4]},${px[k * 4 + 1]},${px[k * 4 + 2]})`} />
          ))}
          <rect x={ZX} y={ZY} width={ZS} height={ZS} fill="none" stroke={m.col} strokeWidth={1.5} />
          <Txt x={ZX + ZS / 2} y={ZY + ZS + 24} s={`${P} × ${P} pixels`} size={15} w={700} c={T.ink} />
          <Txt x={ZX + ZS / 2} y={ZY + ZS + 44} s="× 3 colours (R, G, B)" size={14} c={T.soft} />
        </g>
        {/* 3 · flatten into one long list of numbers */}
        <g opacity={f}>
          <Txt x={VX} y={22} a="start" s="flatten → numbers" size={15} w={700} c={T.ink} />
          {vals.slice(0, nCells).map((v, k) => <rect key={k} x={VX + k * vw} y={36} width={vw - 0.5} height={24} fill={chColor(k, v)} />)}
          <Txt x={VX + nCells * vw + 4} y={53} a="start" s="…" size={15} c={T.soft} />
          <Txt x={VX} y={82} a="start" s={`${P}·${P}·3 = ${P * P * 3}`} size={15} c={T.soft} />
          {frames > 1 && <Txt x={VX} y={101} a="start" s={`× 2 frames = ${m.pp}`} size={14} c={m.col} />}
        </g>
        {/* 4 · one matmul: × W_patch → the patch vector */}
        <g opacity={d}>
          <path d={`M${VX + 120},${frames > 1 ? 108 : 90} V116`} stroke={T.dim} />
          <rect x={VX} y={116} width={240} height={56} rx={8} fill={`${m.col}12`} stroke={m.col} strokeWidth={1.4} />
          <Txt x={VX + 120} y={139} s={`[1, ${m.pp}] × W_patch`} size={15} w={700} c={T.ink} />
          <Txt x={VX + 120} y={160} s={`W: [${m.pp}, ${m.vitD}]`} size={14} c={m.col} />
          <path d={`M${VX + 120},172 V184`} stroke={T.dim} />
          {out.slice(0, 32).map((v, j) => {
            const h = (Math.abs(v) / mx) * 24;
            return <rect key={j} x={VX + j * 7.5} y={v >= 0 ? 212 - h : 212} width={6} height={Math.max(1, h)} fill={m.col} opacity={0.85} />;
          })}
          <line x1={VX} x2={VX + 240} y1={212} y2={212} stroke={T.rule} />
          <Txt x={VX} y={258} a="start" s={`patch vector [${m.vitD}]`} size={15} w={700} c={m.col} />
          <Txt x={VX} y={278} a="start" s="same W for every patch" size={14} c={T.soft} />
        </g>
        {/* 5 · all patches at once → ViT */}
        <g>
          <Txt x={IX} y={320} a="start" s={`same for all ${g * g} patches${ds ? " · in each of the 5 views" : ""}`} size={15} w={700} c={T.ink} op={e} />
          {chip(IX, 122, `[${g * g}, ${m.pp}]`, "pixels", T.soft, e)}
          {arrow(IX + 122, IX + 142, e)}
          {chip(IX + 142, 122, `[${g * g}, ${m.vitD}]`, "× W_patch", m.col, e)}
          {arrow(IX + 264, IX + 284, e)}
          {chip(IX + 284, 128, "+ positions", VIEW[m.id], T.soft, e)}
          {arrow(IX + 412, IX + 432, e)}
          {chip(IX + 432, 112, `${m.vitL} blocks`, `ViT · ${m.vitD}`, m.col, e)}
          {arrow(IX + 544, IX + 562, e)}
          {chip(IX + 562, 124, ds ? "5×[729,1152]" : `[${g * g}, ${m.vitD}]`, "patch vectors", m.col, e)}
        </g>
      </svg>
    </div>
  );
}

// ------------------------------------------------------------ 1 · 2×2 neighbours → one vector
const barW = (d) => (d / 7168) * 170;
function MergeLane({ m, p, pic }) {
  const t1 = seg(p, 0.05, 0.4), t2 = seg(p, 0.45, 0.9), bw = barW(m.mergeD), sw = bw / 4;
  const ds = m.id === "deepseek", X0 = 61, Y0 = 14;
  return (
    <g>
      {pic.c4.map((col, k) => {
        const r = Math.floor(k / 4), c = k % 4, g = (r >> 1) * 2 + (c >> 1), o = (r & 1) * 2 + (c & 1);
        const p0 = [X0 + c * 17 + 7, Y0 + r * 17 + 7];
        const gc = [X0 + (c >> 1) * 34 + 15.5, Y0 + (r >> 1) * 34 + 15.5], sp = [gc[0] + (gc[0] - 95) * 0.5, gc[1] + (gc[1] - 47) * 0.5];
        const p1 = [sp[0] + ((c & 1) - 0.5) * 13, sp[1] + ((r & 1) - 0.5) * 13];
        const p2 = [95 - bw / 2 + o * sw + sw / 2, 132 + g * 32];
        const x = t2 > 0 ? lerp(p1[0], p2[0], t2) : lerp(p0[0], p1[0], t1), y = t2 > 0 ? lerp(p1[1], p2[1], t2) : lerp(p0[1], p1[1], t1);
        const w = t2 > 0 ? lerp(12, sw - 0.6, t2) : lerp(14, 12, t1), h = t2 > 0 ? lerp(12, 20, t2) : lerp(14, 12, t1);
        return <Sq key={k} x={x} y={y} w={w} h={h} c={col} sw={1} stroke={t2 > 0.95 ? m.col : undefined} />;
      })}
      {[0, 1, 2, 3].map((g) => <rect key={g} x={95 - bw / 2 - 1} y={121 + g * 32} width={bw + 2} height={22} rx={3} fill="none" stroke={m.col} strokeWidth={1.4} opacity={seg(p, 0.85, 1)} />)}
      <Txt x={95} y={110} s={`4 × ${m.vitD} = ${m.mergeD}`} size={13} c={m.col} w={700} op={seg(p, 0.6, 0.9)} />
      <g opacity={seg(p, 0.75, 1)}>
        <Txt x={95} y={278} s={ds ? "5 × [196, 4608]" : `[${m.nVis}, ${m.mergeD}]`} size={15} w={700} c={m.col} />
        <Txt x={95} y={298} s={ds ? "27 → pad 28 → 14" : `${m.side} → ${m.side / 2} per side`} size={13} c={T.soft} />
        <Txt x={95} y={316} s="4× fewer tokens" size={13} />
        {m.id === "qwen" && <Txt x={95} y={350} s="taps 8·16·24: same" size={13} c={m.col} />}
        {ds && <Txt x={95} y={350} s="every tile + thumbnail" size={13} c={m.col} />}
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 1 · MLP → LLM width
function MlpLane({ m, p, pic }) {
  const sy = lerp(-10, 250, clamp01(p / 0.8)), lit = (y) => sy >= y, bi = barW(m.mergeD), bo = barW(m.D);
  const ys = [56, 96, 136, 176];
  return (
    <g>
      {[0, 1, 2, 3].map((o) => <rect key={o} x={95 - bi / 2 + (o * bi) / 4} y={8} width={bi / 4 - 0.6} height={20} rx={2} fill={rgb(pic.c4[[0, 1, 4, 5][o]])} />)}
      <rect x={95 - bi / 2 - 1} y={7} width={bi + 2} height={22} rx={3} fill="none" stroke={m.col} strokeWidth={1.4} />
      <Txt x={95} y={46} s={`${m.mergeD}`} size={13} c={T.soft} />
      {m.mlp.map((s, i) => {
        const on = lit(ys[i]), now = on && sy < ys[i] + 40, none = s.startsWith("(");
        return (
          <g key={i}>
            <line x1={95} x2={95} y1={ys[i] - 6} y2={ys[i]} stroke={on ? m.col : T.rule} strokeWidth={1.4} />
            <rect x={8} y={ys[i]} width={174} height={28} rx={7} fill={now ? `${m.col}30` : on ? `${m.col}14` : T.bg} stroke={on && !none ? m.col : T.rule} strokeWidth={now ? 2.2 : 1.3} strokeDasharray={none ? "4 3" : undefined} />
            <Txt x={95} y={ys[i] + 19} s={s} size={14} w={none ? 400 : 700} c={none ? T.dim : on ? T.ink : T.dim} />
          </g>
        );
      })}
      <line x1={95} x2={95} y1={204} y2={218} stroke={lit(218) ? m.col : T.rule} strokeWidth={1.4} />
      {sy > 0 && sy < 210 && <circle cx={95} cy={sy + 14} r={4} fill={m.col} />}
      <g opacity={seg(p, 0.7, 0.85)}>
        <rect x={95 - bo / 2} y={220} width={bo} height={20} rx={3} fill={rgb(pic.merged[0])} stroke={m.col} strokeWidth={1.6} />
        <Txt x={95} y={258} s={`${m.D}`} size={15} w={700} c={m.col} />
        <Txt x={95} y={276} s="= LLM width" size={13} />
      </g>
      <g opacity={seg(p, 0.85, 1)}>
        <Txt x={95} y={312} s={m.id === "deepseek" ? "[980, 2560]" : `[${m.nVis}, ${m.D}]`} size={15} w={700} c={T.ink} />
        {m.id === "deepseek" && <Txt x={95} y={332} s="+ 42 ↵ + 1 sep = 1023" size={13} c={m.col} />}
        {m.id === "qwen" && <Txt x={95} y={332} s="+ 3 taps × [196, 4096]" size={13} c={m.col} />}
        <Txt x={95} y={370} s={`${m.nVis}`} size={26} w={700} c={m.col} />
        <Txt x={95} y={390} s="image tokens" size={13} />
        <Txt x={95} y={424} s={`${m.w} weights`} size={13} c={T.soft} />
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 2 · the sequence
// toy lines of chips; DeepSeek's thumbnail and tiles end each row with ↵
function seqLines(m) {
  const I = (k, pos) => ({ t: "img", k, pos }), B = (s, pos) => ({ t: "bnd", s, pos }), X = (s, pos) => ({ t: "txt", s, pos });
  const words = ["What", "is", "this", "?"];
  if (m.id === "deepseek") {
    const lines = [], nl = (pos) => ({ t: "nl", s: "↵", pos });
    let n = 0;
    for (let r = 0; r < 2; r++) lines.push([I(r * 2, n++), I(r * 2 + 1, n++), nl(n++)]);
    lines.push([{ t: "sep", s: "‖", pos: n++ }]);
    for (let r = 0; r < 4; r++) lines.push([...[0, 1, 2, 3].map((c) => ({ t: "img8", k: r * 2 * 8 + c * 2, pos: n++ })), nl(n++)]);
    lines.push(words.map((w) => X(w, n++)));
    return lines;
  }
  const q = m.id === "qwen", k = m.id === "kimi";
  const pos = (i, h, w) => (k ? "" : q ? `1,${h},${w}` : `${i}`);
  return [
    [B("▶", k ? "" : "0")],
    [I(0, pos(1, 1, 1)), I(1, pos(2, 1, 2))],
    [I(2, pos(3, 2, 1)), I(3, pos(4, 2, 2))],
    [B("◀", k ? "" : q ? "3" : "5")],
    words.map((w, i) => X(w, k ? "" : `${(q ? 4 : 6) + i}`)),
  ];
}
function SeqLane({ m, p, pic }) {
  const lines = seqLines(m), ds = m.id === "deepseek", q = m.id === "qwen";
  const LH = ds ? 22 : 30, CH = ds ? 18 : 22;
  let idx = 0;
  const total = lines.flat().length;
  return (
    <g>
      {lines.map((ln, li) => {
        const isTxt = ln[0].t === "txt", y = 14 + li * LH + (isTxt ? 8 : 0);
        const ws = ln.map((c) => (c.t === "txt" ? 40 : ds ? 20 : q && c.t === "img" ? 46 : 28));
        const tw = ws.reduce((a, b) => a + b + 3, -3);
        let x = 95 - tw / 2;
        return ln.map((c, ci) => {
          const w = ws[ci], x0 = x, op = seg(p, (idx / total) * 0.6, (idx / total) * 0.6 + 0.12);
          x += w + 3; idx += 1;
          const col = c.t === "img" ? pic.c4[[0, 1, 4, 5][c.k] ?? 0] : c.t === "img8" ? pic.c8[c.k] : null;
          const fill = col ? rgb(col) : c.t === "txt" ? T.rule : `${m.col}33`;
          return (
            <g key={`${li}.${ci}`} opacity={op}>
              <rect x={x0} y={y} width={w} height={CH} rx={3} fill={fill} stroke={col ? "none" : m.col} strokeWidth={c.t === "txt" ? 0 : 1.2} />
              {c.s && <Txt x={x0 + w / 2} y={y + CH / 2 + 5} s={c.s} size={c.t === "txt" ? 13 : 14} c={c.t === "txt" ? T.ink : m.col} w={700} />}
              {c.pos !== "" && c.pos != null && !c.s && <Txt x={x0 + w / 2} y={y + CH / 2 + 5} s={c.pos} size={13} c={T.bg} w={700} />}
              {c.pos !== "" && c.pos != null && c.s && c.t !== "txt" && c.t !== "nl" && c.t !== "sep" && <Txt x={x0 + w + 4} y={y + CH / 2 + 5} s={c.pos} size={13} c={T.dim} a="start" />}
              {c.t === "txt" && c.pos !== "" && <Txt x={x0 + w / 2} y={y + CH + 15} s={c.pos} size={13} c={T.dim} />}
            </g>
          );
        });
      })}
      <g opacity={seg(p, 0.7, 1)}>
        <Txt x={95} y={300} s={`[${m.seq}, ${m.D}]`} size={15} w={700} c={T.ink} />
        <Txt x={95} y={320} s="into the LLM" size={13} />
        <Txt x={95} y={354} s={{ qwen: "3D positions (t, h, w)", deepseek: "1D positions", kimi: "no position ids", mimo: "1D positions" }[m.id]} size={13} c={m.col} w={700} />
        <Txt x={95} y={374} s={{ qwen: "h = row, w = column", deepseek: "↵ marks each row", kimi: "order comes from KDA", mimo: "image = a flat run" }[m.id]} size={13} c={T.soft} />
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 2 · who can see whom
const TOY = { def: ["b", "i", "i", "i", "i", "b", "t", "t", "t", "t"], deepseek: ["i", "i", "n", "i", "i", "n", "s", "t", "t", "t"] };
const typeCol = (m, t) => (t === "i" ? m.col : t === "t" ? T.soft : T.dim);
function Mask({ m, x, y, cs, kind, p, title, sub }) {
  const ty = TOY[m.id] ?? TOY.def, n = ty.length, rows = p * n;
  const allowed = (i, j) => j <= i && (kind !== "swa" || j > i - 4);
  const alpha = (i, j) => (kind === "kda" ? 0.78 ** (i - j) : 1);
  return (
    <g>
      <Txt x={x + (n * cs) / 2} y={y - 18} s={title} size={13} c={T.ink} w={700} />
      {ty.map((t, j) => <rect key={`c${j}`} x={x + j * cs + 0.5} y={y - 8} width={cs - 1} height={4} fill={typeCol(m, t)} />)}
      {ty.map((t, i) => <rect key={`r${i}`} x={x - 8} y={y + i * cs + 0.5} width={4} height={cs - 1} fill={typeCol(m, t)} />)}
      {ty.map((_, i) => ty.map((tj, j) => {
        const on = allowed(i, j), vis = i < rows;
        return <rect key={`${i}.${j}`} x={x + j * cs + 0.5} y={y + i * cs + 0.5} width={cs - 1} height={cs - 1}
          fill={on && vis ? typeCol(m, tj) : "none"} fillOpacity={on && vis ? 0.85 * alpha(i, j) : 1} stroke={T.rule} strokeWidth={0.5} />;
      }))}
      {p >= 1 && <rect x={x} y={y + (n - 1) * cs} width={n * cs} height={cs} fill="none" stroke={T.ink} strokeWidth={1.4} />}
      {sub && <Txt x={x + (n * cs) / 2} y={y + n * cs + 18} s={sub} size={13} c={m.col} />}
    </g>
  );
}
function MaskLane({ m, p }) {
  const r = clamp01(p / 0.8);
  const two = m.id === "kimi" || m.id === "mimo";
  return (
    <g>
      {two ? <>
        <Mask m={m} x={14} y={44} cs={7} p={r} kind={m.id === "kimi" ? "kda" : "swa"} title={m.id === "kimi" ? "KDA × 69" : "SWA × 39"} sub={m.id === "kimi" ? "state" : "window"} />
        <Mask m={m} x={106} y={44} cs={7} p={r} kind="full" title={m.id === "kimi" ? "MLA × 24" : "global × 9"} sub="all" />
      </> : <Mask m={m} x={40} y={44} cs={12} p={r} kind="full" title={`all ${m.L} layers`} sub="causal" />}
      <g opacity={seg(p, 0.8, 1)}>
        <Txt x={95} y={232} s="row = query · col = key" size={13} c={T.dim} />
        <Txt x={95} y={290} s={{ qwen: "“?” sees all 196 img", deepseek: "“?” sees all 1023 img", kimi: "3 of 4 layers:", mimo: "39 of 48 layers:" }[m.id]} size={13} c={T.ink} w={700} />
        <Txt x={95} y={310} s={{ qwen: "in every layer", deepseek: "in every layer", kimi: "a fixed-size memory", mimo: "only the last 128" }[m.id]} size={13} c={T.soft} />
        <Txt x={95} y={346} s={{ qwen: "", deepseek: "", kimi: "1 of 4: every token", mimo: "1 of ~5: every token" }[m.id]} size={13} c={m.col} />
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 3 · “?” predicts the next token
const NC = 8, CX = 64, CG = 15, CW = 12, RY = 44, RH = 13;
const colX = (c) => CX + c * CG;
const layerKind = (m, c) => (m.id === "kimi" ? (c % 4 === 3 ? "MLA" : "KDA") : m.id === "mimo" ? (c === 0 || c === 5 ? "global" : "SWA") : "");
const attnW = (i, c) => 0.15 + 0.85 * Math.abs(Math.sin(i * 1.7 + c * 2.3 + 0.4));
const VOC = ["dog", "cat", "a", "grass", "fox"];
const PROBS = [0.81, 0.08, 0.05, 0.03, 0.03];
function PredictLane({ m, p, pic }) {
  const ty = TOY[m.id] ?? TOY.def, n = ty.length, q = n - 1;
  const sw = clamp01((p - 0.05) / 0.6) * NC, cur = Math.min(NC - 1, Math.floor(sw)), done = p > 0.66, kind = layerKind(m, cur);
  const lt = seg(p, 0.68, 0.95);
  const cellFill = (i, c) => {
    if (c !== cur || done) return c < cur || done ? { s: `${m.col}44` } : {};
    if (i === q) return { f: m.col, o: 1, s: T.ink, w: 1.6 };
    if (i > q) return {};
    const k = layerKind(m, c);
    if (k === "SWA" && i < q - 3) return {};
    const a = k === "KDA" ? 0.78 ** (q - i) : attnW(i, c) * 0.9;
    return { f: typeCol(m, ty[i]), o: a, s: `${m.col}88` };
  };
  const ds = m.id === "qwen" && [0, 1, 2].includes(cur) && !done;
  return (
    <g>
      {m.id === "qwen" && <g>
        <Txt x={CX - 8} y={30} s="taps" size={13} a="end" />
        {[0, 1, 2].map((c) => <g key={c}>{[0, 1, 2, 3].map((o) => <rect key={o} x={colX(c) + (o & 1) * 6} y={14 + (o >> 1) * 6} width={5.5} height={5.5} fill={rgb(pic.merged[o])} stroke={ds && cur === c ? m.col : "none"} />)}<line x1={colX(c) + CW / 2} x2={colX(c) + CW / 2} y1={27} y2={RY - 2} stroke={m.col} strokeWidth={ds && cur === c ? 2.2 : 1} strokeDasharray={ds && cur === c ? undefined : "3 3"} /></g>)}
      </g>}
      {ty.map((t, i) => <rect key={`l${i}`} x={CX - 18} y={RY + i * RH + 1} width={12} height={RH - 2} rx={1.5} fill={t === "i" ? rgb(pic.c4[[0, 1, 4, 5][i % 4]]) : t === "t" ? T.rule : `${m.col}33`} stroke={t === "t" ? "none" : m.col} strokeWidth={0.6} />)}
      <Txt x={CX - 22} y={RY + q * RH + 10} s="?" size={14} w={700} c={T.ink} a="end" />
      {ty.map((_, i) => Array.from({ length: NC }, (_, c) => {
        const st = cellFill(i, c);
        return <rect key={`${i}.${c}`} x={colX(c)} y={RY + i * RH + 1} width={CW} height={RH - 2} rx={1.5} fill={st.f ?? "none"} fillOpacity={st.o ?? 1} stroke={st.s ?? T.rule} strokeWidth={st.w ?? 0.7} />;
      }))}
      {ds && ty.map((t, i) => t === "i" && <Txt key={i} x={colX(cur) + CW / 2} y={RY + i * RH + 10} s="+" size={13} w={700} c={T.bg} />)}
      {!done && <Txt x={colX(cur) + CW / 2} y={RY + n * RH + 16} s={kind || `layer ${Math.round(((cur + 1) * m.L) / NC)}`} size={13} c={m.col} w={700} />}
      {done && <Txt x={CX + (NC * CG) / 2} y={RY + n * RH + 16} s={`${m.L} layers →`} size={13} />}
      <g opacity={lt > 0 ? 1 : 0.25}>
        {VOC.map((w, i) => {
          const pr = PROBS[i];
          return (
            <g key={w}>
              <Txt x={56} y={232 + i * 18 + 12} s={w} size={13} c={i === 0 && lt > 0.6 ? m.col : T.dim} a="end" w={i === 0 ? 700 : 400} />
              <rect x={60} y={232 + i * 18 + 3} width={Math.max(1, pr * 100 * lt)} height={11} rx={2} fill={i === 0 ? m.col : T.rule} />
              {lt > 0.6 && <Txt x={64 + pr * 100} y={232 + i * 18 + 12} s={pr.toFixed(2)} size={13} c={i === 0 ? m.col : T.dim} a="start" />}
            </g>
          );
        })}
        <Txt x={95} y={346} s={`of ${m.vocab.toLocaleString("en-US")}`} size={13} op={lt} />
        <Txt x={95} y={384} s="“?” → “dog”" size={16} w={700} c={T.ink} op={lt} />
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 4 · text loss → gradients flow down
function LossLane({ m, p, pic }) {
  const g = clamp01((p - 0.15) / 0.7), gy = lerp(40, 330, g);
  const box = (y, h, t, s, extra) => {
    const hot = gy >= y;
    return (
      <g>
        <rect x={20} y={y} width={150} height={h} rx={8} fill={hot ? `${GRAD}22` : `${m.col}10`} stroke={hot ? GRAD : m.col} strokeWidth={hot ? 2 : 1.3} />
        <Txt x={78} y={y + (s ? 18 : h / 2 + 5)} s={t} size={14} w={700} c={T.ink} />
        {s && <Txt x={78} y={y + 35} s={s} size={13} c={T.soft} />}
        {extra && hot && <Txt x={95} y={y + h + 15} s={extra} size={13} c={GRAD} />}
      </g>
    );
  };
  const extraVit = { deepseek: "lr × 0.1", kimi: "its only teacher", qwen: "", mimo: "" }[m.id];
  return (
    <g>
      <rect x={8} y={4} width={174} height={30} rx={7} fill={`${GRAD}22`} stroke={GRAD} strokeWidth={1.6} />
      <Txt x={95} y={24} s="−log 0.81 = 0.21" size={14} w={700} c={GRAD} />
      {box(56, 50, "LLM", `${m.L} layers`)}
      {box(146, 34, "MLP")}
      {box(220, 50, "ViT", `${m.vitL} blocks`, extraVit)}
      <image href={pic.url} x={75} y={300} width={40} height={40} />
      {[[95, 300, 95, 272], [95, 220, 95, 182], [95, 146, 95, 108], [95, 56, 95, 36]].map(([x1, y1, x2, y2], i) => <line key={i} x1={x1 - 22} y1={y1} x2={x2 - 22} y2={y2} stroke={T.rule} strokeWidth={1.4} />)}
      {m.id === "qwen" && [0, 1, 2].map((k) => <path key={k} d={`M170,${232 + k * 10} H${178 - k * 4} V${70 + k * 10} H170`} fill="none" stroke={gy >= 220 ? GRAD : m.col} strokeWidth={1.1} strokeDasharray="3 2" />)}
      {g > 0 && g < 1 && <g>
        <line x1={150} x2={150} y1={36} y2={gy} stroke={GRAD} strokeWidth={3} strokeLinecap="round" />
        <path d={`M144,${gy - 6} L150,${gy + 2} L156,${gy - 6}`} fill="none" stroke={GRAD} strokeWidth={2.4} />
      </g>}
      {g >= 1 && <line x1={150} x2={150} y1={36} y2={272} stroke={GRAD} strokeWidth={2} strokeDasharray="5 4" />}
      <Txt x={132} y={50} s="∂L" size={14} w={700} c={GRAD} op={seg(p, 0.1, 0.3)} />
      <g opacity={seg(p, 0.85, 1)}>
        <Txt x={95} y={378} s="loss on text tokens only" size={13} c={T.soft} />
        <Txt x={95} y={398} s="image tokens: no loss" size={13} />
        {m.id === "qwen" && <Txt x={95} y={426} s="taps carry ∂L too" size={13} c={m.col} />}
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 4 · init + frozen / trained per stage
const TRAIN = {
  qwen: { st: ["0", "1+"], rows: [["ViT", "SigLIP 2", "❄", "🔥"], ["MLP", "random", "🔥", "🔥"], ["LLM", "Qwen3", "❄", "🔥"]], note: ["0: merger only", "1+: everything"] },
  deepseek: { st: ["1", "2", "3"], rows: [["ViT", "SigLIP", "🔥", "🔥", "🔥"], ["MLP", "random", "🔥", "🔥", "🔥"], ["LLM", "DS-MoE", "❄", "🔥", "🔥"]], note: ["1: LLM frozen", "2, 3: everything", "ViT lr × 0.1"] },
  kimi: { st: ["1"], rows: [["ViT", "random", "🔥"], ["MLP", "random", "🔥"], ["LLM", "random", "🔥"]], note: ["all random at step 0", "one loss from the start"] },
  mimo: { st: ["0", "1", "2"], rows: [["ViT", "random", "🔥", "–", "🔥"], ["MLP", "random", "–", "–", "🔥"], ["LLM", "random", "🔥", "🔥", "🔥"]], note: ["0: ViT + a small LLM", "1: big LLM, text only", "2: all together"] },
};
function TrainLane({ m, p }) {
  const t = TRAIN[m.id], sx = (j) => (t.st.length === 1 ? 160 : t.st.length === 2 ? 145 + j * 30 : 138 + j * 22);
  return (
    <g>
      <Txt x={92} y={30} s="start" size={13} c={T.dim} />
      {t.st.map((s, j) => <Txt key={s} x={sx(j)} y={30} s={s} size={14} w={700} c={T.soft} />)}
      {t.rows.map(([name, init, ...ic], i) => {
        const y = 46 + i * 50, op = seg(p, i * 0.15, i * 0.15 + 0.3);
        return (
          <g key={name} opacity={op}>
            <rect x={4} y={y} width={50} height={36} rx={7} fill={`${m.col}12`} stroke={m.col} />
            <Txt x={29} y={y + 23} s={name} size={14} c={T.ink} w={700} />
            <Txt x={92} y={y + 23} s={init} size={13} c={init === "random" ? m.col : T.soft} />
            {ic.map((c, j) => <text key={j} x={sx(j)} y={y + 25} textAnchor="middle" fontSize={c === "–" ? 15 : 17} fill={T.dim}>{c}</text>)}
          </g>
        );
      })}
      <g opacity={seg(p, 0.6, 0.9)}>
        {t.note.map((s, i) => <Txt key={i} x={95} y={238 + i * 22} s={s} size={13} c={i === 0 ? m.col : T.soft} />)}
      </g>
    </g>
  );
}

// ------------------------------------------------------------ 5 · side by side
const ROWS = [
    ["448² image → tokens", (m) => `${m.nVis}`],
    ["resolution", (m) => (m.id === "deepseek" ? "384 tiles + thumb" : "native")],
    ["ViT", (m) => `${m.vitL} × ${m.vitD}`],
    ["ViT starts from", (m) => m.init],
    ["compress", (m) => `2×2 → ${m.mergeD}`],
    ["MLP → width", (m) => `${m.D}`],
    ["extra", (m) => ({ qwen: "DeepStack × 3", deepseek: "↵ + sep tokens", kimi: "—", mimo: "—" }[m.id])],
    ["positions", (m) => ({ qwen: "3D (t, h, w)", deepseek: "1D", kimi: "none", mimo: "1D" }[m.id])],
    ["LLM attention", (m) => ({ qwen: "full", deepseek: "full (MLA)", kimi: "3 KDA : 1 MLA", mimo: "SWA 128 + global" }[m.id])],
    ["LLM", (m) => `${m.L} layers × ${m.D}`],
];
function Compare({ sel }) {
  const rows = ROWS;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "190px repeat(4, minmax(0, 1fr))", gap: "2px 10px", fontFamily: mono, padding: 6 }}>
      <span />
      {M4.map((m) => <div key={m.id} style={{ fontSize: 17, fontWeight: 700, color: m.col, padding: "4px 0 8px", opacity: m.id === sel ? 1 : 0.55 }}>{m.name}{m.id === sel ? " ◂" : ""}</div>)}
      {rows.map(([k, f]) => [
        <div key={k} style={{ fontSize: 14, color: T.dim, padding: "9px 0", borderTop: `1px solid ${T.rule}` }}>{k}</div>,
        ...M4.map((m) => <div key={m.id + k} style={{ fontSize: 16, color: m.id === sel ? T.ink : T.soft, fontWeight: m.id === sel ? 700 : 400, padding: "9px 0", borderTop: `1px solid ${T.rule}` }}>{f(m)}</div>),
      ])}
    </div>
  );
}

// ------------------------------------------------------------ the view
const CODE = {
  vit: [
    "qwen:     f, taps = vit(patches)          # [784, 1152] + layers 8, 16, 24",
    "deepseek: f = siglip(tiles)               # 5 tiles × [729, 1152]",
    "kimi:     f = moonvit_v2(patches)         # [1024, 1024]",
    "mimo:     f = mimo_vit(patches)           # [784, 1280]",
  ],
  merge: [
    "x = f.view(-1, 4 * d)     # 2×2 neighbours side by side",
    "# qwen [196, 4608] · deepseek [5, 196, 4608] · kimi [256, 4096] · mimo [196, 5120]",
  ],
  mlp: [
    "v = W2(gelu(W1(norm(f).view(-1, 4 * d))))",
    "# deepseek: no norm · then insert ↵ and sep",
  ],
  seq: [
    "x = embed(input_ids)          # text rows",
    "x[image_slots] = v            # image rows",
    "# qwen: position_ids = 3 × [T] (t, h, w)",
  ],
  mask: [
    "full:   mask[i, j] = j <= i",
    "SWA:    mask[i, j] = i - 128 < j <= i",
    "KDA:    S = decay * S + k vᵀ   # no mask matrix",
  ],
  predict: [
    "for i, layer in enumerate(llm.layers):",
    "    h = layer(h)",
    "    if qwen and i < 3: h[image_slots] += ds[i]",
    "logits = lm_head(h[-1])",
  ],
  loss: [
    "loss = cross_entropy(logits[text_pos], targets[text_pos])",
    "loss.backward()      # LLM → MLP → ViT",
  ],
  train: ["module.requires_grad_(stage in its trained stages)"],
};

const FLOW = [
  { group: "1 · to LLM width", items: [
    { id: "vit", label: "image → patch vectors", sub: "native or tiles", code: CODE.vit },
    { id: "merge", label: "2×2 → 1 vector", sub: "4× fewer tokens", code: CODE.merge },
    { id: "mlp", label: "MLP → LLM width", sub: "image tokens", code: CODE.mlp },
  ] },
  { group: "2 · into the sequence", items: [
    { id: "seq", label: "[img] [text]", sub: "boundaries · positions", code: CODE.seq },
    { id: "mask", label: "who sees whom", sub: "attention masks", code: CODE.mask },
  ] },
  { group: "3 · predict text", items: [{ id: "predict", label: "“?” → “dog”", sub: "up the layers", code: CODE.predict }] },
  { group: "4 · text loss trains", items: [
    { id: "loss", label: "loss → ∂L down", sub: "LLM → MLP → ViT", code: CODE.loss },
    { id: "train", label: "start · ❄ / 🔥", sub: "per stage", code: CODE.train },
  ] },
  { id: "compare", label: "5 · side by side", sub: "what each one changes" },
];

const STEPS = [
  { id: "vit", block: "vit", caption: "one 448 × 448 image → patch vectors" },
  { id: "merge", block: "merge", caption: "2×2 neighbours → one long vector" },
  { id: "mlp", block: "mlp", caption: "MLP → the LLM's width" },
  { id: "seq", block: "seq", caption: "image tokens join the text" },
  { id: "mask", block: "mask", caption: "who can see whom" },
  { id: "predict", block: "predict", caption: "“?” reads the image → “dog”" },
  { id: "loss", block: "loss", caption: "text loss → gradients flow down" },
  { id: "train", block: "train", caption: "starting weights · frozen ❄ / trained 🔥" },
  { id: "compare", block: "compare", caption: "side by side" },
];

const LANE = { vit: VitLane, merge: MergeLane, mlp: MlpLane, seq: SeqLane, mask: MaskLane, predict: PredictLane, loss: LossLane, train: TrainLane };
function Scene({ st, m }) {
  const pic = usePic(), p = useProgress(st.id + m.id, ["predict", "mlp", "loss"].includes(st.id) ? 3200 : 2400);
  if (st.id === "compare") return <Compare sel={m.id} />;
  if (st.id === "vit") return <PatchScene m={m} />;
  const L = LANE[st.id];
  return <Solo m={m} stepId={st.id} render={(mm) => <L m={mm} p={p} pic={pic} />} />;
}

// the picked model, so the matmul card and code only show its lines
const SelCtx = createContext(null);

const Tag = ({ m }) => <span style={{ display: "inline-block", minWidth: 120, color: m.col, fontWeight: 700 }}>{m.name}</span>;
const ML = ({ m, children }) => useContext(SelCtx) !== m.id ? null : <div style={{ display: "grid", gridTemplateColumns: "124px minmax(0, 1fr)", alignItems: "baseline", fontFamily: mono, fontSize: 15 }}><Tag m={m} /><div style={{ display: "grid", gap: 2 }}>{children}</div></div>;

function matmuls(st) {
  const mlpMM = (m) => {
    const hid = m.id === "deepseek" ? m.D : m.mergeD, n = m.id === "deepseek" ? 980 : m.nVis;
    return <><MM a={["x", [n, m.mergeD]]} b={["W1", [m.mergeD, hid]]} c={["h", [n, hid]]} note="GELU" /><MM a={["h", [n, hid]]} b={["W2", [hid, m.D]]} c={["v", [n, m.D]]} /></>;
  };
  const body = {
    vit: <>
      <ML m={QW}><MM a={["patches", [784, 1536]]} b={["W_patch", [1536, 1152]]} c={["f", [784, 1152]]} note="2 frames × 16×16×3 · then 27 blocks" /></ML>
      <ML m={DS}><MM a={["patches", [3645, 588]]} b={["W_patch", [588, 1152]]} c={["f", [3645, 1152]]} note="5 tiles × 729 · 14×14×3 · then 27 blocks" /></ML>
      <ML m={KM}><MM a={["patches", [1024, 588]]} b={["W_patch", [588, 1024]]} c={["f", [1024, 1024]]} note="14×14×3 · then 27 blocks" /></ML>
      <ML m={MI}><MM a={["patches", [784, 1536]]} b={["W_patch", [1536, 1280]]} c={["f", [784, 1280]]} note="2 frames × 16×16×3 · then 28 blocks" /></ML>
    </>,
    merge: <>
      <ML m={QW}><MMNote>[784, 1152] → [196, 4 × 1152] = [196, 4608] · no math, a reshape</MMNote></ML>
      <ML m={DS}><MMNote>[5, 27, 27, 1152] → pad → [5, 28, 28, 1152] → [5, 196, 4608]</MMNote></ML>
      <ML m={KM}><MMNote>[1024, 1024] → [256, 4096]</MMNote></ML>
      <ML m={MI}><MMNote>[784, 1280] → [196, 5120]</MMNote></ML>
    </>,
    mlp: <>{M4.map((m) => <ML key={m.id} m={m}>{mlpMM(m)}</ML>)}</>,
    seq: <>
      <ML m={QW}><MMNote>embed(ids) [6, 4096] + v [196, 4096] → x [202, 4096]</MMNote></ML>
      <ML m={DS}><MMNote>embed(ids) [4, 2560] + v [980, 2560] + 42 ↵ + 1 sep → x [1027, 2560]</MMNote></ML>
      <ML m={KM}><MMNote>embed(ids) [6, 7168] + v [256, 7168] → x [262, 7168]</MMNote></ML>
      <ML m={MI}><MMNote>embed(ids) [6, 4096] + v [196, 4096] → x [202, 4096]</MMNote></ML>
    </>,
    mask: <>
      <ML m={QW}><MM a={["q", [32, 202, 128]]} b={["kᵀ", [8, 128, 202]]} c={["S", [32, 202, 202]]} times={36} /></ML>
      <ML m={DS}><MM a={["q", [32, 1027, "d"]]} b={["kᵀ", [32, "d", 1027]]} c={["S", [32, 1027, 1027]]} times={30} /></ML>
      <ML m={KM}><MM a={["q", [96, 262, "d"]]} b={["kᵀ", [96, "d", 262]]} c={["S", [96, 262, 262]]} times={24} note="MLA layers · the 69 KDA layers keep a fixed-size state" /></ML>
      <ML m={MI}><MM a={["q", [64, 202, 192]]} b={["kᵀ", [8, 192, 128]]} c={["S", [64, 202, 128]]} times={39} note="SWA: at most 128 keys" /><MM a={["q", [64, 202, 192]]} b={["kᵀ", [4, 192, 202]]} c={["S", [64, 202, 202]]} times={9} note="global" /></ML>
    </>,
    predict: <>{M4.map((m) => <ML key={m.id} m={m}><MM a={["h[“?”]", [1, m.D]]} b={["W_vocab", [m.D, m.vocab]]} c={["logits", [1, m.vocab]]} /></ML>)}</>,
    loss: <MMNote>L = −log p(“dog”) = 0.21 · every 🔥 weight on the path gets ∂L/∂W</MMNote>,
    train: <MMNote>no new math · ❄ = no gradient update in that stage</MMNote>,
    compare: <MMNote>same path everywhere: ViT → 2×2 → MLP → LLM → text loss</MMNote>,
  };
  return <MatMulCard>{body[st.id]}</MatMulCard>;
}

// keep only this model's code lines ("qwen: …" lines of the others are dropped)
const flowFor = (id) => FLOW.map((b) => {
  const fix = (it) => (it.code ? { ...it, code: it.code.filter((l) => !M4.some((o) => o.id !== id && l.startsWith(`${o.id}:`))) } : it);
  return b.items ? { ...b, items: b.items.map(fix) } : fix(b);
});

export default function ModernVlmView() {
  const [sel, setSel] = useState(() => {
    try { return M4.find((m) => m.id === localStorage.getItem("modern-vlm"))?.id ?? "qwen"; } catch { return "qwen"; }
  });
  const pick = (id) => { setSel(id); try { localStorage.setItem("modern-vlm", id); } catch { /* ignore */ } };
  const m = M4.find((x) => x.id === sel);
  // each sub-tab keeps its own step: first visit starts at step 1, later visits resume
  const [stepOf, setStepOf] = useState(() => {
    try { return JSON.parse(localStorage.getItem("modern-vlm-steps")) ?? {}; } catch { return {}; }
  });
  const setStep = (i) => setStepOf((o) => {
    const n = { ...o, [sel]: i };
    try { localStorage.setItem("modern-vlm-steps", JSON.stringify(n)); } catch { /* ignore */ }
    return n;
  });
  const flow = useMemo(() => flowFor(sel), [sel]);
  return (
    <SelCtx.Provider value={sel}>
      <StepShell
        title="Modern VLMs"
        subtitle="one 448 × 448 image · toy picture, real shapes"
        blocks={flow}
        steps={STEPS}
        index={stepOf[sel] ?? 0}
        onIndex={setStep}
        scene={(st) => <div style={{ minHeight: 420 }}><Scene key={st.id + sel} st={st} m={m} /></div>}
        matmuls={matmuls}
        controls={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {M4.map((x) => (
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
