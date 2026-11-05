import { useEffect, useMemo, useRef, useState } from "react";
import { ease, lerp } from "./anim";
import { T, mono, btn } from "./theme";
import { Tex } from "./tex";

// Dense FFN -> sparse MoE, drawn as its two matrices. The dense FFN is
// y = x + W_down · act(W_up · x) with W_up [M,H] and W_down [H,M]. Cutting the
// hidden width H into experts slices W_up into column strips and W_down into
// row strips: every expert owns one W_up slice [M, r·H] and one W_down slice
// [r·H, M] (r = fine-grained ratio). A router then runs only s + k of them.
//   stage 0 dense · 1 split into experts · 2 shared vs routed + router · 3 route a token

const W = 860, H = 540;
const AREA = { x: 190, y: 70, w: 540, h: 330 };
const SH_H = 72, GAP = 6, ROUTED_Y = AREA.y + SH_H + 40;

// dense matrices as trapezoids, data flows left to right: W_up widens M -> H (H = 4M),
// W_down narrows H -> M. Edge heights are drawn to the real ratio.
const MID = AREA.y + AREA.h / 2;
const DM = { m: 60, h: 240, w: 190 }; // M-side height, H-side height, width
const UP = { x: AREA.x + 10, w: DM.w };
const DN = { x: AREA.x + 10 + DM.w + 50, w: DM.w };

const CAPTION = [
  "dense: W_up widens every token from M to H, W_down narrows it back",
  "cut W_up and W_down into one up/down pair per expert",
  "s shared experts always run; N routed experts wait for a router",
  "the router scores the token and only the top-k routed experts run",
];


function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const hue = (i) => `hsl(${(i * 137.5) % 360} 62% 62%)`;

function gridRects(n, box, gap) {
  const cols = Math.min(n, Math.max(1, Math.round(Math.sqrt((n * box.w) / box.h))));
  const rows = Math.ceil(n / cols);
  const w = (box.w - gap * (cols - 1)) / cols;
  const h = (box.h - gap * (rows - 1)) / rows;
  return Array.from({ length: n }, (_, i) => ({ x: box.x + (i % cols) * (w + gap), y: box.y + Math.floor(i / cols) * (h + gap), w, h }));
}

// one rect per expert for the stage (index 0..s-1 shared, then routed)
function expertRects(stage, N, s) {
  const n = N + s;
  if (stage === 0) return Array.from({ length: n }, () => ({ x: AREA.x + AREA.w / 2, y: MID, w: 0, h: 0 })); // collapsed inside the dense matrices
  if (stage === 1) return gridRects(n, AREA, GAP);
  const shared = s ? gridRects(s, { x: AREA.x, y: AREA.y, w: AREA.w, h: SH_H }, GAP) : [];
  const routed = gridRects(N, { x: AREA.x, y: ROUTED_Y, w: AREA.w, h: AREA.y + AREA.h - ROUTED_Y }, GAP);
  return [...shared, ...routed];
}

// tween a bag of numbers toward `target`; every change restarts from the value currently on screen
function useTweenNums(target, ms = 850) {
  const sig = JSON.stringify(target);
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    const from = cur.current, keys = Object.keys(target);
    if (keys.every((k) => from[k] === target[k])) return;
    let raf, t0;
    const tick = (now) => {
      if (t0 == null) t0 = now;
      const q = Math.min(1, (now - t0) / ms), e = ease(q);
      const next = q < 1 ? Object.fromEntries(keys.map((k) => [k, lerp(from[k], target[k], e)])) : target;
      cur.current = next; setV(next);
      if (q < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, ms]);
  return v;
}

// morph a list of rects [{x,y,w,h}] into a new list: shared indices glide/resize, new ones are born
// (born(t,i) -> start row), vanished ones shrink away. Returns [x,y,w,h,opacity] rows.
function useMorph(rects, born, ms = 850) {
  const sig = rects.map((r) => `${r.x.toFixed(1)},${r.y.toFixed(1)},${r.w.toFixed(1)},${r.h.toFixed(1)}`).join("|");
  const toArr = rects.map((r) => [r.x, r.y, r.w, r.h, 1]);
  const [v, setV] = useState(toArr);
  const cur = useRef(toArr);
  useEffect(() => {
    const from = cur.current;
    if (from.length === toArr.length && from.every((f, i) => f.every((x, j) => Math.abs(x - toArr[i][j]) < 0.01))) return;
    const L = Math.max(from.length, toArr.length), A = [], B = [];
    for (let i = 0; i < L; i++) {
      const f = from[i], t = toArr[i];
      A.push(f ?? born(t, i));
      B.push(t ?? [f[0] + f[2] / 2, f[1] + f[3] / 2, 0, 0, 0]);
    }
    let raf, t0;
    const tick = (now) => {
      if (t0 == null) t0 = now;
      const q = Math.min(1, (now - t0) / ms), e = ease(q);
      const next = q < 1 ? A.map((a, i) => a.map((x, j) => lerp(x, B[i][j], e))) : toArr;
      cur.current = next; setV(next);
      if (q < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, ms]);
  return v;
}

// trapezoid in a unit box: left/right edge heights as fractions of the box height
const trap = (l, r) => `0,${0.5 - l / 2} 1,${0.5 - r / 2} 1,${0.5 + r / 2} 0,${0.5 + l / 2}`;

// one expert's pair inside its block. W_up is M tall on the left and r·H tall on the
// right; W_down is its mirror image. Heights keep the true ratio r·H / M = 4r.
function sliceGeo(rect, r) {
  const ar = 4 * r, pad = 4, g = 6, top = rect.h > 30 ? 10 : 0;
  const aw = rect.w - 2 * pad, ah = rect.h - 2 * pad - top;
  const hmax = Math.max(1.5, Math.min(ah, (aw - g) / 1.7));
  const u = hmax / Math.max(1, ar);
  const ws = hmax * 0.85, total = 2 * ws + g;
  const x0 = rect.x + (rect.w - total) / 2, cy = rect.y + top + pad + ah / 2;
  return {
    u, hmax, ar,
    up: { x: x0, y: cy - hmax / 2, w: ws, h: hmax, pts: trap(u / hmax, (ar * u) / hmax) },
    dn: { x: x0 + ws + g, y: cy - hmax / 2, w: ws, h: hmax, pts: trap((ar * u) / hmax, u / hmax) },
  };
}

function Slider({ label, value, min, max, onChange, fmt = (v) => v, hint }) {
  return (
    <label style={{ display: "grid", gap: 3, minWidth: 150, flex: 1 }}>
      <span style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11, color: T.dim }}>
        <span>{label}</span><b style={{ color: T.accent, fontSize: 13 }}>{fmt(value)}</b>
      </span>
      <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(+e.target.value)} style={{ accentColor: T.accent, width: "100%" }} />
      {hint && <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{hint}</span>}
    </label>
  );
}

const fmtP = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`);
const signed = (v) => `${v >= 0 ? "+" : "−"}${fmtP(Math.abs(v))}`;
const pct = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0)}%`;

const Stat = ({ k, v, sub, tone }) => (
  <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 8, padding: "8px 12px", minWidth: 150, flex: 1 }}>
    <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.08em", color: T.dim }}>{k}</div>
    <div style={{ fontFamily: mono, fontSize: 18, fontWeight: 700, color: tone ?? T.ink, marginTop: 2 }}>{v}</div>
    {sub && <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginTop: 2 }}>{sub}</div>}
  </div>
);

function Bar({ label, sub, segs, max, on = true, f = 1 }) {
  return (
    <div style={{ display: "grid", gap: 4, opacity: on ? 1 : 0.2, transition: "opacity .5s" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11.5 }}>
        <b style={{ color: T.ink }}>{label}</b>
        <span style={{ color: T.dim }}>{fmtP(segs.reduce((a, [v]) => a + v, 0))}</span>
      </div>
      <div style={{ display: "flex", height: 16, background: T.well, borderRadius: 4, overflow: "hidden" }}>
        {segs.map(([v, c], i) => <div key={i} style={{ width: `${(on ? Math.max((v / max) * 100, v > 0 ? 0.8 : 0) : 0) * f}%`, background: c }} />)}
      </div>
      <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim }}>{sub}</div>
    </div>
  );
}

export default function DenseToMoEView() {
  const [stage, setStage] = useState(0);
  const [N0, setN0] = useState(8); // experts in the conventional MoE, each as big as the dense FFN
  const [sT, setS] = useState(1);
  const [k0, setK0] = useState(2); // its top-k
  const [tokId, setTokId] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [m0, setM0] = useState(2); // fine-grained factor: each expert is cut into m pieces
  const [MT, setM] = useState(1024);
  // DeepSeekMoE fine-grained segmentation: cut each expert into m smaller ones
  // (width H/m) and activate m times as many, so total and active params stay put.
  const m = Math.min(m0, Math.floor(256 / N0), 32);
  const NT = N0 * m;
  const kT = Math.min(k0 * m, NT);
  const nT = NT + sT;
  const inv = m;
  // every quantity that is drawn or printed is tweened from its previous value (counts are rounded)
  const a = useTweenNums({ N: NT, kk: kT, s: sT, M: MT, r: 1 / m, f1: stage >= 1 ? 1 : 0, f2: stage >= 2 ? 1 : 0, f3: stage >= 3 ? 1 : 0 });
  const N = Math.round(a.N), kk = Math.round(a.kk), s = Math.round(a.s), M = Math.round(a.M);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setStage((i) => Math.min(i + 1, 3));
      if (e.key === "ArrowLeft") setStage((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (!playing) return;
    if (stage >= 3) { setPlaying(false); return; }
    const t = setTimeout(() => setStage(stage + 1), 1700);
    return () => clearTimeout(t);
  }, [playing, stage]);

  const n = N + s;
  const rects = useMemo(() => expertRects(stage, NT, sT), [stage, NT, sT]);
  const rv = useMorph(rects, (t, i) => (stage >= 2 && i < sT ? [t[0] - 90, t[1], t[2], t[3], 0] : [t[0] + t[2] / 2, t[1] + t[3] / 2, 0, 0, 0])).map(([x, y, w, h, o]) => ({ x, y, w, h, o }));
  // the sample token: router scores for the N routed experts, top-k of them run
  const route = useMemo(() => {
    const r = mulberry(tokId * 7919 + NT);
    const sc = Array.from({ length: NT }, () => 0.05 + 0.9 * r() ** 1.6);
    const order = sc.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, kT);
    const sum = order.reduce((a, [v]) => a + v, 0);
    return { sc, picked: order.map(([v, i]) => ({ i, g: v / sum })) };
  }, [tokId, NT, kT]);
  const pickedSet = new Set(route.picked.map((p) => p.i));

  const active = (s + kk) / n;
  const ratio = kk / N;
  const routing = stage >= 3;
  const split = stage >= 1;

  // parameter accounting (two matrices per FFN: W_up [M,H] and W_down [H,M])
  const Hd = 4 * M;
  const r = a.r;
  const He = r * Hd;
  const denseP = 2 * M * Hd;
  const expertP = 2 * M * He;
  const routerP = M * N;
  const expertsTotal = n * expertP;
  const totalP = expertsTotal + routerP;
  const activeP = (s + kk) * expertP + routerP;
  const maxP = Math.max(denseP, totalP, activeP);
  const heTxt = inv === 1 ? "H" : `H/${inv}`;
  const rTxt = inv === 1 ? "1" : `1/${inv}`;

  const colorOf = (e) => (stage === 0 ? T.accent : stage >= 2 && e < sT ? T.accent2 : stage === 1 ? hue(e) : hue(e - sT + 3));
  const onE = (e) => !routing || e < sT || pickedSet.has(e - sT);
  const repE = stage >= 3 ? sT + route.picked[0].i : stage >= 2 ? sT : 0; // expert whose shapes are written out
  const rHn = Math.round(He);

  // wires for stage 3
  const mid = AREA.y + AREA.h / 2;
  const wires = [];
  if (routing) {
    rv.forEach((r, e) => {
      if (e >= nT || !onE(e) || r.w < 1) return;
      const cy = r.y + r.h / 2;
      const shared = e < sT;
      wires.push({ key: `in${e}`, d: `M74,${mid} C130,${mid} ${r.x - 40},${cy} ${r.x},${cy}`, c: shared ? T.accent2 : colorOf(e) });
      wires.push({ key: `out${e}`, d: `M${r.x + r.w},${cy} C${r.x + r.w + 30},${cy} ${AREA.x + AREA.w + 10},${mid} 772,${mid}`, c: shared ? T.accent2 : colorOf(e) });
    });
  }

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>Dense FFN → sparse MoE</div>
          <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>set the sparsity, press next: W_up and W_down are cut into one up/down pair per expert</div>
        </div>
        <div style={{ flex: 1 }} />
      </header>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", gap: 20, flexWrap: "wrap" }}>
        <Slider label="experts before splitting N₀" value={N0} min={2} max={32} onChange={(v) => { setN0(v); setK0((q) => Math.min(q, v)); }} hint={`a conventional MoE: ${N0} experts, each as big as the dense FFN`} />
        <Slider label="top-k₀ of those" value={Math.min(k0, N0)} min={1} max={N0} onChange={setK0} hint={`${Math.min(k0, N0)} of ${N0} experts run per token`} />
        <Slider label="fine-grained factor m" value={m} min={1} max={Math.min(32, Math.floor(256 / N0))} onChange={setM0} fmt={(v) => `×${v}`} hint={`each expert → ${m} pieces of width H/${m}: N = ${N0}·${m} = ${NT}, k = ${Math.min(k0, N0)}·${m} = ${kT}`} />
        <Slider label="shared experts s" value={sT} min={0} max={4} onChange={setS} hint={sT ? `${sT} block${sT > 1 ? "s" : ""} every token passes through` : "none: every expert is routed"} />
        <label style={{ fontFamily: mono, fontSize: 11, color: T.dim, alignSelf: "end" }}>M ={" "}
          <select value={MT} onChange={(e) => setM(+e.target.value)} style={{ fontFamily: mono, background: T.well, color: T.ink, border: `1px solid ${T.rule}`, borderRadius: 5, padding: "2px 4px" }}>
            {[512, 1024, 2048, 4096, 7168].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>{" "}· H = 4M
        </label>
      </div>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12 }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          <text x={AREA.x} y={AREA.y - 16} fontFamily={mono} fontSize="11" letterSpacing="0.1em" fill={T.dim}>
            {stage === 0 ? "DENSE FFN · y = x + W_down · act(W_up · x)"
              : stage === 1 ? `${n} EXPERTS · EACH OWNS W_up [${M} × ${rHn}] AND W_down [${rHn} × ${M}]`
              : sT ? "SHARED · ALWAYS ON" : "NO SHARED EXPERTS"}
          </text>
          {stage >= 2 && <text x={AREA.x} y={ROUTED_Y - 8} fontFamily={mono} fontSize="11" letterSpacing="0.1em" fill={T.dim}>ROUTED · {routing ? `TOP-${kk} OF ${N} RUN · idx, g [k] = [${kk}]` : `${N} EXPERTS WAIT FOR THE ROUTER`}</text>}

          {/* token in */}
          <g>
            {[10, 18, 7, 22, 14, 9].map((h, i) => (
              <rect key={i} x={22 + i * 8} y={mid - h / 2} width={5} height={h} rx={1.5} fill={T.soft} />
            ))}
            <text x={46} y={mid + 34} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.soft}>token x</text>
            <Tex x={46} y={mid + 52} w={110} tex={`x\\in\\mathbb{R}^{M}`} size={12} color={T.dim} align="center" />
          </g>

          {/* router */}
          <g style={{ opacity: stage >= 2 ? 1 : 0, transition: "opacity .5s" }}>
            <rect x={92} y={mid - 66} width={64} height={132} rx={8} fill={T.well} stroke={T.wire} />
            <text x={124} y={mid - 52} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={T.wire}>router</text>
            <Tex x={124} y={mid - 46} w={70} h={16} tex={`W_g\\in\\mathbb{R}^{M\\times N}`} size={8.5} color={T.wire} align="center" />
            {Array.from({ length: Math.min(NT, 12) }, (_, i) => {
              const v = route.sc[Math.floor((i * NT) / Math.min(NT, 12))];
              return <rect key={i} x={100} y={mid - 30 + i * 6} width={48 * v} height={4} rx={1.5} fill={T.wire} opacity={routing ? 0.9 : 0.35} style={{ transition: "all .5s" }} />;
            })}
            <Tex x={124} y={mid + 50} w={70} h={16} tex={`xW_g\\in\\mathbb{R}^{N}`} size={9} color={T.dim} align="center" />
            <Tex x={124} y={mid + 72} w={90} h={16} tex={`${M}\\cdot ${N}=\\mathrm{${fmtP(routerP)}}`} size={9} color={T.dim} align="center" />
          </g>
          {stage >= 2 && <path d={`M70,${mid} L92,${mid}`} stroke={T.soft} strokeWidth={1.5} />}

          {wires.map((w) => <path key={w.key} d={w.d} fill="none" stroke={w.c} strokeWidth={1.6} opacity={0.5} className="lane" />)}

          {/* dense FFN: two trapezoids with their shapes written on them */}
          <g fontFamily={mono} fontSize="11" fill={T.soft} style={{ opacity: stage === 0 ? 1 : 0, transition: "opacity .6s", pointerEvents: "none" }}>
            <path d={`M70,${MID} L${UP.x},${MID}`} stroke={T.soft} strokeWidth={1.5} opacity={0.6} />
            <polygon points={`${UP.x},${MID - DM.m / 2} ${UP.x + UP.w},${MID - DM.h / 2} ${UP.x + UP.w},${MID + DM.h / 2} ${UP.x},${MID + DM.m / 2}`} fill={T.accent} opacity={0.92} />
            <path d={`M${UP.x + UP.w},${MID} L${DN.x},${MID}`} stroke={T.wire} strokeWidth={1.5} />
            <text x={(UP.x + UP.w + DN.x) / 2} y={MID - 8} textAnchor="middle" fill={T.wire} fontSize="10">act</text>
            <polygon points={`${DN.x},${MID - DM.h / 2} ${DN.x + DN.w},${MID - DM.m / 2} ${DN.x + DN.w},${MID + DM.m / 2} ${DN.x},${MID + DM.h / 2}`} fill={T.accent} opacity={0.55} />
            <path d={`M${DN.x + DN.w},${MID} L772,${MID}`} stroke={T.soft} strokeWidth={1.5} opacity={0.6} />
            {/* W_up */}
            <Tex x={UP.x + 70} y={MID - 26} w={120} tex={"W_{\\mathrm{up}}"} size={17} color={T.bg} align="center" />
            <Tex x={UP.x + 70} y={MID - 2} w={140} tex={"\\in\\mathbb{R}^{M\\times H}"} size={12} color={T.bg} align="center" />
            <Tex x={UP.x - 6} y={MID - 34} w={120} tex={`M=${M}`} size={13} color={T.accent} align="right" />
            <Tex x={UP.x + UP.w} y={MID - DM.h / 2 - 26} w={120} tex={`H=${Hd}`} size={13} color={T.accent} align="right" />
            <Tex x={UP.x} y={MID + DM.h / 2 + 6} w={250} tex={`\\mathbb{R}^{${M}\\times ${Hd}}\\;=\\;\\mathrm{${fmtP(M * Hd)}}\\text{ params}`} size={12} color={T.dim} />
            {/* W_down */}
            <Tex x={DN.x + 120} y={MID - 26} w={140} tex={"W_{\\mathrm{down}}"} size={17} color={T.bg} align="center" />
            <Tex x={DN.x + 120} y={MID - 2} w={140} tex={"\\in\\mathbb{R}^{H\\times M}"} size={12} color={T.bg} align="center" />
            <Tex x={DN.x} y={MID - DM.h / 2 - 26} w={120} tex={`H=${Hd}`} size={13} color={T.accent} />
            <Tex x={DN.x + DN.w + 6} y={MID - 34} w={120} tex={`M=${M}`} size={13} color={T.accent} />
            <Tex x={DN.x} y={MID + DM.h / 2 + 6} w={250} tex={`\\mathbb{R}^{${Hd}\\times ${M}}\\;=\\;\\mathrm{${fmtP(M * Hd)}}\\text{ params}`} size={12} color={T.dim} />
          </g>

          {/* expert blocks */}
          {rv.map((r, e) => {
            const on = e >= nT || onE(e);
            const c = colorOf(e);
            if (r.w < 0.5 || r.h < 0.5) return null;
            return (
              <g key={e} opacity={r.o}>
                <rect x={r.x} y={r.y} width={r.w} height={r.h} rx={6} fill={T.well} stroke={e === repE && split ? T.ink : c} strokeWidth={routing && on ? 2.2 : e === repE && split ? 1.8 : 1.2} opacity={on ? 1 : 0.35} />
                {split && r.w > 30 && r.h > 22 && (
                  <text x={r.x + 5} y={r.y + 11} fontFamily={mono} fontSize="8.5" fontWeight="700" fill={c} opacity={on ? 1 : 0.4}>{stage >= 2 && e < sT ? `S${e}` : `E${e - (stage >= 2 ? sT : 0)}`}</text>
                )}
              </g>
            );
          })}

          {/* each expert's W_up and W_down trapezoids */}
          {rv.map((rr, e) => {
            if (rr.w < 0.5 || rr.h < 0.5) return null;
            const g = sliceGeo(rr, r);
            const on = e >= nT || onE(e), c = colorOf(e);
            const vis = stage === 0 ? 0 : rr.o;
            const tr = (b, o) => ({
              transform: `translate(${b.x}px, ${b.y}px) scale(${b.w}, ${b.h})`, opacity: vis * o,
              transition: "opacity .6s, fill .5s", // geometry is already animated frame by frame
            });
            const big = split && g.hmax >= 34;
            return (
              <g key={e}>
                <polygon points={g.up.pts} fill={c} style={tr(g.up, on ? 0.95 : 0.2)} />
                <polygon points={g.dn.pts} fill={c} style={tr(g.dn, on ? 0.55 : 0.12)} />
                {big && (
                  <g fontFamily={mono} fontSize="7.5" fill={T.ink} opacity={on ? 1 : 0.4}>
                    <text x={g.up.x + g.up.w / 2} y={g.up.y + g.hmax / 2 + 3} textAnchor="middle" fill={T.bg} fontWeight="700">up</text>
                    <text x={g.dn.x + g.dn.w / 2} y={g.dn.y + g.hmax / 2 + 3} textAnchor="middle" fill={T.bg} fontWeight="700">down</text>
                    <text x={g.up.x + g.up.w + 1} y={g.up.y + g.hmax / 2 - g.hmax * Math.min(g.ar, 1.0) / 2 - 1 + (g.ar >= 1 ? 0 : 0)} fontSize="6.5" fill={T.soft}>{inv === 1 ? "H" : `H/${inv}`}</text>
                  </g>
                )}
              </g>
            );
          })}

          {/* the chosen expert's shapes, written out in full */}
          {split && (
            <g fontFamily={mono} fontSize="10.5">
              <Tex x={AREA.x} y={H - 92} w={560} h={24} size={13} color={colorOf(repE)} tex={`${stage >= 2 && repE < sT ? `\\mathrm{S}_{${repE}}` : `\\mathrm{E}_{${repE - (stage >= 2 ? sT : 0)}}`}:\\;\\; r=${inv === 1 ? "1" : `\\tfrac{1}{${inv}}`}\\;\\Rightarrow\\; rH=${rHn},\\quad 2\\cdot M\\cdot rH=\\mathrm{${fmtP(expertP)}}\\text{ params}`} />
              <Tex x={AREA.x} y={H - 66} w={620} h={26} size={13} color={T.soft} tex={`W_{\\mathrm{up}}\\in\\mathbb{R}^{M\\times rH}=\\mathbb{R}^{${M}\\times ${rHn}}\\qquad W_{\\mathrm{down}}\\in\\mathbb{R}^{rH\\times M}=\\mathbb{R}^{${rHn}\\times ${M}}`} />
              <polygon points="0,2 26,0 26,13 0,11" transform={`translate(${AREA.x},${H - 32})`} fill={T.soft} opacity={0.95} />
              <text x={AREA.x + 32} y={H - 21} fill={T.dim}>W_up slice: M → rH</text>
              <polygon points="0,0 26,2 26,11 0,13" transform={`translate(${AREA.x + 190},${H - 32})`} fill={T.soft} opacity={0.55} />
              <text x={AREA.x + 222} y={H - 21} fill={T.dim}>W_down slice: rH → M</text>
              <text x={AREA.x + 400} y={H - 21} fill={T.dim}>{4 * r === 1 ? "rH = M: the pair is rectangular" : 4 * r > 1 ? "rH > M: up widens" : "rH < M: up narrows"}</text>
            </g>
          )}

          {/* output */}
          <g>
            <circle cx={784} cy={mid} r={12} fill={T.well} stroke={T.soft} />
            <text x={784} y={mid + 5} textAnchor="middle" fontFamily={mono} fontSize="15" fill={T.ink}>{stage >= 2 ? "+" : "="}</text>
            <text x={784} y={mid + 34} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.soft}>y</text>
            <Tex x={784} y={mid + 52} w={110} tex={`y\\in\\mathbb{R}^{M}`} size={12} color={T.dim} align="center" />
          </g>

          {routing && (
            <text x={AREA.x} y={AREA.y + AREA.h + 36} fontFamily={mono} fontSize="12" fill={T.soft}>
              y = x + {sT ? `Σ shared(x)  + ` : ""}Σ<tspan baselineShift="sub" fontSize="9">top-{kk}</tspan> g<tspan baselineShift="sub" fontSize="9">i</tspan> · FFN<tspan baselineShift="sub" fontSize="9">i</tspan>(x)
            </text>
          )}
          {stage === 0 && <text x={AREA.x} y={AREA.y + AREA.h + 36} fontFamily={mono} fontSize="12" fill={T.soft}>both matrices in full for every token: 2 · M · H = {fmtP(denseP)} params read per token</text>}
        </svg>
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Stat k="ACTIVE NEURONS / TOKEN" v={`${(active * 100).toFixed(active < 0.1 ? 1 : 0)}%`} sub={`(s + k) / (N + s) = ${s + kk} / ${n} experts`} tone={stage >= 3 ? T.accent : T.ink} />
        <Stat k="FFN COMPUTE VS DENSE" v={(s + kk) * r <= 1 ? `${(1 / ((s + kk) * r)).toFixed(1)}× less` : `${((s + kk) * r).toFixed(1)}× more`} sub={`(s + k) · r = ${s + kk} · ${rTxt} of a dense FFN · router: one [M, N] matmul`} tone={stage >= 3 ? T.accent : T.ink} />
        <Stat k="SPARSITY k / N" v={`${(ratio * 100).toFixed(ratio < 0.1 ? 1 : 0)}%`} sub={`${N - kk} of ${N} routed experts idle per token`} />
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "start" }}>
        <div style={{ flex: "1 1 100%", minWidth: 0, background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 12 }}>
          <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>PARAMETERS · FFN LAYER</div>
          <Bar label="dense FFN" sub={`2 · M · H = ${fmtP(denseP)}`} segs={[[denseP, T.accent]]} max={maxP} />
          <Bar label="MoE · all experts" sub={stage >= 2 ? `${n} · 2·M·${heTxt} + router` : `${n} · 2·M·${heTxt}`} on={stage >= 1} f={a.f1} segs={[[expertsTotal, T.soft], [routerP * a.f2, T.wire]]} max={maxP} />
          <Bar label="MoE · used per token" sub={`(${s} + ${kk}) · 2·M·${heTxt} + router`} on={stage >= 3} f={a.f3} segs={[[(s + kk) * expertP, T.accent2], [routerP, T.wire]]} max={maxP} />
          <div style={{ display: "grid", gap: 6, fontFamily: mono, fontSize: 12, color: T.soft, opacity: stage >= 1 ? 1 : 0.35, transition: "opacity .4s" }}>
            <div>stored: <b style={{ color: totalP > denseP ? T.wire : T.accent }}>{signed(totalP - denseP)}</b> vs dense ({pct(totalP / denseP - 1)})<span style={{ color: T.dim }}> · {n} · r = {(n * r).toFixed(2)}× the dense FFN, plus router M·N = {fmtP(routerP)}</span></div>
            <div style={{ opacity: stage >= 3 ? 1 : 0.35 }}>per token: <b style={{ color: activeP < denseP ? T.accent : T.bad }}>{signed(activeP - denseP)}</b> vs dense ({pct(activeP / denseP - 1)}){activeP > denseP && <span style={{ color: T.dim }}> · more compute than dense: (s + k) · r = {((s + kk) * r).toFixed(2)} dense FFNs per token</span>}</div>
          </div>
          <div style={{ display: "flex", gap: 14, fontFamily: mono, fontSize: 10.5, color: T.dim, flexWrap: "wrap" }}>
            <span><b style={{ color: T.soft }}>■</b> experts</span><span><b style={{ color: T.wire }}>■</b> router M·N (new)</span><span><b style={{ color: T.accent2 }}>■</b> active experts</span>
          </div>
        </div>
      </div>

      {routing && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", fontFamily: mono, fontSize: 11.5, color: T.dim }}>
          <span>token {tokId} → </span>
          {sT > 0 && <span style={{ color: T.accent2 }}>{sT} shared (weight 1)</span>}
          {route.picked.map((p) => <span key={p.i} style={{ color: colorOf(p.i + sT), border: `1px solid ${colorOf(p.i + sT)}66`, borderRadius: 10, padding: "1px 8px" }}>E{p.i} · g {p.g.toFixed(2)}</span>)}
          <button onClick={() => setTokId((t) => t + 1)} style={{ ...btn(false, false), height: 26, fontSize: 11.5 }}>↻ another token</button>
        </div>
      )}

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <button onClick={() => setStage(Math.max(stage - 1, 0))} disabled={stage === 0} style={btn(false, stage === 0)}>◀ prev</button>
        <button onClick={() => setStage(Math.min(stage + 1, 3))} disabled={stage === 3} style={btn(true, stage === 3)}>next ▶</button>
        <button onClick={() => { if (stage >= 3) setStage(0); setPlaying(!playing); }} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
        <div style={{ flex: 1, minWidth: 220 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{stage === 1 ? `cut W_up into ${n} column strips and W_down into ${n} row strips: each expert keeps one up/down pair, r = ${rTxt} of the dense width (${(n * r).toFixed(2)}× all together)` : CAPTION[stage]}</div>
          <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
            {CAPTION.map((c, i) => (
              <button key={i} onClick={() => setStage(i)} aria-label={`stage ${i + 1}`} style={{ flex: 1, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: i === stage ? T.accent : i < stage ? `${T.accent}66` : T.rule }} />
            ))}
          </div>
        </div>
        <span style={{ fontSize: 11, color: T.dim }}>{stage + 1} / 4 · ← →</span>
      </div>
    </div>
  );
}
