import { T, mono } from "./theme";

const W = 860;
const H = 340;
const PAD = { top: 20, right: 24, bottom: 42, left: 64 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;
const LX0 = 0; // log10(B) range 1..10^4
const LX1 = 4;

const xPx = (b) => PAD.left + ((Math.log10(Math.max(b, 1)) - LX0) / (LX1 - LX0)) * PW;

function sci(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-2) return v.toExponential(0).replace("e+", "e");
  return Number(v.toPrecision(3)).toString();
}

const BS = Array.from({ length: 161 }, (_, i) => Math.pow(10, (i / 160) * (LX1 - LX0)));
const path = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

function Frame({ children, yTicks, yPx, yLabel }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {[1, 10, 100, 1000, 10000].map((b) => (
        <g key={b}>
          <line x1={xPx(b)} x2={xPx(b)} y1={PAD.top} y2={PAD.top + PH} stroke={T.rule} strokeOpacity=".45" />
          <text x={xPx(b)} y={H - 20} fill={T.dim} fontSize="11" fontFamily={mono} textAnchor="middle">{b.toLocaleString()}</text>
        </g>
      ))}
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={PAD.left + PW} y1={yPx(v)} y2={yPx(v)} stroke={T.rule} strokeOpacity=".45" />
          <text x={PAD.left - 8} y={yPx(v) + 4} fill={T.dim} fontSize="11" fontFamily={mono} textAnchor="end">{sci(v)}</text>
        </g>
      ))}
      <text x={PAD.left + PW / 2} y={H - 4} fill={T.soft} fontSize="12" fontFamily={mono} textAnchor="middle">batch size B (log)</text>
      <text transform={`translate(14 ${PAD.top + PH / 2}) rotate(-90)`} fill={T.soft} fontSize="12" fontFamily={mono} textAnchor="middle">{yLabel}</text>
      {children}
    </svg>
  );
}

function VLine({ x, color, label, dash, top = 0 }) {
  if (x < PAD.left - 1 || x > PAD.left + PW + 1) return null;
  return (
    <g>
      <line x1={x} x2={x} y1={PAD.top} y2={PAD.top + PH} stroke={color} strokeWidth="1.5" strokeDasharray={dash} />
      <text x={x + 5} y={PAD.top + 12 + top} fill={color} fontSize="11.5" fontFamily={mono}>{label}</text>
    </g>
  );
}

function Legend({ items }) {
  return (
    <g fontFamily={mono} fontSize="11.5">
      {items.map((it, i) => (
        <g key={it.label} transform={`translate(${PAD.left + 12 + i * 150} ${PAD.top + PH - 10 - 0})`}>
          <line x1="0" x2="18" y1="-4" y2="-4" stroke={it.color} strokeWidth="2.5" strokeDasharray={it.dash} />
          <text x="24" y="0" fill={T.soft}>{it.label}</text>
        </g>
      ))}
    </g>
  );
}

// Log-log: weight term (flat), KV term (slope 1), total; crossing at B*.
export function StepTimeChart({ P, Ckv, W: Whbm, Bcrit = Infinity, B, Bstar }) {
  const tW = P / Whbm;
  const mlp = (b) => (P * Math.max(b / Bcrit, 1)) / Whbm;
  const tot = (b) => (b * Ckv + P * Math.max(b / Bcrit, 1)) / Whbm;
  const kv = (b) => (b * Ckv) / Whbm;
  const lo = Math.min(tW, kv(1)) * 0.6;
  const hi = tot(1e4) * 1.4;
  const l0 = Math.log10(lo);
  const l1 = Math.log10(hi);
  const yPx = (v) => PAD.top + (1 - (Math.log10(Math.max(v, lo)) - l0) / (l1 - l0)) * PH;
  const ticks = [];
  for (let e = Math.ceil(l0); e <= Math.floor(l1); e++) ticks.push(Math.pow(10, e));
  const line = (f) => path(BS.filter((b) => f(b) >= lo).map((b) => [xPx(b), yPx(f(b))]));
  return (
    <Frame yTicks={ticks} yPx={yPx} yLabel="step time (log)">
      <path d={line(kv)} fill="none" stroke={T.accent2} strokeWidth="2.2" strokeDasharray="6 4" />
      <path d={line(mlp)} fill="none" stroke={T.wire} strokeWidth="2.2" strokeDasharray="6 4" />
      <path d={line(tot)} fill="none" stroke={T.ink} strokeWidth="3" />
      <VLine x={xPx(Bstar)} color={T.accent} label={`B* = ${sci(Bstar)}`} dash="3 4" />
      {isFinite(Bcrit) && <VLine x={xPx(Bcrit)} color={T.bad} label={`B_crit = ${sci(Bcrit)}`} dash="3 4" top={16} />}
      <line x1={xPx(B)} x2={xPx(B)} y1={PAD.top} y2={PAD.top + PH} stroke={T.accent} strokeWidth="1.5" strokeOpacity=".8" />
      <circle cx={xPx(B)} cy={yPx(tot(B))} r="5.5" fill={T.ink} stroke={T.accent} strokeWidth="2" />
      <Legend items={[
        { label: isFinite(Bcrit) ? "MLP max(·)" : "weights P/W", color: T.wire, dash: "6 4" },
        { label: "KV B·Ckv/W", color: T.accent2, dash: "6 4" },
        { label: "total T_step", color: T.ink },
      ]} />
    </Frame>
  );
}

// Linear y: throughput saturating to cap; ghost curves for MHA/GQA/MQA.
export function ThroughputChart({ model, B, Bstar, cap, variants }) {
  const thr = (b, m = model) => (b * m.W) / (b * m.Ckv + m.P * Math.max(b / m.Bcrit, 1));
  const capOf = (m) => m.W / (m.Ckv + m.P / m.Bcrit);
  const ymax = variants.length ? Math.max(...variants.map((v) => capOf(v.m))) : cap;
  const top = Math.max(ymax, cap) * 1.08;
  // when ghost caps are huge (MQA), clip to keep the main curve readable
  const yTop = variants.length ? Math.min(top, cap * 6) : top;
  const yPx = (v) => PAD.top + (1 - Math.min(v, yTop) / yTop) * PH;
  const step = niceStep(yTop / 5);
  const ticks = [];
  for (let v = 0; v <= yTop; v += step) ticks.push(v);
  const curve = (m) => path(BS.map((b) => [xPx(b), yPx(thr(b, m))]));
  return (
    <Frame yTicks={ticks} yPx={yPx} yLabel="tokens / time">
      {variants.map((v) => (
        <path key={v.label} d={curve(v.m)} fill="none" stroke={v.color} strokeOpacity=".55" strokeWidth="1.8" strokeDasharray="4 4" />
      ))}
      <path d={curve(model)} fill="none" stroke={T.accent} strokeWidth="3" />
      <line x1={PAD.left} x2={PAD.left + PW} y1={yPx(cap)} y2={yPx(cap)} stroke={T.wire} strokeDasharray="6 4" />
      <text x={PAD.left + PW - 4} y={yPx(cap) - 6} fill={T.wire} fontSize="11.5" fontFamily={mono} textAnchor="end">cap = {sci(cap)}</text>
      <VLine x={xPx(Bstar)} color={T.accent2} label="B*: half of max" dash="3 4" />
      <circle cx={xPx(Bstar)} cy={yPx(thr(Bstar))} r="5" fill={T.bg} stroke={T.accent2} strokeWidth="2" />
      <line x1={xPx(B)} x2={xPx(B)} y1={PAD.top} y2={PAD.top + PH} stroke={T.accent} strokeWidth="1.5" strokeOpacity=".8" />
      <circle cx={xPx(B)} cy={yPx(thr(B))} r="5.5" fill={T.ink} stroke={T.accent} strokeWidth="2" />
      <Legend items={[
        { label: "current model", color: T.accent },
        ...variants.map((v) => ({ label: `${v.label} (K=${v.k})`, color: v.color, dash: "4 4" })),
      ].slice(0, 4)} />
    </Frame>
  );
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}
