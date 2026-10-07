import { T, mono } from "./theme";
import { DEV_COLORS, deviceStep, ownChunks, progress, stepTimes } from "./ringModel";

// Causal attention matrix at 2N-chunk resolution (token order on both
// axes). A cell is coloured by the GPU that computes it: faded once done,
// bright on the current step. The strip above shows which GPU owns which
// chunk of the sequence.

function ownerOf(variant, N) {
  const own = [];
  for (let r = 0; r < N; r++) for (const c of ownChunks(variant, N, r)) own[c] = r;
  return own;
}

export function SequenceStrip({ variant, N }) {
  const own = ownerOf(variant, N);
  const C = 2 * N;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${C}, 1fr)`, gap: 2 }}>
        {own.map((r, c) => (
          <div key={c} style={{ background: `${DEV_COLORS[r]}30`, border: `1px solid ${DEV_COLORS[r]}`, borderRadius: 4, padding: "3px 0", textAlign: "center", fontFamily: mono, fontSize: 10.5, color: DEV_COLORS[r] }}>
            <div style={{ fontWeight: 700 }}>{r}</div>
            <div style={{ fontSize: 9, opacity: 0.8 }}>c{c}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 10, color: T.dim, marginTop: 4 }}>
        <span>token 0</span><span>sequence →</span><span>token S</span>
      </div>
    </div>
  );
}

export default function AttentionMatrix({ variant, N, phase, sel }) {
  const C = 2 * N;
  const size = 300, pad = 26;
  const cs = size / C;
  const own = ownerOf(variant, N);
  const { doneBelow, current } = progress(phase);

  const cells = [];
  for (let s = 0; s < N; s++) {
    if (s >= doneBelow && s !== current) continue;
    for (let r = 0; r < N; r++) {
      const d = deviceStep(variant, N, r, s);
      for (const c of d.cells) cells.push({ ...c, r, cur: s === current });
    }
  }

  return (
    <svg viewBox={`0 0 ${size + pad + 6} ${size + pad + 6}`} style={{ width: "100%", maxWidth: 380, display: "block", margin: "0 auto" }}>
      <defs>
        <pattern id={`hatch-${variant}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke={T.rule} strokeWidth="1" opacity="0.5" />
        </pattern>
      </defs>
      <g transform={`translate(${pad},${pad})`}>
        {/* masked upper triangle (k chunk > q chunk) */}
        {Array.from({ length: C }, (_, q) => (
          <rect key={q} x={(q + 1) * cs} y={q * cs} width={(C - q - 1) * cs} height={cs} fill={`url(#hatch-${variant})`} />
        ))}
        {/* grid */}
        {Array.from({ length: C + 1 }, (_, i) => (
          <g key={i} stroke={T.rule} strokeWidth={variant === "ring" && i % 2 === 0 ? 1 : 0.5} opacity={0.7}>
            <line x1={0} y1={i * cs} x2={size} y2={i * cs} />
            <line x1={i * cs} y1={0} x2={i * cs} y2={size} />
          </g>
        ))}
        {/* computed blocks */}
        {cells.map((c, i) => {
          const x = c.k0 * cs, y = c.q0 * cs, w = c.kn * cs, h = c.qn * cs;
          const col = DEV_COLORS[c.r];
          const fill = c.cur ? `${col}c0` : `${col}48`;
          const stroke = c.cur && c.r === sel ? T.ink : c.cur ? col : "none";
          const sw = c.cur && c.r === sel ? 2.2 : 1.2;
          return c.tri ? (
            <polygon key={i} points={`${x},${y} ${x},${y + h} ${x + w},${y + h}`} fill={fill} stroke={stroke} strokeWidth={sw} />
          ) : (
            <rect key={i} x={x + 0.5} y={y + 0.5} width={w - 1} height={h - 1} fill={fill} stroke={stroke} strokeWidth={sw} />
          );
        })}
        {/* owner labels */}
        {own.map((r, c) => (
          <g key={c} fontFamily={mono} fontSize={C > 10 ? 8 : 9.5} fontWeight="700" fill={DEV_COLORS[r]} textAnchor="middle">
            <text x={-12} y={c * cs + cs / 2 + 3.5}>{r}</text>
            <text x={c * cs + cs / 2} y={-9}>{r}</text>
          </g>
        ))}
      </g>
      <text x={4} y={12} fontFamily={mono} fontSize="9" fill={T.dim}>Q↓ K→</text>
    </svg>
  );
}

// Gantt of compute per step. Every step lasts as long as its slowest GPU
// (the ring is synchronised by comm.wait()), so idle time is hatched.
export function WorkChart({ variant, N, phase, sel }) {
  const times = stepTimes(variant, N);
  const ringTotal = stepTimes("ring", N).reduce((a, b) => a + b, 0);
  const total = times.reduce((a, b) => a + b, 0);
  const ideal = 2 * N;
  const W = 100 / ringTotal; // % per unit, same scale in both views
  const { doneBelow, current } = progress(phase);
  return (
    <div>
      <div style={{ display: "grid", gap: 4 }}>
        {Array.from({ length: N }, (_, r) => {
          const busy = times.reduce((acc, _, s) => acc + (s < doneBelow || s === current ? deviceStep(variant, N, r, s).work : 0), 0);
          return (
            <div key={r} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 40, fontFamily: mono, fontSize: 10.5, color: DEV_COLORS[r], fontWeight: r === sel ? 700 : 400 }}>GPU {r}</span>
              <div style={{ flex: 1, display: "flex", height: 16, background: T.well, borderRadius: 3, overflow: "hidden", outline: r === sel ? `1px solid ${T.ink}55` : "none" }}>
                {times.map((t, s) => {
                  const w = deviceStep(variant, N, r, s).work;
                  const shown = s < doneBelow || s === current;
                  return (
                    <div key={s} style={{ width: `${t * W}%`, display: "flex", borderRight: `1px solid ${T.bg}` }}>
                      <div style={{ width: `${(w / t) * 100}%`, background: shown ? `${DEV_COLORS[r]}${s === current ? "e0" : "70"}` : `${DEV_COLORS[r]}18`, transition: "background .3s" }} />
                      <div style={{ flex: 1, background: shown ? `repeating-linear-gradient(45deg, transparent 0 3px, ${T.rule}88 3px 4px)` : "transparent" }} />
                    </div>
                  );
                })}
              </div>
              <span style={{ width: 28, fontFamily: mono, fontSize: 10.5, color: T.soft, textAlign: "right" }}>{busy}</span>
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, fontFamily: mono, fontSize: 11, color: T.soft, marginTop: 10 }}>
        <span>wall time <b style={{ color: T.ink }}>{total}</b> units</span>
        <span>ideal (perfect balance) <b style={{ color: T.accent }}>{ideal}</b></span>
        <span>efficiency <b style={{ color: total === ideal ? T.accent : T.bad }}>{Math.round((100 * ideal) / total)}%</b></span>
      </div>
    </div>
  );
}
