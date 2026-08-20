import { mono, sans, T } from "../lib/theme";
import { fmtNum } from "../lib/steps";

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y} Z`;
}

export default function PieChart({ title, segments, unit }) {
  const total = segments.reduce((a, s) => a + s.value, 0);
  const size = 132;
  const r = size / 2 - 4;
  const cx = size / 2;
  const cy = size / 2;

  let angle = 0;
  const slices = segments
    .filter((s) => s.value > 0)
    .map((s) => {
      const frac = total > 0 ? s.value / total : 0;
      const start = angle;
      const end = angle + frac * 360;
      angle = end;
      return { ...s, start, end, frac };
    });

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: size, height: size, flexShrink: 0 }}>
        {total <= 0 ? (
          <circle cx={cx} cy={cy} r={r} fill="none" stroke={T.rule} strokeWidth="1.5" strokeDasharray="4 4" />
        ) : slices.length === 1 ? (
          <circle cx={cx} cy={cy} r={r} fill={slices[0].color} opacity="0.9" />
        ) : (
          slices.map((s) => (
            <path key={s.label} d={arcPath(cx, cy, r, s.start, s.end)} fill={s.color} opacity="0.9" stroke={T.well} strokeWidth="1.5" />
          ))
        )}
        <circle cx={cx} cy={cy} r={r * 0.56} fill={T.well} />
        <text x={cx} y={cy - 3} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim} letterSpacing="0.06em">
          {title}
        </text>
        <text x={cx} y={cy + 11} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.ink}>
          {fmtNum(total)}
        </text>
      </svg>
      <div style={{ display: "grid", gap: 7, minWidth: 130 }}>
        {segments.map((s) => {
          const frac = total > 0 ? (s.value / total) * 100 : 0;
          return (
            <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ width: 9, height: 9, borderRadius: 2, background: s.color, flexShrink: 0 }} />
              <span style={{ fontFamily: sans, fontSize: 12, color: T.soft, flex: 1 }}>{s.label}</span>
              <span style={{ fontFamily: mono, fontSize: 11.5, color: T.ink }}>
                {fmtNum(s.value)}
                {unit ? ` ${unit}` : ""} · {frac.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
