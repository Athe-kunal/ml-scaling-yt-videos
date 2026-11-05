import { T, mono } from "./theme";
import { XC } from "./lbModel";

// Small SVG chart pieces shared by the balancing tabs.

export const f2 = (x) => x.toFixed(2).replace(/^(-?)0\./, "$1.");

// Vertical bars from a zero baseline (negative values hang below it).
export function Bars({
  x, y, w, h, values, max, min = 0, colors = XC, ideal, idealLabel = "balanced", title, fmt = (v) => v.toFixed(0), labels, dim = [], flag = [], showVals = true, arrows, hl, onPick,
}) {
  const n = values.length;
  const hi = max ?? Math.max(...values, ideal ?? 0) * 1.12;
  const lo0 = Math.min(min, ...values);
  const lo = lo0 < 0 ? lo0 * 1.3 : lo0; // headroom so value labels clear the axis names
  const sc = h / (hi - lo || 1);
  const zero = y + h - (0 - lo) * sc;
  const bw = w / n;
  return (
    <g>
      {title && <text x={x} y={y - 10} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{title}</text>}
      <line x1={x} x2={x + w} y1={zero} y2={zero} stroke={T.rule} />
      {values.map((v, j) => {
        const bh = Math.abs(v) * sc;
        const by = v >= 0 ? zero - bh : zero;
        const col = Array.isArray(colors) ? colors[j] : colors;
        return (
          <g key={j} opacity={dim.includes(j) ? 0.3 : 1}>
            <rect x={x + j * bw + 3} y={by} width={bw - 6} height={Math.max(bh, 0.5)} rx={3} fill={col} stroke={flag.includes(j) ? T.bad : "none"} strokeWidth={2} style={{ transition: "all .5s" }} />
            {showVals && (
              <text x={x + j * bw + bw / 2} y={v >= 0 ? by - 5 : by + bh + 12} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={flag.includes(j) ? T.bad : T.soft}>{fmt(v)}</text>
            )}
            {hl === j && <rect x={x + j * bw + 1} y={Math.min(by, zero) - (v >= 0 ? 16 : 2)} width={bw - 2} height={Math.max(bh, 0.5) + 20} rx={5} fill="none" stroke={T.ink} strokeWidth={2.2} />}
            {onPick && <rect x={x + j * bw} y={y - 8} width={bw} height={h + 40} fill="transparent" style={{ cursor: "pointer" }} onClick={() => onPick(j)} />}
            {arrows && arrows[j] && (
              <text x={x + j * bw + bw / 2} y={y + h + 30} textAnchor="middle" fontFamily={mono} fontSize="12" fill={arrows[j] > 0 ? T.accent : T.bad}>{arrows[j] > 0 ? "▲" : "▼"}</text>
            )}
            <text x={x + j * bw + bw / 2} y={y + h + 14 + (lo < 0 ? 0 : 0)} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={col}>{labels ? labels[j] : `E${j}`}</text>
          </g>
        );
      })}
      {ideal != null && (
        <g>
          <line x1={x} x2={x + w} y1={zero - (ideal - 0) * sc} y2={zero - ideal * sc} stroke={T.wire} strokeDasharray="5 4" />
          <text x={x + w} y={zero - ideal * sc - 4} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.wire}>{idealLabel} {fmt(ideal)}</text>
        </g>
      )}
    </g>
  );
}

// Polyline chart. series: [{ pts, color, label }]; only points <= upTo are drawn.
export function Lines({ x, y, w, h, series, ymax, upTo, title, ref1 }) {
  const N = Math.max(...series.map((s) => s.pts.length));
  const hi = ymax ?? Math.max(...series.flatMap((s) => s.pts)) * 1.1;
  const px = (i) => x + (i / (N - 1)) * w;
  const py = (v) => y + h - (Math.min(v, hi) / hi) * h;
  return (
    <g>
      {title && <text x={x} y={y - 10} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{title}</text>}
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      {ref1 != null && <line x1={x} x2={x + w} y1={py(ref1)} y2={py(ref1)} stroke={T.wire} strokeDasharray="4 4" />}
      {series.map((s, k) => {
        const cut = upTo == null ? s.pts.length - 1 : Math.min(upTo, s.pts.length - 1);
        const d = s.pts.slice(0, cut + 1).map((v, i) => `${i ? "L" : "M"}${px(i)},${py(v)}`).join(" ");
        return (
          <g key={k}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={2} />
            <circle cx={px(cut)} cy={py(s.pts[cut])} r={4} fill={s.color} />
            <text x={x + 8} y={y + 14 + k * 14} fontFamily={mono} fontSize="10" fontWeight="700" fill={s.color}>{s.label}</text>
          </g>
        );
      })}
    </g>
  );
}

// diverging color for margins: warm = positive, cool = negative
export const div = (v, s = 0.25) => {
  const a = Math.min(1, Math.abs(v) / s);
  return `color-mix(in srgb, ${v >= 0 ? T.accent2 : "#5B8DEF"} ${Math.round(a * 75)}%, ${T.well})`;
};
