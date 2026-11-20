import { Canvas, heat } from "./draw";
import { T, mono } from "./theme";

// Canvas plots for the audio scenes: waveform, heatmap (row 0 at the bottom),
// line/bar spectrum, histogram. All take real arrays.

export const W0 = 760; // default plot width (fits the scene panel at a 1200px window)

// waveform of y[from:to] as min/max per pixel; optional highlighted sample ranges
export function Wave({ y, from = 0, to = y.length, w = W0, h = 120, color = T.accent, boxes = [], shade, ticks, stems }) {
  return (
    <Canvas w={w} h={h} deps={[y, from, to, color, JSON.stringify(boxes), JSON.stringify(shade), stems]} draw={(x) => {
      const n = to - from, mid = h / 2, amp = h / 2 - 6;
      let m = 1e-6;
      for (let i = from; i < to; i++) m = Math.max(m, Math.abs(y[i] ?? 0));
      const sc = amp / Math.max(m, 0.05);
      if (shade) {
        x.fillStyle = shade.fill ?? "rgba(148,161,178,0.10)";
        x.fillRect(((shade.a - from) / n) * w, 0, ((shade.b - shade.a) / n) * w, h);
      }
      x.strokeStyle = T.rule; x.lineWidth = 1;
      x.beginPath(); x.moveTo(0, mid); x.lineTo(w, mid); x.stroke();
      for (const b of boxes) {
        const a = ((b.a - from) / n) * w, z = ((b.b - from) / n) * w;
        x.fillStyle = b.fill ?? `${b.color ?? T.wire}22`;
        x.fillRect(a, b.y0 ?? 2, z - a, (b.y1 ?? h - 2) - (b.y0 ?? 2));
        x.strokeStyle = b.color ?? T.wire; x.lineWidth = b.lw ?? 1.5;
        x.strokeRect(a + 0.5, (b.y0 ?? 2) + 0.5, z - a - 1, (b.y1 ?? h - 2) - (b.y0 ?? 2) - 1);
      }
      if (stems) {
        // one stem per sample (zoomed in)
        const u = w / n;
        for (let i = 0; i < n; i++) {
          const v = y[from + i] ?? 0, px = (i + 0.5) * u;
          x.strokeStyle = typeof stems === "function" ? stems(from + i) : color; x.lineWidth = Math.min(3, u * 0.4);
          x.beginPath(); x.moveTo(px, mid); x.lineTo(px, mid - v * sc); x.stroke();
          x.fillStyle = x.strokeStyle; x.beginPath(); x.arc(px, mid - v * sc, Math.min(4, u * 0.3), 0, 7); x.fill();
        }
      } else {
        x.fillStyle = color;
        const step = n / w;
        for (let px = 0; px < w; px++) {
          const a = from + Math.floor(px * step), b = Math.max(a + 1, from + Math.floor((px + 1) * step));
          let lo = 0, hi = 0;
          for (let i = a; i < b && i < to; i++) { const v = y[i] ?? 0; if (v < lo) lo = v; if (v > hi) hi = v; }
          x.fillRect(px, mid - hi * sc, 1, Math.max(1, (hi - lo) * sc));
        }
      }
      if (ticks) {
        x.fillStyle = T.dim; x.font = `12px ${mono}`;
        for (const [pos, lab] of ticks) {
          const px = ((pos - from) / n) * w;
          x.fillRect(px, h - 5, 1, 5);
          x.textAlign = px < 20 ? "left" : px > w - 20 ? "right" : "center";
          x.fillText(lab, px, h - 8);
        }
      }
    }} />
  );
}

// heatmap of data[r * cols + c] for rows × [c0, c1); row 0 drawn at the bottom
export function Heat({ data, rows, cols, c0 = 0, c1 = cols, w = W0, h = 200, lo, hi, map = (v) => v, col, row, cell, vline, smooth = true }) {
  return (
    <Canvas w={w} h={h} deps={[data, rows, cols, c0, c1, lo, hi, col, row, JSON.stringify(cell), JSON.stringify(vline)]} draw={(x) => {
      const nc = c1 - c0, off = document.createElement("canvas");
      off.width = nc; off.height = rows;
      const o = off.getContext("2d"), im = o.createImageData(nc, rows);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < nc; c++) {
          const v = map(data[r * cols + c0 + c]);
          const [R, G, B] = heat((v - lo) / (hi - lo));
          const p = ((rows - 1 - r) * nc + c) * 4;
          im.data[p] = R; im.data[p + 1] = G; im.data[p + 2] = B; im.data[p + 3] = 255;
        }
      o.putImageData(im, 0, 0);
      x.imageSmoothingEnabled = smooth && nc > w;
      x.drawImage(off, 0, 0, w, h);
      const uc = w / nc, ur = h / rows;
      x.lineWidth = 1.5;
      if (vline) for (const [c, color] of vline) { x.strokeStyle = color; x.beginPath(); x.moveTo((c - c0) * uc, 0); x.lineTo((c - c0) * uc, h); x.stroke(); }
      if (col != null) { x.strokeStyle = T.accent; x.strokeRect((col - c0) * uc, 0.5, Math.max(uc, 2), h - 1); }
      if (row != null) { x.strokeStyle = T.accent2; x.strokeRect(0.5, (rows - 1 - row) * ur, w - 1, Math.max(ur, 2)); }
      if (cell) { x.strokeStyle = T.ink; x.lineWidth = 2; x.strokeRect((cell[1] - c0) * uc - 2, (rows - 1 - cell[0]) * ur - 2, Math.max(uc, 2) + 4, Math.max(ur, 2) + 4); }
    }} />
  );
}

// simple x/y line + bar plot in SVG
export function Plot({ w = W0, h = 180, xs, series = [], bars, xlab, ylab, xticks = [], vlines = [], pad = { l: 44, r: 10, t: 10, b: 30 }, ymin, ymax, children }) {
  const all = [...series.flatMap((s) => s.y), ...(bars ? bars.y : [])];
  const y0 = ymin ?? Math.min(0, ...all), y1 = ymax ?? Math.max(...all, 1e-9);
  const xa = xs[0], xb = xs[xs.length - 1];
  const X = (v) => pad.l + ((v - xa) / (xb - xa)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0)) * (h - pad.t - pad.b);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ maxWidth: "100%", height: "auto", display: "block", fontFamily: mono }}>
      <line x1={pad.l} x2={w - pad.r} y1={Y(y0)} y2={Y(y0)} stroke={T.rule} />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={h - pad.b} stroke={T.rule} />
      {xticks.map(([v, lab]) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={h - pad.b} y2={h - pad.b + 4} stroke={T.dim} />
          <text x={X(v)} y={h - pad.b + 17} fill={T.dim} fontSize={12} textAnchor="middle">{lab}</text>
        </g>
      ))}
      {xlab && <text x={w - pad.r} y={h - 2} fill={T.dim} fontSize={12} textAnchor="end">{xlab}</text>}
      {ylab && <text x={pad.l + 4} y={pad.t + 10} fill={T.dim} fontSize={12}>{ylab}</text>}
      {vlines.map(([v, color, lab], i) => (
        <g key={i}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={h - pad.b} stroke={color} strokeWidth={1.5} strokeDasharray="4 3" />
          {lab && <text x={X(v) + 4} y={pad.t + 12} fill={color} fontSize={13} fontWeight={700}>{lab}</text>}
        </g>
      ))}
      {bars && bars.y.map((v, i) => {
        const bw = Math.max(1, (w - pad.l - pad.r) / bars.y.length - (bars.gap ?? 0.5));
        return <rect key={i} x={X(xs[i]) - bw / 2} y={Math.min(Y(v), Y(Math.max(0, y0)))} width={bw} height={Math.abs(Y(Math.max(0, y0)) - Y(v))} fill={bars.color?.(i) ?? T.accent} opacity={0.9} />;
      })}
      {series.map((s, k) => (
        <path key={k} d={s.y.map((v, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join("")} fill={s.fill ?? "none"} fillOpacity={s.fillOpacity ?? 0.2}
          stroke={s.color ?? T.accent} strokeWidth={s.lw ?? 1.5} opacity={s.opacity ?? 1} strokeDasharray={s.dash} />
      ))}
      {children && children({ X, Y })}
    </svg>
  );
}

// histogram of values on a fixed x range
export function Hist({ data, lo, hi, nb = 90, w = W0, h = 150, color = T.accent, vlines = [], xticks, xlab }) {
  const cnt = new Array(nb).fill(0);
  for (const v of data) { const b = Math.floor(((v - lo) / (hi - lo)) * nb); if (b >= 0 && b < nb) cnt[b]++; else if (b >= nb) cnt[nb - 1]++; else cnt[0]++; }
  const xs = cnt.map((_, i) => lo + ((i + 0.5) * (hi - lo)) / nb);
  const ly = cnt.map((c) => Math.log10(1 + c));
  return <Plot w={w} h={h} xs={[lo, ...xs, hi]} bars={{ y: [0, ...ly, 0], color: () => color, gap: 1 }} vlines={vlines} xticks={xticks} xlab={xlab} ylab="count (log)" ymin={0} />;
}

export const Cap = ({ children, color = T.soft, size = 15 }) => <div style={{ fontFamily: mono, fontSize: size, color }}>{children}</div>;
