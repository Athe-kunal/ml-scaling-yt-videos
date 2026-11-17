import { createContext, useContext, useEffect, useRef } from "react";
import { T, mono } from "./theme";
import { dimCol } from "./shapes";

// Canvas + small drawing primitives shared by the ViT and CLIP scenes.

// a DPR-aware canvas: draw(ctx, w, h) runs whenever deps change
export function Canvas({ w, h, draw, deps, style }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const x = c.getContext("2d");
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, w, h);
    draw(x, w, h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, h, ...deps]);
  return <canvas ref={ref} style={{ width: w, maxWidth: "100%", aspectRatio: `${w} / ${h}`, ...style }} />;
}

// dark -> pink -> amber ramp for attention weights / probabilities
const STOPS = [[16, 21, 31], [120, 50, 120], [255, 138, 198], [255, 201, 77], [255, 248, 220]];
export function heat(t) {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
  return STOPS[i].map((v, k) => Math.round(v + (STOPS[i + 1][k] - v) * f));
}
export const heatCss = (t) => `rgb(${heat(t).join(",")})`;

// [T, T] attention matrix as an image; values shown relative to each row's max
export function Heatmap({ A, n, size = 300, mark }) {
  return (
    <Canvas
      w={size} h={size} deps={[A, n, mark]}
      draw={(x) => {
        const off = document.createElement("canvas");
        off.width = off.height = n;
        const o = off.getContext("2d"), im = o.createImageData(n, n);
        for (let i = 0; i < n; i++) {
          let m = 0;
          for (let j = 0; j < n; j++) m = Math.max(m, A[i * n + j]);
          for (let j = 0; j < n; j++) {
            const [r, g, b] = heat(Math.sqrt(A[i * n + j] / m));
            const p = (i * n + j) * 4;
            im.data[p] = r; im.data[p + 1] = g; im.data[p + 2] = b; im.data[p + 3] = 255;
          }
        }
        o.putImageData(im, 0, 0);
        x.imageSmoothingEnabled = n > size;
        x.drawImage(off, 0, 0, size, size);
        const u = size / n;
        if (mark != null) {
          x.strokeStyle = T.accent;
          x.lineWidth = 1.5;
          x.strokeRect(0.5, mark * u, size - 1, Math.max(u, 2));
        }
      }}
    />
  );
}

// the image with an optional patch grid, a per-patch overlay (e.g. CLS attention)
// and a highlighted patch
export function PatchImage({ img, P, size = 260, grid = 1, overlay, hi, dim = 0 }) {
  return (
    <Canvas
      w={size} h={size} deps={[img, P, grid, overlay, hi, dim]}
      draw={(x) => {
        const g = Math.floor(img.size / P), u = size / g;
        x.drawImage(img.canvas, 0, 0, g * P, g * P, 0, 0, size, size);
        if (dim) { x.fillStyle = `rgba(10,14,20,${dim})`; x.fillRect(0, 0, size, size); }
        if (overlay) {
          const m = Math.max(...overlay);
          overlay.forEach((v, k) => {
            const t = Math.sqrt(v / m);
            x.fillStyle = `rgba(10,14,20,${0.82 * (1 - t)})`;
            x.fillRect((k % g) * u, Math.floor(k / g) * u, u, u);
          });
        }
        if (grid > 0) {
          x.strokeStyle = `rgba(245,248,251,${0.55 * grid})`;
          x.lineWidth = g > 40 ? 0.4 : 0.8;
          x.beginPath();
          for (let k = 0; k <= g; k++) { x.moveTo(k * u, 0); x.lineTo(k * u, size); x.moveTo(0, k * u); x.lineTo(size, k * u); }
          x.stroke();
        }
        if (hi != null) {
          x.strokeStyle = T.accent;
          x.lineWidth = 2.5;
          x.strokeRect((hi % g) * u, Math.floor(hi / g) * u, u, u);
        }
      }}
    />
  );
}

// ---------- shapes on the diagrams ----------
// views provide {sym: [meaning, value]}; a shape is written "B T D" and shows as
// [B, T=197, D=768] with each symbol in its dimension colour (hover = meaning)
export const DimCtx = createContext({});
const syms = (shape) => (Array.isArray(shape) ? shape : String(shape).split(" ").filter(Boolean));

function DimChip({ s, short }) {
  const G = useContext(DimCtx), g = G[s], v = g?.[1];
  const lit = /^\d+$/.test(s);
  return (
    <span title={g ? `${s}: ${g[0]}` : undefined} style={{ color: lit ? T.soft : dimCol(s), fontWeight: 700 }}>
      {s}{!short && !lit && v != null && v !== s && <span style={{ color: T.soft, fontWeight: 400 }}>={v}</span>}
    </span>
  );
}

export function ShapeTag({ name, shape, color = T.ink, size = 11.5 }) {
  const ds = syms(shape);
  return (
    <span style={{ fontFamily: mono, fontSize: size, whiteSpace: "nowrap" }}>
      {name && <b style={{ color, marginRight: 5 }}>{name}</b>}
      <span style={{ color: T.dim }}>[</span>
      {ds.length === 0 ? <span style={{ color: T.soft }}>scalar</span> : ds.map((x, i) => <span key={i}>{i > 0 && <span style={{ color: T.dim }}>, </span>}<DimChip s={x} /></span>)}
      <span style={{ color: T.dim }}>]</span>
    </span>
  );
}

// in → out strip drawn at the top of a scene: rows [{ name, dims, io }]
export function TagStrip({ rows }) {
  if (!rows || !rows.length) return null;
  const ins = rows.filter((r) => r.io === "in"), outs = rows.filter((r) => r.io === "out"), mid = rows.filter((r) => !r.io);
  const tag = (r, i) => <ShapeTag key={i} name={r.name} shape={r.dims} color={r.io === "out" ? T.accent : T.ink} size={13} />;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", padding: "8px 12px", background: T.well, border: `1px solid ${T.rule}`, borderRadius: 8 }}>
      {ins.map(tag)}
      {mid.length > 0 && <>{ins.length > 0 && <span style={{ color: T.accent2 }}>·</span>}{mid.map(tag)}</>}
      {outs.length > 0 && <>{(ins.length > 0 || mid.length > 0) && <span style={{ color: T.accent2, fontFamily: mono, fontSize: 14 }}>→</span>}{outs.map(tag)}</>}
    </div>
  );
}

// a tensor drawn as a box whose sides grow (log-scaled) with its dims; with
// `shape`, the axes are labelled by the last two symbols and the full shape sits below
const side = (v) => Math.max(16, Math.min(150, 13 * Math.log2(v + 1)));
export function Mat({ name, rows, cols, rl, cl, shape, color = T.accent, stripes = 0, fill, note, glow }) {
  const h = side(rows), w = side(cols), ds = shape ? syms(shape) : null;
  const rowLab = rl ?? (ds && ds.length >= 2 ? <DimChip s={ds[ds.length - 2]} short /> : rows);
  const colLab = cl ?? (ds ? <DimChip s={ds[ds.length - 1]} short /> : cols);
  return (
    <div style={{ display: "grid", justifyItems: "center", gap: 4 }}>
      <div style={{ fontFamily: mono, fontSize: 12, fontWeight: 700, color }}>{name}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, writingMode: "vertical-rl", transform: "rotate(180deg)" }}>{rowLab}</span>
        <div style={{
          width: w, height: h, borderRadius: 4, border: `1.5px solid ${color}`, background: fill ?? `${color}22`, position: "relative", overflow: "hidden",
          boxShadow: glow ? `0 0 0 4px ${color}22` : "none", transition: "all .5s",
        }}>
          {stripes > 1 && (
            <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: `repeat(${stripes}, 1fr)` }}>
              {Array.from({ length: stripes }, (_, k) => (
                <div key={k} style={{ borderRight: k < stripes - 1 ? `1px solid ${T.bg}` : 0, background: `hsl(${(k * 360) / stripes} 70% 65% / 0.35)` }} />
              ))}
            </div>
          )}
        </div>
      </div>
      <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{colLab}</span>
      {ds && <ShapeTag shape={ds} size={12.5} />}
      {note && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire }}>{note}</span>}
    </div>
  );
}

export const Op = ({ children, color = T.accent2 }) => (
  <div style={{ fontFamily: mono, fontSize: 13, color, display: "grid", placeItems: "center", padding: "0 4px", whiteSpace: "nowrap" }}>{children}</div>
);

export const Row = ({ children, gap = 14, wrap = true, style }) => (
  <div style={{ display: "flex", alignItems: "center", gap, flexWrap: wrap ? "wrap" : "nowrap", ...style }}>{children}</div>
);

export const Label = ({ children, color = T.dim }) => (
  <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.1em", color, textTransform: "uppercase" }}>{children}</div>
);

// a vector drawn as a row of bars (values centred on 0)
export function Bars({ v, w = 220, h = 46, color = T.accent, zero = true }) {
  const m = Math.max(1e-6, ...v.map(Math.abs)), bw = w / v.length;
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      {zero && <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke={T.rule} />}
      {v.map((x, i) => {
        const bh = (Math.abs(x) / m) * (h / 2 - 2);
        return <rect key={i} x={i * bw + 0.5} width={Math.max(1, bw - 1)} y={x >= 0 ? h / 2 - bh : h / 2} height={bh} fill={color} opacity={0.85} style={{ transition: "all .5s" }} />;
      })}
    </svg>
  );
}

// ---------- explicit matmuls ----------
// one line: A [.., m, k] @ B [.., k, n] -> C [.., m, n]. The contracted dim k is
// pink and underlined on both sides; output dims are teal. dims may be numbers or
// symbols ("B" counts as 1 for the MAC estimate).
const fmtD = (v) => String(v); // no thousands separator: "3,072" would read as two dims
function Shape({ dims, hi = [], tone = T.soft, hiTone = T.accent2 }) {
  return (
    <span style={{ color: T.dim }}>
      [{dims.map((v, i) => (
        <span key={i}>
          {i > 0 && ", "}
          <span style={{ color: hi.includes(i) ? hiTone : tone, fontWeight: hi.includes(i) ? 700 : 400, textDecoration: hi.includes(i) ? "underline" : "none", textUnderlineOffset: 3 }}>{fmtD(v)}</span>
        </span>
      ))}]
    </span>
  );
}
const num = (v) => (typeof v === "number" ? v : 1);
export function MM({ a, b, c, op = "@", note, times }) {
  const [an, ad] = a, [bn, bd] = b, [cn, cd] = c;
  const k = ad[ad.length - 1];
  const macs = cd.reduce((p, v) => p * num(v), 1) * num(k) * (times ?? 1);
  const mac = macs >= 1e9 ? `${(macs / 1e9).toFixed(2)} GMAC` : macs >= 1e6 ? `${(macs / 1e6).toFixed(1)} MMAC` : macs >= 1e3 ? `${(macs / 1e3).toFixed(1)} KMAC` : `${macs} MAC`;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "baseline", fontFamily: mono, fontSize: 12.5, lineHeight: 1.5 }}>
      <div style={{ overflowWrap: "anywhere" }}>
        <b style={{ color: T.ink }}>{an}</b> <Shape dims={ad} hi={[ad.length - 1]} />
        <span style={{ color: T.accent2, fontWeight: 700 }}> {op} </span>
        <b style={{ color: T.ink }}>{bn}</b> <Shape dims={bd} hi={[bd.length - 2]} />
        <span style={{ color: T.dim }}> → </span>
        <b style={{ color: T.accent }}>{cn}</b> <Shape dims={cd} tone={T.accent} />
        {times && <span style={{ color: T.wire }}>  × {times}</span>}
        {note && <div style={{ fontSize: 11, color: T.dim, paddingLeft: 12 }}>{note}</div>}
      </div>
      <span style={{ fontSize: 10.5, color: T.dim, whiteSpace: "nowrap" }}>{mac}</span>
    </div>
  );
}
// a non-matmul line in the same card (reshape, softmax, gather, …)
export const MMNote = ({ children }) => <div style={{ fontFamily: mono, fontSize: 11.5, color: T.soft, paddingLeft: 12, borderLeft: `2px solid ${T.rule}` }}>{children}</div>;

export function MatMulCard({ children, footer }) {
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>
        <span>TENSOR MULTIPLICATIONS · <span style={{ color: T.accent2 }}>pink = contracted dim</span></span><span>multiply-adds (symbol B counted as 1)</span>
      </div>
      {children}
      {footer && <div style={{ fontFamily: mono, fontSize: 11, color: T.dim, borderTop: `1px solid ${T.rule}`, paddingTop: 6 }}>{footer}</div>}
    </div>
  );
}

// the six matmuls inside one pre-LN transformer block, optionally repeated × L
export function BlockMMs({ B, T: t, D, h, d, M, L, causal, names = {} }) {
  const x = names.x ?? "x";
  return (
    <>
      <MM a={[`LN(${x})`, [B, t, D]]} b={["W_qkv", [D, 3 * D]]} c={["qkv", [B, t, 3 * D]]} times={L} note={`split + reshape → q, k, v each [${B}, ${h}, ${t}, ${d}]`} />
      <MM a={["q", [B, h, t, d]]} b={["kᵀ", [B, h, d, t]]} c={["S", [B, h, t, t]]} times={L} note={`÷ √${d}${causal ? ", + causal mask (−∞ above diagonal)" : ""}, softmax over the last dim → A`} />
      <MM a={["A", [B, h, t, t]]} b={["v", [B, h, t, d]]} c={["o", [B, h, t, d]]} times={L} note={`concat heads → [${B}, ${t}, ${D}]`} />
      <MM a={["o", [B, t, D]]} b={["W_O", [D, D]]} c={["attn", [B, t, D]]} times={L} note={`${x} = ${x} + attn`} />
      <MM a={[`LN(${x})`, [B, t, D]]} b={["W_1", [D, M]]} c={["h", [B, t, M]]} times={L} note="GELU" />
      <MM a={["h", [B, t, M]]} b={["W_2", [M, D]]} c={["mlp", [B, t, D]]} times={L} note={`${x} = ${x} + mlp`} />
    </>
  );
}
