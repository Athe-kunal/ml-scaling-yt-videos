import { T, mono } from "./theme";

// Shape notes, as in the ring-attention walkthrough: every step lists the
// tensors it touches with their shapes, what each dimension means, and the
// numbers used in the toy example. Code lines carry the same shapes as
// trailing comments.

const DIM_COL = {
  G: "#6EF3DD", S: "#FFC94D", E: "#FF8AC6", C: "#9BE564", M: "#8DB4FF", H: "#C79BFF",
  T: "#FFC94D", N: "#FF8AC6", K: "#9BE564", d: "#8DB4FF", B: "#FFC94D", R: "#6EF3DD", L: "#C79BFF",
  Dsrc: "#6EF3DD", Ddst: "#FF8AC6",
};
const dimCol = (s) => DIM_COL[s] ?? T.soft;

const Chip = ({ s, size }) => (
  <span style={{ fontFamily: mono, fontSize: 11.5, fontWeight: 700, color: dimCol(s), padding: "1px 6px", borderRadius: 5, border: `1px solid ${dimCol(s)}66`, background: `${dimCol(s)}12` }}>
    {s}{size != null && <span style={{ color: T.soft, fontWeight: 500 }}>={size}</span>}
  </span>
);

// rows: [{ name, dims: ["G","S","E"], what, io?: "in"|"out", note? }]
// glossary: { G: ["groups, one per device", 4], ... }
export function ShapeCard({ rows, glossary }) {
  if (!rows || !rows.length) return null;
  const used = [];
  rows.forEach((r) => r.dims.forEach((d) => { if (!used.includes(d)) used.push(d); }));
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 10 }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>SHAPES</div>
      {rows.map((r, i) => (
        <div key={i} style={{ display: "grid", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", fontFamily: mono, fontSize: 12.5 }}>
            {r.io && <span style={{ fontSize: 9.5, letterSpacing: "0.08em", color: r.io === "out" ? T.accent : T.dim }}>{r.io === "out" ? "OUT" : "IN"}</span>}
            <span style={{ fontWeight: 700, color: T.ink }}>{r.name}</span>
            <span style={{ color: T.dim }}>[</span>
            {r.dims.length === 0 ? <span style={{ color: T.soft }}>scalar</span> : r.dims.map((d, j) => <Chip key={j} s={d} />)}
            <span style={{ color: T.dim }}>]</span>
            {r.dims.length > 0 && (
              <span style={{ color: T.soft }}>
                = [{r.dims.map((d) => (glossary[d]?.[1] != null ? glossary[d][1] : d)).join(", ")}]
              </span>
            )}
            {r.note && <span style={{ fontSize: 10.5, color: T.wire, border: `1px solid ${T.wire}55`, borderRadius: 10, padding: "1px 8px" }}>{r.note}</span>}
          </div>
          {r.what && <div style={{ fontSize: 12, color: T.dim, lineHeight: 1.5 }}>{r.what}</div>}
        </div>
      ))}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px", paddingTop: 8, borderTop: `1px solid ${T.rule}` }}>
        {used.map((d) => glossary[d] && (
          <span key={d} style={{ display: "inline-flex", gap: 6, alignItems: "center", fontFamily: mono, fontSize: 11, color: T.dim }}>
            <Chip s={d} size={glossary[d][1]} /> {glossary[d][0]}
          </span>
        ))}
      </div>
    </div>
  );
}

// one line of code; a trailing "# …" comment is dimmed and set apart
export function CodeLine({ n, text }) {
  const i = text.indexOf("#");
  const code = i < 0 ? text : text.slice(0, i).replace(/\s+$/, "");
  const com = i < 0 ? null : text.slice(i);
  const wrap = { whiteSpace: "pre-wrap", overflowWrap: "anywhere" };
  const pad = n != null ? 24 : 0;
  const tone = com && (com.includes("[") || com.includes("→")) ? T.wire : T.dim;
  return (
    <div style={{ marginBottom: com && code ? 3 : 0 }}>
      {code && (
        <div style={{ ...wrap, paddingLeft: pad, textIndent: -pad }}>
          {n != null && <span style={{ color: T.dim, display: "inline-block", width: 24, textIndent: 0 }}>{String(n).startsWith("s") ? "" : n}</span>}
          {code}
        </div>
      )}
      {com && <div style={{ ...wrap, paddingLeft: pad + (code ? 12 : 0), color: tone, fontSize: 10.5 }}>{com}</div>}
    </div>
  );
}
