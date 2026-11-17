import { T, mono } from "./theme";

// Dimension colours (used by the shape tags drawn on the diagrams) and the
// code-line renderer for the flowchart's "{ }" panel.

const DIM_COL = {
  B: "#FFC94D", C: "#9BE564", H: "#8DB4FF", W: "#8DB4FF", P: "#FF8AC6", N: "#6EF3DD", T: "#6EF3DD",
  D: "#C79BFF", h: "#FFB86B", d: "#FF8AC6", M: "#9BE564", K: "#FFC94D", "P²C": "#FF8AC6", "3D": "#C79BFF",
  n: "#FFC94D", g: "#6EF3DD", L: "#6EF3DD", V: "#FFB86B", E: "#FF8AC6", Di: "#C79BFF", Dt: "#8DB4FF", "3Di": "#C79BFF", "3Dt": "#8DB4FF",
};
export const dimCol = (s) => DIM_COL[s] ?? T.soft;

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
