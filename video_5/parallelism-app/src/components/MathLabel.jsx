import { katexHtml } from "../lib/latex";

// SVG <text> can't host arbitrary HTML, so this renders a pure-math label
// (a diagram box's identifier, e.g. "G_i^{(0)}" or "dWout[F,D]") as real
// KaTeX inside a <foreignObject>, centered in the given box.
export default function MathLabel({ x, y, w, h, text, color = "#fff", fontSize = 12.5 }) {
  const html = katexHtml(text, false);
  return (
    <foreignObject x={x} y={y} width={w} height={h} style={{ overflow: "visible", pointerEvents: "none" }}>
      <div
        xmlns="http://www.w3.org/1999/xhtml"
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color,
          fontSize,
          lineHeight: 1,
          transition: "color .35s ease",
        }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </foreignObject>
  );
}
