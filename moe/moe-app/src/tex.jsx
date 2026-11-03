import katex from "katex";
import { T } from "./theme";

// KaTeX inside the SVG canvases: a foreignObject box whose HTML is the rendered math.
// (x, y) is the top-left corner, or top-centre / top-right for align "center" / "right".
const html = (tex) => {
  try {
    return katex.renderToString(tex, { throwOnError: false, strict: false });
  } catch {
    return tex;
  }
};

export function Tex({ x, y, tex, w = 320, h = 28, size = 13, color = T.soft, align = "left", opacity = 1 }) {
  const x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  return (
    <foreignObject x={x0} y={y} width={w} height={h} style={{ overflow: "visible", pointerEvents: "none" }} opacity={opacity}>
      <div style={{ fontSize: size, color, textAlign: align, whiteSpace: "nowrap", lineHeight: 1.25 }} dangerouslySetInnerHTML={{ __html: html(tex) }} />
    </foreignObject>
  );
}
