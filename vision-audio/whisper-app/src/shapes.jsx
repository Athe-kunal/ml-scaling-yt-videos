import { T, mono } from "./theme";

// Dimension colours (used by the shape tags drawn on the diagrams) and the
// code-line renderer for the flowchart's "{ }" panel.

const DIM_COL = {
  // whisper: S samples · F freq bins · M mel bins · T mel frames · A audio tokens · n text tokens · V vocab
  S: "#9BE564", F: "#FFB86B", M: "#FF8AC6", T: "#8DB4FF", A: "#6EF3DD", n: "#FFC94D", V: "#FFB86B",
  D: "#C79BFF", h: "#FFB86B", d: "#FF8AC6", "4D": "#C79BFF", "3M": "#FF8AC6", "3D": "#C79BFF", B: "#FFC94D", K: "#9BE564", L: "#6EF3DD",
  // qwen2-audio: N audio tokens kept · P pooled 750 · E LLM width · I SwiGLU width · Q sequence
  N: "#6EF3DD", P: "#9BE564", E: "#FF8AC6", I: "#FFB86B", Q: "#FFC94D",
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
      {com && <div style={{ ...wrap, paddingLeft: pad + (code ? 12 : 0), color: tone, fontSize: 13 }}>{com}</div>}
    </div>
  );
}
