import katex from "katex";

// Renders a string that is ENTIRELY math (diagram box labels, e.g.
// "W_i^{(0)}" or "dWout[F,D]") as real KaTeX HTML.
export function katexHtml(math, displayMode = false) {
  try {
    return katex.renderToString(math, { throwOnError: false, displayMode, strict: false });
  } catch {
    return math;
  }
}

const INLINE_MATH_RE = /\$([^$]+)\$/g;

// For prose strings (notation/formula/note/body) that mix plain English
// with math spans delimited by $...$ — e.g. "computed locally on device
// $X$" — replaces each $...$ span with rendered KaTeX HTML and leaves
// everything else (including existing <b>/<i> tags) untouched. Safe to
// feed the result straight into dangerouslySetInnerHTML.
export function withInlineMath(str) {
  if (!str) return str;
  return str.replace(INLINE_MATH_RE, (_, math) => katexHtml(math, false));
}
