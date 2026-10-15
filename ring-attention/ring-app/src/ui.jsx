import { withInlineMath, katexHtml } from "./latex";

// KaTeX-rendering components shared by the ring-family and tree views.

export const Rich = ({ html, style }) => <div style={style} dangerouslySetInnerHTML={{ __html: withInlineMath(html) }} />;
export const Tex = ({ tex }) => <div style={{ overflowX: "auto", overflowY: "hidden", padding: "2px 0" }} dangerouslySetInnerHTML={{ __html: katexHtml(tex, true) }} />;
