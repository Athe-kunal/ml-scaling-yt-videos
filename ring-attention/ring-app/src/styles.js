import { T, mono, sans } from "./theme";

// Style helpers shared by the ring-family and tree views.

export const panel = { background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 16, minWidth: 0 };
export const h2 = { margin: "0 0 12px", fontFamily: sans, fontSize: 12, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600 };
export const chip = (c) => ({
  display: "inline-block", fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase",
  padding: "3px 9px", borderRadius: 5, color: c, border: `1px solid ${c}55`, background: `${c}14`,
});
export const navBtn = (disabled) => ({
  fontFamily: mono, fontSize: 13, height: 32, padding: "0 12px", borderRadius: 6, flexShrink: 0,
  border: `1px solid ${T.rule}`, background: disabled ? "transparent" : T.well,
  color: disabled ? T.rule : T.ink, cursor: disabled ? "default" : "pointer",
});
