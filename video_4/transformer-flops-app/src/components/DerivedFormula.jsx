import { fmtNum, fmtPlain, exprValue } from "../lib/steps";
import { T, mono } from "../lib/theme";

// Shows a formula three ways so the final number is never just dropped on
// the reader: the symbolic product (e.g. "6 · B · T · D · N · H"), the same
// product with the current dimension values substituted in, and the
// resulting total — exactly how the number was derived, not just what it is.
export default function DerivedFormula({ label, expr, dims, formatResult = fmtNum, resultColor }) {
  if (!expr) return null;
  const { coef, symbols } = expr;
  const showCoef = coef !== 1;
  const symbolic = [showCoef ? String(coef) : null, ...symbols].filter((v) => v !== null).join(" · ");
  const substituted = [showCoef ? String(coef) : null, ...symbols.map((s) => fmtPlain(dims[s]))]
    .filter((v) => v !== null)
    .join(" · ");
  const result = exprValue(expr, dims);

  return (
    <div
      style={{
        fontFamily: mono,
        fontSize: 12,
        background: T.well,
        border: `1px solid ${T.rule}`,
        borderRadius: 8,
        padding: "10px 12px",
        display: "grid",
        gap: 4,
      }}
    >
      <div style={{ color: T.soft }}>
        <span style={{ color: T.dim }}>{label} = </span>
        {symbolic}
      </div>
      <div style={{ color: T.soft, paddingLeft: 14 }}>= {substituted}</div>
      <div style={{ color: resultColor || T.accent, fontWeight: 700, paddingLeft: 14 }}>
        = {formatResult(result)}
      </div>
    </div>
  );
}
