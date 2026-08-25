import { STEPS, fmtNum, fmtBytes, cumulativeTotals, stepParams, stepFlops, stepMem } from "../lib/steps";
import { T, mono, sans, cond } from "../lib/theme";
import DerivedFormula from "./DerivedFormula";

const navBtn = (disabled) => ({
  fontFamily: mono,
  fontSize: 15,
  width: 34,
  height: 30,
  borderRadius: 6,
  border: `1px solid ${T.rule}`,
  background: disabled ? "transparent" : T.well,
  color: disabled ? T.rule : T.ink,
  cursor: disabled ? "default" : "pointer",
  transition: "all .16s",
});

function Stat({ label, value, accent, color }) {
  return (
    <div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 9.5,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: T.dim,
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 17,
          fontWeight: 600,
          color: color || (accent ? T.accent : T.ink),
        }}
      >
        {value}
      </div>
    </div>
  );
}

export default function StepPanel({ step, dims, onStep }) {
  const idx = step;
  const s = STEPS[idx];
  const thisStepParams = stepParams(s, dims);
  const thisStepFlops = stepFlops(s, dims);
  const thisStepMem = stepMem(s, dims);
  const totals = cumulativeTotals(dims, idx);

  return (
    <div
      style={{
        background: T.panel,
        border: `1px solid ${T.rule}`,
        borderRadius: 10,
        padding: 18,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <button
          onClick={() => onStep(idx - 1)}
          disabled={idx === 0}
          aria-label="Previous step"
          style={navBtn(idx === 0)}
        >
          ←
        </button>
        <div style={{ display: "flex", gap: 4, flex: 1, justifyContent: "center", flexWrap: "wrap" }}>
          {STEPS.map((st, i) => (
            <button
              key={st.id}
              onClick={() => onStep(i)}
              aria-label={`Step ${i + 1}: ${st.title}`}
              style={{
                width: i === idx ? 20 : 7,
                height: 7,
                borderRadius: 4,
                border: "none",
                cursor: "pointer",
                padding: 0,
                background: i === idx ? T.accent : i < idx ? T.dim : T.rule,
                transition: "all .25s cubic-bezier(.2,.7,.3,1)",
              }}
            />
          ))}
        </div>
        <button
          onClick={() => onStep(idx + 1)}
          disabled={idx === STEPS.length - 1}
          aria-label="Next step"
          style={navBtn(idx === STEPS.length - 1)}
        >
          →
        </button>
      </div>

      <div
        style={{
          display: "inline-block",
          fontFamily: mono,
          fontSize: 9.5,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          padding: "3px 9px",
          borderRadius: 5,
          marginBottom: 10,
          color: s.phase === "attn" ? T.accent : T.accent2,
          border: `1px solid ${s.phase === "attn" ? T.accent : T.accent2}55`,
          background: `${s.phase === "attn" ? T.accent : T.accent2}14`,
        }}
      >
        step {idx + 1} / {STEPS.length} · {s.phase === "attn" ? "attention" : "mlp"}
      </div>

      <div
        style={{
          fontFamily: cond,
          fontSize: 19,
          fontWeight: 700,
          color: T.ink,
          marginBottom: 6,
        }}
      >
        {s.title}
      </div>
      <div
        style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: T.soft, marginBottom: 14 }}
        dangerouslySetInnerHTML={{ __html: s.body }}
      />

      {s.memExpr ? (
        <div style={{ display: "grid", gap: 8, marginBottom: 6 }}>
          <DerivedFormula label="bytes" expr={s.memExpr} dims={dims} formatResult={fmtBytes} resultColor={T.wire} />
        </div>
      ) : (s.paramsExpr || s.flopsExpr) ? (
        <div style={{ display: "grid", gap: 8, marginBottom: s.excludedFromTotals ? 6 : 16 }}>
          <DerivedFormula label="params" expr={s.paramsExpr} dims={dims} />
          <DerivedFormula label="flops" expr={s.flopsExpr} dims={dims} />
        </div>
      ) : null}

      {s.memExpr && (
        <div
          style={{
            fontFamily: mono,
            fontSize: 11,
            color: T.dim,
            marginBottom: 16,
          }}
        >
          ↳ memory, not compute — not counted in the params/flops totals below.
        </div>
      )}

      {s.excludedFromTotals && (
        <div
          style={{
            fontFamily: mono,
            fontSize: 11,
            color: T.dim,
            marginBottom: 16,
          }}
        >
          ↳ computed above, but excluded from the running totals below — negligible next to the surrounding O(D²) matmuls.
        </div>
      )}

      {!s.paramsExpr && !s.flopsExpr && !s.memExpr && !s.excludedFromTotals && (
        <div
          style={{
            fontFamily: mono,
            fontSize: 11.5,
            color: T.dim,
            background: T.well,
            border: `1px solid ${T.rule}`,
            borderRadius: 8,
            padding: "9px 12px",
            marginBottom: 16,
          }}
        >
          shape / elementwise op — no weights, no formula to derive
        </div>
      )}

      {s.memExpr ? (
        <div
          style={{
            marginBottom: 16,
            paddingBottom: 16,
            borderBottom: `1px solid ${T.rule}`,
          }}
        >
          <Stat label="this step · KV cache" value={fmtBytes(thisStepMem)} accent color={T.wire} />
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            marginBottom: 16,
            paddingBottom: 16,
            borderBottom: `1px solid ${T.rule}`,
          }}
        >
          <Stat label="this step · params" value={fmtNum(thisStepParams)} accent={thisStepParams > 0} />
          <Stat label="this step · flops" value={fmtNum(thisStepFlops)} accent={thisStepFlops > 0} />
        </div>
      )}

      <div
        style={{
          fontFamily: mono,
          fontSize: 9.5,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: T.dim,
          marginBottom: 10,
        }}
      >
        Cumulative through this step
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Stat label="params so far" value={fmtNum(totals.params)} accent />
        <Stat label="flops so far" value={fmtNum(totals.flops)} accent />
        <Stat label="attention params" value={fmtNum(totals.attnParams)} />
        <Stat label="attention flops" value={fmtNum(totals.attnFlops)} />
        <Stat label="mlp params" value={fmtNum(totals.mlpParams)} />
        <Stat label="mlp flops" value={fmtNum(totals.mlpFlops)} />
      </div>
    </div>
  );
}
