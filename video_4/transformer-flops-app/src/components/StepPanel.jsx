import { STEPS, fmtNum, cumulativeTotals } from "../lib/steps";
import { T, mono, sans, cond } from "../lib/theme";

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

function Stat({ label, value, accent }) {
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
          color: accent ? T.accent : T.ink,
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
  const stepParams = s.params(dims);
  const stepFlops = s.flops(dims);
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
        style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: "#B6C2CF", marginBottom: 14 }}
        dangerouslySetInnerHTML={{ __html: s.body }}
      />

      {s.formulaLabel && (
        <div
          style={{
            fontFamily: mono,
            fontSize: 12.5,
            color: T.ink,
            background: T.well,
            border: `1px solid ${T.rule}`,
            borderRadius: 8,
            padding: "9px 12px",
            marginBottom: 14,
          }}
        >
          {s.formulaLabel}
        </div>
      )}

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
        <Stat label="this step · params" value={fmtNum(stepParams)} accent={stepParams > 0} />
        <Stat label="this step · flops" value={fmtNum(stepFlops)} accent={stepFlops > 0} />
      </div>

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
