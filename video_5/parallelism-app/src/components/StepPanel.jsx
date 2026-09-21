import { T, mono, sans, cond } from "../lib/theme";
import { withInlineMath } from "../lib/latex";

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
  flexShrink: 0,
});

export default function StepPanel({ steps, step, onStep }) {
  const idx = step;
  const s = steps[idx];

  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <button onClick={() => onStep(idx - 1)} disabled={idx === 0} aria-label="Previous step" style={navBtn(idx === 0)}>
          ←
        </button>
        <div style={{ display: "flex", gap: 4, flex: 1, justifyContent: "center", flexWrap: "wrap" }}>
          {steps.map((st, i) => (
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
          disabled={idx === steps.length - 1}
          aria-label="Next step"
          style={navBtn(idx === steps.length - 1)}
        >
          →
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <div
          style={{
            display: "inline-block",
            fontFamily: mono,
            fontSize: 9.5,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            padding: "3px 9px",
            borderRadius: 5,
            color: T.accent,
            border: `1px solid ${T.accent}55`,
            background: `${T.accent}14`,
          }}
        >
          step {idx + 1} / {steps.length}
        </div>
        {s.sectionLabel && (
          <div
            style={{
              display: "inline-block",
              fontFamily: mono,
              fontSize: 9.5,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              padding: "3px 9px",
              borderRadius: 5,
              color: T.wire,
              border: `1px solid ${T.wire}55`,
              background: `${T.wire}14`,
            }}
          >
            {s.sectionLabel}
          </div>
        )}
      </div>

      <div style={{ fontFamily: cond, fontSize: 19, fontWeight: 700, color: T.ink, marginBottom: 8 }}>{s.title}</div>

      <div
        style={{
          fontSize: 14,
          lineHeight: 1.8,
          color: T.accent2,
          background: T.well,
          border: `1px solid ${T.rule}`,
          borderRadius: 8,
          padding: "10px 12px",
          marginBottom: 12,
          overflowX: "auto",
        }}
        dangerouslySetInnerHTML={{ __html: withInlineMath(s.notation) }}
      />

      {(s.flops || s.comms) && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          {s.flops && (
            <div
              style={{
                flex: "1 1 200px",
                fontSize: 12.5,
                lineHeight: 1.6,
                color: T.accent,
                background: `${T.accent}0f`,
                border: `1px solid ${T.accent}40`,
                borderRadius: 8,
                padding: "7px 10px",
                overflowX: "auto",
              }}
            >
              <span style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: "0.1em", textTransform: "uppercase", opacity: 0.85, marginRight: 8 }}>
                FLOPs
              </span>
              <span dangerouslySetInnerHTML={{ __html: withInlineMath(s.flops) }} />
            </div>
          )}
          {s.comms && (
            <div
              style={{
                flex: "1 1 200px",
                fontSize: 12.5,
                lineHeight: 1.6,
                color: T.wire,
                background: `${T.wire}0f`,
                border: `1px solid ${T.wire}40`,
                borderRadius: 8,
                padding: "7px 10px",
                overflowX: "auto",
              }}
            >
              <span style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: "0.1em", textTransform: "uppercase", opacity: 0.85, marginRight: 8 }}>
                Comms
              </span>
              <span dangerouslySetInnerHTML={{ __html: withInlineMath(s.comms) }} />
            </div>
          )}
        </div>
      )}

      <div
        style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: T.soft, marginBottom: s.note || s.formula ? 12 : 0 }}
        dangerouslySetInnerHTML={{ __html: withInlineMath(s.body) }}
      />

      {s.note && (
        <div
          style={{
            fontFamily: sans,
            fontSize: 12,
            lineHeight: 1.55,
            color: T.dim,
            borderLeft: `2px solid ${T.wire}`,
            paddingLeft: 10,
            marginBottom: s.formula ? 12 : 0,
          }}
          dangerouslySetInnerHTML={{ __html: withInlineMath(s.note) }}
        />
      )}

      {s.formula && (
        <div
          style={{
            fontSize: 13.5,
            lineHeight: 1.7,
            color: T.wire,
            background: T.well,
            border: `1px solid ${T.rule}`,
            borderRadius: 8,
            padding: "10px 12px",
            overflowX: "auto",
          }}
          dangerouslySetInnerHTML={{ __html: withInlineMath(s.formula) }}
        />
      )}
    </div>
  );
}
