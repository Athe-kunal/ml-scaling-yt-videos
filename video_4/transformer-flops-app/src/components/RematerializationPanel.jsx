import { STEPS, CHECKPOINT_META, fmtNum, stepRemat, totalRemat, layerTotals } from "../lib/steps";
import { T, mono, sans, cond } from "../lib/theme";
import DerivedFormula from "./DerivedFormula";

const COLOR = { accent: T.accent, bad: T.bad, wire: T.wire, soft: T.soft };

function LegendRow({ category }) {
  const meta = CHECKPOINT_META[category];
  const color = COLOR[meta.colorKey];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 3,
          background: color,
          flexShrink: 0,
        }}
      />
      <span style={{ fontFamily: mono, fontSize: 11.5, color: T.ink, minWidth: 76 }}>{meta.label}</span>
      <span style={{ fontFamily: sans, fontSize: 11.5, color: T.soft }}>{meta.detail}</span>
    </div>
  );
}

export default function RematerializationPanel({ dims }) {
  const rematSteps = STEPS.filter((s) => s.rematExpr);
  const extra = totalRemat(dims);
  const totals = layerTotals(dims);
  const normalTrainFlops = totals.flops; // 6x forward+backward, this layer
  const overheadPct = normalTrainFlops > 0 ? (extra / normalTrainFlops) * 100 : 0;

  return (
    <div
      style={{
        background: T.panel,
        border: `1px solid ${T.rule}`,
        borderRadius: 10,
        padding: 18,
      }}
    >
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
          color: T.bad,
          border: `1px solid ${T.bad}55`,
          background: `${T.bad}14`,
        }}
      >
        Question 6 · rematerialization
      </div>

      <div style={{ fontFamily: cond, fontSize: 19, fontWeight: 700, color: T.ink, marginBottom: 6 }}>
        Only 7 matmul outputs saved
      </div>
      <p style={{ fontFamily: sans, fontSize: 13.5, lineHeight: 1.55, color: T.soft, marginBottom: 14 }}>
        Say we only save the output of the 7 main matmuls (Q, K, V, O + the three FFW
        matrices) from the forward pass — not every intermediate activation. Everything
        else must be <b style={{ color: T.ink }}>rematerialized</b> in the backward pass.
        To get <b style={{ color: T.ink }}>∂L/∂W_O</b>, we need <code>softmax(QKᵀ)·V</code>{" "}
        back — but it wasn't saved, only Q, K, V were. So we recompute both attention
        matmuls from the saved Q, K, V.
      </p>

      <div style={{ display: "grid", gap: 6, marginBottom: 16 }}>
        <LegendRow category="saved" />
        <LegendRow category="remat-heavy" />
        <LegendRow category="remat-cheap" />
        <LegendRow category="free" />
      </div>

      <div style={{ display: "grid", gap: 8, marginBottom: 6 }}>
        {rematSteps.map((s) => (
          <DerivedFormula
            key={s.id}
            label={`extra flops · ${s.title.toLowerCase()}`}
            expr={s.rematExpr}
            dims={dims}
            resultColor={T.bad}
          />
        ))}
      </div>
      <p style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginBottom: 16 }}>
        ↳ forward-only cost (coef 2, not the usual 6) — we need this value as an input,
        not something to differentiate through again here.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          marginBottom: 14,
          paddingBottom: 14,
          borderBottom: `1px solid ${T.rule}`,
        }}
      >
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
            total extra flops
          </div>
          <div style={{ fontFamily: mono, fontSize: 17, fontWeight: 600, color: T.bad }}>{fmtNum(extra)}</div>
        </div>
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
            vs. normal layer training flops
          </div>
          <div style={{ fontFamily: mono, fontSize: 17, fontWeight: 600, color: T.ink }}>
            +{overheadPct.toFixed(1)}%
          </div>
        </div>
      </div>

      <p style={{ fontFamily: sans, fontSize: 12, color: T.dim, lineHeight: 1.55 }}>
        The <span style={{ color: T.wire }}>amber</span> steps (norm, reshape, softmax,
        GELU gate) also get recomputed, but they're elementwise — asymptotically ~0 FLOPs
        next to the quadratic attention recompute above, so they don't move this number.
        The <span style={{ color: T.soft }}>grey</span> steps (input, residual adds) were
        never lost — nothing to redo there.
      </p>
    </div>
  );
}
