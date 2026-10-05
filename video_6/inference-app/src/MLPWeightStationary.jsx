import { useState } from "react";
import { T, mono, sans } from "./theme";
import { withInlineMath } from "./latex";
import { STEPS_1D, STEPS_2D, HEADER_1D, HEADER_2D } from "./mlpSteps";
import MLPFlowDiagram from "./MLPFlowDiagram";
import MatrixShapes from "./MatrixShapes";

const panel = { background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 18 };
const navBtn = (disabled) => ({
  fontFamily: mono, fontSize: 15, width: 34, height: 30, borderRadius: 6, flexShrink: 0,
  border: `1px solid ${T.rule}`, background: disabled ? "transparent" : T.well,
  color: disabled ? T.rule : T.ink, cursor: disabled ? "default" : "pointer",
});
const Rich = ({ html, style }) => <div style={style} dangerouslySetInnerHTML={{ __html: withInlineMath(html) }} />;

function Stepper({ label, intro, steps }) {
  const [idx, setIdx] = useState(0);
  const last = steps.length - 1;
  const s = steps[idx];
  const isComm = !!s.diagram.comm;

  return (
    <section style={panel}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4, marginBottom: 4 }}>{label.toUpperCase()}</div>
        <Rich style={{ color: T.soft, lineHeight: 1.55, fontSize: 14, maxWidth: 820 }} html={intro} />
      </div>

      <div style={{ display: "grid", gap: 14 }}>
        <div>
          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.2, color: T.dim, textTransform: "uppercase", marginBottom: 8 }}>Tensor flow</div>
          <MLPFlowDiagram nodes={s.diagram.nodes} comm={s.diagram.comm} />
        </div>

        <div>
          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.2, color: T.dim, textTransform: "uppercase", marginBottom: 8 }}>Shapes &amp; sharding at this step</div>
          <MatrixShapes row={s.matrices} accent={isComm ? T.wire : T.accent} />
        </div>

        <div style={{ borderTop: `1px solid ${T.rule}`, paddingTop: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} aria-label={`Previous ${label} step`} style={navBtn(idx === 0)}>←</button>
            <div style={{ display: "flex", gap: 4, flex: 1, justifyContent: "center", flexWrap: "wrap" }}>
              {steps.map((st, i) => (
                <button
                  key={st.id} onClick={() => setIdx(i)} aria-label={`${label} step ${i + 1}: ${st.title}`}
                  style={{ width: i === idx ? 20 : 7, height: 7, borderRadius: 4, border: "none", padding: 0, cursor: "pointer",
                    background: i === idx ? T.accent : i < idx ? T.dim : T.rule, transition: "all .25s cubic-bezier(.2,.7,.3,1)" }}
                />
              ))}
            </div>
            <button onClick={() => setIdx(idx + 1)} disabled={idx === last} aria-label={`Next ${label} step`} style={navBtn(idx === last)}>→</button>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", padding: "3px 9px", borderRadius: 5, color: T.accent, border: `1px solid ${T.accent}55`, background: `${T.accent}14` }}>
              step {idx + 1} / {steps.length}
            </span>
            {isComm && (
              <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", padding: "3px 9px", borderRadius: 5, color: T.wire, border: `1px solid ${T.wire}55`, background: `${T.wire}14` }}>
                collective
              </span>
            )}
          </div>

          <div style={{ fontSize: 17, fontWeight: 700, color: T.ink, marginBottom: 8 }}>{s.title}</div>

          <Rich
            style={{ fontSize: 13.5, lineHeight: 1.8, color: T.accent2, background: T.well, border: `1px solid ${T.rule}`, borderRadius: 8, padding: "9px 11px", marginBottom: 10, overflowX: "auto" }}
            html={s.notation}
          />

          {s.comms && (
            <Rich
              style={{ fontSize: 12.5, lineHeight: 1.6, color: T.wire, background: `${T.wire}0f`, border: `1px solid ${T.wire}40`, borderRadius: 8, padding: "7px 10px", marginBottom: 10 }}
              html={`<span style="font-family:${mono};font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;opacity:.85;margin-right:8px">Comms</span>${s.comms}`}
            />
          )}

          <Rich style={{ fontSize: 13.5, lineHeight: 1.6, color: T.soft }} html={s.body} />
        </div>
      </div>
    </section>
  );
}

export default function MLPWeightStationary() {
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: sans, padding: "28px clamp(16px, 3vw, 40px) 60px" }}>
      <header style={{ maxWidth: 1100, margin: "0 auto 22px" }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SCALING BOOK · INFERENCE</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>MLP weight-stationary: 1D vs 2D</h1>
        <Rich
          style={{ color: T.soft, maxWidth: 900, lineHeight: 1.55, fontSize: 15 }}
          html="Same feedforward block, two ways to lay the weights across the chip mesh. Step through each independently and watch what moves, when, and how much."
        />
      </header>

      <div style={{ maxWidth: 1100, margin: "0 auto", display: "grid", gap: 22 }}>
        <Stepper label={HEADER_1D.title} intro={HEADER_1D.intro} steps={STEPS_1D} />
        <Stepper label={HEADER_2D.title} intro={HEADER_2D.intro} steps={STEPS_2D} />
      </div>
    </div>
  );
}
