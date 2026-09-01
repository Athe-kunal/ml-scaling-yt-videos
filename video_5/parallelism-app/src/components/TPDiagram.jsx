import { T, mono } from "../lib/theme";
import { STATE_STYLE, CommArrow, CommBar } from "./DiagramPrimitives";

// Renderer for TP's "matmul" diagram kind: two GPU panels, each a vertical
// stack of stage boxes (activation/weight/grad, plain/partial/active),
// connected top-to-bottom by arrows, with a shared comm bar when a step
// involves an AllReduce. Purely driven by the `diagram` object each tp.js
// step hands it — same contract as DeviceDiagram, different shape because
// TP's story is a data-flow through matmul stages, not a per-layer chain.

const ROLE_COLOR = {
  act: T.soft,
  weight: T.wire,
  grad: T.accent2,
};

const PANEL = [
  { x: 40, label: "GPU 0", sub: "Y=0" },
  { x: 480, label: "GPU 1", sub: "Y=1" },
];
const PANEL_W = 380;
const PANEL_Y = 46;
const STAGE_W = 300;
const STAGE_H = 42;
const STAGE_GAP = 26;
const STAGE_X_PAD = 40;

function StageBox({ x, y, w, h, stage }) {
  const st = STATE_STYLE[stage.state] || STATE_STYLE.solid;
  const roleColor = ROLE_COLOR[stage.role] || T.soft;
  return (
    <g opacity={st.opacity} style={{ transition: "all .35s cubic-bezier(.2,.7,.3,1)" }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={7}
        fill={st.fill}
        stroke={st.stroke}
        strokeWidth={stage.state === "active" ? 2 : 1.2}
        strokeDasharray={st.dash}
      />
      <rect x={x} y={y} width={4} height={h} rx={2} fill={stage.state === "hidden" ? "transparent" : roleColor} opacity={0.85} />
      <text x={x + w / 2} y={y + h / 2 - 3} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={st.text}>
        {stage.label}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={st.text} opacity={0.75}>
        {stage.shape}
      </text>
    </g>
  );
}

export default function TPDiagram({ diagram }) {
  const { devices, comm } = diagram;
  const maxStages = Math.max(...devices.map((d) => d.stages.length));
  const svgH = PANEL_Y + maxStages * (STAGE_H + STAGE_GAP) + (comm ? 46 : 18);

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "18px 14px 14px" }}>
      <svg viewBox={`0 0 900 ${svgH}`} style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="tp-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
        </defs>

        {comm && <CommArrow type={comm.type} x1={190} x2={710} y={20} />}

        {devices.map((dev, di) => {
          const p = PANEL[di];
          return (
            <g key={di}>
              <rect
                x={p.x}
                y={PANEL_Y}
                width={PANEL_W}
                height={maxStages * (STAGE_H + STAGE_GAP) - STAGE_GAP + 24}
                rx={10}
                fill="rgba(255,255,255,0.02)"
                stroke={T.rule}
                strokeWidth={1}
              />
              <text x={p.x + 12} y={PANEL_Y - 10} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
                {p.label.toUpperCase()} · {p.sub}
              </text>

              {dev.stages.map((stage, si) => {
                const x = p.x + STAGE_X_PAD;
                const y = PANEL_Y + 12 + si * (STAGE_H + STAGE_GAP);
                return (
                  <g key={si}>
                    <StageBox x={x} y={y} w={STAGE_W} h={STAGE_H} stage={stage} />
                    {si < dev.stages.length - 1 && (
                      <line
                        x1={x + STAGE_W / 2}
                        y1={y + STAGE_H}
                        x2={x + STAGE_W / 2}
                        y2={y + STAGE_H + STAGE_GAP}
                        stroke={T.dim}
                        strokeWidth={1.2}
                        opacity={0.7}
                        markerEnd="url(#tp-arrow)"
                      />
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      <CommBar comm={comm} />
    </div>
  );
}
