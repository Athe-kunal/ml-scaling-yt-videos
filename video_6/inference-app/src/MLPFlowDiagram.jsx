import { T, mono } from "./theme";
import { withInlineMath } from "./latex";
import { codeHtml } from "./notation";

// One row of named boxes (Act -> Win -> Tmp -> Wout -> Out -> Final) for the
// MLP weight-stationary steppers. Each box cycles through a small state
// machine as you step: hidden (not introduced yet) -> ghost (exists, but
// sharded/not yet touched) -> active (this step's highlighted result) ->
// solid (settled, done) -> partial (unreduced, needs a collective before
// it's the true value). A comm arrow + label bar is drawn above whichever
// box just changed, when this step is a collective.

const STATE_STYLE = {
  hidden: { stroke: T.rule, fill: "transparent", dash: "3 4", opacity: 0.22, text: T.rule },
  ghost: { stroke: T.dim, fill: "rgba(199,209,221,0.07)", dash: "4 4", opacity: 0.75, text: T.dim },
  solid: { stroke: T.soft, fill: T.panel, dash: "none", opacity: 1, text: T.ink },
  active: { stroke: T.accent, fill: "rgba(110,243,221,0.16)", dash: "none", opacity: 1, text: T.accent },
  partial: { stroke: T.wire, fill: "rgba(255,201,77,0.12)", dash: "none", opacity: 1, text: T.wire },
};

const COMM_COLOR = { allgather: T.accent2, reducescatter: T.wire };
const COMM_TITLE = { allgather: "AllGather", reducescatter: "ReduceScatter" };

const IDS = ["Act", "Win", "Tmp", "Wout", "Out", "Final"];
const BOX_W = 138;
const BOX_H = 46;
const GAP = 14;
const ROW_Y = 44;
const X0 = 10;

const XS = IDS.map((_, i) => X0 + i * (BOX_W + GAP));

export default function MLPFlowDiagram({ nodes, comm }) {
  const commIdx = comm ? IDS.indexOf(comm.targetId) : -1;
  const totalW = XS[XS.length - 1] + BOX_W + X0;

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "16px 14px 14px" }}>
      <svg viewBox={`0 0 ${totalW} 150`} style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="mlp-flow-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
          {comm && (
            <marker id="mlp-comm-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 z" fill={COMM_COLOR[comm.type]} />
            </marker>
          )}
        </defs>

        {IDS.map((id, i) => {
          const n = nodes[id] || { label: "", state: "hidden" };
          const st = STATE_STYLE[n.state] || STATE_STYLE.hidden;
          const x = XS[i];
          return (
            <g key={id}>
              <rect
                x={x} y={ROW_Y} width={BOX_W} height={BOX_H} rx={8}
                fill={st.fill} stroke={st.stroke} strokeWidth={n.state === "active" || n.state === "partial" ? 2 : 1.2}
                strokeDasharray={st.dash} opacity={st.opacity}
                style={{ transition: "all .3s ease" }}
              />
              {n.state === "partial" && (
                <circle cx={x + BOX_W - 13} cy={ROW_Y + 12} r={9} fill={T.well} stroke={T.wire} strokeWidth={1.2} />
              )}
              {n.state === "partial" && (
                <text x={x + BOX_W - 13} y={ROW_Y + 16} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.wire}>
                  Σ
                </text>
              )}
              <foreignObject x={x} y={ROW_Y} width={BOX_W} height={BOX_H}>
                <div
                  style={{
                    width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
                    fontFamily: mono, fontSize: 11, color: st.text, textAlign: "center", padding: "0 6px", lineHeight: 1.25,
                  }}
                  dangerouslySetInnerHTML={{ __html: codeHtml(n.label || id) }}
                />
              </foreignObject>
              {i < IDS.length - 1 && (
                <line
                  x1={x + BOX_W} y1={ROW_Y + BOX_H / 2} x2={XS[i + 1]} y2={ROW_Y + BOX_H / 2}
                  stroke={T.dim} strokeWidth={1} opacity={0.4} markerEnd="url(#mlp-flow-arrow)"
                />
              )}
            </g>
          );
        })}

        {comm && commIdx >= 0 && (
          <g>
            <line
              x1={XS[commIdx]} y1={16} x2={XS[commIdx] + BOX_W} y2={16}
              stroke={COMM_COLOR[comm.type]} strokeWidth={2}
              markerStart="url(#mlp-comm-arrow)" markerEnd="url(#mlp-comm-arrow)"
            />
            <text
              x={XS[commIdx] + BOX_W / 2} y={10} textAnchor="middle"
              fontFamily={mono} fontSize="9.5" fontWeight="700" letterSpacing="0.08em" fill={COMM_COLOR[comm.type]}
            >
              {COMM_TITLE[comm.type]}
            </text>
          </g>
        )}
      </svg>

      {comm && (
        <div
          style={{ marginTop: 6, fontSize: 12.5, color: COMM_COLOR[comm.type], textAlign: "center", overflowX: "auto" }}
          dangerouslySetInnerHTML={{ __html: withInlineMath(comm.label) }}
        />
      )}
    </div>
  );
}
