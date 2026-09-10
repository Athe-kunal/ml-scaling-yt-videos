import { T, mono } from "../lib/theme";
import { STATE_STYLE, CommArrow, CommBar } from "./DiagramPrimitives";
import MathLabel from "./MathLabel";

// Renderer for the book's per-line pseudocode walkthroughs (FSDP and
// Tensor Parallelism, kinds "fsdp" and "tp"): one row of forward tensors
// (In -> Win -> Tmp -> Wout -> Out -> Loss) and one row of backward
// tensors (dOut -> dWout -> dTmp -> dWin -> dIn) — the same named-tensor
// shape both topics trace through, just with different boxes cycling
// between gathered/sharded/partial states (FSDP moves the weights, TP
// moves the activations). Win/Wout are shared boxes between the two rows
// — the same tensor gets touched in both the forward and backward pass,
// and the diagram should show that as the same box lighting up twice,
// not two different ones.

const ROW1_IDS = ["In", "Win", "Tmp", "Wout", "Out", "Loss"];
const ROW2_IDS = ["dOut", "dWout", "dTmp", "dWin", "dIn"];
const BOX_W = 128;
const BOX_H = 44;
const GAP = 18;
const ROW1_Y = 66;
const ROW2_Y = 186;
const ROW1_X0 = 10;
const ROW2_X0 = 86;

function positions(x0, count) {
  return Array.from({ length: count }, (_, i) => x0 + i * (BOX_W + GAP));
}
const ROW1_X = positions(ROW1_X0, ROW1_IDS.length);
const ROW2_X = positions(ROW2_X0, ROW2_IDS.length);

function boxRect(id) {
  let idx = ROW1_IDS.indexOf(id);
  if (idx >= 0) return { x: ROW1_X[idx], y: ROW1_Y, w: BOX_W, h: BOX_H };
  idx = ROW2_IDS.indexOf(id);
  if (idx >= 0) return { x: ROW2_X[idx], y: ROW2_Y, w: BOX_W, h: BOX_H };
  return null;
}

function Row({ ids, xs, y, nodes }) {
  return (
    <>
      {ids.map((id, i) => {
        const n = nodes[id] || { label: id, state: "hidden" };
        const st = STATE_STYLE[n.state] || STATE_STYLE.hidden;
        const x = xs[i];
        return (
          <g key={id}>
            <rect
              x={x}
              y={y}
              width={BOX_W}
              height={BOX_H}
              rx={7}
              fill={st.fill}
              stroke={st.stroke}
              strokeWidth={n.state === "active" ? 2 : 1.2}
              strokeDasharray={st.dash}
              opacity={st.opacity}
              style={{ transition: "all .3s ease" }}
            />
            <MathLabel x={x} y={y} w={BOX_W} h={BOX_H} text={n.label} color={st.text} fontSize={12} />
            {i < ids.length - 1 && (
              <line
                x1={x + BOX_W}
                y1={y + BOX_H / 2}
                x2={xs[i + 1]}
                y2={y + BOX_H / 2}
                stroke={T.dim}
                strokeWidth={1}
                opacity={0.45}
                markerEnd="url(#fsdp-arrow)"
              />
            )}
          </g>
        );
      })}
    </>
  );
}

export default function FSDPDiagram({ diagram }) {
  const { forward, backward, comm } = diagram;
  const fwdNodes = Object.fromEntries(forward.map((n) => [n.id, n]));
  const bwdNodes = Object.fromEntries((backward || []).map((n) => [n.id, n]));
  const commRect = comm && comm.targetId ? boxRect(comm.targetId) : null;

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "18px 14px 14px" }}>
      <svg viewBox="0 0 900 246" style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="fsdp-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
        </defs>

        <text x={ROW1_X0} y={ROW1_Y - 14} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          FORWARD
        </text>
        <Row ids={ROW1_IDS} xs={ROW1_X} y={ROW1_Y} nodes={fwdNodes} />

        <text x={ROW2_X0} y={ROW2_Y - 14} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          BACKWARD
        </text>
        <Row ids={ROW2_IDS} xs={ROW2_X} y={ROW2_Y} nodes={bwdNodes} />

        {comm && commRect && <CommArrow type={comm.type} x1={commRect.x} x2={commRect.x + commRect.w} y={commRect.y - 10} />}
      </svg>

      <CommBar comm={comm} />
    </div>
  );
}
