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
const GAP = 10;
const ROW1_Y = 66;
const ROW2_Y = 186;
// Columns start past a left margin reserved for the rotated row labels
// below, so a comm arrow over the leftmost aligned column (In/dIn) never
// collides with the "FORWARD"/"BACKWARD" text.
const ROW1_X0 = 46;
const LABEL_X = 20;

function positions(x0, count) {
  return Array.from({ length: count }, (_, i) => x0 + i * (BOX_W + GAP));
}
const ROW1_X = positions(ROW1_X0, ROW1_IDS.length);

// The backward row is column-aligned under its forward counterpart —
// dOut sits under Out, dWout under Wout, etc. — rather than left-aligned
// in isolation, since that's what actually makes it read as "the same
// pipeline, run in reverse" instead of a second, unrelated sequence.
const BACKWARD_ALIGN = { dOut: "Out", dWout: "Wout", dTmp: "Tmp", dWin: "Win", dIn: "In" };
const ROW2_X = ROW2_IDS.map((id) => ROW1_X[ROW1_IDS.indexOf(BACKWARD_ALIGN[id])]);

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
            {i < ids.length - 1 &&
              (() => {
                // Forward flows left-to-right (increasing x); the
                // column-aligned backward row flows right-to-left
                // (decreasing x) since it's walking the same pipeline in
                // reverse — draw the connecting arrow in whichever
                // direction the columns actually go.
                const xNext = xs[i + 1];
                const forwardFlow = xNext > x;
                const x1 = forwardFlow ? x + BOX_W : x;
                const x2 = forwardFlow ? xNext : xNext + BOX_W;
                return (
                  <line
                    x1={x1}
                    y1={y + BOX_H / 2}
                    x2={x2}
                    y2={y + BOX_H / 2}
                    stroke={T.dim}
                    strokeWidth={1}
                    opacity={0.45}
                    markerEnd="url(#fsdp-arrow)"
                  />
                );
              })()}
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

        <text
          x={LABEL_X}
          y={ROW1_Y + BOX_H / 2}
          transform={`rotate(-90 ${LABEL_X} ${ROW1_Y + BOX_H / 2})`}
          textAnchor="middle"
          fontFamily={mono}
          fontSize="10"
          letterSpacing="0.12em"
          fill={T.dim}
        >
          FORWARD
        </text>
        <Row ids={ROW1_IDS} xs={ROW1_X} y={ROW1_Y} nodes={fwdNodes} />

        <text
          x={LABEL_X}
          y={ROW2_Y + BOX_H / 2}
          transform={`rotate(-90 ${LABEL_X} ${ROW2_Y + BOX_H / 2})`}
          textAnchor="middle"
          fontFamily={mono}
          fontSize="10"
          letterSpacing="0.12em"
          fill={T.dim}
        >
          BACKWARD
        </text>
        <Row ids={ROW2_IDS} xs={ROW2_X} y={ROW2_Y} nodes={bwdNodes} />

        {comm && commRect && <CommArrow type={comm.type} x1={commRect.x} x2={commRect.x + commRect.w} y={commRect.y - 10} />}
      </svg>

      <CommBar comm={comm} />
    </div>
  );
}
