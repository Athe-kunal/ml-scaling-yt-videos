import { T, mono } from "../lib/theme";

// Shared visual vocabulary for every diagram (DeviceDiagram, TPDiagram,
// EmbedDiagram): the hidden/ghost/solid/active box states, the comm-op
// color/title lookup, and the comm arrow + label bar drawn under an SVG
// whenever a step involves cross-device communication.

export const STATE_STYLE = {
  hidden: { stroke: T.rule, fill: "transparent", dash: "3 4", opacity: 0.22, text: T.rule },
  ghost: { stroke: T.dim, fill: "rgba(199,209,221,0.07)", dash: "4 4", opacity: 0.75, text: T.dim },
  solid: { stroke: T.soft, fill: T.panel, dash: "none", opacity: 1, text: T.ink },
  active: { stroke: T.accent, fill: "rgba(110,243,221,0.16)", dash: "none", opacity: 1, text: T.accent },
  partial: { stroke: T.wire, fill: "rgba(255,201,77,0.12)", dash: "none", opacity: 1, text: T.wire },
};

export const COMM_COLOR = {
  allreduce: T.accent,
  reducescatter: T.wire,
  allgather: T.accent2,
  broadcast: T.accent2,
};
export const COMM_TITLE = {
  allreduce: "AllReduce",
  reducescatter: "ReduceScatter",
  allgather: "AllGather",
  broadcast: "Broadcast",
};

// A colored double-headed arrow spanning [x1,x2] at height y, with a title
// above it — the visual marker for "communication happens here".
export function CommArrow({ type, x1, x2, y = 20, label }) {
  const markerId = `dd-comm-arrow-${type}`;
  return (
    <g>
      <defs>
        <marker id={markerId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 z" fill={COMM_COLOR[type]} />
        </marker>
      </defs>
      <line
        x1={x1}
        y1={y}
        x2={x2}
        y2={y}
        stroke={COMM_COLOR[type]}
        strokeWidth={2}
        markerStart={`url(#${markerId})`}
        markerEnd={`url(#${markerId})`}
      />
      <text
        x={(x1 + x2) / 2}
        y={y - 8}
        textAnchor="middle"
        fontFamily={mono}
        fontSize="10"
        fontWeight="700"
        fill={COMM_COLOR[type]}
        letterSpacing="0.06em"
      >
        {label || COMM_TITLE[type]}
      </text>
    </g>
  );
}

// The text strip under the SVG that spells out the comm op, e.g.
// "G_i = AllReduce_X(G_i^{(0)}, G_i^{(1)})".
export function CommBar({ comm }) {
  if (!comm) return null;
  return (
    <div
      style={{
        marginTop: 10,
        fontFamily: mono,
        fontSize: 11,
        color: COMM_COLOR[comm.type],
        textAlign: "center",
        wordBreak: "break-word",
      }}
    >
      {comm.label}
    </div>
  );
}
