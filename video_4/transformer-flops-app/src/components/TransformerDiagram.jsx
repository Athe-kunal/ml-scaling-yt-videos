import { STEPS } from "../lib/steps";
import { T, mono } from "../lib/theme";

const NODE_ORDER = STEPS.map((s) => s.node);

// Layout mirrors the reference scaling-book diagram: attention block on top,
// MLP block below, both flowing top-to-bottom with the residual stream
// running down the left edge.
const NODES = [
  { id: "x-input", type: "label", x: 320, y: 18, text: "X" },
  { id: "norm1", type: "box", x: 260, y: 34, w: 120, h: 28, label: "norm" },

  { id: "q-proj", type: "box", x: 130, y: 100, w: 100, h: 44, label: "W_Q · X", sub: "→ Q  [B,T,N,H]" },
  { id: "k-proj", type: "box", x: 270, y: 100, w: 100, h: 44, label: "W_K · X", sub: "→ K  [B,S,K,H]" },
  { id: "v-proj", type: "box", x: 410, y: 100, w: 100, h: 44, label: "W_V · X", sub: "→ V  [B,S,K,H]" },

  { id: "reshape-qkv", type: "box", x: 240, y: 172, w: 160, h: 26, label: "reshape", sub: "BTNH → BTKGH", dashed: true },

  { id: "attn-scores", type: "box", x: 220, y: 226, w: 200, h: 40, label: "Q · Kᵀ  + mask", sub: "→ [B,T,S,N,H]" },
  { id: "softmax", type: "box", x: 220, y: 284, w: 200, h: 26, label: "softmax", dashed: true },
  { id: "weighted-sum", type: "box", x: 220, y: 328, w: 200, h: 40, label: "softmax(QKᵀ) · V", sub: "→ [B,T,N,H]" },

  { id: "out-proj", type: "box", x: 240, y: 396, w: 160, h: 44, label: "reshape · W_O", sub: "→ [B,T,D]" },
  { id: "residual1", type: "circle", x: 320, y: 466, r: 13, label: "+" },

  { id: "norm2", type: "box", x: 260, y: 506, w: 120, h: 28, label: "norm" },

  { id: "in1-proj", type: "box", x: 210, y: 562, w: 110, h: 44, label: "W_in1 · X", sub: "→ [B,T,F]" },
  { id: "in2-proj", type: "box", x: 340, y: 562, w: 110, h: 44, label: "W_in2 · X", sub: "→ [B,T,F]" },

  { id: "gelu-gate", type: "circle", x: 320, y: 638, r: 13, label: "×", sub: "gelu(in1) · in2" },

  { id: "out-proj-mlp", type: "box", x: 260, y: 682, w: 120, h: 44, label: "W_out", sub: "→ [B,T,D]" },
  { id: "residual2", type: "circle", x: 320, y: 752, r: 13, label: "+" },
];

const NODE_BY_ID = Object.fromEntries(NODES.map((n) => [n.id, n]));

function anchor(node, side) {
  if (node.type === "label") {
    const { x, y } = node;
    if (side === "top") return { x, y: y - 12 };
    return { x, y: y + 6 };
  }
  if (node.type === "circle") {
    const { x, y, r } = node;
    if (side === "top") return { x, y: y - r };
    if (side === "bottom") return { x, y: y + r };
    if (side === "left") return { x: x - r, y };
    return { x: x + r, y };
  }
  const { x, y, w, h } = node;
  if (side === "top") return { x: x + w / 2, y };
  if (side === "bottom") return { x: x + w / 2, y: y + h };
  if (side === "left") return { x, y: y + h / 2 };
  return { x: x + w, y: y + h / 2 };
}

// residual bypass line runs down x=60, rejoining at each "+" circle
const EDGES = [
  { id: "e-x-norm1", from: ["x-input", "bottom"], to: ["norm1", "top"], after: "norm1" },
  { id: "e-norm1-q", from: ["norm1", "bottom"], to: ["q-proj", "top"], after: "q-proj" },
  { id: "e-norm1-k", from: ["norm1", "bottom"], to: ["k-proj", "top"], after: "k-proj" },
  { id: "e-norm1-v", from: ["norm1", "bottom"], to: ["v-proj", "top"], after: "v-proj" },
  { id: "e-q-reshape", from: ["q-proj", "bottom"], to: ["reshape-qkv", "top"], after: "reshape-qkv" },
  { id: "e-k-reshape", from: ["k-proj", "bottom"], to: ["reshape-qkv", "top"], after: "reshape-qkv" },
  { id: "e-v-reshape", from: ["v-proj", "bottom"], to: ["reshape-qkv", "top"], after: "reshape-qkv" },
  { id: "e-reshape-scores", from: ["reshape-qkv", "bottom"], to: ["attn-scores", "top"], after: "attn-scores" },
  { id: "e-scores-softmax", from: ["attn-scores", "bottom"], to: ["softmax", "top"], after: "softmax" },
  { id: "e-softmax-wsum", from: ["softmax", "bottom"], to: ["weighted-sum", "top"], after: "weighted-sum" },
  { id: "e-wsum-out", from: ["weighted-sum", "bottom"], to: ["out-proj", "top"], after: "out-proj" },
  { id: "e-out-res1", from: ["out-proj", "bottom"], to: ["residual1", "top"], after: "residual1" },
  { id: "e-res1-norm2", from: ["residual1", "bottom"], to: ["norm2", "top"], after: "norm2" },
  { id: "e-norm2-in1", from: ["norm2", "bottom"], to: ["in1-proj", "top"], after: "in1-proj" },
  { id: "e-norm2-in2", from: ["norm2", "bottom"], to: ["in2-proj", "top"], after: "in2-proj" },
  { id: "e-in1-gate", from: ["in1-proj", "bottom"], to: ["gelu-gate", "top"], after: "gelu-gate" },
  { id: "e-in2-gate", from: ["in2-proj", "bottom"], to: ["gelu-gate", "top"], after: "gelu-gate" },
  { id: "e-gate-out", from: ["gelu-gate", "bottom"], to: ["out-proj-mlp", "top"], after: "out-proj-mlp" },
  { id: "e-out-res2", from: ["out-proj-mlp", "bottom"], to: ["residual2", "top"], after: "residual2" },
];

const RESIDUAL_BYPASSES = [
  { id: "bypass1", y1: 12, y2: 466, after: "residual1" },
  { id: "bypass2", y1: 466, y2: 752, after: "residual2" },
];

function isRevealed(id, currentIndex) {
  const idx = NODE_ORDER.indexOf(id);
  return idx !== -1 && idx <= currentIndex;
}

export default function TransformerDiagram({ currentStepIndex, currentNode }) {
  return (
    <div
      style={{
        background: T.well,
        border: `1px solid ${T.rule}`,
        borderRadius: 10,
        padding: "18px 14px 10px",
      }}
    >
      <svg viewBox="0 0 620 790" style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
        </defs>

        {/* residual bypass lines along the left */}
        {RESIDUAL_BYPASSES.map((b) => {
          const on = isRevealed(b.after, currentStepIndex);
          return (
            <path
              key={b.id}
              d={`M 60 ${b.y1} L 60 ${b.y2 - 13} L ${320 - 13} ${b.y2}`}
              fill="none"
              stroke={on ? T.accent : T.rule}
              strokeWidth={on ? 1.6 : 1}
              strokeDasharray={on ? "none" : "4 4"}
              opacity={on ? 0.85 : 0.35}
              markerEnd={on ? "url(#arrow)" : undefined}
              style={{ transition: "all .35s ease" }}
            />
          );
        })}
        {/* tap points where the bypass leaves the main line */}
        {["norm1", "norm2"].map((id) => {
          const n = NODE_BY_ID[id];
          const on = isRevealed(id, currentStepIndex);
          return (
            <line
              key={`tap-${id}`}
              x1={60}
              y1={n.y - 8}
              x2={n.x}
              y2={n.y - 8}
              stroke={on ? T.rule : T.rule}
              strokeWidth="1"
              strokeDasharray="2 3"
              opacity={on ? 0.5 : 0}
            />
          );
        })}

        {EDGES.map((e) => {
          const on = isRevealed(e.after, currentStepIndex);
          const from = anchor(NODE_BY_ID[e.from[0]], e.from[1]);
          const to = anchor(NODE_BY_ID[e.to[0]], e.to[1]);
          return (
            <line
              key={e.id}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={on ? T.dim : T.rule}
              strokeWidth="1.3"
              strokeDasharray={on ? "none" : "3 4"}
              opacity={on ? 0.9 : 0.3}
              markerEnd={on ? "url(#arrow)" : undefined}
              style={{ transition: "all .35s ease" }}
            />
          );
        })}

        {NODES.map((n) => {
          const on = isRevealed(n.id, currentStepIndex);
          const active = n.id === currentNode;
          const stroke = active ? T.accent : on ? T.soft : T.rule;
          const strokeWidth = active ? 2.2 : 1.2;
          const fill = active
            ? "rgba(94,234,212,0.14)"
            : on
            ? T.panel
            : "transparent";
          const textColor = on ? T.ink : T.dim;
          const subColor = on ? T.soft : T.dim;

          if (n.type === "label") {
            return (
              <text
                key={n.id}
                x={n.x}
                y={n.y}
                textAnchor="middle"
                fontFamily={mono}
                fontSize="16"
                fontWeight="700"
                fill={on ? T.accent : T.dim}
                style={{ transition: "fill .35s ease" }}
              >
                {n.text}
              </text>
            );
          }

          if (n.type === "circle") {
            return (
              <g key={n.id} style={{ transition: "all .35s ease" }}>
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={n.r}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  strokeDasharray={on ? "none" : "3 3"}
                />
                <text
                  x={n.x}
                  y={n.y + 4}
                  textAnchor="middle"
                  fontFamily={mono}
                  fontSize="13"
                  fontWeight="700"
                  fill={textColor}
                >
                  {n.label}
                </text>
                {n.sub && (
                  <text
                    x={n.x}
                    y={n.y + n.r + 13}
                    textAnchor="middle"
                    fontFamily={mono}
                    fontSize="8.5"
                    fill={subColor}
                  >
                    {n.sub}
                  </text>
                )}
              </g>
            );
          }

          return (
            <g key={n.id} style={{ transition: "all .35s ease" }}>
              <rect
                x={n.x}
                y={n.y}
                width={n.w}
                height={n.h}
                rx="6"
                fill={fill}
                stroke={stroke}
                strokeWidth={strokeWidth}
                strokeDasharray={!on || n.dashed ? "4 4" : "none"}
              />
              <text
                x={n.x + n.w / 2}
                y={n.y + n.h / 2 + (n.sub ? -2 : 4)}
                textAnchor="middle"
                fontFamily={mono}
                fontSize="10.5"
                fontWeight="600"
                fill={textColor}
              >
                {n.label}
              </text>
              {n.sub && (
                <text
                  x={n.x + n.w / 2}
                  y={n.y + n.h / 2 + 11}
                  textAnchor="middle"
                  fontFamily={mono}
                  fontSize="8.5"
                  fill={subColor}
                >
                  {n.sub}
                </text>
              )}
            </g>
          );
        })}

        {/* phase labels */}
        <text x={12} y={110} fontFamily={mono} fontSize="10" fill={T.dim} letterSpacing="0.14em" transform="rotate(-90 12 110)">
          ATTENTION
        </text>
        <text x={12} y={620} fontFamily={mono} fontSize="10" fill={T.dim} letterSpacing="0.14em" transform="rotate(-90 12 620)">
          MLP
        </text>
      </svg>
    </div>
  );
}
