import { T, mono } from "../lib/theme";
import { STATE_STYLE, COMM_COLOR, COMM_TITLE, CommBar } from "./DiagramPrimitives";
import MathLabel from "./MathLabel";

// Generic renderer for DP/ZeRO-1/2/3's per-step diagram descriptor: two
// device panels, each holding the book's 2-matmul MLP weights — Win and
// Wout — plus a gradient row and an optimizer-state row, and an optional
// cross-device communication arrow. Nothing here is topic-specific; the
// `diagram` object each topic's steps.js hands it drives everything.

const PANEL = [
  { x: 60, mbX: 0 },
  { x: 500, mbX: 460 },
];
const PANEL_W = 370;
const PANEL_H = 190;
const PANEL_Y = 40;
const LAYER_KEYS = ["in", "out"];
const LAYER_OFFSETS = [20, 195];
const LAYER_W = 150;
const LAYER_H = 40;
const LAYER_Y = 96;
const GRAD_Y = 154;
const OPTIM_Y = 186;
const OPTIM_W = 90;
const OPTIM_H = 26;

function Box({ x, y, w, h, label, state, small }) {
  const st = STATE_STYLE[state] || STATE_STYLE.hidden;
  return (
    <g opacity={st.opacity} style={{ transition: "all .35s cubic-bezier(.2,.7,.3,1)" }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        fill={st.fill}
        stroke={st.stroke}
        strokeWidth={state === "active" ? 2 : 1.2}
        strokeDasharray={st.dash}
      />
      <MathLabel x={x} y={y} w={w} h={h} text={label} color={st.text} fontSize={small ? 10.5 : 12} />
    </g>
  );
}

export default function DeviceDiagram({ diagram }) {
  const { devices, comm } = diagram;

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "18px 14px 14px" }}>
      <svg viewBox="0 0 900 250" style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="dd-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
          <marker id="dd-comm-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={comm ? COMM_COLOR[comm.type] : T.dim} />
          </marker>
        </defs>

        {devices.map((dev, di) => {
          const p = PANEL[di];
          return (
            <g key={di}>
              <rect
                x={p.x}
                y={PANEL_Y}
                width={PANEL_W}
                height={PANEL_H}
                rx={10}
                fill="rgba(255,255,255,0.02)"
                stroke={T.rule}
                strokeWidth={1}
              />
              <text x={p.x + 12} y={PANEL_Y - 10} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
                DEVICE {di} · X={di}
              </text>

              {dev.mb && (
                <>
                  <rect x={p.mbX} y={100} width={34} height={24} rx={4} fill={T.panel} stroke={T.rule} strokeWidth={1} />
                  <text x={p.mbX + 17} y={116} textAnchor="middle" fontFamily={mono} fontSize="8.5" fill={T.soft}>
                    {dev.mb}
                  </text>
                  <line
                    x1={p.mbX + 34}
                    y1={112}
                    x2={p.x}
                    y2={112}
                    stroke={T.dim}
                    strokeWidth={1.2}
                    markerEnd="url(#dd-arrow)"
                    opacity={0.7}
                  />
                </>
              )}

              {dev.input && (
                <text x={p.x + 12} y={PANEL_Y + 18} fontFamily={mono} fontSize="9.5" fill={T.soft}>
                  {dev.input}
                </text>
              )}

              {LAYER_OFFSETS.map((off, li) => (
                <Box
                  key={`w${li}`}
                  x={p.x + off}
                  y={LAYER_Y}
                  w={LAYER_W}
                  h={LAYER_H}
                  label={dev.layers[li].label}
                  state={dev.layers[li].state}
                />
              ))}

              {LAYER_OFFSETS.slice(0, -1).map((off, li) => {
                const a = dev.layers[li];
                const b = dev.layers[li + 1];
                const on = a.state !== "hidden" && b.state !== "hidden";
                const x1 = p.x + off + LAYER_W;
                const x2 = p.x + LAYER_OFFSETS[li + 1];
                return (
                  <line
                    key={`e${li}`}
                    x1={x1}
                    y1={LAYER_Y + LAYER_H / 2}
                    x2={x2}
                    y2={LAYER_Y + LAYER_H / 2}
                    stroke={on ? T.dim : T.rule}
                    strokeWidth={1}
                    opacity={on ? 0.8 : 0.2}
                    markerEnd={on ? "url(#dd-arrow)" : undefined}
                  />
                );
              })}

              {LAYER_OFFSETS.map((off, li) => {
                const g = dev.grads[li];
                if (g.state === "hidden") return null;
                return (
                  <MathLabel
                    key={`g${li}`}
                    x={p.x + off}
                    y={GRAD_Y - 11}
                    w={LAYER_W}
                    h={16}
                    text={g.label}
                    color={g.state === "active" ? T.accent : T.wire}
                    fontSize={10.5}
                  />
                );
              })}

              {LAYER_OFFSETS.map((off, li) => (
                <Box
                  key={`o${li}`}
                  x={p.x + off + (LAYER_W - OPTIM_W) / 2}
                  y={OPTIM_Y}
                  w={OPTIM_W}
                  h={OPTIM_H}
                  label={`m_{${LAYER_KEYS[li]}},v_{${LAYER_KEYS[li]}}`}
                  state={dev.optim[li].state}
                  small
                />
              ))}
            </g>
          );
        })}

        {comm && (
          <g>
            <line
              x1={220}
              y1={20}
              x2={680}
              y2={20}
              stroke={COMM_COLOR[comm.type]}
              strokeWidth={2}
              markerStart="url(#dd-comm-arrow)"
              markerEnd="url(#dd-comm-arrow)"
            />
            <text
              x={450}
              y={12}
              textAnchor="middle"
              fontFamily={mono}
              fontSize="10"
              fontWeight="700"
              fill={COMM_COLOR[comm.type]}
              letterSpacing="0.06em"
            >
              {COMM_TITLE[comm.type]}
            </text>
          </g>
        )}
      </svg>

      <CommBar comm={comm} />
    </div>
  );
}
