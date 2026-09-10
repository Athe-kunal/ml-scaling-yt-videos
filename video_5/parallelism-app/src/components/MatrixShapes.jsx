import { T, mono, sans } from "../lib/theme";
import { COMM_COLOR } from "./DiagramPrimitives";
import MathLabel from "./MathLabel";

// A row of "modern matrix" icons for one pseudocode line — the actual
// operand(s) and result of that line, drawn as labeled rectangles with
// their sharding made visible, rather than just named boxes. Driven by a
// step's `matrices` array (see tp.js's `mat`/`op` helpers):
//   mat(id, rows, cols, { shardAxis, state, tone })
//     shardAxis: null (fully materialized) | "cols" | "rows" (split in two
//       bands along that axis — band 0, "this device", drawn solid; band 1
//       drawn as a hatched outline to mean "lives on the other device")
//     state: "solid" (default) | "active" (this line's highlighted result)
//       | "partial" (unreduced — drawn as two offset cards with a Σ badge)
//     tone: "act" | "weight" | "grad" — which named-tensor role's color
//   op(symbol, { comm })  — "×" / "=" / "→" between two matrices; a "→"
//     with a comm type ("allgather" | "reducescatter") is tinted and
//     colored to match that collective, echoing the flow diagram above it.

const TONE_COLOR = { act: T.accent, weight: T.wire, grad: T.accent2 };

const BOX_W = 96;
const BOX_H = 74;
// The operator glyph sits left-biased in its gap (OP_GLYPH_X from the gap's
// start) so the room it needs never collides with the row-label zone that
// belongs to the box right after it (that label lives in the gap's last
// ROW_LABEL_W px, reserved via PAD_LEFT below).
const OP_W = 64;
const OP_GLYPH_X = 18;
const PAD_TOP = 28;
const PAD_LEFT = 30;
const PAD_RIGHT = 14;
const PAD_BOTTOM = 30;

function hatchId(tone) {
  return `mx-hatch-${tone}`;
}

function MatrixBox({ x, y, spec }) {
  const { id, rows, cols, shardAxis, state = "solid", tone = "act" } = spec;
  const color = TONE_COLOR[tone] || T.soft;
  const active = state === "active";

  const colLabel = (
    <MathLabel x={x} y={y - 22} w={BOX_W} h={18} text={cols} color={T.dim} fontSize={12} />
  );
  const rowLabel = (
    <MathLabel x={x - PAD_LEFT + 4} y={y + BOX_H / 2 - 10} w={PAD_LEFT - 8} h={20} text={rows} color={T.dim} fontSize={12} />
  );
  const idLabel = (
    <MathLabel x={x} y={y + BOX_H + 4} w={BOX_W} h={16} text={id} color={T.dim} fontSize={10.5} />
  );

  if (state === "partial") {
    return (
      <g>
        <rect x={x + 7} y={y + 7} width={BOX_W} height={BOX_H} rx={9} fill={`${color}14`} stroke={color} strokeWidth={1.1} strokeDasharray="4 4" opacity={0.55} />
        <rect x={x} y={y} width={BOX_W} height={BOX_H} rx={9} fill={`${color}26`} stroke={color} strokeWidth={1.6} style={{ transition: "all .35s ease" }} />
        <circle cx={x + BOX_W - 13} cy={y + 13} r={10} fill={T.well} stroke={color} strokeWidth={1.3} />
        <text x={x + BOX_W - 13} y={y + 17} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={color}>
          Σ
        </text>
        {colLabel}
        {rowLabel}
        {idLabel}
      </g>
    );
  }

  if (shardAxis === "cols" || shardAxis === "rows") {
    const vertical = shardAxis === "cols";
    const bandW = vertical ? BOX_W / 2 : BOX_W;
    const bandH = vertical ? BOX_H : BOX_H / 2;
    const mineX = x;
    const mineY = y;
    const otherX = vertical ? x + bandW : x;
    const otherY = vertical ? y : y + bandH;
    return (
      <g style={{ transition: "all .35s ease" }}>
        <rect x={otherX} y={otherY} width={bandW} height={bandH} fill={`url(#${hatchId(tone)})`} stroke={T.rule} strokeWidth={1} strokeDasharray="3 4" opacity={0.6} />
        <rect
          x={mineX}
          y={mineY}
          width={bandW}
          height={bandH}
          fill={`${color}30`}
          stroke={color}
          strokeWidth={active ? 2.2 : 1.4}
          style={{ filter: active ? `drop-shadow(0 0 4px ${color}88)` : "none" }}
        />
        <rect x={x} y={y} width={BOX_W} height={BOX_H} rx={9} fill="none" stroke={T.rule} strokeWidth={1} />
        {colLabel}
        {rowLabel}
        {idLabel}
      </g>
    );
  }

  return (
    <g style={{ transition: "all .35s ease" }}>
      <rect
        x={x}
        y={y}
        width={BOX_W}
        height={BOX_H}
        rx={9}
        fill={`${color}${active ? "33" : "22"}`}
        stroke={color}
        strokeWidth={active ? 2.2 : 1.4}
        style={{ filter: active ? `drop-shadow(0 0 4px ${color}88)` : "none" }}
      />
      {colLabel}
      {rowLabel}
      {idLabel}
    </g>
  );
}

function widthOf(item) {
  return item.op ? OP_W : BOX_W;
}

// Extra gap reserved for a matrix's row-dimension label when it sits
// directly next to another matrix with no "×"/"="/"→" between them (e.g.
// the setup row's three side-by-side icons) — without it the label crowds
// the previous box's right edge.
const ADJACENT_GAP = 30;

export default function MatrixShapes({ items }) {
  if (!items || items.length === 0) return null;

  const placed = items.reduce((acc, item, i) => {
    const prev = acc[acc.length - 1];
    const prevItem = i > 0 ? items[i - 1] : null;
    const adjacent = prev && !prevItem.op && !item.op;
    const x = prev ? prev.x + prev.w + (adjacent ? ADJACENT_GAP : 0) : PAD_LEFT;
    return [...acc, { item, x, w: widthOf(item) }];
  }, []);

  const last = placed[placed.length - 1];
  const totalW = last.x + last.w + PAD_RIGHT;
  const totalH = PAD_TOP + BOX_H + PAD_BOTTOM;

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "14px 14px 10px" }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim, marginBottom: 4, textTransform: "uppercase" }}>
        Shapes &amp; sharding
      </div>
      <svg viewBox={`0 0 ${totalW} ${totalH}`} style={{ width: "100%", display: "block", maxWidth: totalW + 40 }}>
        <defs>
          {["act", "weight", "grad"].map((tone) => (
            <pattern key={tone} id={hatchId(tone)} width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
              <rect width="7" height="7" fill="transparent" />
              <line x1="0" y1="0" x2="0" y2="7" stroke={TONE_COLOR[tone]} strokeWidth="1" opacity="0.25" />
            </pattern>
          ))}
          <marker id="mx-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
          </marker>
        </defs>

        {placed.map(({ item, x }, i) => {
          if (item.op) {
            const cy = PAD_TOP + BOX_H / 2;
            // Left-biased within the gap: the box right after this operator
            // reserves its own row-label space at the gap's tail end (see
            // MatrixBox's rowLabel), so the glyph/arrow must stay clear of it.
            if (item.comm) {
              const c = COMM_COLOR[item.comm];
              return (
                <g key={i}>
                  <line x1={x + 4} y1={cy} x2={x + OP_GLYPH_X + 12} y2={cy} stroke={c} strokeWidth={2} markerEnd="url(#mx-arrow)" />
                </g>
              );
            }
            return (
              <text key={i} x={x + OP_GLYPH_X} y={cy + 6} textAnchor="middle" fontFamily={sans} fontSize="20" fontWeight="600" fill={T.dim}>
                {item.op}
              </text>
            );
          }
          return <MatrixBox key={i} x={x} y={PAD_TOP} spec={item} />;
        })}
      </svg>
      <div style={{ fontFamily: sans, fontSize: 10.5, color: T.dim, marginTop: 2 }}>
        solid = this device's shard · hatched = the other device's · stacked <span style={{ fontFamily: mono }}>Σ</span> = unreduced, not yet summed
      </div>
    </div>
  );
}
