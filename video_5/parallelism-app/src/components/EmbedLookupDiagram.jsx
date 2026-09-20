import { T, mono, sans } from "../lib/theme";
import { CommArrow, CommBar } from "./DiagramPrimitives";

// Renderer for the vocab-parallel-embedding topic's worked numeric example
// (kind: "vpe") — a concrete 6-row embedding table split by row-range
// across 2 GPUs, looked up at 4 token indices. Unlike FSDPDiagram's
// abstract named-tensor boxes, everything here is real numbers, because
// the book's own derivation of this one is a numeric example rather than
// shape-level pseudocode.
//
// diagram = {
//   embeddings: number[6],           // the table, row-indexed 0..5
//   ranges: null | [{ gpu: 1|2, lo, hi }, ...],  // row ownership, once sharded
//   index: number[4],                // X — the token ids being looked up
//   rows: [{ label, cells: [{ value, tone, dim }] }],  // per-token derived rows
//   comm: { type, label } | null,
// }
// cell.tone: null (neutral) | "gpu1" | "gpu2". cell.dim: true = hatched/
// faded — "zeroed out by a mask" or "about to be discarded", not real data.

const GPU_COLOR = { gpu1: T.accent, gpu2: T.wire };

const EMB_CELL = 52;
const EMB_GAP = 8;
const TOK_CELL = 52;
const TOK_GAP = 8;
const LABEL_W = 132;
const LEFT_PAD = 16;

function hatchDefs() {
  return (
    <pattern id="vpe-hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="transparent" />
      <line x1="0" y1="0" x2="0" y2="7" stroke={T.dim} strokeWidth="1" opacity="0.35" />
    </pattern>
  );
}

function Cell({ x, y, w, h, value, tone, dim }) {
  const color = tone ? GPU_COLOR[tone] : T.soft;
  if (dim) {
    return (
      <g style={{ transition: "all .3s ease" }}>
        <rect x={x} y={y} width={w} height={h} rx={7} fill="url(#vpe-hatch)" stroke={T.rule} strokeWidth={1} strokeDasharray="3 4" opacity={0.75} />
        <text x={x + w / 2} y={y + h / 2 + 5} textAnchor="middle" fontFamily={mono} fontSize="13" fill={T.dim}>
          {value}
        </text>
      </g>
    );
  }
  return (
    <g style={{ transition: "all .3s ease" }}>
      <rect x={x} y={y} width={w} height={h} rx={7} fill={tone ? `${color}26` : T.panel} stroke={color} strokeWidth={tone ? 1.6 : 1.2} />
      <text x={x + w / 2} y={y + h / 2 + 5} textAnchor="middle" fontFamily={mono} fontSize="13" fontWeight="600" fill={tone ? color : T.ink}>
        {value}
      </text>
    </g>
  );
}

export default function EmbedLookupDiagram({ diagram }) {
  const { embeddings, ranges, index, rows, comm } = diagram;

  const embRowW = embeddings.length * (EMB_CELL + EMB_GAP) - EMB_GAP;
  const tokRowW = LABEL_W + index.length * (TOK_CELL + TOK_GAP) - TOK_GAP;
  const totalW = LEFT_PAD + Math.max(embRowW, tokRowW) + 16;

  const embY = ranges ? 56 : 40;
  const tokStartY = embY + 40 + 46;
  const rowH = 34;
  const rowGap = 14;
  const totalH = tokStartY + (rows.length + 1) * (rowH + rowGap) + (comm ? 30 : 6);

  function ownerOf(rowIdx) {
    if (!ranges) return null;
    const r = ranges.find((rr) => rowIdx >= rr.lo && rowIdx <= rr.hi);
    return r ? `gpu${r.gpu}` : null;
  }

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "18px 14px 14px" }}>
      <svg viewBox={`0 0 ${totalW} ${totalH}`} style={{ width: "100%", display: "block", maxWidth: totalW + 40 }}>
        <defs>{hatchDefs()}</defs>

        <text x={LEFT_PAD} y={embY - 30} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          EMBEDDING TABLE (vocab = {embeddings.length})
        </text>

        {ranges &&
          ranges.map((r) => {
            const x1 = LEFT_PAD + r.lo * (EMB_CELL + EMB_GAP);
            const x2 = LEFT_PAD + r.hi * (EMB_CELL + EMB_GAP) + EMB_CELL;
            const color = GPU_COLOR[`gpu${r.gpu}`];
            return (
              <g key={r.gpu}>
                <line x1={x1} y1={embY - 10} x2={x2} y2={embY - 10} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
                <text x={(x1 + x2) / 2} y={embY - 16} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={color} letterSpacing="0.06em">
                  GPU {r.gpu}
                </text>
              </g>
            );
          })}

        {embeddings.map((v, i) => (
          <g key={i}>
            <Cell x={LEFT_PAD + i * (EMB_CELL + EMB_GAP)} y={embY} w={EMB_CELL} h={40} value={v} tone={ownerOf(i)} dim={false} />
            <text x={LEFT_PAD + i * (EMB_CELL + EMB_GAP) + EMB_CELL / 2} y={embY + 40 + 14} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.dim}>
              row {i}
            </text>
          </g>
        ))}

        <text x={LEFT_PAD} y={tokStartY - 12} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          TOKEN INDICES X = [{index.join(", ")}]
        </text>

        <text x={LEFT_PAD} y={tokStartY + rowH / 2 + 4} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.soft}>
          X
        </text>
        {index.map((v, i) => (
          <Cell key={i} x={LABEL_W + i * (TOK_CELL + TOK_GAP)} y={tokStartY} w={TOK_CELL} h={rowH} value={v} tone={null} dim={false} />
        ))}

        {rows.map((row, ri) => {
          const y = tokStartY + (ri + 1) * (rowH + rowGap);
          return (
            <g key={ri}>
              <text x={LEFT_PAD} y={y + rowH / 2 + 4} fontFamily={mono} fontSize="11.5" fontWeight="600" fill={T.soft}>
                {row.label}
              </text>
              {row.cells.map((c, ci) => (
                <Cell key={ci} x={LABEL_W + ci * (TOK_CELL + TOK_GAP)} y={y} w={TOK_CELL} h={rowH} value={c.value} tone={c.tone} dim={c.dim} />
              ))}
            </g>
          );
        })}

        {comm && (
          <CommArrow
            type={comm.type}
            x1={LABEL_W}
            x2={LABEL_W + index.length * (TOK_CELL + TOK_GAP) - TOK_GAP}
            y={tokStartY + (rows.length + 1) * (rowH + rowGap) - rowGap / 2 + 6}
          />
        )}
      </svg>

      <CommBar comm={comm} />
      <div style={{ fontFamily: sans, fontSize: 10.5, color: T.dim, marginTop: 2 }}>
        <span style={{ color: T.accent }}>teal</span> = GPU1's row range · <span style={{ color: T.wire }}>gold</span> = GPU2's · hatched = zeroed out /
        discarded
      </div>
    </div>
  );
}
