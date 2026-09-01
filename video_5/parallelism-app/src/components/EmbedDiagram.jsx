import { T, mono } from "../lib/theme";
import { CommArrow, CommBar } from "./DiagramPrimitives";

// Renderer for TP's "embed" diagram kind — the deck's concrete
// vocab-parallel embedding example: a 6-row table split by row range across
// 2 GPUs, a token-index lookup, per-GPU masked-gather results, and the
// AllReduce that combines them. Driven entirely by the `diagram` object
// tp.js hands it (table, indices, ranges, gpuResults, combined, comm).

const CELL_W = 46;
const CELL_H = 30;
const PANEL = [{ x: 40, label: "GPU 0", color: T.accent }, { x: 480, label: "GPU 1", color: T.accent2 }];

function Row({ x, y, values, colorFn, dim }) {
  return (
    <g>
      {values.map((v, i) => (
        <g key={i} opacity={dim && dim(i) ? 0.28 : 1}>
          <rect
            x={x + i * (CELL_W + 6)}
            y={y}
            width={CELL_W}
            height={CELL_H}
            rx={5}
            fill={T.panel}
            stroke={colorFn ? colorFn(i) : T.rule}
            strokeWidth={1.3}
          />
          <text
            x={x + i * (CELL_W + 6) + CELL_W / 2}
            y={y + CELL_H / 2 + 4}
            textAnchor="middle"
            fontFamily={mono}
            fontSize="12"
            fontWeight="600"
            fill={T.ink}
          >
            {v}
          </text>
        </g>
      ))}
    </g>
  );
}

export default function EmbedDiagram({ diagram }) {
  const { table, indices, ranges, gpuResults, combined, comm, stage } = diagram;

  const tableY = 30;
  const idxY = 96;
  const resultY = 150;
  const combinedY = 214;
  const svgH = combined ? 256 : gpuResults ? 200 : 140;

  return (
    <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "18px 14px 14px" }}>
      <svg viewBox={`0 0 900 ${svgH}`} style={{ width: "100%", display: "block" }}>
        {comm && <CommArrow type={comm.type} x1={190} x2={710} y={18} />}

        <text x={40} y={tableY - 8} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          EMBEDDING TABLE (vocab = {table.length})
        </text>
        <Row
          x={40}
          y={tableY}
          values={table}
          colorFn={(i) => (i <= ranges[0].hi ? PANEL[0].color : PANEL[1].color)}
        />

        <text x={40} y={idxY - 8} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.dim}>
          TOKEN INDICES X = [{indices.join(", ")}]
        </text>
        <Row x={40} y={idxY} values={indices} colorFn={() => T.rule} />

        {gpuResults &&
          PANEL.map((p, pi) => (
            <g key={pi}>
              <text x={p.x} y={resultY - 8} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={p.color}>
                {p.label.toUpperCase()} MASKED RESULT (owns rows {ranges[pi].lo}–{ranges[pi].hi})
              </text>
              <Row
                x={p.x}
                y={resultY}
                values={gpuResults[pi]}
                colorFn={(i) => (gpuResults[pi][i] !== 0 ? p.color : T.rule)}
                dim={(i) => gpuResults[pi][i] === 0}
              />
            </g>
          ))}

        {combined && (
          <g>
            <text x={40} y={combinedY - 8} fontFamily={mono} fontSize="10" letterSpacing="0.12em" fill={T.accent}>
              Emb(X) — COMBINED
            </text>
            <Row x={40} y={combinedY} values={combined} colorFn={() => T.accent} />
          </g>
        )}

        {stage === "backward" && (
          <text x={40} y={resultY + 10} fontFamily={mono} fontSize="11" fill={T.dim}>
            dW_emb[row] += dOut[t] — scatter-add directly into owned rows, no cross-GPU step
          </text>
        )}
      </svg>

      <CommBar comm={comm} />
    </div>
  );
}
