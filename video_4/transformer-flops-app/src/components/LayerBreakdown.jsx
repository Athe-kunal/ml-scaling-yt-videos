import { layerTotals, crossoverT, fmtNum } from "../lib/steps";
import { T, mono, sans, cond } from "../lib/theme";
import PieChart from "./PieChart";

export default function LayerBreakdown({ dims }) {
  const totals = layerTotals(dims);
  const tStar = crossoverT(dims);
  const attnDominates = totals.attnFlops > totals.mlpFlops;
  const ratio = totals.mlpFlops > 0 ? totals.attnFlops / totals.mlpFlops : Infinity;

  return (
    <div
      style={{
        background: T.panel,
        border: `1px solid ${T.rule}`,
        borderRadius: 10,
        padding: 18,
        display: "grid",
        gap: 18,
      }}
    >
      <div>
        <div
          style={{
            fontFamily: mono,
            fontSize: 9.5,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: T.dim,
            marginBottom: 6,
          }}
        >
          Full-layer breakdown · attention vs MLP
        </div>
        <div style={{ fontFamily: cond, fontSize: 17, fontWeight: 700, color: T.ink }}>
          {attnDominates ? (
            <>
              Attention <span style={{ color: T.accent }}>dominates</span> compute —{" "}
              {ratio.toFixed(2)}× the MLP's FLOPs at T = {fmtNum(dims.T)}
            </>
          ) : (
            <>
              MLP <span style={{ color: T.accent2 }}>dominates</span> compute — attention is{" "}
              {(ratio * 100).toFixed(0)}% of the MLP's FLOPs at T = {fmtNum(dims.T)}
            </>
          )}
        </div>
        <p style={{ fontFamily: sans, fontSize: 12.5, color: T.soft, marginTop: 6, lineHeight: 1.55 }}>
          {tStar != null ? (
            <>
              With the current D, F, N, K, H — and assuming the KV context grows with the
              query (S = T) — attention FLOPs overtake the MLP's at{" "}
              <b style={{ color: T.ink, fontFamily: mono }}>T ≈ {fmtNum(tStar)}</b> tokens.
              Drag the T slider above past that point to watch attention take over.
            </>
          ) : (
            <>Attention already dominates MLP compute at every sequence length for these dimensions.</>
          )}
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 20,
        }}
      >
        <PieChart
          title="params"
          segments={[
            { label: "Attention params", value: totals.attnParams, color: T.accent },
            { label: "MLP params", value: totals.mlpParams, color: T.accent2 },
          ]}
        />
        <PieChart
          title="activations"
          segments={[
            { label: "Attention FLOPs", value: totals.attnFlops, color: T.accent },
            { label: "MLP FLOPs", value: totals.mlpFlops, color: T.accent2 },
          ]}
        />
      </div>
    </div>
  );
}
