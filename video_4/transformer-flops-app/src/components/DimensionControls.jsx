import { DEFAULT_DIMS, DIM_INFO, DIM_RANGES, DIM_STEP } from "../lib/steps";
import { T, mono } from "../lib/theme";

export default function DimensionControls({ dims, onChange }) {
  const setDim = (key, value) => {
    const v = Math.max(1, Math.round(Number(value) || 1));
    const next = { ...dims, [key]: v };
    // "sequence length" is one knob in practice: dragging T keeps S in
    // lockstep so the attention-vs-MLP crossover exploration below reflects
    // a growing context, not just a growing query chunk. S can still be
    // pulled independently afterwards for a custom KV-cache length.
    if (key === "T") next.S = v;
    onChange(next);
  };

  return (
    <div
      style={{
        background: T.panel,
        border: `1px solid ${T.rule}`,
        borderRadius: 10,
        padding: 16,
        display: "grid",
        gap: 14,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div
          style={{
            fontFamily: mono,
            fontSize: 9.5,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: T.dim,
          }}
        >
          Dimensions &nbsp;
          <span style={{ color: T.dim, textTransform: "none", letterSpacing: "normal" }}>
            (arrows / drag step by {DIM_STEP})
          </span>
        </div>
        <button
          onClick={() => onChange(DEFAULT_DIMS)}
          style={{
            fontFamily: mono,
            fontSize: 10,
            padding: "4px 9px",
            borderRadius: 5,
            border: `1px solid ${T.rule}`,
            background: "transparent",
            color: T.soft,
            cursor: "pointer",
          }}
        >
          reset to book defaults
        </button>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          gap: 14,
        }}
      >
        {DIM_INFO.map(({ key, label }) => {
          const { min, max } = DIM_RANGES[key];
          return (
            <div key={key}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                  fontFamily: mono,
                  fontSize: 10.5,
                  marginBottom: 6,
                }}
              >
                <span>
                  <span style={{ color: T.accent }}>{key}</span>{" "}
                  <span style={{ color: T.dim }}>{label}</span>
                </span>
                <input
                  type="number"
                  min="1"
                  step={DIM_STEP}
                  value={dims[key]}
                  onChange={(e) => setDim(key, e.target.value)}
                  style={{
                    width: 76,
                    fontFamily: mono,
                    fontSize: 12,
                    color: T.ink,
                    background: T.well,
                    border: `1px solid ${T.rule}`,
                    borderRadius: 6,
                    padding: "3px 6px",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={DIM_STEP}
                value={dims[key]}
                onChange={(e) => setDim(key, e.target.value)}
                style={{ width: "100%", accentColor: T.accent }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
