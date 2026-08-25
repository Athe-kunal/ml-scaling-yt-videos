import { useCallback, useEffect, useState } from "react";
import { DEFAULT_DIMS, STEPS } from "./lib/steps";
import { T, mono, sans, cond } from "./lib/theme";
import TransformerDiagram from "./components/TransformerDiagram";
import StepPanel from "./components/StepPanel";
import DimensionControls from "./components/DimensionControls";
import LayerBreakdown from "./components/LayerBreakdown";
import RematerializationPanel from "./components/RematerializationPanel";

function useArrowKeys(setStep, total) {
  const onKey = useCallback(
    (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setStep((s) => Math.min(s + 1, total - 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
    },
    [setStep, total]
  );
  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);
}

const PAGES = [
  { id: "walkthrough", label: "Walkthrough", sub: "diagram · step-by-step" },
  { id: "breakdown", label: "Breakdown", sub: "params · activations" },
];

export default function App() {
  const [step, setStep] = useState(0);
  const [dims, setDims] = useState(DEFAULT_DIMS);
  const [page, setPage] = useState("walkthrough");
  const [checkpointMode, setCheckpointMode] = useState(false);
  useArrowKeys(setStep, STEPS.length);

  const clampedStep = Math.min(Math.max(step, 0), STEPS.length - 1);
  const currentNode = STEPS[clampedStep].node;

  return (
    <div
      style={{
        background: `radial-gradient(1100px 500px at 12% -12%, rgba(94,234,212,0.06), transparent), ${T.bg}`,
        minHeight: "100vh",
        color: T.ink,
        padding: "26px 20px 56px",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        button:focus-visible, input:focus-visible { outline: 2px solid ${T.accent}; outline-offset: 2px; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.6; }
      `}</style>

      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              letterSpacing: "0.22em",
              color: T.dim,
              textTransform: "uppercase",
              marginBottom: 8,
            }}
          >
            transformer layer &nbsp;/&nbsp; flops = 6 · params
          </div>
          <h1
            style={{
              fontFamily: cond,
              fontWeight: 700,
              fontSize: "clamp(26px, 4vw, 40px)",
              margin: 0,
              lineHeight: 1.05,
            }}
          >
            One block at a time,
            <br />
            <span style={{ color: T.accent }}>counting params and FLOPs as we go</span>
          </h1>
          <p
            style={{
              fontFamily: sans,
              fontSize: 14,
              color: T.soft,
              maxWidth: 640,
              marginTop: 12,
              lineHeight: 1.55,
            }}
          >
            Step through a single transformer layer's dataflow — attention, then MLP.
            Each revealed block shows its own parameter count and training FLOPs
            (6× params for weighted matmuls); totals accumulate as you go. Arrow keys
            work.
          </p>
        </header>

        <div style={{ marginBottom: 16 }}>
          <DimensionControls dims={dims} onChange={setDims} />
        </div>

        <nav
          style={{
            display: "grid",
            gap: 8,
            marginBottom: 16,
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          }}
        >
          {PAGES.map((p) => {
            const on = p.id === page;
            return (
              <button
                key={p.id}
                onClick={() => setPage(p.id)}
                style={{
                  textAlign: "left",
                  padding: "11px 13px",
                  borderRadius: 9,
                  cursor: "pointer",
                  border: `1px solid ${on ? T.accent : T.rule}`,
                  background: on ? "rgba(94,234,212,0.08)" : T.panel,
                  transition: "all .18s cubic-bezier(.2,.7,.3,1)",
                }}
              >
                <div
                  style={{
                    fontFamily: mono,
                    fontSize: 10,
                    letterSpacing: "0.14em",
                    color: on ? T.accent : T.dim,
                    marginBottom: 4,
                    textTransform: "uppercase",
                  }}
                >
                  {p.sub}
                </div>
                <div
                  style={{
                    fontFamily: cond,
                    fontWeight: 600,
                    fontSize: 15,
                    color: on ? T.ink : T.soft,
                  }}
                >
                  {p.label}
                </div>
              </button>
            );
          })}
        </nav>

        {page === "walkthrough" && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              marginBottom: 16,
              flexWrap: "wrap",
            }}
          >
            <div style={{ fontFamily: mono, fontSize: 11.5, color: T.soft }}>
              {checkpointMode
                ? "Showing what's saved vs. rematerialized in the backward pass."
                : "Step through the forward-pass dataflow."}
            </div>
            <button
              onClick={() => setCheckpointMode((v) => !v)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontFamily: mono,
                fontSize: 11,
                letterSpacing: "0.04em",
                padding: "7px 12px",
                borderRadius: 999,
                cursor: "pointer",
                border: `1px solid ${checkpointMode ? T.bad : T.rule}`,
                background: checkpointMode ? "rgba(255,138,138,0.1)" : T.panel,
                color: checkpointMode ? T.bad : T.soft,
                transition: "all .18s cubic-bezier(.2,.7,.3,1)",
              }}
            >
              <span
                style={{
                  width: 30,
                  height: 16,
                  borderRadius: 999,
                  background: checkpointMode ? T.bad : T.rule,
                  position: "relative",
                  transition: "background .18s ease",
                  flexShrink: 0,
                }}
              >
                <span
                  style={{
                    position: "absolute",
                    top: 2,
                    left: checkpointMode ? 16 : 2,
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    background: T.bg,
                    transition: "left .18s ease",
                  }}
                />
              </span>
              rematerialization view
            </button>
          </div>
        )}

        {page === "walkthrough" ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.15fr) minmax(300px, 0.85fr)",
              gap: 16,
              alignItems: "start",
            }}
          >
            <TransformerDiagram
              currentStepIndex={clampedStep}
              currentNode={currentNode}
              checkpointMode={checkpointMode}
            />
            {checkpointMode ? (
              <RematerializationPanel dims={dims} />
            ) : (
              <StepPanel step={clampedStep} dims={dims} onStep={setStep} />
            )}
          </div>
        ) : (
          <LayerBreakdown dims={dims} />
        )}

        <footer
          style={{
            fontFamily: sans,
            fontSize: 12,
            color: T.dim,
            marginTop: 28,
            paddingTop: 14,
            borderTop: `1px solid ${T.rule}`,
            lineHeight: 1.6,
          }}
        >
          Formulas and dataflow follow the{" "}
          <span style={{ fontFamily: mono, color: T.soft }}>jax-ml scaling book</span>{" "}
          transformer accounting: B batch, T/S sequence length, D model dim, F MLP
          hidden dim, N/K query/kv heads, H head dim.
        </footer>
      </div>
    </div>
  );
}
