import { useMemo, useState } from "react";
import { T, mono, sans, cond } from "../lib/theme";

// Interactive reproduction of the scaling-book's "ratio of FLOPs to comms
// time" figure (https://jax-ml.github.io/scaling-book/training/, the
// "Combining FSDP and tensor parallelism" section): three strategies —
// pure FSDP, pure TP, and the FSDP+TP mix optimized per batch size — each
// plotted as FLOPs:comms ratio vs. batch tokens per chip (r = B/N), with
// alpha = C/W_ici (per-chip compute flops/s over ICI bandwidth) and F
// (feed-forward width) as the two knobs that set where the "mixing is
// required to stay compute-bound" regime sits.
//
// Closed-form model, derived from the book's own compute-bound
// inequalities (B/X > alpha for FSDP; F > Y*alpha for TP; B/N >
// alpha^2/F for the jointly-optimized mix), self-consistently:
//   Ratio_FSDP(r)       = r / alpha                (pure FSDP, X=N,Y=1)
//   Ratio_TP             = F / (alpha * N)          (pure TP, X=1,Y=N — flat, no r-dependence)
//   Ratio_Mixed_opt(r)   = max(r, sqrt(r*F)) / alpha  (jointly optimal X*Y=N split)
// The mixed curve's sqrt(r*F) branch is the continuous optimum found by
// balancing FSDP-comm against TP-comm (X_opt = sqrt(B*N/F)); the max(...)
// clamps it to degenerate into plain FSDP once r exceeds F, where the
// unconstrained optimum would need X > N (not physically available).
// Both thresholds (r_low = alpha^2/F, r_high = alpha) are independent of
// N — a nice, checkable consequence of the model, not an assumption.

const R_MIN = 10;
const R_MAX = 100000;
const LOG_R_MIN = Math.log10(R_MIN);
const LOG_R_MAX = Math.log10(R_MAX);
const Y_LOG_MIN = -2; // ratio 0.01
const Y_LOG_MAX = 2.15; // ratio ~140

const CHART_W = 860;
const CHART_H = 460;
const PAD = { top: 26, right: 108, bottom: 46, left: 56 };
const PLOT_W = CHART_W - PAD.left - PAD.right;
const PLOT_H = CHART_H - PAD.top - PAD.bottom;

const N_PRESETS = [
  { label: "4×4×1", n: 16 },
  { label: "4×4×4", n: 64 },
  { label: "8×8×4", n: 256 },
  { label: "8×8×16", n: 1024 },
];

const SERIES = {
  fsdp: { color: T.wire, label: "Pure FSDP" },
  tp: { color: T.accent2, label: "Pure TP" },
  mixed: { color: T.accent, label: "Mixed FSDP + TP (optimal split)" },
};

function xToPx(r) {
  const t = (Math.log10(r) - LOG_R_MIN) / (LOG_R_MAX - LOG_R_MIN);
  return PAD.left + t * PLOT_W;
}
function yToPx(ratio) {
  const clamped = Math.max(Math.min(Math.log10(Math.max(ratio, 1e-6)), Y_LOG_MAX), Y_LOG_MIN);
  const t = (clamped - Y_LOG_MIN) / (Y_LOG_MAX - Y_LOG_MIN);
  return PAD.top + (1 - t) * PLOT_H;
}

function ratioFsdp(r, alpha) {
  return r / alpha;
}
function ratioMixed(r, alpha, F) {
  return Math.max(r, Math.sqrt(r * F)) / alpha;
}
function ratioTp(alpha, F, N) {
  return F / (alpha * N);
}

function fmt(n) {
  if (n >= 1000) return `${Math.round(n / 100) / 10}k`;
  if (n >= 1) return Math.round(n).toString();
  return n.toFixed(2);
}

function Slider({ label, value, onChange, min, max, step, display }) {
  return (
    <label style={{ display: "block", marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontFamily: mono, fontSize: 11.5, color: T.soft }}>{label}</span>
        <span style={{ fontFamily: mono, fontSize: 11.5, color: T.accent, fontWeight: 600 }}>{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: T.accent }}
      />
    </label>
  );
}

export default function FlopsCommsChart() {
  const [alpha, setAlpha] = useState(850);
  const [F, setF] = useState(7225);
  const [N, setN] = useState(64);
  const [logR, setLogR] = useState(Math.log10(300));

  const r = Math.pow(10, logR);

  const { rLow, rHigh, curves } = useMemo(() => {
    const rLow = (alpha * alpha) / F;
    const rHigh = alpha;
    const steps = 160;
    const pts = { fsdp: [], mixed: [] };
    for (let i = 0; i <= steps; i++) {
      const lr = LOG_R_MIN + (i / steps) * (LOG_R_MAX - LOG_R_MIN);
      const rr = Math.pow(10, lr);
      pts.fsdp.push([rr, ratioFsdp(rr, alpha)]);
      pts.mixed.push([rr, ratioMixed(rr, alpha, F)]);
    }
    return { rLow, rHigh, curves: pts };
  }, [alpha, F]);

  const tp = ratioTp(alpha, F, N);
  const atR = {
    fsdp: ratioFsdp(r, alpha),
    mixed: ratioMixed(r, alpha, F),
    tp,
  };

  const pathFor = (pts) => pts.map(([rr, ratio], i) => `${i === 0 ? "M" : "L"} ${xToPx(rr).toFixed(1)} ${yToPx(ratio).toFixed(1)}`).join(" ");

  const xTicks = [10, 100, 1000, 10000, 100000];
  const yTicks = [0.01, 0.1, 1, 10, 100];

  const bandX1 = xToPx(Math.max(rLow, R_MIN));
  const bandX2 = xToPx(Math.min(rHigh, R_MAX));
  const bandVisible = rHigh > R_MIN && rLow < R_MAX && bandX2 > bandX1;

  return (
    <div>
      <header style={{ marginBottom: 18 }}>
        <div style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: "0.16em", color: T.dim, textTransform: "uppercase", marginBottom: 6 }}>
          Scaling ML Models
        </div>
        <h1 style={{ fontFamily: cond, fontSize: 26, fontWeight: 700, margin: 0, color: T.ink }}>FLOPs : Comms — Mixed FSDP + TP</h1>
        <p style={{ fontFamily: sans, fontSize: 13, color: T.soft, maxWidth: 640, marginTop: 8, lineHeight: 1.55 }}>
          Reproduces the book's claim: pure FSDP dominates at very large batch/chip, but there's a regime where neither pure
          strategy alone is compute-bound and mixing is required. Drag the sliders — the shaded band tracks the same math live.
        </p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.4fr) minmax(220px,0.85fr)", gap: 18, alignItems: "start" }}>
        <div style={{ background: T.well, border: `1px solid ${T.rule}`, borderRadius: 10, padding: "16px 14px 10px" }}>
          <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} style={{ width: "100%", display: "block" }}>
            {/* gridlines */}
            {xTicks.map((t) => (
              <line key={`gx${t}`} x1={xToPx(t)} y1={PAD.top} x2={xToPx(t)} y2={PAD.top + PLOT_H} stroke={T.rule} strokeWidth={1} opacity={0.35} />
            ))}
            {yTicks.map((t) => (
              <line key={`gy${t}`} x1={PAD.left} y1={yToPx(t)} x2={PAD.left + PLOT_W} y2={yToPx(t)} stroke={T.rule} strokeWidth={1} opacity={0.35} />
            ))}

            {/* mixing-required band */}
            {bandVisible && (
              <rect
                x={bandX1}
                y={PAD.top}
                width={bandX2 - bandX1}
                height={PLOT_H}
                fill={T.accent}
                opacity={0.08}
              />
            )}

            {/* compute-bound = 1 line */}
            <line x1={PAD.left} y1={yToPx(1)} x2={PAD.left + PLOT_W} y2={yToPx(1)} stroke={T.soft} strokeWidth={1.3} strokeDasharray="5 4" opacity={0.7} />
            <text x={PAD.left + PLOT_W} y={yToPx(1) - 6} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.soft}>
              compute-bound (ratio = 1)
            </text>

            {/* curves */}
            <path d={pathFor(curves.fsdp)} fill="none" stroke={SERIES.fsdp.color} strokeWidth={2} opacity={0.9} />
            <path d={pathFor(curves.mixed)} fill="none" stroke={SERIES.mixed.color} strokeWidth={2.4} />
            <line
              x1={PAD.left}
              y1={yToPx(tp)}
              x2={PAD.left + PLOT_W}
              y2={yToPx(tp)}
              stroke={SERIES.tp.color}
              strokeWidth={2}
              opacity={0.9}
            />

            {/* direct end-labels — nudged apart when curves converge (e.g. FSDP
                and Mixed are identical once B/N exceeds F) so labels never overlap */}
            {(() => {
              const MIN_GAP = 12;
              const raw = [
                { key: "fsdp", label: "FSDP", y: yToPx(curves.fsdp[curves.fsdp.length - 1][1]) },
                { key: "mixed", label: "Mixed", y: yToPx(curves.mixed[curves.mixed.length - 1][1]) },
                { key: "tp", label: "TP", y: yToPx(tp) },
              ].sort((a, b) => a.y - b.y);
              for (let i = 1; i < raw.length; i++) {
                if (raw[i].y - raw[i - 1].y < MIN_GAP) raw[i].y = raw[i - 1].y + MIN_GAP;
              }
              return raw.map((d) => (
                <text key={d.key} x={PAD.left + PLOT_W + 6} y={d.y + 3} fontFamily={mono} fontSize="10.5" fill={SERIES[d.key].color}>
                  {d.label}
                </text>
              ));
            })()}

            {/* marker */}
            <line x1={xToPx(r)} y1={PAD.top} x2={xToPx(r)} y2={PAD.top + PLOT_H} stroke={T.ink} strokeWidth={1} opacity={0.5} strokeDasharray="2 3" />
            {[atR.fsdp, atR.mixed, atR.tp].map((v, i) => {
              const color = [SERIES.fsdp.color, SERIES.mixed.color, SERIES.tp.color][i];
              return <circle key={i} cx={xToPx(r)} cy={yToPx(v)} r={4.5} fill={T.well} stroke={color} strokeWidth={2} />;
            })}

            {/* axes */}
            <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + PLOT_H} stroke={T.rule} strokeWidth={1.2} />
            <line x1={PAD.left} y1={PAD.top + PLOT_H} x2={PAD.left + PLOT_W} y2={PAD.top + PLOT_H} stroke={T.rule} strokeWidth={1.2} />
            {xTicks.map((t) => (
              <text key={`xt${t}`} x={xToPx(t)} y={PAD.top + PLOT_H + 18} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.dim}>
                {fmt(t)}
              </text>
            ))}
            {yTicks.map((t) => (
              <text key={`yt${t}`} x={PAD.left - 8} y={yToPx(t) + 3} textAnchor="end" fontFamily={mono} fontSize="10" fill={T.dim}>
                {fmt(t)}
              </text>
            ))}
            <text x={PAD.left + PLOT_W / 2} y={CHART_H - 6} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.soft} letterSpacing="0.05em">
              batch tokens per chip — B/N
            </text>
            <text
              x={16}
              y={PAD.top + PLOT_H / 2}
              transform={`rotate(-90 16 ${PAD.top + PLOT_H / 2})`}
              textAnchor="middle"
              fontFamily={mono}
              fontSize="10.5"
              fill={T.soft}
              letterSpacing="0.05em"
            >
              FLOPs : comms ratio
            </text>
          </svg>

          <div style={{ fontFamily: sans, fontSize: 11.5, color: T.dim, marginTop: 6, lineHeight: 1.5 }}>
            {bandVisible ? (
              <>
                Shaded band: mixing is <b style={{ color: T.ink }}>required</b> to stay compute-bound for{" "}
                <span style={{ fontFamily: mono, color: T.accent }}>{fmt(rLow)}</span> ≲ B/N ≲{" "}
                <span style={{ fontFamily: mono, color: T.accent }}>{fmt(rHigh)}</span>. Below it, even mixing can't help;
                above it, plain FSDP already suffices.
              </>
            ) : (
              <>At these settings the mixing-required band falls outside the plotted range.</>
            )}
          </div>
        </div>

        <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 16 }}>
          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim, textTransform: "uppercase", marginBottom: 12 }}>
            Hardware &amp; model
          </div>

          <Slider
            label="α = C / W_ici  (compute ÷ ICI bandwidth)"
            value={alpha}
            onChange={setAlpha}
            min={100}
            max={3000}
            step={10}
            display={fmt(alpha)}
          />
          <Slider label="F  (feed-forward width)" value={F} onChange={setF} min={500} max={50000} step={25} display={fmt(F)} />
          <Slider label="N  (total chips)" value={N} onChange={setN} min={4} max={1024} step={4} display={fmt(N)} />
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 18, marginTop: -6 }}>
            {N_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setN(p.n)}
                style={{
                  fontFamily: mono,
                  fontSize: 9.5,
                  padding: "3px 8px",
                  borderRadius: 5,
                  border: `1px solid ${N === p.n ? T.accent : T.rule}`,
                  background: N === p.n ? `${T.accent}18` : "transparent",
                  color: N === p.n ? T.accent : T.dim,
                  cursor: "pointer",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim, textTransform: "uppercase", marginBottom: 12 }}>
            Inspect a point
          </div>
          <Slider
            label="B/N  (batch tokens per chip)"
            value={logR}
            onChange={setLogR}
            min={LOG_R_MIN}
            max={LOG_R_MAX}
            step={0.01}
            display={fmt(r)}
          />

          <div style={{ display: "grid", gap: 8, marginTop: 4 }}>
            {[
              { key: "mixed", v: atR.mixed },
              { key: "fsdp", v: atR.fsdp },
              { key: "tp", v: atR.tp },
            ].map(({ key, v }) => {
              const bound = v >= 1;
              return (
                <div
                  key={key}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "7px 10px",
                    borderRadius: 7,
                    border: `1px solid ${T.rule}`,
                    background: T.well,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 4, background: SERIES[key].color, display: "inline-block" }} />
                    <span style={{ fontFamily: sans, fontSize: 11.5, color: T.soft }}>{SERIES[key].label}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontFamily: mono, fontSize: 12, color: T.ink, fontWeight: 600 }}>{v.toFixed(2)}×</span>
                    <span
                      style={{
                        fontFamily: mono,
                        fontSize: 8.5,
                        letterSpacing: "0.06em",
                        textTransform: "uppercase",
                        padding: "2px 6px",
                        borderRadius: 4,
                        color: bound ? T.accent : T.bad,
                        border: `1px solid ${bound ? T.accent : T.bad}55`,
                        background: bound ? `${T.accent}14` : `${T.bad}14`,
                      }}
                    >
                      {bound ? "compute" : "comm"}-bound
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <footer
        style={{
          fontFamily: sans,
          fontSize: 11.5,
          color: T.dim,
          marginTop: 22,
          paddingTop: 14,
          borderTop: `1px solid ${T.rule}`,
          lineHeight: 1.6,
        }}
      >
        Model: <span style={{ fontFamily: mono, color: T.soft }}>Ratio_FSDP = (B/N)/α</span>,{" "}
        <span style={{ fontFamily: mono, color: T.soft }}>Ratio_TP = F/(αN)</span>,{" "}
        <span style={{ fontFamily: mono, color: T.soft }}>Ratio_Mixed = max(B/N, √((B/N)·F)) / α</span> — derived from the book's
        own compute-bound conditions (B/X&gt;α for FSDP, F&gt;Y·α for TP, B/N&gt;α²/F for the jointly-optimized mix); a mesh
        bandwidth bonus (M_X·M_Y in the book) is taken as 1 here. Defaults (α=850, F=7225) are chosen to land the "mixing
        required" band around the book's own cited range of roughly 100–850 tokens/chip on a TPUv5p 4×4×4 slice.
      </footer>
    </div>
  );
}
