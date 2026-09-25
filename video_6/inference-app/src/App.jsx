import { useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { withInlineMath, katexHtml } from "./latex";
import { StepTimeChart, ThroughputChart } from "./Charts";
import AlgorithmSteps from "./AlgorithmSteps";

// Interactive version of "Theoretical estimates for LLM latency and
// throughput" (scaling-book, Inference chapter).
//
//   T_step  = (P + B*Ckv) / W_hbm          (memory-bound decode)
//   tok/s   = B / T_step
//   B*      = P / Ckv                      (where the two byte terms are equal)
//   B*      ~ 2F/S  when H*K = D           (MHA);  GQA/MQA push it up
//
// "Model dims" mode sets W_hbm := P, i.e. the time unit is one bare weight
// load, so T_step = 1 + B/B*  and only the ratio B* matters (layers and
// bytes/element cancel, exactly as in the text).

const TOY = { P: 100, Ckv: 10, W: 1 };
const LLAMA = { D: 8192, F: 28672, S: 8192, H: 128, K: 64 };
const K_PRESETS = [
  { label: "MHA", k: 64, color: T.bad },
  { label: "GQA", k: 8, color: T.accent },
  { label: "MQA", k: 1, color: T.wire },
];

const Math_ = ({ tex, display }) => (
  <span dangerouslySetInnerHTML={{ __html: katexHtml(tex, !!display) }} />
);
const Prose = ({ html, style }) => (
  <div style={style} dangerouslySetInnerHTML={{ __html: withInlineMath(html) }} />
);

function fmt(n, d = 3) {
  if (!isFinite(n)) return "∞";
  if (n >= 1e6) return n.toExponential(2);
  if (n >= 100) return Math.round(n).toLocaleString();
  return Number(n.toPrecision(d)).toString();
}

function Slider({ label, tex, value, onChange, min, max, step = 1, display }) {
  return (
    <label style={{ display: "block", marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 12, color: T.soft, marginBottom: 4 }}>
        <span>{tex ? <Math_ tex={tex} /> : label}</span>
        <span style={{ color: T.ink }}>{display ?? value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: T.accent }}
      />
    </label>
  );
}

const panel = { background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 18 };
const h2 = { margin: "0 0 10px", fontFamily: sans, fontSize: 13, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600 };

function Segmented({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          style={{
            fontFamily: mono, fontSize: 12, padding: "5px 11px", borderRadius: 6, cursor: "pointer",
            background: value === o.value ? T.accent : T.well,
            color: value === o.value ? T.bg : T.soft,
            border: `1px solid ${value === o.value ? T.accent : T.rule}`,
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function ThroughputPage() {
  const [mode, setMode] = useState("toy");
  const [logB, setLogB] = useState(1); // B = 10^logB
  const [toy, setToy] = useState(TOY);
  const [dims, setDims] = useState(LLAMA);
  const [kv2, setKv2] = useState(false); // strict 2x for K and V
  const [computeOn, setComputeOn] = useState(false); // general Eq. (1): MLP can be compute-bound
  const [Bcrit, setBcrit] = useState(240); // C / W_hbm (bf16): ~240 on TPU v5e

  const B = Math.max(1, Math.round(Math.pow(10, logB)));

  // Resolve (P, Ckv, W) for the active mode.
  const model = useMemo(() => {
    const bc = computeOn ? Bcrit : Infinity;
    if (mode === "toy") return { ...toy, Bcrit: bc, unit: "bytes", timeUnit: "s" };
    const P = 2 * dims.D * dims.F;
    const Ckv = (kv2 ? 2 : 1) * dims.S * dims.H * dims.K;
    return { P, Ckv, W: P, Bcrit: bc, unit: "elements", timeUnit: "weight-loads" };
  }, [mode, toy, dims, kv2, computeOn, Bcrit]);

  const { P, Ckv, W } = model;
  const Bstar = P / Ckv;
  // Eq. (1): T = B*Ckv/W + max(2*B*Pcount/C, P/W). With bf16 params the compute
  // term is (B/Bcrit) * P/W, so the MLP term is P/W * max(B/Bcrit, 1).
  const step = (b, m = model) => (b * m.Ckv + m.P * Math.max(b / m.Bcrit, 1)) / m.W;
  const thr = (b, m = model) => b / step(b, m);
  const capOf = (m) => m.W / (m.Ckv + m.P / m.Bcrit);
  const cap = capOf(model);
  const bytesW = P;
  const bytesKV = B * Ckv;
  const kvFrac = bytesKV / (bytesW + bytesKV);
  const pctMax = thr(B) / cap;

  // Architecture variants for the ghost curves (model mode only).
  const variants = useMemo(() => {
    if (mode !== "model") return [];
    return K_PRESETS.map((p) => {
      const m = { P, W, Bcrit: model.Bcrit, Ckv: (kv2 ? 2 : 1) * dims.S * dims.H * p.k };
      return { ...p, m, Bstar: m.P / m.Ckv };
    });
  }, [mode, P, W, model.Bcrit, kv2, dims.S, dims.H]);

  const regime =
    computeOn && B > Bcrit ? "compute-bound MLP — FLOPs, not weight loads, set the MLP time"
    : B < 0.5 * Bstar ? "weight-dominated — extra sequences are nearly free"
    : B > 2 * Bstar ? "KV-dominated — every extra sequence pays its own bytes"
    : "near the bend, B ≈ B*";

  const tableRows = [1, 10, 100, 1000];
  const t0 = thr(1);

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: sans, padding: "28px clamp(16px, 3vw, 40px) 60px" }}>
      <header style={{ maxWidth: 1320, margin: "0 auto 22px" }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SCALING BOOK · INFERENCE</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Latency &amp; throughput: weights are shared, KV caches are not</h1>
        <Prose
          style={{ color: T.soft, maxWidth: 900, lineHeight: 1.55, fontSize: 15 }}
          html="Small-batch decode is memory-bound, so $T_{\text{step}} \approx \frac{P + B\,C_{\text{kv}}}{W_{\text{hbm}}}$. The weights $P$ load once per step regardless of $B$; the KV caches load $B\cdot C_{\text{kv}}$. Drag the batch size and watch where the bend happens."
        />
      </header>

      <div style={{ maxWidth: 1320, margin: "0 auto", display: "grid", gridTemplateColumns: "minmax(280px, 340px) 1fr", gap: 20, alignItems: "start" }}>
        {/* ---------- controls ---------- */}
        <aside style={{ ...panel, position: "sticky", top: 16 }}>
          <div style={h2}>Parameters</div>
          <Segmented
            value={mode} onChange={setMode}
            options={[{ value: "toy", label: "Toy numbers" }, { value: "model", label: "Model dims" }]}
          />

          <Slider
            tex="B" label="Batch size" value={Number(logB.toFixed(2))} min={0} max={4} step={0.01}
            onChange={setLogB} display={B.toLocaleString()}
          />

          {mode === "toy" ? (
            <>
              <Slider tex="P" value={toy.P} min={10} max={500} step={5} onChange={(v) => setToy({ ...toy, P: v })} display={`${toy.P}`} />
              <Slider tex="C_{\text{kv}}" value={toy.Ckv} min={1} max={100} step={1} onChange={(v) => setToy({ ...toy, Ckv: v })} display={`${toy.Ckv}`} />
              <Slider tex="W_{\text{hbm}}" value={toy.W} min={0.5} max={5} step={0.5} onChange={(v) => setToy({ ...toy, W: v })} display={`${toy.W}`} />
            </>
          ) : (
            <>
              <Segmented
                value={dims.K}
                onChange={(k) => setDims({ ...dims, K: k })}
                options={K_PRESETS.map((p) => ({ value: p.k, label: `${p.label} (K=${p.k})` }))}
              />
              <Slider tex="D" value={dims.D} min={1024} max={16384} step={256} onChange={(v) => setDims({ ...dims, D: v })} />
              <Slider tex="F" value={dims.F} min={2048} max={65536} step={512} onChange={(v) => setDims({ ...dims, F: v })} />
              <Slider tex="S" label="Context" value={dims.S} min={512} max={131072} step={512} onChange={(v) => setDims({ ...dims, S: v })} display={dims.S.toLocaleString()} />
              <Slider tex="H" value={dims.H} min={32} max={256} step={32} onChange={(v) => setDims({ ...dims, H: v })} />
              <Slider tex="K" label="KV heads" value={dims.K} min={1} max={64} step={1} onChange={(v) => setDims({ ...dims, K: v })} />
              <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12.5, color: T.soft, fontFamily: mono }}>
                <input type="checkbox" checked={kv2} onChange={(e) => setKv2(e.target.checked)} style={{ accentColor: T.accent }} />
                count K and V (2SHK)
              </label>
            </>
          )}

          <div style={{ borderTop: `1px solid ${T.rule}`, marginTop: 14, paddingTop: 12 }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12.5, color: T.soft, fontFamily: mono, marginBottom: 10 }}>
              <input type="checkbox" checked={computeOn} onChange={(e) => setComputeOn(e.target.checked)} style={{ accentColor: T.accent }} />
              add compute term (Eq. 1)
            </label>
            {computeOn && (
              <Slider tex="B_{\text{crit}}=C/W_{\text{hbm}}" value={Bcrit} min={20} max={2000} step={10} onChange={setBcrit} />
            )}
          </div>

          <div style={{ borderTop: `1px solid ${T.rule}`, marginTop: 10, paddingTop: 14, fontFamily: mono, fontSize: 12.5, lineHeight: 1.9, color: T.soft }}>
            <div><Math_ tex="P" /> = <span style={{ color: T.wire }}>{fmt(P)}</span>{mode === "model" && <span style={{ color: T.dim }}> = 2DF</span>}</div>
            <div><Math_ tex="C_{\text{kv}}" /> = <span style={{ color: T.accent2 }}>{fmt(Ckv)}</span>{mode === "model" && <span style={{ color: T.dim }}> = {kv2 ? "2" : ""}SHK</span>}</div>
            <div><Math_ tex="B^*=P/C_{\text{kv}}" /> = <span style={{ color: T.accent }}>{fmt(Bstar)}</span></div>
            {mode === "model" && <div style={{ color: T.dim }}>HK/D = {fmt((dims.H * dims.K) / dims.D)} · 2F/S = {fmt((2 * dims.F) / dims.S)}</div>}
          </div>
        </aside>

        {/* ---------- main column ---------- */}
        <main style={{ display: "grid", gap: 20, minWidth: 0 }}>
          {/* readout */}
          <section style={{ ...panel, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 16 }}>
            <Stat label="Step time" value={fmt(step(B))} sub={mode === "model" ? "weight-load units" : "time units"} color={T.ink} />
            <Stat label="Throughput" value={fmt(thr(B))} sub={mode === "model" ? "tokens / weight-load" : "tokens / time"} color={T.accent} />
            <Stat label="% of max throughput" value={`${(pctMax * 100).toFixed(1)}%`} sub={`cap = ${fmt(cap)}`} color={T.wire} />
            <div>
              <div style={{ fontSize: 11.5, color: T.dim, fontFamily: mono, marginBottom: 6 }}>BYTES PER STEP AT B={B.toLocaleString()}</div>
              <div style={{ display: "flex", height: 22, borderRadius: 5, overflow: "hidden", border: `1px solid ${T.rule}` }}>
                <div style={{ width: `${(1 - kvFrac) * 100}%`, background: T.wire, transition: "width .15s" }} />
                <div style={{ width: `${kvFrac * 100}%`, background: T.accent2, transition: "width .15s" }} />
              </div>
              <div style={{ fontFamily: mono, fontSize: 11.5, color: T.soft, marginTop: 6 }}>
                <span style={{ color: T.wire }}>weights {((1 - kvFrac) * 100).toFixed(0)}%</span> · <span style={{ color: T.accent2 }}>KV {(kvFrac * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div style={{ gridColumn: "1 / -1", fontSize: 13.5, color: T.soft, borderTop: `1px solid ${T.rule}`, paddingTop: 10 }}>
              Regime: <b style={{ color: T.ink }}>{regime}</b>
            </div>
          </section>

          <section style={panel}>
            <div style={h2}>Step time vs batch size</div>
            <StepTimeChart P={P} Ckv={Ckv} W={W} Bcrit={model.Bcrit} B={B} Bstar={Bstar} />
          </section>

          <section style={panel}>
            <div style={h2}>Throughput vs batch size</div>
            <ThroughputChart model={model} B={B} Bstar={Bstar} cap={cap} variants={variants} />
          </section>

          {/* toy table */}
          <section style={panel}>
            <div style={h2}>Diminishing returns at the current parameters</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: mono, fontSize: 13 }}>
                <thead>
                  <tr style={{ color: T.dim, textAlign: "right" }}>
                    <th style={th("left")}>B</th>
                    <th style={th()}>bytes / step</th>
                    <th style={th()}>tokens/s</th>
                    <th style={th()}>× vs B=1</th>
                    <th style={th()}>step × prev row</th>
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((b, i) => (
                    <tr key={b} style={{ color: T.soft, textAlign: "right" }}>
                      <td style={td("left")}>{b.toLocaleString()}</td>
                      <td style={td()}>{fmt(P + b * Ckv)}</td>
                      <td style={{ ...td(), color: T.accent }}>{fmt(thr(b))}</td>
                      <td style={td()}>{(thr(b) / t0).toFixed(2)}×</td>
                      <td style={td()}>{i === 0 ? "—" : `${(thr(b) / thr(tableRows[i - 1])).toFixed(2)}× throughput`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Prose
              style={{ color: T.dim, fontSize: 13, marginTop: 10, lineHeight: 1.5 }}
              html="With the toy numbers ($P=100$, $C_{\text{kv}}=10$, $W_{\text{hbm}}=1$) the rows read 0.009 → 0.050 → 0.091 → 0.099: about 5.5× from B=1→10 but only 1.09× from B=100→1000."
            />
          </section>

          {/* architecture comparison */}
          {mode === "model" && (
            <section style={panel}>
              <div style={h2}>Architecture changes B*</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14 }}>
                {variants.map((v) => (
                  <div key={v.label} style={{ background: T.well, border: `1px solid ${dims.K === v.k ? v.color : T.rule}`, borderRadius: 8, padding: 14 }}>
                    <div style={{ color: v.color, fontFamily: mono, fontSize: 13, fontWeight: 600 }}>{v.label} · K={v.k}</div>
                    <div style={{ fontSize: 26, margin: "6px 0 2px" }}>B* ≈ {fmt(v.Bstar, 2)}</div>
                    <div style={{ fontSize: 12, color: T.dim, fontFamily: mono }}>
                      {fmt((thr(B, v.m) / capOf(v.m)) * 100, 3)}% of max at B={B.toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
              <Prose
                style={{ color: T.dim, fontSize: 13, marginTop: 12, lineHeight: 1.55 }}
                html="Llama-70B-like ($D=8192$, $F=28672$, $S=8192$, $H=128$): MHA gives $B^*\approx 7$, GQA ($K=8$) gives $\approx 56$. Fewer KV heads means $HK \ll D$, a smaller $C_{\text{kv}}$, and a later bend."
              />
            </section>
          )}

          {/* derivation */}
          <section style={panel}>
            <div style={h2}>Where the bend comes from</div>
            <div style={{ display: "grid", gap: 8, color: T.soft, fontSize: 14.5, lineHeight: 1.6 }}>
              <Math_ display tex={"T_{\\text{step}} = \\underbrace{\\frac{B\\,C_{\\text{kv}}}{W_{\\text{hbm}}}}_{\\text{attention: always bandwidth-bound}} + \\underbrace{\\max\\left(\\frac{2\\,B\\,P_{\\text{count}}}{C},\\; \\frac{P}{W_{\\text{hbm}}}\\right)}_{\\text{MLP: can be compute-bound}}"} />
              <Prose html="Turn on the compute term to plot this general form. The two MLP arguments cross at $B_{\text{crit}} = C/W_{\text{hbm}}$; past it the MLP time grows with $B$ and throughput caps at $W_{\text{hbm}}/(C_{\text{kv}} + P/B_{\text{crit}})$ instead." />
              <Math_ display tex={"P = B\\,C_{\\text{kv}} \\;\\Rightarrow\\; B^* = \\frac{P}{C_{\\text{kv}}} \\approx \\frac{2DF}{SHK} \\overset{HK\\approx D}{=} \\frac{2F}{S}"} />
              <Prose html="Bigger $F$: more weight bytes to amortize, so batching helps for longer. Longer $S$: each sequence's KV cache is bigger, so you hit the wall sooner. At $B=B^*$ you are at exactly half of max throughput; as $B\to\infty$ throughput caps at $W_{\text{hbm}}/C_{\text{kv}}$." />
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

const th = (align = "right") => ({ textAlign: align, padding: "6px 10px", fontWeight: 500, borderBottom: `1px solid ${T.rule}` });
const td = (align = "right") => ({ textAlign: align, padding: "7px 10px", borderBottom: `1px solid ${T.well}` });

function Stat({ label, value, sub, color }) {
  return (
    <div>
      <div style={{ fontSize: 11.5, color: T.dim, fontFamily: mono, marginBottom: 4 }}>{label.toUpperCase()}</div>
      <div style={{ fontSize: 28, color, fontFamily: mono, lineHeight: 1.1 }}>{value}</div>
      <div style={{ fontSize: 11.5, color: T.dim, fontFamily: mono, marginTop: 3 }}>{sub}</div>
    </div>
  );
}

const VIEWS = [
  { id: "throughput", label: "Step time & throughput", sub: "roofline · interactive chart" },
  { id: "algo", label: "Sharded attention", sub: "full algorithm · per pseudocode line" },
];

export default function App() {
  const [view, setView] = useState("throughput");
  return (
    <div style={{ minHeight: "100vh", background: T.bg }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 40px) 0", maxWidth: 1320, margin: "0 auto" }}>
        {VIEWS.map((v) => {
          const active = v.id === view;
          return (
            <button
              key={v.id} onClick={() => setView(v.id)}
              style={{ fontFamily: mono, fontSize: 12, padding: "8px 14px", borderRadius: 8, textAlign: "left", cursor: "pointer",
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}18` : T.panel, color: active ? T.accent : T.soft }}
            >
              <div style={{ fontWeight: 700 }}>{v.label}</div>
              <div style={{ fontSize: 9.5, opacity: 0.75, marginTop: 2, color: active ? T.accent : T.dim }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      {view === "algo" ? <AlgorithmSteps /> : <ThroughputPage />}
    </div>
  );
}
