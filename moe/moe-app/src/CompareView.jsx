import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";
import { CodeLine } from "./shapes";
import { DsScene, DS_STEPS, DS_BLOCKS } from "./DeepSeekView";
import { KimiScene, KIMI_STEPS, KIMI_BLOCKS } from "./KimiView";
import { LB, simulate } from "./lbModel";

// same runs the two tabs draw (γ = 0.02 on this toy). Row u = the state after u bias updates.
const SG = simulate("sign", 60, 0.02);
const QBR = simulate("qb", 60);
const cv = (R, u) => R[u].cv.toFixed(2);

// DeepSeek-V3 and Kimi K3 load balancing, side by side. Every pair below puts a
// DeepSeek step next to the Kimi step that does the same job, so one prev / next
// walks both codes and both pictures together. A side with no counterpart shows
// a short note instead. The scenes and code are the ones from the two tabs.

const PAIRS = [
  { part: "pick experts", ds: "topk", ki: "topk1", say: "Both start the same way: every token keeps its top-k experts. They differ in what they do about imbalance." },
  { part: "measure", ds: "monitor", ki: "margin", say: "DeepSeek counts tokens per expert. Kimi measures how far each score sits above its token's cutoff." },
  { part: "target", ds: "err", ki: "q", say: "DeepSeek asks only: over or under the mean? Kimi asks: what threshold leaves exactly k/N of the tokens above it?" },
  { part: "update b", ds: "update", ki: "quantile", say: "DeepSeek moves b by a fixed ±γ: a sign step on the load error. Kimi sets b straight to the quantile, the exact answer to the same objective, so there is no step size to tune." },
  { part: "update b", ds: null, dsNote: "no equivalent step: each bias just moves by ±γ", ki: "center", say: "Kimi removes the mean of b. That changes no ranking, it only keeps the biases small." },
  { part: "at scale", ds: null, dsNote: "only the token count per expert is needed: N numbers, one small all-reduce", ki: "hist", say: "Across GPUs Kimi cannot sort all margins, so every rank bins its own." },
  { part: "at scale", ds: null, dsNote: "only the token count per expert is needed: N numbers, one small all-reduce", ki: "reduce", say: "The ranks add their bin counts: N × L numbers, whatever the number of tokens." },
  { part: "at scale", ds: null, dsNote: "only the token count per expert is needed: N numbers, one small all-reduce", ki: "bhat", say: "The quantile is read off the pooled histogram. This is the price of Kimi's method: a histogram all-reduce." },
  { part: "apply", ds: "select", ki: "apply", say: "Both apply the bias to the NEXT batch. It only decides who is picked; the gate still uses the raw score." },
  // DeepSeek loop step t shows the load under t-1 updates; Kimi's t shows the state after t updates.
  // So Kimi t2 pairs with DeepSeek t3 (2 updates each) and Kimi t5 with DeepSeek t6 (5 each).
  { part: "race", ds: "t3", ki: "t2", say: `After 2 updates, same toy, same batches: load CV ${cv(SG, 2)} for the sign step, ${cv(QBR, 2)} for the quantile. Kimi is almost there.` },
  { part: "race", ds: "t6", ki: "t5", say: `After 5 updates: CV ${cv(SG, 5)} for the sign step, ${cv(QBR, 5)} for the quantile. The sign step is still closing the gap.` },
  { part: "race", ds: "select10", ki: "t10", say: `After 10 updates the sign step has caught up on this toy (CV ${cv(SG, 10)} vs ${cv(QBR, 10)}): 4 experts and a large γ = 0.02. DeepSeek's real γ is 0.001 and K3 has 896 experts, where a fixed step is far slower. Both biases are frozen at inference.` },
  { part: "loss", ds: "f", ki: null, kiNote: "no balance-loss term in this walkthrough: Kimi balances with the bias alone", say: "DeepSeek also keeps a tiny balance loss as a safety net. First f: how much of its fair share each expert got." },
  { part: "loss", ds: "P", ki: null, kiNote: "no balance-loss term in this walkthrough: Kimi balances with the bias alone", say: "P: how much the router likes each expert. Smooth, so it carries the gradient." },
  { part: "loss", ds: "L", ki: null, kiNote: "no balance-loss term in this walkthrough: Kimi balances with the bias alone", say: "L = α Σ f·P. It stays tiny: the bias does the real balancing." },
];

const PARTS = [...new Set(PAIRS.map((p) => p.part))];
const find = (arr, id) => (id ? arr.find((s) => s.id === id) : null);

function Side({ name, sub, color, blocks, step, note, scene }) {
  const block = step ? blocks.find((b) => b.id === step.block) : null;
  const lines = step?.lines ?? [];
  const vars = step?.vars ? step.vars() : [];
  return (
    <div style={{ minWidth: 0, display: "grid", gap: 10, alignContent: "start" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, borderBottom: `2px solid ${color}`, paddingBottom: 6 }}>
        <span style={{ fontFamily: sans, fontSize: 15, fontWeight: 700, color }}>{name}</span>
        <span style={{ fontSize: 11, color: T.dim }}>{sub}</span>
      </div>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 10, minHeight: 190 }}>
        {block ? (
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={{ fontFamily: mono, fontSize: 12.5, fontWeight: 700, color }}>{block.label}</span>
              <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{block.sub}</span>
            </div>
            <div style={{ padding: "6px 0", background: T.bg, border: `1px solid ${T.rule}`, borderRadius: 6, fontFamily: mono, fontSize: 11, lineHeight: 1.6 }}>
              {block.code.map((l, i) => {
                const on = lines.includes(i);
                return (
                  <div key={i} style={{ padding: "2px 10px", borderLeft: `3px solid ${on ? color : "transparent"}`, background: on ? `${color}1c` : "transparent", color: on ? T.ink : T.dim, transition: "background .2s" }}>
                    <CodeLine text={l} />
                  </div>
                );
              })}
            </div>
            {vars.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {vars.map(([k, v, tone]) => (
                  <span key={k} style={{ fontFamily: mono, fontSize: 11, padding: "2px 7px", borderRadius: 5, background: T.bg, border: `1px solid ${tone === "bad" ? T.bad : T.rule}`, color: tone === "bad" ? T.bad : T.ink }}>
                    <span style={{ color: T.dim }}>{k} </span>{v}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: T.dim, fontSize: 12, display: "grid", placeItems: "center", height: 170 }}>no code for this step</div>
        )}
      </div>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 8 }}>
        {step ? scene(step) : (
          <div style={{ display: "grid", placeItems: "center", minHeight: 330, padding: 24, textAlign: "center", color: T.soft, fontSize: 14, lineHeight: 1.6 }}>{note}</div>
        )}
      </div>
      {step && <div style={{ fontSize: 13, color: T.soft, padding: "0 4px" }}>{step.caption}</div>}
    </div>
  );
}

export default function CompareView() {
  const [i0, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ex, setEx] = useState(0);
  const [tok, setTok] = useState(0);
  const last = PAIRS.length - 1;
  const i = Math.min(i0, last);
  const pair = PAIRS[i];
  const ds = find(DS_STEPS, pair.ds);
  const ki = find(KIMI_STEPS, pair.ki);
  const partIdx = PARTS.indexOf(pair.part);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") setI((v) => Math.min(v + 1, last));
      if (e.key === "ArrowLeft") setI((v) => Math.max(v - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  useEffect(() => {
    if (!playing) return;
    if (i >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setI(i + 1), 2200);
    return () => clearTimeout(t);
  }, [playing, i, last]);

  const picker = (label, val, set, count) => (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>{label}</span>
      {Array.from({ length: count }, (_, k) => (
        <button key={k} onClick={() => set(k)} style={{ ...btn(k === val, false), height: 28, padding: "0 9px" }}>{k}</button>
      ))}
    </div>
  );

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>DeepSeek-V3 vs Kimi K3 · load balancing</div>
          <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>same toy router both sides: {LB.n} experts, top-{LB.k}, {LB.m}-token batches · each step puts the matching step of each method side by side</div>
        </div>
        <div style={{ flex: 1 }} />
        {picker("follow expert", ex, setEx, LB.n)}
        {ki?.pick === "tok" && picker("follow token", tok, setTok, LB.T)}
      </header>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {PARTS.map((p, k) => (
          <button
            key={p} onClick={() => setI(PAIRS.findIndex((x) => x.part === p))}
            style={{ fontFamily: mono, fontSize: 11, padding: "4px 10px", borderRadius: 14, cursor: "pointer", border: `1px solid ${k === partIdx ? T.accent : T.rule}`, background: k === partIdx ? `${T.accent}18` : "transparent", color: k === partIdx ? T.accent : k < partIdx ? T.soft : T.dim }}
          >
            {k + 1} · {p}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(520px, 1fr))", gap: 16 }}>
        <Side
          name="DeepSeek-V3" sub="bias nudged by ±γ" color={T.accent2} blocks={DS_BLOCKS} step={ds} note={pair.dsNote}
          scene={(s) => <DsScene step={s} ex={ex} setEx={setEx} />}
        />
        <Side
          name="Kimi K3" sub="bias set from a quantile" color={T.accent} blocks={KIMI_BLOCKS} step={ki} note={pair.kiNote}
          scene={(s) => <KimiScene step={s} tok={tok} ex={ex} setEx={setEx} />}
        />
      </div>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <button onClick={() => setI(Math.max(i - 1, 0))} disabled={i === 0} style={btn(false, i === 0)}>◀ previous</button>
        <button onClick={() => setI(Math.min(i + 1, last))} disabled={i === last} style={btn(true, i === last)}>next ▶</button>
        <button onClick={() => setPlaying(!playing)} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
        <div style={{ flex: 1, minWidth: 260 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{pair.say}</div>
          <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
            {PAIRS.map((p, k) => (
              <button key={k} onClick={() => setI(k)} title={p.part} aria-label={`step ${k + 1}`} style={{ flex: 1, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: k === i ? T.accent : k < i ? `${T.accent}66` : T.rule }} />
            ))}
          </div>
        </div>
        <span style={{ fontSize: 11, color: T.dim }}>step {i + 1} / {PAIRS.length} · ← →</span>
      </div>
    </div>
  );
}
