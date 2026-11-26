import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";
import { CHAPTERS, STEPS } from "./steps";
import { BitEditor } from "./Bits";

// Left: chapter flowchart (the current step's node enlarges). Right: transport + caption,
// the step's picture, and the live bit editor that every step starts from.

function Flow({ idx, go }) {
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14, display: "grid", gap: 10 }}>
      {CHAPTERS.map((c) => {
        const items = STEPS.map((s, i) => [s, i]).filter(([s]) => s.ch === c.id);
        const here = STEPS[idx].ch === c.id;
        return (
          <div key={c.id} style={{ border: `1px dashed ${here ? T.accent + "88" : T.rule}`, borderRadius: 10, padding: "8px 8px 10px", background: `${T.bg}66` }}>
            <div style={{ fontFamily: mono, fontSize: 14, fontWeight: 700, color: here ? T.accent : T.soft, margin: "0 4px 2px" }}>{c.label}</div>
            <div style={{ fontFamily: mono, fontSize: 12, color: T.dim, margin: "0 4px 6px" }}>{c.sub}</div>
            <div style={{ display: "grid", gap: 3 }}>
              {items.map(([s, i]) => {
                const active = i === idx, done = i < idx;
                return (
                  <button key={s.id} onClick={() => go(i)}
                    style={{ textAlign: "left", cursor: "pointer", fontFamily: mono, transition: "all .25s",
                      fontSize: active ? 16 : 13, fontWeight: active ? 700 : 400, padding: active ? "8px 10px" : "3px 10px", borderRadius: 7,
                      border: `1px solid ${active ? T.accent : "transparent"}`, background: active ? `${T.accent}14` : "transparent",
                      color: active ? T.accent : done ? T.soft : T.dim }}>
                    <span style={{ display: "inline-block", width: 16, color: done ? T.accent : T.rule }}>{done ? "✓" : "·"}</span>{s.label}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [idx, setIdx] = useState(() => Math.max(0, STEPS.findIndex((s) => `#${s.id}` === location.hash)));
  const step = STEPS[idx];
  const [fmt, setFmt] = useState(step.fmt);
  const [raw, setRaw] = useState(step.raw);
  const last = STEPS.length - 1;
  const go = (i) => setIdx(Math.max(0, Math.min(last, i)));
  const reset = () => { setFmt(step.fmt); setRaw(step.raw); };

  useEffect(() => { reset(); history.replaceState(null, "", `#${step.id}`); }, [idx]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") go(idx + 1);
      if (e.key === "ArrowLeft") go(idx - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const { Scene } = step;
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
      <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
        <header>
          <div style={{ fontSize: 20, fontWeight: 700, fontFamily: sans }}>Floating point, step by step</div>
          <div style={{ fontSize: 13, color: T.dim, marginTop: 2 }}>IEEE-754 · fp32 · fp16 · bf16</div>
        </header>
        <div className="vit-grid">
          <Flow idx={idx} go={go} />
          <div style={{ display: "grid", gap: 12, alignContent: "start", minWidth: 0 }}>
            <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <button onClick={() => go(idx - 1)} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev</button>
              <button onClick={() => go(idx + 1)} disabled={idx === last} style={btn(true, idx === last)}>next ▶</button>
              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 22, fontWeight: 600, color: T.ink, fontFamily: sans }}>{step.caption}</div>
                <div style={{ display: "flex", gap: 3, marginTop: 8 }}>
                  {STEPS.map((s, k) => (
                    <button key={s.id} onClick={() => go(k)} title={s.label} aria-label={s.label}
                      style={{ flex: "1 1 0", minWidth: 0, padding: 0, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: k === idx ? T.accent : k < idx ? `${T.accent}66` : T.rule }} />
                  ))}
                </div>
              </div>
              <span style={{ fontSize: 14, color: T.dim }}>step {idx + 1} / {STEPS.length} · ← →</span>
            </div>
            <div key={step.id} className="scene" style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12, minWidth: 0, overflowX: "auto" }}>
              <Scene />
            </div>
            <BitEditor fmt={fmt} raw={raw} setFmt={setFmt} setRaw={setRaw} hl={step.hl} onReset={reset} />
          </div>
        </div>
      </div>
    </div>
  );
}
