import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";

// Generic prev/next shell: flowchart panel on the left (active block enlarged,
// live variables, code on demand), scene + controls on the right.
//   blocks: [{ id, label, sub, loop?, code: string[] }]
//   steps:  [{ id, block, caption, vars?: () => [[k, v, tone?]] }]
//   scene:  (step, idx) => node     header: (step) => node (pickers)

function Panel({ title, blocks, steps, idx, showCode, setShowCode }) {
  const step = steps[idx];
  const order = blocks.map((b) => b.id);
  const cur = order.indexOf(step.block);
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14, display: "grid", gap: 0 }}>
      <div style={{ fontFamily: sans, fontSize: 11, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600, marginBottom: 10 }}>{title}</div>
      {blocks.map((b, i) => {
        const active = i === cur;
        const state = active ? "active" : i < cur ? "done" : "todo";
        const vars = active && step.vars ? step.vars() : [];
        return (
          <div key={b.id}>
            {i > 0 && (
              <div style={{ display: "grid", justifyItems: "center", height: 14, color: T.rule }}>
                <svg width="10" height="14" viewBox="0 0 10 14"><path d="M5 0 V11 M1 7 L5 12 L9 7" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
              </div>
            )}
            <div
              style={{
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}10` : T.well, borderRadius: active ? 10 : 7,
                padding: active ? "12px 14px" : "6px 10px", opacity: state === "todo" ? 0.5 : 1, transition: "all .3s", boxShadow: active ? `0 0 0 3px ${T.accent}14` : "none",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 14, height: 14, borderRadius: 7, border: `1.5px solid ${state === "done" ? T.accent : T.rule}`, background: state === "done" ? T.accent : "transparent", flexShrink: 0, display: "grid", placeItems: "center", fontSize: 9, color: T.bg, fontWeight: 700 }}>
                  {state === "done" ? "✓" : ""}
                </span>
                <span style={{ fontFamily: mono, fontSize: active ? 14 : 11.5, fontWeight: 700, color: active ? T.accent : T.soft, flex: 1 }}>{b.label}</span>
                {b.loop && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire, border: `1px solid ${T.wire}66`, borderRadius: 10, padding: "1px 7px" }}>↻ steps</span>}
              </div>
              {state !== "todo" && <div style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginLeft: 22, marginTop: 2 }}>{b.sub}</div>}
              {active && (
                <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
                  {vars.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {vars.map(([k, v, tone]) => (
                        <span key={k} style={{ fontFamily: mono, fontSize: 11.5, padding: "3px 8px", borderRadius: 5, background: T.bg, border: `1px solid ${tone === "bad" ? T.bad : T.rule}`, color: tone === "bad" ? T.bad : T.ink }}>
                          <span style={{ color: T.dim }}>{k} </span>{v}
                        </span>
                      ))}
                    </div>
                  )}
                  <button onClick={() => setShowCode(!showCode)} style={{ justifySelf: "start", fontFamily: mono, fontSize: 10.5, padding: "3px 9px", borderRadius: 5, border: `1px solid ${T.rule}`, background: "transparent", color: T.soft, cursor: "pointer" }}>
                    {showCode ? "hide code ▴" : "‹/› show code ▾"}
                  </button>
                  {showCode && (
                    <pre style={{ margin: 0, padding: "8px 10px", background: T.bg, border: `1px solid ${T.rule}`, borderRadius: 6, fontFamily: mono, fontSize: 11, lineHeight: 1.65, color: T.ink, overflowX: "auto" }}>
                      {b.code.join("\n")}
                    </pre>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Stepper({ title, subtitle, blocks, steps, scene, header, legend }) {
  const [idx, setIdx] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [playing, setPlaying] = useState(false);
  const last = steps.length - 1;
  const step = steps[idx];

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") setIdx((i) => Math.min(i + 1, last));
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  useEffect(() => {
    if (!playing) return;
    if (idx >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx((i) => i + 1), 2400);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{title}</div>
          <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>{subtitle}</div>
        </div>
        <div style={{ flex: 1 }} />
        {header && header(step)}
      </header>

      <div className="moe-grid">
        <Panel title={title} blocks={blocks} steps={steps} idx={idx} showCode={showCode} setShowCode={setShowCode} />
        <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12 }}>{scene(step, idx)}</div>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev</button>
            <button onClick={() => setIdx(Math.min(idx + 1, last))} disabled={idx === last} style={btn(true, idx === last)}>next ▶</button>
            <button onClick={() => setPlaying(!playing)} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
            <div style={{ flex: 1, minWidth: 220 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{step.caption}</div>
              <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
                {steps.map((s, i) => (
                  <button key={s.id} onClick={() => setIdx(i)} title={s.id} aria-label={s.id} style={{ flex: 1, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: i === idx ? T.accent : i < idx ? `${T.accent}66` : T.rule }} />
                ))}
              </div>
            </div>
            <span style={{ fontSize: 11, color: T.dim }}>{idx + 1} / {steps.length} · ← →</span>
          </div>
          {legend && <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 11, color: T.dim, padding: "0 4px" }}>{legend}</div>}
        </div>
      </div>
    </div>
  );
}
