import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";
import { CodeLine } from "./shapes";
import { TagStrip } from "./draw";

// Step-level prev/next shell (one step = one job, not one code line).
// Left: flowchart of grouped blocks; the running block enlarges and shows its
// live variables, with its code behind a "{ }" toggle. Right: the scene, the
// shape card, and the transport bar.
//   blocks: [{ id, label, sub, code?: string[] } | { group, loop?, items: [...] }]
//   steps:  [{ id, block, caption, vars?: () => [[k, v, tone?]] }]
//   scene(step, idx)   tags(step) -> [{ name, dims, io }] drawn atop the scene   matmuls(step)   controls: node

const Arrow = () => (
  <div style={{ display: "grid", justifyItems: "center", height: 14, color: T.rule }}>
    <svg width="10" height="14" viewBox="0 0 10 14"><path d="M5 0 V11 M1 7 L5 12 L9 7" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
  </div>
);

function Node({ b, state, step, showCode, setShowCode }) {
  const active = state === "active";
  const vars = active && step.vars ? step.vars() : [];
  return (
    <div
      style={{
        border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}10` : T.well, borderRadius: active ? 10 : 7,
        padding: active ? "11px 13px" : "6px 10px", opacity: state === "todo" ? 0.5 : 1, transition: "all .3s", boxShadow: active ? `0 0 0 3px ${T.accent}14` : "none",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 14, height: 14, borderRadius: 7, border: `1.5px solid ${state === "done" ? T.accent : T.rule}`, background: state === "done" ? T.accent : "transparent", flexShrink: 0, display: "grid", placeItems: "center", fontSize: 12, color: T.bg, fontWeight: 700 }}>
          {state === "done" ? "✓" : ""}
        </span>
        <span style={{ fontFamily: mono, fontSize: active ? 17 : 14, fontWeight: 700, color: active ? T.accent : T.soft, flex: 1 }}>{b.label}</span>
        {active && b.code && (
          <button onClick={() => setShowCode(!showCode)} title="show the code for this block" style={{ ...btn(showCode, false), height: 22, fontSize: 13, padding: "0 7px" }}>{"{ }"}</button>
        )}
      </div>
      {state !== "todo" && <div style={{ fontFamily: mono, fontSize: 13, color: T.dim, marginLeft: 22, marginTop: 2 }}>{b.sub}</div>}
      {active && (vars.length > 0 || (showCode && b.code)) && (
        <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
          {vars.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {vars.map(([k, v, tone]) => (
                <span key={k} style={{ fontFamily: mono, fontSize: 14, padding: "3px 8px", borderRadius: 5, background: T.bg, border: `1px solid ${tone === "bad" ? T.bad : T.rule}`, color: tone === "bad" ? T.bad : T.ink }}>
                  <span style={{ color: T.dim }}>{k} </span>{v}
                </span>
              ))}
            </div>
          )}
          {showCode && b.code && (
            <div style={{ padding: "6px 10px", background: T.bg, border: `1px solid ${T.rule}`, borderRadius: 6, fontFamily: mono, fontSize: 14, lineHeight: 1.6, color: T.ink }}>
              {b.code.map((l, i) => <CodeLine key={i} text={l} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Flow({ title, blocks, step, showCode, setShowCode }) {
  const nodes = blocks.flatMap((b) => (b.group ? b.items : [b]));
  const cur = nodes.findIndex((b) => b.id === step.block);
  const stateOf = (b) => { const i = nodes.indexOf(b); return i === cur ? "active" : i < cur ? "done" : "todo"; };
  const node = (b) => <Node b={b} state={stateOf(b)} step={step} showCode={showCode} setShowCode={setShowCode} />;
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14 }}>
      <div style={{ fontFamily: sans, fontSize: 14, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600, marginBottom: 10 }}>{title}</div>
      {blocks.map((b, i) => (
        <div key={b.id ?? b.group}>
          {i > 0 && <Arrow />}
          {b.group ? (
            <div style={{ border: `1px dashed ${T.rule}`, borderRadius: 12, padding: "8px 8px 10px", background: `${T.bg}66` }}>
              <div style={{ display: "flex", alignItems: "center", margin: "0 4px 8px", gap: 8 }}>
                <span style={{ fontFamily: mono, fontSize: 13, letterSpacing: "0.1em", color: T.wire, flex: 1 }}>{b.group.toUpperCase()}</span>
                {b.loop && <span style={{ fontFamily: mono, fontSize: 13, color: T.wire, border: `1px solid ${T.wire}66`, borderRadius: 10, padding: "1px 7px" }}>{b.loop}</span>}
              </div>
              {b.items.map((it, j) => (
                <div key={it.id}>
                  {j > 0 && <Arrow />}
                  {node(it)}
                </div>
              ))}
            </div>
          ) : node(b)}
        </div>
      ))}
    </div>
  );
}

export default function StepShell({ title, subtitle, blocks, steps, scene, tags, controls, matmuls }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const last = steps.length - 1;
  const step = steps[Math.min(idx, last)];
  const go = (i) => setIdx(Math.max(0, Math.min(last, i)));

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") go(idx + 1);
      if (e.key === "ArrowLeft") go(idx - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    if (!playing) return;
    if (idx >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx(idx + 1), 2600);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{title}</div>
          <div style={{ fontSize: 13, color: T.dim, marginTop: 2 }}>{subtitle}</div>
        </div>
        <div style={{ flex: 1 }} />
        {controls}
      </header>

      <div className="vit-grid">
        <Flow title={title} blocks={blocks} step={step} showCode={showCode} setShowCode={setShowCode} />
        <div style={{ display: "grid", gap: 12, alignContent: "start", minWidth: 0 }}>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <button onClick={() => go(idx - 1)} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev</button>
            <button onClick={() => go(idx + 1)} disabled={idx === last} style={btn(true, idx === last)}>next ▶</button>
            <button onClick={() => { if (!playing && idx >= last) setIdx(0); setPlaying(!playing); }} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
            <div style={{ flex: 1, minWidth: 220 }}>
              <div style={{ fontSize: 21, fontWeight: 600, color: T.ink, fontFamily: sans }}>{step.caption}</div>
              <div style={{ display: "flex", gap: 3, marginTop: 8 }}>
                {steps.map((s, k) => (
                  <button key={s.id} onClick={() => go(k)} title={s.id} aria-label={s.id}
                    style={{ flex: "1 1 0", minWidth: 0, padding: 0, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: k === idx ? T.accent : k < idx ? `${T.accent}66` : T.rule }} />
                ))}
              </div>
            </div>
            <span style={{ fontSize: 14, color: T.dim }}>step {idx + 1} / {steps.length} · ← →</span>
          </div>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12, minWidth: 0, display: "grid", gap: 10 }}>
            {tags && <TagStrip rows={tags(step)} />}
            {scene(step, idx)}
          </div>
          {matmuls && matmuls(step)}
        </div>
      </div>
    </div>
  );
}

export function Seg({ label, value, options, onChange, fmt = (v) => v, disabled = [], hint }) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, letterSpacing: "0.08em" }}>{label}</span>
      <div style={{ display: "flex", gap: 4 }}>
        {options.map((o) => (
          <button key={o} onClick={() => !disabled.includes(o) && onChange(o)} disabled={disabled.includes(o)} title={disabled.includes(o) ? hint : undefined}
            style={{ ...btn(o === value, disabled.includes(o)), height: 28, fontSize: 15, padding: "0 10px", textDecoration: disabled.includes(o) ? "line-through" : "none" }}>{fmt(o)}</button>
        ))}
      </div>
    </div>
  );
}

export function Stat({ k, v, tone }) {
  return (
    <div style={{ display: "grid", gap: 2, padding: "4px 10px", borderLeft: `2px solid ${tone ?? T.rule}` }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>{k}</span>
      <span style={{ fontFamily: mono, fontSize: 17, fontWeight: 700, color: tone ?? T.ink }}>{v}</span>
    </div>
  );
}
