import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";
import { ShapeCard, CodeLine } from "./shapes";

// Line-by-line prev/next shell. The flowchart on the left enlarges the running
// block and lists that block's code; ◀ ▶ move ONE LINE at a time and the line
// being executed is highlighted. The canvas on the right only changes when the
// line does something visual (a line that merely names a value leaves it alone).
//   blocks: [{ id, label, sub, loop?, code: string[] } | { group, items: [...] }]
//   steps:  [{ id, block, lines: [index into block.code], visAt?, caption, vars? }]
//           (or pass `flat`, a ready-made list of line items, e.g. unrolled loops)
//   scene:  (item, idx) => node        header: (item) => node        shapes: (item) => rows

// One step -> one item per code line. Lines before `visAt` still show the
// previous step's picture; the step's own picture appears when that line runs.
export function expand(steps) {
  const out = [];
  steps.forEach((st, i) => {
    const lines = st.lines ?? [];
    const n = Math.max(lines.length, 1);
    const at = st.visAt ?? n - 1;
    for (let j = 0; j < n; j++) {
      const vis = j < at && i > 0 ? steps[i - 1] : st;
      out.push({ ...vis, block: st.block, caption: st.caption, vars: st.vars, pick: st.pick, tensors: st.tensors, sid: st.id, line: lines.length ? lines[j] : null, first: j === 0 });
    }
  });
  return out;
}

const Arrow = () => (
  <div style={{ display: "grid", justifyItems: "center", height: 14, color: T.rule }}>
    <svg width="10" height="14" viewBox="0 0 10 14"><path d="M5 0 V11 M1 7 L5 12 L9 7" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
  </div>
);

function Node({ b, state, item }) {
  const active = state === "active";
  const vars = active && item.vars ? item.vars() : [];
  return (
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
        {b.loop && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire, border: `1px solid ${T.wire}66`, borderRadius: 10, padding: "1px 7px" }}>{b.loop === true ? "↻ steps" : b.loop}</span>}
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
          {b.code && (
            <div style={{ margin: 0, padding: "6px 0", background: T.bg, border: `1px solid ${T.rule}`, borderRadius: 6, fontFamily: mono, fontSize: 11, lineHeight: 1.6 }}>
              {b.code.map((l, i) => {
                const on = i === item.line;
                return (
                  <div
                    key={i}
                    style={{
                      padding: "2px 10px", borderLeft: `3px solid ${on ? T.accent : "transparent"}`, background: on ? `${T.accent}1c` : "transparent",
                      color: on ? T.ink : item.line != null && i < item.line ? T.soft : T.dim, transition: "background .2s",
                    }}
                  >
                    <CodeLine text={l} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Panel({ title, blocks, item }) {
  const nodes = blocks.flatMap((b) => (b.group ? b.items : [b]));
  const cur = nodes.findIndex((b) => b.id === item.block);
  const stateOf = (b) => { const i = nodes.indexOf(b); return i === cur ? "active" : i < cur ? "done" : "todo"; };
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14, display: "grid", gap: 0 }}>
      <div style={{ fontFamily: sans, fontSize: 11, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600, marginBottom: 10 }}>{title}</div>
      {blocks.map((b, i) => (
        <div key={b.id ?? b.group}>
          {i > 0 && <Arrow />}
          {b.group ? (
            <div style={{ border: `1px dashed ${T.rule}`, borderRadius: 12, padding: "8px 8px 10px", display: "grid", gap: 0, background: `${T.bg}66` }}>
              <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.1em", color: T.wire, margin: "0 4px 8px" }}>{b.group.toUpperCase()}</div>
              {b.items.map((it, j) => (
                <div key={it.id}>
                  {j > 0 && <Arrow />}
                  <Node b={it} state={stateOf(it)} item={item} />
                </div>
              ))}
            </div>
          ) : (
            <Node b={b} state={stateOf(b)} item={item} />
          )}
        </div>
      ))}
    </div>
  );
}

export default function Stepper({ title, subtitle, blocks, steps, flat, scene, header, legend, shapes, glossary, extra }) {
  const items = flat ?? expand(steps);
  const [i0, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = items.length - 1;
  const idx = Math.min(i0, last);
  const item = items[idx];

  const firsts = items.map((it, i) => (it.first ? i : -1)).filter((i) => i >= 0);
  const prevStep = () => setIdx([...firsts].reverse().find((f) => f < idx) ?? 0);
  const nextStep = () => setIdx(firsts.find((f) => f > idx) ?? last);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") (e.shiftKey ? nextStep : () => setIdx((i) => Math.min(i + 1, last)))();
      if (e.key === "ArrowLeft") (e.shiftKey ? prevStep : () => setIdx((i) => Math.max(i - 1, 0)))();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    if (!playing) return;
    if (idx >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx(idx + 1), 1500);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  const stepNo = firsts.filter((f) => f <= idx).length;

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{title}</div>
          <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>{subtitle}</div>
        </div>
        <div style={{ flex: 1 }} />
        {header && header(item)}
      </header>

      <div className="moe-grid">
        <Panel title={title} blocks={blocks} item={item} />
        <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12 }}>{scene(item, idx)}</div>
          {shapes && <ShapeCard rows={shapes(item)} glossary={typeof glossary === "function" ? glossary(item) : glossary} />}
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <button onClick={prevStep} disabled={idx === 0} title="previous step (⇧←)" style={{ ...btn(false, idx === 0), padding: "0 10px" }}>⏮</button>
            <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev line</button>
            <button onClick={() => setIdx(Math.min(idx + 1, last))} disabled={idx === last} style={btn(true, idx === last)}>next line ▶</button>
            <button onClick={nextStep} disabled={idx === last} title="next step (⇧→)" style={{ ...btn(false, idx === last), padding: "0 10px" }}>⏭</button>
            <button onClick={() => setPlaying(!playing)} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
            <div style={{ flex: 1, minWidth: 220, overflow: "hidden" }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{item.caption}</div>
              <div style={{ display: "flex", gap: 3, marginTop: 8, minWidth: 0 }}>
                {firsts.map((f, k) => {
                  const end = firsts[k + 1] ?? last + 1;
                  const cur = idx >= f && idx < end;
                  return <button key={f} onClick={() => setIdx(f)} title={items[f].sid} aria-label={items[f].sid} style={{ flex: "1 1 0", minWidth: 0, padding: 0, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: cur ? T.accent : idx >= end ? `${T.accent}66` : T.rule }} />;
                })}
              </div>
            </div>
            <span style={{ fontSize: 11, color: T.dim }}>line {idx + 1} / {items.length} · step {stepNo} / {firsts.length} · ← → (⇧ = step)</span>
          </div>
          {legend && <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 11, color: T.dim, padding: "0 4px" }}>{typeof legend === "function" ? legend(item) : legend}</div>}
          {extra && extra(item)}
        </div>
      </div>
    </div>
  );
}
