import { useEffect, useState } from "react";
import { T, mono } from "../theme";
import { navBtn, chip } from "../styles";
import { KIND } from "../sim/trace";

// ← / → one line, shift+← / shift+→ one group (engine step or demo section).
export function useStepper(events) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = events.length - 1;
  const i = Math.min(idx, last);

  const groupStart = (g) => events.findIndex((e) => e.group === g);
  const nextGroup = () => {
    const k = events.findIndex((e, j) => j > i && e.group !== events[i].group);
    return k < 0 ? last : k;
  };
  const prevGroup = () => {
    const g = events[i].group;
    const start = groupStart(g);
    if (start < i) return start;
    const pg = events[Math.max(0, start - 1)].group;
    return groupStart(pg);
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") { e.preventDefault(); setIdx(e.shiftKey ? nextGroup() : Math.min(i + 1, last)); }
      if (e.key === "ArrowLeft") { e.preventDefault(); setIdx(e.shiftKey ? prevGroup() : Math.max(i - 1, 0)); }
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const isPlaying = playing && i < last;
  useEffect(() => {
    if (!isPlaying) return;
    const t = setTimeout(() => setIdx(i + 1), 900);
    return () => clearTimeout(t);
  }, [isPlaying, i]);

  return { i, setIdx, playing: isPlaying, setPlaying, last, nextGroup, prevGroup, groupStart };
}

export function StepperControls({ events, st, groupLabel }) {
  const { i, setIdx, playing, setPlaying, last, nextGroup, prevGroup, groupStart } = st;
  const ev = events[i];
  const groups = [...new Set(events.map((e) => e.group))];
  return (
    <div style={{ display: "grid", gap: 10, width: "100%" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
        <button onClick={() => setIdx(0)} disabled={i === 0} style={navBtn(i === 0)} aria-label="Restart">⟲</button>
        <button onClick={() => setIdx(prevGroup())} disabled={i === 0} style={navBtn(i === 0)} title="shift + ←">⇤ {groupLabel}</button>
        <button onClick={() => setIdx(i - 1)} disabled={i === 0} style={navBtn(i === 0)}>← prev</button>
        <button onClick={() => { if (i >= last) { setIdx(0); setPlaying(true); } else setPlaying(!playing); }}
          style={{ ...navBtn(false), color: T.accent, borderColor: `${T.accent}88` }}>{playing ? "❚❚ pause" : "▶ play"}</button>
        <button onClick={() => setIdx(i + 1)} disabled={i === last} style={navBtn(i === last)}>next →</button>
        <button onClick={() => setIdx(nextGroup())} disabled={i === last} style={navBtn(i === last)} title="shift + →">{groupLabel} ⇥</button>
        <span style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginLeft: "auto" }}>{i + 1} / {events.length}</span>
      </div>

      {/* one cell per event, a gap between groups */}
      <div style={{ display: "flex", gap: 6 }}>
        {groups.map((g) => {
          const evs = events.map((e, k) => [e, k]).filter(([e]) => e.group === g);
          const cur = ev.group === g;
          return (
            <div key={g} style={{ flex: evs.length, minWidth: 0, display: "grid", gap: 3 }}>
              <button onClick={() => setIdx(groupStart(g))} style={{
                fontFamily: mono, fontSize: 9.5, border: "none", background: "none", padding: 0, cursor: "pointer", textAlign: "left",
                color: cur ? T.ink : T.dim, fontWeight: cur ? 700 : 400, whiteSpace: "nowrap", overflow: "hidden",
              }}>{g}</button>
              <div style={{ display: "flex", gap: 1 }}>
                {evs.map(([e, k]) => (
                  <button key={k} onClick={() => setIdx(k)} aria-label={`event ${k + 1}`} title={e.title}
                    style={{ flex: 1, minWidth: 2, height: 9, border: "none", padding: 0, borderRadius: 2, cursor: "pointer",
                      background: k === i ? e.color : k < i ? `${e.color}70` : T.rule, outline: k === i ? `1px solid ${T.ink}` : "none" }} />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 26 }}>
        <span style={chip(ev.color)}>{KIND[ev.kind].label}</span>
        <span key={i} className="title-in" style={{ fontFamily: mono, fontSize: 14, color: T.ink }}>{ev.title}</span>
      </div>
    </div>
  );
}

export function NumCtl({ label, value, min, max, onChange }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginRight: 2 }}>{label}</span>
      <button onClick={() => onChange(Math.max(min, value - 1))} disabled={value <= min} style={{ ...navBtn(value <= min), height: 26, padding: "0 8px" }}>−</button>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.ink, minWidth: 20, textAlign: "center" }}>{value}</span>
      <button onClick={() => onChange(Math.min(max, value + 1))} disabled={value >= max} style={{ ...navBtn(value >= max), height: 26, padding: "0 8px" }}>+</button>
    </span>
  );
}

export function Choice({ label, value, options, onChange }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginRight: 2 }}>{label}</span>
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} style={{ ...navBtn(false), height: 26, padding: "0 10px",
          background: o === value ? T.accent : T.well, color: o === value ? T.bg : T.soft, borderColor: o === value ? T.accent : T.rule }}>{o}</button>
      ))}
    </span>
  );
}
