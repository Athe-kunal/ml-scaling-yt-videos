import { useEffect, useState } from "react";
import { T, mono } from "../theme";
import { navBtn } from "../styles";
import { FNS } from "../flow/dsl";
import { MODS } from "../flow/world";

// Debugger-style stepping over the recorded events.
//   into  = the very next event (descends into calls)
//   over  = next event of this thread at the same depth or shallower
//   out   = next event of this thread shallower than now (finish this function)
export function useDebugger(events) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(650);
  const last = events.length - 1;
  const ev = events[Math.min(i, last)];

  const find = (from, dir, ok) => {
    for (let j = from + dir; j >= 0 && j <= last; j += dir) if (ok(events[j])) return j;
    return -1;
  };
  const same = (e) => e.thread === ev.thread;
  const go = {
    into: () => Math.min(i + 1, last),
    over: () => { const j = find(i, 1, (e) => same(e) && e.depth <= ev.depth); return j < 0 ? Math.min(i + 1, last) : j; },
    out: () => { const j = find(i, 1, (e) => same(e) && e.depth < ev.depth); return j < 0 ? last : j; },
    back: () => Math.max(i - 1, 0),
    backOver: () => { const j = find(i, -1, (e) => same(e) && e.depth <= ev.depth); return j < 0 ? 0 : j; },
    nextStep: () => { const j = find(i, 1, (e) => e.group > ev.group); return j < 0 ? last : j; },
    prevStep: () => {
      const start = events.findIndex((e) => e.group === ev.group);
      if (start < i) return start;
      const j = find(i, -1, (e) => e.group < ev.group);
      return j < 0 ? 0 : events.findIndex((e) => e.group === events[j].group);
    },
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      const k = e.key;
      let to = null;
      if (k === "ArrowRight") to = e.shiftKey ? go.over() : go.into();
      else if (k === "ArrowLeft") to = e.shiftKey ? go.backOver() : go.back();
      else if (k === "ArrowUp") to = go.out();
      else if (k === "]" || k === "PageDown") to = go.nextStep();
      else if (k === "[" || k === "PageUp") to = go.prevStep();
      else if (k === " ") { e.preventDefault(); setPlaying((p) => !p); return; }
      if (to !== null) { e.preventDefault(); setI(to); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const isPlaying = playing && i < last;
  useEffect(() => {
    if (!isPlaying) return;
    const t = setTimeout(() => setI((x) => Math.min(x + 1, last)), speed);
    return () => clearTimeout(t);
  }, [isPlaying, i, speed, last]);

  return { i: Math.min(i, last), setI, ev, go, playing: isPlaying, setPlaying, speed, setSpeed, last };
}

const Btn = ({ onClick, disabled, children, title, accent }) => (
  <button onClick={onClick} disabled={disabled} title={title}
    style={{ ...navBtn(disabled), height: 30, padding: "0 10px", fontSize: 12, ...(accent ? { color: accent, borderColor: `${accent}88` } : {}) }}>
    {children}
  </button>
);

export function DebugBar({ dbg, events }) {
  const { i, setI, go, playing, setPlaying, speed, setSpeed, last, ev } = dbg;
  const groups = [...new Set(events.map((e) => e.group))];
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
        <Btn onClick={() => setI(0)} disabled={i === 0} title="restart">⟲</Btn>
        <Btn onClick={() => setI(go.prevStep())} disabled={i === 0} title="previous engine step  [">⇤ step</Btn>
        <Btn onClick={() => setI(go.backOver())} disabled={i === 0} title="back over  shift+←">⤺ back over</Btn>
        <Btn onClick={() => setI(go.back())} disabled={i === 0} title="back  ←">← back</Btn>
        <Btn onClick={() => setPlaying(!playing)} accent={T.accent} title="space">{playing ? "❚❚ pause" : "▶ play"}</Btn>
        <Btn onClick={() => setI(go.into())} disabled={i === last} accent="#FFC94D" title="step into  →">↘ into</Btn>
        <Btn onClick={() => setI(go.over())} disabled={i === last} accent="#6EF3DD" title="step over  shift+→">⤼ over</Btn>
        <Btn onClick={() => setI(go.out())} disabled={i === last || ev.depth <= 1} accent="#FF8AC6" title="step out  ↑">↗ out</Btn>
        <Btn onClick={() => setI(go.nextStep())} disabled={i === last} title="next engine step  ]">step ⇥</Btn>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)}
          style={{ fontFamily: mono, fontSize: 11, height: 30, background: T.well, color: T.soft, border: `1px solid ${T.rule}`, borderRadius: 6 }}>
          <option value={1200}>slow</option><option value={650}>normal</option><option value={250}>fast</option><option value={60}>very fast</option>
        </select>
        <span style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginLeft: "auto" }}>{i + 1} / {events.length}</span>
      </div>
      <div style={{ display: "flex", gap: 3 }}>
        {groups.map((g) => {
          const idx = events.map((e, k) => [e, k]).filter(([e]) => e.group === g);
          const a = idx[0][1], b = idx[idx.length - 1][1];
          const cur = ev.group === g;
          const pct = cur ? ((i - a) / Math.max(1, b - a)) * 100 : i > b ? 100 : 0;
          return (
            <button key={g} onClick={() => setI(a)} title={g === 0 ? "startup" : `engine step ${g}`}
              style={{ flex: Math.max(4, Math.sqrt(idx.length)), minWidth: 0, border: "none", padding: 0, background: "none", cursor: "pointer", display: "grid", gap: 2 }}>
              <span style={{ fontFamily: mono, fontSize: 9, color: cur ? T.ink : T.dim, fontWeight: cur ? 700 : 400, textAlign: "left", whiteSpace: "nowrap", overflow: "hidden" }}>{g === 0 ? "startup" : `step ${g}`}</span>
              <span style={{ height: 6, borderRadius: 3, background: T.rule, overflow: "hidden", outline: cur ? `1px solid ${T.ink}` : "none" }}>
                <span style={{ display: "block", height: "100%", width: `${pct}%`, background: T.accent }} />
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// where we are: thread + call stack breadcrumb + the current line's note
export function NowCard({ ev }) {
  const t = ev.threads.find((x) => x.name === ev.thread);
  const top = t.frames[t.frames.length - 1];
  const mod = MODS[FNS[top.fn].mod];
  return (
    <div style={{ background: T.panel, border: `1px solid ${mod.color}88`, borderRadius: 10, padding: "9px 12px", display: "grid", gap: 6 }}>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginRight: 4 }}>{ev.thread}</span>
        {t.frames.map((f, k) => {
          const c = MODS[FNS[f.fn].mod].color;
          const last = k === t.frames.length - 1;
          return (
            <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              {k > 0 && <span style={{ color: T.rule, fontSize: 10 }}>›</span>}
              <span style={{ fontFamily: mono, fontSize: last ? 11.5 : 10, color: last ? T.bg : c, background: last ? c : "transparent", borderRadius: 4, padding: last ? "1px 6px" : 0, fontWeight: last ? 700 : 400 }}>
                {f.fn.split(".").pop()}
              </span>
            </span>
          );
        })}
      </div>
      {ev.note && <div key={ev.node + ev.note} className="title-in" style={{ fontFamily: mono, fontSize: 13, color: T.ink }}>{ev.note}</div>}
    </div>
  );
}
