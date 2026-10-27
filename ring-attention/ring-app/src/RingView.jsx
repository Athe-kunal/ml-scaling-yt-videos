import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS, PHASE, buildPhases, mod } from "./ringModel";
import { COPY, codeFocus, codeNotes } from "./ringLogic";
import { CodeBlock } from "./CodePanel";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich } from "./ui";
import RingDiagram, { OutLegend } from "./RingDiagram";
import AttentionMatrix, { OutLseKey, SequenceStrip } from "./AttentionMatrix";

// Rank switcher shown in the code panel header: the comments follow this rank.
export function RankPicker({ N, sel, onSel }) {
  return (
    <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
      <span style={{ marginRight: 4 }}>comments for rank</span>
      {Array.from({ length: N }, (_, r) => (
        <button
          key={r} onClick={() => onSel(r)} aria-label={`Show rank ${r}`}
          style={{ ...navBtn(false), height: 22, padding: "0 8px", fontSize: 11, color: DEV_COLORS[r],
            borderColor: r === sel ? DEV_COLORS[r] : T.rule, background: r === sel ? `${DEV_COLORS[r]}22` : T.well, fontWeight: r === sel ? 700 : 400 }}
        >
          {r}
        </button>
      ))}
    </span>
  );
}

export default function RingView({ variant }) {
  const [N, setN] = useState(4);
  const [sel, setSel] = useState(2);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const phases = useMemo(() => buildPhases(N), [N]);
  const last = phases.length - 1;
  const phase = phases[Math.min(idx, last)];
  const copy = COPY[variant];

  const changeN = (n) => {
    setN(n);
    setIdx(0);
    setSel((r) => Math.min(r, n - 1));
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setIdx((i) => Math.min(i + 1, last));
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(i - 1, 0));
      if (e.key === "ArrowUp") setSel((r) => mod(r + 1, N));
      if (e.key === "ArrowDown") setSel((r) => mod(r - 1, N));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last, N]);

  useEffect(() => {
    if (!playing || idx >= last) return;
    const t = setTimeout(() => setIdx((i) => i + 1), 1500);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  const focus = codeFocus(variant, N, phase, sel);
  const notes = codeNotes(variant, N, phase, sel);
  const ph = PHASE[phase.type];

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>{copy.eyebrow}</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>{copy.title}</h1>
        <Rich style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 960 }} html={copy.intro} />
      </header>

      {/* controls */}
      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center", marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>GPUs N</span>
          {[2, 3, 4, 6, 8].map((n) => (
            <button key={n} onClick={() => changeN(n)} style={{ ...navBtn(false), height: 28, padding: "0 10px", background: n === N ? T.accent : T.well, color: n === N ? T.bg : T.soft, borderColor: n === N ? T.accent : T.rule }}>{n}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>follow GPU</span>
          {Array.from({ length: N }, (_, r) => (
            <button key={r} onClick={() => setSel(r)} style={{ ...navBtn(false), height: 28, padding: "0 10px", color: DEV_COLORS[r], borderColor: r === sel ? DEV_COLORS[r] : T.rule, background: r === sel ? `${DEV_COLORS[r]}22` : T.well, fontWeight: r === sel ? 700 : 400 }}>{r}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: "1 1 320px", justifyContent: "flex-end" }}>
          <button onClick={() => setIdx(0)} disabled={idx === 0} style={navBtn(idx === 0)} aria-label="Restart">⟲</button>
          <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} style={navBtn(idx === 0)}>← prev</button>
          <button onClick={() => { if (idx >= last) { setIdx(0); setPlaying(true); } else setPlaying(!playing); }} style={{ ...navBtn(false), color: T.accent, borderColor: `${T.accent}88` }}>{playing && idx < last ? "❚❚ pause" : "▶ play"}</button>
          <button onClick={() => setIdx(idx + 1)} disabled={idx === last} style={navBtn(idx === last)}>next →</button>
        </div>
        <div style={{ display: "flex", gap: 3, width: "100%", flexWrap: "wrap" }}>
          {phases.map((p, i) => (
            <button
              key={i} onClick={() => setIdx(i)} aria-label={`Phase ${i + 1}`} title={`${p.type}${p.s >= 0 ? ` · step ${p.s}` : ""}`}
              style={{ flex: 1, minWidth: 8, height: 8, border: "none", padding: 0, borderRadius: 3, cursor: "pointer",
                background: i === idx ? PHASE[p.type].color : i < idx ? `${PHASE[p.type].color}66` : T.rule, outline: i === idx ? `1px solid ${T.ink}` : "none" }}
            />
          ))}
        </div>
      </div>

      <div className="ra-grid">
        {/* ---------- visuals ---------- */}
        <main style={{ display: "grid", gap: 18, minWidth: 0 }}>
          <section style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
              <div style={h2}>The ring · click a GPU to follow it</div>
              <div style={{ display: "flex", gap: 6 }}>
                {phase.s >= 0 && phase.type !== "final" && <span style={chip(T.accent)}>step {phase.s}</span>}
                <span style={chip(ph.color)}>{ph.label}</span>
              </div>
            </div>
            <RingDiagram variant={variant} N={N} phase={phase} sel={sel} onSel={setSel} />
            <OutLegend variant={variant} N={N} phase={phase} sel={sel} />
          </section>

          <section style={panel}>
            <div style={h2}>Causal attention matrix</div>
            <SequenceStrip variant={variant} N={N} tokens />
            <div style={{ height: 12 }} />
            <AttentionMatrix variant={variant} N={N} phase={phase} sel={sel} tokens />
            <OutLseKey />
            <Rich
              style={{ fontSize: 12.5, color: T.dim, marginTop: 8, lineHeight: 1.5 }}
              html={copy.matrixNote(N)}
            />
          </section>
        </main>

        {/* ---------- code ---------- */}
        <aside className="ra-side">
          <CodeBlock
            title={copy.src[0]}
            lines={copy.src[1]}
            focus={focus.main}
            notes={notes.main}
            maxHeight="calc(100vh - 90px)"
            note={<RankPicker N={N} sel={sel} onSel={setSel} />}
          />
        </aside>
      </div>
    </div>
  );
}
