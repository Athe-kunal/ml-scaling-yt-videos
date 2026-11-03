import { useEffect, useState } from "react";
import { T, mono, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";
import { STEPS } from "./steps";
import Flow from "./Flow";
import GroupScene from "./GroupScene";
import DevicesScene from "./DevicesScene";
import TensorView from "./TensorView";

export default function GatingView() {
  const [idx, setIdx] = useState(0);
  const [view, setView] = useState("token");
  const [g, setG] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [playing, setPlaying] = useState(false);
  const last = STEPS.length - 1;
  const step = STEPS[idx];
  const gm = MODEL.groups[g];

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
    const t = setTimeout(() => setIdx((i) => i + 1), 2200);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  const canvas = view === "tensor"
    ? <TensorView step={step} gm={gm} />
    : step.scene === "group" ? <GroupScene gm={gm} stage={step.stage} /> : <DevicesScene stage={step.stage} sel={g} />;

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
      <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
        <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>GShard MoE · top-2 gating</div>
            <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>G={CFG.G} groups · S={CFG.S} tokens · E={CFG.E} experts · C={CFG.C}</div>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow group</span>
            {Array.from({ length: CFG.G }, (_, k) => (
              <button key={k} onClick={() => setG(k)} style={{ ...btn(k === g, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <button onClick={() => setView("token")} style={btn(view === "token", false)}>tokens</button>
            <button onClick={() => setView("tensor")} style={btn(view === "tensor", false)}>tensors</button>
          </div>
        </header>

        <div className="moe-grid">
          <Flow idx={idx} gm={gm} showCode={showCode} setShowCode={setShowCode} />

          <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
            <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12 }}>
              {canvas}
            </div>

            <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev</button>
              <button onClick={() => setIdx(Math.min(idx + 1, last))} disabled={idx === last} style={btn(true, idx === last)}>next ▶</button>
              <button onClick={() => setPlaying(!playing)} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{step.caption}</div>
                <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
                  {STEPS.map((s, i) => (
                    <button
                      key={s.id} onClick={() => setIdx(i)} title={s.id} aria-label={s.id}
                      style={{ flex: 1, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: i === idx ? T.accent : i < idx ? `${T.accent}66` : T.rule }}
                    />
                  ))}
                </div>
              </div>
              <span style={{ fontSize: 11, color: T.dim }}>{idx + 1} / {STEPS.length} · ← →</span>
            </div>

            {step.scene === "devices" && view === "token" && (
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 11, color: T.dim, padding: "0 4px" }}>
                <span><b style={{ color: T.soft }}>●</b> 1st choice ticket</span>
                <span><b style={{ color: T.soft }}>○</b> 2nd choice ticket</span>
                {EC.map((c, e) => <span key={e} style={{ color: c }}>■ E{e}</span>)}
                <span style={{ color: T.wire }}>━ lanes of followed group {g}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
