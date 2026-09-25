import { useEffect, useState } from "react";
import { T, mono, sans } from "./theme";
import { withInlineMath } from "./latex";
import { STEPS, HEADER } from "./algoSteps";
import { AXIS_COLOR, codeHtml } from "./notation";
import MatrixShapes from "./MatrixShapes";

const KIND = {
  load: { label: "state", color: T.dim },
  compute: { label: "compute", color: T.accent },
  comm: { label: "comms", color: T.wire },
  reshape: { label: "reshape", color: T.soft },
};

const navBtn = (disabled) => ({
  fontFamily: mono, fontSize: 15, width: 34, height: 30, borderRadius: 6, flexShrink: 0,
  border: `1px solid ${T.rule}`, background: disabled ? "transparent" : T.well,
  color: disabled ? T.rule : T.ink, cursor: disabled ? "default" : "pointer",
});
const chip = (c) => ({
  display: "inline-block", fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase",
  padding: "3px 9px", borderRadius: 5, color: c, border: `1px solid ${c}55`, background: `${c}14`,
});
const panel = { background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 18 };
const Rich = ({ html, style }) => <div style={style} dangerouslySetInnerHTML={{ __html: withInlineMath(html) }} />;

export default function AlgorithmSteps() {
  const [idx, setIdx] = useState(0);
  const last = STEPS.length - 1;
  const s = STEPS[idx];
  const kind = KIND[s.kind];

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setIdx((i) => Math.min(i + 1, last));
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "28px 20px 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 20 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SCALING BOOK · INFERENCE</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Sharded attention, line by line</h1>
        <Rich style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 820 }} html={HEADER.intro} />
        <div style={{ display: "flex", gap: 14, marginTop: 12, fontFamily: mono, fontSize: 12, color: T.soft }}>
          {[["Y", "Y"], ["Z", "Z"], ["YZ", "Y and Z"]].map(([k, l]) => (
            <span key={k}><span style={{ color: AXIS_COLOR[k], fontWeight: 700 }}>■</span> sharded over {l}</span>
          ))}
        </div>
      </header>

      <div style={{ display: "grid", gap: 18 }}>
        {/* pseudocode listing */}
        <div style={{ ...panel, padding: "14px 10px" }}>
          <ol style={{ listStyle: "none", margin: 0, padding: 0, fontFamily: mono, fontSize: 13.5 }}>
            {STEPS.map((st, i) => {
              const active = i === idx;
              return (
                <li
                  key={i}
                  onClick={() => setIdx(i)}
                  style={{
                    display: "flex", gap: 12, padding: "5px 10px", borderRadius: 6, cursor: "pointer",
                    borderLeft: `3px solid ${active ? KIND[st.kind].color : "transparent"}`,
                    background: active ? `${KIND[st.kind].color}18` : "transparent",
                    color: active ? T.ink : i < idx ? T.soft : T.dim,
                    transition: "all .2s",
                  }}
                >
                  <span style={{ width: 22, textAlign: "right", color: active ? KIND[st.kind].color : T.rule, flexShrink: 0 }}>{i + 1}.</span>
                  <span dangerouslySetInnerHTML={{ __html: codeHtml(st.code) }} />
                </li>
              );
            })}
          </ol>
        </div>

        {/* shapes & sharding strip */}
        <div style={panel}>
          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.2, color: T.dim, textTransform: "uppercase", marginBottom: 12 }}>Shapes &amp; sharding at this line</div>
          <MatrixShapes row={s.row} accent={kind.color} />
          {s.comm && <div style={{ marginTop: 10 }}><span style={chip(T.wire)}>{s.comm}</span></div>}
        </div>

        {/* step panel */}
        <div style={panel}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} aria-label="Previous line" style={navBtn(idx === 0)}>←</button>
            <div style={{ display: "flex", gap: 4, flex: 1, justifyContent: "center", flexWrap: "wrap" }}>
              {STEPS.map((_, i) => (
                <button
                  key={i} onClick={() => setIdx(i)} aria-label={`Line ${i + 1}`}
                  style={{ width: i === idx ? 20 : 7, height: 7, borderRadius: 4, border: "none", padding: 0, cursor: "pointer",
                    background: i === idx ? T.accent : i < idx ? T.dim : T.rule, transition: "all .25s cubic-bezier(.2,.7,.3,1)" }}
                />
              ))}
            </div>
            <button onClick={() => setIdx(idx + 1)} disabled={idx === last} aria-label="Next line" style={navBtn(idx === last)}>→</button>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={chip(T.accent)}>line {idx + 1} / {STEPS.length}</span>
            <span style={chip(kind.color)}>{kind.label}</span>
          </div>

          <div style={{ fontSize: 19, fontWeight: 700, marginBottom: 8 }}>{s.title}</div>
          <Rich style={{ fontSize: 14, lineHeight: 1.7, color: T.soft, marginBottom: 12 }} html={s.body} />

          {s.perChip && (
            <Rich
              style={{ fontSize: 12.5, lineHeight: 1.6, color: T.accent, background: `${T.accent}0f`, border: `1px solid ${T.accent}40`, borderRadius: 8, padding: "7px 10px", marginBottom: 12 }}
              html={`<span style="font-family:${mono};font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;opacity:.85;margin-right:8px">Per chip</span>${s.perChip}`}
            />
          )}
          {s.note && (
            <Rich style={{ fontSize: 12.5, lineHeight: 1.55, color: T.dim, borderLeft: `2px solid ${T.wire}`, paddingLeft: 10 }} html={s.note} />
          )}
          {idx === last && <Rich style={{ fontSize: 13.5, lineHeight: 1.6, color: T.soft, marginTop: 14 }} html={HEADER.outro} />}
        </div>
      </div>
    </div>
  );
}
