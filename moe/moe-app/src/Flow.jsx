import { T, mono, sans } from "./theme";
import { FLOW, STEPS } from "./steps";

// Flowchart from main: the running block is enlarged, finished blocks are
// ticked, and live variables sit inside the active node. Code lines are
// hidden until the viewer opens them.

const blockState = (id, idx) => {
  const own = STEPS.map((s, i) => [s, i]).filter(([s]) => s.block === id).map(([, i]) => i);
  if (own.includes(idx)) return "active";
  return Math.max(...own) < idx ? "done" : "todo";
};

const Arrow = () => (
  <div style={{ display: "grid", justifyItems: "center", height: 14, color: T.rule }}>
    <svg width="10" height="14" viewBox="0 0 10 14"><path d="M5 0 V11 M1 7 L5 12 L9 7" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
  </div>
);

function Node({ b, state, step, gm, showCode, setShowCode }) {
  const active = state === "active";
  const col = active ? T.accent : state === "done" ? T.rule : T.rule;
  return (
    <div
      style={{
        border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}10` : T.well, borderRadius: active ? 10 : 7,
        padding: active ? "12px 14px" : "6px 10px", opacity: state === "todo" ? 0.5 : 1, transition: "all .3s",
        boxShadow: active ? `0 0 0 3px ${T.accent}14` : "none",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 14, height: 14, borderRadius: 7, border: `1.5px solid ${state === "done" ? T.accent : col}`, background: state === "done" ? T.accent : "transparent", flexShrink: 0, display: "grid", placeItems: "center", fontSize: 9, color: T.bg, fontWeight: 700 }}>
          {state === "done" ? "✓" : ""}
        </span>
        <span style={{ fontFamily: mono, fontSize: active ? 14 : 11.5, fontWeight: 700, color: active ? T.accent : T.soft, flex: 1 }}>{b.label}</span>
        {b.loop && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire, border: `1px solid ${T.wire}66`, borderRadius: 10, padding: "1px 7px" }}>↻ ×S</span>}
      </div>
      {(active || state !== "todo") && <div style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginLeft: 22, marginTop: 2 }}>{b.sub}</div>}

      {active && (
        <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {step.vars(gm).map(([k, v, tone]) => (
              <span key={k} style={{ fontFamily: mono, fontSize: 11.5, padding: "3px 8px", borderRadius: 5, background: T.bg, border: `1px solid ${tone === "bad" ? T.bad : T.rule}`, color: tone === "bad" ? T.bad : T.ink }}>
                <span style={{ color: T.dim }}>{k} </span>{v}
              </span>
            ))}
          </div>
          <button
            onClick={() => setShowCode(!showCode)}
            style={{ justifySelf: "start", fontFamily: mono, fontSize: 10.5, padding: "3px 9px", borderRadius: 5, border: `1px solid ${T.rule}`, background: "transparent", color: T.soft, cursor: "pointer" }}
          >
            {showCode ? "hide code ▴" : "‹/› show code ▾"}
          </button>
          {showCode && (
            <pre style={{ margin: 0, padding: "8px 10px", background: T.bg, border: `1px solid ${T.rule}`, borderRadius: 6, fontFamily: mono, fontSize: 11, lineHeight: 1.65, color: T.ink, overflowX: "auto" }}>
              {step.code.map((l) => (
                <div key={l.n}>
                  <span style={{ color: T.dim, display: "inline-block", width: 24 }}>{String(l.n).startsWith("s") ? "" : l.n}</span>{l.t}
                </div>
              ))}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Flow({ idx, gm, showCode, setShowCode }) {
  const step = STEPS[idx];
  const nodeProps = (b) => ({ b, state: blockState(b.id, idx), step, gm, showCode, setShowCode });
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14, display: "grid", gap: 0 }}>
      <div style={{ fontFamily: sans, fontSize: 11, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600, marginBottom: 10 }}>
        moe_layer(x)
      </div>
      {FLOW.map((b, i) => (
        <div key={b.id ?? b.group}>
          {i > 0 && <Arrow />}
          {b.group ? (
            <div style={{ border: `1px dashed ${T.rule}`, borderRadius: 12, padding: "8px 8px 10px", display: "grid", gap: 0, background: `${T.bg}66` }}>
              <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.1em", color: T.wire, margin: "0 4px 8px" }}>{b.group.toUpperCase()}</div>
              {b.items.map((it, j) => (
                <div key={it.id}>
                  {j > 0 && <Arrow />}
                  <Node {...nodeProps(it)} />
                </div>
              ))}
            </div>
          ) : (
            <Node {...nodeProps(b)} />
          )}
        </div>
      ))}
    </div>
  );
}
