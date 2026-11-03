import { useEffect, useState } from "react";
import { T, mono, sans, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";

// All-to-all as a stepper. Device s holds D chunks; chunk t is addressed to
// device t. After the collective, device t holds the chunks from every s:
// the D×D chunk grid is transposed. Chunk (s→t) is drawn at device s / slot t
// before it moves and at device t / slot s after it arrives. Schedule shown:
// D-1 pairwise rounds, round r sends to (d+r) mod D, one chunk per link.

const D = CFG.G;
const W = 860, H = 350;
const DX = (d) => 20 + d * 210, DW = 190, BY = 92, BH = 215;
const slotCX = (d, j) => DX(d) + 10 + j * 44 + 20;
const CY1 = BY + 84, CY2 = BY + 176; // send row, receive row
const mod = (a, n) => ((a % n) + n) % n;

// tokens group s sends to expert t (every chunk is padded to C slots)
const COUNT = Array.from({ length: D }, (_, s) => {
  const row = Array(D).fill(0);
  MODEL.groups[s].toks.forEach((k) => k.final.forEach((f) => (row[f.e] += 1)));
  return row;
});

const BLOCKS = [
  { id: "start", label: "x[d] = D chunks", sub: "chunk t is for device t", code: ["# every device d holds", "x[d] = [chunk_0, …, chunk_D-1]"] },
  { id: "keep", label: "keep own chunk", sub: "no network", code: ["out[d][d] = x[d][d]"] },
  {
    id: "loop", label: "for r in 1..D-1", sub: "one chunk per link", loop: true,
    code: ["for r in 1 .. D-1:", "  send x[d][(d+r) % D]  → device (d+r) % D", "  recv from device (d-r) % D → out[d][(d-r) % D]"],
  },
  { id: "done", label: "rows → columns", sub: "a transpose of the chunk grid", code: ["# one library call does all of it:", "dist.all_to_all_single(out, x)", "# out[j] = x_on_device_j[d]"] },
  { id: "moe", label: "in MoE", sub: "chunk = expert buffer slice", code: ["# chunk s→e = [C, M] tokens of group s", "# for expert e, padded to C slots"] },
];

const STEPS = [
  { id: "start", block: "start", round: -2, caption: "each device holds one chunk per destination" },
  { id: "addr", block: "start", round: -2, fan: true, caption: "chunk t is addressed to device t" },
  { id: "keep", block: "keep", round: 0, caption: "chunk d→d is already home" },
  { id: "r1", block: "loop", round: 1, caption: "round 1 · everyone sends to d+1" },
  { id: "r2", block: "loop", round: 2, caption: "round 2 · everyone sends to d+2" },
  { id: "r3", block: "loop", round: 3, caption: "round 3 · everyone sends to d+3" },
  { id: "done", block: "done", round: 3, caption: "rows became columns: a transpose" },
  { id: "moe", block: "moe", round: 3, detail: true, caption: "in MoE: chunk s→e = group s's tokens for expert e" },
];

function Chunk({ s, t, round, detail, sel, moving }) {
  const r = mod(t - s, D);
  const arrived = r <= round;
  const dev = arrived ? t : s;
  const slot = arrived ? s : t;
  const x = slotCX(dev, slot);
  const home = s === t;
  const c = EC[t];
  return (
    <g
      style={{
        transform: `translate(${x}px, ${arrived ? CY2 : CY1}px)`,
        transition: `transform .9s cubic-bezier(.3,.7,.3,1) ${(s * 60) % 240}ms`,
      }}
    >
      <rect
        x={-20} y={-22} width={40} height={44} rx={7} fill={`${c}${detail ? "1c" : "55"}`} stroke={moving ? T.ink : c}
        strokeWidth={moving ? 2.6 : home && round === 0 ? 2.4 : 1.4} opacity={sel === s || sel == null ? 1 : 0.6}
      />
      <text y={detail ? -8 : 4} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={T.ink}>{s}→{t}</text>
      {detail &&
        Array.from({ length: CFG.C }, (_, k) => (
          <rect key={k} x={-16 + k * 11} y={2} width={9} height={14} rx={2} fill={k < Math.min(COUNT[s][t], CFG.C) ? c : "transparent"} stroke={c} strokeDasharray={k < COUNT[s][t] ? "none" : "2 2"} />
        ))}
    </g>
  );
}

function Scene({ step, sel }) {
  const { round, fan, detail } = step;
  const done = round >= D - 1;
  const arcs = [];
  if (fan) for (let t = 0; t < D; t++) if (t !== sel) arcs.push([sel, t]);
  if (round >= 1 && !done) for (let s = 0; s < D; s++) arcs.push([s, mod(s + round, D)]);
  if (round === D - 1 && step.id === "r3") for (let s = 0; s < D; s++) arcs.push([s, mod(s + round, D)]);
  const chunks = [];
  for (let s = 0; s < D; s++) for (let t = 0; t < D; t++) chunks.push([s, t]);
  const movingR = step.block === "loop" ? round : -1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        <marker id="a2a-arrow" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={T.wire} />
        </marker>
      </defs>

      {Array.from({ length: D }, (_, d) => (
        <g key={d}>
          <rect x={DX(d)} y={BY} width={DW} height={BH} rx={10} fill={T.panel} stroke={d === sel ? T.soft : T.rule} strokeWidth={d === sel ? 1.8 : 1.2} />
          <text x={DX(d) + 12} y={BY + 22} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.ink}>Device {d}</text>
          {[[CY1, "send x[j]  → device j"], [CY2, "recv out[j]  ← device j"]].map(([cy, label]) => (
            <g key={cy}>
              <text x={DX(d) + 10} y={cy - 32} fontFamily={mono} fontSize="9.5" fill={cy === CY2 && done ? T.accent : T.dim}>{label}</text>
              {Array.from({ length: D }, (_, j) => (
                <rect key={j} x={DX(d) + 10 + j * 44} y={cy - 24} width={40} height={48} rx={7} fill="none" stroke={T.rule} strokeDasharray="3 3" />
              ))}
            </g>
          ))}
        </g>
      ))}

      {arcs.map(([s, t], i) => {
        const x0 = DX(s) + DW / 2, x1 = DX(t) + DW / 2;
        const h = 28 + Math.abs(t - s) * 16;
        return (
          <path
            key={i} d={`M${x0},${BY - 4} Q${(x0 + x1) / 2},${BY - 4 - h * 1.6} ${x1},${BY - 4}`} fill="none" stroke={T.wire} strokeWidth={2}
            opacity={0.8} className="lane" markerEnd="url(#a2a-arrow)"
          />
        );
      })}

      {chunks.map(([s, t]) => (
        <Chunk key={`${s}-${t}`} s={s} t={t} round={round} detail={detail} sel={fan ? sel : null} moving={mod(t - s, D) === movingR || (fan && s === sel && t !== s)} />
      ))}

      {step.block === "loop" && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          each device sends 1 chunk and receives 1 chunk — no link is shared
        </text>
      )}
    </svg>
  );
}

function Panel({ idx, showCode, setShowCode }) {
  const step = STEPS[idx];
  const order = BLOCKS.map((b) => b.id);
  const cur = order.indexOf(step.block);
  const vars = () => {
    if (step.block === "loop") return [["r", `${step.round}`], ["d sends to", Array.from({ length: D }, (_, d) => `${d}→${mod(d + step.round, D)}`).join(" ")]];
    if (step.block === "keep") return [["stays", Array.from({ length: D }, (_, d) => `${d}→${d}`).join(" ")]];
    if (step.block === "start") return [["D", `${D}`], ["chunks / device", `${D}`]];
    if (step.block === "done") return [["moved", `${D * (D - 1)} chunks`], ["kept", `${D}`]];
    return [["C", `${CFG.C}`], ["unused slots", `${D * D * CFG.C - COUNT.flat().reduce((a, b) => a + b, 0)} (padding)`]];
  };
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 14, display: "grid", gap: 0 }}>
      <div style={{ fontFamily: sans, fontSize: 11, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600, marginBottom: 10 }}>all_to_all(x)</div>
      {BLOCKS.map((b, i) => {
        const active = i === cur;
        const state = active ? "active" : i < cur ? "done" : "todo";
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
                {b.loop && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire, border: `1px solid ${T.wire}66`, borderRadius: 10, padding: "1px 7px" }}>↻ ×{D - 1}</span>}
              </div>
              {state !== "todo" && <div style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginLeft: 22, marginTop: 2 }}>{b.sub}</div>}
              {active && (
                <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {vars().map(([k, v]) => (
                      <span key={k} style={{ fontFamily: mono, fontSize: 11.5, padding: "3px 8px", borderRadius: 5, background: T.bg, border: `1px solid ${T.rule}`, color: T.ink }}>
                        <span style={{ color: T.dim }}>{k} </span>{v}
                      </span>
                    ))}
                  </div>
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

export default function A2AView() {
  const [idx, setIdx] = useState(0);
  const [sel, setSel] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [playing, setPlaying] = useState(false);
  const last = STEPS.length - 1;
  const step = STEPS[idx];

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

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "16px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 14 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>All-to-all collective</div>
          <div style={{ fontSize: 10.5, color: T.dim, marginTop: 2 }}>D={D} devices · every device sends a different chunk to every device</div>
        </div>
        <div style={{ flex: 1 }} />
        {step.fan && (
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow device</span>
            {Array.from({ length: D }, (_, k) => (
              <button key={k} onClick={() => setSel(k)} style={{ ...btn(k === sel, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
        )}
      </header>

      <div className="moe-grid">
        <Panel idx={idx} showCode={showCode} setShowCode={setShowCode} />
        <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: 12 }}>
            <Scene step={step} sel={sel} />
          </div>
          <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={idx === 0} style={btn(false, idx === 0)}>◀ prev</button>
            <button onClick={() => setIdx(Math.min(idx + 1, last))} disabled={idx === last} style={btn(true, idx === last)}>next ▶</button>
            <button onClick={() => setPlaying(!playing)} style={btn(playing, false)}>{playing ? "❚❚" : "▶ play"}</button>
            <div style={{ flex: 1, minWidth: 220 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: T.ink }}>{step.caption}</div>
              <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
                {STEPS.map((s, i) => (
                  <button key={s.id} onClick={() => setIdx(i)} title={s.id} aria-label={s.id} style={{ flex: 1, height: 6, borderRadius: 3, border: 0, cursor: "pointer", background: i === idx ? T.accent : i < idx ? `${T.accent}66` : T.rule }} />
                ))}
              </div>
            </div>
            <span style={{ fontSize: 11, color: T.dim }}>{idx + 1} / {STEPS.length} · ← →</span>
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 11, color: T.dim, padding: "0 4px" }}>
            <span>chunk <b style={{ color: T.soft }}>s→t</b> = from device s, for device t</span>
            {EC.map((c, e) => <span key={e} style={{ color: c }}>■ for device {e}</span>)}
          </div>
        </div>
      </div>
    </div>
  );
}
