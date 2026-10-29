import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS } from "./ringModel";
import { ownerOf, localTokens } from "./shardModel";
import { panel, h2, navBtn } from "./styles";
import { Rich } from "./ui";

// One sequence of S = 4N tokens, each a Q/K/V column. Stage 0 shows the raw
// sequence, stage 1 colours every token by the GPU it is headed to, stage 2
// flies them into N GPUs on a ring (token 0 first). Switching method while
// distributed re-flies the tokens. Below: the causal matrix of all three
// layouts, each row coloured by the GPU that holds that query.

const METHODS = [
  { id: "ring", label: "Ring", rule: "contiguous: GPU r gets tokens 4r … 4r+3" },
  { id: "zigzag", label: "Zig-zag", rule: "2N chunks of 2: GPU r gets chunks r and 2N−1−r" },
  { id: "stripe", label: "Stripe", rule: "round-robin: token t → GPU t mod N" },
];
const STAGES = [
  { id: "seq", label: "1 · all Q, K, V" },
  { id: "assign", label: "2 · assign to GPUs" },
  { id: "dist", label: "3 · distribute" },
];
const QKV = [
  { l: "Q", c: "#6EF3DD" },
  { l: "K", c: "#FFC94D" },
  { l: "V", c: "#FF8AC6" },
];

const PLAY_ORDER = [["seq", "ring"], ["assign", "ring"], ["dist", "ring"], ["assign", "zigzag"], ["dist", "zigzag"], ["assign", "stripe"], ["dist", "stripe"]];

const W = 1000;
const TW = 22; // token width
const GAPX = 4;
const CELL = 13; // height of one Q/K/V cell

function Token({ t, x, y, color, delay, hl, dim }) {
  return (
    <g style={{ transform: `translate(${x}px, ${y}px)`, transition: `transform .9s cubic-bezier(.35,.7,.25,1) ${delay}ms, opacity .3s`, opacity: dim ? 0.25 : 1 }}>
      <text x={TW / 2} y={-4} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight={hl ? 700 : 500} fill={color}>{t}</text>
      <rect x={-1.5} y={-1.5} width={TW + 3} height={3 * CELL + 3} rx={4} fill="none" stroke={hl ? T.ink : color} strokeWidth={hl ? 1.6 : 1} style={{ transition: "stroke .4s" }} />
      {QKV.map((m, i) => (
        <g key={m.l}>
          <rect x={0} y={i * CELL} width={TW} height={CELL - 1} rx={2} fill={`${m.c}2e`} />
          <text x={TW / 2} y={i * CELL + 9.5} textAnchor="middle" fontFamily={mono} fontSize="8.5" fontWeight="700" fill={m.c}>{m.l}</text>
        </g>
      ))}
    </g>
  );
}

function Stage({ N, method, stage, sel, onSel }) {
  const S = 4 * N;
  const per = S / N;
  const rowW = S * (TW + GAPX) - GAPX;
  const x0 = (W - rowW) / 2;
  const srcY = 34;
  // the ring is an ellipse: wide enough for the boxes, short enough to fit on screen
  const BW = per * (TW + GAPX) + 18, BH = 3 * CELL + 42;
  const Rx = N <= 4 ? 230 : 300, Ry = N <= 4 ? 135 : 175;
  const cx = W / 2, cy = srcY + 3 * CELL + 36 + Ry + BH / 2;
  const H = cy + Ry + BH / 2 + 12;
  const ang = (r) => -Math.PI / 2 + (2 * Math.PI * r) / N;
  const pt = (a) => [cx + Rx * Math.cos(a), cy + Ry * Math.sin(a)];
  const box = (r) => { const [x, y] = pt(ang(r)); return [x - BW / 2, y - BH / 2]; };
  const lens = [S];
  const slot = useMemo(() => {
    const m = {};
    for (let r = 0; r < N; r++) localTokens(method, N, [S], r).forEach((t, k) => { m[t] = k; });
    return m;
  }, [method, N, S]);
  const dist = stage === "dist";
  const colored = stage !== "seq";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 1060, display: "block", margin: "0 auto" }}>
      <defs>
        <marker id="dist-arrow" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M0,0 L9,4.5 L0,9 z" fill={T.rule} />
        </marker>
      </defs>

      {/* where the sequence starts */}
      <text x={x0} y={srcY - 18} fontFamily={mono} fontSize="11" fill={T.dim}>sequence, token 0 → {S - 1}</text>
      <rect x={x0 - 6} y={srcY - 14} width={rowW + 12} height={3 * CELL + 20} rx={8} fill="none" stroke={T.rule} strokeDasharray={dist ? "3 4" : "none"} opacity={dist ? 0.5 : 1} />

      {/* ring: send direction r -> r+1 */}
      {Array.from({ length: N }, (_, r) => {
        const gap = N <= 4 ? 0.42 : 0.27;
        const [x1, y1] = pt(ang(r) + gap), [x2, y2] = pt(ang(r + 1) - gap);
        return (
          <path key={r} d={`M${x1},${y1} A${Rx},${Ry} 0 0 1 ${x2},${y2}`}
            fill="none" stroke={T.rule} strokeWidth={1.2} markerEnd="url(#dist-arrow)" opacity={dist ? 0.9 : 0.4} />
        );
      })}
      <text x={cx} y={cy - 6} textAnchor="middle" fontFamily={mono} fontSize="13" fill={T.dim}>{N} GPUs</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fontFamily={mono} fontSize="16" fontWeight="700" fill={dist ? T.accent : T.dim}>{METHODS.find((m) => m.id === method).label}</text>

      {/* GPU boxes */}
      {Array.from({ length: N }, (_, r) => {
        const [bx, by] = box(r);
        const on = r === sel;
        return (
          <g key={r} onClick={() => onSel(on ? null : r)} style={{ cursor: "pointer" }}>
            <rect x={bx} y={by} width={BW} height={BH} rx={9} fill={on ? `${DEV_COLORS[r]}22` : T.well}
              stroke={DEV_COLORS[r]} strokeWidth={on ? 2.4 : 1.2} strokeDasharray={dist ? "none" : "4 4"} opacity={dist || on ? 1 : 0.6} style={{ transition: "all .3s" }} />
            <text x={bx + 8} y={by + 14} fontFamily={mono} fontSize="11" fontWeight="700" fill={DEV_COLORS[r]}>GPU {r}</text>
          </g>
        );
      })}

      {/* tokens (drawn last so they fly over the boxes) */}
      {Array.from({ length: S }, (_, t) => {
        const own = ownerOf(method, N, lens, t);
        const [bx, by] = box(own);
        const x = dist ? bx + 9 + slot[t] * (TW + GAPX) : x0 + t * (TW + GAPX);
        const y = dist ? by + 30 : srcY;
        return (
          <Token key={t} t={t} x={x} y={y} color={colored ? DEV_COLORS[own] : T.soft}
            delay={dist ? t * 45 : 0} hl={sel === own && colored} dim={sel != null && colored && sel !== own} />
        );
      })}
    </svg>
  );
}

// Causal matrix at token resolution, row q coloured by the GPU holding query q.
function CausalMatrix({ N, method, sel, active }) {
  const S = 4 * N;
  const size = 280, pad = 22, cs = size / S;
  const own = Array.from({ length: S }, (_, t) => ownerOf(method, N, [S], t));
  const work = Array.from({ length: N }, () => 0);
  own.forEach((r, q) => { work[r] += q + 1; });
  const max = Math.max(...work), avg = work.reduce((a, b) => a + b, 0) / N;
  return (
    <div style={{ ...panel, borderColor: active ? T.accent : T.rule }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: active ? T.accent : T.ink }}>{METHODS.find((m) => m.id === method).label}</span>
        <span style={{ fontFamily: mono, fontSize: 11, color: max / avg > 1.2 ? T.bad : T.accent }}>slowest GPU = {(max / avg).toFixed(2)}× average</span>
      </div>
      <svg viewBox={`0 0 ${size + pad + 4} ${size + pad + 4}`} style={{ width: "100%", maxWidth: 340, display: "block", margin: "0 auto" }}>
        <g transform={`translate(${pad},${pad})`}>
          <rect x={0} y={0} width={size} height={size} fill="none" stroke={T.rule} />
          {own.map((r, q) => {
            const dim = sel != null && r !== sel;
            return <rect key={q} x={0} y={q * cs} width={(q + 1) * cs} height={cs} fill={DEV_COLORS[r]} opacity={dim ? 0.1 : sel === r ? 0.95 : 0.6} style={{ transition: "opacity .3s" }} />;
          })}
          {own.map((r, q) => (
            <rect key={`l${q}`} x={-pad + 4} y={q * cs + 0.5} width={pad - 8} height={cs - 1} rx={1} fill={DEV_COLORS[r]} opacity={sel != null && r !== sel ? 0.25 : 1} />
          ))}
          {Array.from({ length: S + 1 }, (_, i) => (
            <line key={i} x1={0} x2={size} y1={i * cs} y2={i * cs} stroke={T.bg} strokeWidth={0.6} />
          ))}
          <line x1={0} y1={0} x2={size} y2={size} stroke={T.dim} strokeDasharray="3 3" strokeWidth={0.8} />
        </g>
        <text x={pad} y={13} fontFamily={mono} fontSize="10" fill={T.dim}>query ↓ · key →</text>
      </svg>
      <div style={{ display: "grid", gap: 3, marginTop: 10 }}>
        {work.map((w, r) => (
          <div key={r} style={{ display: "flex", alignItems: "center", gap: 6, opacity: sel != null && sel !== r ? 0.35 : 1 }}>
            <span style={{ width: 38, fontFamily: mono, fontSize: 10.5, color: DEV_COLORS[r] }}>GPU {r}</span>
            <div style={{ flex: 1, height: 10, background: T.well, borderRadius: 3 }}>
              <div style={{ width: `${(100 * w) / max}%`, height: "100%", background: DEV_COLORS[r], borderRadius: 3, transition: "width .4s" }} />
            </div>
            <span style={{ width: 30, fontFamily: mono, fontSize: 10.5, color: T.soft, textAlign: "right" }}>{w}</span>
          </div>
        ))}
      </div>
      <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginTop: 6 }}>bar = q·k scores the GPU computes (its rows of the triangle)</div>
      {method === "stripe" && <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginTop: 3 }}>only 4 tokens per GPU here; with long sequences this tends to 1.00×</div>}
      {method === "ring" && <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginTop: 3 }}>tends to 2× as N grows: GPU N−1 does ~twice the average</div>}
    </div>
  );
}

export default function DistributeView() {
  const [N, setN] = useState(8);
  const [method, setMethod] = useState("ring");
  const [stage, setStage] = useState("seq");
  const [sel, setSel] = useState(null);
  const [playing, setPlaying] = useState(false);
  const si = STAGES.findIndex((s) => s.id === stage);

  // play: walk the stages, then distribute ring → zig-zag → stripe
  const cur = PLAY_ORDER.findIndex(([s, m]) => s === stage && m === method);
  const atEnd = cur === PLAY_ORDER.length - 1;
  useEffect(() => {
    if (!playing || atEnd) return;
    const seqOrder = PLAY_ORDER;
    const t = setTimeout(() => {
      const [s, m] = seqOrder[cur + 1] ?? seqOrder[0];
      setStage(s); setMethod(m);
    }, stage === "dist" ? 2800 : 1500);
    return () => clearTimeout(t);
  }, [playing, stage, cur, atEnd]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") setPlaying(false);
      if (e.key === "ArrowRight") setStage((s) => STAGES[Math.min(STAGES.findIndex((x) => x.id === s) + 1, STAGES.length - 1)].id);
      if (e.key === "ArrowLeft") setStage((s) => STAGES[Math.max(STAGES.findIndex((x) => x.id === s) - 1, 0)].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const btn = (on, c = T.accent) => ({ ...navBtn(false), height: 30, padding: "0 12px", fontSize: 12, color: on ? T.bg : T.soft, background: on ? c : T.well, borderColor: on ? c : T.rule });
  const m = METHODS.find((x) => x.id === method);

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SEQUENCE PARALLELISM · FROM ONE SEQUENCE TO {N} GPUs</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Where every query, key and value goes</h1>
        <Rich
          style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 1000 }}
          html={`One sequence of ${4 * N} tokens, split across ${N} GPUs on a ring. Colour each token by the GPU it goes to, then switch the layout to watch the tokens re-shuffle.`}
        />
      </header>

      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center", marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {STAGES.map((s) => <button key={s.id} onClick={() => { setPlaying(false); setStage(s.id); }} style={btn(stage === s.id)}>{s.label}</button>)}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>layout</span>
          {METHODS.map((x) => <button key={x.id} onClick={() => { setPlaying(false); setMethod(x.id); }} style={btn(method === x.id, T.wire)}>{x.label}</button>)}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>GPUs</span>
          {[4, 8].map((n) => <button key={n} onClick={() => { setN(n); setSel(null); }} style={btn(N === n)}>{n}</button>)}
        </div>
        <div style={{ display: "flex", gap: 8, flex: "1 1 260px", justifyContent: "flex-end" }}>
          <button onClick={() => { setPlaying(false); setStage(STAGES[Math.max(si - 1, 0)].id); }} disabled={si === 0} style={navBtn(si === 0)}>← prev</button>
          <button onClick={() => { if (!playing || atEnd) { setStage("seq"); setMethod("ring"); setPlaying(true); } else setPlaying(false); }} style={{ ...navBtn(false), color: T.accent, borderColor: `${T.accent}88` }}>{playing && !atEnd ? "❚❚ pause" : "▶ play all three"}</button>
          <button onClick={() => { setPlaying(false); setStage(STAGES[Math.min(si + 1, 2)].id); }} disabled={si === 2} style={navBtn(si === 2)}>next →</button>
        </div>
      </div>

      <section style={{ ...panel, marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, marginBottom: 6 }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>
            {stage === "seq" ? "One sequence: every token has a query, a key and a value" : stage === "assign" ? `${m.label}: colour each token by its GPU` : `${m.label}: tokens sent to their GPUs`}
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center", fontFamily: mono, fontSize: 11 }}>
            {QKV.map((x) => <span key={x.l} style={{ color: x.c }}>■ {x.l}</span>)}
            <span style={{ color: T.dim }}>· click a GPU to highlight it</span>
          </div>
        </div>
        {stage !== "seq" && <div style={{ fontFamily: mono, fontSize: 12, color: T.wire, marginBottom: 4 }}>rule: {m.rule}</div>}
        <Stage N={N} method={method} stage={stage} sel={sel} onSel={setSel} />
        <Rich
          style={{ fontSize: 13, color: T.dim, lineHeight: 1.55 }}
          html={stage === "dist"
            ? "Each GPU keeps its <b>Q</b>. Its <b>K, V</b> travel around the ring (arrows), so every query meets every key it can see."
            : stage === "assign"
              ? "Nothing has moved yet: the colour is just the destination. Compare the patterns: a block (ring), an early and late pair (zig-zag), or every N-th token (stripe)."
              : "On one GPU all of this sits on one device. Sequence parallelism cuts it along the token axis."}
        />
      </section>

      <div style={h2}>Causal attention matrix · who computes which rows</div>
      <div className="three-col">
        {METHODS.map((x) => <CausalMatrix key={x.id} N={N} method={x.id} sel={sel} active={x.id === method && stage !== "seq"} />)}
      </div>
      <Rich
        style={{ fontSize: 13, color: T.dim, lineHeight: 1.6, marginTop: 12, maxWidth: 1100 }}
        html={`Row $q$ is the query of token $q$, and it sees keys $0..q$. The GPU that holds $q$ does all of row $q$'s work. Ring gives GPU ${N - 1} the longest rows; zig-zag and stripe balance them. Click a GPU on the ring to see its rows in all three.`}
      />
    </div>
  );
}
