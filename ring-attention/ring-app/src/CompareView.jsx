import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS, PHASE, STEP_PHASES, buildPhases, deviceStep, heldKV, mod, tokensOf, shardLabel, STRIPE_L } from "./ringModel";
import { COPY, codeFocus, codeNotes } from "./ringLogic";
import { RankPicker } from "./RingView";
import { CodeBlock } from "./CodePanel";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich } from "./ui";
import RingDiagram, { OutLegend } from "./RingDiagram";
import AttentionMatrix, { OutLseKey, SequenceStrip, WorkChart } from "./AttentionMatrix";

// Zig-zag and stripe driven by one stepper: same ring, same phases, so each
// "next" shows what moves and what each kernel computes in both, side by side.

const SIDES = [
  { v: "zigzag", name: "Zig-zag", color: "#8AB4FF" },
  { v: "stripe", name: "Stripe", color: "#FF8AC6" },
];

// What the kernel call looks like for GPU r at step s.
function kernel(variant, N, r, s) {
  const d = deviceStep(variant, N, r, s);
  if (variant === "zigzag") {
    if (d.branch === "0") return { br: "step == 0", call: "forward(q, k, v, causal=True)", shape: "L × L ◣", slice: "—", d };
    if (d.branch === "lo") return { br: `step <= rank`, call: "forward(q, k0, v0, causal=False)", shape: "L × L/2 ▭", slice: "—", d };
    return { br: "else", call: "forward(q1, k, v, causal=False)", shape: "L/2 × L ▭", slice: "(:, L/2:)", d };
  }
  if (d.branch === "lo") return { br: "step <= rank", call: "fa(q, k, v, causal=True)", shape: "L × L ◣", slice: "—", d };
  return { br: "else", call: "fa(q[:,1:], k[:,:-1], v[:,:-1], causal=True)", shape: "(L−1) × (L−1) ◣", slice: "(:, 1:)", d };
}

function Pair({ children, title }) {
  return (
    <section>
      {title && <div style={{ ...h2, margin: "4px 2px 10px" }}>{title}</div>}
      <div className="cmp-two">{children}</div>
    </section>
  );
}

function SideHead({ side, extra }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
      <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: side.color, letterSpacing: 0.5 }}>{side.name}</span>
      {extra}
    </div>
  );
}

// Which tokens a KV shard holds, in words (for S = 4N tokens, chunk = 2 tokens).
function tokenPreview(variant, N, j) {
  return `tokens ${tokensOf(variant, N, j).join(",")}`;
}

function DiffTable({ N, phase, sel }) {
  const { type, s } = phase;
  const th = { textAlign: "left", padding: "6px 8px", fontWeight: 500, color: T.dim, borderBottom: `1px solid ${T.rule}`, whiteSpace: "nowrap" };
  const td = { padding: "6px 8px", borderBottom: `1px solid ${T.well}`, verticalAlign: "top" };
  const stepPhase = STEP_PHASES.includes(type);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: mono, fontSize: 12 }}>
        <thead>
          <tr>
            <th style={th}>GPU</th>
            {SIDES.map((sd) => (
              <th key={sd.v} colSpan={stepPhase ? 3 : 2} style={{ ...th, color: sd.color, borderLeft: `1px solid ${T.rule}` }}>{sd.name}</th>
            ))}
          </tr>
          <tr>
            <th style={th} />
            {SIDES.map((sd) => (stepPhase ? (
              [<th key={`${sd.v}a`} style={{ ...th, borderLeft: `1px solid ${T.rule}` }}>holds k, v</th>, <th key={`${sd.v}b`} style={th}>branch → kernel call</th>, <th key={`${sd.v}c`} style={th}>shape · slice_</th>]
            ) : (
              [<th key={`${sd.v}a`} style={{ ...th, borderLeft: `1px solid ${T.rule}` }}>owns Q, K, V</th>, <th key={`${sd.v}b`} style={th}>tokens (S = {STRIPE_L * N})</th>]
            )))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: N }, (_, r) => {
            const isSel = r === sel;
            return (
              <tr key={r} style={{ background: isSel ? `${DEV_COLORS[r]}14` : "transparent" }}>
                <td style={{ ...td, color: DEV_COLORS[r], fontWeight: isSel ? 700 : 400 }}>{r}</td>
                {SIDES.map((sd) => {
                  if (!stepPhase) {
                    return [
                      <td key={`${sd.v}a`} style={{ ...td, borderLeft: `1px solid ${T.rule}`, color: T.ink }}>{sd.v === "stripe" ? `x[:, ${r}::${N}]` : `chunks c${r}, c${2 * N - 1 - r}`}</td>,
                      <td key={`${sd.v}b`} style={{ ...td, color: T.soft }}>{tokenPreview(sd.v, N, r).replace("tokens ", "")}</td>,
                    ];
                  }
                  const k = kernel(sd.v, N, r, s);
                  const computing = type === "compute" || type === "merge";
                  return [
                    <td key={`${sd.v}a`} style={{ ...td, borderLeft: `1px solid ${T.rule}`, color: DEV_COLORS[heldKV(N, r, phase)] }}>{shardLabel(sd.v, N, heldKV(N, r, phase), "KV")}</td>,
                    <td key={`${sd.v}b`} style={{ ...td, color: computing ? T.ink : T.soft }}>
                      <span style={{ color: computing ? PHASE.compute.color : T.dim }}>{k.br}</span>
                      <div style={{ whiteSpace: "nowrap" }}>{k.call}</div>
                    </td>,
                    <td key={`${sd.v}c`} style={{ ...td, color: T.soft, whiteSpace: "nowrap" }}>
                      {k.shape}
                      <div style={{ color: k.slice === "—" ? T.dim : PHASE.merge.color }}>{k.slice === "—" ? "no slice" : `slice_ ${k.slice}`}</div>
                    </td>,
                  ];
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const TAKEAWAY = {
  init: "Zig-zag gives each GPU two contiguous chunks (one early, one late). Stripe deals single tokens round-robin. Q, K, V never leave this layout; only K, V rotate.",
  send: "The transfer is identical: each GPU sends the KV it holds to rank+1 and receives from rank−1. Only what is inside the shard differs.",
  compute: "Here they differ. Zig-zag slices the inputs in half and runs an <b>unmasked</b> rectangle. Stripe keeps full-size inputs, shifts them by one token when the KV came from a later rank, and always runs a <b>causal</b> kernel.",
  merge: "Both merge with the same <code>update_out_and_lse</code>. On the <code>else</code> branch zig-zag merges only the late half of the rows (<code>slice_=(:, L/2:)</code>); stripe merges every row except row 0 (<code>slice_=(:, 1:)</code>).",
  wait: "Same sync point in both: <code>comm.wait()</code>, then <code>k, v = next_k, next_v</code>.",
  final: "Both return output in their own permuted order: undo the zig-zag chunk swap, or interleave the stripes.",
};

export default function CompareView() {
  const [N, setN] = useState(4);
  const [sel, setSel] = useState(2);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const phases = useMemo(() => buildPhases(N), [N]);
  const last = phases.length - 1;
  const phase = phases[Math.min(idx, last)];
  const ph = PHASE[phase.type];

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
    const t = setTimeout(() => setIdx((i) => i + 1), 1800);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  return (
    <div style={{ maxWidth: 1600, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SEQUENCE PARALLELISM · ZIG-ZAG vs STRIPE</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Same ring, different deal</h1>
        <Rich
          style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 1000 }}
          html="One stepper drives both. The ring, the hops and the sync points are identical; what differs is which tokens sit in each shard and therefore what each GPU feeds the kernel at every step. Both views use $S = 4N$ tokens so the token numbers line up."
        />
      </header>

      {/* controls */}
      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center", marginBottom: 18, padding: "12px 16px", position: "sticky", top: 0, zIndex: 5 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>GPUs N</span>
          {[2, 3, 4, 6, 8].map((n) => (
            <button key={n} onClick={() => changeN(n)} style={{ ...navBtn(false), height: 28, padding: "0 10px", background: n === N ? T.accent : T.well, color: n === N ? T.bg : T.soft, borderColor: n === N ? T.accent : T.rule }}>{n}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>follow GPU</span>
          {Array.from({ length: N }, (_, r) => (
            <button key={r} onClick={() => setSel(r)} style={{ ...navBtn(false), height: 28, padding: "0 10px", color: DEV_COLORS[r], borderColor: r === sel ? DEV_COLORS[r] : T.rule, background: r === sel ? `${DEV_COLORS[r]}22` : T.well, fontWeight: r === sel ? 700 : 400 }}>{r}</button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {phase.s >= 0 && phase.type !== "final" && <span style={chip(T.accent)}>step {phase.s}</span>}
          <span style={chip(ph.color)}>{ph.label}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: "1 1 320px", justifyContent: "flex-end" }}>
          <button onClick={() => setIdx(0)} disabled={idx === 0} style={navBtn(idx === 0)} aria-label="Restart">⟲</button>
          <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} style={navBtn(idx === 0)}>← prev</button>
          <button onClick={() => { if (idx >= last) { setIdx(0); setPlaying(true); } else setPlaying(!playing); }} style={{ ...navBtn(false), color: T.accent, borderColor: `${T.accent}88` }}>{playing && idx < last ? "❚❚ pause" : "▶ play"}</button>
          <button onClick={() => setIdx(idx + 1)} disabled={idx === last} style={navBtn(idx === last)}>next →</button>
        </div>
        <div style={{ display: "flex", gap: 3, width: "100%" }}>
          {phases.map((p, i) => (
            <button
              key={i} onClick={() => setIdx(i)} aria-label={`Phase ${i + 1}`} title={`${p.type}${p.s >= 0 ? ` · step ${p.s}` : ""}`}
              style={{ flex: 1, minWidth: 8, height: 8, border: "none", padding: 0, borderRadius: 3, cursor: "pointer",
                background: i === idx ? PHASE[p.type].color : i < idx ? `${PHASE[p.type].color}66` : T.rule, outline: i === idx ? `1px solid ${T.ink}` : "none" }}
            />
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gap: 22 }}>
        <div style={{ ...panel, borderLeft: `3px solid ${ph.color}` }}>
          <Rich style={{ fontSize: 14.5, color: T.soft, lineHeight: 1.6 }} html={`<b style="color:${ph.color}">${ph.label}.</b> ${TAKEAWAY[phase.type]}`} />
        </div>

        <Pair title="The ring · what moves this phase">
          {SIDES.map((sd) => (
            <section key={sd.v} style={panel}>
              <SideHead side={sd} />
              <RingDiagram variant={sd.v} N={N} phase={phase} sel={sel} onSel={setSel} />
              <OutLegend variant={sd.v} N={N} phase={phase} sel={sel} />
            </section>
          ))}
        </Pair>

        <section style={panel}>
          <div style={h2}>{STEP_PHASES.includes(phase.type) ? `What each GPU feeds the kernel · step ${phase.s}` : "Who owns which tokens"}</div>
          <DiffTable N={N} phase={phase} sel={sel} />
          <Rich
            style={{ fontSize: 12.5, color: T.dim, marginTop: 10, lineHeight: 1.55 }}
            html="▭ = dense rectangle, no mask. ◣ = causal mask. $L$ = tokens per GPU. Zig-zag halves the inputs and drops the mask; stripe keeps full inputs and lets the mask (shifted by one token on the <code>else</code> branch) discard the invisible half."
          />
        </section>

        <Pair title="The code · same phase, both implementations">
          {SIDES.map((sd) => {
            const focus = codeFocus(sd.v, N, phase, sel);
            return (
              <div key={sd.v} style={{ minWidth: 0 }}>
                <CodeBlock title={COPY[sd.v].src[0]} lines={COPY[sd.v].src[1]} focus={focus.main} notes={codeNotes(sd.v, N, phase, sel).main} maxHeight={470} note={<RankPicker N={N} sel={sel} onSel={setSel} />} />
              </div>
            );
          })}
        </Pair>

        <Pair title="Causal attention matrix · token resolution for both">
          {SIDES.map((sd) => (
            <section key={sd.v} style={panel}>
              <SideHead side={sd} />
              <SequenceStrip variant={sd.v} N={N} tokens />
              <div style={{ height: 12 }} />
              <AttentionMatrix variant={sd.v} N={N} phase={phase} sel={sel} tokens />
              <OutLseKey />
              <div style={{ height: 14 }} />
              <WorkChart variant={sd.v} N={N} phase={phase} sel={sel} />
            </section>
          ))}
        </Pair>

      </div>

    </div>
  );
}
