import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { katexHtml } from "./latex";
import { DEV_COLORS, PHASE, STEP_PHASES, shardLabel, STRIPE_L, buildPhases, deviceStep, heldKV, inFlight, mod, covered } from "./ringModel";
import { RING_SRC, ZIGZAG_SRC, STRIPE_SRC, UTILS_SRC, ZIGZAG_UTILS_SRC, STRIPE_UTILS_SRC } from "./codeSources";
import { CodeBlock } from "./CodePanel";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich, Tex } from "./ui";
import RingDiagram from "./RingDiagram";
import AttentionMatrix, { SequenceStrip, WorkChart } from "./AttentionMatrix";
import { equations, transfers } from "./equations";

const COPY = {
  ring: {
    eyebrow: "SEQUENCE PARALLELISM · RING ATTENTION",
    title: "Ring attention: Q stays, KV goes around",
    intro: "Each GPU keeps its query block and passes its key/value block to the next GPU, $N$ times. Every hop is posted <i>before</i> the attention kernel, so the transfer hides behind compute. With a causal mask, half the blocks it receives are all future tokens, and that is where the imbalance comes from.",
    src: ["ring_flash_attn.py", RING_SRC, 430],
    utils: ["utils.py", UTILS_SRC],
    matrixNote: () => "GPU $r$ owns two neighbouring chunk rows ($c_{2r}, c_{2r+1}$). The top pair has 3 visible cells (2 chunk²), the bottom pair $4N-1$ cells ($4N-2$ chunk²). White outline = the GPU you follow.",
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$). A step lasts as long as its slowest GPU. Step 0 costs 2, every later step 4 (a full block on GPU ${N - 1}), so wall time is $4N-2 = ${4 * N - 2}$ against an ideal $2N = ${2 * N}$.`,
  },
  zigzag: {
    eyebrow: "SEQUENCE PARALLELISM · ZIG-ZAG RING ATTENTION",
    title: "Zig-zag ring attention: balance the causal triangle",
    intro: "Same ring and same hops, but the sequence is cut into $2N$ chunks and GPU $r$ takes chunks $r$ and $2N-1-r$: one early, one late. Every off-diagonal step now costs exactly half a block on every GPU, picked by one of two slices: <code>k[:, :L/2]</code> or <code>q[:, L/2:]</code>.",
    src: ["zigzag_ring_flash_attn.py", ZIGZAG_SRC, 470],
    utils: ["extract_local + utils.py", ZIGZAG_UTILS_SRC],
    matrixNote: () => "Each GPU owns two rows of chunks, one near the top (few keys) and one near the bottom (many keys), so every GPU's total share of the triangle is the same.",
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$). Every step costs 2 on every GPU: wall time $2N = ${2 * N}$, no idle time. Same bar scale as the ring view.`,
  },
  stripe: {
    eyebrow: "SEQUENCE PARALLELISM · STRIPE ATTENTION",
    title: "Stripe attention: deal tokens round-robin",
    intro: "Same ring and same hops, but tokens are dealt like cards: token $t$ goes to GPU $t \\bmod N$. Every GPU then holds a thin stripe spread over the whole sequence, so each KV block it receives is about half visible, at every step. If the block came from an earlier rank the plain causal mask is right; from a later rank the mask shifts by one token: <code>q[:, 1:]</code> against <code>k[:, :-1]</code>.",
    src: ["stripe_flash_attn.py", STRIPE_SRC, 470],
    utils: ["extract_local + utils.py", STRIPE_UTILS_SRC],
    matrixNote: (N) => `Token resolution: $S = ${STRIPE_L}N = ${STRIPE_L * N}$ tokens, row $t$ belongs to GPU $t \\bmod ${N}$. Every GPU's rows are spread evenly down the triangle. A step fills $\\tfrac{L(L+1)}{2}$ cells (from an earlier rank) or $\\tfrac{L(L-1)}{2}$ (later rank): with $L=${STRIPE_L}$ that is ${(STRIPE_L * (STRIPE_L + 1)) / 2}$ vs ${(STRIPE_L * (STRIPE_L - 1)) / 2}$, but both tend to $L^2/2$ as $L$ grows.`,
    workNote: (N) => `Units = chunk² ($\\tfrac{L}{2}\\times\\tfrac{L}{2}$), large-$L$ limit: every step costs $L^2/2 = 2$ on every GPU, wall time $2N = ${2 * N}$. The leftover imbalance is one token's diagonal, $L$ cells per step, negligible for long sequences.`,
  },
};

// tag -> color for the main function and utils listings at this phase.
function codeFocus(variant, N, phase, r) {
  const { type, s } = phase;
  const C = (t) => PHASE[t].color;
  const main = {}, utils = {};
  if (type === "init") {
    main.init = C("init");
    utils.init = C("init");
    utils.layout = C("init");
    return { main, utils };
  }
  if (type === "final") {
    main.final = C("final");
    return { main, utils };
  }
  main.loop = `dim:${T.soft}`;
  const d = deviceStep(variant, N, r, s);
  if (type === "send") {
    main.send = C("send");
    utils.send = C("send");
  } else if (type === "wait") {
    main.wait = C("wait");
    utils.wait = C("wait");
  } else if (type === "compute") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (variant === "ring") {
      main.cond = d.active ? C("compute") : T.bad;
      if (d.active) main.comp = C("compute");
    } else {
      main[`cond:${d.branch}`] = C("compute");
      main[`comp:${d.branch}`] = C("compute");
      main.fwd = `dim:${C("compute")}`;
    }
  } else if (type === "merge") {
    if (inFlight(N, phase)) utils.pending = `dim:${T.wire}`;
    if (d.active) {
      if (variant === "ring") main.merge = C("merge");
      else main[`merge:${d.branch}`] = C("merge");
      utils.upd = C("merge");
      if (s === 0) utils.first = C("merge");
      else {
        utils[d.branch === "hi" ? "slice" : "else"] = C("merge");
        utils.core = C("merge");
      }
    } else if (variant === "ring") main.cond = T.bad;
  }
  return { main, utils };
}

function variables(variant, N, phase, r) {
  const { type, s } = phase;
  const kvLbl = (j) => shardLabel(variant, N, j, "KV");
  const rows = [
    ["comm.rank", r],
    ["comm.send_rank", `${mod(r + 1, N)}  (next)`],
    ["comm.recv_rank", `${mod(r - 1, N)}  (prev)`],
    ["step", type === "init" ? "—" : type === "final" ? `${N - 1} (done)` : s],
    ["k, v", kvLbl(heldKV(N, r, phase))],
  ];
  if (inFlight(N, phase)) rows.push(["next_k, next_v", `← ${kvLbl(mod(r - s - 1, N))} (in flight)`]);
  else if (type === "wait") rows.push(["next_k, next_v", `${kvLbl(mod(r - s - 1, N))} (landed)`]);
  else rows.push(["next_k, next_v", type === "init" ? "None" : "—"]);
  if (STEP_PHASES.includes(type)) {
    const d = deviceStep(variant, N, r, s);
    if (variant === "ring") {
      rows.push(["step <= comm.rank", `${s} <= ${r} → ${s <= r}`]);
      rows.push(["causal and step == 0", `${s === 0}`]);
    } else if (variant === "stripe") {
      rows.push(["step <= comm.rank", `${s} <= ${r} → ${s <= r}`]);
      rows.push(["kernel inputs", d.branch === "lo" ? "q, k, v  causal" : "q[:,1:], k[:,:-1], v[:,:-1]  causal"]);
      rows.push(["slice_", d.branch === "hi" ? "(:, 1:)" : "None"]);
    } else {
      rows.push(["branch", d.branch === "0" ? "step == 0" : d.branch === "lo" ? `step <= rank (${s} <= ${r})` : `else (${s} > ${r})`]);
      rows.push(["kernel inputs", d.branch === "0" ? "q, k, v  causal" : d.branch === "lo" ? "q, k0, v0" : "q1, k, v"]);
      rows.push(["slice_", d.branch === "hi" ? "(:, L/2:)" : "None"]);
    }
  }
  const js = covered(variant, N, r, phase);
  rows.push(["out covers", js.length ? js.map(kvLbl).join("  ") : "None"]);
  return rows;
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
  const eqs = equations(variant, N, sel, phase);
  const xfers = transfers(variant, N, phase);
  const vars = variables(variant, N, phase, sel);
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
          </section>

          <section style={panel}>
            <div style={h2}>Equations · GPU {sel} at this phase</div>
            <div style={{ display: "grid", gap: 6, color: T.soft, fontSize: 14.5, lineHeight: 1.6 }}>
              {eqs.map((e, i) => (e.m ? <Tex key={i} tex={e.m} /> : <Rich key={i} html={e.p} />))}
            </div>
          </section>

          <section style={panel}>
            <div style={h2}>Transfers this step{xfers.length ? ` · s = ${phase.s}` : ""}</div>
            {xfers.length === 0 ? (
              <div style={{ fontFamily: mono, fontSize: 12.5, color: T.dim }}>
                {phase.type === "init" ? "No transfers yet." : phase.type === "final" ? `Done: ${N - 1} hops per GPU.` : "Last step: step + 1 == world_size, nothing sent."}
              </div>
            ) : (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 6 }}>
                  {xfers.map((x) => {
                    const hot = x.from === sel || x.to === sel;
                    return (
                      <div key={x.from} style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: mono, fontSize: 12, padding: "4px 10px", borderRadius: 6,
                        background: hot ? `${T.wire}14` : T.well, border: `1px solid ${hot ? `${T.wire}66` : T.rule}` }}>
                        <span style={{ color: DEV_COLORS[x.from] }}>GPU {x.from}</span>
                        <span style={{ color: T.wire }}>→</span>
                        <span dangerouslySetInnerHTML={{ __html: katexHtml(x.tex) }} />
                        <span style={{ color: T.wire }}>→</span>
                        <span style={{ color: DEV_COLORS[x.to] }}>GPU {x.to}</span>
                      </div>
                    );
                  })}
                </div>
                <Rich
                  style={{ fontSize: 13, color: T.dim, marginTop: 10, lineHeight: 1.5 }}
                  html={String.raw`Rule for every hop: $\;KV^{(s+1)}_{(i+1)\bmod N} = KV^{(s)}_{i} = KV_{(i-s)\bmod N}$. ${phase.type === "wait" ? "Landed: buffers swapped." : "In flight while the kernel runs."}`}
                />
              </>
            )}
          </section>

          <div className="ra-two">
            <section style={panel}>
              <div style={h2}>Causal attention matrix</div>
              <SequenceStrip variant={variant} N={N} />
              <div style={{ height: 12 }} />
              <AttentionMatrix variant={variant} N={N} phase={phase} sel={sel} />
              <Rich
                style={{ fontSize: 12.5, color: T.dim, marginTop: 8, lineHeight: 1.5 }}
                html={copy.matrixNote(N)}
              />
            </section>
            <section style={panel}>
              <div style={h2}>Compute per step</div>
              <WorkChart variant={variant} N={N} phase={phase} sel={sel} />
              <Rich
                style={{ fontSize: 12.5, color: T.dim, marginTop: 12, lineHeight: 1.55 }}
                html={copy.workNote(N)}
              />
            </section>
          </div>
        </main>

        {/* ---------- code ---------- */}
        <aside className="ra-side">
          <CodeBlock
            title={copy.src[0]}
            lines={copy.src[1]}
            focus={focus.main}
            maxHeight={copy.src[2]}
            note={`following GPU ${sel}`}
          />
          <div style={{ ...panel, padding: "10px 14px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
              <div style={{ ...h2, margin: 0 }}>Live variables · rank {sel}</div>
              <div style={{ display: "flex", gap: 4 }}>
                {Array.from({ length: N }, (_, r) => (
                  <button
                    key={r} onClick={() => setSel(r)} aria-label={`Show rank ${r}`}
                    style={{ ...navBtn(false), height: 24, padding: "0 9px", fontSize: 11.5, color: DEV_COLORS[r],
                      borderColor: r === sel ? DEV_COLORS[r] : T.rule, background: r === sel ? `${DEV_COLORS[r]}22` : T.well, fontWeight: r === sel ? 700 : 400 }}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr", columnGap: 14, rowGap: 3, fontFamily: mono, fontSize: 12 }}>
              {vars.map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <span style={{ color: T.dim }}>{k}</span>
                  <span style={{ color: k === "step <= comm.rank" && String(v).endsWith("false") ? T.bad : T.ink }}>{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
          <CodeBlock
            title={copy.utils[0]}
            lines={copy.utils[1]}
            focus={focus.utils}
            maxHeight={300}
            note={phase.type === "compute" ? (inFlight(N, phase) ? "kernel runs; _reqs still pending" : "kernel runs (flash-attn)") : null}
          />
        </aside>
      </div>
    </div>
  );
}
