import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS } from "./ringModel";
import { TREE_PHASE, PER, D, makeData, buildPhases, stateAt, role, fmt, vec, covList } from "./treeModel";
import { TREE_SRC, TREE_AR_SRC } from "./codeSources";
import { RankPicker } from "./RingView";
import { CodeBlock } from "./CodePanel";
import TreeDiagram from "./TreeDiagram";
import { StageStrip, TreeWeights } from "./TreeWeights";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich } from "./ui";

function phaseTitle(ph) {
  if (ph.type === "up" || ph.type === "down") return `${ph.op === "max" ? "max" : "sum"} ${TREE_PHASE[ph.type].label} · level ${ph.lvl}`;
  return TREE_PHASE[ph.type].label;
}

// Permanent comments on the unsqueeze in the decode code, as einops / einsum.
// Layout here: q, local_out [B, H, 1, d]; local_lse, lse_max [B, H, 1].
const TREE_SHAPE_NOTES = [
  { m: "local_out, local_lse = flash_attn_decode(q, k, v)", t: `scores = einsum("bhqd,bhsd->bhqs", q, k) / √d   # q·kᵀ → [B, H, 1, S/N]` },
  { m: "local_out, local_lse = flash_attn_decode(q, k, v)", t: "local_lse = logsumexp(scores, dim=-1)   # [B, H, 1]" },
  { m: "local_out, local_lse = flash_attn_decode(q, k, v)", t: `local_out = einsum("bhqs,bhsd->bhqd", softmax(scores, dim=-1), v)   # p·v → [B, H, 1, ${D}]` },
  { m: "w = torch.exp(local_lse - lse_max).unsqueeze(-1)", t: `[B, H, 1] → unsqueeze(-1) adds a size-1 last dim → [B, H, 1, 1], so it broadcasts over d=${D}  ≡  rearrange(w, "b h q -> b h q 1")` },
  { m: "num = local_out * w", t: `[B, H, 1, ${D}] × [B, H, 1, 1]  ≡  einsum("bhqd,bhq->bhqd", local_out, exp(local_lse − lse_max))` },
  { m: "num, den = buf[..., :-1], buf[..., -1:]", t: `buf is [B, H, 1, ${D}+1]: :-1 keeps the ${D} numerator dims, -1: keeps den as [B, H, 1, 1] (a slice, so no squeeze needed)` },
];

function codeFocus(ph, r) {
  const main = {}, ar = {};
  const C = TREE_PHASE[ph.type].color;
  if (ph.type === "init") { main.local = `dim:${T.soft}`; return { main, ar }; }
  if (ph.type === "local") main.local = C;
  if (ph.type === "rescale") main.rescale = C;
  if (ph.type === "final") main.final = C;
  if (ph.type === "up" || ph.type === "down") {
    main[ph.op] = C;
    ar.ar = `dim:${T.soft}`;
    ar[ph.type] = `dim:${C}`;
    const ro = role(r, ph);
    if (ro.kind !== "idle") ar[`${ph.type}_${ro.kind}`] = C;
  }
  return { main, ar };
}

// Live comments for GPU r at phase idx: what each executed line of the decode
// code and of tree_allreduce evaluates to (see CodeBlock `notes`).
function treeCodeNotes(data, phases, idx, r) {
  const ph = phases[idx];
  const st = stateAt(data, phases, idx);
  const prev = idx > 0 ? stateAt(data, phases, idx - 1) : st;
  const g = data.gpus[r], s = st.gpus[r], N = data.N;
  const n = (m, t, after) => ({ m, t, after });
  const at = (type, op) => phases.findIndex((p) => p.type === type && (!op || p.op === op));
  const reached = (type, op) => at(type, op) <= idx;
  const inAR = (op) => (ph.type === "up" || ph.type === "down") && ph.op === op;
  const covers = (mask) => covList(mask, N).join(", ");
  const keys = `t${PER * r}–t${PER * r + PER - 1}`;
  const main = [];
  const ar = [];

  main.push(n("local_out, local_lse = flash_attn_decode(q, k, v)", reached("local")
    ? `this rank's keys ${keys} → local_lse = ${fmt(g.lse, 3)}, local_out = ${vec(g.o)}`
    : `k, v = this rank's keys ${keys}; q = the new token`));
  if (reached("up", "max")) main.push(n("lse_max = local_lse.clone()", fmt(g.lse, 3)));
  if (inAR("max")) main.push(n("tree_allreduce(lse_max, op=torch.maximum", `buffer now ${fmt(s.mx, 3)} = max over GPUs ${covers(s.cov)}`));
  else if (reached("rescale")) main.push(n("tree_allreduce(lse_max, op=torch.maximum", `→ m = ${fmt(s.mx, 3)}, the same on every rank`));
  if (reached("rescale")) {
    const m = stateAt(data, phases, at("rescale")).gpus[r].mx;
    const w = Math.exp(g.lse - m);
    main.push(n("w = torch.exp(local_lse - lse_max)", `exp(${fmt(g.lse, 3)} − ${fmt(m, 3)}) = ${fmt(w, 4)}`));
    main.push(n("num = local_out * w", vec(g.o.map((x) => x * w))));
    main.push(n("den = w", fmt(w, 4)));
    main.push(n("buf = torch.cat([num, den], dim=-1)", `[${vec(g.o.map((x) => x * w)).slice(1, -1)}, ${fmt(w, 4)}]`));
  }
  if (inAR("sum")) main.push(n("tree_allreduce(buf, op=torch.add", `den part now ${fmt(s.den, 4)} = sum over GPUs ${covers(s.cov)}`));
  if (ph.type === "final") {
    main.push(n("tree_allreduce(buf, op=torch.add", `→ global n, d on every rank`));
    main.push(n("num, den = buf[..., :-1], buf[..., -1:]", `num = ${vec(s.num)}, den = ${fmt(s.den, 4)}`));
    main.push(n("return num / den", `${vec(s.out)} = exact softmax over all ${PER * N} keys`));
  }

  if (ph.type === "up" || ph.type === "down") {
    const ro = role(r, ph), step = ph.step, md = r % (2 * step);
    const isMax = ph.op === "max";
    const val = (x) => (isMax ? fmt(x.mx, 3) : `d = ${fmt(x.den, 4)}`);
    ar.push(n("rank = dist.get_rank(group)", `${r}`), n("N = dist.get_world_size(group)", `${N}`));
    if (ph.type === "up") {
      ar.push(n("while step < N:", `step = ${step} (level ${ph.lvl}): ${step} < ${N}`));
      ar.push(n("if rank % (2 * step) == 0:", `${r} % ${2 * step} = ${md} → ${md === 0}`, "while step < N:"));
      if (ro.kind === "recv") {
        const mine = prev.gpus[r], peer = prev.gpus[ro.peer];
        ar.push(n("dist.recv(buf, src=rank + step", `src = ${r} + ${step} = ${ro.peer}; buf = ${val(peer)}`));
        ar.push(n("x.copy_(op(x, buf))", isMax
          ? `x = max(${fmt(mine.mx, 3)}, ${fmt(peer.mx, 3)}) = ${fmt(s.mx, 3)}`
          : `d = ${fmt(mine.den, 4)} + ${fmt(peer.den, 4)} = ${fmt(s.den, 4)} (n adds the same way)`));
      } else {
        ar.push(n("elif rank % (2 * step) == step:", `${md} == ${step} → ${ro.kind === "send"}${ro.kind === "idle" ? ": already sent at a lower level" : ""}`, "while step < N:"));
        if (ro.kind === "send") ar.push(n("dist.send(x, dst=rank - step", `dst = ${r} − ${step} = ${ro.peer}; sends ${val(prev.gpus[r])}`));
      }
    } else {
      ar.push(n("while step >= 1:", `step = ${step} (level ${ph.lvl})`));
      ar.push(n("if rank % (2 * step) == 0:", `${r} % ${2 * step} = ${md} → ${md === 0}`, "while step >= 1:"));
      if (ro.kind === "send") ar.push(n("dist.send(x, dst=rank + step", `dst = ${r} + ${step} = ${ro.peer}; sends the global ${val(prev.gpus[r])}`));
      else {
        ar.push(n("elif rank % (2 * step) == step:", `${md} == ${step} → ${ro.kind === "recv"}${ro.kind === "idle" ? (s.got || r === 0 ? ": already has the result" : ": receives at a lower level") : ""}`, "while step >= 1:"));
        if (ro.kind === "recv") ar.push(n("dist.recv(x, src=rank - step", `src = ${r} − ${step} = ${ro.peer}; x = ${val(s)} (global)`));
      }
    }
  }
  return { main: [...main, ...TREE_SHAPE_NOTES], ar };
}

export default function TreeView() {
  const [N, setN] = useState(8);
  const [sel, setSel] = useState(0);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const data = useMemo(() => makeData(N), [N]);
  const phases = useMemo(() => buildPhases(N), [N]);
  const last = phases.length - 1;
  const ph = phases[Math.min(idx, last)];
  const state = useMemo(() => stateAt(data, phases, Math.min(idx, last)), [data, phases, idx, last]);

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
      if (e.key === "ArrowUp") setSel((r) => (r + 1) % N);
      if (e.key === "ArrowDown") setSel((r) => (r - 1 + N) % N);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last, N]);

  useEffect(() => {
    if (!playing || idx >= last) return;
    const t = setTimeout(() => setIdx((i) => i + 1), 1500);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  const i = Math.min(idx, last);
  const focus = codeFocus(ph, sel);
  const notes = treeCodeNotes(data, phases, i, sel);
  const pc = TREE_PHASE[ph.type].color;

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SEQUENCE PARALLELISM · TREE ATTENTION (DECODING)</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Tree attention: combine partial softmaxes in log N</h1>
        <Rich
          style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 980 }}
          html="Decoding: one new query token, a long KV cache split over $N$ GPUs. Each GPU attends to its shard, then two tree allreduces (max, then sum) combine the results in $\log_2 N$ levels. Only small partial results move, never K or V (Shyam et al., 2024)."
        />
      </header>

      {/* controls */}
      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center", marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>GPUs N</span>
          {[2, 4, 8].map((n) => (
            <button key={n} onClick={() => changeN(n)} style={{ ...navBtn(false), height: 28, padding: "0 10px", background: n === N ? T.accent : T.well, color: n === N ? T.bg : T.soft, borderColor: n === N ? T.accent : T.rule }}>{n}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
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
        <div style={{ display: "flex", gap: 3, width: "100%" }}>
          {phases.map((p, k) => (
            <button
              key={k} onClick={() => setIdx(k)} aria-label={`Phase ${k + 1}`} title={phaseTitle(p)}
              style={{ flex: 1, minWidth: 8, height: 8, border: "none", padding: 0, borderRadius: 3, cursor: "pointer",
                background: k === idx ? TREE_PHASE[p.type].color : k < idx ? `${TREE_PHASE[p.type].color}66` : T.rule, outline: k === idx ? `1px solid ${T.ink}` : "none" }}
            />
          ))}
        </div>
      </div>

      <div className="ra-grid">
        <main style={{ display: "grid", gap: 18, minWidth: 0 }}>
          <section style={panel}>
            <div style={h2}>The recipe · five stages, two of them collectives</div>
            <StageStrip N={N} phase={ph} />
            <Rich
              style={{ fontSize: 12.5, color: T.dim, marginTop: 10, lineHeight: 1.5 }}
              html="Why it works: $o = \sum_i e^{\ell_i - \ell}\,o_i$. Each GPU's output only needs re-weighting by its share of the global normaliser; stages 2–4 compute those shares."
            />
          </section>

          <section style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
              <div style={h2}>The tree · click a GPU to follow it</div>
              <div style={{ display: "flex", gap: 6 }}>
                {ph.op && <span style={chip(T.accent)}>{ph.op === "max" ? "allreduce max" : "allreduce sum"}</span>}
                <span style={chip(pc)}>{phaseTitle(ph)}</span>
              </div>
            </div>
            <TreeDiagram N={N} phase={ph} state={state} data={data} sel={sel} onSel={setSel} />
            <Rich
              style={{ fontSize: 12.5, color: T.dim, marginTop: 6, lineHeight: 1.5 }}
              html="Bottom-up = reduce, top-down = broadcast. Solid edges send, dashed ones stay on the same GPU. Coloured squares: whose data a GPU's buffer already includes."
            />
          </section>

          <section style={panel}>
            <div style={h2}>What the numbers mean · attention weights over all {data.N * 3} key tokens</div>
            <TreeWeights data={data} phase={ph} state={state} sel={sel} />
          </section>

        </main>

        <aside className="ra-side">
          <CodeBlock title="tree_attn_decode.py" lines={TREE_SRC} focus={focus.main} notes={notes.main} maxHeight={520} note={<RankPicker N={N} sel={sel} onSel={setSel} />} />
          <CodeBlock title="tree_allreduce (binomial tree)" lines={TREE_AR_SRC} focus={focus.ar} notes={notes.ar} maxHeight={420} note={ph.op ? `${ph.op} · ${ph.type === "up" ? "reduce" : "broadcast"} · step ${ph.step}` : null} />
        </aside>
      </div>
    </div>
  );
}
