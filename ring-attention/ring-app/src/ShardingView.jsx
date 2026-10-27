import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS } from "./ringModel";
import { SHARD_SRC, PREP_SRC, DRIVER_SRC, LSE_SRC, RING_LSE_SRC, program } from "./shardModel";
import { CodeBlock } from "./CodePanel";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich } from "./ui";

const METHODS = [
  { id: "ring", label: "Ring", file: "test utils · varlen ring (extract_local)" },
  { id: "zigzag", label: "Zig-zag", file: "test utils · varlen zig-zag (extract_local)" },
  { id: "stripe", label: "Stripe", file: "test utils · stripe (extract_local)" },
  { id: "tree", label: "Tree (decode)", file: "illustrative · decode sharding" },
];
const PREP_ONE = [
  { text: "value = seq0", tags: ["cu"] },
  { text: "cu_seqlens = torch.tensor([0, S], dtype=torch.int32)", tags: ["cu"] },
];
const STEP_COLOR = { t_qkv: T.wire, t_ref: T.soft, t_out: T.accent, t_cut: T.accent2, t_ring: T.wire, t_rcut: T.accent2, t_share: T.accent, t_cmp: T.accent, lens: T.soft, pack: T.wire, cu: T.accent2, def: T.soft, seq: T.wire, chunk: T.accent, pick: T.accent2, cat: T.accent, split: T.wire, stack: T.accent, transpose: T.accent2, slice: T.accent, q: T.wire, kv: T.accent, ret: T.soft };

function Cell({ it, small }) {
  const w = small ? 22 : 27;
  if (it.gap) return <span style={{ width: small ? 6 : 10, flexShrink: 0 }} />;
  if (it.pad) return <span title="padding" style={{ width: w, height: w, flexShrink: 0, borderRadius: 5, border: `1px dashed ${T.rule}`, background: `repeating-linear-gradient(45deg, transparent 0 3px, ${T.rule} 3px 4px)` }} />;
  if (it.empty) return <span style={{ fontFamily: mono, fontSize: 12, color: T.dim, padding: "0 4px" }}>{it.empty}</span>;
  const c = it.q ? T.accent : it.neutral ? T.soft : DEV_COLORS[it.own];
  return (
    <span
      title={it.q ? "query token" : it.lse ? `lse of query token ${it.t} (GPU ${it.own})` : `token ${it.t} → GPU ${it.own}`}
      style={{
        width: w, height: w, flexShrink: 0, display: "inline-flex", alignItems: "center", justifyContent: "center",
        fontFamily: mono, fontSize: small ? 9.5 : 11, fontWeight: it.hl ? 700 : 500, borderRadius: it.q || it.lse ? w / 2 : 5,
        color: it.hl ? T.bg : c, background: it.hl ? c : `${c}26`,
        border: `1.5px solid ${it.isNew ? T.ink : c}`, opacity: it.dim ? 0.22 : 1,
        boxShadow: it.isNew ? `0 0 0 2px ${T.ink}55` : "none", transition: "all .3s",
      }}
    >
      {it.q ? "q" : it.t}
    </span>
  );
}

function Block({ b, sel, small }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginBottom: 6 }}>{b.title}</div>
      <div style={{ display: "grid", gap: 5 }}>
        {b.rows.map((row, ri) => {
          const on = row.rank === sel;
          let pos = 0;
          return (
            <div key={ri} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 56, flexShrink: 0, fontFamily: mono, fontSize: 11, textAlign: "right",
                color: row.rank != null ? DEV_COLORS[row.rank] : T.soft, fontWeight: on ? 700 : 400 }}>{row.label}</span>
              {row.items.some((x) => x.mark != null) ? (
                <SegmentRow items={row.items} small={small} />
              ) : b.chunkLabels ? (
                <ChunkRow items={row.items} labels={b.chunkLabels} small={small} />
              ) : (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3, alignItems: "flex-start", paddingTop: on ? 3 : 0, paddingLeft: on ? 3 : 0, paddingRight: on ? 3 : 0, paddingBottom: on ? 3 : 0, borderRadius: 7, outline: on ? `1px solid ${DEV_COLORS[row.rank]}88` : "none" }}>
                  {row.items.map((it, k) => {
                    const idx = !it.gap && !it.empty ? pos++ : null;
                    return (
                      <span key={k} style={{ display: "inline-flex", flexDirection: "column", alignItems: "center" }}>
                        <Cell it={it} small={small} />
                        {b.localIdx && idx != null && <span style={{ fontFamily: mono, fontSize: 8.5, color: T.dim, marginTop: 1 }}>p{idx}</span>}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// A packed row cut at its cu_seqlens marks: each sequence is one unbreakable
// segment headed by the cu_seqlens entry where it starts.
function SegmentRow({ items, small }) {
  const segs = [];
  items.forEach((it) => {
    if (it.mark != null) segs.push({ mark: it.mark, name: it.markName || "cu_seqlens", items: [] });
    else if (!it.gap && segs.length) segs[segs.length - 1].items.push(it);
  });
  const tick = (k, mark, last, name) => (
    <span style={{ fontFamily: mono, fontSize: 10, fontWeight: 700, color: T.accent2, whiteSpace: "nowrap" }}>
      {name}[{k}] = {mark}{last ? "  (end)" : ""}
    </span>
  );
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10, rowGap: 8, flex: 1, minWidth: 0 }}>
      {segs.map((sg, k) => (
        <div key={k} style={{ display: "flex", flexDirection: "column", gap: 3, borderLeft: `2px solid ${T.accent2}`, paddingLeft: 5, maxWidth: "100%", minWidth: 0, boxSizing: "border-box" }}>
          {tick(k, sg.mark, sg.items.length === 0, sg.name)}
          {/* a sequence longer than the panel wraps; its tick stays at the head */}
          {sg.items.length > 0 && <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>{sg.items.map((it, j) => <Cell key={j} it={it} small={small} />)}</div>}
        </div>
      ))}
    </div>
  );
}

// A chunked sequence: one labelled group per chunk ("c3 → GPU 3").
function ChunkRow({ items, labels, small }) {
  const groups = [[]];
  items.forEach((it) => (it.gap ? groups.push([]) : groups[groups.length - 1].push(it)));
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: small ? 6 : 10 }}>
      {groups.map((g, j) => {
        const l = labels[j];
        return (
          <div key={j} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
            <span style={{ fontFamily: mono, fontSize: 9.5, color: l.mine ? T.ink : DEV_COLORS[l.own], fontWeight: l.mine ? 700 : 400 }}>c{l.j}→{l.own}</span>
            <div style={{ display: "flex", gap: 3 }}>{g.map((it, k) => <Cell key={k} it={it} small={small} />)}</div>
          </div>
        );
      })}
    </div>
  );
}

export default function ShardingView() {
  const [method, setMethod] = useState("ring");
  const [packed, setPacked] = useState(true);
  const [N, setN] = useState(4);
  const [sel, setSel] = useState(1);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dim3, setDim3] = useState(false);
  const varlenOk = method === "ring" || method === "zigzag";
  const prog = useMemo(() => program(method, N, varlenOk && packed, sel, dim3), [method, N, packed, sel, varlenOk, dim3]);
  const last = prog.steps.length - 1;
  const step = prog.steps[Math.min(idx, last)];
  const small = prog.S > 24;

  const reset = (fn) => (v) => { fn(v); setIdx(0); setPlaying(false); };
  const changeN = (n) => { setN(n); setIdx(0); setSel((r) => Math.min(r, n - 1)); };

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
    const t = setTimeout(() => setIdx((i) => i + 1), 1700);
    return () => clearTimeout(t);
  }, [playing, idx, last]);

  const sc = STEP_COLOR[step.tag] || T.accent;
  const isPrep = step.code === "prep";
  const isTest = step.code === "test";
  const inLocal = !isPrep && !isTest; // stepping inside extract_local(qkv)
  const focus = {};
  if (inLocal) for (const t of step.focus || [step.tag]) focus[t] = t === "loop" ? `dim:${T.soft}` : sc;
  if (step.tag === "t_out") for (const t of step.focus) focus[t] = `dim:${sc}`;
  const driverFocus = isTest ? { [step.tag]: sc } : { t_qkv: `dim:${T.wire}` };
  const lseFocus = step.lse ? { [dim3 ? "d3" : "d2"]: sc } : {};
  const showLocal = inLocal || step.tag === "t_qkv" || step.tag === "t_out";
  const prepFocus = isPrep ? { [step.tag]: sc } : {};
  // once past the prep steps, the prep code keeps its final values as comments
  const prepFinal = prog.steps.filter((x) => x.code === "prep").at(-1)?.prepNotes ?? [];
  const m = METHODS.find((x) => x.id === method);
  const btn = (on, c = T.accent) => ({ ...navBtn(false), height: 28, padding: "0 11px", fontSize: 12, color: on ? T.bg : T.soft, background: on ? c : T.well, borderColor: on ? c : T.rule });

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SEQUENCE PARALLELISM · HOW THE DATA IS SHARDED</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>How the sharding code works, line by line</h1>
        <Rich
          style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 1000 }}
          html="Step through each method's <code>extract_local</code>: the lit line is the one running, and the comments beside the code show what each line evaluates to right now. Ring and zig-zag use the varlen version, which needs <code>cu_seqlens</code>: the walkthrough starts by showing where that comes from, for a packed batch or a single sequence. Then it follows the repo's test back out: the same <code>extract_local</code> cuts the reference <code>out</code>, and <code>extract_lse</code> cuts its <code>lse</code>, so they can be compared row by row with what ring attention returns."
        />
      </header>

      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center", marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {METHODS.map((x) => <button key={x.id} onClick={() => reset(setMethod)(x.id)} style={btn(method === x.id)}>{x.label}</button>)}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", opacity: varlenOk ? 1 : 0.4 }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>input</span>
          <button disabled={!varlenOk} onClick={() => reset(setPacked)(false)} style={btn(varlenOk && !packed, T.wire)}>one sequence</button>
          <button disabled={!varlenOk} onClick={() => reset(setPacked)(true)} style={btn(varlenOk && packed, T.wire)}>packed (varlen)</button>
        </div>
        {varlenOk && (
          <div style={{ display: "flex", gap: 6, alignItems: "center" }} title="flash-attn returns varlen lse packed (newer) or padded (older)">
            <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>lse layout</span>
            <button onClick={() => setDim3(false)} style={btn(!dim3, T.accent2)}>dim 2 · [H, total]</button>
            <button onClick={() => setDim3(true)} style={btn(dim3, T.accent2)}>dim 3 · [num_seq, H, max]</button>
          </div>
        )}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>N</span>
          {[2, 4, 8].map((n) => <button key={n} onClick={() => changeN(n)} style={btn(n === N)}>{n}</button>)}
        </div>
        <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>rank</span>
          {Array.from({ length: N }, (_, r) => (
            <button key={r} onClick={() => setSel(r)} style={{ ...navBtn(false), height: 28, padding: "0 9px", color: DEV_COLORS[r], borderColor: r === sel ? DEV_COLORS[r] : T.rule, background: r === sel ? `${DEV_COLORS[r]}22` : T.well, fontWeight: r === sel ? 700 : 400 }}>{r}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: "1 1 300px", justifyContent: "flex-end" }}>
          <button onClick={() => setIdx(0)} disabled={idx === 0} style={navBtn(idx === 0)} aria-label="Restart">⟲</button>
          <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} style={navBtn(idx === 0)}>← prev</button>
          <button onClick={() => { if (idx >= last) { setIdx(0); setPlaying(true); } else setPlaying(!playing); }} style={{ ...navBtn(false), color: T.accent, borderColor: `${T.accent}88` }}>{playing && idx < last ? "❚❚ pause" : "▶ play"}</button>
          <button onClick={() => setIdx(idx + 1)} disabled={idx === last} style={navBtn(idx === last)}>next →</button>
        </div>
        <div style={{ display: "flex", gap: 3, width: "100%" }}>
          {prog.steps.map((s, i) => (
            <button key={i} onClick={() => setIdx(i)} title={s.title} aria-label={`Step ${i + 1}`}
              style={{ flex: 1, height: 8, border: "none", padding: 0, borderRadius: 3, cursor: "pointer",
                background: i === idx ? STEP_COLOR[s.tag] : i < idx ? `${STEP_COLOR[s.tag]}66` : T.rule, outline: i === idx ? `1px solid ${T.ink}` : "none" }} />
          ))}
        </div>
      </div>

      <div className="ra-grid">
        <main style={{ display: "grid", gap: 18, minWidth: 0 }}>
          <section style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{step.title}</div>
              <div style={{ display: "flex", gap: 6 }}>
                <span style={chip(T.accent)}>step {Math.min(idx, last) + 1} / {last + 1}</span>
                {varlenOk && <span style={chip(T.wire)}>cu_seqlens [{prog.cu.join(", ")}]</span>}
              </div>
            </div>
            <Rich style={{ fontSize: 14, color: T.soft, lineHeight: 1.65, marginBottom: 14 }} html={step.desc} />
            {step.blocks.map((b, i) => <Block key={`${method}-${i}`} b={b} sel={sel} small={small} />)}
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontFamily: mono, fontSize: 11, color: T.dim, borderTop: `1px solid ${T.rule}`, paddingTop: 10 }}>
              <span>grey = not assigned yet · colour = GPU the token ends up on</span>
              {varlenOk && <span style={{ color: T.accent2 }}>| = cu_seqlens boundary</span>}
              <span>filled = rank {sel}'s tokens at this step</span>
              <span>white ring = just appended</span>
              {isTest && <span>round = one query row's lse (one head)</span>}
              {method === "tree" && <span>q = new query token</span>}
            </div>
          </section>

        </main>

        <aside className="ra-side">
          {varlenOk && (isPrep || inLocal) && (
            <CodeBlock title={packed ? "building a varlen batch (typical, not from the repo)" : "one sequence"} lines={packed ? PREP_SRC : PREP_ONE} focus={prepFocus} notes={isPrep ? step.prepNotes : prepFinal} maxHeight={220} />
          )}
          {varlenOk && !isPrep && (
            <CodeBlock title={`test_${method === "zigzag" ? "zigzag_" : ""}ring_flash_attn_varlen_func.py (abridged)`} lines={DRIVER_SRC[method]} focus={driverFocus} notes={step.driverNotes} maxHeight={inLocal ? 150 : 330} note={`rank = ${sel}, world_size = ${N}`} />
          )}
          {(!varlenOk || showLocal) && (
            <CodeBlock title={m.file} lines={SHARD_SRC[method]} focus={focus} notes={isPrep ? [] : step.notes} maxHeight={420} note={`rank = ${sel}, world_size = ${N}`} />
          )}
          {step.lse && <CodeBlock title="extract_lse (test)" lines={LSE_SRC} focus={lseFocus} notes={step.lseNotes} maxHeight={330} />}
          {step.tag === "t_ring" && (
            <CodeBlock title="ring_flash_attn_varlen.py · lse layout inside the ring" lines={RING_LSE_SRC} focus={{ [dim3 ? "d3" : "d2"]: T.wire, merge: T.accent2 }} notes={step.ringNotes} maxHeight={300} />
          )}
        </aside>
      </div>

    </div>
  );
}
