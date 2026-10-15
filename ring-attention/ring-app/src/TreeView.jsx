import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { katexHtml } from "./latex";
import { DEV_COLORS } from "./ringModel";
import { TREE_PHASE, PER, makeData, buildPhases, stateAt, role, log2, fmt, vec } from "./treeModel";
import { TREE_SRC, TREE_AR_SRC } from "./codeSources";
import { CodeBlock } from "./CodePanel";
import TreeDiagram from "./TreeDiagram";
import { panel, h2, chip, navBtn } from "./styles";
import { Rich, Tex } from "./ui";

const tvec = (v) => String.raw`\begin{bmatrix}${v.map((x) => fmt(x)).join(" & ")}\end{bmatrix}`;

function phaseTitle(ph) {
  if (ph.type === "up" || ph.type === "down") return `${ph.op === "max" ? "max" : "sum"} ${TREE_PHASE[ph.type].label} · level ${ph.lvl}`;
  return TREE_PHASE[ph.type].label;
}

function equations(data, phases, idx, r) {
  const ph = phases[idx];
  const st = stateAt(data, phases, idx);
  const prev = idx > 0 ? stateAt(data, phases, idx - 1) : st;
  const g = data.gpus[r], N = data.N;
  const out = [];
  const P = (p) => out.push({ p });
  const M = (m) => out.push({ m });

  if (ph.type === "init") {
    P(`Decoding: one new query token $q$ (replicated on every GPU) must attend to a KV cache of $S = ${PER}N = ${PER * N}$ tokens, sharded by sequence. GPU ${r} holds keys ${PER * r}..${PER * r + PER - 1}. The target is plain softmax attention over all of them:`);
    M(String.raw`o = \sum_{j=0}^{S-1} \frac{e^{s_j}}{\sum_{j'} e^{s_{j'}}}\, v_j,\qquad s_j = \frac{q\cdot k_j}{\sqrt d}`);
    P("Ring attention would rotate the KV shards through every GPU, $N-1$ hops of big tensors. Here K and V never move: only small partial results do.");
    M(String.raw`q = ${tvec(data.q)}`);
    return out;
  }
  if (ph.type === "local") {
    P(`Every GPU runs flash-attention on its own shard, in parallel. For GPU ${r}:`);
    M(String.raw`s = \big[${g.scores.map((x) => fmt(x)).join(",\\ ")}\big]`);
    M(String.raw`\ell_{${r}} = \log\sum_{j \in \text{shard } ${r}} e^{s_j} = ${fmt(g.lse, 3)},\qquad o_{${r}} = \sum_{j} e^{s_j - \ell_{${r}}}\,v_j = ${tvec(g.o)}`);
    P("Each $o_i$ is normalised only over its own shard. To combine them we need the global normaliser, which is a sum over GPUs: an Allreduce.");
    return out;
  }
  if (ph.type === "rescale") {
    const m = st.gpus[r].mx;
    P(`Every GPU now has the same $m = \\max_i \\ell_i = ${fmt(m, 3)}$. Rescale locally (no comms):`);
    M(String.raw`w_{${r}} = e^{\ell_{${r}} - m} = e^{${fmt(g.lse, 3)} - ${fmt(m, 3)}} = ${fmt(st.gpus[r].w, 4)}`);
    M(String.raw`n_{${r}} = o_{${r}}\,w_{${r}} = \sum_{j\in ${r}} e^{s_j - m}\,v_j = ${tvec(st.gpus[r].num)},\qquad d_{${r}} = w_{${r}} = \sum_{j\in ${r}} e^{s_j - m}`);
    P("Subtracting the global max means every $w_i \\le 1$, so nothing overflows however long the context is.");
    return out;
  }
  if (ph.type === "final") {
    const s = st.gpus[r];
    P("After the sum-allreduce every GPU holds the global $(n, d)$. One division:");
    M(String.raw`o = \frac{\sum_i n_i}{\sum_i d_i} = \frac{\sum_j e^{s_j - m}\,v_j}{\sum_j e^{s_j - m}} = ${tvec(s.out)}`);
    const err = Math.max(...s.out.map((x, t) => Math.abs(x - data.exact[t])));
    M(String.raw`\text{exact softmax over all } ${PER * N} \text{ keys} = ${tvec(data.exact)},\qquad \max|\Delta| = ${err.toExponential(1)}`);
    P("Same answer as one GPU holding the whole cache. The same works with the ring tab's merge $\\oplus$ (<code>_update_out_and_lse</code>): it is associative, so it can be reduced in a tree too. Max + sum is that merge split into two collectives.");
    return out;
  }

  // up / down
  const ro = role(r, ph);
  const isMax = ph.op === "max";
  const sym = isMax ? "m" : "d";
  const v = (s) => (isMax ? s.mx : s.den);
  const span = 2 ** ph.lvl;
  if (ph.type === "up") {
    P(`Reduce, level ${ph.lvl} (<code>step = ${ph.step}</code>). Ranks with <code>rank % ${2 * ph.step} == 0</code> receive from <code>rank + ${ph.step}</code>; ranks with remainder ${ph.step} send; the rest already sent at a lower level.`);
    if (ro.kind === "recv") {
      const a = v(prev.gpus[r]), b = v(prev.gpus[ro.peer]);
      if (isMax) M(String.raw`m_{[${r}..${r + span - 1}]} = \max\big(m_{[${r}..${r + ph.step - 1}]},\ m_{[${ro.peer}..${ro.peer + ph.step - 1}]}\big) = \max(${fmt(a, 3)},\ ${fmt(b, 3)}) = ${fmt(v(st.gpus[r]), 3)}`);
      else {
        M(String.raw`d_{[${r}..${r + span - 1}]} = d_{[${r}..${r + ph.step - 1}]} + d_{[${ro.peer}..${ro.peer + ph.step - 1}]} = ${fmt(a, 3)} + ${fmt(b, 3)} = ${fmt(v(st.gpus[r]), 3)}`);
        M(String.raw`n_{[${r}..${r + span - 1}]} = ${tvec(prev.gpus[r].num)} + ${tvec(prev.gpus[ro.peer].num)} = ${tvec(st.gpus[r].num)}`);
      }
      P(`GPU ${r} receives from GPU ${ro.peer} and now holds the result for GPUs ${r}..${r + span - 1}.`);
    } else if (ro.kind === "send") {
      P(`GPU ${r} sends its partial ${isMax ? "max" : "$(n, d)$"} for GPUs ${r}..${r + ph.step - 1} to GPU ${ro.peer} and is done until the broadcast.`);
      M(String.raw`\text{GPU}_{${r}} \xrightarrow{\;${sym} = ${fmt(v(prev.gpus[r]), 3)}\;} \text{GPU}_{${ro.peer}}`);
    } else {
      P(`GPU ${r} is idle at this level: it handed its value up at an earlier level.`);
    }
    M(String.raw`\text{after level } \ell:\ \text{GPU } k\,2^{\ell} \text{ holds the ${isMax ? "max" : "sum"} over GPUs } [k\,2^{\ell},\,(k+1)\,2^{\ell})`);
    if (ph.lvl === log2(N)) P(`Root reached: GPU 0 has the global ${isMax ? "$m$" : "$(n, d)$"} after only $\\log_2 N = ${log2(N)}$ levels.`);
  } else {
    P(`Broadcast, level ${ph.lvl} (<code>step = ${ph.step}</code>): the same tree in reverse. Ranks with <code>rank % ${2 * ph.step} == 0</code> send to <code>rank + ${ph.step}</code>.`);
    if (ro.kind === "recv") M(String.raw`\text{GPU}_{${ro.peer}} \xrightarrow{\;${sym} = ${fmt(v(st.gpus[r]), 3)}\;} \text{GPU}_{${r}}\quad(\text{global})`);
    else if (ro.kind === "send") M(String.raw`\text{GPU}_{${r}} \xrightarrow{\;${sym} = ${fmt(v(st.gpus[r]), 3)}\;} \text{GPU}_{${ro.peer}}`);
    else P(`GPU ${r} is idle at this level (${st.gpus[r].got || r === 0 ? "it already has the global value" : "it receives at a lower level"}).`);
    if (ph.lvl === 1) P(`Every GPU now holds the global ${isMax ? "max" : "$(n, d)$"}: $2\\log_2 N = ${2 * log2(N)}$ levels for this allreduce.`);
  }
  return out;
}

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

function variables(data, phases, idx, r) {
  const ph = phases[idx];
  const st = stateAt(data, phases, idx).gpus[r];
  const g = data.gpus[r];
  const rows = [["rank", r], ["N", data.N]];
  if (ph.type === "up" || ph.type === "down") {
    const ro = role(r, ph);
    rows.push(["allreduce", ph.op === "max" ? "lse_max (op=max)" : "buf = [num, den] (op=add)"]);
    rows.push(["loop", ph.type === "up" ? "reduce" : "broadcast"]);
    rows.push(["step", ph.step]);
    rows.push([`rank % ${2 * ph.step}`, r % (2 * ph.step)]);
    rows.push(["action", ro.kind === "recv" ? `recv from ${ro.peer}` : ro.kind === "send" ? `send to ${ro.peer}` : "idle"]);
  }
  rows.push(["local_lse", ph.type === "init" ? "—" : fmt(g.lse, 3)]);
  rows.push(["lse_max", st.mx == null ? "—" : fmt(st.mx, 3)]);
  rows.push(["w", st.w == null ? "—" : fmt(st.w, 4)]);
  rows.push(["den", st.den == null ? "—" : fmt(st.den, 4)]);
  rows.push(["num", vec(st.num)]);
  if (st.out) rows.push(["out", vec(st.out)]);
  return rows;
}

// Decode-time comms for one query token, per head, bf16: ring rotates the KV
// shards; tree reduces (lse) then (num, den).
function CostTable({ N }) {
  const S = 131072, d = 128, B = 2;
  const Ns = [...new Set([N, 8, 64, 512])].sort((a, b) => a - b);
  const bytes = (x) => (x >= 1e6 ? `${(x / 1e6).toFixed(1)} MB` : x >= 1e3 ? `${(x / 1e3).toFixed(1)} KB` : `${x} B`);
  const th = { textAlign: "right", padding: "5px 8px", fontWeight: 500, color: T.dim, borderBottom: `1px solid ${T.rule}` };
  const td = { textAlign: "right", padding: "5px 8px", borderBottom: `1px solid ${T.well}` };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: mono, fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ ...th, textAlign: "left" }}>N</th>
            <th style={th}>ring: sequential hops</th>
            <th style={th}>ring: bytes / GPU</th>
            <th style={th}>tree: depth</th>
            <th style={th}>tree: bytes / GPU</th>
          </tr>
        </thead>
        <tbody>
          {Ns.map((n) => {
            const L = log2(n);
            return (
              <tr key={n} style={{ color: n === N ? T.ink : T.soft }}>
                <td style={{ ...td, textAlign: "left", color: n === N ? T.accent : T.soft }}>{n}</td>
                <td style={td}>{n - 1}</td>
                <td style={td}>{bytes((n - 1) * 2 * (S / n) * d * B)}</td>
                <td style={td}>{4 * L}</td>
                <td style={td}>{bytes(2 * L * (1 + d + 1) * B)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function TreeView() {
  const [N, setN] = useState(8);
  const [sel, setSel] = useState(5);
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
  const eqs = equations(data, phases, i, sel);
  const focus = codeFocus(ph, sel);
  const vars = variables(data, phases, i, sel);
  const pc = TREE_PHASE[ph.type].color;

  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>SEQUENCE PARALLELISM · TREE ATTENTION (DECODING)</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Tree attention: combine partial softmaxes in log N</h1>
        <Rich
          style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 980 }}
          html="When decoding, there is one new query token but a very long KV cache, sharded across $N$ GPUs. Each GPU attends to its own shard; the partial results are combined with two Allreduces (a <b>max</b>, then a <b>sum</b>). Max and sum are associative, so a tree finishes each in $\log_2 N$ levels instead of $N-1$ ring hops, and only $(\ell, n, d)$ moves, never K or V (Shyam et al., 2024)."
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
              html="Dashed edge: left child, same GPU as the parent, no message. Solid edge: the right child's GPU sends. Toy numbers: $d=4$, 3 keys per GPU, seeded."
            />
          </section>

          <section style={panel}>
            <div style={h2}>Equations · GPU {sel} at this phase</div>
            <div style={{ display: "grid", gap: 6, color: T.soft, fontSize: 14.5, lineHeight: 1.6 }}>
              {eqs.map((e, k) => (e.m ? <Tex key={k} tex={e.m} /> : <Rich key={k} html={e.p} />))}
            </div>
          </section>

          <section style={panel}>
            <div style={h2}>Messages this phase</div>
            {state.msgs.length === 0 ? (
              <div style={{ fontFamily: mono, fontSize: 12.5, color: T.dim }}>No communication: {ph.type === "local" || ph.type === "rescale" || ph.type === "final" ? "local compute only." : "nothing sent yet."}</div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: 6 }}>
                {state.msgs.map((m) => {
                  const hot = m.from === sel || m.to === sel;
                  const tex = ph.op === "max" ? String.raw`m = ${fmt(m.val, 3)}` : String.raw`d = ${fmt(m.val, 3)},\ n = [\dots]`;
                  return (
                    <div key={`${m.from}-${m.to}`} style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: mono, fontSize: 12, padding: "4px 10px", borderRadius: 6,
                      background: hot ? `${pc}14` : T.well, border: `1px solid ${hot ? `${pc}66` : T.rule}` }}>
                      <span style={{ color: DEV_COLORS[m.from] }}>GPU {m.from}</span>
                      <span style={{ color: pc }}>→</span>
                      <span dangerouslySetInnerHTML={{ __html: katexHtml(tex) }} />
                      <span style={{ color: pc }}>→</span>
                      <span style={{ color: DEV_COLORS[m.to] }}>GPU {m.to}</span>
                    </div>
                  );
                })}
              </div>
            )}
            <Rich
              style={{ fontSize: 13, color: T.dim, marginTop: 10, lineHeight: 1.5 }}
              html={"Level $\\ell$ of the reduce: GPU $i$ with $i \\bmod 2^{\\ell} = 2^{\\ell-1}$ sends to GPU $i - 2^{\\ell-1}$. The broadcast mirrors it. Message size: one scalar per head for the max, $d+1$ numbers for the sum."}
            />
          </section>

          <section style={panel}>
            <div style={h2}>Ring vs tree for decoding · one query token, one head</div>
            <CostTable N={N} />
            <Rich
              style={{ fontSize: 12.5, color: T.dim, marginTop: 10, lineHeight: 1.55 }}
              html="$S = 131072$, $d = 128$, bf16. Ring attention rotates K and V shards ($2\,\tfrac{S}{N}\,d$ elements per hop, $N-1$ hops). Tree attention never moves KV; it sends $\ell$, then $(n, d)$, along a tree: depth $2\log_2 N$ per allreduce, $4\log_2 N$ for both. At small $N$ the depth can exceed the ring's, but the bytes are tiny; at large $N$ both win. Fusing the two into one allreduce of the $\oplus$ merge halves the depth."
            />
          </section>
        </main>

        <aside className="ra-side">
          <CodeBlock title="tree_attn_decode.py" lines={TREE_SRC} focus={focus.main} maxHeight={420} note={`following GPU ${sel}`} />
          <div style={{ ...panel, padding: "10px 14px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
              <div style={{ ...h2, margin: 0 }}>Live variables · rank {sel}</div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
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
                  <span style={{ color: k === "action" ? (String(v).startsWith("idle") ? T.dim : pc) : T.ink }}>{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
          <CodeBlock title="tree_allreduce (binomial tree)" lines={TREE_AR_SRC} focus={focus.ar} maxHeight={330} note={ph.op ? `${ph.op} · ${ph.type === "up" ? "reduce" : "broadcast"} · step ${ph.step}` : null} />
        </aside>
      </div>
    </div>
  );
}
