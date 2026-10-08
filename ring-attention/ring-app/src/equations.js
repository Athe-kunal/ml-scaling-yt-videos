import { DEV_COLORS, mod, deviceStep, covered, ownChunks } from "./ringModel";

// Equations for the selected GPU r at the current phase, numbers plugged in.
// Items are { p: html-with-$math$ } or { m: display tex }.

const col = (j, tex) => String.raw`\textcolor{${DEV_COLORS[j]}}{${tex}}`;
const KV = (j) => col(j, `KV_{${j}}`);

export function equations(variant, N, r, phase) {
  const { type, s } = phase;
  const zig = variant === "zigzag";
  const nxt = mod(r + 1, N), prv = mod(r - 1, N);
  const a = r, b = 2 * N - 1 - r;
  const out = [];
  const P = (p) => out.push({ p });
  const M = (m) => out.push({ m });

  if (type === "init") {
    P("<code>RingComm</code> fixes each GPU's two neighbours once, before the loop. Data only ever moves clockwise:");
    M(String.raw`\texttt{send\_rank} = (r+1) \bmod N = (${r}+1) \bmod ${N} = ${nxt}`);
    M(String.raw`\texttt{recv\_rank} = (r-1) \bmod N = (${r}-1) \bmod ${N} = ${prv}`);
    if (!zig) {
      P(`The sequence is cut into $2N=${2 * N}$ chunks $c_0,\\dots,c_{${2 * N - 1}}$ of $L/2$ tokens (same chunks as the zig-zag view). Plain ring gives GPU $r$ two <b>neighbouring</b> chunks:`);
      M(String.raw`Q_{r} = \big[\,Q_{c_{2r}}\,;\,Q_{c_{2r+1}}\big] = \big[\,Q_{c_{${2 * r}}}\,;\,Q_{c_{${2 * r + 1}}}\big] = X\big[\,${r}L : ${r + 1}L\,\big]\,W_Q`);
      P("Same for $K, V$. $Q_r$ never moves. The $K,V$ blocks rotate, so after $N$ steps every query chunk has met every key chunk. Because both chunks sit together, GPU 0 holds only early tokens and GPU $N-1$ only late ones: that is the imbalance."); 
    } else {
      P(`The sequence is cut into $2N=${2 * N}$ chunks $c_0,\\dots,c_{${2 * N - 1}}$ of $L/2$ tokens. <code>extract_local</code> gives GPU $r$ one early and one late chunk:`);
      M(String.raw`Q_{r} = \big[\,Q_{c_{r}}\,;\,Q_{c_{2N-1-r}}\big] = \big[\,Q_{c_{${a}}}\,;\,Q_{c_{${b}}}\big]`);
      P(String.raw`Same for $K, V$. Inside a shard the order is still increasing ($c_{${a}} < c_{${b}}$), which is what lets step 0 use an ordinary causal mask. <code>q1 = q[:, L/2:]</code> is the late half $Q_{c_{${b}}}$.`);
    }
    P("Accumulators start empty: <code>out = lse = None</code>.");
    return out;
  }

  if (type === "final") {
    const js = covered(variant, N, r, phase);
    P(`After $N=${N}$ steps GPU ${r} has merged every visible block:`);
    M(String.raw`\ell_{${r}} = \log\!\sum_{j\in\{${js.join(",")}\}} e^{\ell^{(j)}_{${r}}},\qquad O_{${r}} = \sum_{j\in\{${js.join(",")}\}} e^{\,\ell^{(j)}_{${r}}-\ell_{${r}}}\;O^{(j)}_{${r}}`);
    P("That is exactly softmax over the concatenated keys: each block's output is reweighted by its share of the total normaliser. <code>out</code> is cast back to <code>q.dtype</code>; <code>lse</code> is reshaped to $[B, H, L]$ for the backward pass.");
    P(String.raw`Comms per GPU: $N-1 = ${N - 1}$ hops of one KV block, each hidden behind one step of compute.`);
    if (zig) P("The output is still in zig-zag order; gather and undo <code>extract_local</code> to get token order back.");
    return out;
  }

  const d = deviceStep(variant, N, r, s);
  const j = d.j, jn = mod(r - s - 1, N);
  const [j0, j1] = ownChunks(variant, N, j);
  const [q0, q1] = ownChunks(variant, N, r);
  const kvj = String.raw`[K_{c_{${j0}}};K_{c_{${j1}}}]`;

  if (type === "send") {
    P(`Step ${s}: before any compute, every GPU posts a non-blocking <code>isend</code> of the KV it holds to <b>next</b> and an <code>irecv</code> from <b>prev</b> into fresh buffers <code>next_k, next_v</code>. <code>commit()</code> fires all four ops in one <code>batch_isend_irecv</code>.`);
    M(String.raw`\text{GPU}_{${r}} \text{ holds } KV^{(${s})}_{${r}} = KV_{(r-s)\bmod N} = ${KV(j)}`);
    M(String.raw`${KV(j)} \;\xrightarrow{\;\texttt{isend}\;}\; \text{GPU}_{${nxt}}\quad(\texttt{send\_rank})`);
    M(String.raw`\texttt{next\_k},\texttt{next\_v} \;\xleftarrow{\;\texttt{irecv}\;}\; \text{GPU}_{${prv}}\ \text{sends}\ KV^{(${s})}_{${prv}} = ${KV(jn)}`);
    P("The call returns immediately; the bytes move while the kernel below runs.");
    return out;
  }

  if (type === "wait") {
    P("<code>comm.wait()</code> blocks until this step's isend/irecv pair completes, then the buffers swap. This is the only sync point in the loop.");
    M(String.raw`K^{(${s + 1})}_{${r}},V^{(${s + 1})}_{${r}} \leftarrow \texttt{next\_k},\texttt{next\_v} = KV^{(${s})}_{${prv}} = ${KV(jn)}`);
    M(String.raw`KV^{(s+1)}_{r} = KV^{(s)}_{(r-1)\bmod N} \;\Rightarrow\; KV^{(s)}_{r} = KV_{(r-s)\bmod N}`);
    P(`So at step $s$ GPU $r$ sees the block that started $s$ hops upstream. Next step GPU ${r} works on ${"$"}${KV(jn)}${"$"}.`);
    return out;
  }

  if (type === "compute") {
    if (s === N - 1) P("Last step: <code>step + 1 == world_size</code>, so nothing was sent. Compute only.");
    if (!zig) {
      if (d.branch === "0") {
        P(`Step 0 is the diagonal block: GPU ${r}'s queries against its own keys, so the causal mask is needed (<code>causal and step == 0</code> is True).`);
        M(String.raw`O^{(0)}_{${r}},\ \ell^{(0)}_{${r}} = \mathrm{FlashAttn}\big([Q_{c_{${q0}}};Q_{c_{${q1}}}],\ [K_{c_{${q0}}};K_{c_{${q1}}}],\ [V_{c_{${q0}}};V_{c_{${q1}}}];\ \text{causal}\big)`);
        P(String.raw`Chunk by chunk: $Q_{c_{${q0}}}{\cdot}K_{c_{${q0}}}$ and $Q_{c_{${q1}}}{\cdot}K_{c_{${q1}}}$ are triangles, $Q_{c_{${q1}}}{\cdot}K_{c_{${q0}}}$ is full. Work: $\tfrac12+1+\tfrac12 = 2$ chunk², the same on every GPU.`);
      } else if (d.active) {
        P(`<code>step &lt;= rank</code> (${s} ≤ ${r}), so the block came from <b>earlier</b> in the sequence: $j = ${j} < r = ${r}$. Every key precedes every query, so no mask.`);
        M(String.raw`c_{${j0}},\,c_{${j1}} \;<\; c_{${q0}},\,c_{${q1}} \quad\Rightarrow\quad \text{all 4 chunk pairs visible}`);
        M(String.raw`O^{(${s})}_{${r}} = \mathrm{softmax}\!\Big(\tfrac{[Q_{c_{${q0}}};Q_{c_{${q1}}}]\,${col(j, kvj)}^{\!\top}}{\sqrt d}\Big)\,${col(j, "V")},\qquad \ell^{(${s})}_{${r}} = \log\sum_k \exp\!\big(\cdot\big)_k`);
        P("Full $L\\times L$ block = 4 chunk²: twice the diagonal step, and twice what any zig-zag step costs.");
      } else {
        P(`<code>step &lt;= rank</code> is False (${s} > ${r}): the block wrapped around from <b>later</b> in the sequence, $j = ${j} > r = ${r}$.`);
        M(String.raw`c_{${j0}},\,c_{${j1}} \;>\; c_{${q0}},\,c_{${q1}} \;\Rightarrow\; \text{all 4 chunk pairs masked: } 0 \text{ chunk}^2`);
        P(`GPU ${r} skips the kernel and just relays the KV. Meanwhile GPU ${N - 1} computes a full block, so GPU ${r} sits idle. This is the load imbalance.`);
      }
      return out;
    }
    if (d.branch === "0") {
      P("Step 0: the GPU's own two chunks. Local order matches global order, so a plain causal mask is exact:");
      M(String.raw`\mathrm{FlashAttn}\big([Q_{c_{${a}}};Q_{c_{${b}}}],\ [K_{c_{${a}}};K_{c_{${b}}}],\ [V_{c_{${a}}};V_{c_{${b}}}];\ \text{causal}\big)`);
      P(String.raw`$Q_{c_{${a}}}{\cdot}K_{c_{${a}}}$ and $Q_{c_{${b}}}{\cdot}K_{c_{${b}}}$ are triangles, $Q_{c_{${b}}}{\cdot}K_{c_{${a}}}$ is full, $Q_{c_{${a}}}{\cdot}K_{c_{${b}}}$ is masked. Work: $\tfrac12+1+\tfrac12 = 2$ chunk².`);
    } else if (d.branch === "lo") {
      P(String.raw`<code>step &lt;= rank</code>: the KV came from $j=${j} < r=${r}$. Check which chunks are visible:`);
      M(String.raw`c_{${j}} < c_{${a}} < c_{${b}} \quad\Rightarrow\quad \text{both query chunks see } K_{c_{${j}}}`);
      M(String.raw`c_{${2 * N - 1 - j}} > c_{${b}} \quad\Rightarrow\quad K_{c_{${2 * N - 1 - j}}} \text{ is in everyone's future}`);
      M(String.raw`O_b = \mathrm{FlashAttn}\big([Q_{c_{${a}}};Q_{c_{${b}}}],\ ${col(j, `K_{c_{${j}}}`)},\ ${col(j, `V_{c_{${j}}}`)}\big)\quad(\texttt{k0 = k[:, :L/2]})`);
      P("All of $Q$ × the first half of $KV$: $L \\times L/2$, no mask. Work: 2 chunk².");
    } else {
      P(String.raw`<code>else</code>: the KV came from $j=${j} > r=${r}$. Now:`);
      M(String.raw`c_{${j}} > c_{${a}} \quad\Rightarrow\quad Q_{c_{${a}}} \text{ sees nothing (drop it)}`);
      M(String.raw`c_{${j}},\ c_{${2 * N - 1 - j}} < c_{${b}} \quad\Rightarrow\quad Q_{c_{${b}}} \text{ sees all of } ${kvj}`);
      M(String.raw`O_b = \mathrm{FlashAttn}\big(Q_{c_{${b}}},\ ${col(j, kvj)},\ ${col(j, "V")}\big)\quad(\texttt{q1 = q[:, L/2:]})`);
      P("Second half of $Q$ × all of $KV$: $L/2 \\times L$, no mask. Work: 2 chunk², the same as the other branch.");
    }
    P("Every GPU does the same amount of work at every step, so nobody waits at <code>comm.wait()</code>.");
    return out;
  }

  // merge
  if (!d.active) {
    P(`GPU ${r} skipped the kernel, so <code>update_out_and_lse</code> is not called: <code>out</code> and <code>lse</code> are unchanged.`);
    return out;
  }
  const js = covered(variant, N, r, phase);
  if (s === 0) {
    P("First block: <code>out is None</code>, so <code>update_out_and_lse</code> just stores it (upcast to fp32).");
    M(String.raw`\text{out} \leftarrow O^{(0)}_{${r}},\qquad \text{lse} \leftarrow \ell^{(0)}_{${r}}`);
    return out;
  }
  P("Online-softmax merge. With running $(O, \\ell)$ and the new block $(O_b, \\ell_b)$:");
  M(String.raw`\ell' = \log\!\big(e^{\ell} + e^{\ell_b}\big) = \ell - \log\sigma(\ell - \ell_b)`);
  M(String.raw`O' = \frac{e^{\ell}O + e^{\ell_b}O_b}{e^{\ell'}} = O - \sigma(\ell_b - \ell)\,\big(O - O_b\big)`);
  P(String.raw`Because $e^{\ell_b-\ell'} = \sigma(\ell_b-\ell)$ and $e^{\ell-\ell'} = 1-\sigma(\ell_b-\ell)$. The sigmoid form never exponentiates a large number, so it is stable in fp32.`);
  if (d.branch === "hi") {
    P(String.raw`Only the late half $Q_{c_{${b}}}$ was computed, so the merge runs on <code>slice_ = (:, L/2:)</code>. Rows of $Q_{c_{${a}}}$ keep their old $(O,\ell)$:`);
    M(String.raw`O[c_{${b}}] \leftarrow O[c_{${b}}] - \sigma\big(\ell_b - \ell[c_{${b}}]\big)\big(O[c_{${b}}] - O_b\big)`);
  }
  M(String.raw`\text{out}_{${r}} \text{ now covers } KV_{\{${js.join(",")}\}}`);
  return out;
}

// One line per GPU for the transfers of this step.
export function transfers(variant, N, phase) {
  const { type, s } = phase;
  if (!["send", "compute", "merge", "wait"].includes(type) || s >= N - 1) return [];
  return Array.from({ length: N }, (_, i) => {
    const j = mod(i - s, N);
    const [c0, c1] = ownChunks(variant, N, j);
    const tex = String.raw`[K,V]_{c_{${c0}}, c_{${c1}}}`;
    return { from: i, to: mod(i + 1, N), j, tex: col(j, tex) };
  });
}
