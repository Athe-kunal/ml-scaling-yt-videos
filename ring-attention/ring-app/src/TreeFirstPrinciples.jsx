import { useEffect, useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { DEV_COLORS } from "./ringModel";
import { Q, K, V, NTOK, scores, lse, attention, shard, partials, energy, fp16exp, f, fv } from "./tfpModel";
import { CodeBlock } from "./CodePanel";
import { panel, h2, navBtn } from "./styles";
import { Rich, Tex } from "./ui";

// Tree attention from first principles, following the paper's own argument
// (Shyam et al., "Tree Attention", arXiv 2408.04093): attention for one query
// is a ratio of sums → sums can be added in any order → a tree does it in
// log p steps → a global max keeps exp() finite → the paper's view: attention
// is the gradient of a logsumexp "energy" → Algorithm 3 → why it beats ring.

const CHAPTERS = [
  { id: "problem", title: "The decoding problem" },
  { id: "sums", title: "Attention is two sums" },
  { id: "tree", title: "Any order → a tree" },
  { id: "max", title: "Overflow and the max" },
  { id: "energy", title: "The paper's lens: an energy" },
  { id: "algo", title: "The algorithm" },
  { id: "ring", title: "Why not ring attention?" },
];

const Takeaway = ({ html }) => (
  <Rich
    style={{ borderLeft: `3px solid ${T.accent}`, background: `${T.accent}10`, padding: "10px 14px", borderRadius: 6, fontSize: 15, color: T.ink, lineHeight: 1.55, marginTop: 16 }}
    html={html}
  />
);
const P_ = ({ html }) => <Rich style={{ fontSize: 15, color: T.soft, lineHeight: 1.7, margin: "0 0 10px" }} html={html} />;
const th = { textAlign: "right", padding: "5px 9px", fontWeight: 500, color: T.dim, borderBottom: `1px solid ${T.rule}`, whiteSpace: "nowrap" };
const td = { textAlign: "right", padding: "5px 9px", borderBottom: `1px solid ${T.well}`, whiteSpace: "nowrap" };
const gpuOf = (P, a) => Math.floor(a / (NTOK / P));

// a row of key/value tokens coloured by the GPU that holds them
function TokenRow({ P, top }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {K.map((k, a) => {
        const g = gpuOf(P, a);
        return (
          <div key={a} style={{ border: `1.5px solid ${DEV_COLORS[g]}`, background: `${DEV_COLORS[g]}18`, borderRadius: 8, padding: "6px 8px", fontFamily: mono, fontSize: 11.5, minWidth: 92 }}>
            <div style={{ color: DEV_COLORS[g], fontWeight: 700 }}>token {a} · GPU {g}</div>
            <div style={{ color: T.soft }}>k = {fv(k, 1)}</div>
            <div style={{ color: T.soft }}>v = [{V[a].join(", ")}]</div>
            {top && <div style={{ color: T.accent, marginTop: 2 }}>{top(a)}</div>}
          </div>
        );
      })}
    </div>
  );
}

function ChProblem({ P }) {
  const s = scores();
  return (
    <>
      <P_ html={`When a language model <b>decodes</b>, it produces one new token at a time. That token has one query vector $q$, and it must attend to the keys and values of <b>every</b> earlier token: the KV cache. With a million-token context the cache does not fit on one GPU, so it is split by position across $p$ GPUs (here $p = ${P}$, ${NTOK / P} tokens each). The query is tiny, so every GPU gets a copy.`} />
      <Tex tex={String.raw`z \;=\; \sum_{a=1}^{N} \operatorname{softmax}_a(q\cdot k_a)\; v_a \;=\; \sum_{a} \frac{e^{\,q\cdot k_a}}{\sum_b e^{\,q\cdot k_b}}\, v_a`} />
      <div style={{ fontFamily: mono, fontSize: 13, color: T.accent, margin: "8px 0" }}>q = [{Q.join(", ")}] (on every GPU)</div>
      <TokenRow P={P} top={(a) => `q·k = ${f(s[a])}`} />
      <Takeaway html={String.raw`The question for the whole page: every GPU can only see its own keys, yet the softmax needs a denominator over <b>all</b> keys. How do the GPUs agree on the answer without shipping keys around?`} />
    </>
  );
}

function ChSums({ P }) {
  const s = scores();
  const e = s.map((x) => Math.exp(x));
  const sub = Array.from({ length: P }, (_, g) => {
    const idx = shard(P, g);
    return { n: [0, 1].map((c) => idx.reduce((acc, a) => acc + e[a] * V[a][c], 0)), d: idx.reduce((acc, a) => acc + e[a], 0) };
  });
  const n = [0, 1].map((c) => sub.reduce((acc, x) => acc + x.n[c], 0));
  const d = sub.reduce((acc, x) => acc + x.d, 0);
  const z = attention();
  return (
    <>
      <P_ html="Pull the softmax apart. It is a weighted sum divided by the sum of the weights: two plain sums over the tokens (the memory-efficient attention of Rabe &amp; Staats, eq. 1–3 in the paper)." />
      <Tex tex={String.raw`z = \frac{n}{d},\qquad n = \sum_a e^{\,q\cdot k_a}\, v_a,\qquad d = \sum_a e^{\,q\cdot k_a}`} />
      <P_ html="A sum over all tokens is the sum of each GPU's sum over <b>its own</b> tokens. So every GPU can compute its share with no communication, and only the small totals need combining:" />
      <Tex tex={String.raw`n = \sum_{i=1}^{p} n_i,\qquad d = \sum_{i=1}^{p} d_i,\qquad n_i = \sum_{a \in \text{GPU } i} e^{\,q\cdot k_a} v_a,\quad d_i = \sum_{a \in \text{GPU } i} e^{\,q\cdot k_a}`} />
      <div style={{ overflowX: "auto", marginTop: 8 }}>
        <table style={{ borderCollapse: "collapse", fontFamily: mono, fontSize: 12.5, width: "100%" }}>
          <thead><tr><th style={{ ...th, textAlign: "left" }}>GPU</th><th style={th}>its tokens</th><th style={th}>e^(q·k) per token</th><th style={th}>n_i (2 numbers)</th><th style={th}>d_i (1 number)</th></tr></thead>
          <tbody>
            {sub.map((x, g) => (
              <tr key={g}>
                <td style={{ ...td, textAlign: "left", color: DEV_COLORS[g], fontWeight: 700 }}>GPU {g}</td>
                <td style={td}>{shard(P, g).join(", ")}</td>
                <td style={{ ...td, color: T.soft }}>{shard(P, g).map((a) => f(e[a])).join(" + ")}</td>
                <td style={{ ...td, color: T.ink }}>{fv(x.n)}</td>
                <td style={{ ...td, color: T.ink }}>{f(x.d)}</td>
              </tr>
            ))}
            <tr><td style={{ ...td, textAlign: "left", color: T.accent }}>total</td><td style={td} /><td style={td} /><td style={{ ...td, color: T.accent, fontWeight: 700 }}>{fv(n)}</td><td style={{ ...td, color: T.accent, fontWeight: 700 }}>{f(d)}</td></tr>
          </tbody>
        </table>
      </div>
      <div style={{ fontFamily: mono, fontSize: 13, marginTop: 10, color: T.soft }}>
        n / d = {fv(n.map((x) => x / d), 4)}   ·   softmax computed directly = {fv(z, 4)} ✓
      </div>
      <Takeaway html={String.raw`Each GPU only has to send <b>3 numbers</b> (its $n_i$ and $d_i$), whatever its number of tokens. The whole problem is now: add up $p$ small things.`} />
    </>
  );
}

// chain vs tree for adding the p partial denominators
function ChTree({ P }) {
  const vals = useMemo(() => {
    const e = scores().map((x) => Math.exp(x));
    return Array.from({ length: P }, (_, g) => shard(P, g).reduce((acc, a) => acc + e[a], 0));
  }, [P]);
  const W = 640, colW = W / P, levels = Math.log2(P);
  const leafY = 40 + levels * 58;
  const H = leafY + 50;
  const cx = (a, b) => ((a + b) / 2) * colW;
  const nodes = [];
  for (let l = 1; l <= levels; l++) for (let k = 0; k < P / 2 ** l; k++) {
    const lo = k * 2 ** l, hi = lo + 2 ** l;
    nodes.push({ l, k, x: cx(lo, hi), y: leafY - l * 58, v: vals.slice(lo, hi).reduce((a, b) => a + b, 0), lc: cx(lo, lo + 2 ** (l - 1)), rc: cx(lo + 2 ** (l - 1), hi) });
  }
  let run = 0;
  const chain = vals.map((v) => (run += v));
  return (
    <>
      <P_ html="Addition does not care about order or grouping: $(a+b)+(c+d) = ((a+b)+c)+d$. That property, <b>associativity</b>, is all a tree needs. Instead of one GPU adding the partial results one after another, pairs of GPUs add at the same time, then pairs of pairs, and so on." />
      <div className="ra-two" style={{ marginTop: 6 }}>
        <div>
          <div style={h2}>One after another: {P - 1} sequential steps</div>
          <div style={{ display: "grid", gap: 4, fontFamily: mono, fontSize: 12 }}>
            {chain.map((c, g) => (
              <div key={g} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ color: T.dim, width: 52 }}>{g === 0 ? "start" : `step ${g}`}</span>
                <span style={{ color: DEV_COLORS[g] }}>{g === 0 ? `d₀ = ${f(vals[0])}` : `+ d${g} (${f(vals[g])})`}</span>
                <span style={{ color: T.soft, marginLeft: "auto" }}>= {f(c)}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div style={h2}>A tree: {levels} parallel level{levels > 1 ? "s" : ""}</div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
            {nodes.map((nd) => (
              <g key={`${nd.l}-${nd.k}`}>
                {[nd.lc, nd.rc].map((x, j) => (
                  <line key={j} x1={x} y1={nd.l === 1 ? leafY : nd.y + 58} x2={nd.x} y2={nd.y + 22} stroke={T.dim} strokeWidth={1.2} />
                ))}
              </g>
            ))}
            {vals.map((v, g) => (
              <g key={g}>
                <rect x={cx(g, g + 1) - 34} y={leafY} width={68} height={30} rx={6} fill={`${DEV_COLORS[g]}22`} stroke={DEV_COLORS[g]} />
                <text x={cx(g, g + 1)} y={leafY + 19} textAnchor="middle" fontFamily={mono} fontSize="11" fill={DEV_COLORS[g]}>d{g}={f(v)}</text>
              </g>
            ))}
            {nodes.map((nd) => (
              <g key={`n${nd.l}-${nd.k}`}>
                <rect x={nd.x - 38} y={nd.y} width={76} height={24} rx={6} fill={T.well} stroke={nd.l === levels ? T.accent : T.soft} />
                <text x={nd.x} y={nd.y + 16} textAnchor="middle" fontFamily={mono} fontSize="11" fill={nd.l === levels ? T.accent : T.ink}>{f(nd.v)}</text>
              </g>
            ))}
            {Array.from({ length: levels }, (_, l) => (
              <text key={l} x={4} y={leafY - (l + 1) * 58 + 16} fontFamily={mono} fontSize="10" fill={T.dim}>level {l + 1}</text>
            ))}
          </svg>
        </div>
      </div>
      <div style={{ overflowX: "auto", marginTop: 8 }}>
        <table style={{ borderCollapse: "collapse", fontFamily: mono, fontSize: 12.5 }}>
          <thead><tr><th style={{ ...th, textAlign: "left" }}>GPUs p</th><th style={th}>chain: steps</th><th style={th}>tree: levels = log₂ p</th></tr></thead>
          <tbody>{[8, 64, 128, 1024].map((p) => <tr key={p}><td style={{ ...td, textAlign: "left" }}>{p}</td><td style={td}>{p - 1}</td><td style={{ ...td, color: T.accent }}>{Math.log2(p)}</td></tr>)}</tbody>
        </table>
      </div>
      <Takeaway html={String.raw`This is the paper's Theorem 1: an associative reduction over $N$ items on $p$ processors takes $O(N/p + \log p)$ time, local sums first, then a tree. In collective-communication terms it is an <b>Allreduce</b>: reduce up the tree, then broadcast the result back down so every GPU has it.`} />
    </>
  );
}

function ChMax({ P }) {
  const [alpha, setAlpha] = useState(1);
  const s = scores(alpha);
  const m = Math.max(...s);
  const naiveD = s.reduce((acc, x) => acc + fp16exp(x), 0);
  const naiveN = [0, 1].map((c) => s.reduce((acc, x, a) => acc + fp16exp(x) * V[a][c], 0));
  const naiveZ = naiveN.map((x) => x / naiveD);
  const safeZ = attention(alpha);
  const pr = partials(P, alpha);
  return (
    <>
      <P_ html="There is a catch. $e^{x}$ grows fast: in float16 the largest number is 65504, so any score above $\ln 65504 \approx 11.1$ overflows to <code>inf</code>, and $\infty/\infty$ is <code>NaN</code>. Real attention scores get that big. Drag the slider to scale the scores up:" />
      <label style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: mono, fontSize: 13, color: T.soft, margin: "6px 0 12px" }}>
        scores × {alpha.toFixed(1)}
        <input type="range" min={1} max={12} step={0.5} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ flex: 1, maxWidth: 360, accentColor: T.accent }} />
        largest score = {f(m)}
      </label>
      <div className="ra-two">
        <div style={{ ...panel, background: T.well }}>
          <div style={{ ...h2, color: naiveZ.some((x) => !Number.isFinite(x)) ? T.bad : T.dim }}>Naive, in float16</div>
          <div style={{ fontFamily: mono, fontSize: 12.5, color: T.soft, lineHeight: 1.8 }}>
            <div>d = Σ e^s = {f(naiveD)}</div>
            <div>z = n / d = <span style={{ color: naiveZ.some((x) => !Number.isFinite(x)) ? T.bad : T.ink }}>{fv(naiveZ, 3)}</span></div>
          </div>
        </div>
        <div style={{ ...panel, background: T.well }}>
          <div style={{ ...h2, color: T.accent }}>Subtract the max first</div>
          <div style={{ fontFamily: mono, fontSize: 12.5, color: T.soft, lineHeight: 1.8 }}>
            <div>m = max score = {f(m)}; every e^(s − m) ≤ 1</div>
            <div>z = {fv(safeZ, 3)} ✓</div>
          </div>
        </div>
      </div>
      <Tex tex={String.raw`z \;=\; \frac{\sum_a e^{\,s_a - m}\,v_a}{\sum_a e^{\,s_a - m}} \qquad (\text{the } e^{-m} \text{ cancels, so the answer is unchanged})`} />
      <P_ html={`But $m$ must be the max over <b>all</b> tokens, and each GPU only sees its own. Max is associative too ($\\max(\\max(a,b),c) = \\max(a,\\max(b,c))$), so it is one more tree reduction, done <b>before</b> the sums. Here: the GPUs' local maxima of their log-sums are ${pr.gpus.map((g) => f(g.lse)).join(", ")} and the global one is ${f(pr.m)}.`} />
      <Takeaway html={String.raw`So decoding needs two tree reductions: first a <b>max</b> (to agree on a safe shift), then a <b>sum</b> of the shifted $n_i, d_i$.`} />
    </>
  );
}

// F(ζ) along one coordinate of ζ, with its tangent at 0 (slope = attention)
function ChEnergy({ P }) {
  const [k, setK] = useState(0);
  const z = attention();
  const dir = (t) => (k === 0 ? [t, 0] : [0, t]);
  const F = (t) => energy(dir(t));
  const eps = 1e-4;
  const fd = (F(eps) - F(-eps)) / (2 * eps);
  const W = 520, H = 230, pad = 34;
  const ts = Array.from({ length: 61 }, (_, i) => -1.5 + (3 * i) / 60);
  const Fs = ts.map(F);
  const lo = Math.min(...Fs) - 0.2, hi = Math.max(...Fs) + 0.2;
  const X = (t) => pad + ((t + 1.5) / 3) * (W - 2 * pad);
  const Y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  const F0 = F(0);
  const pr = partials(P);
  const lseAll = lse(scores());
  return (
    <>
      <P_ html="The paper finds a single scalar function whose <b>gradient</b> is attention. Add an extra input $\zeta$ (the 'source', a vector the size of $v$) and define the energy:" />
      <Tex tex={String.raw`F(\zeta) \;=\; \log \sum_{a} \exp\!\big(q\cdot k_a + \zeta\cdot v_a\big) \;=\; \operatorname{logsumexp}_a\big(q\cdot k_a + \zeta\cdot v_a\big)`} />
      <Tex tex={String.raw`\frac{\partial F}{\partial \zeta}\Big|_{\zeta=0} \;=\; \frac{\sum_a e^{\,q\cdot k_a}\, v_a}{\sum_a e^{\,q\cdot k_a}} \;=\; \sum_a \operatorname{softmax}_a(q\cdot k_a)\, v_a \;=\; z`} />
      <P_ html="Differentiating $\log\sum\exp$ gives back the softmax weights, and the chain rule through $\zeta\cdot v_a$ multiplies each weight by its $v_a$. Check it numerically: below is $F$ along one coordinate of $\zeta$, and its slope at $0$." />
      <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
        {[0, 1].map((c) => (
          <button key={c} onClick={() => setK(c)} style={{ ...navBtn(false), height: 28, fontSize: 12, color: k === c ? T.bg : T.soft, background: k === c ? T.accent : T.well, borderColor: k === c ? T.accent : T.rule }}>
            ζ along coordinate {c}
          </button>
        ))}
      </div>
      <div className="ra-two">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: T.well, borderRadius: 8 }}>
          <line x1={X(0)} x2={X(0)} y1={pad / 2} y2={H - pad} stroke={T.rule} strokeDasharray="3 4" />
          <polyline points={ts.map((t, i) => `${X(t)},${Y(Fs[i])}`).join(" ")} fill="none" stroke={T.wire} strokeWidth={2.2} />
          <line x1={X(-1.2)} y1={Y(F0 - 1.2 * z[k])} x2={X(1.2)} y2={Y(F0 + 1.2 * z[k])} stroke={T.accent} strokeWidth={1.8} strokeDasharray="6 4" />
          <circle cx={X(0)} cy={Y(F0)} r={4.5} fill={T.accent} />
          <text x={X(0) + 8} y={Y(F0) - 8} fontFamily={mono} fontSize="11" fill={T.accent}>slope = z[{k}] = {f(z[k], 3)}</text>
          <text x={W - pad} y={H - 10} textAnchor="end" fontFamily={mono} fontSize="10.5" fill={T.dim}>ζ[{k}] →</text>
          <text x={pad} y={pad / 2 + 4} fontFamily={mono} fontSize="10.5" fill={T.wire}>F(ζ)</text>
        </svg>
        <div style={{ fontFamily: mono, fontSize: 12.5, color: T.soft, lineHeight: 1.9 }}>
          <div>finite difference (F(+ε) − F(−ε)) / 2ε = <b style={{ color: T.ink }}>{f(fd, 4)}</b></div>
          <div>attention output z[{k}] = <b style={{ color: T.accent }}>{f(z[k], 4)}</b> ✓</div>
          <div style={{ marginTop: 10, color: T.dim }}>F is a logsumexp, and logsumexp is associative:</div>
          <div>logsumexp over all 8 tokens = {f(lseAll, 4)}</div>
          <div>logsumexp of the {P} GPUs' own logsumexps ({pr.gpus.map((g) => f(g.lse)).join(", ")}) = {f(lse(pr.gpus.map((g) => g.lse)), 4)} ✓</div>
        </div>
      </div>
      <Takeaway html={String.raw`This is the paper's key observation (eq. 5–7): attention is the gradient of one big <b>logsumexp</b>, and logsumexp (like max) is associative, so it can be computed with a tree reduction across GPUs. Automatic differentiation then gives its gradient, the attention output, for about the same cost.`} />
    </>
  );
}

const JAX = `def tree_flash_decode(q, k, v):
    loc_res, loc_lse = flash_num_lse(q, k, v)
    a_max_global = lax.pmax(loc_lse, axis_name='i')
    num_global = lax.psum(loc_res * jnp.exp(loc_lse - a_max_global), axis_name='i')
    den_global = lax.psum(jnp.exp(loc_lse - a_max_global), axis_name='i')
    return (num_global / den_global)`.split("\n").map((text) => ({ text, tags: [] }));

function ChAlgo({ P }) {
  const pr = partials(P);
  const exact = attention();
  const [sel, setSel] = useState(0);
  const g = pr.gpus[Math.min(sel, P - 1)];
  const notes = [
    { m: "loc_res, loc_lse = flash_num_lse", t: `GPU ${g.i}: tokens ${g.idx.join(",")} → loc_res = ${fv(g.o, 3)}, loc_lse = ${f(g.lse, 3)}` },
    { m: "a_max_global = lax.pmax", t: `Allreduce(max) over the ${P} GPUs → ${f(pr.m, 3)}` },
    { m: "num_global = lax.psum", t: `this GPU adds ${fv(g.n, 3)} = loc_res · e^(${f(g.lse, 2)} − ${f(pr.m, 2)}); Allreduce(sum) → ${fv(pr.n, 3)}` },
    { m: "den_global = lax.psum", t: `this GPU adds e^(${f(g.lse, 2)} − ${f(pr.m, 2)}) = ${f(g.d, 3)}; Allreduce(sum) → ${f(pr.d, 3)}` },
    { m: "return (num_global / den_global)", t: `${fv(pr.z, 4)} on every GPU (exact: ${fv(exact, 4)})` },
  ];
  return (
    <>
      <P_ html="Putting it together gives the paper's Algorithm 3. Each GPU runs ordinary flash-attention on its own shard, which returns its <b>locally normalised</b> output $o_i$ and its log-sum $\ell_i = \log\sum_{a\in i} e^{q\cdot k_a}$. Then three Allreduces:" />
      <Tex tex={String.raw`m = \max_i \ell_i,\qquad n = \sum_i o_i\, e^{\ell_i - m},\qquad d = \sum_i e^{\ell_i - m},\qquad z = \frac{n}{d}`} />
      <P_ html="Why $o_i e^{\ell_i - m}$? Because $o_i = n_i/e^{\ell_i}$ is already divided by its local sum; multiplying by $e^{\ell_i}$ undoes that, and $e^{-m}$ is the shared safety shift. The paper's JAX code is exactly these lines (<code>pmax</code> / <code>psum</code> are Allreduces):" />
      <div className="ra-two">
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontFamily: mono, fontSize: 12 }}>
            <thead><tr><th style={{ ...th, textAlign: "left" }}>GPU</th><th style={th}>ℓ_i</th><th style={th}>o_i</th><th style={th}>e^(ℓ_i − m)</th><th style={th}>n_i</th></tr></thead>
            <tbody>
              {pr.gpus.map((x) => (
                <tr key={x.i} onClick={() => setSel(x.i)} style={{ cursor: "pointer", background: x.i === g.i ? `${DEV_COLORS[x.i]}18` : "transparent" }}>
                  <td style={{ ...td, textAlign: "left", color: DEV_COLORS[x.i], fontWeight: 700 }}>GPU {x.i}</td>
                  <td style={td}>{f(x.lse, 3)}</td>
                  <td style={td}>{fv(x.o)}</td>
                  <td style={td}>{f(x.w, 3)}</td>
                  <td style={td}>{fv(x.n)}</td>
                </tr>
              ))}
              <tr><td style={{ ...td, textAlign: "left", color: T.accent }}>reduce</td><td style={{ ...td, color: T.accent }}>m = {f(pr.m, 3)}</td><td style={td} /><td style={{ ...td, color: T.accent }}>d = {f(pr.d, 3)}</td><td style={{ ...td, color: T.accent }}>n = {fv(pr.n)}</td></tr>
            </tbody>
          </table>
          <div style={{ fontFamily: mono, fontSize: 12.5, color: T.soft, marginTop: 10 }}>z = n / d = {fv(pr.z, 4)}   ·   exact = {fv(exact, 4)} ✓</div>
          <div style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginTop: 6 }}>click a row to put that GPU's numbers in the code comments</div>
        </div>
        <CodeBlock title="tree_flash_decode (paper, Appendix D)" lines={JAX} focus={{}} notes={notes} maxHeight={360} />
      </div>
      <Takeaway html={String.raw`Nothing about K or V ever leaves its GPU. What travels is one scalar per head for the max, and $d_h + 1$ numbers per head for the sums, along a tree. The Tree attention tab steps through exactly these Allreduces, level by level.`} />
    </>
  );
}

function ChRing() {
  // the paper's example: 640k context on 8 GPUs, hidden 2048 (16 heads × 128), bf16
  const N = 640000, p = 8, d = 2048, nh = 16, b = 1;
  const t = N / p;
  const ring = 2 * b * t * d * p; // eq. 10: elements moved per iteration
  const tree = 2 * ((p - 1) / p) * (b * d + 2 * b * nh); // eq. 14
  const B = (x) => (x * 2 >= 1e9 ? `${(x * 2 / 1e9).toFixed(1)} GB` : x * 2 >= 1e6 ? `${(x * 2 / 1e6).toFixed(1)} MB` : `${(x * 2 / 1e3).toFixed(1)} KB`);
  return (
    <>
      <P_ html="Ring attention also computes exact attention across GPUs, but it was built for training, where every GPU has a long block of queries. Each GPU keeps its queries and passes its <b>keys and values</b> to the next GPU, $p-1$ times in a row; the transfer hides behind the large attention compute." />
      <P_ html="In decoding there is only <b>one</b> query. The compute per GPU is tiny (the paper measures about $10^{-5}$ s), while moving a KV chunk takes about $10^{-3}$ s, so nothing hides the communication. Tree attention instead moves only the small partial results:" />
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontFamily: mono, fontSize: 12.5 }}>
          <thead><tr><th style={{ ...th, textAlign: "left" }} /><th style={th}>what moves</th><th style={th}>sequential steps</th><th style={th}>volume (paper eq. 10, 14)</th></tr></thead>
          <tbody>
            <tr><td style={{ ...td, textAlign: "left", color: T.bad }}>Ring</td><td style={td}>K, V chunks (t × d each)</td><td style={td}>p − 1 = {p - 1}</td><td style={td}>{ring.toExponential(2)} elements ≈ {B(ring)}</td></tr>
            <tr><td style={{ ...td, textAlign: "left", color: T.accent }}>Tree</td><td style={td}>m, n, d (≈ d + 2·heads numbers)</td><td style={td}>~2 log₂ p = {2 * Math.log2(p)} per Allreduce</td><td style={td}>{Math.round(tree)} elements ≈ {B(tree)}</td></tr>
          </tbody>
        </table>
      </div>
      <div style={{ fontFamily: mono, fontSize: 11, color: T.dim, marginTop: 6 }}>example from the paper: {N.toLocaleString()} tokens of context on {p} GPUs, hidden size {d} ({nh} heads × 128), batch 1, bf16</div>
      <P_ html="<br/>It also fits real clusters better: GPUs inside a node talk over fast NVLink, nodes talk over much slower InfiniBand. A ring is only as fast as its slowest link, while an Allreduce lets NCCL use a ring inside the node and a tree across nodes (paper §5.3)." />
      <Takeaway html={String.raw`Result in the paper: up to 8× faster than ring attention when decoding on 128 GPUs with 5.12M tokens of context, about half the peak memory, and 2–5× faster end-to-end decoding on Llama 3.1/3.2.`} />
    </>
  );
}

export default function TreeFirstPrinciples() {
  const [ch, setCh] = useState(0);
  const [P, setP] = useState(4);
  const last = CHAPTERS.length - 1;

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setCh((c) => Math.min(c + 1, last));
      if (e.key === "ArrowLeft") setCh((c) => Math.max(c - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  const C = CHAPTERS[ch];
  const body = { problem: ChProblem, sums: ChSums, tree: ChTree, max: ChMax, energy: ChEnergy, algo: ChAlgo, ring: ChRing }[C.id];
  const Body = body;

  return (
    <div style={{ maxWidth: 1240, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>TREE ATTENTION · FROM FIRST PRINCIPLES</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Why decoding attention can be added up like a tree</h1>
        <Rich style={{ color: T.soft, lineHeight: 1.55, fontSize: 15, maxWidth: 950 }} html="Seven short steps, following the argument of Shyam et al., <i>Tree Attention</i> (arXiv 2408.04093). One toy query, 8 keys, 2-dimensional vectors: every number on the page is computed live." />
      </header>

      <div style={{ ...panel, display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 18, padding: "10px 14px" }}>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: "1 1 600px" }}>
          {CHAPTERS.map((c, i) => (
            <button key={c.id} onClick={() => setCh(i)} style={{ ...navBtn(false), height: 30, fontSize: 11.5, padding: "0 10px",
              color: i === ch ? T.bg : i < ch ? T.soft : T.dim, background: i === ch ? T.accent : T.well, borderColor: i === ch ? T.accent : T.rule }}>
              {i + 1}. {c.title}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>GPUs p</span>
          {[2, 4, 8].map((p) => (
            <button key={p} onClick={() => setP(p)} style={{ ...navBtn(false), height: 28, fontSize: 12, padding: "0 10px", color: p === P ? T.bg : T.soft, background: p === P ? T.wire : T.well, borderColor: p === P ? T.wire : T.rule }}>{p}</button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setCh(ch - 1)} disabled={ch === 0} style={navBtn(ch === 0)}>← prev</button>
          <button onClick={() => setCh(ch + 1)} disabled={ch === last} style={navBtn(ch === last)}>next →</button>
        </div>
      </div>

      <section style={{ ...panel, padding: "20px 24px" }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.dim, marginBottom: 4 }}>step {ch + 1} of {CHAPTERS.length}</div>
        <h2 style={{ margin: "0 0 14px", fontSize: 22, fontWeight: 650 }}>{C.title}</h2>
        <Body P={P} />
      </section>
    </div>
  );
}
