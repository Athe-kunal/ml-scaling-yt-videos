import { useState } from "react";
import { T, mono, btn } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, cvOf, qbStep, simulate } from "./lbModel";
import { Bars, Lines, f2, div } from "./charts";
import Stepper from "./Stepper";
import { Tex } from "./tex";
import { KIMI_GLOSSARY, KIMI_SHAPES } from "./lbShapes";
import { useProgress, ease, lerp, clamp01 } from "./anim";
import K3Panel, { K3_STEPS } from "./KimiK3Panel";

// Kimi K3 Quantile Balancing (arXiv 2607.24653, §2.3.3). Instead of nudging b_j by a
// fixed ±γ, set it so expert j is chosen by exactly k/n of the batch:
//   alpha_i  = (k+1)-th largest of  s_i + b        (cutoff a token's experts must beat)
//   margin   = s_ij - alpha_i
//   b^_j     = - quantile_{1-k/n}( margin_:j )
//   b        = b^ - mean(b^)
// At scale the quantile comes from per-expert histograms summed by one all-reduce.

const { n, k, m, ranks } = LB;
const MU = (m * k) / n;
const Z = Array(n).fill(0);
const B0 = scoresOf(1);
const RT0 = route(B0, Z);
const QB0 = qbStep(B0, Z);
const SIGN = simulate("sign", 60, 0.02);
const SLOW = simulate("sign", 60, 0.003);
const FAST = simulate("sign", 60, 0.08);
const QB = simulate("qb", 60);
const B2 = scoresOf(2);
const SNAPS = [2, 5, 10, 25, 50];
const NB = 16;
const W = 860, H = 430;
const lerpArr = (a, b, e) => a.map((v, i) => lerp(v, b[i], e));
const firstBelow = (rec, thr) => rec.findIndex((r) => r.cv < thr);

const BLOCKS = [
  { id: "limit", label: "why not ± γ?", sub: "fixed step: slow or jittery", code: [
    "b += gamma * sign(load.mean() - load)   # load [N] → mean over N → scalar; b [N] · gamma = fixed step size",
    "# small gamma: slow   large gamma: oscillates"] },
  { id: "scores", label: "affinity scores", sub: "logit → sigmoid → s + b", code: [
    "z = x @ W   # x [B,D] @ W [D,N] → [B,N] · z[t,i] = how strongly token t wants expert i (logit)",
    "s = sigmoid(z)   # [B,N] → [B,N] · each score squashed into (0,1); no softmax across experts",
    "biased = s + b   # [B,N] + [N] → [B,N] · b = 0 on the very first batch"] },
  { id: "cutoff", label: "top-(k+1) cutoff", sub: "α_i per token", code: [
    "vals = topk(biased, k + 1)   # biased [B,N] → top k+1 over N → [B,k+1] · B tokens, N experts, b[i] = bias of expert i",
    "alpha = vals[:, k]   # [B,k+1] → [B] · alpha[t] = token t's (k+1)-th best score = its cutoff"] },
  { id: "margin", label: "routing margin", sub: "s_ij − α_i", code: [
    "margin = s - alpha[:, None]   # [B,N] − [B,1] → [B,N] · margin[t,i] = how far expert i is above token t's cutoff"] },
  { id: "quantile", label: "per-expert quantile", sub: "choose b so k/n of tokens pass", code: [
    "q     = 1 - k / n   # scalar · share of tokens that must stay below the threshold",
    "b_hat = -quantile(margin, q, dim=0)   # [B,N] → quantile over B → [N] · b_hat[i] = bias that gives expert i k/N of the tokens"] },
  { id: "center", label: "mean-centre", sub: "selection unchanged", code: [
    "b = b_hat - b_hat.mean()   # [N] − (mean over N → scalar) → [N] · same shift for every expert, so rankings do not change"] },
  { id: "onegpu", label: "one GPU: exact sort", sub: "what we assumed so far", code: [
    "# so far everything sat on ONE GPU: all B tokens' margins in one place",
    "b_hat = -torch.quantile(margin, q, dim=0)   # [B,N] sort over B → [N] · exact, but needs the whole [B,N] in memory"] },
  { id: "allreduce", label: "histogram + all-reduce", sub: "scales to ~1000 experts", code: [
    "hist = histogram(margin, bins)   # [B,N] → count over B per bin → [N,L] on each rank · L bins, each rank holds B/R tokens",
    "all_reduce(hist)   # sum over R ranks → [N,L], identical on every rank",
    "b_hat = -quantile_from(hist, q)   # [N,L] → find the q point per expert → [N]"] },
  { id: "apply", label: "use next step", sub: "bias applies to the next batch", code: [
    "# batch t is routed with b from batch t-1",
    "idx = topk(s + b, k)   # [B,N] + [N] → top-k over N → [B,k] · idx[t] = token t's k expert ids"] },
  { id: "race", label: "training loop", sub: "sign rule vs quantile", loop: true, code: [
    "for batch in data:   # one batch per step",
    "    route(batch, b)   # [B,N] → idx [B,k]",
    "    b = quantile_balance(batch, b)   # [N]",
    "# bias frozen at inference"] },
];

const STEPS = [
  { id: "limit", block: "limit", lines: [0], scene: "limit", caption: "A fixed ± γ nudge cannot fit every case: too small and balancing crawls, too big and the loads keep overshooting.", vars: () => [["n experts (K3)", "896"], ["k", "16"]] },
  { id: "logit", block: "scores", lines: [0], scene: "scores", stage: 0, pick: "tok", caption: "The router gives every token a raw number for each expert saying how much that token wants it.", vars: () => [["B tokens", `${m}`], ["N experts", `${n}`]] },
  { id: "sigmoid", block: "scores", lines: [1], scene: "scores", stage: 1, pick: "tok", caption: "Squash each raw number into a 0-to-1 score, one expert at a time, so the scores are easy to compare.", vars: () => [["s", "σ(z)"]] },
  { id: "addb", block: "scores", lines: [2], scene: "scores", stage: 2, pick: "tok", caption: "Add each expert's bias to its score: this nudge only decides who gets picked, not how much they count.", vars: () => [["b", "0 for now"]] },
  { id: "topk1", block: "cutoff", lines: [0], scene: "cutoff", stage: 0, pick: "tok", caption: "For each token, rank the experts by biased score and keep the best k+1: the k winners plus the first runner-up.", vars: () => [["k", `${k}`], ["keep", `k+1 = ${k + 1}`]] },
  { id: "cutoff", block: "cutoff", lines: [1], scene: "cutoff", stage: 1, pick: "tok", caption: "The runner-up's score is that token's cutoff α: an expert has to beat it to be one of the k winners.", vars: () => [["k", `${k}`], ["α", "(k+1)-th"]] },
  { id: "margin", block: "margin", lines: [0], scene: "margin", pick: "tok", caption: "Subtract the cutoff from every score, so each number says how far an expert is above (win) or below (lose) for that token.", vars: () => [["margin", "s_ij − α_i"]] },
  { id: "q", block: "quantile", lines: [0], scene: "quantile", stage: 0, pick: "ex", caption: "Each expert should win k/n of the tokens, so we need a line that leaves exactly that fraction of its margins on the right.", vars: () => [["target", `k/n = ${(k / n).toFixed(2)}`], ["quantile", `${(1 - k / n).toFixed(2)}`], ["K3", "1 − 16/896 ≈ 0.982"]] },
  { id: "quantile", block: "quantile", lines: [1], scene: "quantile", stage: 1, pick: "ex", caption: "Find that line, then shift the expert's scores by minus its position, so exactly k/n of the tokens now win it.", vars: () => [["target", `k/n = ${(k / n).toFixed(2)}`], ["quantile", `${(1 - k / n).toFixed(2)}`]] },
  { id: "center", block: "center", lines: [0], scene: "center", pick: "ex", caption: "Subtract the average bias from every expert: no ranking changes, but the biases cannot drift away over time.", vars: () => [["mean b̂", Math.abs(QB0.bhat.reduce((a, x) => a + x, 0) / n) < 1e-9 ? "0" : (QB0.bhat.reduce((a, x) => a + x, 0) / n).toFixed(3)]] },
  { id: "onegpu", block: "onegpu", lines: [1], scene: "onegpu", pick: "ex", caption: "Everything so far assumed one GPU holds all the tokens, so we can simply sort each column and read the quantile exactly.", vars: () => [["tokens on the GPU", `${m}`], ["cost", "sort B per expert"]] },
  { id: "hist", block: "allreduce", lines: [0], scene: "allreduce", stage: 0, pick: "ex", caption: "A real batch is too big to sort, so each GPU just counts its own tokens' margins into bins.", vars: () => [["ranks", `${ranks}`], ["bins / expert", `${NB} here · few hundred in K3`]] },
  { id: "reduce", block: "allreduce", lines: [1], scene: "allreduce", stage: 1, pick: "ex", caption: "Add the GPUs' bin counts together: that is the whole batch's histogram, for the cost of a few thousand numbers.", vars: () => [["ranks", `${ranks}`], ["traffic", "bins × experts"]] },
  { id: "bhat", block: "allreduce", lines: [2], scene: "allreduce", stage: 2, pick: "ex", caption: "Walk the bins until the running count reaches the target load: that spot is the quantile, with no sorting needed.", vars: () => [["same as", "sorting all tokens"]] },
  { id: "apply", block: "apply", lines: [0, 1], visAt: 1, scene: "apply", caption: "Use the new bias on the next batch, so every batch is balanced by what the one before it revealed.", vars: () => [] },
  ...SNAPS.map((t) => ({
    id: `t${t}`, block: "race", lines: [1, 2], visAt: 1, scene: "race", t, caption: `After ${t} updates: quantile balancing is already even, while the sign rule is still creeping toward it.`,
    vars: () => [["CV sign", SIGN[t].cv.toFixed(2)], ["CV quantile", QB[t].cv.toFixed(2)], ...(t === 50 ? [["inference", "bias frozen"]] : [])],
  })),
];

const hist = (vals, lo, hi) => {
  const c = Array(NB).fill(0);
  vals.forEach((v) => (c[Math.min(NB - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * NB)))] += 1));
  return c;
};

// histogram bars with a threshold line; bins right of the line are coloured
function Hist({ x, y, w, h, counts, lo, hi, thr, color, title, maxC, hlBin }) {
  const bw = w / NB;
  const mx = maxC ?? Math.max(...counts, 1);
  const tx = thr == null ? null : x + ((thr - lo) / (hi - lo)) * w;
  const zx = x + ((0 - lo) / (hi - lo)) * w;
  return (
    <g>
      {title && <text x={x} y={y - 8} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{title}</text>}
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      {counts.map((c, i) => {
        const bx = lo + ((i + 0.5) / NB) * (hi - lo);
        return <rect key={i} x={x + i * bw + 1.5} y={y + h - (c / mx) * h} width={bw - 3} height={(c / mx) * h} rx={2} fill={thr == null || bx > thr ? color : T.rule} />;
      })}
      {hlBin != null && <rect x={x + hlBin * bw} y={y} width={bw} height={h} fill="none" stroke={T.ink} strokeWidth={2} rx={2} />}
      <line x1={zx} x2={zx} y1={y} y2={y + h} stroke={T.dim} strokeDasharray="2 3" />
      {tx != null && <line x1={tx} x2={tx} y1={y - 4} y2={y + h} stroke={T.wire} strokeWidth={2} />}
    </g>
  );
}

// rk = replay counter: bumping it restarts the animation on the same step
function Scene({ step, tok, ex, setEx, rk = 0 }) {
  const p = useProgress(`${step.id}:${rk}`, 1500); // 0 → 1 each time a new picture arrives; ↻ replay re-runs it
  const e = ease(p);
  let body = null;
  const d2 = (v) => v.toFixed(2);
  const c = (x, j) => `{\\color{${XC[j]}}\\mathbf{${x}}}`; // a number to point at, coloured like its expert
  const col = (j) => B0.map((_, i) => QB0.margins[i][j]);

  if (step.scene === "limit") {
    const r = (rec) => rec.map((q) => q.loads[0] / MU);
    body = (
      <g>
        <Lines x={30} y={70} w={380} h={200} series={[{ pts: r(SLOW), color: T.accent, label: "γ small: slow" }]} upTo={Math.floor(e * 60)} ymax={4} ref1={1} title="EXPERT 0 LOAD / FAIR SHARE" />
        <Lines x={450} y={70} w={380} h={200} series={[{ pts: r(FAST), color: T.bad, label: "γ large: oscillates" }]} upTo={Math.floor(e * 60)} ymax={4} ref1={1} title="EXPERT 0 LOAD / FAIR SHARE" />
        <Tex x={30} y={296} w={500} h={30} size={15} color={T.accent2} tex={`b_j\\leftarrow b_j+\\gamma\\,\\mathrm{sign}(\\bar\\ell-\\ell_j)`} />
        <Tex x={330} y={298} w={520} h={26} size={11.5} color={T.dim} tex={`\\bar\\ell=\\text{mean load},\\;\\ell_j=\\text{load of expert }j,\\;\\gamma=\\text{fixed step}`} />
        <text x={30} y={344} fontFamily={mono} fontSize="11" fill={T.dim}>one step size for every expert and every imbalance; K3 has 896 experts and k = 16</text>
        <text x={30} y={364} fontFamily={mono} fontSize="11" fill={T.dim}>yellow dashed = fair share · x axis: training step 0–60</text>
      </g>
    );
  } else if (step.scene === "scores") {
    const sc = B0[tok];
    const z = sc.map((v) => Math.log(v / (1 - v))); // logit recovered from s = sigmoid(z)
    const zmx = Math.max(...z, 1) * 1.12, zmn = Math.min(...z, 0);
    const st = step.stage;
    const sAnim = sc.map((v) => lerp(0, v, e)); // s grows from 0 as the sigmoid is applied
    const sbAnim = sc.map((v) => lerp(0, v, e));
    const all = (f) => sc.map((_, j) => f(j)).join(",\\;\\;");
    body = (
      <g>
        {st === 0 && <Bars x={40} y={80} w={520} h={230} values={z.map((v) => lerp(0, v, e))} max={zmx} min={zmn} fmt={f2} title={`TOKEN x${tok} · LOGIT z = x · w_j`} labels={XC.map((_, j) => `E${j}`)} />}
        {st === 1 && <Bars x={30} y={80} w={380} h={230} values={z} max={zmx} min={zmn} fmt={f2} title="LOGIT z" labels={XC.map((_, j) => `E${j}`)} />}
        {st === 1 && <Bars x={450} y={80} w={380} h={230} values={sAnim} max={1} fmt={f2} title="SCORE s = σ(z)" labels={XC.map((_, j) => `E${j}`)} />}
        {st === 2 && <Bars x={30} y={80} w={380} h={230} values={sc} max={1} fmt={f2} title="SCORE s" labels={XC.map((_, j) => `E${j}`)} />}
        {st === 2 && <Bars x={450} y={80} w={380} h={230} values={sbAnim} max={1} fmt={f2} title="s + b  (b = 0)" labels={XC.map((_, j) => `E${j}`)} />}
        <Tex x={40} y={340} w={800} h={32} size={14} color={T.ink} tex={st === 0
          ? all((j) => `\\mathrm{E}_{${j}}\\!:\\,z=${c(d2(z[j]), j)}`)
          : st === 1
            ? all((j) => `\\mathrm{E}_{${j}}\\!:\\,${d2(z[j])}\\to\\sigma=${c(d2(sc[j]), j)}`)
            : all((j) => `\\mathrm{E}_{${j}}\\!:\\,${d2(sc[j])}{+}0\\to ${c(d2(sc[j]), j)}`)} />
        <text x={40} y={392} fontFamily={mono} fontSize="11.5" fill={T.dim}>
          {st === 0 ? `token x${tok}: one logit per expert, any real number · expert 0 is the popular one` : st === 1 ? "each expert scored on its own: scores need not sum to 1" : "b is selection-only: it steers the top-k, the output still uses s"}
        </text>
      </g>
    );
  } else if (step.scene === "cutoff") {
    const r = RT0[tok];
    const cut = order(B0[tok]);
    const sv = B0[tok]; // s + b already shown in the previous steps
    const aNow = lerp(1, r.cutoff, e); // the cutoff line slides down to the (k+1)-th score
    body = (
      <g>
        <Bars
          x={40} y={80} w={520} h={230} values={sv} max={1} ideal={step.stage >= 1 ? aNow : undefined} idealLabel="α" fmt={f2}
          title={`TOKEN x${tok} · SCORE s + b  (b = 0 for now)`} labels={XC.map((_, j) => `E${j}`)} dim={B0[tok].map((_, j) => j).filter((j) => !r.sel.includes(j) && j !== cut[k])} flag={step.stage >= 1 ? [cut[k]] : []}
        />
        <Tex x={40} y={340} w={800} h={32} size={14} color={T.ink} tex={step.stage === 0
          ? `${cut.slice(0, k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${c(d2(B0[tok][j]), j)}`).join("\\;\\ge\\;")}\\;\\Big|\\;${cut.slice(k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${d2(B0[tok][j])}`).join("\\;\\ge\\;")}`
          : `${cut.slice(0, k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${c(d2(B0[tok][j]), j)}`).join("\\;\\to\\;")}\\quad\\Rightarrow\\quad \\alpha_{x_{${tok}}}=${c(d2(r.cutoff), cut[k])}`} />
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          <text x={600} y={120}>top-{k}: routed</text>
          {step.stage >= 1 && <text x={600} y={142} fill={T.bad}>(k+1)-th, red ring:</text>}
          {step.stage >= 1 && <text x={600} y={160} fill={T.bad}>defines the cutoff α_i</text>}
          <text x={600} y={200} fill={T.dim}>an expert wins this token</text>
          <text x={600} y={218} fill={T.dim}>iff it stays above α_i</text>
        </g>
      </g>
    );
  } else if (step.scene === "margin") {
    const x0 = 40, cw = 90, ch = 34, y0 = 56;
    const rows = LB.T;
    const alpha = RT0[tok].cutoff;
    const shift = (i, j) => QB0.margins[i][j] + RT0[i].cutoff * (1 - e); // s_ij → s_ij − α_i
    const tokVals = QB0.margins[tok].map((_, j) => shift(tok, j));
    body = (
      <g>
        {Array.from({ length: n }, (_, j) => <text key={j} x={x0 + j * cw + cw / 2 - 2} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>)}
        {Array.from({ length: rows }, (_, i) => (
          <g key={i} opacity={i === tok ? 1 : 0.7}>
            <text x={x0 - 6} y={y0 + i * ch + ch / 2 + 3} textAnchor="end" fontFamily={mono} fontSize="11" fill={i === tok ? T.ink : T.dim}>x{i}</text>
            {QB0.margins[i].map((v, j) => {
              const val = shift(i, j); // colour animates from s_ij to s_ij − α_i; the text shows both ends at once
              const fin = v; // final margin
              const fm = Math.abs(fin) < 1e-9 ? "0" : Math.abs(fin) < 0.01 ? fin.toFixed(3) : (fin > 0 ? "+" : "") + fin.toFixed(2);
              return (
                <g key={j}>
                  <rect x={x0 + j * cw} y={y0 + i * ch} width={cw - 3} height={ch - 3} rx={3} fill={div(val)} stroke={i === tok ? T.ink : "none"} />
                  <text x={x0 + j * cw + (cw - 3) / 2} y={y0 + i * ch + ch / 2 + 3} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight={i === tok ? 700 : 400} fill={T.ink}>{B0[i][j].toFixed(2)}<tspan fill={T.dim}> → </tspan>{fm}</text>
                </g>
              );
            })}
          </g>
        ))}
        <text x={x0} y={y0 + rows * ch + 20} fontFamily={mono} fontSize="10" fill={T.dim}>each cell: score s → margin s − α</text>
        <text x={x0} y={y0 + rows * ch + 36} fontFamily={mono} fontSize="10" fill={T.dim}>warm = above the cutoff · cool = below</text>
        <Bars x={470} y={90} w={370} h={200} values={tokVals} max={Math.max(...B0[tok]) * 1.12} min={Math.min(...QB0.margins[tok])} ideal={alpha * (1 - e)} idealLabel="α" fmt={f2} title={`TOKEN x${tok} · MARGIN s − α`} labels={XC.map((_, j) => `E${j}`)} />
        <Tex x={470} y={314} w={390} h={66} size={12.5} color={T.ink} tex={`\\begin{gathered}${order(B0[tok]).slice(0, k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${d2(B0[tok][j])}-${d2(alpha)}=${c(d2(B0[tok][j] - alpha), j)}`).join("\\\\")}\\end{gathered}`} />
        <text x={470} y={400} fontFamily={mono} fontSize="11" fill={T.dim}>routed experts &gt; 0 · the (k+1)-th is exactly 0</text>
      </g>
    );
  } else if (step.scene === "onegpu") {
    const v = col(ex).slice().sort((a, b) => a - b); // all m margins of one expert, sorted: [m]
    const lo = v[0], hi = v[v.length - 1], thr = QB0.thr[ex];
    const bx = (i) => 62 + i * 11.9, by = (x) => 290 - ((x - lo) / (hi - lo)) * 170;
    body = (
      <g>
        <rect x={40} y={56} width={780} height={252} rx={10} fill={T.well} stroke={T.wire} strokeWidth={1.5} />
        <text x={56} y={78} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.wire}>GPU 0 · all {m} tokens</text>
        <text x={W - 56} y={78} textAnchor="end" fontFamily={mono} fontSize="10.5" fill={T.dim}>E{ex}'s margins, sorted small → large</text>
        {v.map((x, i) => <rect key={i} x={bx(i)} y={290 - (290 - by(x)) * e} width={9.5} height={(290 - by(x)) * e + 1} rx={1.5} fill={x > thr ? XC[ex] : T.rule} />)}
        <line x1={bx(m / 2) - 1.2} x2={bx(m / 2) - 1.2} y1={96} y2={296} stroke={T.wire} strokeWidth={2} />
        <text x={bx(m / 2) - 8} y={104} textAnchor="end" fontFamily={mono} fontSize="10.5" fill={T.dim}>{m / 2} below</text>
        <text x={bx(m / 2) + 6} y={104} fontFamily={mono} fontSize="10.5" fill={XC[ex]} fontWeight="700">{m / 2} above</text>
        <Tex x={40} y={322} w={780} h={34} size={14} color={T.ink} tex={`\\text{sort the }${m}\\text{ margins},\\ \\mathrm{quantile}_{${d2(1 - k / n)}}=${c(d2(thr), ex)}\\ \\Rightarrow\\ \\hat b_{\\mathrm{E}_{${ex}}}=${c(d2(-thr), ex)}`} />
        <text x={40} y={392} fontFamily={mono} fontSize="11.5" fill={T.dim}>exact, but every margin has to be on this one device</text>
      </g>
    );
  } else if (step.scene === "quantile") {
    const v = col(ex);
    const lo = Math.min(...v), hi = Math.max(...v);
    const cnt = hist(v, lo, hi);
    const thr = step.stage >= 1 ? lerp(hi, QB0.thr[ex], e) : null; // threshold slides in from the right
    const bhA = QB0.bhat.map((x) => lerp(0, x, e));
    const above = thr == null ? 0 : v.filter((x) => x > thr).length;
    body = (
      <g>
        <Hist x={40} y={80} w={520} h={220} counts={step.stage === 0 ? cnt.map((x) => x * e) : cnt} maxC={Math.max(...cnt, 1)} lo={lo} hi={hi} thr={thr} color={XC[ex]} title={`EXPERT E${ex} · MARGINS OF ALL ${m} TOKENS`} />
        {thr == null ? (
          <g>
            <Tex x={40} y={312} w={500} h={42} size={15} color={T.ink} tex={`q=1-\\dfrac{k}{N}=1-\\dfrac{${k}}{${n}}=${c(d2(1 - k / n), ex)}`} />
            <Tex x={40} y={362} w={700} h={26} size={11.5} color={T.dim} tex={`\\text{the threshold must leave }k/N=${d2(k / n)}\\text{ of the ${m} tokens on its right}`} />
          </g>
        ) : (
          <g>
            <Tex x={40} y={308} w={520} h={30} size={14} color={T.ink} tex={`\\mathrm{quantile}_{q=${d2(1 - k / n)}}(\\mathrm{margin}_{:,\\mathrm{E}_{${ex}}})=${c(d2(thr), ex)}`} />
            <Tex x={40} y={342} w={520} h={44} size={14} color={T.ink} tex={`\\dfrac{${c(above, ex)}}{${m}}=${d2(above / m)}\\approx\\dfrac{k}{N}=\\dfrac{${k}}{${n}}\\;\\Rightarrow\\;\\text{load}\\approx${MU}`} />
            <Tex x={40} y={390} w={520} h={30} size={15} color={T.accent2} tex={`\\hat b_{\\mathrm{E}_{${ex}}}=-\\mathrm{quantile}=-(${d2(thr)})=${c(d2(-thr), ex)}`} />
          </g>
        )}
        {thr != null && <Bars x={610} y={90} w={230} h={170} values={bhA} max={Math.max(...QB0.bhat) * 1.12} min={Math.min(...QB0.bhat)} fmt={f2} title="b̂ FOR ALL EXPERTS" dim={XC.map((_, j) => j).filter((j) => j !== ex)} />}
      </g>
    );
  } else if (step.scene === "center") {
    const mu = QB0.bhat.reduce((a, x) => a + x, 0) / n;
    const mx = Math.max(...QB0.bhat, ...QB0.b) * 1.12, mn = Math.min(...QB0.bhat, ...QB0.b);
    body = (
      <g>
        <Bars x={30} y={80} w={380} h={210} values={QB0.bhat} max={mx} min={mn} ideal={mu} idealLabel="mean" fmt={f2} title="b̂ (RAW)" hl={ex} onPick={setEx} />
        <Bars x={450} y={80} w={380} h={210} values={lerpArr(QB0.bhat, QB0.b, e)} max={mx} min={mn} ideal={mu * (1 - e)} idealLabel="mean" fmt={f2} title="b = b̂ − mean(b̂)" hl={ex} onPick={setEx} />
        <Tex x={30} y={318} w={800} h={44} size={14} color={T.ink} tex={`b_{\\mathrm{E}_{${ex}}}=\\hat b_{\\mathrm{E}_{${ex}}}-\\dfrac{1}{N}\\sum_{i}\\hat b_i=${c(d2(QB0.bhat[ex]), ex)}-(${d2(mu)})=${c(d2(lerp(QB0.bhat[ex], QB0.b[ex], e)), ex)}`} />
        <text x={30} y={384} fontFamily={mono} fontSize="12" fill={T.soft}>adding the same number to every expert never changes a top-k ranking</text>
      </g>
    );
  } else if (step.scene === "allreduce") {
    const v = col(ex);
    const lo = Math.min(...v), hi = Math.max(...v);
    const per = m / ranks;
    const locals = Array.from({ length: ranks }, (_, r) => hist(v.slice(r * per, (r + 1) * per), lo, hi));
    const pooled = locals[0].map((_, i) => locals.reduce((a, h) => a + h[i], 0));
    const thr0 = QB0.thr[ex];
    const thr = step.stage >= 2 ? lerp(hi, thr0, e) : thr0; // stage 2: threshold slides in
    const mx = Math.max(...locals.flat(), 1);
    const part = (r) => (step.stage === 0 ? e : 1); // stage 0: each rank fills its own bins
    const acc = (r) => (step.stage === 1 ? clamp01(e * ranks - r) : 1); // stage 1: ranks add in one by one
    const pooledA = pooled.map((_, i) => locals.reduce((a, h, r) => a + h[i] * acc(r), 0));
    const pooledMax = Math.max(...pooled, 1);
    const bi = Math.min(NB - 1, Math.max(0, Math.floor(((thr0 - lo) / (hi - lo)) * NB))); // the bin holding the threshold
    body = (
      <g>
        {locals.map((c, r) => (
          <Hist key={r} x={20 + r * 210} y={70} w={190} h={80} counts={step.stage === 0 ? c.map((x) => x * part(r)) : c} lo={lo} hi={hi} thr={step.stage >= 2 ? thr : null} color={XC[ex]} maxC={mx} hlBin={bi} title={`RANK ${r} · ${per} TOKENS`} />
        ))}
        {step.stage >= 1 && <text x={W / 2} y={188} textAnchor="middle" fontFamily={mono} fontSize="13" fontWeight="700" fill={T.wire}>↓ all-reduce: add the bin counts ↓</text>}
        {step.stage >= 1 && <Hist x={230} y={210} w={400} h={110} counts={pooledA} maxC={pooledMax} lo={lo} hi={hi} thr={step.stage >= 2 ? thr : null} color={XC[ex]} hlBin={bi} title={`POOLED · EXPERT E${ex}`} />}
        {step.stage === 0 && <Tex x={20} y={176} w={820} h={30} size={14} color={T.ink} tex={`\\mathrm{hist}_r[\\ell=${bi}]=${locals.map((h) => c(Math.round(h[bi] * part()), ex)).join(",\\;")}\\quad(r=0,\\dots,${ranks - 1})`} />}
        {step.stage === 1 && <Tex x={20} y={328} w={820} h={32} size={14} color={T.ink} tex={`\\mathrm{hist}[\\ell=${bi}]=\\sum_{r}\\mathrm{hist}_r[\\ell]=${locals.map((h) => c(h[bi], ex)).join("+")}=${c(Math.round(pooledA[bi]), ex)}`} />}
        {step.stage >= 2 && <Tex x={20} y={330} w={820} h={30} size={14} color={T.accent2} tex={`\\hat b_{\\mathrm{E}_{${ex}}}=-\\mathrm{quantile}_{q}(\\mathrm{hist})=-(${d2(thr)})=${c(d2(-thr), ex)}`} />}
        {step.stage >= 2 && <text x={20} y={376} fontFamily={mono} fontSize="11.5" fill={T.soft}>same quantile as sorting all {m} margins, up to bin width</text>}
        {step.stage >= 1 && <text x={20} y={402} fontFamily={mono} fontSize="11" fill={T.dim}>traffic = bins × experts, whatever the number of tokens or ranks</text>}
      </g>
    );
  } else if (step.scene === "apply") {
    const sets = [
      ["NO BIAS", Z],
      ["SIGN STEP γ=0.02 · AFTER 1 UPDATE", SIGN[0].next],
      ["QUANTILE · AFTER 1 UPDATE", QB[0].next],
    ];
    body = (
      <g>
        {sets.map(([t, b], i) => {
          const l1 = loadsOf(route(B2, b));
          const l = i === 0 ? l1 : lerpArr(loadsOf(route(B2, Z)), l1, e); // bias switched on: s → s + b
          return (
            <g key={i}>
              <Bars x={20 + i * 285} y={80} w={255} h={190} values={l} ideal={MU} title={t} max={42} />
              <text x={20 + i * 285} y={316} fontFamily={mono} fontSize="12" fontWeight="700" fill={i === 2 ? T.accent : T.soft}>load CV {cvOf(l).toFixed(2)}</text>
            </g>
          );
        })}
        <Tex x={20} y={328} w={820} h={50} size={14} color={T.ink} tex={`\\mathrm{CV}=\\dfrac{\\mathrm{std}(\\ell)}{\\bar\\ell},\\quad \\mathrm{std}(\\ell)=\\sqrt{\\tfrac{1}{${n}}\\textstyle\\sum_{j}(\\ell_j-\\bar\\ell)^2},\\quad \\bar\\ell=${MU}\\text{ (fair share)}`} />
        <text x={20} y={402} fontFamily={mono} fontSize="11" fill={T.dim}>a fresh batch, routed with the bias computed from the previous one</text>
      </g>
    );
  } else {
    const t = step.t;
    const prevT = SNAPS[SNAPS.indexOf(t) - 1] ?? 0;
    const tn = Math.round(lerp(prevT, t, e)); // training advances from the last snapshot
    const tl = (R) => lerpArr(R[prevT].loads, R[t].loads, e);
    body = (
      <g>
        <Lines x={20} y={70} w={390} h={230} series={[{ pts: SIGN.map((q) => q.cv), color: T.wire, label: "sign step γ=0.02" }, { pts: QB.map((q) => q.cv), color: T.accent, label: "quantile" }]} ymax={1} upTo={tn} title="LOAD CV (LOWER = BALANCED)" />
        <Bars x={450} y={80} w={390} h={80} values={tl(SIGN)} ideal={MU} title="SIGN STEP · LOAD NOW" max={40} />
        <Bars x={450} y={220} w={390} h={80} values={tl(QB)} ideal={MU} title="QUANTILE · LOAD NOW" max={40} />
        <Tex x={20} y={376} w={820} h={30} size={13.5} color={T.dim} tex={`\\mathrm{CV}=\\mathrm{std}(\\ell)\\,/\\,\\bar\\ell\\ \\text{ over the }${n}\\text{ experts' loads }\\ell;\\ 0=\\text{even}`} />
        <text x={20} y={340} fontFamily={mono} fontSize="12" fill={T.soft}>CV below 0.3 after: sign step {firstBelow(SIGN, 0.3)} steps · quantile {firstBelow(QB, 0.3)} step</text>
        <text x={20} y={362} fontFamily={mono} fontSize="11" fill={T.dim}>the remaining ~0.2 is sampling noise of a {m}-token batch</text>
      </g>
    );
  }
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

const order = (row) => row.map((_, j) => j).sort((a, b) => row[b] - row[a]);

export { Scene as KimiScene, STEPS as KIMI_STEPS, BLOCKS as KIMI_BLOCKS };

export default function KimiView() {
  const [rk, setRk] = useState(0); // replay counter
  const [tok, setTok] = useState(0);
  const [ex, setEx] = useState(0);
  const picker = (label, val, set, count) => (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>{label}</span>
      {Array.from({ length: count }, (_, i) => (
        <button key={i} onClick={() => set(i)} style={{ ...btn(i === val, false), height: 28, padding: "0 9px" }}>{i}</button>
      ))}
    </div>
  );
  return (
    <Stepper
      title="Kimi K3 · Quantile Balancing"
      subtitle={`bias from a score quantile instead of a ± γ nudge · toy: n=${n} experts, k=${k}, ${m} tokens`}
      blocks={BLOCKS}
      steps={STEPS}
      header={(step) => (
        <>
          {step.pick === "tok" ? picker("follow token", tok, setTok, LB.T) : step.pick === "ex" ? picker("follow expert", ex, setEx, n) : null}
          <button onClick={() => setRk((r) => r + 1)} style={{ ...btn(false, false), height: 28, padding: "0 10px" }} title="run this step's animation again">↻ replay</button>
        </>
      )}
      scene={(step) => (
        <>
          <Scene step={step} tok={tok} ex={ex} setEx={setEx} rk={rk} />
          {K3_STEPS.has(step.id) && <K3Panel step={step} rk={rk} />}
        </>
      )}
      glossary={KIMI_GLOSSARY}
      shapes={(step) => KIMI_SHAPES[step.block === "race" ? "race" : step.sid]}
      legend={<span>source: Kimi K3 technical report · toy numbers, not K3's (896 experts, k = 16)</span>}
    />
  );
}
