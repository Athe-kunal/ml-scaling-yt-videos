import { useState } from "react";
import { T, mono, btn } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, cvOf, qbStep, simulate } from "./lbModel";
import { Bars, Lines, f2, div } from "./charts";
import Stepper from "./Stepper";
import { Tex } from "./tex";
import { KIMI_GLOSSARY, KIMI_SHAPES } from "./lbShapes";
import { useProgress, ease, lerp, clamp01 } from "./anim";

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
const SIGN = simulate("sign", 60, 0.01);
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
  { id: "cutoff", label: "top-(k+1) cutoff", sub: "α_i per token", code: [
    "vals = topk(s + b, k + 1)   # s [B,N] + b [N] → [B,N] → top k+1 over N → [B,k+1] · B tokens, N experts, b[i] = bias of expert i",
    "alpha = vals[:, k]   # [B,k+1] → [B] · alpha[t] = token t's (k+1)-th best score = its cutoff"] },
  { id: "margin", label: "routing margin", sub: "s_ij − α_i", code: [
    "margin = s - alpha[:, None]   # [B,N] − [B,1] → [B,N] · margin[t,i] = how far expert i is above token t's cutoff"] },
  { id: "quantile", label: "per-expert quantile", sub: "choose b so k/n of tokens pass", code: [
    "q     = 1 - k / n   # scalar · share of tokens that must stay below the threshold",
    "b_hat = -quantile(margin, q, dim=0)   # [B,N] → quantile over B → [N] · b_hat[i] = bias that gives expert i k/N of the tokens"] },
  { id: "center", label: "mean-centre", sub: "selection unchanged", code: [
    "b = b_hat - b_hat.mean()   # [N] − (mean over N → scalar) → [N] · same shift for every expert, so rankings do not change"] },
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
  { id: "limit", block: "limit", lines: [0], scene: "limit", caption: "fixed γ: slow when small, jittery when large", vars: () => [["n experts (K3)", "896"], ["k", "16"]] },
  { id: "topk1", block: "cutoff", lines: [0], scene: "cutoff", stage: 0, pick: "tok", caption: "take each token's k+1 best experts", vars: () => [["k", `${k}`], ["keep", `k+1 = ${k + 1}`]] },
  { id: "cutoff", block: "cutoff", lines: [1], scene: "cutoff", stage: 1, pick: "tok", caption: "its (k+1)-th best score is the cutoff α", vars: () => [["k", `${k}`], ["α", "(k+1)-th"]] },
  { id: "margin", block: "margin", lines: [0], scene: "margin", pick: "tok", caption: "margin = how far each expert is above that cutoff", vars: () => [["margin", "s_ij − α_i"]] },
  { id: "q", block: "quantile", lines: [0], scene: "quantile", stage: 0, pick: "ex", caption: "we want k/n of the tokens to clear the cutoff", vars: () => [["target", `k/n = ${(k / n).toFixed(2)}`], ["quantile", `${(1 - k / n).toFixed(2)}`], ["K3", "1 − 16/896 ≈ 0.982"]] },
  { id: "quantile", block: "quantile", lines: [1], scene: "quantile", stage: 1, pick: "ex", caption: "slide b until exactly k/n of tokens pass the cutoff", vars: () => [["target", `k/n = ${(k / n).toFixed(2)}`], ["quantile", `${(1 - k / n).toFixed(2)}`]] },
  { id: "center", block: "center", lines: [0], scene: "center", pick: "ex", caption: "subtract the mean: routing unchanged, bias stays small", vars: () => [["mean b̂", Math.abs(QB0.bhat.reduce((a, x) => a + x, 0) / n) < 1e-9 ? "0" : (QB0.bhat.reduce((a, x) => a + x, 0) / n).toFixed(3)]] },
  { id: "hist", block: "allreduce", lines: [0], scene: "allreduce", stage: 0, pick: "ex", caption: "each rank bins its own margins", vars: () => [["ranks", `${ranks}`], ["bins / expert", `${NB} here · few hundred in K3`]] },
  { id: "reduce", block: "allreduce", lines: [1], scene: "allreduce", stage: 1, pick: "ex", caption: "all-reduce: ranks add bin counts into one histogram", vars: () => [["ranks", `${ranks}`], ["traffic", "bins × experts"]] },
  { id: "bhat", block: "allreduce", lines: [2], scene: "allreduce", stage: 2, pick: "ex", caption: "read the quantile off the pooled histogram", vars: () => [["same as", "sorting all tokens"]] },
  { id: "apply", block: "apply", lines: [0, 1], visAt: 1, scene: "apply", caption: "one update later the next batch is already balanced", vars: () => [] },
  ...SNAPS.map((t) => ({
    id: `t${t}`, block: "race", lines: [1, 2], visAt: 1, scene: "race", t, caption: `step ${t}: quantile vs sign-step balance`,
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

function Scene({ step, tok, ex, setEx }) {
  const p = useProgress(step.id); // 0 → 1 each time a new picture arrives
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
  } else if (step.scene === "cutoff") {
    const r = RT0[tok];
    const cut = order(B0[tok]);
    const sv = step.stage === 0 ? B0[tok].map((v) => lerp(0, v, e)) : B0[tok]; // scores rise from 0 (s + b, b = 0)
    const aNow = lerp(1, r.cutoff, e); // the cutoff line slides down to the (k+1)-th score
    const cn = (v) => (step.stage === 0 ? lerp(0, v, e) : v);
    body = (
      <g>
        <Bars
          x={40} y={80} w={520} h={230} values={sv} max={1} ideal={step.stage >= 1 ? aNow : undefined} idealLabel="α" fmt={f2}
          title={`TOKEN x${tok} · SCORE s + b  (b = 0 for now)`} dim={B0[tok].map((_, j) => j).filter((j) => !r.sel.includes(j) && j !== cut[k])} flag={step.stage >= 1 ? [cut[k]] : []}
        />
        <Tex x={40} y={340} w={800} h={32} size={14} color={T.ink} tex={step.stage === 0
          ? `\\mathrm{TopK}(s_{x_{${tok}}}+b,\\,k{+}1)=\\{${cut.slice(0, k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${c(d2(cn(B0[tok][j])), j)}`).join(",\\;")}\\}`
          : `\\alpha_{x_{${tok}}}=(k{+}1)\\text{-th largest}=\\mathrm{E}_{${cut[k]}}\\!:\\,${c(d2(aNow), cut[k])}`} />
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
    const x0 = 40, cw = 46, ch = 19, y0 = 56;
    const rows = 16;
    const alpha = RT0[tok].cutoff;
    const shift = (i, j) => QB0.margins[i][j] + RT0[i].cutoff * (1 - e); // s_ij → s_ij − α_i
    const tokVals = QB0.margins[tok].map((_, j) => shift(tok, j));
    body = (
      <g>
        {Array.from({ length: n }, (_, j) => <text key={j} x={x0 + j * cw + cw / 2 - 2} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>)}
        {Array.from({ length: rows }, (_, i) => (
          <g key={i} opacity={i === tok ? 1 : 0.7}>
            <text x={x0 - 6} y={y0 + i * ch + 13} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={i === tok ? T.ink : T.dim}>x{i}</text>
            {QB0.margins[i].map((v, j) => (
              <rect key={j} x={x0 + j * cw} y={y0 + i * ch} width={cw - 3} height={ch - 3} rx={3} fill={div(shift(i, j))} stroke={i === tok ? T.ink : "none"} />
            ))}
          </g>
        ))}
        <text x={x0} y={y0 + rows * ch + 20} fontFamily={mono} fontSize="10" fill={T.dim}>warm = above the cutoff · cool = below</text>
        <Bars x={470} y={90} w={370} h={200} values={tokVals} max={Math.max(...B0[tok]) * 1.12} min={Math.min(...QB0.margins[tok])} ideal={alpha * (1 - e)} idealLabel="α" fmt={f2} title={`TOKEN x${tok} · MARGIN s − α`} labels={XC.map((_, j) => `E${j}`)} />
        <Tex x={470} y={318} w={390} h={30} size={12.5} color={T.ink} tex={order(B0[tok]).slice(0, k + 1).map((j) => `\\mathrm{E}_{${j}}\\!:\\,${d2(B0[tok][j])}-${d2(alpha * e)}=${c(d2(B0[tok][j] - alpha * e), j)}`).join("\\;\\;")} />
        <text x={470} y={366} fontFamily={mono} fontSize="11" fill={T.dim}>routed experts &gt; 0 · the (k+1)-th is exactly 0</text>
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
      ["SIGN STEP γ=0.01 · AFTER 1 UPDATE", SIGN[0].next],
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
        <text x={20} y={360} fontFamily={mono} fontSize="11" fill={T.dim}>a fresh batch, routed with the bias computed from the previous one</text>
      </g>
    );
  } else {
    const t = step.t;
    const prevT = SNAPS[SNAPS.indexOf(t) - 1] ?? 0;
    const tn = Math.round(lerp(prevT, t, e)); // training advances from the last snapshot
    const tl = (R) => lerpArr(R[prevT].loads, R[t].loads, e);
    body = (
      <g>
        <Lines x={20} y={70} w={390} h={230} series={[{ pts: SIGN.map((q) => q.cv), color: T.wire, label: "sign step γ=0.01" }, { pts: QB.map((q) => q.cv), color: T.accent, label: "quantile" }]} ymax={1} upTo={tn} title="LOAD CV (LOWER = BALANCED)" />
        <Bars x={450} y={80} w={390} h={80} values={tl(SIGN)} ideal={MU} title="SIGN STEP · LOAD NOW" max={40} />
        <Bars x={450} y={220} w={390} h={80} values={tl(QB)} ideal={MU} title="QUANTILE · LOAD NOW" max={40} />
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
      header={(step) => (step.pick === "tok" ? picker("follow token", tok, setTok, 8) : step.pick === "ex" ? picker("follow expert", ex, setEx, n) : null)}
      scene={(step) => <Scene step={step} tok={tok} ex={ex} setEx={setEx} />}
      glossary={KIMI_GLOSSARY}
      shapes={(step) => KIMI_SHAPES[step.block === "race" ? "race" : step.sid]}
      legend={<span>source: Kimi K3 technical report · toy numbers, not K3's (896 experts, k = 16)</span>}
    />
  );
}
