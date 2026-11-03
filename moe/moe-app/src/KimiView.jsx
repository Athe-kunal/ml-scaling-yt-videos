import { useState } from "react";
import { T, mono, btn } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, cvOf, qbStep, simulate } from "./lbModel";
import { Bars, Lines, f2, div } from "./charts";
import Stepper from "./Stepper";

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
const firstBelow = (rec, thr) => rec.findIndex((r) => r.cv < thr);

const BLOCKS = [
  { id: "limit", label: "why not ± γ?", sub: "fixed step: slow or jittery", code: ["b += gamma * sign(load.mean() - load)", "# small gamma: slow   large gamma: oscillates"] },
  { id: "cutoff", label: "top-(k+1) cutoff", sub: "α_i per token", code: ["vals = topk(s + b, k + 1)", "alpha = vals[:, k]            # (k+1)-th score"] },
  { id: "margin", label: "routing margin", sub: "s_ij − α_i", code: ["margin = s - alpha[:, None]   # [tokens, experts]"] },
  { id: "quantile", label: "per-expert quantile", sub: "choose b so k/n of tokens pass", code: ["q     = 1 - k / n", "b_hat = -quantile(margin, q, dim=0)"] },
  { id: "center", label: "mean-centre", sub: "selection unchanged", code: ["b = b_hat - b_hat.mean()"] },
  { id: "allreduce", label: "histogram + all-reduce", sub: "scales to ~1000 experts", code: ["hist = histogram(margin, bins)    # [experts, bins]", "all_reduce(hist)                   # one collective", "b_hat = -quantile_from(hist, q)"] },
  { id: "apply", label: "use next step", sub: "bias applies to the next batch", code: ["# batch t is routed with b from batch t-1", "idx = topk(s + b, k)"] },
  { id: "race", label: "training loop", sub: "sign rule vs quantile", loop: true, code: ["for batch in data:", "    route(batch, b); b = quantile_balance(batch, b)", "# bias frozen at inference"] },
];

const STEPS = [
  { id: "limit", block: "limit", scene: "limit", caption: "fixed γ: slow when small, jittery when large", vars: () => [["n experts (K3)", "896"], ["k", "16"]] },
  { id: "cutoff", block: "cutoff", scene: "cutoff", pick: "tok", caption: "each token has a cutoff: its (k+1)-th best score", vars: () => [["k", `${k}`], ["α", "(k+1)-th"]] },
  { id: "margin", block: "margin", scene: "margin", pick: "tok", caption: "margin = how far each expert is above that cutoff", vars: () => [["margin", "s_ij − α_i"]] },
  { id: "quantile", block: "quantile", scene: "quantile", pick: "ex", caption: "slide b until exactly k/n of tokens pass the cutoff", vars: () => [["target", `k/n = ${(k / n).toFixed(2)}`], ["quantile", `${(1 - k / n).toFixed(2)}`], ["K3", "1 − 16/896 ≈ 0.982"]] },
  { id: "center", block: "center", scene: "center", caption: "subtract the mean: routing unchanged, bias stays small", vars: () => [["mean b̂", QB0.bhat.reduce((a, x) => a + x, 0) / n < 1e-9 ? "0" : (QB0.bhat.reduce((a, x) => a + x, 0) / n).toFixed(3)]] },
  { id: "allreduce", block: "allreduce", scene: "allreduce", pick: "ex", caption: "ranks add bin counts: same quantile, tiny traffic", vars: () => [["ranks", `${ranks}`], ["bins / expert", `${NB} here · few hundred in K3`]] },
  { id: "apply", block: "apply", scene: "apply", caption: "one update later the next batch is already balanced", vars: () => [] },
  ...SNAPS.map((t) => ({
    id: `t${t}`, block: "race", scene: "race", t, caption: `step ${t}: quantile vs sign-step balance`,
    vars: () => [["CV sign", SIGN[t].cv.toFixed(2)], ["CV quantile", QB[t].cv.toFixed(2)], ...(t === 50 ? [["inference", "bias frozen"]] : [])],
  })),
];

const hist = (vals, lo, hi) => {
  const c = Array(NB).fill(0);
  vals.forEach((v) => (c[Math.min(NB - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * NB)))] += 1));
  return c;
};

// histogram bars with a threshold line; bins right of the line are coloured
function Hist({ x, y, w, h, counts, lo, hi, thr, color, title, maxC }) {
  const bw = w / NB;
  const mx = maxC ?? Math.max(...counts, 1);
  const tx = x + ((thr - lo) / (hi - lo)) * w;
  const zx = x + ((0 - lo) / (hi - lo)) * w;
  return (
    <g>
      {title && <text x={x} y={y - 8} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{title}</text>}
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      {counts.map((c, i) => {
        const bx = lo + ((i + 0.5) / NB) * (hi - lo);
        return <rect key={i} x={x + i * bw + 1.5} y={y + h - (c / mx) * h} width={bw - 3} height={(c / mx) * h} rx={2} fill={bx > thr ? color : T.rule} />;
      })}
      <line x1={zx} x2={zx} y1={y} y2={y + h} stroke={T.dim} strokeDasharray="2 3" />
      <line x1={tx} x2={tx} y1={y - 4} y2={y + h} stroke={T.wire} strokeWidth={2} />
    </g>
  );
}

function Scene({ step, tok, ex }) {
  let body = null;
  const col = (j) => B0.map((_, i) => QB0.margins[i][j]);

  if (step.scene === "limit") {
    const r = (rec) => rec.map((q) => q.loads[0] / MU);
    body = (
      <g>
        <Lines x={30} y={70} w={380} h={200} series={[{ pts: r(SLOW), color: T.accent, label: "γ small: slow" }]} ymax={4} ref1={1} title="EXPERT 0 LOAD / FAIR SHARE" />
        <Lines x={450} y={70} w={380} h={200} series={[{ pts: r(FAST), color: T.bad, label: "γ large: oscillates" }]} ymax={4} ref1={1} title="EXPERT 0 LOAD / FAIR SHARE" />
        <text x={30} y={320} fontFamily={mono} fontSize="13" fontWeight="700" fill={T.accent2}>b_j += γ · sign(mean − load_j)</text>
        <text x={30} y={344} fontFamily={mono} fontSize="11" fill={T.dim}>one step size for every expert and every imbalance; K3 has 896 experts and k = 16</text>
        <text x={30} y={364} fontFamily={mono} fontSize="11" fill={T.dim}>yellow dashed = fair share · x axis: training step 0–60</text>
      </g>
    );
  } else if (step.scene === "cutoff") {
    const r = RT0[tok];
    const cut = order(B0[tok]);
    body = (
      <g>
        <Bars
          x={40} y={80} w={520} h={230} values={B0[tok]} max={1} ideal={r.cutoff} idealLabel="α" fmt={f2}
          title={`TOKEN x${tok} · SCORE s + b  (b = 0 for now)`} dim={B0[tok].map((_, j) => j).filter((j) => !r.sel.includes(j) && j !== cut[k])} flag={[cut[k]]}
        />
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          <text x={600} y={120}>top-{k}: routed</text>
          <text x={600} y={142} fill={T.bad}>(k+1)-th, red ring:</text>
          <text x={600} y={160} fill={T.bad}>defines the cutoff α_i</text>
          <text x={600} y={200} fill={T.dim}>an expert wins this token</text>
          <text x={600} y={218} fill={T.dim}>iff it stays above α_i</text>
        </g>
      </g>
    );
  } else if (step.scene === "margin") {
    const x0 = 40, cw = 46, ch = 19, y0 = 56;
    const rows = 16;
    body = (
      <g>
        {Array.from({ length: n }, (_, j) => <text key={j} x={x0 + j * cw + cw / 2 - 2} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>)}
        {Array.from({ length: rows }, (_, i) => (
          <g key={i} opacity={i === tok ? 1 : 0.7}>
            <text x={x0 - 6} y={y0 + i * ch + 13} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={i === tok ? T.ink : T.dim}>x{i}</text>
            {QB0.margins[i].map((v, j) => (
              <rect key={j} x={x0 + j * cw} y={y0 + i * ch} width={cw - 3} height={ch - 3} rx={3} fill={div(v)} stroke={i === tok ? T.ink : "none"} />
            ))}
          </g>
        ))}
        <text x={x0} y={y0 + rows * ch + 20} fontFamily={mono} fontSize="10" fill={T.dim}>warm = above the cutoff · cool = below</text>
        <Bars x={470} y={90} w={370} h={200} values={QB0.margins[tok]} fmt={f2} title={`TOKEN x${tok} · MARGIN s − α`} labels={XC.map((_, j) => `E${j}`)} />
        <text x={470} y={350} fontFamily={mono} fontSize="11" fill={T.dim}>routed experts &gt; 0 · the (k+1)-th is exactly 0</text>
      </g>
    );
  } else if (step.scene === "quantile") {
    const v = col(ex);
    const lo = Math.min(...v), hi = Math.max(...v);
    const thr = QB0.thr[ex];
    const above = v.filter((x) => x > thr).length;
    body = (
      <g>
        <Hist x={40} y={80} w={520} h={220} counts={hist(v, lo, hi)} lo={lo} hi={hi} thr={thr} color={XC[ex]} title={`EXPERT E${ex} · MARGINS OF ALL ${m} TOKENS`} />
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          <text x={40} y={330}>yellow line = the (1 − k/n) quantile = {f2(thr)}</text>
          <text x={40} y={352} fill={XC[ex]}>{above} of {m} tokens lie right of it = k/n of the batch = target load {MU}</text>
          <text x={40} y={376} fontWeight="700" fill={T.accent2}>b̂_{ex} = −(quantile) = {f2(-thr)}</text>
        </g>
        <Bars x={610} y={90} w={230} h={170} values={QB0.bhat} fmt={f2} title="b̂ FOR ALL EXPERTS" dim={XC.map((_, j) => j).filter((j) => j !== ex)} />
      </g>
    );
  } else if (step.scene === "center") {
    const mu = QB0.bhat.reduce((a, x) => a + x, 0) / n;
    body = (
      <g>
        <Bars x={30} y={80} w={380} h={210} values={QB0.bhat} fmt={f2} title="b̂ (RAW)" />
        <Bars x={450} y={80} w={380} h={210} values={QB0.b} fmt={f2} title="b = b̂ − mean(b̂)" />
        <text x={30} y={340} fontFamily={mono} fontSize="12" fill={T.soft}>adding the same number to every expert never changes a top-k ranking</text>
        <text x={30} y={362} fontFamily={mono} fontSize="11" fill={T.dim}>mean removed: {f2(mu)} · old bias only enters through the cutoffs α</text>
      </g>
    );
  } else if (step.scene === "allreduce") {
    const v = col(ex);
    const lo = Math.min(...v), hi = Math.max(...v);
    const per = m / ranks;
    const locals = Array.from({ length: ranks }, (_, r) => hist(v.slice(r * per, (r + 1) * per), lo, hi));
    const pooled = locals[0].map((_, i) => locals.reduce((a, h) => a + h[i], 0));
    const thr = QB0.thr[ex];
    const mx = Math.max(...locals.flat(), 1);
    body = (
      <g>
        {locals.map((c, r) => (
          <Hist key={r} x={20 + r * 210} y={70} w={190} h={80} counts={c} lo={lo} hi={hi} thr={thr} color={XC[ex]} maxC={mx} title={`RANK ${r} · ${per} TOKENS`} />
        ))}
        <text x={W / 2} y={188} textAnchor="middle" fontFamily={mono} fontSize="13" fontWeight="700" fill={T.wire}>↓ all-reduce: add the bin counts ↓</text>
        <Hist x={230} y={210} w={400} h={110} counts={pooled} lo={lo} hi={hi} thr={thr} color={XC[ex]} title={`POOLED · EXPERT E${ex}`} />
        <text x={20} y={360} fontFamily={mono} fontSize="11.5" fill={T.soft}>same quantile as sorting all {m} margins, up to bin width</text>
        <text x={20} y={382} fontFamily={mono} fontSize="11" fill={T.dim}>traffic = bins × experts, whatever the number of tokens or ranks</text>
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
          const l = loadsOf(route(B2, b));
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
    body = (
      <g>
        <Lines x={20} y={70} w={390} h={230} series={[{ pts: SIGN.map((q) => q.cv), color: T.wire, label: "sign step γ=0.01" }, { pts: QB.map((q) => q.cv), color: T.accent, label: "quantile" }]} ymax={1} upTo={t} title="LOAD CV (LOWER = BALANCED)" />
        <Bars x={450} y={80} w={390} h={80} values={SIGN[t].loads} ideal={MU} title="SIGN STEP · LOAD NOW" max={40} />
        <Bars x={450} y={220} w={390} h={80} values={QB[t].loads} ideal={MU} title="QUANTILE · LOAD NOW" max={40} />
        <text x={20} y={340} fontFamily={mono} fontSize="12" fill={T.soft}>CV below 0.3 after: sign step {firstBelow(SIGN, 0.3)} steps · quantile {firstBelow(QB, 0.3)} step</text>
        <text x={20} y={362} fontFamily={mono} fontSize="11" fill={T.dim}>the remaining ~0.2 is sampling noise of a {m}-token batch</text>
      </g>
    );
  }
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

const order = (row) => row.map((_, j) => j).sort((a, b) => row[b] - row[a]);

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
      scene={(step) => <Scene step={step} tok={tok} ex={ex} />}
      legend={<span>source: Kimi K3 technical report, §2.3.3 · toy numbers, not K3's (896 experts, k = 16)</span>}
    />
  );
}
