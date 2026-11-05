import { useState } from "react";
import { T, mono, btn } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, auxTerms, simulate } from "./lbModel";
import { Bars, Lines, f2 } from "./charts";
import Stepper from "./Stepper";
import { DS_GLOSSARY, DS_SHAPES, DEV_GLOSSARY, DEV_SHAPES } from "./lbShapes";
import { DEV, devOf, devScores, routeDev } from "./devRoute";
import { Tex } from "./tex";

// DeepSeek-V3 (arXiv 2412.19437, §2.1.2): sigmoid router, auxiliary balance loss
// (eq. 17-20, sequence-wise), then auxiliary-loss-free balancing:
// a per-expert bias b_i steers top-K selection only; the gate still uses s.

const { n, k, T: TT } = LB;
const GAMMA = 0.01; // paper: 0.001, larger here so the motion is visible
const SEQ = scoresOf(1000, TT);
const AUX = auxTerms(SEQ);
const SIM = simulate("sign", 60, GAMMA);
const ZERO = Array(n).fill(0);
const BIAS_DEMO = SIM[25].b;
const SNAPS = [1, 3, 6, 12, 25, 50];
const W = 860, H = 430;
const DSC = devScores(DEV.SEED);
const RR = DSC.map((r) => routeDev(r, true));
const RU = DSC.map((r) => routeDev(r, false));
const srcDev = (t) => Math.floor(t / (DEV.B / DEV.D));
const crossOf = (R) => R.reduce((a, r, t) => a + r.devs.filter((d) => d !== srcDev(t)).length, 0);

const BLOCKS = [
  { id: "route", label: "route with sigmoid + top-K", sub: "sigmoid scores → top-K → gates", code: [
    "s   = sigmoid(u @ e.T)   # u [T,d] @ e.T [d,N] → [T,N] · s[t,i] = token t's score for expert i",
    "idx = topk(s, K)   # top-K over N: [T,N] → [T,K] · idx[t] = token t's K expert ids",
    "g   = s[idx] / s[idx].sum(-1, keepdim=True)   # [T,K] / (sum over K → [T,1]) → [T,K] · g[t,k] = gate of token t's k-th expert"] },
  { id: "aux", label: "auxiliary balance loss", sub: "f_i · P_i", code: [
    "f = N/(K*T) * onehot(idx).sum((0, 1))   # [T,K,N] → sum over T,K → [N] · f[i] = expert i's share of routes (1 = fair)",
    "P = (s / s.sum(-1, keepdim=True)).mean(0)   # [T,N] / (sum over N → [T,1]) → [T,N] → mean over T → [N] · P[i] = expert i's avg soft score",
    "L_bal = alpha * (f * P).sum()   # [N]·[N] → [N] → sum over N → scalar · alpha = loss weight"] },
  { id: "bias", label: "bias steers selection", sub: "loss-free", code: [
    "idx = topk(s + b, K)   # s [T,N] + b [N] → [T,N] → top-K over N → [T,K] · b[i] = expert i's bias, used only to pick",
    "g   = s[idx] / s[idx].sum(-1, keepdim=True)   # [T,K] · uses the raw s, not s + b"] },
  { id: "monitor", label: "monitor expert load", sub: "whole batch, every step", code: [
    "load = onehot(idx).sum((0, 1))   # [T,K,N] → sum over T,K → [N] · load[i] = tokens expert i received"] },
  { id: "update", label: "bias update", sub: "± γ", code: [
    "err = load.mean() - load   # mean over N → scalar; load [N] → err [N] · err[i] > 0: expert i is underloaded",
    "b  += gamma * sign(err)   # [N] · gamma = step size; b[i] goes up if underloaded, down if overloaded"] },
  { id: "loop", label: "repeat each training step", sub: "balance emerges", loop: true, code: [
    "for step in training:   # one batch per step",
    "    idx  = topk(s + b, K)   # [T,K]",
    "    load = count(idx)   # [N] tokens per expert",
    "    b   += gamma * sign(load.mean() - load)   # [N]"] },
  { id: "devroute", label: "device-limited routing", sub: "≤ M devices per token", code: [
    "dev_score = topk(s.view(D, E // D), K // M).values.sum(-1)   # s [E] → [D, E/D] → top K/M per device → sum over them → [D] · dev_score[d] = score of device d",
    "keep = topk(dev_score, M).indices   # top-M over D: [D] → [M] · keep = device ids this token may use",
    "s_dev = s.masked_fill(~on(keep), -inf)   # [E] · experts outside keep get -inf",
    "idx = topk(s_dev, K).indices   # top-K over E: [E] → [K] · all on ≤ M devices"] },
  { id: "shuffle", label: "router shuffle: all-to-all", sub: "dispatch → experts → combine", code: [
    "dest = device_of(idx)   # [B, K] expert ids → device ids",
    "recv = all_to_all(x, dest)   # a copy of each token goes to every device it needs",
    "out = experts(recv)   # runs locally, no network",
    "y = all_to_all(out) · g   # copies return and are weighted by g"] },
  { id: "recipe", label: "V3 recipe", sub: "bias + tiny α loss" , code: ["loss = lm_loss + alpha * L_bal   # alpha extremely small", "# bias does the batch-level balancing"] },
];

const STEPS = [
  { id: "scores", block: "route", lines: [0], scene: "heat", stage: 0, caption: "sigmoid affinity of every token for every expert", vars: () => [["T tokens", `${TT}`], ["N experts", `${n}`], ["K", `${k}`]] },
  { id: "topk", block: "route", lines: [1], scene: "heat", stage: 1, caption: "keep the top-K scores of each token", vars: () => [["K", `${k}`]] },
  { id: "gate", block: "route", lines: [2], scene: "heat", stage: 1.5, caption: "normalise the kept scores into gates", vars: () => [["gate", "s / Σ s_sel"]] },
  { id: "skew", block: "route", lines: [], scene: "heat", stage: 2, caption: "popular experts hog the tokens: routing collapse", vars: () => [["max load", `${Math.max(...AUX.cnt)}`, "bad"], ["fair share", `${(TT * k) / n}`]] },
  { id: "f", block: "aux", lines: [0], scene: "aux", stage: 0, caption: "f_i: how much of its fair share each expert got", vars: () => [["f", `[${AUX.f.map((x) => f2(x)).join(" ")}]`]] },
  { id: "P", block: "aux", lines: [1], scene: "aux", stage: 1, caption: "P_i: how much the router likes each expert (smooth)", vars: () => [["Σ P", "1"]] },
  { id: "L", block: "aux", lines: [2], scene: "aux", stage: 2, caption: "L_bal = α·Σ f_i·P_i: gradient pushes busy experts down", vars: () => [["Σ f·P", AUX.loss.toFixed(2), "bad"], ["balanced", "1.00"]] },
  { id: "select", block: "bias", lines: [0, 1], visAt: 0, scene: "bias", caption: "add b to the score to choose experts; gate keeps raw s", vars: () => [["select", "s + b"], ["weight", "s"]] },
  { id: "monitor", block: "monitor", lines: [0], scene: "monitor", caption: "count tokens per expert over the whole batch", vars: () => [["batch", `${LB.m} tokens`], ["mean load", `${(LB.m * k) / n}`]] },
  { id: "err", block: "update", lines: [0], scene: "update", stage: 0, caption: "error: how far each expert is from the mean load", vars: () => [["γ", `${GAMMA}`]] },
  { id: "update", block: "update", lines: [1], scene: "update", stage: 1, caption: "overloaded: b − γ · underloaded: b + γ", vars: () => [["γ", `${GAMMA}`], ["paper γ", "0.001"]] },
  ...SNAPS.map((t) => ({ id: `t${t}`, block: "loop", lines: [1, 2, 3], visAt: 2, scene: "loop", t, caption: `after ${t} training step${t > 1 ? "s" : ""}: load flattens`, vars: () => [["step", `${t}`], ["load CV", SIM[t].cv.toFixed(2)]] })),
  { id: "dscore", block: "devroute", lines: [0], scene: "devroute", stage: 0, caption: "device score = sum of the top K/M affinities of its experts", vars: () => [["K", `${DEV.K}`], ["M", `${DEV.M}`], ["K/M", `${DEV.K / DEV.M}`]] },
  { id: "dtop", block: "devroute", lines: [1], scene: "devroute", stage: 1, caption: "keep only the M best devices", vars: () => [["keep", RR[0].keep.map((d) => `D${d}`).join(" ")]] },
  { id: "dmask", block: "devroute", lines: [2], scene: "devroute", stage: 2, caption: "experts on the other devices are masked out", vars: () => [["eligible", `${RR[0].eligible.length} of ${DEV.E}`]] },
  { id: "dpick", block: "devroute", lines: [3], scene: "devroute", stage: 3, caption: "top-K among the rest: at most M devices are touched", vars: () => [["devices", `${RR[0].devs.length}`], ["unrestricted", `${RU[0].devs.length}`, RU[0].devs.length > RR[0].devs.length ? "bad" : null]] },
  { id: "dest", block: "shuffle", lines: [0], scene: "shuffle", stage: 0, caption: "every token knows which devices host its experts", vars: () => [["tokens", `${DEV.B}`], ["devices", `${DEV.D}`]] },
  { id: "a2a", block: "shuffle", lines: [1], scene: "shuffle", stage: 1, caption: "all-to-all: token copies fly to their devices (IB, then NVLink)", vars: () => [["network copies", `${crossOf(RR)}`], ["without limit", `${crossOf(RU)}`, "bad"]] },
  { id: "dffn", block: "shuffle", lines: [2], scene: "shuffle", stage: 2, caption: "each expert runs on the copies it received", vars: () => [["copies", `${RR.reduce((a, r) => a + r.devs.length, 0)}`]] },
  { id: "dcomb", block: "shuffle", lines: [3], scene: "shuffle", stage: 3, caption: "outputs travel back and are weighted by the gates g", vars: () => [["dropped tokens", "0"]] },
  { id: "recipe", block: "recipe", lines: [0, 1], visAt: 0, scene: "recipe", caption: "bias balances batches, tiny loss guards each sequence", vars: () => [["tokens dropped", "0"]] },
];

function Heat({ scores, rt, base, stage, bias }) {
  const x0 = 50, cw = 50, ch = 19, y0 = 56;
  return (
    <g>
      {Array.from({ length: n }, (_, j) => (
        <text key={j} x={x0 + j * cw + cw / 2 - 2} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>
      ))}
      {bias && Array.from({ length: n }, (_, j) => {
        const v = bias[j], hh = Math.min(12, Math.abs(v) * 110);
        const base = y0 - 30;
        return <rect key={j} x={x0 + j * cw + 6} y={v >= 0 ? base - hh : base} width={cw - 14} height={Math.max(hh, 1)} rx={2} fill={v >= 0 ? T.accent : T.bad} opacity={0.85} />;
      })}
      {scores.map((row, t) => {
        const sel = rt[t].sel;
        const gsum = sel.reduce((a, j) => a + row[j], 0);
        return (
          <g key={t}>
            <text x={x0 - 8} y={y0 + t * ch + 13} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.dim}>x{t}</text>
            {row.map((s, j) => {
              const on = sel.includes(j);
              const was = base ? base[t].sel.includes(j) : on;
              const flip = base && on !== was;
              const show = stage === 0 || on || bias;
              return (
                <g key={j} opacity={stage >= 1 && !on ? 0.28 : 1} style={{ transition: "opacity .4s" }}>
                  <rect
                    x={x0 + j * cw} y={y0 + t * ch} width={cw - 3} height={ch - 3} rx={3}
                    fill={`color-mix(in srgb, ${XC[j]} ${Math.round(s * 80)}%, ${T.well})`}
                    stroke={flip ? (on ? T.accent2 : T.wire) : on && stage >= 1 ? T.ink : "none"} strokeWidth={flip ? 2.2 : 1.4} strokeDasharray={flip && !on ? "3 2" : "none"}
                  />
                  {show && (
                    <text x={x0 + j * cw + (cw - 3) / 2} y={y0 + t * ch + 12} textAnchor="middle" fontFamily={mono} fontSize={stage >= 1.5 && on && !bias ? 8.5 : 9.5} fontWeight="600" fill={T.ink}>
                      {stage >= 1.5 && on && !bias ? `${f2(s)}→${f2(s / gsum)}` : f2(s)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

// Token x0 worked through: which experts top-K keeps, and how their gates come out.
function Worked({ rt, stage }) {
  const row = SEQ[0], sel = rt[0].sel;
  const v = (j) => f2(row[j]);
  const sum = sel.reduce((a, j) => a + row[j], 0);
  const names = sel.map((j) => `\\mathrm{E}_{${j}}`).join(",\\;");
  // numbers carry the colour of their expert, matching the ringed cells in the grid
  const cc = (j) => `{\\color{${XC[j]}}\\mathbf{${v(j)}}}`;
  const vals = sel.map((j) => `s_{\\mathrm{E}_{${j}}}=${cc(j)}`).join(",\\;");
  const gates = sel.map((j) => `g_{\\mathrm{E}_{${j}}}=\\dfrac{${cc(j)}}{${sel.map(cc).join("+")}}=${f2(row[j] / sum)}`);
  return (
    <g>
      <rect x={46} y={53} width={n * 50 + 2} height={22} rx={5} fill="none" stroke={T.accent} strokeWidth={1.6} />
      <text x={470} y={156} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>TOKEN x0 (OUTLINED ROW)</text>
      {stage === 1 && (
        <g>
          <Tex x={470} y={166} w={380} h={30} size={14} color={T.ink} tex={`\\mathrm{idx}=\\operatorname{TopK}(s,\\,K=${k})=\\{${names}\\}`} />
          <Tex x={470} y={200} w={380} h={30} size={13} color={T.soft} tex={`${vals}`} />
          <Tex x={470} y={234} w={380} h={30} size={12} color={T.dim} tex={`\\text{the other ${n - k} scores are dropped}`} />
        </g>
      )}
      {stage === 1.5 && (
        <g>
          <Tex x={470} y={166} w={380} h={52} size={15} color={T.ink} tex={`g_i=\\dfrac{s_i}{\\sum_{j\\in\\mathrm{idx}} s_j}\\qquad i\\in\\mathrm{idx}`} />
          <Tex x={470} y={222} w={380} h={28} size={13} color={T.wire} tex={`${vals}\\;\\text{(left number in each ringed cell)}`} />
          {gates.map((g, i) => <Tex key={i} x={470} y={256 + i * 50} w={380} h={46} size={14} color={T.soft} tex={g} />)}
          <Tex x={470} y={256 + gates.length * 50} w={380} h={30} size={13} color={T.accent} tex={`${sel.map((j) => `g_{\\mathrm{E}_{${j}}}`).join("+")}=1`} />
        </g>
      )}
    </g>
  );
}

function Scene({ step, ex, setEx }) {
  const rt0 = route(SEQ, ZERO);
  const mu = (LB.m * k) / n;
  let body = null;

  if (step.scene === "heat") {
    body = (
      <g>
        <Heat scores={SEQ} rt={rt0} stage={step.stage} />
        {step.stage === 2 && (
          <g>
            <Bars x={500} y={90} w={330} h={220} values={AUX.cnt} ideal={(TT * k) / n} title="TOKENS PER EXPERT" flag={AUX.cnt.map((c, j) => (c > 2 * ((TT * k) / n) ? j : -1)).filter((j) => j >= 0)} />
            <text x={500} y={355} fontFamily={mono} fontSize="11" fill={T.dim}>E0 and E1 take {AUX.cnt[0] + AUX.cnt[1]} of {TT * k} routes</text>
          </g>
        )}
        {step.stage >= 1 && step.stage < 2 && <Worked rt={rt0} stage={step.stage} />}
        {step.stage >= 1 && step.stage < 2 && <text x={500} y={120} fontFamily={mono} fontSize="11.5" fill={T.soft}>ringed = chosen{step.stage >= 1.5 ? " · in a cell: score s → gate g" : ""}</text>}
      </g>
    );
  } else if (step.scene === "aux") {
    const st = step.stage;
    const col = XC[ex];
    const c = (x) => `{\\color{${col}}\\mathbf{${x}}}`; // the numbers to point at: same colour as the highlighted bars
    const d2 = (v) => v.toFixed(2);
    const rowSum = (t) => SEQ[t].reduce((a, v) => a + v, 0);
    const term = (t) => `\\dfrac{${c(d2(SEQ[t][ex]))}}{${d2(rowSum(t))}}`;
    const E = `\\mathrm{E}_{${ex}}`;
    body = (
      <g>
        <Bars x={20} y={70} w={250} h={180} values={AUX.f} ideal={1} fmt={f2} title="f_i  (FRACTION ROUTED)" hl={ex} onPick={setEx} flag={AUX.f.map((v, j) => (v > 1.5 ? j : -1)).filter((j) => j >= 0)} />
        {st >= 1 && <Bars x={305} y={70} w={250} h={180} values={AUX.P} ideal={1 / n} fmt={f2} title="P_i  (MEAN SOFT SCORE)" hl={ex} onPick={setEx} arrows={st >= 2 ? AUX.f.map((v) => (v > 1 ? -1 : 0)) : null} />}
        {st >= 2 && <Bars x={590} y={70} w={250} h={180} values={AUX.prod} ideal={1 / n} fmt={f2} title="f_i · P_i" hl={ex} onPick={setEx} />}
        <Tex x={20} y={284} w={820} h={42} size={14} color={T.ink} tex={`f_{${E}}=\\dfrac{N}{K\\,T}\\,n_{${E}}=\\dfrac{${n}}{${k}\\cdot ${TT}}\\cdot ${c(AUX.cnt[ex])}=${c(d2(AUX.f[ex]))}`} />
        {st >= 1 && <Tex x={20} y={330} w={840} h={42} size={14} color={T.ink} tex={`P_{${E}}=\\dfrac{1}{T}\\sum_{t=1}^{T}\\dfrac{s_{${E},t}}{\\sum_{j} s_{j,t}}=\\dfrac{1}{${TT}}\\Big(${term(0)}+${term(1)}+\\cdots+${term(TT - 1)}\\Big)=\\dfrac{${c(d2(AUX.P[ex] * TT))}}{${TT}}=${c(d2(AUX.P[ex]))}`} />}
        {st >= 2 && <Tex x={20} y={376} w={840} h={28} size={14} color={T.accent2} tex={`f_{${E}}P_{${E}}=${d2(AUX.f[ex])}\\cdot ${d2(AUX.P[ex])}=${c(d2(AUX.prod[ex]))}\\qquad L_{\\mathrm{bal}}=\\alpha\\sum_{i} f_i P_i=\\alpha\\,(\\cdots+${c(d2(AUX.prod[ex]))}+\\cdots)=\\alpha\\cdot ${d2(AUX.loss)}`} />}
        <Tex x={20} y={406} w={840} h={22} size={11.5} color={T.dim} tex={st === 0 ? `n_{${E}}=\\text{tokens that picked }${E}\\text{ (of }${TT}\\text{); }f=1\\text{ is a fair share; top-}K\\text{ is a hard choice, so no gradient}` : st === 1 ? `\\text{each token's scores are rescaled to sum to 1, then averaged over tokens: smooth, so it carries the gradient}` : `\\downarrow\\text{ the gradient lowers }P\\text{ of busy experts; a large }\\alpha\\text{ hurts quality, so V3 keeps it tiny}`} />
        <text x={20} y={60 - 36} fontFamily={mono} fontSize="10" fill={T.dim}>click a bar, or pick an expert above: its numbers light up below</text>
      </g>
    );
  } else if (step.scene === "bias") {
    const rb = route(SEQ, BIAS_DEMO);
    const lb = loadsOf(rb);
    body = (
      <g>
        <Heat scores={SEQ} rt={rb} base={rt0} stage={1} bias={BIAS_DEMO} />
        <text x={50} y={10} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>BIAS b_j (ABOVE) · SELECT BY s + b</text>
        <Bars x={500} y={90} w={330} h={190} values={lb} ideal={(TT * k) / n} title="TOKENS PER EXPERT WITH BIAS" />
        <g fontFamily={mono} fontSize="11" fill={T.soft}>
          <text x={500} y={330}><tspan fill={T.accent2}>▢</tspan> newly chosen because of b</text>
          <text x={500} y={350}><tspan fill={T.wire}>⬚</tspan> dropped because of b</text>
          <text x={500} y={376} fill={T.dim}>gate g still computed from raw s</text>
        </g>
      </g>
    );
  } else if (step.scene === "monitor") {
    const l = SIM[0].loads;
    body = (
      <g>
        <Bars x={40} y={70} w={520} h={250} values={l} ideal={mu} title={`TOKENS PER EXPERT · ONE BATCH OF ${LB.m}`} flag={l.map((v, j) => (v > mu ? j : -1)).filter((j) => j >= 0)} />
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          <text x={600} y={120} fill={T.bad}>red ring = overloaded</text>
          <text x={600} y={144} fill={T.accent}>no ring = underloaded</text>
          <text x={600} y={190} fill={T.dim}>no loss term involved:</text>
          <text x={600} y={210} fill={T.dim}>just counting</text>
        </g>
      </g>
    );
  } else if (step.scene === "update") {
    const l = SIM[0].loads;
    const err = l.map((v) => mu - v);
    const b1 = SIM[0].next;
    const col = XC[ex];
    const c = (x) => `{\\color{${col}}\\mathbf{${x}}}`; // the numbers to point at, coloured like the expert's bars
    const E = `\\mathrm{E}_{${ex}}`;
    const sg = Math.sign(err[ex]);
    body = (
      <g>
        <Bars x={30} y={70} w={380} h={220} values={err} fmt={(v) => v.toFixed(0)} title="err_j = mean load − load_j" arrows={err.map((v) => Math.sign(v))} hl={ex} onPick={setEx} />
        {step.stage >= 1 && <Bars x={450} y={70} w={380} h={220} values={b1} min={-GAMMA * 1.5} max={GAMMA * 1.5} fmt={(v) => v.toFixed(3)} title="NEW BIAS b_j  (STARTED AT 0)" hl={ex} onPick={setEx} />}
        <Tex x={30} y={328} w={820} h={28} size={14} color={T.ink} tex={`\\mathrm{err}_{${E}}=\\bar\\ell-\\ell_{${E}}=${mu}-${c(l[ex])}=${c(err[ex])}\\qquad\\bar\\ell=\\dfrac{${LB.m}\\cdot ${k}}{${n}}=${mu}\\text{ (fair share)}`} />
        {step.stage >= 1 && <Tex x={30} y={358} w={820} h={28} size={14} color={T.accent2} tex={`b_{${E}}\\leftarrow b_{${E}}+\\gamma\\,\\mathrm{sign}(\\mathrm{err}_{${E}})=0+${GAMMA}\\cdot(${c(sg)})=${c((sg * GAMMA).toFixed(3))}`} />}
        <Tex x={30} y={392} w={820} h={24} size={11.5} color={T.dim} tex={step.stage === 0 ? `\\mathrm{err}>0:\\text{ underloaded}\;(\\uparrow\\text{ raise }b)\\qquad \\mathrm{err}<0:\\text{ overloaded}\;(\\downarrow\\text{ lower }b)` : `\\gamma\\text{ is fixed: a mild or a huge imbalance moves }b\\text{ by the same }\\pm${GAMMA}`} />
      </g>
    );
  } else if (step.scene === "loop") {
    const r = SIM[step.t];
    body = (
      <g>
        <Bars x={20} y={70} w={260} h={190} values={r.loads} ideal={mu} title="LOAD THIS STEP" />
        <Bars x={305} y={70} w={260} h={190} values={r.b} min={-0.3} max={0.3} fmt={(v) => f2(v)} title="BIAS b_j" />
        <Lines x={600} y={70} w={240} h={190} series={[{ pts: SIM.map((q) => q.cv), color: T.accent2, label: "load CV" }]} ymax={1} upTo={step.t} title="IMBALANCE OVER TIME" />
        <text x={20} y={310} fontFamily={mono} fontSize="12" fill={T.soft}>step {step.t} · bias of popular experts sinks, rare ones rise</text>
        <text x={20} y={334} fontFamily={mono} fontSize="11" fill={T.dim}>γ = {GAMMA} here; the paper uses 0.001 over many more steps</text>
      </g>
    );
  } else if (step.scene === "devroute") {
    const st = step.stage, rr = RR[0], ru = RU[0], row = DSC[0];
    const DX = (d) => 20 + d * 210, bx = (e) => DX(devOf(e)) + 22 + (e % DEV.Ed) * 52, BASE = 232, SC = 130;
    const keepSet = new Set(rr.keep), feeds = new Set(rr.top.flat()), chosen = new Set(rr.idx), ghost = new Set(ru.idx);
    body = (
      <g>
        <text x={20} y={34} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>TOKEN x0 · AFFINITY s OF {DEV.E} EXPERTS ON {DEV.D} DEVICES · K = {DEV.K}, M = {DEV.M}</text>
        {Array.from({ length: DEV.D }, (_, d) => {
          const kept = keepSet.has(d), out = st >= 1 && !kept;
          return (
            <g key={d} opacity={out ? 0.35 : 1} style={{ transition: "opacity .5s" }}>
              <rect x={DX(d)} y={50} width={190} height={226} rx={10} fill={T.panel} stroke={st >= 1 && kept ? T.accent : T.rule} strokeWidth={st >= 1 && kept ? 2.2 : 1.2} />
              <text x={DX(d) + 12} y={72} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.ink}>Device {d}</text>
              <text x={DX(d) + 178} y={72} textAnchor="end" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={st >= 1 ? (kept ? T.accent : T.bad) : T.wire}>
                {st === 0 ? `Σ = ${f2(rr.devScore[d])}` : kept ? `Σ ${f2(rr.devScore[d])} ✓` : `Σ ${f2(rr.devScore[d])} ✕`}
              </text>
            </g>
          );
        })}
        {row.map((v, e) => {
          const d = devOf(e), h = v * SC, x = bx(e);
          const masked = st >= 2 && !keepSet.has(d);
          const lit = st === 0 ? feeds.has(e) : st >= 3 ? chosen.has(e) : !masked;
          return (
            <g key={e} opacity={masked ? (st >= 3 && ghost.has(e) ? 0.55 : 0.2) : 1} style={{ transition: "opacity .5s" }}>
              <rect x={x} y={BASE - h} width={40} height={h} rx={4} fill={XC[d]} opacity={lit ? 0.95 : 0.32} stroke={st === 0 && feeds.has(e) ? T.ink : "none"} strokeWidth={1.6} />
              <text x={x + 20} y={BASE - h - 5} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.soft}>{f2(v)}</text>
              <text x={x + 20} y={BASE + 14} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[d]}>E{e}</text>
              {st >= 3 && chosen.has(e) && <text x={x + 20} y={BASE + 30} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.accent}>✓ g {f2(rr.g[rr.idx.indexOf(e)])}</text>}
              {st >= 3 && ghost.has(e) && !chosen.has(e) && <text x={x + 20} y={BASE - h - 17} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.bad}>▽</text>}
              {masked && <text x={x + 20} y={BASE - h / 2} textAnchor="middle" fontFamily={mono} fontSize="16" fill={T.bad}>×</text>}
            </g>
          );
        })}
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          {st === 0 && <text x={20} y={316}>white outline = the K/M = {DEV.K / DEV.M} best experts of each device: they make its score</text>}
          {st === 1 && <text x={20} y={316}>keep the M = {DEV.M} highest device scores: this bounds the tokens' network fan-out</text>}
          {st === 2 && <text x={20} y={316}>an expert on a dropped device can no longer be picked, however high its own score</text>}
          {st >= 3 && <text x={20} y={316}>chosen: E{rr.idx.join(", E")} on {rr.devs.length} device{rr.devs.length > 1 ? "s" : ""}</text>}
          {st >= 3 && <text x={20} y={340} fill={T.bad}>▽ = would be picked without the limit: E{ru.idx.join(", E")} on {ru.devs.length} devices</text>}
          {st >= 3 && <text x={20} y={364} fill={T.dim}>V3: at most 4 nodes per token; ~3.2 experts per node, so up to 13 routed experts cost the same traffic</text>}
        </g>
      </g>
    );
  } else if (step.scene === "shuffle") {
    const st = step.stage, per = DEV.B / DEV.D;
    const DX = (d) => 20 + d * 210, CX = (t) => DX(srcDev(t)) + 50 + (t % per) * 90, SY = 80, DY = 292;
    const endX = (t, d) => DX(d) + 24 + ((t * 37) % 142);
    const recv = Array(DEV.D).fill(0), expAt = Array(DEV.E).fill(0);
    RR.forEach((r) => { r.devs.forEach((d) => (recv[d] += 1)); r.idx.forEach((e) => (expAt[e] += 1)); });
    body = (
      <g>
        {Array.from({ length: DEV.D }, (_, d) => (
          <g key={`s${d}`}>
            <rect x={DX(d)} y={26} width={190} height={78} rx={10} fill={T.panel} stroke={T.rule} />
            <text x={DX(d) + 10} y={42} fontFamily={mono} fontSize="10" fill={T.dim}>device {d} · tokens</text>
          </g>
        ))}
        {Array.from({ length: DEV.D }, (_, d) => (
          <g key={`d${d}`}>
            <rect x={DX(d)} y={DY} width={190} height={92} rx={10} fill={T.panel} stroke={st >= 2 ? T.accent : T.rule} style={{ transition: "stroke .4s" }} />
            <text x={DX(d) + 10} y={DY + 15} fontFamily={mono} fontSize="10" fill={T.dim}>device {d} · experts</text>
            {st >= 1 && <text x={DX(d) + 180} y={DY + 15} textAnchor="end" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={T.wire}>recv {recv[d]}</text>}
            {Array.from({ length: DEV.Ed }, (_, j) => {
              const e = d * DEV.Ed + j;
              return (
                <g key={e}>
                  <rect x={DX(d) + 10 + j * 59} y={DY + 30} width={52} height={50} rx={6} fill={XC[d]} opacity={st >= 2 ? (expAt[e] ? 0.9 : 0.2) : 0.3} style={{ transition: "opacity .5s" }} />
                  <text x={DX(d) + 36 + j * 59} y={DY + 52} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.bg}>E{e}</text>
                  {st >= 2 && <text x={DX(d) + 36 + j * 59} y={DY + 68} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.bg}>{expAt[e]} tok</text>}
                </g>
              );
            })}
          </g>
        ))}
        {DSC.map((_, t) => (
          <g key={`t${t}`}>
            <circle cx={CX(t)} cy={SY} r={15} fill={XC[t % 8]} />
            <text x={CX(t)} y={SY + 4} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.bg}>x{t}</text>
            {st === 0 && <text x={CX(t)} y={SY + 30} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.soft}>→ D{RR[t].devs.join(" D")}</text>}
          </g>
        ))}
        {st >= 1 && st !== 2 && DSC.map((_, t) => (
          <g key={`l${t}`}>
            {RU[t].devs.filter((d) => !RR[t].devs.includes(d)).map((d) => (
              <path key={`g${d}`} d={`M${CX(t)},${SY + 16} C${CX(t)},190 ${endX(t, d)},190 ${endX(t, d)},${DY}`} fill="none" stroke={T.bad} strokeWidth={1.1} strokeDasharray="3 4" opacity={st === 1 ? 0.45 : 0} />
            ))}
            {RR[t].devs.map((d) => {
              const local = d === srcDev(t);
              const dd = st >= 3 ? `M${endX(t, d)},${DY} C${endX(t, d)},190 ${CX(t)},190 ${CX(t)},${SY + 16}` : `M${CX(t)},${SY + 16} C${CX(t)},190 ${endX(t, d)},190 ${endX(t, d)},${DY}`;
              return <path key={d} d={dd} fill="none" stroke={st >= 3 ? T.accent2 : XC[t % 8]} strokeWidth={local ? 1.2 : 1.8} strokeDasharray={local ? "2 4" : undefined} opacity={local ? 0.4 : 0.75} className={local ? undefined : st >= 3 ? "lane back" : "lane"} />;
            })}
          </g>
        ))}
        <g fontFamily={mono} fontSize="11.5">
          <text x={20} y={H - 26} fill={T.soft}>cross-device copies: <tspan fill={T.accent} fontWeight="700">{crossOf(RR)}</tspan> with the M = {DEV.M} limit · <tspan fill={T.bad} fontWeight="700">{crossOf(RU)}</tspan> without it</text>
          {st >= 1 && <text x={20} y={H - 8} fontSize="10.5" fill={T.dim}>dotted = stays on its own device · red dashes = traffic the limit saves</text>}
        </g>
      </g>
    );
  } else {
    const card = (x, y, c, a, b, d) => (
      <g>
        <rect x={x} y={y} width={380} height={150} rx={12} fill={T.well} stroke={c} strokeWidth={1.6} />
        <text x={x + 20} y={y + 34} fontFamily={mono} fontSize="14" fontWeight="700" fill={c}>{a}</text>
        <text x={x + 20} y={y + 68} fontFamily={mono} fontSize="12" fill={T.ink}>{b}</text>
        <text x={x + 20} y={y + 94} fontFamily={mono} fontSize="12" fill={T.soft}>{d}</text>
      </g>
    );
    body = (
      <g>
        {card(30, 60, T.accent, "bias b_i", "balances the whole batch", "no gradient · cannot hurt the LM loss")}
        {card(450, 60, T.accent2, "L_bal · tiny α", "stops extreme imbalance", "inside any single sequence")}
        <text x={30} y={280} fontFamily={mono} fontSize="12.5" fill={T.soft}>result: balanced load without dropping tokens (train or inference)</text>
      </g>
    );
  }
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

export { Scene as DsScene, STEPS as DS_STEPS, BLOCKS as DS_BLOCKS };

export default function DeepSeekView() {
  const [ex, setEx] = useState(0); // expert the equations are written out for
  return (
    <Stepper
      title="DeepSeek-V3 · load balancing"
      subtitle={`aux loss → auxiliary-loss-free bias · toy: N=${n} experts, K=${k}`}
      blocks={BLOCKS}
      steps={STEPS}
      scene={(step) => <Scene step={step} ex={ex} setEx={setEx} />}
      header={(step) => (step.scene === "aux" || step.scene === "update") && (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>point at expert</span>
          {Array.from({ length: n }, (_, j) => (
            <button key={j} onClick={() => setEx(j)} style={{ ...btn(j === ex, false), height: 28, padding: "0 9px", color: j === ex ? undefined : XC[j] }}>E{j}</button>
          ))}
        </div>
      )}
      glossary={(step) => (step.block === "devroute" || step.block === "shuffle" ? DEV_GLOSSARY : DS_GLOSSARY)}
      shapes={(step) => (step.block === "devroute" || step.block === "shuffle" ? DEV_SHAPES : DS_SHAPES)[step.block === "loop" ? "loop" : step.sid]}
      legend={<span>source: DeepSeek-V3 Technical Report · <span style={{ color: T.wire }}>dashed</span> = balanced level</span>}
    />
  );
}
