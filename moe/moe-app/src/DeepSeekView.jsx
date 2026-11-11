import { useState } from "react";
import { T, mono, btn } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, auxTerms, simulate } from "./lbModel";
import { Bars, Lines, f2 } from "./charts";
import Stepper from "./Stepper";
import { DS_GLOSSARY, DS_SHAPES } from "./lbShapes";
import { Tex } from "./tex";
import { useProgress, ease, lerp, clamp01, logit } from "./anim";

// DeepSeek-V3 (arXiv 2412.19437, §2.1.2): sigmoid router, auxiliary balance loss
// (eq. 17-20, sequence-wise), then auxiliary-loss-free balancing:
// a per-expert bias b_i steers top-K selection only; the gate still uses s.

const { n, k, T: TT } = LB;
const GAMMA = 0.02; // paper: 0.001; a larger toy value so 10 training steps are enough to see the balance emerge
const SEQ = scoresOf(1000, TT);
const AUX = auxTerms(SEQ);
const SIM = simulate("sign", 60, GAMMA);
const cvStd = (l) => { const m0 = l.reduce((a, x) => a + x, 0) / l.length; return Math.sqrt(l.reduce((a, x) => a + (x - m0) ** 2, 0) / l.length); }; // population std over the n experts
const ZERO = Array(n).fill(0);
const repicked = (b) => { const a = route(SEQ, ZERO), c = route(SEQ, b); return a.filter((r, t) => [...r.sel].sort().join() !== [...c[t].sel].sort().join()).length; }; // toy tokens whose top-K set changes under bias b
const LOOPT = Array.from({ length: 10 }, (_, i) => i + 1); // every training step, 1 to 10
const loopStep = (t) => ({ id: `t${t}`, block: "loop", lines: [1, 2, 3], visAt: 0, scene: "loop", t, prev: t - 1, caption: t === 1 ? "training step 1: b starts at 0, count the load, then nudge every b by ±γ" : `training step ${t}: route with the current b, count the load, nudge b again`, vars: () => [["training step", `${t}`], ["load CV", SIM[t - 1].cv.toFixed(2)]] });
const W = 860, H = 430;
const HG = { x0: 50, cw: 84, ch: 34, y0: 56 }; // heat-map grid: few cells, so draw them big

const BLOCKS = [
  { id: "route", label: "route with sigmoid + top-K", sub: "sigmoid scores → top-K → gates", code: [
    "s   = sigmoid(u @ e.T)   # u [T,d] @ e.T [d,N] → [T,N] · s[t,i] = token t's score for expert i",
    "idx = topk(s, K)   # top-K over N: [T,N] → [T,K] · idx[t] = token t's K expert ids",
    "g   = s[idx] / s[idx].sum(-1, keepdim=True)   # [T,K] / (sum over K → [T,1]) → [T,K] · g[t,k] = gate of token t's k-th expert"] },
  { id: "aux", label: "auxiliary balance loss", sub: "f_i · P_i", code: [
    "f = N/(K*T) * onehot(idx).sum((0, 1))   # [T,K,N] → sum over T,K → [N] · f[i] = expert i's share of routes (1 = fair)",
    "P = (s / s.sum(-1, keepdim=True)).mean(0)   # [T,N] / (sum over N → [T,1]) → [T,N] → mean over T → [N] · P[i] = expert i's avg soft score",
    "L_bal = alpha * (f * P).sum()   # [N]·[N] → [N] → sum over N → scalar · alpha = loss weight"] },
  { id: "monitor", label: "monitor expert load", sub: "whole batch, every step", code: [
    "load = onehot(idx).sum((0, 1))   # [T,K,N] → sum over T,K → [N] · load[i] = tokens expert i received"] },
  { id: "update", label: "bias update", sub: "± γ", code: [
    "err = load.mean() - load   # mean over N → scalar; load [N] → err [N] · err[i] > 0: expert i is underloaded",
    "b  += gamma * sign(err)   # [N] · gamma = step size; b[i] goes up if underloaded, down if overloaded"] },
  { id: "bias", label: "bias steers selection", sub: "loss-free", code: [
    "idx = topk(s + b, K)   # s [T,N] + b [N] → [T,N] → top-K over N → [T,K] · b[i] = expert i's bias, used only to pick",
    "g   = s[idx] / s[idx].sum(-1, keepdim=True)   # [T,K] · uses the raw s, not s + b"] },
  { id: "loop", label: "repeat each training step", sub: "balance emerges", loop: true, code: [
    "for step in training:   # one batch per step",
    "    idx  = topk(s + b, K)   # [T,K]",
    "    load = count(idx)   # [N] tokens per expert",
    "    b   += gamma * sign(load.mean() - load)   # [N]"] },
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
  { id: "monitor", block: "monitor", lines: [0], scene: "monitor", caption: "training step 1: route a bigger batch (64 tokens) with b = 0 and count tokens per expert", vars: () => [["batch", `${LB.m} tokens`], ["mean load", `${(LB.m * k) / n}`]] },
  { id: "err", block: "update", lines: [0], scene: "update", stage: 0, caption: "error: how far each expert is from the mean load", vars: () => [["γ", `${GAMMA}`]] },
  { id: "update", block: "update", lines: [1], scene: "update", stage: 1, caption: "overloaded: b − γ · underloaded: b + γ", vars: () => [["γ", `${GAMMA}`], ["paper γ", "0.001"]] },
  { id: "select", block: "bias", lines: [0, 1], visAt: 0, scene: "bias", t: 1, prev: 0, caption: `use the b from step 1 on the 8-token toy: only ${repicked(SIM[1].b)} of ${TT} tokens change their picks`, vars: () => [["bias after", "1 step"], ["select", "s + b"], ["weight", "s"]] },
  ...LOOPT.map(loopStep),
  { id: "select10", block: "bias", lines: [0, 1], visAt: 0, scene: "bias", t: 10, prev: 1, caption: `the b after 10 training steps on the same 8 tokens: ${repicked(SIM[10].b)} of ${TT} tokens now pick differently`, vars: () => [["bias after", "20 steps"], ["select", "s + b"], ["weight", "s"]] },
  { id: "recipe", block: "recipe", lines: [0, 1], visAt: 0, scene: "recipe", caption: "bias balances batches, tiny loss guards each sequence", vars: () => [["tokens dropped", "0"]] },
];

// sigmoid curve: the 8 logits of token x0 rise from the axis onto the curve
function Sigmoid({ x, y, w, h, zs, e }) {
  const px = (z) => x + ((z + 4) / 8) * w, py = (v) => y + h - v * h;
  const pts = Array.from({ length: 41 }, (_, i) => { const z = -4 + (i / 40) * 8; return `${i ? "L" : "M"}${px(z)},${py(1 / (1 + Math.exp(-z)))}`; }).join(" ");
  return (
    <g>
      <text x={x} y={y - 12} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>ROW x0 · LOGIT z → SCORE s</text>
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      <line x1={px(0)} x2={px(0)} y1={y} y2={y + h} stroke={T.rule} strokeDasharray="2 3" />
      <line x1={x} x2={x + w} y1={py(0.5)} y2={py(0.5)} stroke={T.rule} strokeDasharray="2 3" />
      <path d={pts} fill="none" stroke={T.dim} strokeWidth={1.6} />
      {zs.map((z, j) => (
        <g key={j}>
          <line x1={px(z)} x2={px(z)} y1={y + h} y2={py(e / (1 + Math.exp(-z)))} stroke={XC[j]} strokeWidth={1.2} strokeDasharray="2 2" opacity={0.7} />
          <circle cx={px(z)} cy={py((e * 1) / (1 + Math.exp(-z)))} r={5} fill={XC[j]} stroke={T.bg} />
        </g>
      ))}
      <text x={x + 6} y={y + 14} fontFamily={mono} fontSize="10" fill={T.dim}>s = 1</text>
      <text x={x + w - 6} y={y + h + 14} textAnchor="end" fontFamily={mono} fontSize="10" fill={T.dim}>z →</text>
      <Tex x={x} y={y + h + 20} w={w} h={30} size={13} color={T.soft} tex={`s=\\sigma(z)=\\dfrac{1}{1+e^{-z}}`} />
    </g>
  );
}

// f_i × P_i as an AREA: width = P_i, height = f_i, so the rectangle's area is the product
function AreaModel({ x, y, w, h, f, P, hl, onPick, e }) {
  const sp = P.reduce((a, v) => a + v, 0), mf = Math.max(...f);
  let cx = x;
  return (
    <g>
      <text x={x} y={y - 10} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>f_i × P_i = AREA (WIDTH P, HEIGHT f)</text>
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      {P.map((pj, j) => {
        const wj = (pj / sp) * w, hj = (f[j] / mf) * (h - 14) * e, xj = cx;
        cx += wj;
        return (
          <g key={j}>
            <rect x={xj + 1} y={y + h - hj} width={Math.max(wj - 2, 1)} height={Math.max(hj, 0.5)} rx={2} fill={XC[j]} fillOpacity={0.85} stroke={hl === j ? T.ink : "none"} strokeWidth={2.2} />
            {e > 0.95 && wj > 20 && f[j] * pj >= 0.03 && <text x={xj + wj / 2} y={y + h - hj - 4} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={hl === j ? T.ink : T.soft}>{f2(f[j] * pj)}</text>}
            <text x={xj + wj / 2} y={y + h + 14} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>
            {onPick && <rect x={xj} y={y} width={wj} height={h} fill="transparent" style={{ cursor: "pointer" }} onClick={() => onPick(j)} />}
          </g>
        );
      })}
    </g>
  );
}

function Heat({ scores, rt, base, stage, bias, bias0 = ZERO, e = 1 }) {
  const { x0, cw, ch, y0 } = HG;
  return (
    <g>
      {Array.from({ length: n }, (_, j) => (
        <text key={j} x={x0 + j * cw + cw / 2 - 2} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={XC[j]}>E{j}</text>
      ))}
      {bias && Array.from({ length: n }, (_, j) => {
        const v = lerp(bias0[j], bias[j], e), hh = Math.min(12, Math.abs(v) * 110);
        const base = y0 - 30;
        return <rect key={j} x={x0 + j * cw + 6} y={v >= 0 ? base - hh : base} width={cw - 14} height={Math.max(hh, 1)} rx={2} fill={v >= 0 ? T.accent : T.bad} opacity={0.85} />;
      })}
      {scores.map((row, t) => {
        const sel = rt[t].sel;
        const gsum = sel.reduce((a, j) => a + row[j], 0);
        return (
          <g key={t}>
            <text x={x0 - 8} y={y0 + t * ch + ch / 2 + 3} textAnchor="end" fontFamily={mono} fontSize="11" fill={T.dim}>x{t}</text>
            {row.map((s, j) => {
              const on = sel.includes(j);
              const was = base ? base[t].sel.includes(j) : on;
              const flip = base && on !== was;
              const topkAnim = stage === 1 && !bias;
              const show = stage === 0 || on || bias || (topkAnim && e < 1);
              // the variable as the current line changes it
              let shown = s, tint = s;
              if (stage === 0) { shown = lerp(logit(s), s, e); tint = lerp(0.35, s, e); } // logit z → sigmoid(z)
              else if (stage === 1.5 && on && !bias) shown = lerp(s, s / gsum, e); // s → s / Σ s_sel
              else if (stage > 1.5 && on && !bias) shown = s / gsum; // later steps keep the finished gate
              else if (bias) shown = s + lerp(bias0[j], bias[j], e); // s → s + b
              return (
                <g key={j} opacity={stage >= 1 && !on ? (topkAnim ? lerp(1, 0.28, e) : bias ? 0.28 : 0.28) : 1} transform={topkAnim && on ? `translate(0,${-2 * e})` : undefined}>
                  <rect
                    x={x0 + j * cw} y={y0 + t * ch} width={cw - 3} height={ch - 3} rx={3}
                    fill={`color-mix(in srgb, ${XC[j]} ${Math.round(tint * 80)}%, ${T.well})`}
                    stroke={flip ? (on ? T.accent2 : T.wire) : on && stage >= 1 ? T.ink : "none"} strokeWidth={flip ? 2.2 : 1.4} strokeOpacity={topkAnim && on ? e : 1} strokeDasharray={flip && !on ? "3 2" : "none"}
                  />
                  {show && (
                    <text x={x0 + j * cw + (cw - 3) / 2} y={y0 + t * ch + ch / 2 + 2} textAnchor="middle" fontFamily={mono} fontSize={stage >= 1.5 && on && !bias ? 11 : 12} fontWeight="600" fill={T.ink} opacity={topkAnim && !on ? clamp01(1 - 2 * e) : 1}>
                      {stage >= 1.5 && on && !bias ? `${f2(s)}→${f2(shown)}` : f2(shown)}
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
function Worked({ rt, stage, e = 1 }) {
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
      <rect x={HG.x0 - 4} y={HG.y0 - 3} width={n * HG.cw + 2} height={HG.ch + 3} rx={5} fill="none" stroke={T.accent} strokeWidth={1.6} />
      <text x={470} y={156} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>TOKEN x0 (OUTLINED ROW)</text>
      {stage === 1 && (
        <g>
          <Tex x={470} y={166} w={380} h={30} size={14} color={T.ink} tex={`\\mathrm{idx}=\\operatorname{TopK}(s,\\,K=${k})=\\{${names}\\}`} />
          <Tex x={470} y={200} w={380} h={30} size={13} color={T.soft} tex={`${vals}`} />
          <Tex x={470} y={234} w={380} h={30} size={12} color={T.dim} tex={`\\text{the other ${n - k} scores are dropped}`} />
          <text x={470} y={280} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim} opacity={e}>SELECTED SCORES → GATE ROW</text>
          {sel.map((j, i) => {
            const fx = HG.x0 + j * HG.cw + 18, fy = HG.y0 + 9, tx = 470 + i * 72, ty = 288;
            return (
              <g key={j}>
                <rect x={tx} y={ty} width={47} height={16} rx={3} fill="none" stroke={T.rule} strokeDasharray="2 3" />
                <rect x={lerp(fx, tx, e)} y={lerp(fy, ty, e)} width={47} height={16} rx={3} fill={`color-mix(in srgb, ${XC[j]} ${Math.round(row[j] * 80)}%, ${T.well})`} stroke={T.ink} strokeWidth={1.4} />
                <text x={lerp(fx, tx, e) + 23.5} y={lerp(fy, ty, e) + 12} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="600" fill={T.ink}>{f2(row[j])}</text>
              </g>
            );
          })}
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
  const p = useProgress(step.id); // 0 → 1 each time a new picture arrives
  const e = ease(p);
  const rt0 = route(SEQ, ZERO);
  const mu = (LB.m * k) / n;
  let body = null;

  if (step.scene === "heat") {
    body = (
      <g>
        <Heat scores={SEQ} rt={rt0} stage={step.stage} e={e} />
        {step.stage === 0 && <Sigmoid x={480} y={90} w={350} h={190} zs={SEQ[0].map(logit)} e={e} />}
        {step.stage === 2 && (
          <g>
            <Bars x={500} y={90} w={330} h={220} values={AUX.cnt.map((v) => v * e)} ideal={(TT * k) / n} fmt={(v) => v.toFixed(0)} max={Math.max(...AUX.cnt) * 1.12} title="TOKENS PER EXPERT" flag={AUX.cnt.map((c, j) => (c > 2 * ((TT * k) / n) ? j : -1)).filter((j) => j >= 0)} />
            <text x={500} y={355} fontFamily={mono} fontSize="11" fill={T.dim}>E0 and E1 take {AUX.cnt[0] + AUX.cnt[1]} of {TT * k} routes</text>
          </g>
        )}
        {step.stage >= 1 && step.stage < 2 && <Worked rt={rt0} stage={step.stage} e={e} />}
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
        {st === 1 ? (
          <g fontFamily={mono} fontSize="11.5">
            <text x={20} y={62} fontSize="10" letterSpacing="0.1em" fill={T.dim}>{`P_${ex}: ONE ROW PER TOKEN, EXPERT ${ex}`}</text>
            {["token", `s_${ex}`, "Σ_j s_j", "ratio"].map((h, i) => <text key={h} x={[20, 84, 148, 214][i]} y={84} fill={T.dim}>{h}</text>)}
            {SEQ.map((_, t) => {
              const q = ease(clamp01((e * (TT + 1) - t) / 1)), y = 104 + t * 18;
              return (
                <g key={t}>
                  <text x={20} y={y} fill={T.soft}>x{t}</text>
                  <text x={84} y={y} fill={col}>{d2(SEQ[t][ex])}</text>
                  <text x={148} y={y} fill={T.soft}>{d2(rowSum(t))}</text>
                  <text x={214} y={y} fill={col} fontWeight="700" fillOpacity={q}>{d2(SEQ[t][ex] / rowSum(t))}</text>
                </g>
              );
            })}
            <line x1={20} x2={262} y1={250 - 6} y2={250 - 6} stroke={T.rule} />
            <text x={148} y={262} fill={T.soft} textAnchor="end">mean of ratios =</text>
            <text x={214} y={262} fill={col} fontWeight="700" fillOpacity={ease(clamp01(e * 2 - 1))}>{d2(AUX.P[ex])}</text>
          </g>
        ) : (
          <Bars x={20} y={70} w={250} h={180} values={AUX.f.map((v) => v * (st === 0 ? e : 1))} max={Math.max(...AUX.f) * 1.12} ideal={1} fmt={f2} title="f_i  (FRACTION ROUTED)" hl={ex} onPick={setEx} flag={AUX.f.map((v, j) => (v > 1.5 ? j : -1)).filter((j) => j >= 0)} />
        )}
        {st >= 1 && <Bars x={305} y={70} w={250} h={180} values={AUX.P.map((v) => v * (st === 1 ? e : 1))} max={Math.max(...AUX.P) * 1.12} ideal={1 / n} fmt={f2} title="P_i  (MEAN SOFT SCORE)" hl={ex} onPick={setEx} arrows={st >= 2 ? AUX.f.map((v) => (v > 1 ? -1 : 0)) : null} />}
        {st >= 2 && <AreaModel x={590} y={70} w={250} h={180} f={AUX.f} P={AUX.P} hl={ex} onPick={setEx} e={e} />}
        <Tex x={20} y={284} w={820} h={42} size={14} color={T.ink} tex={`f_{${E}}=\\dfrac{N}{K\\,T}\\,n_{${E}}=\\dfrac{${n}}{${k}\\cdot ${TT}}\\cdot ${c(AUX.cnt[ex])}=${c(d2(AUX.f[ex]))}`} />
        {st >= 1 && <Tex x={20} y={330} w={840} h={42} size={14} color={T.ink} tex={`P_{${E}}=\\dfrac{1}{T}\\sum_{t=1}^{T}\\dfrac{s_{${E},t}}{\\sum_{j} s_{j,t}}=\\dfrac{1}{${TT}}\\Big(${term(0)}+${term(1)}+\\cdots+${term(TT - 1)}\\Big)=\\dfrac{${c(d2(AUX.P[ex] * TT))}}{${TT}}=${c(d2(AUX.P[ex]))}`} />}
        {st >= 2 && <Tex x={20} y={376} w={840} h={28} size={14} color={T.accent2} tex={`f_{${E}}P_{${E}}=${d2(AUX.f[ex])}\\cdot ${d2(AUX.P[ex])}=${c(d2(AUX.prod[ex]))}\\qquad L_{\\mathrm{bal}}=\\alpha\\sum_{i} f_i P_i=\\alpha\\,(\\cdots+${c(d2(AUX.prod[ex]))}+\\cdots)=\\alpha\\cdot ${d2(AUX.loss)}`} />}
        <Tex x={20} y={406} w={840} h={22} size={11.5} color={T.dim} tex={st === 0 ? `n_{${E}}=\\text{tokens that picked }${E}\\text{ (of }${TT}\\text{); }f=1\\text{ is a fair share; top-}K\\text{ is a hard choice, so no gradient}` : st === 1 ? `\\text{each token's scores are rescaled to sum to 1, then averaged over tokens: smooth, so it carries the gradient}` : `\\downarrow\\text{ the gradient lowers }P\\text{ of busy experts; a large }\\alpha\\text{ hurts quality, so V3 keeps it tiny}`} />
        <text x={20} y={60 - 36} fontFamily={mono} fontSize="10" fill={T.dim}>click a bar, or pick an expert above: its numbers light up below</text>
      </g>
    );
  } else if (step.scene === "bias") {
    // bias after step.prev training steps -> after step.t: the picture runs through every step in between
    const t0 = step.prev, t1 = step.t;
    const b0 = SIM[t0].b, b1 = SIM[t1].b;
    const r0 = route(SEQ, b0), rb = route(SEQ, b1);
    const l0 = loadsOf(r0), lb = loadsOf(rb);
    const tn = Math.round(lerp(t0, t1, e)); // training steps counted so far
    const bNow = b0.map((v, j) => lerp(v, b1[j], e));
    const hist = SIM.slice(0, t1), mu25 = (l) => l.reduce((a2, v) => a2 + v, 0) / l.length;
    const sgn = (r) => (r.loads[ex] < mu25(r.loads) ? 1 : r.loads[ex] > mu25(r.loads) ? -1 : 0);
    const up = hist.slice(0, tn).filter((r) => sgn(r) > 0).length, down = hist.slice(0, tn).filter((r) => sgn(r) < 0).length;
    const bv = SIM[tn].b[ex], cl = `{\\color{${XC[ex]}}\\mathbf{${bv >= 0 ? "+" : "-"}${Math.abs(bv).toFixed(2)}}}`;
    const cw2 = Math.min(13.6, 330 / t1);
    body = (
      <g>
        <Heat scores={SEQ} rt={rb} base={r0} stage={1} bias={b1} bias0={b0} e={e} />
        <text x={50} y={10} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim}>BAR ABOVE EACH COLUMN = BIAS b_j (GREEN +, RED −) · SELECT BY s + b</text>
        {/* the bias value of every expert, spelled out under its own column */}
        <g fontFamily={mono} textAnchor="middle">
          <text x={HG.x0 - 8} y={362} textAnchor="end" fontSize="12" fontWeight="700" fill={T.wire}>b_j</text>
          <text x={HG.x0 - 8} y={384} textAnchor="end" fontSize="11" fill={T.dim}>load</text>
          {Array.from({ length: n }, (_, j) => {
            const v = bNow[j], cx = HG.x0 + j * HG.cw + HG.cw / 2 - 2;
            return (
              <g key={j}>
                <text x={cx} y={362} fontSize="14" fontWeight="700" fill={v >= 0 ? T.accent : T.bad}>{`${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`}</text>
                <text x={cx} y={384} fontSize="11" fill={T.soft}>{`${l0[j]} → ${Math.round(lerp(l0[j], lb[j], e))}`}</text>
              </g>
            );
          })}
          <text x={HG.x0} y={406} textAnchor="start" fontSize="11" fill={T.dim}>b &lt; 0: was overloaded, harder to pick</text>
          <text x={HG.x0} y={422} textAnchor="start" fontSize="11" fill={T.dim}>b &gt; 0: was underloaded, easier to pick</text>
        </g>
        <Bars x={500} y={80} w={330} h={150} values={l0.map((v, j) => lerp(v, lb[j], e))} ideal={(TT * k) / n} fmt={(v) => v.toFixed(0)} max={Math.max(...l0, ...lb, 6) * 1.12} title="TOKENS PER EXPERT WITH BIAS" idealLabel="" hl={ex} onPick={setEx} />
        <g fontFamily={mono} fontSize="11" fill={T.soft}>
          <text x={500} y={264}><tspan fill={T.accent2}>▢</tspan> newly chosen because of b</text>
          <text x={500} y={280}><tspan fill={T.wire}>⬚</tspan> dropped because of b · dashed line = fair share {(TT * k) / n}</text>
          <text x={500} y={296} fill={T.dim}>gate g still computed from raw s</text>
        </g>
        {/* how b itself is calculated: the sign update, one training step at a time */}
        <Tex x={500} y={304} w={350} h={30} size={13} color={T.ink} tex={t1 === 1 ? `b_j=\\gamma\\,\\mathrm{sign}\\big(\\bar\\ell-\\ell_j\\big)` : `b_j=\\gamma\\sum_{t=1}^{${t1}}\\mathrm{sign}\\big(\\bar\\ell-\\ell_j^{(t)}\\big)`} />
        <Tex x={500} y={330} w={350} h={34} size={13} color={T.accent2} tex={`b_{${ex}}=${GAMMA}\\cdot\\big({\\color{${T.accent}}${up}}-{\\color{${T.bad}}${down}}\\big)=${cl}`} />
        {/* one cell per training step (= one batch of 64 tokens): did E{ex} get fewer or more tokens than the fair share? */}
        {hist.map((r, t) => {
          const d = sgn(r);
          return <rect key={t} x={500 + t * cw2} y={366} width={cw2 - 2.6} height={12} rx={2} fill={d > 0 ? T.accent : d < 0 ? T.bad : T.rule} opacity={t < tn ? 0.9 : 0.15} />;
        })}
        <g fontFamily={mono} fontSize="11">
          <text x={500} y={392} fill={T.accent}>■ {up} {up === 1 ? "step" : "steps"}: E{ex} got FEWER than {mu} tokens → +{GAMMA}</text>
          <text x={500} y={406} fill={T.bad}>■ {down} {down === 1 ? "step" : "steps"}: E{ex} got MORE than {mu} tokens → −{GAMMA}</text>
          <text x={500} y={420} fill={T.dim}>{up + down < tn ? `■ ${tn - up - down} step: exactly ${mu} → no change · ` : ""}each cell = 1 training step (64 tokens)</text>
        </g>
      </g>
    );
  } else if (step.scene === "monitor") {
    const l = SIM[0].loads;
    body = (
      <g>
        <Bars x={40} y={70} w={520} h={250} values={l.map((v) => v * e)} max={Math.max(...l) * 1.12} fmt={(v) => v.toFixed(0)} ideal={mu} title={`TOKENS PER EXPERT · ONE BATCH OF ${LB.m}`} flag={l.map((v, j) => (v > mu ? j : -1)).filter((j) => j >= 0)} />
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
    const maxL = Math.max(...l), maxE = Math.max(...err.map(Math.abs));
    const e1 = ease(clamp01(p / 0.5)), e2 = ease(clamp01((p - 0.5) / 0.5));
    let left, leftTitle, leftMax, leftMin, ideal;
    if (step.stage === 0) {
      // load_j → load_j − mean (slide down by the mean) → mean − load_j (flip)
      left = l.map((v) => (v - e1 * mu) * (1 - 2 * e2));
      leftTitle = e2 > 0 ? "err_j = mean − load_j" : e1 > 0 ? "load_j − mean" : "load_j  (TOKENS PER EXPERT)";
      leftMax = maxL * 1.1; leftMin = -maxL * 0.8;
      ideal = e2 === 0 ? mu * (1 - e1) : undefined;
    } else {
      // err_j → sign(err_j): every bar collapses to the same height, only the direction survives
      left = err.map((v) => lerp(v, Math.sign(v) * maxE * 0.55, e1));
      leftTitle = e1 > 0.5 ? "sign(err_j)" : "err_j";
      leftMax = maxE * 1.1; leftMin = -maxE * 0.9;
    }
    const settled = p >= 1;
    body = (
      <g>
        <Bars
          x={30} y={70} w={380} h={220} values={left} fmt={step.stage === 0 ? (v) => v.toFixed(0) : () => ""} title={leftTitle} max={leftMax} min={leftMin}
          ideal={ideal} idealLabel="mean" arrows={step.stage === 0 && settled ? err.map((v) => Math.sign(v)) : null} hl={ex} onPick={setEx}
        />
        {step.stage >= 1 && <Bars x={450} y={70} w={380} h={220} values={b1.map((v) => v * e2)} min={-GAMMA * 1.5} max={GAMMA * 1.5} fmt={(v) => v.toFixed(3)} title="NEW BIAS b_j  (STARTED AT 0)" hl={ex} onPick={setEx} />}
        {step.stage >= 1 && <text x={420} y={190} textAnchor="middle" fontFamily={mono} fontSize="12" fill={T.wire}>× γ</text>}
        <Tex x={30} y={334} w={820} h={30} size={14} color={T.ink} tex={`\\mathrm{err}_{${E}}=\\bar\\ell-\\ell_{${E}}=${mu}-${c(l[ex])}=${c(err[ex])}`} />
        <text x={30} y={26} fontFamily={mono} fontSize="11.5" fill={T.soft}>ℓ̄ = fair share = {LB.m} tokens × top-{k} ÷ {n} experts = {mu} · ℓ_{ex} = tokens E{ex} received = {l[ex]}</text>
        {step.stage >= 1 && <Tex x={30} y={366} w={820} h={30} size={14} color={T.ink} tex={`\\mathrm{sign}(\\mathrm{err}_{${E}})=\\mathrm{sign}(${c(err[ex])})=${c(sg > 0 ? "+1" : sg < 0 ? "-1" : "0")}\\qquad \\bigl(\\mathrm{sign}(x)=-1\\text{ if }x<0,\\ +1\\text{ if }x>0,\\ 0\\text{ if }x=0\\bigr)`} />}
        {step.stage >= 1 && <Tex x={30} y={398} w={820} h={30} size={14} color={T.accent2} tex={`b_{${E}}\\leftarrow b_{${E}}+\\gamma\\,\\mathrm{sign}(\\mathrm{err}_{${E}})=0+${GAMMA}\\cdot(${c(sg > 0 ? "+1" : sg < 0 ? "-1" : "0")})=${c((sg * GAMMA).toFixed(3))}`} />}
        {step.stage === 0 && <Tex x={30} y={372} w={820} h={24} size={11.5} color={T.dim} tex={`\\mathrm{err}>0:\\text{ underloaded}\\;(\\uparrow\\text{ raise }b)\\qquad \\mathrm{err}<0:\\text{ overloaded}\\;(\\downarrow\\text{ lower }b)`} />}
      </g>
    );
  } else if (step.scene === "loop") {
    // training step t: the batch is routed with the current b (= SIM[t-1].b), its load is counted, then b is nudged to SIM[t].b
    const t = step.t, a0 = SIM[Math.max(step.prev - 1, 0)], a1 = SIM[t - 1], bNext = SIM[t].b;
    const eL = ease(clamp01(p / 0.5)), eB = ease(clamp01((p - 0.5) / 0.5)); // load first, then the bias update
    const loadsNow = a1.loads.map((v, j) => lerp(a0.loads[j], v, eL));
    const bNow = a1.b.map((v, j) => lerp(v, bNext[j], eB));
    const sf = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
    const dv = a1.loads.map((v) => v - mu); // > 0: more tokens than the fair share
    const colX = (j) => 150 + j * 110;
    const names = (pick) => dv.map((d, j) => (pick(d) ? `E${j}` : null)).filter(Boolean).join(" ") || "none";
    body = (
      <g>
        {/* the loop trains on bigger batches than the 8-token toy: say so, and show where 32 comes from */}
        <text x={20} y={22} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.wire}>each training step = a NEW batch of {LB.m} tokens (not the 8-token toy)</text>
        <text x={20} y={42} fontFamily={mono} fontSize="11.5" fill={T.soft}>{LB.m} tokens × top-{k} = {LB.m * k} picks, shared by {n} experts → fair share {LB.m * k} / {n} = {mu}</text>
        <Bars x={20} y={70} w={260} h={190} values={loadsNow} ideal={mu} title={`TOKENS PER EXPERT · STEP ${t}`} />
        <Bars x={305} y={70} w={260} h={190} values={bNow} min={-0.3} max={0.3} fmt={(v) => f2(v)} title="BIAS b_j AFTER THE UPDATE" />
        <Lines x={600} y={70} w={240} h={190} series={[{ pts: SIM.map((q) => q.cv), color: T.accent2, label: "load CV" }]} ymax={1} upTo={t - 1} title="IMBALANCE OVER TIME" />
        {/* the numbers, expert by expert: load, distance from the fair share, which way b moves, b before -> after */}
        <g fontFamily={mono} fontSize="11.5">
          {["tokens", `vs fair ${mu}`, "b moves", "bias b"].map((lab, r) => <text key={lab} x={20} y={310 + r * 18} fill={T.dim}>{lab}</text>)}
          {Array.from({ length: n }, (_, j) => (
            <g key={j} textAnchor="middle">
              <text x={colX(j)} y={292} fontWeight="700" fill={XC[j]}>E{j}</text>
              <text x={colX(j)} y={310} fill={T.ink}>{Math.round(loadsNow[j])}</text>
              <text x={colX(j)} y={328} fontWeight="700" fill={dv[j] > 0 ? T.bad : dv[j] < 0 ? T.accent : T.dim} fillOpacity={eL}>{dv[j] > 0 ? `+${dv[j]} more` : dv[j] < 0 ? `${dv[j]} fewer`.replace("-", "−") : "equal"}</text>
              <text x={colX(j)} y={346} fontWeight="700" fill={dv[j] > 0 ? T.bad : dv[j] < 0 ? T.accent : T.dim} fillOpacity={eB}>{dv[j] > 0 ? `−${GAMMA}` : dv[j] < 0 ? `+${GAMMA}` : "0"}</text>
              <text x={colX(j)} y={364} fill={T.soft}>{`${sf(a1.b[j])} → ${sf(bNow[j])}`}</text>
            </g>
          ))}
        </g>
        <Tex x={600} y={284} w={250} h={44} size={13.5} color={T.accent2} tex={`\\mathrm{CV}=\\dfrac{\\mathrm{std}(\\ell)}{\\bar\\ell}=\\dfrac{${cvStd(a1.loads).toFixed(1)}}{${mu}}=\\mathbf{${a1.cv.toFixed(2)}}`} />
        <text x={600} y={346} fontFamily={mono} fontSize="11" fill={T.dim}>CV = std(tokens) / mean tokens</text>
        <text x={600} y={362} fontFamily={mono} fontSize="11" fill={T.dim}>0 = perfectly even</text>
        <g fontFamily={mono} fontSize="11.5">
          <text x={20} y={396} fill={T.bad}>{names((d) => d > 0)}: more than {mu} tokens → b goes DOWN by {GAMMA} (picked less next time)</text>
          <text x={20} y={414} fill={T.accent}>{names((d) => d < 0)}: fewer than {mu} tokens → b goes UP by {GAMMA} (picked more next time)</text>
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
        <g opacity={ease(clamp01(p / 0.5))} transform={`translate(0,${(1 - ease(clamp01(p / 0.5))) * 14})`}>{card(30, 60, T.accent, "bias b_i", "balances the whole batch", "no gradient · cannot hurt the LM loss")}</g>
        <g opacity={ease(clamp01((p - 0.3) / 0.5))} transform={`translate(0,${(1 - ease(clamp01((p - 0.3) / 0.5))) * 14})`}>{card(450, 60, T.accent2, "L_bal · tiny α", "stops extreme imbalance", "inside any single sequence")}</g>
        <text x={30} y={280} fontFamily={mono} fontSize="12.5" fill={T.soft} opacity={ease(clamp01((p - 0.6) / 0.4))}>result: balanced load without dropping tokens (train or inference)</text>
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
      glossary={() => DS_GLOSSARY}
      shapes={(step) => DS_SHAPES[step.block === "loop" ? "loop" : step.sid]}
      legend={<span>source: DeepSeek-V3 Technical Report · <span style={{ color: T.wire }}>dashed</span> = balanced level</span>}
    />
  );
}
