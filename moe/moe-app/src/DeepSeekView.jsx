import { T, mono } from "./theme";
import { LB, XC, scoresOf, route, loadsOf, auxTerms, simulate } from "./lbModel";
import { Bars, Lines, f2 } from "./charts";
import Stepper from "./Stepper";
import { DS_GLOSSARY, DS_SHAPES } from "./lbShapes";

// DeepSeek-V3 (arXiv 2412.19437, §2.1.2): sigmoid router, auxiliary balance loss
// (eq. 17-20, sequence-wise), then auxiliary-loss-free balancing (eq. 16):
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

const BLOCKS = [
  { id: "route", label: "route with sigmoid + top-K", sub: "eq. 13–15", code: ["s   = sigmoid(u @ e.T)            # [T,d]·[d,N] → [T,N]  (eq 15)", "idx = topk(s, K)                   # [T,N] → [T,K]  (eq 14)", "g   = s[idx] / s[idx].sum()        # [T,K]  (eq 13)"] },
  { id: "aux", label: "auxiliary balance loss", sub: "eq. 17–20 · f_i · P_i", code: ["f = N/(K*T) * onehot(idx).sum(0)   # [T,K,N] → [N]  (eq 18)", "P = (s / s.sum(-1)).mean(0)         # [T,N] → [N]  (eq 19-20)", "L_bal = alpha * (f * P).sum()       # [N]·[N] → scalar  (eq 17)"] },
  { id: "bias", label: "bias steers selection", sub: "loss-free · eq. 16", code: ["idx = topk(s + b, K)               # [T,N] + [N] → [T,K]  (eq 16: bias picks…)", "g   = s[idx] / s[idx].sum()        # [T,K]  …gate uses s only"] },
  { id: "monitor", label: "monitor expert load", sub: "whole batch, every step", code: ["load = onehot(idx).sum(0)          # [T,K,N] → [N]  tokens per expert"] },
  { id: "update", label: "bias update", sub: "± γ", code: ["err = load.mean() - load         # [N]", "b  += gamma * sign(err)           # [N]  overloaded ↓, underloaded ↑"] },
  { id: "loop", label: "repeat each training step", sub: "balance emerges", loop: true, code: ["for step in training:", "    idx  = topk(s + b, K)", "    load = count(idx)", "    b   += gamma * sign(load.mean() - load)"] },
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
                    <text x={x0 + j * cw + (cw - 3) / 2} y={y0 + t * ch + 12} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="600" fill={T.ink}>
                      {stage >= 1.5 && on && !bias ? f2(s / gsum) : f2(s)}
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

function Scene({ step }) {
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
        {step.stage >= 1 && step.stage < 2 && <text x={500} y={120} fontFamily={mono} fontSize="11.5" fill={T.soft}>ringed = chosen{step.stage >= 1.5 ? " · number = gate g" : ""}</text>}
      </g>
    );
  } else if (step.scene === "aux") {
    const st = step.stage;
    body = (
      <g>
        <Bars x={20} y={70} w={250} h={180} values={AUX.f} ideal={1} fmt={f2} title="f_i  (FRACTION ROUTED)" flag={AUX.f.map((v, j) => (v > 1.5 ? j : -1)).filter((j) => j >= 0)} />
        {st >= 1 && <Bars x={305} y={70} w={250} h={180} values={AUX.P} ideal={1 / n} fmt={f2} title="P_i  (MEAN SOFT SCORE)" arrows={st >= 2 ? AUX.f.map((v) => (v > 1 ? -1 : 0)) : null} />}
        {st >= 2 && <Bars x={590} y={70} w={250} h={180} values={AUX.prod} ideal={1 / n} fmt={f2} title="f_i · P_i" />}
        <g fontFamily={mono} fontSize="11.5" fill={T.soft}>
          <text x={20} y={300}>f_i = N/(K·T) · #tokens that picked i      <tspan fill={T.dim}>1 = fair share · top-K, no gradient</tspan></text>
          {st >= 1 && <text x={20} y={324}>P_i = mean_t  s_it / Σ_j s_jt           <tspan fill={T.dim}>smooth · carries the gradient</tspan></text>}
          {st >= 2 && (
            <g>
              <text x={20} y={352} fontWeight="700" fill={T.accent2}>L_bal = α · Σ f_i·P_i = α · {AUX.loss.toFixed(2)}</text>
              <text x={20} y={374} fill={T.dim}>balanced router gives α · 1.00 · ▼ = gradient lowers P of busy experts</text>
              <text x={20} y={396} fill={T.dim}>too large α hurts quality, so V3 avoids leaning on it</text>
            </g>
          )}
        </g>
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
    body = (
      <g>
        <Bars x={30} y={70} w={380} h={220} values={err} fmt={(v) => v.toFixed(0)} title="mean load − load_j" arrows={err.map((v) => Math.sign(v))} />
        {step.stage >= 1 && <Bars x={450} y={70} w={380} h={220} values={b1} min={-GAMMA * 1.5} max={GAMMA * 1.5} fmt={(v) => v.toFixed(3)} title="NEW BIAS b_j  (STARTED AT 0)" />}
        {step.stage >= 1 && <text x={30} y={340} fontFamily={mono} fontSize="13" fontWeight="700" fill={T.accent2}>b_j ← b_j + γ · sign(mean − load_j)</text>}
        {step.stage >= 1 && <text x={30} y={364} fontFamily={mono} fontSize="11" fill={T.dim}>fixed step γ = {GAMMA}: same size for a mild or a huge imbalance</text>}
      </g>
    );
  } else if (step.scene === "loop") {
    const r = SIM[step.t];
    body = (
      <g>
        <Bars x={20} y={70} w={260} h={190} values={r.loads} ideal={mu} title="LOAD THIS STEP" />
        <Bars x={305} y={70} w={260} h={190} values={r.b} min={-0.18} max={0.18} fmt={(v) => f2(v)} title="BIAS b_j" />
        <Lines x={600} y={70} w={240} h={190} series={[{ pts: SIM.map((q) => q.cv), color: T.accent2, label: "load CV" }]} ymax={1} upTo={step.t} title="IMBALANCE OVER TIME" />
        <text x={20} y={310} fontFamily={mono} fontSize="12" fill={T.soft}>step {step.t} · bias of popular experts sinks, rare ones rise</text>
        <text x={20} y={334} fontFamily={mono} fontSize="11" fill={T.dim}>γ = {GAMMA} here; the paper uses 0.001 over many more steps</text>
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

export default function DeepSeekView() {
  return (
    <Stepper
      title="DeepSeek-V3 · load balancing"
      subtitle={`aux loss → auxiliary-loss-free bias · toy: N=${n} experts, K=${k}`}
      blocks={BLOCKS}
      steps={STEPS}
      scene={(step) => <Scene step={step} />}
      glossary={DS_GLOSSARY}
      shapes={(step) => DS_SHAPES[step.block === "loop" ? "loop" : step.sid]}
      legend={<span>source: DeepSeek-V3 Technical Report, §2.1.2 (eq. 12–20) · <span style={{ color: T.wire }}>dashed</span> = balanced level</span>}
    />
  );
}
