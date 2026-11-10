import { useState } from "react";
import { T, mono, btn } from "./theme";
import { XC } from "./lbModel";
import Stepper from "./Stepper";
import { Tex } from "./tex";
import { useProgress, ease, lerp, clamp01 } from "./anim";

// Gating functions: softmax vs sigmoid vs sqrt(softplus). Every number on screen comes
// from the JS functions below (no typed-in results).

const N = 4, K = 2;
const ZF = [2, 1, -0.5, -1.2]; // the followed token's router logits
const CMAX = 6, TSTEPS = 60; // confidence 1 -> 6 over 60 toy training steps
const BIAS = 0.3;
const COUPLE_Z = 2.5; // new logit for E3 in the coupling demo
const BZ = [1.6, 0.4]; // two close experts for the bias demo
const GROUPS = [[3, 0], [2, 2]]; // group A, group B logits

// ---- real implementations ---------------------------------------------------
const sum = (a) => a.reduce((s, v) => s + v, 0);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
const sqrtSoftplus = (z) => Math.sqrt(softplus(z));
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = sum(e); return e.map((v) => v / s); };
const SCORE = { soft: softmax, sig: (z) => z.map(sigmoid), sp: (z) => z.map(sqrtSoftplus) };
const topk = (z, k = K) => z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((a) => a[1]);
// scores -> top-2 -> renormalise the 2 chosen
const gateOf = (kind, z) => {
  const s = SCORE[kind](z), idx = topk(z), t = sum(idx.map((i) => s[i]));
  return { s, idx, g: idx.map((i) => s[i] / t) };
};
const w1At = (kind, c) => gateOf(kind, ZF.map((v) => v * c)).g[0];
const ent = (g) => -sum(g.map((v) => (v > 0 ? v * Math.log(v) : 0)));

// ---- toy collapse simulation ------------------------------------------------
// 8 tokens x 4 experts, fixed seeded logits z0. Training makes the router more
// confident: at step t every logit is scaled, z = c(t) * z0, c(t) = 1 + 5 t/59.
// Per step, per gate: top-2 -> renormalised weights g, then averaged over tokens:
//   w2     = mean g[1]                 (2nd chosen expert's weight)
//   nEff   = mean exp(entropy(g))      (1 = one expert does everything, 2 = both equal)
//   grad2  = mean |d g[1] / d z[2nd]|  (what the 2nd expert's logit learns from; central difference)
const rng = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const NT = 8;
const Z0 = (() => {
  const r = rng(21), pop = [2.2, 1.3, 0.5, 0.1];
  return Array.from({ length: NT }, (_, t) => (t === 0 ? ZF : pop.map((p) => Math.max(0.15, p + 0.3 * gauss(r)))));
})();
const cOf = (t) => 1 + ((CMAX - 1) * t) / (TSTEPS - 1);
const simulate = (kind) => {
  const out = { w2: [], nEff: [], grad: [], w1tok0: [] };
  for (let t = 0; t < TSTEPS; t++) {
    const c = cOf(t);
    let w2 = 0, ne = 0, gr = 0;
    Z0.forEach((z0) => {
      const z = z0.map((v) => v * c), { idx, g } = gateOf(kind, z);
      w2 += g[1]; ne += Math.exp(ent(g));
      const h = 1e-5, zp = z.slice(), zm = z.slice();
      zp[idx[1]] += h; zm[idx[1]] -= h;
      gr += Math.abs(gateOf(kind, zp).g[1] - gateOf(kind, zm).g[1]) / (2 * h);
    });
    out.w2.push(w2 / NT); out.nEff.push(ne / NT); out.grad.push(gr / NT);
    out.w1tok0.push(w1At(kind, c));
  }
  return out;
};
const SIM = { soft: simulate("soft"), sig: simulate("sig"), sp: simulate("sp") };
const GMAX = Math.max(...["soft", "sig", "sp"].map((k) => Math.max(...SIM[k].grad)));

const W = 860, H = 430;
const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const cc = (col, s) => `{\\color{${col}}\\mathbf{${s}}}`;
const lbl = (s, x, y, o = {}) => <text x={x} y={y} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim} {...o}>{s}</text>;
const GATES = [
  { id: "soft", name: "softmax", col: T.bad },
  { id: "sig", name: "sigmoid", col: T.accent },
  { id: "sp", name: "sqrt-softplus", col: T.wire },
];
const GC = { soft: T.bad, sig: T.accent, sp: T.wire };
const GN = { soft: "softmax", sig: "sigmoid", sp: "sqrt-softplus" };

// ---- code ------------------------------------------------------------------
const BLOCKS = [
  { id: "logits", label: "router logits", sub: "one token, 4 experts", code: [
    "z = x @ Wg   # x [d] @ Wg [d,N] → [N] · z[j] = this token's raw score for expert j, N=4"] },
  { id: "score", label: "two ways to score", sub: "same z, different curve", code: [
    "s_soft = softmax(z)   # [N] → [N] · e^z[j] / Σ_j e^z[j], sum over N experts → scores sum to 1 (experts compete)",
    "s_sig = sigmoid(z)   # [N] → [N] · 1 / (1 + e^-z[j]) · each in (0,1), independent of the others",
    "s_sp = sqrt(softplus(z))   # [N] → [N] · sqrt(ln(1 + e^z[j])) · each ≥ 0, no cap, independent of the others"] },
  { id: "couple", label: "who influences whom", sub: "change one expert, watch the rest", code: [
    "z[3] = 2.5   # change ONE expert's logit (z [N]), then recompute the scores",
    "s_soft[j] = exp(z[j]) / exp(z).sum()   # denominator sums over ALL N experts → a bigger E3 shrinks E0..E2",
    "s_sig[j] = 1 / (1 + exp(-z[j]))   # uses z[j] only → E0..E2 do not move",
    "s_sp[j] = sqrt(log(1 + exp(z[j])))   # uses z[j] only → E0..E2 do not move"] },
  { id: "conf", label: "router gets confident", sub: "z → c·z, watch g[0]", loop: "c: 1 → 6", code: [
    "z = c * z   # [N] · c = confidence, 1 → 6 · same ranking, bigger gaps",
    "g = s[idx] / s[idx].sum()   # [2] · weights of the 2 chosen experts, sum over them = 1 · g[0] = top expert's weight"] },
];

const STEPS = [
  { id: "logits", block: "logits", lines: [0], scene: "score", kind: null, caption: "one token, four router logits", vars: () => [["z", `[${ZF.map(f1).join(", ")}]`]] },
  { id: "soft", block: "score", lines: [0], scene: "score", kind: "soft", caption: "softmax: exponentiate and divide by the total, scores compete", vars: () => [["Σ scores", f2(sum(SCORE.soft(ZF)))]] },
  { id: "sig", block: "score", lines: [1], scene: "score", kind: "sig", caption: "sigmoid: every expert squashed on its own into (0,1)", vars: () => [["Σ scores", f2(sum(SCORE.sig(ZF)))]] },
  { id: "sp", block: "score", lines: [2], scene: "score", kind: "sp", caption: "sqrt-softplus: also per expert, but with no upper cap", vars: () => [["Σ scores", f2(sum(SCORE.sp(ZF)))]] },
  { id: "side", block: "score", lines: [0, 1, 2], visAt: 0, scene: "side", caption: "same z, three scorers side by side: same ranking, different sizes", vars: () => GATES.map((g) => [g.name + " Σ", f2(sum(SCORE[g.id](ZF)))]) },
  { id: "couple", block: "couple", lines: [0, 1, 2, 3], visAt: 0, scene: "couple", caption: "softmax: raise E3 and everyone else drops. sigmoid and sqrt-softplus: the others stay put", vars: () => [["E3 logit", `${ZF[3]} → ${COUPLE_Z}`]] },
  { id: "conf", block: "conf", lines: [0, 1], visAt: 0, scene: "conf", caption: "more confident router: softmax runs to top-1, sigmoid to an equal split, sqrt-softplus in between", vars: () => GATES.map((g) => [g.name + " g0 @c=6", f2(w1At(g.id, CMAX))]) },
];

// ---- drawing pieces ----------------------------------------------------------
function VB({ x, y, w, h, vals, lo = 0, hi, cols = XC, hl, rings = [], fmt = f2, title, ghost, labelsOn = true }) {
  const n = vals.length, sc = h / (hi - lo || 1), zero = y + h - (0 - lo) * sc, bw = w / n;
  const rect = (v, j, extra) => {
    const bh = Math.abs(v) * sc;
    return <rect x={x + j * bw + 3} y={v >= 0 ? zero - bh : zero} width={bw - 6} height={Math.max(bh, 0.5)} rx={3} {...extra} />;
  };
  return (
    <g>
      {title && lbl(title, x, y - 10)}
      <line x1={x} x2={x + w} y1={zero} y2={zero} stroke={T.rule} />
      {vals.map((v, j) => (
        <g key={j}>
          {ghost && rect(ghost[j], j, { fill: "none", stroke: T.dim, strokeDasharray: "3 3", strokeWidth: 1.2 })}
          {rect(v, j, { fill: Array.isArray(cols) ? cols[j] : cols, fillOpacity: 0.9 })}
          {labelsOn && <text x={x + j * bw + bw / 2} y={v >= 0 ? zero - v * sc - 5 : zero - v * sc + 12} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.soft}>{fmt(v)}</text>}
          <text x={x + j * bw + bw / 2} y={y + h + 14} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={hl === j ? T.ink : XC[j]}>E{j}</text>
          {hl === j && <rect x={x + j * bw + 1} y={y - 4} width={bw - 2} height={h + 22} rx={5} fill="none" stroke={T.ink} strokeWidth={1.6} />}
          {rings.includes(j) && <rect x={x + j * bw} y={y - 6} width={bw} height={h + 26} rx={6} fill="none" stroke={T.ink} strokeWidth={2} strokeDasharray="5 3" />}
        </g>
      ))}
    </g>
  );
}

// a 2-segment weight track: segment widths in px, coloured by expert
function Split({ x, y, w0 = 300, h = 26, widths, idx, labels }) {
  let cx = x;
  return (
    <g>
      <rect x={x - 1} y={y - 1} width={w0 + 2} height={h + 2} rx={4} fill="none" stroke={T.rule} />
      {widths.map((wd, i) => {
        const x0 = cx; cx += wd;
        return (
          <g key={i}>
            <rect x={x0} y={y} width={Math.max(wd, 0)} height={h} rx={3} fill={XC[idx[i]]} fillOpacity={0.9} stroke={T.bg} />
            {wd > 34 && <text x={x0 + wd / 2} y={y + h / 2 + 4} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.bg}>{labels[i]}</text>}
          </g>
        );
      })}
    </g>
  );
}

// small line chart over the training steps, drawn up to step `upTo`
function Panel({ x, y, w, h, pts, upTo, color, lo, hi, title, fmt = f2, ref1 }) {
  const px = (i) => x + (i / (TSTEPS - 1)) * w, py = (v) => y + h - ((clamp01((v - lo) / (hi - lo))) * h);
  const d = pts.slice(0, upTo + 1).map((v, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
  return (
    <g>
      {lbl(title, x, y - 10, { fill: color, fontWeight: 700 })}
      <rect x={x} y={y} width={w} height={h} fill={T.well} stroke={T.rule} rx={4} />
      {ref1 != null && <line x1={x} x2={x + w} y1={py(ref1)} y2={py(ref1)} stroke={T.dim} strokeDasharray="4 4" />}
      <text x={x - 4} y={y + 8} textAnchor="end" fontFamily={mono} fontSize="9" fill={T.dim}>{fmt(hi)}</text>
      <text x={x - 4} y={y + h} textAnchor="end" fontFamily={mono} fontSize="9" fill={T.dim}>{fmt(lo)}</text>
      <path d={d} fill="none" stroke={color} strokeWidth={2.2} />
      <circle cx={px(upTo)} cy={py(pts[upTo])} r={4.5} fill={color} />
      <text x={x + w} y={y + h + 16} textAnchor="end" fontFamily={mono} fontSize="12" fontWeight="700" fill={color}>{fmt(pts[upTo])}</text>
      <text x={x} y={y + h + 16} fontFamily={mono} fontSize="9.5" fill={T.dim}>step 0 → 59</text>
    </g>
  );
}

// ---- scene -------------------------------------------------------------------
function Scene({ step, J }) {
  const p = useProgress(step.id);
  const e = ease(p);
  const col = XC[J];
  let body = null;

  if (step.scene === "score") {
    const kind = step.kind;
    const s = kind ? SCORE[kind](ZF) : ZF;
    const hiS = Math.max(...s) * 1.25, hiZ = Math.max(...ZF) * 1.3;
    const loZ = Math.min(...ZF) - 0.5;
    const vals = ZF.map((z, j) => lerp(z, s[j], kind ? e : 0));
    let eq = null;
    const zJ = ZF[J];
    if (kind === "soft") {
      const ex = ZF.map(Math.exp), S = sum(ex);
      eq = [
        `\\mathrm{softmax}(z)_j=\\dfrac{e^{z_j}}{\\sum_{i} e^{z_i}}`,
        `\\mathrm{softmax}(z)_{${J}}=\\dfrac{e^{${cc(col, f2(zJ))}}}{${ex.map((v, j) => (j === J ? cc(col, f2(v)) : f2(v))).join("+")}}`,
        `=\\dfrac{${cc(col, f2(ex[J]))}}{${f2(S)}}=${cc(col, f2(ex[J] / S))}`,
      ];
    } else if (kind === "sig") {
      eq = [
        `\\sigma(z)=\\dfrac{1}{1+e^{-z}}`,
        `\\sigma(${cc(col, f2(zJ))})=\\dfrac{1}{1+e^{-(${cc(col, f2(zJ))})}}=\\dfrac{1}{1+${f2(Math.exp(-zJ))}}`,
        `=${cc(col, f2(sigmoid(zJ)))}`,
      ];
    } else if (kind === "sp") {
      eq = [
        `\\sqrt{\\mathrm{softplus}(z)}=\\sqrt{\\ln\\!\\left(1+e^{z}\\right)}`,
        `\\sqrt{\\ln\\!\\left(1+e^{${cc(col, f2(zJ))}}\\right)}=\\sqrt{\\ln(1+${f2(Math.exp(zJ))})}=\\sqrt{${f2(softplus(zJ))}}`,
        `=${cc(col, f2(sqrtSoftplus(zJ)))}`,
      ];
    }
    const note = { soft: "bounded by 1 and shared: one expert up = others down", sig: "each score on its own, nothing is shared", sp: "no cap: a bigger logit keeps giving a bigger score" }[kind];
    body = (
      <g>
        <VB x={40} y={70} w={380} h={250} vals={vals} lo={lerp(loZ, 0, kind ? e : 0)} hi={lerp(hiZ, hiS, kind ? e : 0)} hl={J} ghost={kind ? ZF : null}
          title={kind ? `${GN[kind].toUpperCase()} SCORES  (DASHED = THE LOGITS z)` : "ROUTER LOGITS z"} />
        {kind ? (
          <g>
            {eq.map((t, i) => <Tex key={i} x={450} y={78 + i * 50} w={400} h={44} size={i === 0 ? 14 : 13.5} tex={t} opacity={i === 0 ? 1 : ease(clamp01((p - 0.15 * i) / 0.4))} />)}
            <text x={450} y={252} fontFamily={mono} fontSize="12" fill={T.soft}>Σ scores = <tspan fill={GC[kind]} fontWeight="700">{f2(sum(s))}</tspan></text>
            <text x={450} y={274} fontFamily={mono} fontSize="11.5" fill={T.dim}>{note}</text>
          </g>
        ) : (
          <g>
            <text x={450} y={90} fontFamily={mono} fontSize="12.5" fill={T.soft}>z is the raw output of the router.</text>
            <text x={450} y={114} fontFamily={mono} fontSize="12.5" fill={T.soft}>It can be negative; it is not a weight yet.</text>
            <text x={450} y={150} fontFamily={mono} fontSize="12.5" fill={T.soft}>next: three functions turn z into scores</text>
            <Tex x={450} y={180} w={400} h={34} size={14} tex={`z=[${ZF.map((v, j) => cc(XC[j], f1(v))).join(",\\;")}]`} />
          </g>
        )}
      </g>
    );
  } else if (step.scene === "side") {
    const EQ = {
      soft: `\\dfrac{e^{z_j}}{\\sum_i e^{z_i}}`,
      sig: `\\dfrac{1}{1+e^{-z_j}}`,
      sp: `\\sqrt{\\ln\\!\\left(1+e^{z_j}\\right)}`,
    };
    const NOTE = { soft: "shared: scores sum to 1", sig: "independent, each below 1", sp: "independent, no cap" };
    const idx = topk(ZF);
    body = (
      <g>
        <Tex x={30} y={14} w={800} h={30} size={14} tex={`z=[${ZF.map((v, j) => cc(XC[j], f1(v))).join(",\\;")}]\\quad\\text{same logits into all three}`} />
        {GATES.map((g, k) => {
          const sc = SCORE[g.id](ZF);
          return (
            <g key={g.id}>
              <VB x={30 + k * 280} y={62} w={250} h={170} vals={sc.map((v) => v * e)} hi={Math.max(...sc) * 1.25} rings={e > 0.7 ? idx : []} title={g.name.toUpperCase()} />
              <Tex x={30 + k * 280} y={262} w={250} h={46} size={14} color={g.col} tex={`s_j=${EQ[g.id]}`} opacity={ease(clamp01((p - 0.3) / 0.5))} />
              <text x={30 + k * 280} y={330} fontFamily={mono} fontSize="12" fill={T.soft}>Σ scores = <tspan fill={g.col} fontWeight="700">{f2(sum(sc))}</tspan></text>
              <text x={30 + k * 280} y={350} fontFamily={mono} fontSize="11.5" fill={g.col}>{NOTE[g.id]}</text>
            </g>
          );
        })}
        <text x={30} y={392} fontFamily={mono} fontSize="12.5" fill={T.soft}>ringed = top-2. Every scorer is increasing in z, so the ranking is the same; only the sizes differ.</text>
      </g>
    );
  } else if (step.scene === "topk") {
    body = (
      <g>
        {GATES.map((g, k) => {
          const s = SCORE[g.id](ZF), idx = topk(ZF), order = s.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((a) => "E" + a[1]).join(" > ");
          return (
            <g key={g.id}>
              <VB x={30 + k * 280} y={86} w={250} h={190} vals={s.map((v) => v * e)} hi={Math.max(...s) * 1.25} rings={e > 0.7 ? idx : []} title={g.name.toUpperCase()} />
              <text x={30 + k * 280} y={316} fontFamily={mono} fontSize="11.5" fill={g.col} fontWeight="700">{order}</text>
            </g>
          );
        })}
        <text x={30} y={352} fontFamily={mono} fontSize="12.5" fill={T.soft}>ringed = top-2. Each score is strictly increasing in z, so a bigger z can never lose.</text>
        <text x={30} y={374} fontFamily={mono} fontSize="12.5" fill={T.soft}>The three functions differ in how big the weights are, not in who wins.</text>
      </g>
    );
  } else if (step.scene === "gate") {
    const SC = 110;
    body = (
      <g>
        {lbl(step.mix ? "WEIGHT g[i] SCALES EXPERT i'S OUTPUT" : "RAW SCORE s[idx]  →  g = s / Σ s  (WIDTH = VALUE)", 40, 40)}
        {GATES.map((gt, k) => {
          const { s, idx, g } = gateOf(gt.id, ZF), y = 62 + k * 112;
          const widths = idx.map((i, a) => (step.mix ? 300 * g[a] : lerp(s[i] * SC, g[a] * 300, e)));
          const sI = idx.map((i) => s[i]);
          return (
            <g key={gt.id}>
              <text x={40} y={y - 4} fontFamily={mono} fontSize="12" fontWeight="700" fill={gt.col}>{gt.name}</text>
              <Split x={40} y={y + 4} widths={widths} idx={idx} labels={idx.map((i, a) => `E${i} ${f2(lerp(sI[a], g[a], step.mix ? 1 : e))}`)} />
              <Tex x={380} y={y - 2} w={470} h={44} size={13.5}
                tex={`g_{${idx[0]}}=\\dfrac{${cc(XC[idx[0]], f2(sI[0]))}}{${f2(sI[0])}+${f2(sI[1])}}=${cc(XC[idx[0]], f2(g[0]))}\\quad g_{${idx[1]}}=\\dfrac{${cc(XC[idx[1]], f2(sI[1]))}}{${f2(sum(sI))}}=${cc(XC[idx[1]], f2(g[1]))}`}
                opacity={step.mix ? 1 : ease(clamp01((p - 0.4) / 0.5))} />
            </g>
          );
        })}
        <text x={40} y={400} fontFamily={mono} fontSize="12" fill={T.soft}>logits [2, 1]: softmax {gateOf("soft", ZF).g.map(f2).join("/")} · sigmoid {gateOf("sig", ZF).g.map(f2).join("/")} · sqrt-softplus {gateOf("sp", ZF).g.map(f2).join("/")}</text>
      </g>
    );
  } else if (step.scene === "conf") {
    const c = lerp(1, CMAX, e), zc = ZF.map((v) => v * c);
    const x0 = 470, pw = 340, y0 = 90, ph = 220, lo = 0.4, hi = 1;
    const px = (cv) => x0 + ((cv - 1) / (CMAX - 1)) * pw, py = (v) => y0 + ph - ((v - lo) / (hi - lo)) * ph;
    const curve = (kind, upTo) => Array.from({ length: 61 }, (_, i) => 1 + (i / 60) * (CMAX - 1)).filter((cv) => cv <= upTo + 1e-9).map((cv, i) => `${i ? "L" : "M"}${px(cv).toFixed(1)},${py(w1At(kind, cv)).toFixed(1)}`).join(" ");
    body = (
      <g>
        <Tex x={40} y={30} w={400} h={30} size={14} tex={`c=${cc(T.wire, f2(c))}\\qquad cz=[${zc.slice(0, 2).map((v, j) => cc(XC[j], f1(v))).join(",\\;")},\\dots]`} />
        {GATES.map((gt, k) => {
          const { idx, g } = gateOf(gt.id, zc), y = 86 + k * 90;
          return (
            <g key={gt.id}>
              <text x={40} y={y - 2} fontFamily={mono} fontSize="12" fontWeight="700" fill={gt.col}>{gt.name}</text>
              <Split x={40} y={y + 6} widths={g.map((v) => v * 300)} idx={idx} labels={g.map((v, a) => `E${idx[a]} ${f2(v)}`)} />
            </g>
          );
        })}
        {lbl("TOP EXPERT'S WEIGHT g[0] vs c", x0, y0 - 14)}
        <rect x={x0} y={y0} width={pw} height={ph} fill={T.well} stroke={T.rule} rx={4} />
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((v) => <g key={v}><line x1={x0} x2={x0 + pw} y1={py(v)} y2={py(v)} stroke={T.rule} strokeOpacity={0.4} /><text x={x0 - 4} y={py(v) + 3} textAnchor="end" fontFamily={mono} fontSize="9" fill={T.dim}>{v}</text></g>)}
        {[1, 2, 3, 4, 5, 6].map((v) => <text key={v} x={px(v)} y={y0 + ph + 14} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>{v}</text>)}
        <text x={x0 + pw / 2} y={y0 + ph + 30} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.dim}>c</text>
        {GATES.map((gt) => <g key={gt.id}>
          <path d={curve(gt.id, CMAX)} fill="none" stroke={gt.col} strokeOpacity={0.22} strokeWidth={1.5} />
          <path d={curve(gt.id, c)} fill="none" stroke={gt.col} strokeWidth={2.4} />
          <circle cx={px(c)} cy={py(w1At(gt.id, c))} r={5} fill={gt.col} />
          <text x={x0 + pw + 4} y={py(w1At(gt.id, CMAX)) + 4} fontFamily={mono} fontSize="10" fontWeight="700" fill={gt.col}>{f2(w1At(gt.id, CMAX))}</text>
        </g>)}
        <text x={40} y={382} fontFamily={mono} fontSize="12" fill={T.soft}>softmax → top-1 (the other is squeezed out) · sigmoid → equal average · sqrt-softplus → in between</text>
      </g>
    );
  } else if (step.scene === "ratio") {
    const c = lerp(1, CMAX, e), d = ZF[0] - ZF[1], zc = ZF.map((v) => v * c);
    const rat = (k) => { const g = gateOf(k, zc).g; return g[0] / g[1]; };
    const rows = [
      { k: "soft", tex: `\\dfrac{g_0}{g_1}=e^{c(z_0-z_1)}=e^{${cc(T.wire, f2(c))}\\cdot${d}}=${cc(T.bad, f2(rat("soft")))}\\;\\to\\;\\infty` },
      { k: "sig", tex: `\\dfrac{g_0}{g_1}=\\dfrac{\\sigma(c z_0)}{\\sigma(c z_1)}=\\dfrac{${f3(sigmoid(zc[0]))}}{${f3(sigmoid(zc[1]))}}=${cc(T.accent, f2(rat("sig")))}\\;\\to\\;1` },
      { k: "sp", tex: `\\dfrac{g_0}{g_1}\\approx\\sqrt{\\dfrac{z_0}{z_1}}=\\sqrt{\\dfrac{${ZF[0]}}{${ZF[1]}}}=${f2(Math.sqrt(ZF[0] / ZF[1]))}\\quad\\text{now }${cc(T.wire, f2(rat("sp")))}` },
    ];
    body = (
      <g>
        <Tex x={40} y={26} w={300} h={28} size={14} tex={`c=${cc(T.wire, f2(c))}`} />
        {rows.map((r, k) => {
          const { idx, g } = gateOf(r.k, zc), y = 70 + k * 108;
          return (
            <g key={r.k}>
              <text x={40} y={y} fontFamily={mono} fontSize="12" fontWeight="700" fill={GC[r.k]}>{GN[r.k]}</text>
              <Split x={40} y={y + 8} w0={220} widths={g.map((v) => v * 220)} idx={idx} labels={g.map((v, a) => `${f2(v)}`)} />
              <Tex x={290} y={y - 6} w={560} h={50} size={14} tex={r.tex} />
            </g>
          );
        })}
        <text x={40} y={406} fontFamily={mono} fontSize="11.5" fill={T.dim}>softplus(z) ≈ z for large z, so sqrt-softplus ≈ √z: a fixed ratio, no saturation</text>
      </g>
    );
  } else if (step.scene === "monitor") {
    const c = lerp(1, CMAX, e);
    const x0 = 470, pw = 340, y0 = 90, ph = 200, hi = Math.log(2) * 1.05;
    const px = (cv) => x0 + ((cv - 1) / (CMAX - 1)) * pw, py = (v) => y0 + ph - (v / hi) * ph;
    const Hc = (k, cv) => ent(gateOf(k, ZF.map((v) => v * cv)).g);
    const curve = (kind, upTo) => Array.from({ length: 61 }, (_, i) => 1 + (i / 60) * (CMAX - 1)).filter((cv) => cv <= upTo + 1e-9).map((cv, i) => `${i ? "L" : "M"}${px(cv).toFixed(1)},${py(Hc(kind, cv)).toFixed(1)}`).join(" ");
    body = (
      <g>
        <Tex x={40} y={26} w={400} h={28} size={14} tex={`H=-\sum_i g_i\ln g_i,\quad c=${cc(T.wire, f2(c))}`} />
        {GATES.map((gt, k) => {
          const g = gateOf(gt.id, ZF.map((v) => v * c)).g, y = 80 + k * 96;
          return (
            <g key={gt.id}>
              <text x={40} y={y} fontFamily={mono} fontSize="12" fontWeight="700" fill={gt.col}>{gt.name}</text>
              <text x={40} y={y + 22} fontFamily={mono} fontSize="11.5" fill={T.soft}>top weight <tspan fill={gt.col} fontWeight="700">{f2(g[0])}</tspan></text>
              <text x={40} y={y + 42} fontFamily={mono} fontSize="11.5" fill={T.soft}>entropy H <tspan fill={gt.col} fontWeight="700">{f3(ent(g))}</tspan></text>
              <rect x={230} y={y + 28} width={160} height={10} rx={3} fill={T.well} stroke={T.rule} />
              <rect x={230} y={y + 28} width={160 * ent(g) / Math.log(2)} height={10} rx={3} fill={gt.col} />
            </g>
          );
        })}
        {lbl("ENTROPY OF THE 2 CHOSEN WEIGHTS vs c", x0, y0 - 14)}
        <rect x={x0} y={y0} width={pw} height={ph} fill={T.well} stroke={T.rule} rx={4} />
        <line x1={x0} x2={x0 + pw} y1={py(Math.log(2))} y2={py(Math.log(2))} stroke={T.dim} strokeDasharray="4 4" />
        <text x={x0 + 4} y={py(Math.log(2)) - 4} fontFamily={mono} fontSize="9.5" fill={T.dim}>ln 2 = {f3(Math.log(2))}</text>
        {[1, 2, 3, 4, 5, 6].map((v) => <text key={v} x={px(v)} y={y0 + ph + 14} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>{v}</text>)}
        {GATES.map((gt) => <g key={gt.id}>
          <path d={curve(gt.id, c)} fill="none" stroke={gt.col} strokeWidth={2.4} />
          <circle cx={px(c)} cy={py(Hc(gt.id, c))} r={5} fill={gt.col} />
        </g>)}
        <text x={40} y={382} fontFamily={mono} fontSize="12" fill={T.soft}>low entropy = one expert takes most of the weight. softmax falls to 0; sigmoid stays at ln 2.</text>
      </g>
    );
  } else if (step.scene === "top1") {
    const zs = lerp(-3, 6, e), zl = [-3, 0, 3, 6];
    const one = (z) => softmax([z])[0];
    const dsoft = (z) => (softmax([z + 1e-5])[0] - softmax([z - 1e-5])[0]) / 2e-5;
    const dsig = (z) => (sigmoid(z + 1e-5) - sigmoid(z - 1e-5)) / 2e-5;
    body = (
      <g>
        <Tex x={40} y={30} w={500} h={30} size={14} tex={`z=${cc(T.wire, f1(zs))}\qquad k=1:\ \mathrm{softmax}([z])=\dfrac{e^{z}}{e^{z}}=${cc(T.bad, f1(one(zs)))}`} />
        {lbl("z", 50, 92)}{lbl("softmax([z]), k=1", 150, 92)}{lbl("d softmax / dz", 330, 92)}{lbl("sigmoid(z), for comparison", 520, 92)}
        {zl.map((z, i) => (
          <g key={i} opacity={ease(clamp01((e - i * 0.2) / 0.3))}>
            <text x={50} y={128 + i * 52} fontFamily={mono} fontSize="14" fill={T.wire} fontWeight="700">{f1(z)}</text>
            <rect x={150} y={108 + i * 52} width={150 * one(z)} height={26} rx={3} fill={T.bad} fillOpacity={0.9} />
            <text x={156} y={126 + i * 52} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.bg}>[{f1(one(z))}]</text>
            <text x={330} y={128 + i * 52} fontFamily={mono} fontSize="14" fill={T.bad} fontWeight="700">{f1(Math.abs(dsoft(z)))}</text>
            <rect x={520} y={108 + i * 52} width={200 * sigmoid(z)} height={26} rx={3} fill={T.accent} fillOpacity={0.9} />
            <text x={526} y={126 + i * 52} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.bg}>{f3(sigmoid(z))}  d={f3(dsig(z))}</text>
          </g>
        ))}
        <text x={40} y={350} fontFamily={mono} fontSize="12.5" fill={T.soft}>k=1 softmax = [1.0] for any z: the output cannot depend on z, so the router gets zero gradient.</text>
        <text x={40} y={372} fontFamily={mono} fontSize="12.5" fill={T.soft}>the sharpest form of collapse: one expert takes everything and the router cannot learn.</text>
      </g>
    );
  } else if (step.scene === "couple") {
    const zNew = ZF.map((v, j) => (j === 3 ? lerp(v, COUPLE_Z, e) : v));
    const kinds = ["soft", "sig", "sp"];
    body = (
      <g>
        <Tex x={40} y={22} w={600} h={30} size={14} tex={`z_3:\ ${cc(XC[3], f2(ZF[3]))}\to ${cc(XC[3], f2(zNew[3]))}\quad (z_0,z_1,z_2\text{ unchanged})`} />
        {kinds.map((k, i) => {
          const s0 = SCORE[k](ZF), s1 = SCORE[k](zNew), x = 30 + i * 280, hiS = Math.max(...SCORE[k](ZF.map((v, j) => (j === 3 ? COUPLE_Z : v)))) * 1.2;
          const moved = Math.max(...[0, 1, 2].map((j) => Math.abs(s1[j] - s0[j])));
          return (
            <g key={k}>
              <VB x={x} y={86} w={250} h={190} vals={s1} ghost={s0} hi={hiS} title={GN[k].toUpperCase()} />
              <text x={x} y={312} fontFamily={mono} fontSize="11.5" fill={GC[k]} fontWeight="700">E0 score {f3(s0[0])} → {f3(s1[0])}</text>
              <text x={x} y={332} fontFamily={mono} fontSize="11.5" fill={GC[k]}>E1 score {f3(s0[1])} → {f3(s1[1])}</text>
              <text x={x} y={352} fontFamily={mono} fontSize="11" fill={moved > 1e-9 ? T.bad : T.accent}>{moved > 1e-9 ? `E0-E2 moved by up to ${f3(moved)}` : "E0-E2 did not move"}</text>
            </g>
          );
        })}
        <text x={30} y={380} fontFamily={mono} fontSize="12" fill={T.soft}>softmax: the denominator holds every expert, so one expert growing pushes the rest down.</text>
        <text x={30} y={400} fontFamily={mono} fontSize="12" fill={T.soft}>sigmoid and sqrt-softplus: each expert looks only at its own logit, so the others stay put.</text>
        <text x={30} y={420} fontFamily={mono} fontSize="11" fill={T.dim}>dashed = before the change to E3</text>
      </g>
    );
  } else if (step.scene === "advice") {
    const q = (a) => ease(clamp01((p - a) / 0.4));
    const row = (y, a, col, t1, t2) => (
      <g opacity={q(a)} transform={`translate(0,${(1 - q(a)) * 12})`}>
        <rect x={30} y={y} width={800} height={86} rx={12} fill={T.well} stroke={col} strokeWidth={1.6} />
        <text x={50} y={y + 34} fontFamily={mono} fontSize="15" fontWeight="700" fill={col}>{t1}</text>
        <text x={50} y={y + 62} fontFamily={mono} fontSize="12.5" fill={T.soft}>{t2}</text>
      </g>
    );
    body = (
      <g>
        {row(40, 0, T.accent, "balancing with an expert bias → sigmoid or sqrt-softplus", "scores are independent, so a bias moves one expert without disturbing the rest (DeepSeek-V3 uses sigmoid)")}
        {row(150, 0.25, T.bad, "otherwise → softmax is the common default", "competition keeps weights summing to 1; watch the top weight and the entropy so it does not collapse to top-1")}
        {row(260, 0.5, T.wire, "always monitor", `top expert's weight and routing entropy per layer (here: n_eff ${f2(SIM.soft.nEff[TSTEPS - 1])} / ${f2(SIM.sig.nEff[TSTEPS - 1])} / ${f2(SIM.sp.nEff[TSTEPS - 1])})`)}
      </g>
    );
  } else if (step.scene === "sim") {
    const st = step.stage;
    const t = st === 0 ? Math.round(e * (TSTEPS - 1)) : st === 4 ? TSTEPS - 1 : Math.round(e * (TSTEPS - 1));
    const c = cOf(t);
    if (st === 0) {
      const zc = Z0[0].map((v) => v * c);
      body = (
        <g>
          <VB x={40} y={80} w={360} h={230} vals={zc} lo={Math.min(...ZF) * CMAX - 1} hi={Math.max(...ZF) * CMAX * 1.1} hl={0} fmt={f1} title="TOKEN 0: LOGITS  z = c(t) · z0" ghost={Z0[0]} />
          <Tex x={440} y={70} w={400} h={34} size={14} tex={`z_t=c(t)\\,z_0,\\qquad c(t)=1+5\\,\\dfrac{t}{59}`} />
          <Tex x={440} y={118} w={400} h={30} size={14} tex={`t=${cc(T.wire, String(t))}\\qquad c=${cc(T.wire, f2(c))}`} />
          <text x={440} y={190} fontFamily={mono} fontSize="12" fill={T.soft}>8 tokens × 4 experts, same ranking every step.</text>
          <text x={440} y={212} fontFamily={mono} fontSize="12" fill={T.soft}>Only the confidence grows: a confident router</text>
          <text x={440} y={234} fontFamily={mono} fontSize="12" fill={T.soft}>is rewarded for being confident (rich get richer).</text>
          <text x={440} y={266} fontFamily={mono} fontSize="12" fill={T.soft}>question: what happens to the 2nd chosen expert?</text>
          <text x={40} y={370} fontFamily={mono} fontSize="10.5" fill={T.dim}>toy: the growth of c is imposed, not learned; it stands in for training sharpening the router</text>
        </g>
      );
    } else {
      const metric = { 1: "w2", 2: "nEff", 3: "grad", 4: "nEff" }[st];
      const rng2 = { w2: [0, 0.5], nEff: [1, 2], grad: [0, GMAX * 1.05] }[metric];
      const ttl = { w2: "2nd EXPERT'S WEIGHT", nEff: "EXPERTS ACTUALLY USED", grad: "2nd EXPERT'S GRADIENT" }[metric];
      const msg = {
        1: ["softmax: 2nd expert's weight falls to ~0", "sigmoid: stays at 1/2 (equal)", "sqrt-softplus: settles near 0.4"],
        2: ["softmax: ~1 expert (collapse)", "sigmoid: both used, equally", "sqrt-softplus: both used, fixed ratio"],
        3: ["softmax: gradient dies, expert stops learning", "sigmoid: gradient also saturates, weights stay equal", "sqrt-softplus: shrinks slowly, never dies"],
        4: ["softmax: effectively ONE expert", "sigmoid: BOTH, equal average", "sqrt-softplus: BOTH, fixed ratio"],
      }[st];
      body = (
        <g>
          {GATES.map((gt, k) => {
            const x = 52 + k * 280, sim = SIM[gt.id];
            const { idx, g } = gateOf(gt.id, ZF.map((v) => v * c));
            return (
              <g key={gt.id}>
                <Panel x={x} y={70} w={226} h={170} pts={sim[metric]} upTo={t} color={gt.col} lo={rng2[0]} hi={rng2[1]} title={`${gt.name.toUpperCase()}`} fmt={metric === "grad" ? f3 : f2} ref1={metric === "nEff" ? 2 : metric === "w2" ? 0.5 : null} />
                <text x={x} y={62 - 0} fontFamily={mono} fontSize="0" fill="none">{ttl}</text>
                <Split x={x} y={284} w0={226} widths={g.map((v) => v * 226)} idx={idx} labels={g.map((v, a) => f2(v))} />
                <text x={x} y={276} fontFamily={mono} fontSize="9.5" fill={T.dim}>token 0 weights now</text>
                <text x={x} y={338} fontFamily={mono} fontSize="10.5" fill={gt.col}>{msg[k]}</text>
              </g>
            );
          })}
          {lbl(`${ttl}  ·  mean over ${NT} tokens  ·  t = ${t}, c = ${f2(c)}`, 52, 38)}
          {st === 4 && (
            <text x={52} y={386} fontFamily={mono} fontSize="12.5" fill={T.soft}>
              n_eff after 60 steps: <tspan fill={T.bad} fontWeight="700">{f2(SIM.soft.nEff[TSTEPS - 1])}</tspan> · <tspan fill={T.accent} fontWeight="700">{f2(SIM.sig.nEff[TSTEPS - 1])}</tspan> · <tspan fill={T.wire} fontWeight="700">{f2(SIM.sp.nEff[TSTEPS - 1])}</tspan> (of a maximum 2)
            </text>
          )}
          {st === 3 && <text x={52} y={386} fontFamily={mono} fontSize="12" fill={T.soft}>d g[1] / d z (mean over tokens): no gradient = this expert's logit cannot move any more</text>}
        </g>
      );
    }
  } else if (step.scene === "bias") {
    const b = BIAS * e;
    body = (
      <g>
        <text x={40} y={34} fontFamily={mono} fontSize="12.5" fill={T.soft}>two close experts, logits z = [<tspan fill={XC[0]} fontWeight="700">{BZ[0]}</tspan>, <tspan fill={XC[1]} fontWeight="700">{BZ[1]}</tspan>] · bias b = <tspan fill={T.wire} fontWeight="700">{f2(b)}</tspan> added to E1 for selection</text>
        {[GATES[1], GATES[2]].map((gt, k) => {
          const fn = gt.id === "sig" ? sigmoid : sqrtSoftplus;
          const g0 = fn(BZ[0]), g1 = fn(BZ[1]), gap = g0 - g1, flips = BIAS > gap, nowFlip = g1 + b > g0;
          const x = 40 + k * 420, sc = 190 / 1.5, y0 = 300;
          const bar = (v, xx, c0) => <rect x={xx} y={y0 - v * sc} width={64} height={v * sc} rx={3} fill={c0} fillOpacity={0.9} />;
          return (
            <g key={gt.id}>
              <text x={x} y={70} fontFamily={mono} fontSize="13" fontWeight="700" fill={gt.col}>{gt.name}</text>
              <line x1={x} x2={x + 380} y1={y0} y2={y0} stroke={T.rule} />
              {bar(g0, x + 20, XC[0])}
              {bar(g1, x + 130, XC[1])}
              <rect x={x + 130} y={y0 - (g1 + b) * sc} width={64} height={b * sc} rx={3} fill="none" stroke={T.wire} strokeWidth={1.8} strokeDasharray="4 3" />
              <line x1={x + 10} x2={x + 210} y1={y0 - g0 * sc} y2={y0 - g0 * sc} stroke={T.dim} strokeDasharray="3 3" />
              <text x={x + 52} y={y0 - g0 * sc - 6} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.soft}>{f2(g0)}</text>
              <text x={x + 162} y={y0 - (g1 + b) * sc - 6} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.soft}>{f2(g1 + b)}</text>
              <text x={x + 52} y={y0 + 16} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[0]}>E0</text>
              <text x={x + 162} y={y0 + 16} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[1]}>E1 + b</text>
              <Tex x={x + 230} y={96} w={190} h={34} size={13} tex={`\\text{gap}=${f2(g0)}-${f2(g1)}=${cc(gt.col, f2(gap))}`} />
              <Tex x={x + 230} y={136} w={190} h={30} size={13} tex={`b=${cc(T.wire, f2(BIAS))}\\;${flips ? ">" : "<"}\\;${f2(gap)}`} />
              <text x={x + 230} y={196} fontFamily={mono} fontSize="13" fontWeight="700" fill={flips ? T.bad : T.accent} opacity={e > 0.85 ? 1 : 0.15}>{flips ? "RANK FLIPS" : "rank holds"}</text>
              <text x={x + 230} y={216} fontFamily={mono} fontSize="11" fill={T.dim} opacity={e > 0.85 ? 1 : 0}>{nowFlip ? "E1 now beats E0" : "E0 still wins"}</text>
            </g>
          );
        })}
        <text x={40} y={372} fontFamily={mono} fontSize="12" fill={T.soft}>a bias flips a pair only if b exceeds the score gap; sigmoid gaps are small (always &lt; 1, shrinking with saturation)</text>
        <text x={40} y={392} fontFamily={mono} fontSize="12" fill={T.soft}>so the same b changes the top-k under one gate and not the other. (selection only: the weights still use g(z))</text>
      </g>
    );
  } else if (step.scene === "group") {
    const stage = step.stage;
    const rows = [
      { k: "soft", name: "softmax  (∝ e^z)", f: Math.exp },
      { k: "sig", name: "sigmoid", f: sigmoid },
      { k: "sp", name: "sqrt-softplus", f: sqrtSoftplus },
    ];
    const gsum = (f, gi) => sum(GROUPS[gi].map(f));
    const grow = stage === 0 ? e : 1;
    body = (
      <g>
        {rows.map((r, k) => {
          const y = 40 + k * 118, sums = [gsum(r.f, 0), gsum(r.f, 1)], win = sums[0] >= sums[1] ? 0 : 1;
          const sc = 270 / Math.max(...sums);
          return (
            <g key={r.k}>
              <text x={30} y={y} fontFamily={mono} fontSize="12" fontWeight="700" fill={GC[r.k]}>{r.name}</text>
              {[0, 1].map((gi) => {
                let cx = 90;
                return (
                  <g key={gi}>
                    <text x={30} y={y + 28 + gi * 34} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.soft}>{gi ? "B" : "A"}</text>
                    {GROUPS[gi].map((z, a) => {
                      const v = r.f(z) * sc * grow, x0 = cx; cx += v;
                      return <g key={a}><rect x={x0} y={y + 14 + gi * 34} width={v} height={24} rx={3} fill={a ? XC[1] : XC[0]} fillOpacity={0.85} stroke={T.bg} />
                        {v > 40 && <text x={x0 + v / 2} y={y + 30 + gi * 34} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={T.bg}>{f2(r.f(z))}</text>}</g>;
                    })}
                    <text x={cx + 8} y={y + 31 + gi * 34} fontFamily={mono} fontSize="12" fontWeight="700" fill={stage === 1 && win === gi ? GC[r.k] : T.soft}>Σ {f2(sums[gi] * grow)}{stage === 1 && win === gi ? "  ← wins" : ""}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <Tex x={510} y={52} w={340} h={30} size={13} tex={`A=\\{${GROUPS[0].join(",")}\\}\\quad B=\\{${GROUPS[1].join(",")}\\}`} />
        <Tex x={510} y={96} w={340} h={34} size={13} tex={`e^{${GROUPS[0][0]}}+e^{${GROUPS[0][1]}}=${cc(T.bad, f1(gsum(Math.exp, 0)))}\\;>\\;2e^{2}=${f1(gsum(Math.exp, 1))}`} opacity={stage === 1 ? 1 : 0.35} />
        <Tex x={510} y={146} w={340} h={34} size={13} tex={`\\sigma(3)+\\sigma(0)=${f2(gsum(sigmoid, 0))}\\;<\\;2\\sigma(2)=${cc(T.accent, f2(gsum(sigmoid, 1)))}`} opacity={stage === 1 ? 1 : 0.35} />
        <text x={510} y={226} fontFamily={mono} fontSize="11.5" fill={T.soft}>softmax rewards ONE strong expert in a group</text>
        <text x={510} y={248} fontFamily={mono} fontSize="11.5" fill={T.soft}>sigmoid rewards SEVERAL good experts</text>
        <text x={510} y={282} fontFamily={mono} fontSize="11.5" fill={T.dim}>a monotone transform keeps expert order,</text>
        <text x={510} y={300} fontFamily={mono} fontSize="11.5" fill={T.dim}>but not the order of sums</text>
      </g>
    );
  } else {
    const card = (x, a, c, t1, t2, t3, t4) => {
      const q = ease(clamp01((p - a) / 0.5));
      return (
        <g opacity={q} transform={`translate(0,${(1 - q) * 14})`}>
          <rect x={x} y={50} width={270} height={290} rx={12} fill={T.well} stroke={c} strokeWidth={1.6} />
          <text x={x + 16} y={86} fontFamily={mono} fontSize="15" fontWeight="700" fill={c}>{t1}</text>
          <text x={x + 16} y={120} fontFamily={mono} fontSize="12" fill={T.ink}>{t2}</text>
          {t3.map((l, i) => <text key={i} x={x + 16} y={152 + i * 24} fontFamily={mono} fontSize="11.5" fill={T.soft}>{l}</text>)}
          <text x={x + 16} y={318} fontFamily={mono} fontSize="12" fontWeight="700" fill={c}>{t4}</text>
        </g>
      );
    };
    const n = (k) => f2(SIM[k].nEff[TSTEPS - 1]);
    body = (
      <g>
        {card(10, 0, T.bad, "softmax", "competition, sums to 1", ["pushes toward top-1", "2nd expert's weight → 0", "gradient to it dies", "hard to bias"], `n_eff ${n("soft")}`)}
        {card(295, 0.2, T.accent, "sigmoid", "independent scores", ["saturates to equal average", "both experts keep learning", "easy to steer with a bias", "DeepSeek-V3 uses this"], `n_eff ${n("sig")}`)}
        {card(580, 0.4, T.wire, "sqrt-softplus", "in between", ["no saturation", "fixed ratio ≈ √(z₁/z₂)", "differences compressed", "bias acts less"], `n_eff ${n("sp")}`)}
        <text x={10} y={378} fontFamily={mono} fontSize="12" fill={T.soft} opacity={ease(clamp01((p - 0.7) / 0.3))}>same top-k, different weights: the choice of gate is a choice of how experts share the work</text>
      </g>
    );
  }
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

export default function GatingFunctionsView() {
  const [J, setJ] = useState(0);
  return (
    <Stepper
      title="Gating functions"
      subtitle={`softmax vs sigmoid · toy: N=${N} experts, top-${K}`}
      blocks={BLOCKS}
      steps={STEPS}
      scene={(step) => <Scene step={step} J={J} />}
      header={(step) => step.scene === "score" && (
        <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow expert</span>
          {Array.from({ length: N }, (_, j) => (
            <button key={j} onClick={() => setJ(j)} style={{ ...btn(j === J, false), height: 28, padding: "0 9px", color: j === J ? undefined : XC[j] }}>E{j}</button>
          ))}
        </div>
      )}
      legend={<span><span style={{ color: T.bad }}>softmax</span> · <span style={{ color: T.accent }}>sigmoid</span> · bar colours = experts · bold coloured numbers in equations = the followed expert</span>}
    />
  );
}
