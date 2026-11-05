import { useState } from "react";
import { T, mono, btn } from "./theme";
import { XC } from "./lbModel";
import { div } from "./charts";
import Stepper from "./Stepper";
import { Tex } from "./tex";
import { useProgress, ease, lerp, clamp01 } from "./anim";

// Router z-loss: L_z = (1/B) Σ_i logsumexp_j(x_ij)^2, on the raw router logits
// (before softmax / sigmoid). Every number on screen comes from the functions below.

const B = 8, N = 8;
const COEF = 1e-3; // loss weight used in the code
const ETA = 2; // learning rate of the toy gradient step
const TRAIN_STEPS = 30;

// ---- real implementation (seeded logits) -----------------------------------
const rng = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const lse = (row) => { const m = Math.max(...row); return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0)); };
const softmax = (row) => { const z = lse(row); return row.map((v) => Math.exp(v - z)); };
const sum = (a) => a.reduce((s, v) => s + v, 0);
const zOf = (X) => X.map(lse);
const lossZ = (X) => sum(zOf(X).map((z) => z * z)) / X.length;
const grad = (X) => X.map((row) => { const c = (2 * lse(row)) / X.length, p = softmax(row); return p.map((v) => c * v); }); // dL_z/dx
const sgd = (X) => { const G = grad(X); return X.map((row, i) => row.map((v, j) => v - ETA * G[i][j])); };
const meanMaxP = (X) => sum(X.map((r) => Math.max(...softmax(r)))) / X.length;
const maxAbs = (X) => Math.max(...X.flat().map(Math.abs));
const bf16Step = (v) => Math.pow(2, Math.floor(Math.log2(Math.abs(v))) - 7); // spacing of bf16 numbers near v

const r = rng(7);
const BOOST = [1.5, 2, 6, 1, 9, 3, 7.5, 0.5]; // a few tokens whose router is very confident
const FAV = [3, 0, 5, 2, 6, 1, 4, 7];
const X0 = Array.from({ length: B }, (_, i) => Array.from({ length: N }, (_, j) => gauss(r) * 1.2 + (j === FAV[i] ? BOOST[i] : 0)));
const G0 = grad(X0);
const X1 = sgd(X0); // one step
let XN = X0; for (let s = 0; s < TRAIN_STEPS; s++) XN = sgd(XN); // many steps
const Z0 = zOf(X0), Z2 = Z0.map((z) => z * z), LZ0 = lossZ(X0), LZ1 = lossZ(X1), LZN = lossZ(XN);
const P0 = X0.map(softmax), P1 = X1.map(softmax), PN = XN.map(softmax);
const MAXP0 = meanMaxP(X0), MAXPN = meanMaxP(XN), MAXPN1 = meanMaxP(X1);
const LO = Math.min(0, ...X0.flat()) - 1, HI = Math.max(...X0.flat()) + 1.5;

const W = 860, H = 430;
const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), fN = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
const cc = (col, s) => `{\\color{${col}}\\mathbf{${s}}}`;
const lbl = (s, x, y, o = {}) => <text x={x} y={y} fontFamily={mono} fontSize="10" letterSpacing="0.1em" fill={T.dim} {...o}>{s}</text>;

// ---- code ------------------------------------------------------------------
const BLOCKS = [
  { id: "logits", label: "router logits", sub: "raw scores, before softmax", code: [
    "logits = x @ Wg   # x [B,d] @ Wg [d,N] → [B,N] · logits[i,j] = token i's raw score for expert j (before softmax)"] },
  { id: "lse", label: "logsumexp over experts", sub: "one number per token", code: [
    "z = torch.logsumexp(logits, dim=-1)   # [B,N] → reduce over N (experts) → [B] · z[i] = ln Σ_j e^logits[i,j] ≈ token i's biggest logit"] },
  { id: "loss", label: "square, then average", sub: "big z is punished quadratically", code: [
    "sq = z ** 2   # [B] → [B] · sq[i] = z_i²",
    "z_loss = coef * sq.mean()   # [B] → mean over B (tokens) → scalar · coef = loss weight, e.g. 1e-3"] },
  { id: "total", label: "add to the training loss", sub: "one more term", code: [
    "total = task_loss + aux + z_loss   # scalars · aux = load-balance term"] },
  { id: "back", label: "backward: what the gradient does", sub: "pushes every logit down", code: [
    "z_loss.backward()   # dL/dx[i,j] = (2·z[i]/B) · softmax(x)[i,j] → [B,N] · big where token i is big AND expert j is its favourite",
    "logits -= lr * logits.grad   # [B,N] · every logit of a token falls by lr · (2 z_i / B) · p_ij"] },
  { id: "result", label: "before vs after training", sub: "softmax gets less peaked", code: [
    "# same tokens, after many z-loss steps: logits shrink, softmax flattens, L_z drops"] },
  { id: "note", label: "a different job", sub: "z-loss ≠ load balancing", code: [
    "# z-loss: keeps logits small and numerically safe",
    "# aux loss / bias: spreads tokens over experts"] },
];

const STEPS = [
  { id: "logits", block: "logits", lines: [0], scene: "heat", caption: "router logits: 8 tokens × 8 experts, some tokens shout", vars: () => [["B tokens", `${B}`], ["N experts", `${N}`], ["max logit", f1(Math.max(...X0.flat())), "bad"]] },
  { id: "exp", block: "lse", lines: [0], scene: "exp", stage: 0, caption: "step 1 inside: e^x for every expert, one bar towers", vars: (t) => [["logit → e^logit", "per expert"]] },
  { id: "stack", block: "lse", lines: [0], scene: "exp", stage: 1, caption: "step 2 inside: add them up into one stack S", vars: () => [["S", "Σ_j e^x_j"]] },
  { id: "log", block: "lse", lines: [0], scene: "exp", stage: 2, caption: "step 3 inside: take ln(S): the stack collapses to z", vars: () => [["z", "ln S"], ["gap above max", "≤ ln N = " + f2(Math.log(N))]] },
  { id: "zbars", block: "lse", lines: [0], scene: "zbars", caption: "z_i for every token: roughly its biggest logit", vars: () => [["max z", f1(Math.max(...Z0)), "bad"], ["min z", f1(Math.min(...Z0))]] },
  { id: "sq", block: "loss", lines: [0], scene: "sq", caption: "square it: big tokens blow up, small ones barely count", vars: () => [["max z²", fN(Math.max(...Z2)), "bad"], ["min z²", fN(Math.min(...Z2))]] },
  { id: "mean", block: "loss", lines: [1], scene: "mean", caption: "average over tokens: L_z is the mean height", vars: () => [["L_z", fN(LZ0), "bad"], ["coef · L_z", (COEF * LZ0).toFixed(3)]] },
  { id: "total", block: "total", lines: [0], scene: "total", caption: "the router now pays for large logits", vars: () => [["coef", "1e-3"], ["coef · L_z", (COEF * LZ0).toFixed(3)]] },
  { id: "grad", block: "back", lines: [0], scene: "grad", stage: 0, caption: "gradient: every logit of the token is pushed, by its softmax share", vars: () => [["rule", "(2z/B) · p"]] },
  { id: "apply", block: "back", lines: [1], scene: "grad", stage: 1, caption: "gradient step: all logits shrink, the leader most", vars: () => [["lr", `${ETA}`], ["L_z", `${fN(LZ0)} → ${fN(LZ1)}`]] },
  { id: "result", block: "result", lines: [0], scene: "cmp", caption: `after ${TRAIN_STEPS} z-loss steps: smaller logits, flatter softmax`, vars: () => [["L_z", `${fN(LZ0)} → ${fN(LZN)}`], ["mean max p", `${f2(MAXP0)} → ${f2(MAXPN)}`]] },
  { id: "note", block: "note", lines: [0, 1], visAt: 0, scene: "note", caption: "z-loss keeps logits tame; it does not balance load", vars: () => [] },
];

const GLOSS = { B: ["tokens in the batch", B], N: ["experts", N], d: ["hidden size", null] };
const sh = (name, dims, what, io) => ({ name, dims, what, io });
const SHAPES = {
  logits: [sh("x", ["B", "d"], "token vectors entering the router", "in"), sh("logits", ["B", "N"], "logits[i,j]: token i's raw score for expert j; softmax has not been applied yet", "out")],
  exp: [sh("logits[i]", ["N"], "the followed token's row", "in"), sh("e^logits[i]", ["N"], "positive, and one large logit makes one huge entry", "out")],
  stack: [sh("e^logits[i]", ["N"], "stacked one on top of another", "in"), sh("S_i", [], "Σ_j e^x_ij: height of the stack (this is the softmax denominator)", "out")],
  log: [sh("S_i", [], "stack height", "in"), sh("z_i", [], "ln S_i: lies between max_j x_ij and max_j x_ij + ln N", "out")],
  zbars: [sh("logits", ["B", "N"], "", "in"), sh("z", ["B"], "z[i] = logsumexp over the N experts of token i", "out")],
  sq: [sh("z", ["B"], "", "in"), sh("sq", ["B"], "z_i²", "out")],
  mean: [sh("sq", ["B"], "", "in"), sh("L_z", [], "mean over the B tokens (not over experts); times coef in the code", "out")],
  total: [sh("z_loss", [], "coef · L_z", "in"), sh("total", [], "added to the task loss and the aux loss", "out")],
  grad: [sh("z", ["B"], "", "in"), sh("p", ["B", "N"], "softmax(logits)", "in"), sh("grad", ["B", "N"], "(2 z_i / B) · p_ij: same shape as logits", "out")],
  apply: [sh("logits", ["B", "N"], "", "in"), sh("logits'", ["B", "N"], "logits − lr · grad", "out")],
  result: [sh("logits", ["B", "N"], "before and after training", "in"), sh("L_z", [], "scalar", "out")],
};

// ---- drawing pieces ----------------------------------------------------------
// vertical bars from a zero baseline; ghost = dashed outline of another value set
function VB({ x, y, w, h, vals, lo = 0, hi, cols = XC, labels, hl, fmt = f1, title, ghost, onPick, dimOthers }) {
  const n = vals.length, sc = h / (hi - lo || 1), zero = y + h - (0 - lo) * sc, bw = w / n;
  const col = (j) => (Array.isArray(cols) ? cols[j] : cols);
  const rect = (v, j, extra) => {
    const bh = Math.abs(v) * sc;
    return <rect x={x + j * bw + 3} y={v >= 0 ? zero - bh : zero} width={bw - 6} height={Math.max(bh, 0.5)} rx={3} {...extra} />;
  };
  return (
    <g>
      {title && lbl(title, x, y - 10)}
      <line x1={x} x2={x + w} y1={zero} y2={zero} stroke={T.rule} />
      {vals.map((v, j) => (
        <g key={j} opacity={dimOthers != null && dimOthers !== j ? 0.45 : 1}>
          {ghost && rect(ghost[j], j, { fill: "none", stroke: T.dim, strokeDasharray: "3 3", strokeWidth: 1.2 })}
          {rect(v, j, { fill: col(j), fillOpacity: 0.9 })}
          <text x={x + j * bw + bw / 2} y={v >= 0 ? zero - v * sc - 5 : zero - v * sc + 12} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.soft}>{fmt(v)}</text>
          <text x={x + j * bw + bw / 2} y={y + h + 14} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={hl === j ? T.ink : col(j)}>{labels ? labels[j] : `E${j}`}</text>
          {hl === j && <rect x={x + j * bw + 1} y={y - 4} width={bw - 2} height={h + 22} rx={5} fill="none" stroke={T.ink} strokeWidth={1.8} />}
          {onPick && <rect x={x + j * bw} y={y - 6} width={bw} height={h + 28} fill="transparent" style={{ cursor: "pointer" }} onClick={() => onPick(j)} />}
        </g>
      ))}
    </g>
  );
}
const TOKL = Array.from({ length: B }, (_, i) => `x${i}`);
const TOKC = Array(B).fill(T.accent);

// ---- scene -------------------------------------------------------------------
function Scene({ step, tok, setTok, ex }) {
  const p = useProgress(step.id);
  const e = ease(p);
  const xs = X0[tok], es = xs.map(Math.exp), S = sum(es), z = Z0[tok], mx = Math.max(...xs);
  let body = null;

  if (step.scene === "heat") {
    const x0 = 62, cw = 44, ch = 31, y0 = 70;
    body = (
      <g>
        {lbl("LOGITS x [B tokens × N experts]", x0, 34)}
        {Array.from({ length: N }, (_, j) => <text key={j} x={x0 + j * cw + cw / 2 - 1} y={y0 - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[j]}>E{j}</text>)}
        {X0.map((row, i) => (
          <g key={i} style={{ cursor: "pointer" }} onClick={() => setTok(i)}>
            <text x={x0 - 8} y={y0 + i * ch + 19} textAnchor="end" fontFamily={mono} fontSize="10" fontWeight={i === tok ? 700 : 400} fill={i === tok ? T.ink : T.dim}>x{i}</text>
            {row.map((v, j) => (
              <g key={j}>
                <rect x={x0 + j * cw} y={y0 + i * ch} width={cw - 3} height={ch - 3} rx={3} fill={div(v * e, 8)} />
                <text x={x0 + j * cw + (cw - 3) / 2} y={y0 + i * ch + 18} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="600" fill={T.ink} opacity={e}>{f1(v * e)}</text>
              </g>
            ))}
            {i === tok && <rect x={x0 - 2} y={y0 + i * ch - 2} width={N * cw + 1} height={ch + 1} rx={4} fill="none" stroke={T.ink} strokeWidth={1.8} />}
          </g>
        ))}
        <text x={x0} y={y0 + B * ch + 16} fontFamily={mono} fontSize="10.5" fill={T.dim}>warm = large positive · cool = negative · click a row to follow it</text>
        <VB x={500} y={90} w={330} h={210} vals={xs.map((v) => v * e)} lo={LO} hi={HI} title={`TOKEN x${tok}: ITS ${N} LOGITS`} />
        <text x={500} y={345} fontFamily={mono} fontSize="11.5" fill={T.soft} opacity={e}>largest logit: <tspan fill={T.wire} fontWeight="700">E{xs.indexOf(mx)} = {f1(mx)}</tspan></text>
      </g>
    );
  } else if (step.scene === "exp" && step.stage < 2) {
    const Hh = 230, base = 340, bx = 40, bwid = 55;
    const xm = Math.max(...xs.map(Math.abs));
    const stage = step.stage;
    let cum = 0;
    const segs = es.map((v, j) => {
      const hh = (v / S) * Hh, startPx = (xs[j] / xm) * Hh * 0.45;
      const aX = bx + j * bwid + 3, aW = bwid - 6;
      const y1 = base - cum - hh; cum += hh;
      // stage 0: logit bar → e^x bar. stage 1: e^x bars slide into one stack.
      if (stage === 0) {
        const px = lerp(startPx, hh, e);
        return { x: aX, w: aW, y: px >= 0 ? base - px : base, h: Math.abs(px), val: lerp(xs[j], v, e), j };
      }
      return { x: lerp(aX, 520, e), w: lerp(aW, 80, e), y: lerp(base - hh, y1, e), h: hh, val: v, j };
    });
    body = (
      <g>
        {lbl(stage === 0 ? `TOKEN x${tok}: LOGIT x → e^x   (HEIGHT = SHARE OF THE TOTAL)` : `TOKEN x${tok}: e^x STACKED INTO S`, 40, 52)}
        <line x1={30} x2={stage === 0 ? 485 : 640} y1={base} y2={base} stroke={T.rule} />
        {segs.map((s) => (
          <g key={s.j}>
            <rect x={s.x} y={s.y} width={s.w} height={Math.max(s.h, 0.5)} rx={3} fill={XC[s.j]} fillOpacity={0.9} stroke={T.bg} />
            {stage === 0 && <text x={s.x + s.w / 2} y={(s.val >= 0 ? s.y : s.y + s.h) + (s.val >= 0 ? -5 : 12)} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.soft}>{fN(s.val)}</text>}
            {stage === 0 && <text x={s.x + s.w / 2} y={base + 38} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={XC[s.j]}>E{s.j}</text>}
          </g>
        ))}
        {stage === 1 && (
          <g opacity={e}>
            <line x1={620} x2={620} y1={base - Hh} y2={base} stroke={T.wire} />
            <text x={632} y={base - Hh / 2} fontFamily={mono} fontSize="14" fontWeight="700" fill={T.wire}>S = {fN(S)}</text>
            <text x={632} y={base - Hh / 2 + 20} fontFamily={mono} fontSize="10.5" fill={T.soft}>E{es.indexOf(Math.max(...es))} is {(Math.max(...es) / S * 100).toFixed(0)}% of it</text>
          </g>
        )}
        {stage === 0 && <text x={500} y={120} fontFamily={mono} fontSize="11.5" fill={T.soft}>e^x turns a gap of 1 into a ×{Math.E.toFixed(1)}: the largest logit</text>}
        {stage === 0 && <text x={500} y={138} fontFamily={mono} fontSize="11.5" fill={T.soft}>takes most of the height.</text>}
      </g>
    );
  } else if (step.scene === "exp") {
    // stage 2: ln(S) on a number line
    const lo = Math.floor(Math.min(...xs)) - 1, hi = Math.ceil(z) + 1, nx0 = 70, nw = 740, ny = 200;
    const px = (v) => nx0 + ((v - lo) / (hi - lo)) * nw;
    const zNow = lerp(mx, z, e);
    const gap = z - mx;
    body = (
      <g>
        {lbl(`TOKEN x${tok}: ln S ON THE LOGIT AXIS`, 40, 52)}
        <line x1={nx0} x2={nx0 + nw} y1={ny} y2={ny} stroke={T.rule} strokeWidth={1.5} />
        {Array.from({ length: hi - lo + 1 }, (_, k) => lo + k).map((v) => (
          <g key={v}><line x1={px(v)} x2={px(v)} y1={ny - 4} y2={ny + 4} stroke={T.rule} /><text x={px(v)} y={ny + 18} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={T.dim}>{v}</text></g>
        ))}
        {xs.map((v, j) => (
          <g key={j}>
            <circle cx={px(v)} cy={ny} r={6} fill={XC[j]} stroke={T.bg} />
            <text x={px(v)} y={ny - 14 - (j % 2) * 13} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={XC[j]}>E{j}</text>
          </g>
        ))}
        <line x1={px(mx)} x2={px(mx)} y1={ny - 56} y2={ny + 40} stroke={T.dim} strokeDasharray="4 3" />
        <text x={px(mx)} y={ny + 56} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.dim}>max = {f2(mx)}</text>
        <line x1={px(zNow)} x2={px(zNow)} y1={ny - 56} y2={ny + 40} stroke={T.wire} strokeWidth={2.4} />
        <text x={px(zNow)} y={ny - 62} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={T.wire}>z = {f2(zNow)}</text>
        <g opacity={e}>
          <text x={nx0} y={ny + 96} fontFamily={mono} fontSize="12" fill={T.soft}>z − max = <tspan fill={T.wire} fontWeight="700">{f2(gap)}</tspan> · never more than ln N = {f2(Math.log(N))}</text>
          <text x={nx0} y={ny + 116} fontFamily={mono} fontSize="12" fill={T.soft}>{gap < 0.5 ? "one logit dominates: z ≈ max logit" : "logits are close together: z sits above the max"}</text>
          <Tex x={nx0} y={ny + 134} w={700} h={30} size={14} tex={`z=\\ln S=\\ln(${cc(T.wire, fN(S))})=${cc(T.wire, f2(z))}`} />
        </g>
      </g>
    );
  } else if (step.scene === "zbars") {
    body = (
      <g>
        <VB x={40} y={90} w={520} h={230} vals={Z0.map((v) => v * e)} hi={Math.max(...Z0) * 1.15} cols={TOKC} labels={TOKL} hl={tok} onPick={setTok} title="z_i  (DASHED = THE ROW'S LARGEST LOGIT)" ghost={X0.map((row) => Math.max(...row) * e)} />
        <Tex x={590} y={100} w={250} h={34} size={14} tex={`z_i=\\ln\\sum_j e^{x_{ij}}\\approx\\max_j x_{ij}`} opacity={e} />
        <text x={590} y={160} fontFamily={mono} fontSize="11.5" fill={T.soft} opacity={e}>x{tok}: max {f2(Math.max(...xs))} → z {f2(z)}</text>
        <text x={590} y={188} fontFamily={mono} fontSize="11.5" fill={T.soft} opacity={e}>confident tokens have large z,</text>
        <text x={590} y={206} fontFamily={mono} fontSize="11.5" fill={T.soft} opacity={e}>and also a peaked softmax.</text>
      </g>
    );
  } else if (step.scene === "sq") {
    const hiV = Math.max(...Z2) * 1.15;
    body = (
      <g>
        <VB x={40} y={90} w={520} h={230} vals={Z0.map((v, i) => lerp(v, Z2[i], e))} hi={lerp(Math.max(...Z0) * 1.15, hiV, e)} cols={TOKC} labels={TOKL} hl={tok} onPick={setTok} fmt={fN} title="z_i → z_i²" ghost={Z0} />
        <Tex x={590} y={100} w={250} h={34} size={15} tex={`z_{${tok}}^2=(${cc(T.wire, f2(z))})^2=${cc(T.accent, fN(z * z))}`} />
        <text x={590} y={160} fontFamily={mono} fontSize="11.5" fill={T.soft}>z doubles → penalty ×4</text>
        <text x={590} y={184} fontFamily={mono} fontSize="11.5" fill={T.soft}>x{Z0.indexOf(Math.max(...Z0))}: {f1(Math.max(...Z0))} → {fN(Math.max(...Z2))}</text>
        <text x={590} y={204} fontFamily={mono} fontSize="11.5" fill={T.soft}>x{Z0.indexOf(Math.min(...Z0))}: {f1(Math.min(...Z0))} → {fN(Math.min(...Z2))}</text>
      </g>
    );
  } else if (step.scene === "mean") {
    const e1 = ease(clamp01(p / 0.55)), e2 = ease(clamp01((p - 0.55) / 0.45));
    const tot = sum(Z2), base = 340, Hh = 250, cx = 650;
    let cum = 0;
    const part = e1 * B; // how many segments have been placed
    const kShrink = lerp(1, 1 / B, e2);
    body = (
      <g>
        <VB x={40} y={90} w={420} h={230} vals={Z2} hi={Math.max(...Z2) * 1.15} cols={TOKC} labels={TOKL} hl={tok} onPick={setTok} fmt={fN} title="z_i²" />
        <line x1={40} x2={460} y1={320 - (LZ0 / (Math.max(...Z2) * 1.15)) * 230} y2={320 - (LZ0 / (Math.max(...Z2) * 1.15)) * 230} stroke={T.wire} strokeWidth={2} strokeDasharray="6 4" opacity={e2} />
        <text x={456} y={314 - (LZ0 / (Math.max(...Z2) * 1.15)) * 230} textAnchor="end" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.wire} opacity={e2}>L_z = {fN(LZ0)}</text>
        {lbl("SUM OVER TOKENS, THEN ÷ B", 540, 80)}
        <line x1={560} x2={760} y1={base} y2={base} stroke={T.rule} />
        {Z2.map((v, i) => {
          const hh = (v / tot) * Hh * kShrink, vis = clamp01(part - i);
          const y = base - cum * kShrink - hh * vis; cum += (v / tot) * Hh;
          return <rect key={i} x={cx - 40} y={y} width={80} height={Math.max(hh * vis, 0)} fill={XC[i]} fillOpacity={0.9} stroke={T.bg} />;
        })}
        <text x={cx} y={base + 24} textAnchor="middle" fontFamily={mono} fontSize="13" fontWeight="700" fill={T.wire}>{e2 < 0.02 ? `Σ z² = ${fN(tot * e1)}` : `Σ ÷ ${B} = ${fN(lerp(tot, LZ0, e2))}`}</text>
        <Tex x={540} y={96} w={310} h={30} size={13} tex={`L_z=\\dfrac{1}{B}\\sum_i z_i^2=\\dfrac{${fN(tot)}}{${B}}=${cc(T.wire, fN(LZ0))}`} opacity={e2} />
      </g>
    );
  } else if (step.scene === "total") {
    const boxes = [
      { t: "task_loss", c: T.soft, d: "next-token loss", w: 0 },
      { t: "aux", c: XC[3], d: "load balance", w: 0.25 },
      { t: "z_loss", c: T.accent, d: `${COEF} × ${fN(LZ0)} = ${(COEF * LZ0).toFixed(3)}`, w: 0.5 },
    ];
    body = (
      <g>
        {boxes.map((b, i) => {
          const a = ease(clamp01((p - b.w) / 0.4));
          return (
            <g key={i} opacity={a} transform={`translate(0,${(1 - a) * 14})`}>
              <rect x={40 + i * 270} y={110} width={240} height={120} rx={12} fill={T.well} stroke={b.c} strokeWidth={1.6} />
              <text x={60 + i * 270} y={150} fontFamily={mono} fontSize="16" fontWeight="700" fill={b.c}>{i ? "+ " : ""}{b.t}</text>
              <text x={60 + i * 270} y={184} fontFamily={mono} fontSize="12" fill={T.soft}>{b.d}</text>
            </g>
          );
        })}
        <Tex x={40} y={270} w={780} h={30} size={15} tex={`\\mathcal L=\\mathcal L_{task}+\\mathcal L_{aux}+${cc(T.accent, "c")}\\,L_z`} opacity={ease(clamp01((p - 0.6) / 0.4))} />
      </g>
    );
  } else if (step.scene === "grad") {
    const pr = P0[tok], gr = G0[tok], c = (2 * z) / B, J = ex, col = XC[J];
    if (step.stage === 0) {
      const gmax = Math.max(...gr);
      body = (
        <g>
          <VB x={40} y={90} w={400} h={200} vals={pr.map((v, j) => lerp(v, gr[j], e))} hi={lerp(1.05, gmax * 1.15, e)} fmt={(v) => (e < 0.5 ? f2(v) : f2(v))} title={`TOKEN x${tok}: SOFTMAX p → GRADIENT g = c·p`} hl={J} onPick={() => {}} />
          <Tex x={470} y={90} w={380} h={36} size={14} tex={`\\dfrac{\\partial L_z}{\\partial x_{ij}}=\\dfrac{2\\,z_i}{B}\\,p_{ij}`} />
          <Tex x={470} y={140} w={380} h={34} size={14} tex={`c=\\dfrac{2\\cdot${cc(T.wire, f2(z))}}{${B}}=${cc(T.wire, f2(c))}\\quad\\text{(same for the token's experts)}`} />
          <Tex x={470} y={190} w={380} h={34} size={14} tex={`g_{\\mathrm{E}_{${J}}}=${cc(T.wire, f2(c))}\\cdot${cc(col, f2(pr[J]))}=${cc(col, f2(gr[J]))}`} opacity={e} />
          <text x={470} y={262} fontFamily={mono} fontSize="11.5" fill={T.soft}>big z → big c; the favourite expert (high p)</text>
          <text x={470} y={280} fontFamily={mono} fontSize="11.5" fill={T.soft}>gets almost all of it. Same shape as p.</text>
        </g>
      );
    } else {
      const xa = X0[tok], xb = X1[tok], pa = P0[tok], pb = P1[tok];
      body = (
        <g>
          <VB x={40} y={70} w={380} h={200} vals={xa.map((v, j) => lerp(v, xb[j], e))} lo={LO} hi={HI} ghost={xa} title={`TOKEN x${tok}: LOGITS x → x − lr·g`} hl={J} />
          <VB x={470} y={70} w={360} h={200} vals={pa.map((v, j) => lerp(v, pb[j], e))} hi={1.05} ghost={pa} fmt={f2} title="SOFTMAX p (DASHED = BEFORE)" hl={J} />
          <Tex x={40} y={318} w={420} h={34} size={14} tex={`x'_{\\mathrm{E}_{${J}}}=${cc(col, f2(xa[J]))}-${ETA}\\cdot${cc(col, f2(G0[tok][J]))}=${cc(col, f2(xb[J]))}`} />
          <text x={470} y={316} fontFamily={mono} fontSize="12" fill={T.soft}>max p <tspan fill={T.bad} fontWeight="700">{f2(Math.max(...pa))}</tspan> → <tspan fill={T.accent} fontWeight="700">{f2(lerp(Math.max(...pa), Math.max(...pb), e))}</tspan></text>
          <text x={470} y={338} fontFamily={mono} fontSize="12" fill={T.soft}>z <tspan fill={T.bad} fontWeight="700">{f2(z)}</tspan> → <tspan fill={T.accent} fontWeight="700">{f2(lerp(z, lse(xb), e))}</tspan></text>
          <text x={470} y={360} fontFamily={mono} fontSize="12" fill={T.soft}>L_z (all tokens) {fN(LZ0)} → {fN(lerp(LZ0, LZ1, e))}</text>
        </g>
      );
    }
  } else if (step.scene === "cmp") {
    const xa = X0[tok], xb = XN[tok], pa = P0[tok], pb = PN[tok];
    const ZN = zOf(XN);
    const bigA = Math.max(...xa.map(Math.abs)), bigB = Math.max(...xb.map(Math.abs));
    body = (
      <g>
        <VB x={40} y={56} w={380} h={120} vals={xa.map((v, j) => lerp(v, xb[j], e))} lo={LO} hi={HI} ghost={xa} title={`TOKEN x${tok}: LOGITS`} />
        <VB x={470} y={56} w={360} h={120} vals={pa.map((v, j) => lerp(v, pb[j], e))} hi={1.05} ghost={pa} fmt={f2} title="SOFTMAX (DASHED = BEFORE)" />
        <VB x={40} y={250} w={380} h={100} vals={Z0.map((v, i) => lerp(v, ZN[i], e))} hi={Math.max(...Z0) * 1.15} cols={TOKC} labels={TOKL} ghost={Z0} title="z_i FOR ALL TOKENS" hl={tok} />
        <g fontFamily={mono} fontSize="12.5">
          <text x={470} y={252} fill={T.soft}>L_z  <tspan fill={T.bad} fontWeight="700">{fN(LZ0)}</tspan> → <tspan fill={T.accent} fontWeight="700">{fN(lerp(LZ0, LZN, e))}</tspan></text>
          <text x={470} y={276} fill={T.soft}>mean max p  <tspan fill={T.bad} fontWeight="700">{f2(MAXP0)}</tspan> → <tspan fill={T.accent} fontWeight="700">{f2(lerp(MAXP0, MAXPN, e))}</tspan></text>
          <text x={470} y={300} fill={T.soft}>largest |logit|  <tspan fill={T.bad} fontWeight="700">{f1(bigA)}</tspan> → <tspan fill={T.accent} fontWeight="700">{f1(lerp(bigA, bigB, e))}</tspan></text>
          <text x={470} y={328} fontSize="11" fill={T.dim}>bf16 spacing near the largest logit:</text>
          <text x={470} y={346} fontSize="11" fill={T.dim}>{bf16Step(bigA).toFixed(3)} → {bf16Step(bigB).toFixed(3)} (finer = more precise)</text>
        </g>
        <text x={40} y={392} fontFamily={mono} fontSize="10.5" fill={T.dim}>toy: z-loss alone, {TRAIN_STEPS} steps; in real training the task loss pulls the other way</text>
      </g>
    );
  } else {
    const card = (x, a, c, t1, t2, t3, t4) => {
      const q = ease(clamp01((p - a) / 0.5));
      return (
        <g opacity={q} transform={`translate(0,${(1 - q) * 14})`}>
          <rect x={x} y={70} width={390} height={200} rx={12} fill={T.well} stroke={c} strokeWidth={1.6} />
          <text x={x + 20} y={106} fontFamily={mono} fontSize="16" fontWeight="700" fill={c}>{t1}</text>
          <text x={x + 20} y={140} fontFamily={mono} fontSize="12.5" fill={T.ink}>{t2}</text>
          <text x={x + 20} y={168} fontFamily={mono} fontSize="12.5" fill={T.soft}>{t3}</text>
          <text x={x + 20} y={196} fontFamily={mono} fontSize="12.5" fill={T.soft}>{t4}</text>
        </g>
      );
    };
    body = (
      <g>
        {card(20, 0, T.accent, "router z-loss", "acts on the SIZE of the logits", "keeps softmax unsaturated, bf16 precise", "says nothing about which expert wins")}
        {card(450, 0.3, XC[3], "aux loss / bias", "acts on the SPREAD of tokens", "keeps experts evenly loaded", "says nothing about how big logits get")}
        <text x={20} y={316} fontFamily={mono} fontSize="12.5" fill={T.soft} opacity={ease(clamp01((p - 0.6) / 0.4))}>different jobs, so they are used together</text>
      </g>
    );
  }
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>;
}

export default function RouterZLossView() {
  const [tok, setTok] = useState(4); // token every picture follows
  const [ex, setEx] = useState(6); // expert the gradient equations are written out for
  return (
    <Stepper
      title="Router z-loss"
      subtitle={`penalise big router logits · toy: B=${B} tokens, N=${N} experts`}
      blocks={BLOCKS}
      steps={STEPS}
      scene={(step) => <Scene step={step} tok={tok} setTok={setTok} ex={ex} />}
      header={(step) => step.scene !== "total" && step.scene !== "note" && (
        <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow token</span>
          {Array.from({ length: B }, (_, i) => (
            <button key={i} onClick={() => setTok(i)} style={{ ...btn(i === tok, false), height: 28, padding: "0 9px" }}>x{i}</button>
          ))}
          {step.scene === "grad" && (
            <>
              <span style={{ fontSize: 10.5, color: T.dim, margin: "0 4px 0 10px" }}>expert</span>
              {Array.from({ length: N }, (_, j) => (
                <button key={j} onClick={() => setEx(j)} style={{ ...btn(j === ex, false), height: 28, padding: "0 9px", color: j === ex ? undefined : XC[j] }}>E{j}</button>
              ))}
            </>
          )}
        </div>
      )}
      glossary={GLOSS}
      shapes={(step) => SHAPES[step.sid]}
      legend={<span>L_z = (1/B) Σ_i logsumexp_j(x_ij)² · <span style={{ color: T.wire }}>gold</span> = the numbers being plugged in</span>}
    />
  );
}
