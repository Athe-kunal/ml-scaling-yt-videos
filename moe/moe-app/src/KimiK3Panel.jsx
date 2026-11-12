import { T, mono } from "./theme";
import { XC, rng } from "./lbModel";
import { Tex } from "./tex";
import { useProgress, ease, lerp } from "./anim";

// Kimi K3-scale companion to the toy scenes: real sizes (n = 896, k = 16, B = 1000 bins from §D),
// illustrative distributions. One card per step, drawn under the toy.
const N3 = 896, K3 = 16, M3 = 1_000_000, NBIN = 1000; // m = global tokens per step (illustrative)
const W = 860, H = 232;
const TARGET = (M3 * K3) / N3; // 17,857 tokens per expert

const erf = (x) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const pdf = (z) => Math.exp(-z * z / 2);
const fmt = (v) => Math.round(v).toLocaleString("en-US");

// 896 synthetic raw biases with a non-zero mean, so centring has something to remove
const BH = (() => {
  const r = rng(11);
  return Array.from({ length: N3 }, () => 0.18 + 0.12 * Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r()));
})();
const MEAN = BH.reduce((a, x) => a + x, 0) / N3;

function Strip({ x, y, w, vals, scale, color, mean }) {
  const d = vals.map((v, i) => `M${(x + (i + 0.5) * (w / N3)).toFixed(2)} ${y}V${(y - v * scale).toFixed(1)}`).join("");
  return (
    <g>
      <line x1={x} x2={x + w} y1={y} y2={y} stroke={T.rule} />
      <path d={d} stroke={color} strokeWidth={0.5} fill="none" />
      {mean != null && <line x1={x} x2={x + w} y1={y - mean * scale} y2={y - mean * scale} stroke={T.wire} strokeDasharray="4 3" />}
    </g>
  );
}

export const K3_STEPS = new Set(["q", "quantile", "center", "onegpu", "hist", "reduce", "bhat", "apply"]);

export default function K3Panel({ step, rk = 0 }) {
  const p = useProgress(`k3:${step.id}:${rk}`, 1500);
  const e = ease(p);
  const T0 = (x, y, s, o = {}) => <text x={x} y={y} fontFamily={mono} fontSize={o.size ?? 11.5} fill={o.fill ?? T.soft} fontWeight={o.bold ? 700 : 400} textAnchor={o.anchor}>{s}</text>;
  let body = null;

  if (step.id === "q") {
    body = (
      <g>
        {Array.from({ length: 56 }, (_, i) => (
          <rect key={i} x={20 + i * 15} y={44} width={12} height={22} rx={2} fill={i === 55 ? XC[0] : T.rule} opacity={i === 55 ? e : 1} />
        ))}
        <line x1={20} x2={20 + 55 * 15 - 3} y1={76} y2={76} stroke={T.dim} />
        <line x1={20 + 55 * 15} x2={20 + 55 * 15 + 12} y1={76} y2={76} stroke={XC[0]} strokeWidth={2} />
        {T0(20, 94, "55 of every 56 tokens skip this expert")}
        {T0(W - 20, 94, "1 picks it", { anchor: "end", fill: XC[0], bold: true })}
        <Tex x={20} y={108} w={820} h={46} size={15} color={T.ink} tex={`q=1-\\dfrac{k}{n}=1-\\dfrac{16}{896}=1-\\dfrac{1}{56}\\approx\\mathbf{0.982}`} />
        {T0(20, 176, `a threshold at the 98.2nd percentile of one expert's margins leaves 1.8% above it`)}
        {T0(20, 196, `with m = ${fmt(M3)} tokens that is m·k/n = ${fmt(TARGET)} tokens per expert · toy: q = 0.50, half of the tokens`, { fill: T.dim })}
      </g>
    );
  } else if (step.id === "quantile") {
    const zt = lerp(3.4, 2.1, e), z0 = 1.45;
    const X = (z) => 20 + ((z + 3) / 6.6) * 520, Y = (z) => 176 - pdf(z) * 120;
    const pts = (a, b) => Array.from({ length: 41 }, (_, i) => a + ((b - a) * i) / 40);
    const curve = pts(-3, 3.6).map((z, i) => `${i ? "L" : "M"}${X(z).toFixed(1)} ${Y(z).toFixed(1)}`).join("");
    const area = (a, b) => `M${X(a)} 176` + pts(a, b).map((z) => `L${X(z).toFixed(1)} ${Y(z).toFixed(1)}`).join("") + `L${X(b)} 176Z`;
    body = (
      <g>
        {T0(20, 30, "ONE EXPERT · MARGINS OF ALL TOKENS", { size: 10, fill: T.dim })}
        <path d={area(z0, zt)} fill={T.bad} opacity={0.35} />
        <path d={area(zt, 3.6)} fill={XC[0]} opacity={0.9} />
        <path d={curve} stroke={T.soft} fill="none" strokeWidth={1.5} />
        <line x1={20} x2={540} y1={176} y2={176} stroke={T.rule} />
        <line x1={X(z0)} x2={X(z0)} y1={50} y2={176} stroke={T.dim} strokeDasharray="3 3" />
        <line x1={X(zt)} x2={X(zt)} y1={44} y2={176} stroke={T.wire} strokeWidth={2} />
        {T0(X(z0) - 4, 46, "0 = cutoff", { anchor: "end", fill: T.dim, size: 10.5 })}
        {T0(X(zt) + 4, 40, "threshold", { fill: T.wire, size: 10.5 })}
        {T0(580, 60, `above 0: ${(100 * (1 - Phi(z0))).toFixed(1)}% of tokens`, { fill: T.bad, bold: true })}
        {T0(580, 80, `target:  ${(100 * K3 / N3).toFixed(1)}%`, { fill: XC[0], bold: true })}
        {T0(580, 112, "too many pass → overloaded")}
        {T0(580, 130, "threshold sits right of 0")}
        {T0(580, 156, "b̂ = −threshold < 0", { fill: T.accent2, bold: true })}
        {T0(580, 174, "pushes this expert down")}
        {T0(20, 204, "red = tokens above 0 but below the threshold: the excess the bias must remove", { fill: T.dim, size: 10.5 })}
      </g>
    );
  } else if (step.id === "center") {
    const cen = BH.map((v) => lerp(v, v - MEAN, e));
    body = (
      <g>
        {T0(20, 30, `b̂ FOR ALL ${N3} EXPERTS (RAW)`, { size: 10, fill: T.dim })}
        {T0(450, 30, "b = b̂ − mean(b̂)", { size: 10, fill: T.dim })}
        <Strip x={20} y={122} w={390} vals={BH} scale={150} color={XC[3]} mean={MEAN} />
        <Strip x={450} y={122} w={390} vals={cen} scale={150} color={XC[0]} mean={MEAN * (1 - e)} />
        {T0(20, 176, `mean(b̂) = ${MEAN.toFixed(3)}  (dashed)`, { fill: T.wire })}
        {T0(450, 176, `mean(b) = ${(MEAN * (1 - e)).toFixed(3)}`, { fill: T.wire })}
        {T0(20, 206, "every expert moves by the same amount, so no token's top-16 changes · the bias just stays near 0", { fill: T.dim, size: 10.5 })}
      </g>
    );
  } else if (step.id === "onegpu") {
    const R = 8;
    body = (
      <g>
        {Array.from({ length: R }, (_, i) => (
          <g key={i} opacity={Math.min(1, e * 3 - i * 0.2 + 0.2)}>
            <rect x={20 + i * 105} y={40} width={95} height={54} rx={6} fill={T.well} stroke={T.wire} />
            {T0(67 + i * 105, 62, `GPU ${i}`, { anchor: "middle", fill: T.wire, bold: true })}
            {T0(67 + i * 105, 82, "m/R tokens", { anchor: "middle", fill: T.dim, size: 10 })}
          </g>
        ))}
        {T0(20, 126, `real batch: m ≈ ${fmt(M3)} tokens, spread over many ranks and micro-batches`)}
        {T0(20, 150, `exact quantile needs all m·n = ${fmt(M3 * N3)} margins in one place, in every layer, every step`, { fill: T.bad })}
        {T0(20, 180, "gathering that is not viable, so we send a count per bin instead of every margin", { fill: T.dim, size: 10.5 })}
      </g>
    );
  } else if (step.id === "hist") {
    const r = rng(5);
    const rows = 12, cols = 24;
    const cell = (i, c) => Math.exp(-((c - (5 + i * 1.3)) ** 2) / 40) * (0.8 + 0.4 * r());
    body = (
      <g>
        {Array.from({ length: 4 }, (_, rk2) => (
          <g key={rk2}>
            {T0(20 + rk2 * 210, 26, `RANK ${rk2} · H_r  [${N3}, B]`, { size: 10, fill: T.dim })}
            {Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, c) => (
              <rect key={`${i}-${c}`} x={20 + rk2 * 210 + c * 7.9} y={34 + i * 8} width={7} height={7} fill={XC[0]} opacity={Math.min(1, cell(i, c)) * e} />
            )))}
          </g>
        ))}
        {T0(20, 148, `rows = ${N3} experts (a sample is drawn) · columns = B = ${NBIN} bins of required bias r = α − s`, { fill: T.dim, size: 10.5 })}
        <Tex x={20} y={158} w={820} h={40} size={13} color={T.ink} tex={`r_{ij}\\in[b_{\\min}-1,\\;b_{\\max}+1],\\quad w=\\dfrac{b_{\\max}-b_{\\min}+2}{B}\\quad\\text{(range tracks the current bias)}`} />
        {T0(20, 214, "each rank scatter-adds its own tokens over all micro-batches · no communication yet", { fill: T.soft })}
      </g>
    );
  } else if (step.id === "reduce") {
    const raw = M3 * N3, hist = N3 * NBIN, bw = 800;
    body = (
      <g>
        {T0(20, 40, `RAW MARGINS  m·n = ${fmt(M3)} · ${N3} = ${fmt(raw)} numbers`, { size: 10.5, fill: T.dim })}
        <rect x={20} y={48} width={bw * e} height={26} rx={3} fill={T.bad} opacity={0.8} />
        {T0(20, 108, `HISTOGRAM  n·B = ${N3} · ${NBIN} = ${fmt(hist)} integers`, { size: 10.5, fill: T.dim })}
        <rect x={20} y={116} width={Math.max(1.5, (bw * hist / raw) * e)} height={26} rx={2} fill={XC[0]} />
        {T0(32, 134, "← 1000× smaller", { fill: XC[0], bold: true })}
        <Tex x={20} y={158} w={820} h={36} size={13.5} color={T.ink} tex={`\\mathrm{hist}=\\sum_{r=1}^{R}H_r\\quad\\text{one integer all-reduce per layer per step; size independent of }m\\text{ and of }R`} />
        {T0(20, 214, "counts add, so the pooled histogram is the whole batch however tokens are split across ranks", { fill: T.dim, size: 10.5 })}
      </g>
    );
  } else if (step.id === "bhat") {
    const counts = [2400, 3000, 3500, 4000, 4520, 5000, 5400, 5800, 6100];
    const sel = 5, cj = counts.slice(0, sel).reduce((a, x) => a + x, 0), hj = counts[sel];
    const frac = (TARGET - cj) / hj;
    const bx = (i) => 20 + i * 50, mx = 6100;
    body = (
      <g>
        {T0(20, 26, "POOLED COUNTS, BINS OF r = α − s WALKED FROM THE LOW END", { size: 10, fill: T.dim })}
        {counts.map((c, i) => {
          const h = (c / mx) * 96 * e;
          return (
            <g key={i}>
              <rect x={bx(i)} y={134 - h} width={44} height={h} rx={2} fill={i < sel ? XC[3] : i === sel ? XC[0] : T.rule} />
              {i === sel && <rect x={bx(i) + 44 * frac - 1} y={134 - h} width={2} height={h} fill={T.ink} />}
              {T0(bx(i) + 22, 150, fmt(c), { anchor: "middle", size: 9.5, fill: T.dim })}
            </g>
          );
        })}
        {T0(bx(0), 170, `c_j = ${fmt(cj)} below the chosen bin`, { fill: XC[3] })}
        {T0(bx(sel) - 20, 188, `h_j = ${fmt(hj)}`, { fill: XC[0], bold: true })}
        {T0(480, 44, `target load q = ${fmt(TARGET)}`, { fill: T.wire, bold: true })}
        {T0(480, 62, `${fmt(cj)} < q ≤ ${fmt(cj + hj)}: bin ${sel} is the first to reach q`)}
        <Tex x={480} y={78} w={370} h={60} size={13} color={T.ink} tex={`\\hat b_j=b_{\\min}-1+\\bigl(\\beta_j+\\mathrm{clip}(\\tfrac{q-c_j}{h_j},0,1)\\bigr)\\,w`} />
        <Tex x={480} y={140} w={370} h={34} size={13} color={T.accent2} tex={`\\tfrac{q-c_j}{h_j}=\\tfrac{${fmt(TARGET)}-${fmt(cj)}}{${fmt(hj)}}=${frac.toFixed(2)}\\ \\text{of the way in}`} />
        {T0(480, 196, "error ≤ one bin width w: a few 10⁻³ at B = 1000", { fill: T.dim, size: 10.5 })}
      </g>
    );
  } else {
    const loop = step.id === "apply";
    const boxes = [["batch t", "route with b from t−1"], ["histograms", "scatter-add per rank"], ["one all-reduce", "pooled H, read quantile"], ["b (centred)", "used on batch t+1"]];
    body = (
      <g>
        {boxes.map(([a, b], i) => (
          <g key={i} opacity={Math.min(1, e * 4 - i * 0.6 + 0.4)}>
            <rect x={20 + i * 210} y={44} width={190} height={62} rx={8} fill={T.well} stroke={i === 3 ? XC[0] : T.rule} />
            {T0(115 + i * 210, 70, a, { anchor: "middle", fill: T.ink, bold: true })}
            {T0(115 + i * 210, 90, b, { anchor: "middle", fill: T.dim, size: 10 })}
            {i < 3 && T0(200 + i * 210 + 10, 80, "→", { anchor: "middle", fill: T.wire, size: 16, bold: true })}
          </g>
        ))}
        {T0(20, 140, loop ? "a batch is never routed with a bias derived from itself" : "run in every MoE layer, every step: one integer all-reduce, no step size γ to tune", { fill: T.soft })}
        {T0(20, 164, "optional: an exponential moving average of the quantiles across steps smooths batch-to-batch noise", { fill: T.dim, size: 10.5 })}
        {T0(20, 188, "at inference the bias is frozen: nothing is measured or updated", { fill: T.dim, size: 10.5 })}
      </g>
    );
  }

  return (
    <div style={{ borderTop: `1px solid ${T.rule}`, marginTop: 8, paddingTop: 8 }}>
      <div style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: "0.08em", color: T.accent2, padding: "0 4px 2px" }}>
        AT KIMI K3 SCALE · n = {N3} experts, k = {K3} · real formulas, illustrative numbers
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>{body}</svg>
    </div>
  );
}
