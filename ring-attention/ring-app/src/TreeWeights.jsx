import { T, mono } from "./theme";
import { katexHtml } from "./latex";
import { DEV_COLORS } from "./ringModel";
import { TREE_PHASE, PER, log2, fmt, covList, stageOf } from "./treeModel";

// The five stages of tree decoding, with the one in progress lit. Allreduce
// stages show one dot per tree level (log2 N up, then log2 N down).
const STAGES = [
  { key: "local", title: "1 · local attn", tex: String.raw`\ell_i,\ o_i`, sub: "flash-attn on own shard", code: "L4" },
  { key: "max", title: "2 · allreduce max", tex: String.raw`m = \max_i \ell_i`, sub: "tree ↑ then ↓", code: "L7–8" },
  { key: "rescale", title: "3 · rescale", tex: String.raw`w_i = e^{\ell_i - m}`, sub: "n_i = w_i·o_i, d_i = w_i", code: "L11–13" },
  { key: "sum", title: "4 · allreduce sum", tex: String.raw`\textstyle\sum n_i,\ \sum d_i`, sub: "tree ↑ then ↓", code: "L16–17" },
  { key: "final", title: "5 · divide", tex: String.raw`o = n / d`, sub: "same on every GPU", code: "L18–19" },
];

export function StageStrip({ N, phase }) {
  const cur = stageOf(phase);
  const L = log2(N);
  const order = STAGES.map((s) => s.key);
  const curIdx = order.indexOf(cur);
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${STAGES.length}, minmax(0, 1fr))`, gap: 6 }}>
      {STAGES.map((st, i) => {
        const on = st.key === cur;
        const done = curIdx > i;
        const isAR = st.key === "max" || st.key === "sum";
        const c = isAR ? (on ? TREE_PHASE[phase.type].color : T.wire) : st.key === "rescale" ? T.accent2 : st.key === "local" ? T.accent : T.soft;
        // which level dot is active: up levels 1..L, then down levels L..1
        const dot = on && isAR ? (phase.type === "up" ? phase.lvl - 1 : L + (L - phase.lvl)) : -1;
        return (
          <div key={st.key} style={{ borderRadius: 8, padding: "8px 10px", minWidth: 0, background: on ? `${c}1c` : T.well,
            border: `1px solid ${on ? c : T.rule}`, opacity: done || on ? 1 : 0.6, transition: "all .25s" }}>
            <div style={{ fontFamily: mono, fontSize: 10.5, fontWeight: 700, color: on ? c : done ? T.soft : T.dim, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{st.title}{done ? " ✓" : ""}</div>
            <div style={{ fontSize: 13, color: T.ink, margin: "5px 0 1px", whiteSpace: "nowrap", overflow: "hidden" }} dangerouslySetInnerHTML={{ __html: katexHtml(st.tex) }} />
            <div style={{ fontFamily: mono, fontSize: 9.5, color: T.dim, marginBottom: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{st.sub}</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
              <span style={{ fontFamily: mono, fontSize: 9.5, color: T.dim }}>{st.code}</span>
              {isAR && (
                <span style={{ display: "flex", gap: 2 }} title="tree levels: reduce ↑ then broadcast ↓">
                  {Array.from({ length: 2 * L }, (_, k) => (
                    <span key={k} style={{ width: 6, height: 6, borderRadius: k < L ? 2 : 7, background: k === dot ? c : on && k < dot ? `${c}88` : done ? `${T.wire}66` : T.rule }} />
                  ))}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Attention weights over every key token, as each stage leaves them:
//   local/max:   e^{s_j - l_i}   each shard normalised on its own (sums to 1 per GPU)
//   rescale/sum: e^{s_j - m}     shared shift, unnormalised; shard sums = d_i
//   final:       e^{s_j - L}     true softmax over all S keys (sums to 1)
// A faint outline shows the previous stage's bars so the change is visible.
export function TreeWeights({ data, phase, state, sel }) {
  const { N, gpus } = data;
  const stage = stageOf(phase);
  const W = 700, H = 210, padL = 8, padT = 46, plotH = 112, gap = 12;
  const nb = N * PER;
  const bw = (W - padL * 2 - gap * (N - 1)) / nb;
  const bx = (i, j) => padL + i * (PER * bw + gap) + j * bw;
  const m = Math.max(...gpus.map((g) => g.lse));
  const local = (i, j) => Math.exp(gpus[i].scores[j] - gpus[i].lse);
  const shifted = (i, j) => Math.exp(gpus[i].scores[j] - m);
  const exact = (i, j) => Math.exp(gpus[i].scores[j] - data.lseAll);
  const show = stage !== "init";
  const val = stage === "local" || stage === "max" ? local : stage === "rescale" || stage === "sum" ? shifted : stage === "final" ? exact : () => 0;
  const ghost = stage === "rescale" ? local : stage === "final" ? shifted : null;
  const y = (v) => padT + plotH * (1 - Math.min(1, v));
  const cov = state.gpus[sel].cov;
  const isAR = phase.type === "up" || phase.type === "down";
  const inCov = (i) => !isAR || (cov & (1 << i));
  const covered = covList(cov, N);
  const groupNote = (i) => {
    if (stage === "local" || stage === "max") return `ℓ=${fmt(gpus[i].lse)}`;
    if (stage === "rescale" || stage === "sum") return `d=${fmt(Math.exp(gpus[i].lse - m))}`;
    if (stage === "final") return `Σ=${fmt(gpus[i].scores.reduce((a, s) => a + Math.exp(s - data.lseAll), 0))}`;
    return "";
  };
  const caption = {
    init: "No attention yet. Every GPU has the same query token and its own 3 keys.",
    local: "Each GPU's softmax over its own 3 keys: every shard sums to 1, so the shards disagree on scale.",
    max: `Still the local weights. GPU ${sel}'s buffer covers the shards in colour; lse_max over them = ${fmt(state.gpus[sel].mx)}.`,
    rescale: `Bars become e^(s−m) with the shared m = ${fmt(m)}: every shard is now on one scale (outline = before). d_i is each shard's total.`,
    sum: `GPU ${sel}'s buffer covers the shards in colour: den = Σ d over them = ${fmt(state.gpus[sel].den)}.`,
    final: "n/d: the true softmax over all keys (sums to 1). Outline = the unnormalised bars from step 3.",
  }[stage];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
        {/* y = 1 guide */}
        <line x1={padL} x2={W - padL} y1={y(1)} y2={y(1)} stroke={T.rule} strokeDasharray="3 4" />
        <text x={W - padL} y={y(1) - 4} textAnchor="end" fontFamily={mono} fontSize="9" fill={T.dim}>1.0</text>
        <line x1={padL} x2={W - padL} y1={y(0)} y2={y(0)} stroke={T.rule} />

        {/* bracket over the shards in the followed GPU's buffer */}
        {isAR && covered.length > 0 && (() => {
          const x1 = bx(covered[0], 0), x2 = bx(covered[covered.length - 1], PER);
          const c = TREE_PHASE[phase.type].color;
          return (
            <g>
              <path d={`M${x1},${padT - 10} L${x1},${padT - 16} L${x2},${padT - 16} L${x2},${padT - 10}`} fill="none" stroke={c} strokeWidth={1.6} />
              <text x={(x1 + x2) / 2} y={padT - 21} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={c}>
                {`GPU ${sel}'s ${stage === "max" ? "lse_max" : "den"} covers GPU${covered.length > 1 ? "s" : ""} ${covered.length > 2 ? `${covered[0]}–${covered[covered.length - 1]}` : covered.join(", ")}`}
              </text>
            </g>
          );
        })()}

        {gpus.map((g, i) => (
          <g key={i} opacity={inCov(i) ? 1 : 0.28} style={{ transition: "opacity .3s" }}>
            {Array.from({ length: PER }, (_, j) => {
              const x = bx(i, j) + 1.5, w = bw - 3;
              const v = show ? val(i, j) : 0;
              return (
                <g key={j}>
                  {ghost && <rect x={x} y={y(ghost(i, j))} width={w} height={y(0) - y(ghost(i, j))} fill="none" stroke={DEV_COLORS[i]} strokeDasharray="2 2" strokeWidth={1} opacity={0.7} />}
                  <rect x={x} y={y(v)} width={w} height={y(0) - y(v)} rx={2} fill={DEV_COLORS[i]} opacity={i === sel ? 1 : 0.75} style={{ transition: "all .45s cubic-bezier(.3,.7,.3,1)" }} />
                  <text x={x + w / 2} y={y(0) + 12} textAnchor="middle" fontFamily={mono} fontSize={N > 4 ? 8 : 9.5} fill={T.dim}>t{PER * i + j}</text>
                </g>
              );
            })}
            <text x={bx(i, 0) + (PER * bw) / 2} y={y(0) + 27} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight={i === sel ? 700 : 500} fill={DEV_COLORS[i]}>GPU {i}</text>
            {show && <text x={bx(i, 0) + (PER * bw) / 2} y={y(0) + 40} textAnchor="middle" fontFamily={mono} fontSize={N > 4 ? 8.5 : 10} fill={T.soft}>{groupNote(i)}</text>}
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 12.5, color: T.dim, lineHeight: 1.5, marginTop: 4 }}>{caption}</div>
    </div>
  );
}
