import { useMemo, useState } from "react";
import katex from "katex";
import { T, mono, btn } from "./theme";
import { CFG, SEED, EXPERT_COLORS as EC, buildModel } from "./moeModel";
import Stepper, { expand } from "./Stepper";
import GroupScene from "./GroupScene";

// A short, standalone explainer of the numbers everyone asks about in top-2
// MoE routing: the capacity factor, how each token's two experts are chosen,
// when and why a choice overflows, how balanced the result is (CV), and
// exactly how l_aux is computed from the same counters. Same toy batch as
// the "MoE layer" tab (same seed), just re-run at whatever capacity factor
// is picked here — nothing else changes, so any difference in overflow/CV
// is purely the capacity effect.
//
// Fill/overflow steps are unrolled ONE LINE PER TOKEN, same loop vocabulary
// GroupScene already understands from the "MoE layer" tab (phases 1,4,5,6,7
// for the 1st choice; 1..8 for the 2nd) — so stepping through is exactly as
// slow, and exactly as granular, as the GShard tab's own pass1/pass2.

const { G, S, E } = CFG;
const FACTORS = [0.5, 1, 2, 3]; // → C = ceil(factor · S/E)
const f2 = (x) => x.toFixed(2);

function katexHtml(tex) {
  try {
    return katex.renderToString(tex, { throwOnError: false, strict: false });
  } catch {
    return tex;
  }
}
const Eq = ({ tex, style }) => <span style={style} dangerouslySetInnerHTML={{ __html: katexHtml(tex) }} />;

// The five formulas this page walks through, always visible so you can check
// a step's numbers against the symbol they came from.
const FORMULAS = [
  { k: "capacity", label: "capacity factor", tex: "C=\\left\\lceil\\dfrac{\\text{capacity\\_factor}\\cdot S}{E}\\right\\rceil" },
  { k: "fill1", label: "expert assignment", tex: "g_1,e_1,g_2,e_2=\\mathrm{top\\_2}\\big(\\mathrm{softmax}(W_g\\!\\cdot\\!x_s)\\big),\\quad \\tilde g_i=\\dfrac{g_i}{g_1+g_2}" },
  { k: "fill1b", label: "overflow (1st choice)", tex: "\\text{keep at }e_1 \\iff c_{e_1} < C \\quad\\text{(first-come-first-served, token order)}" },
  { k: "fill2", label: "overflow (2nd choice)", tex: "\\text{keep at }e_2 \\iff c_{e_2} < C \\ \\text{and}\\ 2\\tilde g_2 > u,\\quad u\\sim\\mathrm{Uniform}(0,1)" },
  { k: "cv", label: "load balance", tex: "CV=\\dfrac{\\mathrm{std}(\\text{load}_E)}{\\mathrm{mean}(\\text{load}_E)}" },
  { k: "me", label: "m_e (mean gate)", tex: "m_e=\\dfrac{1}{S}\\sum_{s=1}^{S} g_{s,e},\\quad g_{s,\\cdot}=\\mathrm{softmax}(W_g\\!\\cdot\\!x_s)" },
  { k: "aux", label: "ℓ_aux", tex: "\\ell_{aux}=\\dfrac{1}{E}\\sum_e \\dfrac{c_e}{S}\\,m_e" },
];

const BLOCKS = [
  { id: "capacity", label: "capacity factor", sub: "buffer size per expert, per group", code: ["C = ceil(capacity_factor · S / E)   # round UP: never below the fair share"] },
  {
    id: "fill1", label: "1st choice: assign & fill", sub: "one token at a time", loop: true,
    code: [
      "c_E = 0   # one counter per expert",
      "for s in 1..S:",
      "  g1,e1,g2,e2 = top_2(softmax(w_g · x_s))",
      "  g1 = g1 / (g1 + g2)",
      "  c = c_E[e1]",
      "  if c < C:",
      "    keep x_s at slot c of E[e1]",
      "  c_E[e1] = c + 1",
    ],
  },
  {
    id: "fill2", label: "2nd choice: dice & fill", sub: "one token at a time", loop: true,
    code: [
      "for s in 1..S:",
      "  g1,e1,g2,e2 = top_2(softmax(w_g · x_s))",
      "  g2 = g2 / (g1 + g2)",
      "  rnd = uniform(0, 1)",
      "  c = c_E[e2]",
      "  if c < C and 2·g2 > rnd:",
      "    keep x_s at slot c of E[e2]",
      "  c_E[e2] = c + 1",
    ],
  },
  { id: "cv", label: "load balance", sub: "coefficient of variation", code: ["CV = std(load_E) / mean(load_E)   # 0 = perfectly even"] },
  { id: "mean", label: "m_e: mean gate", sub: "average each column over the S tokens", code: ["m_E = mean(gates, over S)   # [E], each entry = column average"] },
  { id: "aux", label: "ℓ_aux, term by term", sub: "the differentiable stand-in", code: ["ℓ_aux = (1/E) · Σ_e (c_e/S) · m_e"] },
];

// One line per token-phase, mirroring gshardFlat.js's pass1/pass2 exactly
// (same phase numbers GroupScene already reads) but driven by THIS page's
// own capacity C rather than the fixed CFG.C the "MoE layer" tab uses.
function fill1Steps(gm, C) {
  const out = [];
  const counts = (n, cur, ph) => {
    const c = Array(E).fill(0);
    gm.toks.forEach((k) => { if (k.s < n || (k.s === cur && ph >= 7)) c[k.e1] += 1; });
    return c;
  };
  const mk = (line, loop, caption, first, extra) => ({
    sid: "fill1", block: "fill1", line, first: !!first, loop, caption,
    vars: () => [...extra, ["c_E", `[${counts(loop.n, loop.cur, loop.ph).join(", ")}]`], ["C", `${C}`]],
  });
  out.push(mk(0, { n: 0, cur: -1, ph: 0 }, "the counters start at zero, one per expert", true, []));
  gm.toks.forEach((k) => {
    const ok = k.t1.ok;
    const L = (ph) => ({ n: k.s, cur: k.s, ph });
    const tok = [["token", `x${k.s}`]];
    out.push(mk(1, L(1), `token x${k.s} is next in the loop`, true, tok));
    out.push(mk(2, L(2), `x${k.s}: best two experts are E${k.e1} (${f2(k.g1)}) and E${k.e2} (${f2(k.g2)})`, false, tok));
    out.push(mk(3, L(3), `renormalise: g̃1 = ${f2(k.g1n)}`, false, tok));
    out.push(mk(4, L(4), `bucket E${k.e1} already holds c = ${k.t1.pos}`, false, tok));
    out.push(mk(5, L(5), ok ? `${k.t1.pos} < ${C}: there is room` : `${k.t1.pos} ≥ ${C}: the bucket is full, skip`, false, [...tok, ["c < C", ok ? "yes" : "no", ok ? null : "bad"]]));
    if (ok) out.push(mk(6, L(6), `x${k.s} takes slot ${k.t1.pos} of E${k.e1}, weight ${f2(k.g1n)}`, false, tok));
    out.push(mk(7, L(7), ok ? `counter c_E[E${k.e1}] becomes ${k.t1.pos + 1}` : `overflow: x${k.s} is not stored, yet the counter still goes to ${k.t1.pos + 1}`, false, [...tok, ...(ok ? [] : [["overflow", `x${k.s}→E${k.e1}`, "bad"]])]));
  });
  return out;
}

function fill2Steps(gm, C) {
  const out = [];
  const counts = (n, cur, ph) => {
    const c = [...gm.cPass1];
    gm.toks.forEach((k) => { if (k.s < n || (k.s === cur && ph >= 8)) c[k.e2] += 1; });
    return c;
  };
  const mk = (line, loop, caption, first, extra) => ({
    sid: "fill2", block: "fill2", line, first: !!first, loop, caption,
    vars: () => [...extra, ["c_E", `[${counts(loop.n, loop.cur, loop.ph).join(", ")}]`], ["C", `${C}`]],
  });
  gm.toks.forEach((k) => {
    const t = k.t2;
    const L = (ph) => ({ n: k.s, cur: k.s, ph });
    const tok = [["token", `x${k.s}`]];
    out.push(mk(0, L(1), `token x${k.s} is next in the loop`, true, tok));
    out.push(mk(1, L(2), `x${k.s}: second choice is E${k.e2} (${f2(k.g2)})`, false, tok));
    out.push(mk(2, L(3), `renormalise: g̃2 = ${f2(k.g2n)}`, false, tok));
    out.push(mk(3, L(4), `roll the dice: rnd = ${f2(k.rnd)}`, false, [...tok, ["rnd", f2(k.rnd)]]));
    out.push(mk(4, L(5), `bucket E${k.e2} already holds c = ${t.pos}`, false, tok));
    const why = !t.capOk ? `${t.pos} ≥ ${C}: bucket full` : !t.diceOk ? `2·g̃2 = ${f2(2 * k.g2n)} ≤ rnd = ${f2(k.rnd)}: dice say skip` : `c < C and 2·g̃2 = ${f2(2 * k.g2n)} > rnd = ${f2(k.rnd)}`;
    out.push(mk(5, L(6), t.ok ? `keep it: ${why}` : `drop the 2nd choice: ${why}`, false, [...tok, ["condition", t.ok ? "true" : "false", t.ok ? null : "bad"]]));
    if (t.ok) out.push(mk(6, L(7), `x${k.s} takes slot ${t.pos} of E${k.e2}, weight ${f2(k.g2n)}`, false, tok));
    out.push(mk(7, L(8), `counter c_E[E${k.e2}] becomes ${t.pos + 1}`, false, tok));
  });
  return out;
}

function Stat({ label, value, color = T.ink }) {
  return (
    <div>
      <div style={{ fontSize: 9.5, color: T.dim, letterSpacing: "0.08em" }}>{label.toUpperCase()}</div>
      <div style={{ fontFamily: mono, fontSize: 20, fontWeight: 700, color }}>{value}</div>
    </div>
  );
}

function Formulas({ active }) {
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 8 }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>FORMULAS ON THIS PAGE</div>
      {FORMULAS.map((f) => (
        <div key={f.k} style={{ display: "flex", gap: 10, alignItems: "baseline", padding: "4px 6px", borderRadius: 6, background: (active === f.k || (active === "aux" && f.k === "me")) ? `${T.wire}14` : "transparent" }}>
          <span style={{ fontFamily: mono, fontSize: 10, color: (active === f.k || (active === "aux" && f.k === "me")) ? T.wire : T.dim, width: 118, flexShrink: 0 }}>{f.label}</span>
          <Eq tex={f.tex} style={{ fontSize: 13.5, color: (active === f.k || (active === "aux" && f.k === "me")) ? T.ink : T.soft, overflowX: "auto" }} />
        </div>
      ))}
    </div>
  );
}

function CVScene({ load }) {
  const mean = load.reduce((a, x) => a + x, 0) / E;
  const variance = load.reduce((a, x) => a + (x - mean) ** 2, 0) / E;
  const std = Math.sqrt(variance);
  const cv = mean > 0 ? std / mean : 0;
  const hi = Math.max(...load, 1);
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <Stat label="mean load" value={f2(mean)} />
        <Stat label="std" value={f2(std)} />
        <Stat label="CV" value={f2(cv)} color={cv < 0.15 ? T.accent : cv < 0.4 ? T.wire : T.bad} />
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "flex-end", height: 90 }}>
        {load.map((v, e) => (
          <div key={e} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, width: 44 }}>
            <div style={{ width: 28, height: `${(v / hi) * 70}px`, background: EC[e], borderRadius: "4px 4px 0 0" }} />
            <span style={{ fontFamily: mono, fontSize: 10.5, color: EC[e] }}>E{e}</span>
            <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{v}</span>
          </div>
        ))}
        <div style={{ borderLeft: `1.5px dashed ${T.dim}`, height: 70, position: "relative", marginLeft: 6 }}>
          <span style={{ position: "absolute", left: 6, top: 70 - (mean / hi) * 70 - 6, fontSize: 9.5, color: T.dim, whiteSpace: "nowrap" }}>mean {f2(mean)}</span>
        </div>
      </div>
      <div style={{ fontSize: 10.5, color: T.dim }}>load = tokens actually dispatched to each expert, summed over all {G} groups</div>
    </div>
  );
}

export default function CapacityView() {
  const [factor, setFactor] = useState(2);
  const [g, setG] = useState(0);
  const fair = (factor * S) / E; // exact slots per expert at this factor, before rounding up
  const capNote = `${factor}× fair share = ${f2(fair)}${Number.isInteger(fair) ? "" : ", rounded up"}`;
  const C = Math.max(1, Math.ceil((factor * S) / E));
  const model = useMemo(() => buildModel(SEED, E, C), [factor]);
  const gm = model.groups[g];

  const mTop = gm.m.indexOf(Math.max(...gm.m)); // the expert the router likes most: its column makes the clearest example
  const simpleSteps = [
    { id: "capacity", block: "capacity", lines: [0], caption: `factor ${factor}× → C = ceil(${factor} · ${S}/${E}) = ceil(${f2(fair)}) = ${C} slots per expert, per group`, vars: () => [["C", `${C}`], ["factor", `${factor}×`]] },
    { id: "cv", block: "cv", lines: [0], caption: "how evenly the real, post-capacity load lands across experts", vars: () => [["load", `[${model.load.join(", ")}]`]] },
    { id: "mean", block: "mean", lines: [0], caption: `m_e = column average of the gates: e.g. m_${mTop} = (${gm.gates.map((r) => f2(r[mTop])).join(" + ")}) / ${S} = ${f2(gm.m[mTop])}`, vars: () => [["m_E", `[${gm.m.map((x) => f2(x)).join(", ")}]`]] },
    { id: "aux", block: "aux", lines: [0], caption: `group ${g}: the exact sum that produces ℓ_aux = ${gm.aux.toFixed(3)}`, vars: () => [["ℓ_aux", gm.aux.toFixed(3)]] },
  ];
  const byId = Object.fromEntries(expand(simpleSteps).map((s, i) => [`${s.sid}:${i}`, s]));
  const flat = [
    ...expand([simpleSteps[0]]),
    ...fill1Steps(gm, C),
    ...fill2Steps(gm, C),
    ...expand([simpleSteps[1]]),
    ...expand([simpleSteps[2]]),
    ...expand([simpleSteps[3]]),
  ];
  void byId; // kept for clarity of intent; Stepper only needs the flat array below

  const activeFormula = (sid) => (sid === "capacity" ? "capacity" : sid === "fill1" ? "fill1" : sid === "fill2" ? "fill2" : sid === "cv" ? "cv" : sid === "mean" ? "me" : "aux");

  return (
    <Stepper
      title="Capacity, overflow, CV &amp; ℓ_aux"
      subtitle={`gate: softmax(w_g·x) · same ${G}×${S} toy batch as "MoE layer" · E=${E} experts`}
      blocks={BLOCKS}
      flat={flat}
      header={() => (
        <>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>capacity factor</span>
            {FACTORS.map((v) => (
              <button key={v} onClick={() => setFactor(v)} style={{ ...btn(v === factor, false), height: 28, padding: "0 10px" }}>{v}×</button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>group</span>
            {Array.from({ length: G }, (_, k) => (
              <button key={k} onClick={() => setG(k)} style={{ ...btn(k === g, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
        </>
      )}
      scene={(item) =>
        // Reuse the exact "MoE layer" tab scene: tickets fill expert buffers in
        // token order and bounce onto the red overflow stack past C, one token
        // at a time (fill1/fill2 carry a `loop` so GroupScene animates each
        // phase individually) — just replayed at whatever capacity factor is
        // picked above. stage 3 = 1st choice fills the buffers, stage 5 = 2nd
        // choice (capacity + dice), stage 4 = the aux-loss sum building term by term.
        item.sid === "capacity" ? <GroupScene gm={gm} stage={3} C={C} capNote={capNote} />
          : item.sid === "fill1" ? <GroupScene gm={gm} stage={3} C={C} capNote={capNote} loop={item.loop} line={item.line} />
          : item.sid === "fill2" ? <GroupScene gm={gm} stage={5} C={C} capNote={capNote} loop={item.loop} line={item.line} />
          : item.sid === "cv" ? <CVScene load={model.load} />
          : item.sid === "mean" ? <GroupScene gm={gm} stage={1} C={C} capNote={capNote} />
          : <GroupScene gm={gm} stage={4} C={C} capNote={capNote} />
      }
      extra={(item) => <Formulas active={activeFormula(item.sid)} />}
      legend={<span>C = per-expert, per-group buffer · ℓ_aux is added to the loss as <span style={{ color: T.wire }}>L = ℓ_nll + k·ℓ_aux</span> · gate is softmax everywhere on this page (see "Gating functions" for alternatives)</span>}
    />
  );
}
