import { CFG, MODEL } from "./moeModel";
import { STEPS } from "./steps";
import { expand } from "./Stepper";

// Line-level program for the GShard tab. Most steps expand one item per code
// line; the two `for s in 1..S` passes of Algorithm 1 are unrolled so a viewer
// walks token by token through the lines that actually execute.

const { C } = CFG;
const f2 = (x) => x.toFixed(2).replace(/^0/, "");
const byId = Object.fromEntries(STEPS.map((s) => [s.id, s]));

function pass1(gm) {
  const base = byId.pass1;
  const out = [];
  const counts = (n, cur, ph) => {
    const c = Array(CFG.E).fill(0);
    gm.toks.forEach((k) => { if (k.s < n || (k.s === cur && ph >= 7)) c[k.e1] += 1; });
    return c;
  };
  const mk = (line, loop, caption, first = false, extra = []) => ({
    ...base, sid: "pass1", line, first, loop, caption,
    vars: () => [...extra, ["c_E", `[${counts(loop.n, loop.cur, loop.ph).join(", ")}]`], ["C", `${C}`]],
  });
  out.push(mk(0, { n: 0, cur: -1, ph: 0 }, "the counters start at zero, one per expert", true));
  gm.toks.forEach((k) => {
    const ok = k.t1.ok;
    const L = (ph) => ({ n: k.s, cur: k.s, ph });
    const tok = [["token", `x${k.s}`]];
    out.push(mk(1, L(1), `token x${k.s} is next in the loop`, true, tok));
    out.push(mk(2, L(2), `x${k.s}: best two experts are E${k.e1} (${f2(k.g1)}) and E${k.e2} (${f2(k.g2)})`, false, tok));
    out.push(mk(3, L(3), `renormalise: g1 = ${f2(k.g1n)}`, false, tok));
    out.push(mk(4, L(4), `bucket E${k.e1} already holds c = ${k.t1.pos}`, false, tok));
    out.push(mk(5, L(5), ok ? `${k.t1.pos} < ${C}: there is room` : `${k.t1.pos} ≥ ${C}: the bucket is full, skip the assignment`, false, [...tok, ["c < C", ok ? "yes" : "no", ok ? null : "bad"]]));
    if (ok) out.push(mk(6, L(6), `x${k.s} takes slot ${k.t1.pos} of E${k.e1}, weight ${f2(k.g1n)}`, false, tok));
    out.push(mk(7, L(7), ok ? `counter c_E[E${k.e1}] becomes ${k.t1.pos + 1}` : `overflow: x${k.s} is not stored, yet the counter still goes to ${k.t1.pos + 1}`, false, [...tok, ...(ok ? [] : [["overflow", `x${k.s}→E${k.e1}`, "bad"]])]));
  });
  return out;
}

function pass2(gm) {
  const base = byId.pass2;
  const out = [];
  const counts = (n, cur, ph) => {
    const c = [...gm.cPass1];
    gm.toks.forEach((k) => { if (k.s < n || (k.s === cur && ph >= 8)) c[k.e2] += 1; });
    return c;
  };
  const mk = (line, loop, caption, first = false, extra = []) => ({
    ...base, sid: "pass2", line, first, loop, caption,
    vars: () => [...extra, ["c_E", `[${counts(loop.n, loop.cur, loop.ph).join(", ")}]`], ["C", `${C}`]],
  });
  gm.toks.forEach((k) => {
    const t = k.t2;
    const L = (ph) => ({ n: k.s, cur: k.s, ph });
    const tok = [["token", `x${k.s}`]];
    out.push(mk(0, L(1), `token x${k.s} is next in the loop`, true, tok));
    out.push(mk(1, L(2), `x${k.s}: second choice is E${k.e2} (${f2(k.g2)})`, false, tok));
    out.push(mk(2, L(3), `renormalise: g2 = ${f2(k.g2n)}`, false, tok));
    out.push(mk(3, L(4), `roll the dice: rnd = ${f2(k.rnd)}`, false, [...tok, ["rnd", f2(k.rnd)]]));
    out.push(mk(4, L(5), `bucket E${k.e2} already holds c = ${t.pos}`, false, tok));
    const why = !t.capOk ? `${t.pos} ≥ ${C}: bucket full` : !t.diceOk ? `2·g2 = ${f2(2 * k.g2n)} ≤ rnd = ${f2(k.rnd)}: dice say skip` : `c < C and 2·g2 = ${f2(2 * k.g2n)} > rnd = ${f2(k.rnd)}`;
    out.push(mk(5, L(6), t.ok ? `keep it: ${why}` : `drop the 2nd choice: ${why}`, false, [...tok, ["condition", t.ok ? "true" : "false", t.ok ? null : "bad"]]));
    if (t.ok) out.push(mk(6, L(7), `x${k.s} takes slot ${t.pos} of E${k.e2}, weight ${f2(k.g2n)}`, false, tok));
    out.push(mk(7, L(8), `counter c_E[E${k.e2}] becomes ${t.pos + 1}`, false, tok));
  });
  return out;
}

export function buildFlat(gm) {
  const bound = STEPS.map((st) => ({ ...st, vars: () => st.vars(gm) }));
  const items = expand(bound);
  return items.flatMap((it) => (it.sid === "pass1" ? pass1(gm) : it.sid === "pass2" ? pass2(gm) : [it]));
}
