import { T, mono } from "./theme";
import { CFG, EXPERT_COLORS as EC } from "./moeModel";
import { Tex } from "./tex";
import { useProgress, ease, lerp, clamp01 } from "./anim";

// Algorithm 1 for ONE group on one device. Rows are the group's tokens, the
// four gate bars per row are g_SE, the buckets on the right are the expert
// buffers (capacity C). Tickets (solid = 1st choice, ring = 2nd choice) fly
// from the token row into a bucket slot, or bounce onto the red overflow
// stack when c_e >= C.
//
// stage: 0 gates · 1 mean · 2 top-2 · 3 pass 1 · 4 aux loss · 5 pass 2 · 6 result

const { S, E } = CFG;
const W = 860, H = 490;
const ROW0 = 62, ROWH = 38;
const GX = 92, GW = 58;
const T1X = 348, T2X = 382, INFOX = 412;
const BX = 520, BXW = 84, BOT = 300, SLOT = 28;

const rowY = (s) => ROW0 + s * ROWH;
const f2 = (x) => x.toFixed(2).replace(/^0/, "");

function Ticket({ x, y, e, label, hollow, bad, on, delay }) {
  return (
    <g
      style={{
        transform: `translate(${x}px, ${y}px)`,
        opacity: on ? 1 : 0,
        transition: `transform .75s cubic-bezier(.3,.7,.3,1) ${delay}ms, opacity .35s ${delay}ms`,
      }}
    >
      <circle r={10} fill={hollow ? T.bg : EC[e]} stroke={bad ? T.bad : EC[e]} strokeWidth={hollow ? 2.2 : bad ? 2.4 : 0} />
      <text textAnchor="middle" y={4} fontFamily={mono} fontSize="11" fontWeight="700" fill={hollow ? EC[e] : "#0A0E14"}>
        {label}
      </text>
    </g>
  );
}

// loop = { n, cur, ph }: tokens before n are finished, token cur is mid-iteration
// at phase ph (pass 1: 1 row · 4 read c · 5 compare · 6 store · 7 count;
// pass 2: 1 row · 4 dice · 5 read c · 6 condition · 7 store · 8 count)
export default function GroupScene({ gm, stage, loop, line, C: Cprop }) {
  const C = Cprop ?? CFG.C; // overridable so a variable capacity factor can replay this exact scene
  const slotXY = (e, k) => [BX + e * BXW + 35, BOT - (k + 1) * SLOT + 13];
  const overXY = (e, j) => [BX + e * BXW + 35, BOT - C * SLOT - 16 - Math.min(j, 3) * 24];
  const lp = loop && (stage === 3 || stage === 5) ? loop : null;
  // the picture id: replays when it changes, holds still between lines that share one picture
  const pkey = lp ? `${stage}:${lp.cur}:${lp.ph}` : stage === 2 ? `2:${line}` : `${stage}`;
  const p = useProgress(pkey);
  const pe = ease(p);
  // centred logits z = ln g - mean(ln g): what the softmax of stage 0 turns into gates
  const logits = gm.gates.map((row) => { const l = row.map(Math.log); const mu = l.reduce((a, v) => a + v, 0) / l.length; return l.map((v) => v - mu); });
  const done = (k, ph) => lp && (k.s < lp.n || (k.s === lp.cur && lp.ph >= ph));
  let counts = stage < 3 ? Array(E).fill(0) : stage < 5 ? gm.cPass1 : gm.cFinal;
  if (lp && stage === 3) { counts = Array(E).fill(0); gm.toks.forEach((k) => { if (done(k, 7)) counts[k.e1] += 1; }); }
  if (lp && stage === 5) { counts = [...gm.cPass1]; gm.toks.forEach((k) => { if (done(k, 8)) counts[k.e2] += 1; }); }
  const showT1 = stage >= 2;
  const curTok = lp && lp.cur >= 0 ? gm.toks[lp.cur] : null;
  // the counter that this very line increments ticks up from its old value
  let tick = -1;
  if (curTok && lp) { if (stage === 3 && lp.ph === 7) tick = curTok.e1; if (stage === 5 && lp.ph === 8) tick = curTok.e2; }
  const shownCount = (c, ei) => (ei === tick ? Math.round(lerp(c - 1, c, pe)) : c);
  // running mean over the first nTok tokens while line "m_E = mean(g)" sweeps down the rows
  const nTok = stage === 1 ? Math.max(1, Math.ceil(pe * S)) : S;
  const mShown = (ei) => (stage === 1 ? gm.gates.slice(0, nTok).reduce((a, r) => a + r[ei], 0) / nTok : gm.m[ei]);
  const hotE = curTok && (stage === 3 ? (lp.ph >= 4 ? curTok.e1 : -1) : lp.ph >= 5 ? curTok.e2 : -1);

  // where does each ticket sit at this stage
  const t1Pos = (k) => {
    if (stage < 3) return [T1X, rowY(k.s)];
    if (lp && stage === 3 && !done(k, k.t1.ok ? 6 : 7)) return [T1X, rowY(k.s)];
    return k.t1.ok ? slotXY(k.e1, k.t1.pos) : overXY(k.e1, k.t1.pos - C);
  };
  const t2Pos = (k) => {
    if (stage < 5) return [T2X, rowY(k.s)];
    if (lp && !done(k, k.t2.ok ? 7 : 8)) return [T2X, rowY(k.s)];
    if (k.t2.ok) return slotXY(k.e2, k.t2.pos);
    return k.t2.capOk ? [T2X, rowY(k.s)] : overXY(k.e2, k.t2.pos - C);
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      {/* column headers */}
      <g fontFamily={mono} fontSize="10" fill={T.dim} letterSpacing="0.1em">
        <text x={20} y={34}>TOKEN</text>
        <text x={GX} y={34}>GATE g · ONE BAR PER EXPERT</text>
        {showT1 && <text x={T1X - 12} y={34}>TOP-2</text>}
        {stage >= 5 && <text x={INFOX} y={34}>{stage >= 6 ? "WEIGHTS" : "DICE"}</text>}
        <text x={BX} y={34}>EXPERT BUFFERS · CAP C = {C}</text>
      </g>
      {Array.from({ length: E }, (_, e) => (
        <text key={e} x={GX + e * GW + (GW - 8) / 2} y={52} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill={EC[e]}>E{e}</text>
      ))}

      {/* token rows: gate bars */}
      {gm.toks.map((k) => {
        const y = rowY(k.s);
        const dropped = stage >= 6 && k.final.length === 0;
        return (
          <g key={k.s} opacity={dropped ? 0.9 : 1}>
            {curTok && curTok.s === k.s && <rect x={12} y={y - 16} width={398} height={32} rx={8} fill="none" stroke={T.accent} strokeWidth={1.6} />}
            <rect x={20} y={y - 11} width={44} height={22} rx={5} fill={T.well} stroke={dropped ? T.bad : T.rule} strokeDasharray={dropped ? "3 3" : "none"} />
            <text x={42} y={y + 4} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="600" fill={T.ink}>x{k.s}</text>
            {gm.gates[k.s].map((g, e) => {
              const top = e === k.e1 || e === k.e2;
              const lit = stage < 2 || top;
              const z = logits[k.s][e];
              const gv = stage === 0 ? lerp(z, g, pe) : g; // logit z -> softmax gate
              const frac = stage === 0 ? lerp(clamp01(0.5 + z / 4), g, pe) : g;
              const op = stage === 2 ? lerp(0.9, lit ? 0.9 : 0.22, pe) : lit ? 0.9 : 0.22;
              return (
                <g key={e}>
                  <rect x={GX + e * GW} y={y - 8} width={GW - 8} height={16} rx={3} fill={T.well} />
                  <rect
                    x={GX + e * GW} y={y - 8} width={(GW - 8) * frac} height={16} rx={3} fill={EC[e]}
                    opacity={op}
                  />
                  <text
                    x={GX + e * GW + (frac > 0.45 ? 4 : (GW - 8) * frac + 4)} y={y + 3.5} fontFamily={mono} fontSize="9.5" fontWeight="600"
                    fill={frac > 0.45 && lit ? "#0A0E14" : T.soft}
                  >{f2(gv)}</text>
                </g>
              );
            })}

            {/* g1 / g2: raw, then renormalised g / (g1 + g2) */}
            {(() => {
              const second = stage === 5;
              const show = stage === 2 || (lp && k.s === lp.cur && lp.ph >= 2 && lp.ph <= 3);
              if (!show) return null;
              const norm = stage === 2 ? line === 1 : lp.ph === 3;
              const fresh = stage === 2 ? line === 0 : lp.ph === 2;
              const a = (raw, n) => (fresh ? raw * pe : norm ? lerp(raw, n, pe) : raw); // count up / morph
              return (
                <text x={INFOX} y={y + 4} fontFamily={mono} fontSize="10.5">
                  {!second && <tspan fill={EC[k.e1]}>{"g\u2081 "}{f2(a(k.g1, k.g1n))}</tspan>}
                  {!second && stage === 2 && <tspan fill={EC[k.e2]}>{"  g\u2082 "}{f2(a(k.g2, k.g2n))}</tspan>}
                  {second && <tspan fill={EC[k.e2]}>{"g\u2082 "}{f2(a(k.g2, k.g2n))}</tspan>}
                </text>
              );
            })()}

            {/* dice / weights column */}
            {stage === 5 && (!lp || done(k, 6) || (k.s === lp.cur && lp.ph >= 4)) && (() => {
              const live = lp && !done(k, 6);
              const flick = lp && k.s === lp.cur && lp.ph === 4 && pe < 0.8 ? ((Math.floor(p * 14) * 0.6180339) % 1) : k.rnd; // the dice tumble, then land
              const verdict = lp && k.s === lp.cur && lp.ph === 6 ? pe : 1;
              return (
                <text x={INFOX} y={y + 4} fontFamily={mono} fontSize="10.5" fill={live ? T.wire : k.t2.capOk ? (k.t2.diceOk ? T.accent : T.bad) : T.bad}>
                  {live
                    ? `rnd = ${f2(flick)}${lp.ph >= 5 ? `  c = ${lp.ph === 5 ? Math.round(k.t2.pos * pe) : k.t2.pos}` : ""}`
                    : k.t2.capOk
                      ? <>{`${f2(k.rnd)} ${k.t2.diceOk ? "<" : "\u2265"} ${f2(2 * k.g2n)} `}<tspan fillOpacity={verdict}>{k.t2.diceOk ? "\u2713" : "\u2715"}</tspan></>
                      : <tspan fillOpacity={verdict}>buffer full ✕</tspan>}
                </text>
              );
            })()}
            {stage === 3 && lp && k.s === lp.cur && lp.ph >= 4 && (
              <text x={INFOX} y={y + 4} fontFamily={mono} fontSize="10.5" fill={lp.ph >= 5 ? (k.t1.ok ? T.accent : T.bad) : T.wire}>
                {lp.ph >= 5
                  ? <>{`${k.t1.pos} ${k.t1.ok ? "<" : "\u2265"} ${C} `}<tspan fillOpacity={lp.ph === 5 ? pe : 1}>{k.t1.ok ? "\u2713" : "\u2715"}</tspan></>
                  : `c = ${lp.ph === 4 ? Math.round(k.t1.pos * pe) : k.t1.pos}`}
              </text>
            )}
            {stage >= 6 && (
              k.final.length === 0 ? (
                <text x={INFOX} y={y + 4} fontFamily={mono} fontSize="10.5" fill={T.bad} fillOpacity={pe}>∅ residual</text>
              ) : (
                <g>
                  {(() => {
                    let off = 0;
                    return k.final.map((f, i) => {
                      const w = 92 * f.w * (stage === 6 ? pe : 1); // weights grow in: G[s,e] = g
                      const r = <rect key={i} x={INFOX + off} y={y - 6} width={w} height={12} rx={3} fill={EC[f.e]} />;
                      off += w;
                      return r;
                    });
                  })()}
                  <text x={INFOX + 98} y={y + 4} fontFamily={mono} fontSize="9" fill={T.dim} fillOpacity={stage === 6 ? pe : 1}>{k.final.length === 2 ? "×2" : "×1"}</text>
                </g>
              )
            )}
          </g>
        );
      })}

      {/* mean row */}
      {stage >= 1 && (
        <g>
          <line x1={20} x2={GX + E * GW - 8} y1={rowY(S - 1) + 24} y2={rowY(S - 1) + 24} stroke={T.rule} />
          <text x={42} y={rowY(S - 1) + 46} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.wire}>m_e</text>
          {stage === 1 && pe < 1 && <rect x={12} y={rowY(nTok - 1) - 16} width={398} height={32} rx={8} fill="none" stroke={T.wire} strokeWidth={1.6} />}
          {gm.m.map((_, e) => {
            const m = mShown(e);
            return (
              <g key={e}>
                <rect x={GX + e * GW} y={rowY(S - 1) + 36} width={GW - 8} height={16} rx={3} fill={T.well} />
                <rect x={GX + e * GW} y={rowY(S - 1) + 36} width={(GW - 8) * m} height={16} rx={3} fill={T.wire} opacity={0.9} />
                <text x={GX + e * GW + (GW - 8) * m + 4} y={rowY(S - 1) + 47.5} fontFamily={mono} fontSize="9.5" fontWeight="600" fill={T.wire}>{f2(m)}</text>
              </g>
            );
          })}
        </g>
      )}

      {/* expert buffers */}
      {Array.from({ length: E }, (_, e) => {
        const x = BX + e * BXW;
        const top = BOT - C * SLOT - 2;
        const over = counts[e] > C;
        return (
          <g key={e}>
            <rect x={x} y={top} width={70} height={C * SLOT + 4} rx={6} fill={T.well} stroke={over ? T.bad : e === hotE ? T.ink : EC[e]} strokeWidth={over || e === hotE ? 2.6 : 1.3} style={{ transition: "stroke .3s" }} />
            {Array.from({ length: C }, (_, k) => (
              <rect key={k} x={x + 5} y={BOT - (k + 1) * SLOT + 2} width={60} height={SLOT - 4} rx={4} fill="none" stroke={EC[e]} strokeOpacity={0.3} strokeDasharray="3 3" />
            ))}
            <text x={x + 35} y={BOT + 22} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={EC[e]}>E{e}</text>
            <text x={x + 35} y={BOT + 40} textAnchor="middle" fontFamily={mono} fontSize={e === tick ? 11.5 + 3 * Math.sin(Math.PI * p) : 11.5} fontWeight="600" fill={over ? T.bad : e === tick ? T.accent : T.soft}>
              c = {shownCount(counts[e], e)}<tspan fill={T.dim} fontWeight="400"> / {C}</tspan>
            </text>
          </g>
        );
      })}

      {/* tickets */}
      {gm.toks.map((k) => {
        const [x1, y1] = t1Pos(k);
        const [x2, y2] = t2Pos(k);
        const dec2 = !lp || stage !== 5 || done(k, 6);
        const gone2 = stage >= 5 && !k.t2.ok && k.t2.capOk && dec2;
        return (
          <g key={k.s}>
            <Ticket x={x1} y={y1} e={k.e1} label={k.s} on={showT1} bad={stage >= 3 && !k.t1.ok && (!lp || stage !== 3 || done(k, 5))} delay={stage === 3 && !lp ? k.s * 110 : 0} />
            <Ticket x={x2} y={y2} e={k.e2} label={k.s} hollow on={showT1 && !gone2} bad={stage >= 5 && !k.t2.ok && dec2} delay={stage === 5 && !lp ? k.s * 110 : 0} />
          </g>
        );
      })}

      {/* aux loss panel: each (c_e/S)·m_e product grows in, expert by expert, and the sum builds */}
      {stage === 4 && (() => {
        const q = Array.from({ length: E }, (_, e) => ease(clamp01((p - e * 0.15) / 0.4)));
        const prods = gm.cPass1.map((c, e) => (c / S) * gm.m[e] * q[e]);
        const total = prods.reduce((a, v) => a + v, 0) / E;
        const terms = prods.map((v, e) => `{\\color{${EC[e]}}${f2(v)}}`).join("+");
        return (
          <g>
            <g fontFamily={mono} fontSize="10.5" fill={T.dim}>
              <rect x={20} y={352} width={12} height={12} rx={2} fill={T.soft} />
              <text x={38} y={362}>c_e / S  (fraction routed)</text>
              <rect x={20} y={374} width={12} height={12} rx={2} fill="none" stroke={T.soft} strokeWidth={1.5} />
              <text x={38} y={384}>m_e  (mean gate)</text>
            </g>
            <Tex x={20} y={402} w={480} h={30} size={13} color={T.accent2} tex={`\\ell_{aux}=\\tfrac{1}{${E}}\\left(${terms}\\right)=\\mathbf{${total.toFixed(3)}}`} />
            <text x={20} y={446} fontFamily={mono} fontSize="11" fill={T.dim}>perfectly balanced = {(1 / (E * E)).toFixed(4)}</text>
            {Array.from({ length: E }, (_, e) => {
              const x = BX + e * BXW;
              const a = (gm.cPass1[e] / S) * q[e], b = gm.m[e] * q[e];
              const sc = 130;
              return (
                <g key={e}>
                  <rect x={x + 8} y={470 - a * sc} width={22} height={a * sc} rx={3} fill={EC[e]} />
                  <rect x={x + 34} y={470 - b * sc} width={22} height={b * sc} rx={3} fill="none" stroke={EC[e]} strokeWidth={2} />
                  <text x={x + 32} y={486} textAnchor="middle" fontFamily={mono} fontSize="10" fill={T.soft}>{f2(a * gm.m[e])}</text>
                </g>
              );
            })}
            <text x={BX} y={346} fontFamily={mono} fontSize="10" fill={T.dim} letterSpacing="0.1em">(c_e/S)·m_e  → SUM / E</text>
          </g>
        );
      })()}
    </svg>
  );
}
