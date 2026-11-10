import { T, mono } from "./theme";
import { EXPERT_COLORS as EC } from "./moeModel";
import { useProgress, ease, lerp, clamp01 } from "./anim";
import { f2 } from "./charts";

// The primary picture for the "All-to-all" tab: the whole MoE block, bottom
// (previous layer) to top (next layer). The real token movement happens here,
// driven by the router decisions of the followed device `sel`:
//   router steps   : the device's tokens leave the Router as chips with their top-2 choices
//   dispatch steps : chips cross the All-to-All dispatch box into the queue under their expert
//   FFN steps      : experts whose queues hold tokens light up (W_up, then W_down)
//   return steps   : chips go back through the All-to-All combine box to the "+"
// Chip = token id tN, coloured by the device hosting the chosen expert; a pink
// outline marks a chip in flight. Hatched experts live on the followed device.

const W = 1000, CX = W / 2, MX = 50, RW = W - 2 * MX;
const ADD2_Y = 46, COMB_Y = 74, BOX_H = 44;
const EXP_Y = 186, EXP_H = 44;
const QY = 238, QP = 22; // queue rows: one per source device
const DISP_Y = 344;
const BAND_Y = DISP_Y + BOX_H, BAND_H = 104;
const ROUTER_Y = BAND_Y + BAND_H, ROUTER_H = 50;
const ADD1_Y = ROUTER_Y + ROUTER_H + 40;
const ATTN_Y = ADD1_Y + 32, ATTN_H = 44;
const H = ATTN_Y + ATTN_H + 50;
const CHW = 24, CHH = 20, LBL = 18; // queue chip size, source label width

const mod = (a, n) => ((a % n) + n) % n;
const names = (ids) => (ids.length ? ids.map((i) => `t${i}`).join(" ") : "none");
const why = (ch, second) => (!ch.capOk && second ? "capacity" : second && !ch.diceOk ? "dice" : "capacity");

function Glow({ x, y, w, h, on }) {
  if (!on) return null;
  return <rect x={x - 7} y={y - 7} width={w + 14} height={h + 14} rx={12} fill={T.wire} fillOpacity={0.3} />;
}

function Box({ x, y, w, h, label, sub: sb, on, right, rightCol }) {
  return (
    <g>
      <Glow x={x} y={y} w={w} h={h} on={on} />
      <rect x={x} y={y} width={w} height={h} rx={9} fill={T.panel} stroke={on ? T.ink : T.rule} strokeWidth={on ? 2.2 : 1.4} />
      {right ? (
        <>
          <text x={x + 16} y={y + h / 2 + 5} fontFamily={mono} fontSize="15" fontWeight="700" fill={T.ink}>{label}</text>
          <text x={x + w - 16} y={y + h / 2 + 5} textAnchor="end" fontFamily={mono} fontSize="13" fontWeight="700" fill={rightCol ?? T.accent}>{right}</text>
        </>
      ) : (
        <>
          <text x={x + w / 2} y={y + h / 2 + (sb ? -3 : 5)} textAnchor="middle" fontFamily={mono} fontSize="15" fontWeight="700" fill={T.ink}>{label}</text>
          {sb && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize="12" fill={T.dim}>{sb}</text>}
        </>
      )}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, dashed, on }) {
  return <path d={`M${x1},${y1} L${x2},${y2}`} fill="none" stroke={on ? T.wire : T.rule} strokeWidth={on ? 2.4 : 1.3} strokeDasharray={dashed ? "4 4" : "none"} markerEnd={`url(#mbm-arrow${on ? "-on" : ""})`} opacity={on ? 1 : 0.8} />;
}

// one token chip; `fly` = pink outline (the vector is in flight)
function Chip({ x, y, id, col, w = CHW, h = CHH, fly, hi, dim, opacity = 1 }) {
  return (
    <g opacity={opacity}>
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={5} fill={col} fillOpacity={dim ? 0.2 : 0.6} stroke={fly ? T.accent2 : hi ? T.accent : col} strokeWidth={fly || hi ? 2.2 : 1.2} />
      <text x={x} y={y + 4.5} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="800" fill={T.ink} stroke={T.bg} strokeWidth="2.4" paintOrder="stroke">t{id}</text>
    </g>
  );
}

// stage ∈ "router" | "dispatch" | "expert" | "combine"
function stageOf(step) {
  if (step.id === "combine" || step.back != null) return "combine";
  if (step.ffn || step.id === "moe") return "expert";
  if (step.id === "start" || step.id === "addr") return "router";
  return "dispatch"; // keep, the D-1 send/recv rounds, done
}

export default function MoEBlockMap({ step, Ed, E, sel, cfg }) {
  const pid = step.id; // picture id: lines before visAt keep the previous picture
  const stage = stageOf(step);
  const e = ease(useProgress(pid, 900));
  const { groups, C } = cfg;
  const D = groups.length;
  const devOf = (x) => Math.floor(x / Ed);
  const cw = RW / E;
  const exL = (i) => MX + i * cw; // expert cell left edge
  const exCx = (i) => exL(i) + cw / 2;
  const slotXY = (s, ex, slot) => [exL(ex) + 2 + LBL + CHW / 2 + slot * (CHW + 2), QY + s * QP + QP / 2 - 1];
  const toksTo = (s, t) => groups[s].toks.filter((tk) => tk.final.some((f) => devOf(f.e) === t)).map((tk) => tk.s);

  const { round, phase, ffn, back } = step;
  const route = pid === "start" || pid === "addr";
  const combine = pid === "combine", keep = pid === "keep";
  const ret = back != null && !combine;
    const hot = keep ? 0 : phase === "send" ? round : ret ? back : null;
  const landed = route ? -1 : keep ? -1 : phase === "send" ? round - 1 : phase === "recv" ? round : D - 1;

  // every surviving (token, expert) ticket of every group
  const tickets = [];
  groups.forEach((g, s) => g.toks.forEach((tk) => tk.final.forEach((f) => tickets.push({ s, id: tk.s, ex: f.e, dev: devOf(f.e), slot: f.slot, w: f.w, r: mod(devOf(f.e) - s, D) }))));

  // the followed token for the combine equation
  const fol = groups[sel].toks.find((tk) => tk.final.length === 2) ?? groups[sel].toks.find((tk) => tk.final.length === 1) ?? null;

  // expert occupancy (tokens that have landed in its queue)
  const has = Array(E).fill(false);
  tickets.forEach((t) => { if (t.r <= landed) has[t.ex] = true; });
  const lit = Array.from({ length: E }, (_, i) => devOf(i) === sel);
  const anyLit = lit.some(Boolean);

  // ---- queue chips (dispatch → landed → returned) ----
  const chips = [];
  if (!route) {
    tickets.forEach((t) => {
      const [qx, qy] = slotXY(t.s, t.ex, t.slot);
      const mine = t.s === sel, base = mine ? 1 : 0.5;
      const key = `${t.s}-${t.id}-${t.ex}`, tag = { s: t.s, ex: t.ex };
      const col = EC[t.dev];
      const recvHi = phase === "recv" && t.dev === sel && t.s === mod(sel - round, D);
      if (combine) {
        const f = mine && fol && fol.s === t.id && fol.final.some((x) => x.e === t.ex);
        if (f) {
          chips.push({ ...tag, key, x: lerp(qx, CX, e), y: lerp(qy, ADD2_Y, e), col, id: t.id, fly: e < 1, opacity: 1 - clamp01((e - 0.9) / 0.1), line: [qx, qy] });
          chips.push({ ...tag, key: key + "d", x: qx, y: qy, col, id: t.id, dim: true, opacity: 0.3 * e });
        } else chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, dim: true, opacity: mine ? 0.35 : 0.18 });
      } else if (ret) {
        if (t.r > back) chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, opacity: base });
        else if (t.r < back) chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, dim: true, opacity: mine ? 0.35 : 0.18 });
        else if (mine) {
          chips.push({ ...tag, key, x: lerp(qx, CX, e), y: lerp(qy, ADD2_Y, e), col, id: t.id, fly: e < 1, opacity: 1 - clamp01((e - 0.9) / 0.1) });
          chips.push({ ...tag, key: key + "d", x: qx, y: qy, col, id: t.id, dim: true, opacity: 0.35 * e });
        } else chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, opacity: lerp(0.5, 0.18, e) });
      } else if (hot != null && t.r === hot) {
        if (mine) chips.push({ ...tag, key, x: lerp(CX, qx, e), y: lerp(ROUTER_Y + 6, qy, e), col, id: t.id, fly: e < 1 });
        else chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, opacity: 0.5 * e });
      } else if (t.r <= landed) chips.push({ ...tag, key, x: qx, y: qy, col, id: t.id, opacity: base, hi: recvHi });
    });
  }

  // ---- text inside the two collective boxes ----
  let dispTxt = null, dispCol = T.accent, combTxt = null, combCol = T.accent;
  if (keep) { dispTxt = `keep · D${sel} → D${sel}: ${names(toksTo(sel, sel))}`; dispCol = EC[sel]; }
  else if (phase === "send") { const t = mod(sel + round, D); dispTxt = `round ${round} · D${sel} → D${t}: ${names(toksTo(sel, t))}`; dispCol = EC[t]; }
  else if (phase === "recv") { const s = mod(sel - round, D); dispTxt = `round ${round} · D${s} → D${sel} landed: ${names(toksTo(s, sel))}`; dispCol = EC[sel]; }
  else if (pid === "done") dispTxt = `D${sel} now holds bundles from all ${D} devices`;
  else if (pid === "moe") dispTxt = "queue per expert, grouped by source device";
  else if (ffn) dispTxt = ffn === "up" ? "queues → W_up   (M → H)" : "queues → W_down   (H → M)";
  if (ret) { const t = mod(sel + back, D); combTxt = `return ${back} · D${t} → D${sel}: ${names(toksTo(sel, t))}`; combCol = EC[t]; }
  else if (combine) combTxt = fol ? `out[t${fol.s}] = ${fol.final.map((f) => `${f2(f.w)}·y(E${f.e})`).join(" + ")}` : `D${sel}: every token dropped`;

  // ---- router step: chips leaving the Router ----
  const routerItems = [];
  if (route) {
    groups[sel].toks.forEach((tk, k) => {
      const colI = Math.floor(k / 3), rowI = k % 3;
      const x0 = MX + colI * 450, y0 = BAND_Y + 6 + rowI * 33 + 14;
      const okc = tk.final[0] ?? null;
      const pills = [[tk.t1, false], [tk.t2, true]].map(([ch, second]) => {
        const dv = devOf(ch.e), stay = dv === sel;
        let txt;
        if (!ch.ok) txt = `E${ch.e} ✕ ${why(ch, second)}`;
        else if (pid === "addr") txt = `E${ch.e} ${f2(ch.w)} → ${stay ? "stays" : "D" + dv}`;
        else txt = `E${ch.e} ${f2(ch.w)}${stay ? " stays" : ""}`;
        return { ch, dv, txt, w: txt.length * 7.4 + 16 };
      });
      routerItems.push({ tk, x: lerp(CX, x0 + 16, e), y: lerp(ROUTER_Y + 10, y0, e), x0, y0, col: okc ? EC[devOf(okc.e)] : T.bad, pills });
    });
  }
  const pillOp = clamp01((e - 0.55) / 0.45);

  const expTagged = (i) => (ffn ? (has[i] ? (ffn === "up" ? "W_up" : "W_down") : "") : "");

  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 14px" }}>
      <div style={{ fontFamily: mono, fontSize: 12, letterSpacing: "0.12em", color: T.dim, marginBottom: 6 }}>HOW ALL-TO-ALL FITS IN THE MoE BLOCK · FOLLOWING DEVICE {sel}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
        <defs>
          <marker id="mbm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={T.rule} /></marker>
          <marker id="mbm-arrow-on" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={T.wire} /></marker>
          {EC.map((c, d) => (
            <pattern key={d} id={`mbm-hatch-${d}`} patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
              <rect width="8" height="8" fill={c} fillOpacity="0.14" />
              <line x1="0" y1="0" x2="0" y2="8" stroke={c} strokeOpacity="0.6" strokeWidth="3" />
            </pattern>
          ))}
        </defs>

        {/* residual skip: branches off add1, bypasses router/dispatch/experts/combine, re-enters at add2 */}
        <path d={`M${CX - 15},${ADD1_Y} L24,${ADD1_Y} L24,${ADD2_Y} L${CX - 15},${ADD2_Y}`} fill="none" stroke={T.rule} strokeWidth={1.3} markerEnd="url(#mbm-arrow)" />
        <text transform={`translate(13 ${(ADD1_Y + ADD2_Y) / 2}) rotate(-90)`} textAnchor="middle" fontFamily={mono} fontSize="12" fill={T.dim}>residual</text>

        {/* previous layer → self-attention → add1 */}
        <text x={CX} y={H - 10} textAnchor="middle" fontFamily={mono} fontSize="12" fill={T.dim}>from previous layer</text>
        <Arrow x1={CX} y1={H - 26} x2={CX} y2={ATTN_Y + ATTN_H} />
        <Box x={CX - 150} y={ATTN_Y} w={300} h={ATTN_H} label="Self-Attention" />
        <Arrow x1={CX} y1={ATTN_Y} x2={CX} y2={ADD1_Y + 15} />
        <circle cx={CX} cy={ADD1_Y} r={15} fill={T.panel} stroke={T.rule} strokeWidth={1.4} />
        <text x={CX} y={ADD1_Y + 6} textAnchor="middle" fontFamily={mono} fontSize="16" fill={T.ink}>+</text>

        {/* add1 → router → (band) → dispatch */}
        <Arrow x1={CX} y1={ADD1_Y - 15} x2={CX} y2={ROUTER_Y + ROUTER_H} on={stage === "router"} />
        <Box x={CX - 150} y={ROUTER_Y} w={300} h={ROUTER_H} label="Router" sub="softmax(w_g·x) → top-2" on={stage === "router"} />
        <Arrow x1={CX} y1={ROUTER_Y} x2={CX} y2={DISP_Y + BOX_H} on={stage === "dispatch"} />
        <Box x={MX} y={DISP_Y} w={RW} h={BOX_H} label="All-to-All dispatch" on={stage === "dispatch"} right={dispTxt} rightCol={dispCol} />

        {/* dispatch → expert queues → experts → combine */}
        {Array.from({ length: E }, (_, i) => {
          const cx = exCx(i), dv = devOf(i), l = lit[i];
          const exOn = (stage === "expert" || ffn) && has[i];
          return (
            <g key={i}>
              <Arrow x1={cx} y1={DISP_Y} x2={cx} y2={EXP_Y + EXP_H} on={stage === "dispatch" && l} dashed={!l} />
              <Glow x={exL(i) + 2} y={EXP_Y} w={cw - 4} h={EXP_H} on={exOn} />
              <rect x={exL(i) + 2} y={EXP_Y} width={cw - 4} height={EXP_H} rx={7} fill={l ? `url(#mbm-hatch-${dv})` : T.panel} stroke={exOn ? T.ink : EC[dv]} strokeWidth={exOn ? 2.4 : 1.4} fillOpacity={1} />
              <text x={cx} y={EXP_Y + (expTagged(i) ? 18 : 28)} textAnchor="middle" fontFamily={mono} fontSize="14" fontWeight="800" fill={T.ink} stroke={T.bg} strokeWidth="3" paintOrder="stroke">E{i}</text>
              {expTagged(i) && <text x={cx} y={EXP_Y + 36} textAnchor="middle" fontFamily={mono} fontSize="12" fontWeight="700" fill={T.wire} stroke={T.bg} strokeWidth="3" paintOrder="stroke">{expTagged(i)}</text>}
              <Arrow x1={cx} y1={EXP_Y - 2} x2={cx} y2={COMB_Y + BOX_H} on={stage === "combine" && has[i]} dashed={!l} />
            </g>
          );
        })}
        {Array.from({ length: D }, (_, d) => (
          <text key={d} x={MX + d * Ed * cw + 4} y={EXP_Y - 8} fontFamily={mono} fontSize="12" fontWeight="700" fill={EC[d]}>{Ed * cw >= 110 ? `device ${d}${d === sel ? " · following" : ""}` : `D${d}`}</text>
        ))}

        {/* queue labels: source device per row, only where something sits */}
        {!route && Array.from({ length: E }, (_, i) => Array.from({ length: D }, (_, s) => {
          const here = chips.some((c) => c.ex === i && c.s === s);
          return here ? <text key={`${i}-${s}`} x={exL(i) + 3} y={QY + s * QP + QP / 2 + 3} fontFamily={mono} fontSize="12" fill={s === sel ? T.accent : T.dim}>D{s}</text> : null;
        }))}

        {/* combine → add2 → next layer */}
        <Box x={MX} y={COMB_Y} w={RW} h={BOX_H} label="All-to-All combine" on={stage === "combine"} right={combTxt} rightCol={combCol} />
        <Arrow x1={CX} y1={COMB_Y} x2={CX} y2={ADD2_Y + 15} on={stage === "combine"} />
        <circle cx={CX} cy={ADD2_Y} r={15} fill={T.panel} stroke={stage === "combine" ? T.ink : T.rule} strokeWidth={stage === "combine" ? 2.2 : 1.4} />
        <text x={CX} y={ADD2_Y + 6} textAnchor="middle" fontFamily={mono} fontSize="16" fill={T.ink}>+</text>
        <text x={CX + 30} y={ADD2_Y + 5} fontFamily={mono} fontSize="12" fill={T.dim}>→ to next layer</text>

        {/* combine step: the followed token's two outputs, merged at "+" */}
        {combine && chips.filter((c) => c.line).map((c) => (
          <line key={"l" + c.key} x1={c.line[0]} y1={c.line[1]} x2={CX} y2={ADD2_Y + 15} stroke={T.accent2} strokeOpacity={0.55} strokeWidth={1.6} />
        ))}

        {/* router step: chips + their top-2 choices */}
        {routerItems.map((it) => (
          <g key={it.tk.s}>
            {it.pills.map((pl, h) => {
              const px = it.x0 + 38 + (h === 0 ? 0 : it.pills[0].w + 8);
              const col = EC[pl.dv];
              return (
                <g key={h} opacity={pillOp}>
                  {pl.ch.ok ? (
                    <rect x={px} y={it.y0 - 13} width={pl.w} height={26} rx={7} fill={`${col}26`} stroke={col} strokeWidth={1.4} />
                  ) : (
                    <rect x={px} y={it.y0 - 13} width={pl.w} height={26} rx={7} fill="none" stroke={T.bad} strokeWidth={1.4} strokeDasharray="4 3" />
                  )}
                  <text x={px + 8} y={it.y0 + 5} fontFamily={mono} fontSize="13" fill={pl.ch.ok ? T.ink : T.bad}>{pl.txt}</text>
                </g>
              );
            })}
            <Chip x={it.x} y={it.y} id={it.tk.s} col={it.col} w={30} h={24} fly={e < 1} />
          </g>
        ))}

        {/* tokens in the queues / in flight */}
        {chips.map((c) => (
          <Chip key={c.key} x={c.x} y={c.y} id={c.id} col={c.col} fly={c.fly} hi={c.hi} dim={c.dim} opacity={c.opacity ?? 1} />
        ))}
      </svg>
      <div style={{ fontSize: 12, color: T.dim, marginTop: 6, textAlign: "center" }}>
        hatched · {anyLit ? `the ${Ed > 1 ? "experts" : "expert"} living on device ${sel}` : "no local expert"} · <span style={{ color: T.wire }}>yellow</span> = this step · <span style={{ color: T.accent2 }}>pink outline</span> = a token's vector in flight · chip colour = device of its expert · dashed red = dropped
      </div>
    </div>
  );
}
