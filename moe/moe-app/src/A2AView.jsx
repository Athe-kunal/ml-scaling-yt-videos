import { useState } from "react";
import { useProgress, ease, lerp, clamp01 } from "./anim";
import { T, mono, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, SEED, buildModel } from "./moeModel";
import Stepper from "./Stepper";

// All-to-all as a stepper. D devices each own Ed experts (E = D·Ed in total).
// Device s holds D token bundles; bundle t carries the tokens bound for the Ed
// experts living on device t. After the collective, device t holds the bundles
// from every s: the D×D bundle grid is transposed. Bundle (s→t) is drawn at
// device s / slot t before it moves and at device t / slot s after it arrives.
// Schedule shown: D-1 pairwise rounds, round r sends to (d+r) mod D.

const D = CFG.G;
const W = 860, H = 430;
const DX = (d) => 20 + d * 210, DW = 190, BY = 92, BH = 300;
const slotCX = (d, j) => DX(d) + 10 + j * 44 + 20;
const CY1 = BY + 84, CY2 = BY + 160; // send row, receive row
const MY = BY + BH - 54; // centre line of the experts' W_up / W_down
const mod = (a, n) => ((a % n) + n) % n;
const SHADE = [1, 0.55, 0.8, 0.35]; // one shade per local expert

// everything that depends on the number of experts per device
const cache = {};
function makeCfg(Ed) {
  if (cache[Ed]) return cache[Ed];
  const E = D * Ed;
  const C = Math.max(1, Math.round((2 * CFG.S) / E)); // GShard capacity per expert per group
  const B = Ed * C; // slots in one bundle
  const model = buildModel(SEED, E, C);
  // cnt[s][t][i] = tokens group s sends to the i-th expert of device t
  const cnt = Array.from({ length: D }, () => Array.from({ length: D }, () => Array(Ed).fill(0)));
  model.groups.forEach((g, s) => g.toks.forEach((k) => k.final.forEach((f) => (cnt[s][Math.floor(f.e / Ed)][f.e % Ed] += 1))));
  // items[s][t][i][j] = id of the token in slot j of the i-th expert of device t (from group s)
  const items = Array.from({ length: D }, () => Array.from({ length: D }, () => Array.from({ length: Ed }, () => Array(C).fill(null))));
  model.groups.forEach((g, s) => g.toks.forEach((k) => k.final.forEach((f) => { items[s][Math.floor(f.e / Ed)][f.e % Ed][f.slot] = k.s; })));
  const real = (s, t) => cnt[s][t].reduce((a, c) => a + Math.min(c, C), 0);
  const used = Array.from({ length: D }, (_, s) => Array.from({ length: D }, (_, t) => real(s, t))).flat().reduce((a, b) => a + b, 0);
  const tag = Ed === 1 ? "E" : "D"; // bundles are addressed to an expert, or to a device holding several
  const list = (f) => Array.from({ length: D }, (_, d) => f(d)).join(" ");

  const BLOCKS = [
    { id: "start", label: "x[d] = D token bundles", sub: Ed === 1 ? "bundle t is for expert t" : "bundle t is for the experts on device t", code: ["# every device d holds one bundle of tokens per destination device", `x[d] = [tokens_for_${tag}0, …, tokens_for_${tag}${D - 1}]   # [Ddst, B, M] on each device, B = Ed·C`] },
    { id: "keep", label: "keep own bundle", sub: "no network", code: ["out[d][d] = x[d][d]   # one bundle [B, M], copied locally"] },
    {
      id: "loop", label: "for r in 1..D-1", sub: "one bundle per link", loop: true,
      code: ["for r in 1 .. D-1:", "  send x[d][(d+r) % D]  → device (d+r) % D   # [B, M]", "  recv from device (d-r) % D → out[d][(d-r) % D]   # [B, M]"],
    },
    { id: "done", label: "rows → columns", sub: "a transpose of the bundle grid", code: ["# one library call does all of it:", "dist.all_to_all_single(out, x)   # x [Dsrc,Ddst,B,M] → out [Ddst,Dsrc,B,M]", "# out[j] = x_on_device_j[d]"] },
    { id: "moe", label: "in MoE", sub: "bundle = tokens for the local experts", code: ["# bundle s→d = [B, M] = Ed experts × C slots   (Dsrc = G, Ddst = devices)", "# split on arrival: tok[i] = bundle[i·C : (i+1)·C] for local expert i; spare slots are padding"] },
    {
      id: "ffn", label: "expert FFN", sub: Ed === 1 ? "device d applies only expert d" : "device d applies its Ed experts",
      code: [
        "for i in 0 .. Ed-1:   # device d owns experts d·Ed … d·Ed+Ed-1",
        "  h = act(tok[i] @ W_up[d,i])   # [Dsrc·C, M] @ [M, H] → [Dsrc·C, H]",
        "  y[i] = h @ W_down[d,i]   # [Dsrc·C, H] @ [H, M] → [Dsrc·C, M]",
        "# a second all-to-all returns y to the token's home group",
      ],
    },
    {
      id: "back", label: "return all-to-all", sub: "the dispatch, run backwards",
      code: [
        "# same collective, reversed: results go back where the tokens came from",
        "dist.all_to_all_single(y_home, y)   # y [Ddst,Dsrc,B,M] → y_home [Dsrc,Ddst,B,M]",
        "for r in 1 .. D-1:   send y[(d-r) % D] → device (d-r) % D   # [B, M]",
        "out = einsum(\"SEC,ECM->SM\", combine_weights, y_home)   # y_home [Ddst,B,M] = [E,C,M] → out [S, M]: gate-weighted sum",
      ],
    },
  ];

  const sendsTo = (r) => list((d) => `${d}→${mod(d + r, D)}`);
  const round = (r) => [
    { id: `r${r}a`, block: "loop", lines: [0, 1], visAt: 1, round: r, phase: "send", caption: `round ${r} · every device sends a token bundle to d+${r}`, vars: () => [["r", `${r}`], ["d sends to", sendsTo(r)]] },
    { id: `r${r}b`, block: "loop", lines: [2], round: r, phase: "recv", caption: `round ${r} · the tokens land in the receive buffers`, vars: () => [["r", `${r}`], ["d receives from", list((d) => `${mod(d - r, D)}→${d}`)]] },
  ];
  const stay = () => [["stays", list((d) => `${d}→${d}`)]];
  const STEPS = [
    { id: "start", block: "start", lines: [1], round: -2, caption: "each device holds a bundle of tokens for every device", vars: () => [["D", `${D}`], ["experts / device", `${Ed}`], ["bundles / device", `${D}`]] },
    { id: "addr", block: "start", lines: [], round: -2, fan: true, caption: Ed === 1 ? "the bundle for expert t is addressed to device t" : "the bundle for device t carries the tokens of all its experts", vars: () => [["E", `${E}`], ["slots / bundle", `${Ed}·${C} = ${B}`]] },
    { id: "keep", block: "keep", lines: [0], round: 0, caption: Ed === 1 ? "tokens for the local expert stay home" : "tokens for the local experts stay home", vars: stay },
    ...[1, 2, 3].flatMap((r) => round(r)),
    { id: "done", block: "done", lines: [1], round: 3, caption: "rows became columns: every device has the tokens for its experts", vars: () => [["moved", `${D * (D - 1)} bundles`], ["kept", `${D}`]] },
    { id: "moe", block: "moe", lines: [0, 1], visAt: 0, round: 3, detail: true, caption: "in MoE: bundle s→d = group s's tokens for the experts on device d", vars: () => [["C / expert", `${C}`], ["unused slots", `${D * D * B - used} (padding)`]] },
    { id: "ffn_up", block: "ffn", lines: [0, 1], visAt: 1, round: 3, ffn: "up", caption: Ed === 1 ? "each expert multiplies its tokens by its own W_up" : "each local expert multiplies its tokens by its own W_up", vars: () => [["tokens / expert", `${D * C}`], ["W_up", "[M, H]"]] },
    { id: "ffn_dn", block: "ffn", lines: [2, 3], visAt: 0, round: 3, ffn: "dn", caption: "then by W_down; a reverse all-to-all sends results home", vars: () => [["W_down", "[H, M]"], ["output", `[${D}·${C}, M] per expert`]] },
    { id: "back0", block: "back", lines: [0, 1], visAt: 1, round: 3, back: 0, caption: "the dispatch runs backwards: results return to their home group", vars: () => [["y", "[Ddst, Dsrc, B, M]"], ...stay()] },
    ...[1, 2, 3].map((r) => ({ id: `back${r}`, block: "back", lines: [2], round: 3, back: r, caption: `return round ${r} · every device sends results to d−${r}`, vars: () => [["r", `${r}`], ["d sends to", list((d) => `${d}→${mod(d - r, D)}`)]] })),
    { id: "combine", block: "back", lines: [3], round: 3, back: 3, caption: "each token sums its expert outputs, weighted by the gates", vars: () => [["per token", "≤ 2 outputs"], ["out", "[S, M]"]] },
  ];

  const GLOSSARY = {
    Dsrc: ["sending device (row of the grid)", D],
    Ddst: ["receiving device (column of the grid)", D],
    Ed: ["experts living on one device", Ed],
    C: ["token slots per expert", C],
    B: [`slots in one bundle = Ed·C = ${Ed}·${C}`, B],
    E: ["experts in total = D·Ed", E],
    M: ["model dimension of a token vector", null],
    S: ["tokens per group", CFG.S],
    H: ["expert hidden width (W_up out / W_down in)", null],
  };
  const row = (name, dims, what, io, note) => ({ name, dims, what, io, note });
  const XALL = row("x", ["Dsrc", "Ddst", "B", "M"], "x[s,t] = the tokens device s wants to give the experts on device t; device s holds only its own row x[s]", "in", "split on Dsrc");
  const SHAPES = {
    start: [XALL],
    addr: [XALL, row("x[s,t]", ["B", "M"], `one bundle: ${B} token slots of M numbers (${C} per local expert), addressed to device t`, "in")],
    keep: [row("out[d,d]", ["B", "M"], "the diagonal bundle never leaves device d", "out")],
    r1a: [row("bundle", ["B", "M"], "every link carries exactly one token bundle this round; all D links are busy at once", "in"), row("out[d,(d−1)]", ["B", "M"], "lands in the receive buffer slot of the sender", "out")],
    r2a: [row("bundle", ["B", "M"], "same size each round, so D−1 rounds move D−1 bundles per device", "in"), row("out[d,(d−2)]", ["B", "M"], "lands in the receive buffer slot of the sender", "out")],
    r3a: [row("bundle", ["B", "M"], "last round: after it every receive slot is filled", "in"), row("out[d,(d−3)]", ["B", "M"], "lands in the receive buffer slot of the sender", "out")],
    done: [XALL, row("out", ["Ddst", "Dsrc", "B", "M"], "same data with the first two dims swapped: now device t holds column t, every bundle addressed to it", "out", "split on Ddst")],
    moe: [row("dispatched", ["Ddst", "Dsrc", "Ed", "C", "M"], "B is really (Ed, C): device d keeps a separate buffer per local expert, C slots from each of the G groups; unused slots are padding", "out", "→ [E,G,C,M] once Ed moves next to Ddst")],
    ffn_up: [
      row("tokens", ["Dsrc", "Ed", "C", "M"], "everything device d received, grouped by local expert i", "in"),
      row("W_up[d]", ["Ed", "M", "H"], "one up projection per local expert: only device d stores them, so tokens had to travel here", "in"),
      row("h", ["Dsrc", "Ed", "C", "H"], "wider activations after act(tokens @ W_up)", "out"),
    ],
    ffn_dn: [
      row("h", ["Dsrc", "Ed", "C", "H"], "hidden activations on device d", "in"),
      row("W_down[d]", ["Ed", "H", "M"], "projects each expert's result back to the model dimension", "in"),
      row("y", ["Dsrc", "Ed", "C", "M"], "same shape as the tokens that arrived; a second all-to-all returns slot (s, c) to group s", "out"),
    ],
    back0: [
      row("y", ["Ddst", "Dsrc", "B", "M"], "expert outputs, laid out as the tokens arrived: row d on device d", "in"),
      row("y_home", ["Dsrc", "Ddst", "B", "M"], "the transpose undone: device s holds y for every token it sent, one bundle per device", "out", "split on Dsrc"),
    ],
    back1: [
      row("bundle", ["B", "M"], "the same size as on the way in; D−1 rounds again, one link per device", "in"),
      row("y_home[s,t]", ["B", "M"], "lands in the slot the tokens were sent from, so their original positions are known", "out"),
    ],
    combine: [
      row("y_home", ["Dsrc", "Ddst", "B", "M"], "outputs of the ≤2 experts each token visited (padding slots are ignored)", "in"),
      row("out", ["S", "M"], "per token: g1·y(e1) + g2·y(e2), the combine tensor of Algorithm 2 applied to y", "out"),
    ],
  };
  SHAPES.back2 = SHAPES.back1;
  SHAPES.back3 = SHAPES.back1;
  [1, 2, 3].forEach((r) => { SHAPES[`r${r}b`] = SHAPES[`r${r}a`]; });

  return (cache[Ed] = { Ed, E, C, B, cnt, items, groups: model.groups, tag, BLOCKS, STEPS, GLOSSARY, SHAPES });
}

// A bundle's route: one smooth curve from its slot in the sender's send row
// (u = 0) over the wire to its slot in the receiver's recv row (u = 1). The same
// curve is run backwards on the return trip. Same arcs as the wires drawn above.
const arcH = (s, t) => 28 + Math.abs(t - s) * 16;
const route = (s, t, u) => {
  const x0 = slotCX(s, t), y0 = CY1, x1 = slotCX(t, s), y1 = CY2;
  if (s === t) return [lerp(x0, x1, u), lerp(y0, y1, u)];
  const peak = BY - 4 - 0.8 * arcH(s, t);
  const cx = (x0 + x1) / 2, cy = (4 * peak - y0 - y1) / 2;
  const a = (1 - u) * (1 - u), b = 2 * u * (1 - u), c = u * u;
  return [a * x0 + b * cx + c * x1, a * y0 + b * cy + c * y1];
};
const hump = (x) => Math.sin(Math.PI * clamp01(x));

// A bundle = the tokens one group sends to one device. Each token is a bar (a
// vector of M numbers), grouped by local expert (one shade each); dashed bars
// are padding up to the capacity C. qb → qa is the bundle's position on its
// route before / after this line; the bars trail the box a little so you see
// the tokens themselves travel. hb = bar height (M → H → M inside the FFN),
// act = how much of the activation's zeroing is visible.
function Bundle({ cfg, s, t, qb, qa, p, e, sel, detail, retNow, hb, act }) {
  const { C, B, cnt, tag } = cfg;
  const qNow = lerp(qb, qa, e);
  const [x, y] = route(s, t, qNow);
  const moving = qb !== qa;
  const home = s === t;
  const c = EC[t];
  const n = cnt[s][t].reduce((a, v) => a + Math.min(v, C), 0);
  const pitch = Math.min(9, 38 / B), x0 = -(B * pitch - 2) / 2;
  const lag = moving ? 0.22 : 0;
  const nc = Math.max(2, Math.round(hb / 9)), chh = hb / nc;
  return (
    <g>
      <g transform={`translate(${x},${y})`}>
        <rect
          x={-20} y={-22} width={40} height={44 + hb - 18} rx={7} fill={`${c}${retNow ? "44" : "22"}`} stroke={moving ? T.ink : c}
          strokeWidth={moving ? 2.6 : home && qNow === 0 && cnt[s][t].length ? 1.8 : 1.4} opacity={sel === s || sel == null ? 1 : 0.45}
        />
        <text y={-9} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={T.ink}>
          {detail ? `${n}/${B}` : retNow ? `y ${tag}${t}` : qNow >= 1 ? `G${s}` : `→${tag}${t}`}
        </text>
      </g>
      {Array.from({ length: B }, (_, k) => {
        const i = Math.floor(k / C), j = k % C;
        const on = j < Math.min(cnt[s][t][i], C);
        const id = on && sel === s && hb <= 18.5 ? cfg.items[s][t][i][j] : null;
        const ek = ease(clamp01((p - (lag * k) / Math.max(1, B - 1)) / (1 - lag)));
        const [bx, by] = route(s, t, lerp(qb, qa, ek));
        const px = bx + x0 + k * pitch, py = by - 1;
        return (
          <g key={k} transform={`translate(${px},${py})`} opacity={sel === s || sel == null ? 1 : 0.45}>
            {on ? (
              Array.from({ length: nc }, (_, ci) => {
                const zero = act > 0 && (ci * 7 + s * 3 + t * 5 + k * 2) % 3 === 0;
                return <rect key={ci} x={0} y={ci * chh} width={pitch - 2} height={chh - (nc > 2 ? 0.6 : 0)} rx={2} fill={c} fillOpacity={SHADE[i] * (zero ? lerp(1, 0.12, act) : 1)} stroke={c} strokeOpacity={SHADE[i]} />;
              })
            ) : (
              <rect x={0} y={0} width={pitch - 2} height={hb} rx={2} fill="transparent" stroke={c} strokeOpacity={SHADE[i]} strokeDasharray="2 2" />
            )}
            {id != null && <text x={(pitch - 2) / 2} y={13} textAnchor="middle" fontFamily={mono} fontSize="8.5" fontWeight="800" fill={T.bg}>{id}</text>}
          </g>
        );
      })}
    </g>
  );
}


// the Ed experts on device d, each with its two weight matrices drawn as in the
// Dense → MoE tab: W_up widens M → H, W_down narrows H → M
const pairGeo = (d, i, Ed) => {
  const pw = (DW - 20) / Ed, gap = Ed === 1 ? 14 : 10;
  const mw = (pw - gap) / 2;
  const x0 = DX(d) + 10 + i * pw;
  return { x0, upX: x0, dnX: x0 + mw + gap, mw, cx: x0 + pw / 2 };
};
function Experts({ d, ffn, Ed }) {
  const c = EC[d];
  const lo = Ed === 1 ? 14 : 12, hi = Ed === 1 ? 44 : 40;
  const poly = (x0, w, l, r) => `${x0},${MY - l / 2} ${x0 + w},${MY - r / 2} ${x0 + w},${MY + r / 2} ${x0},${MY + l / 2}`;
  const lit = (k) => ffn === k;
  return (
    <g>
      {Array.from({ length: Ed }, (_, i) => {
        const g = pairGeo(d, i, Ed);
        const op = SHADE[i];
        const big = g.mw >= 30;
        return (
          <g key={i}>
            <polygon points={poly(g.upX, g.mw, lo, hi)} fill={c} fillOpacity={(lit("up") ? 0.42 : 0.13) * op + 0.05} stroke={lit("up") ? T.ink : c} strokeWidth={lit("up") ? 2.4 : 1.3} style={{ transition: "all .3s" }} />
            <polygon points={poly(g.dnX, g.mw, hi, lo)} fill={c} fillOpacity={(lit("dn") ? 0.42 : 0.13) * op + 0.05} stroke={lit("dn") ? T.ink : c} strokeWidth={lit("dn") ? 2.4 : 1.3} style={{ transition: "all .3s" }} />
            {big && <text x={g.upX + g.mw / 2} y={MY + 4} textAnchor="middle" fontFamily={mono} fontSize={Ed === 1 ? 9.5 : 8.5} fontWeight="700" fill={T.ink}>W_up</text>}
            {big && <text x={g.dnX + g.mw / 2} y={MY + 4} textAnchor="middle" fontFamily={mono} fontSize={Ed === 1 ? 9.5 : 8.5} fontWeight="700" fill={T.ink}>W_down</text>}
            {Ed === 1 && <text x={g.upX + g.mw / 2} y={MY - 30} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>M→H</text>}
            {Ed === 1 && <text x={g.dnX + g.mw / 2} y={MY - 30} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>H→M</text>}
            <text x={g.cx} y={MY + 40} textAnchor="middle" fontFamily={mono} fontSize={Ed === 1 ? 10.5 : 9.5} fontWeight="700" fill={c} fillOpacity={0.4 + 0.6 * op}>{Ed === 1 ? `Expert ${d}` : `E${d * Ed + i}`}</text>
          </g>
        );
      })}
    </g>
  );
}

// combine: the followed group's first tokens. Each token's two expert outputs
// shrink by their gate weights (g·y), then slide together and stack into one bar.
function Combine({ cfg, sel, e }) {
  const { Ed, groups } = cfg;
  const toks = groups[sel].toks.slice(0, 3);
  const u1 = ease(clamp01(e / 0.5)), u2 = ease(clamp01((e - 0.5) / 0.5));
  const Yb = CY2 + 34, HM = 36;
  const x0 = DX(sel);
  return (
    <g opacity={ease(clamp01(e * 5))}>
      <rect x={x0 + 6} y={CY2 - 34} width={DW - 12} height={88} rx={8} fill={T.panel} stroke={T.rule} />
      {toks.map((tk, ti) => {
        const cx = x0 + 36 + ti * 60;
        const chs = [tk.t1, tk.t2].filter((ch) => ch.ok);
        const hs = chs.map((ch) => HM * lerp(1, ch.w, u1));
        const sum = hs.reduce((a, b) => a + b, 0);
        return (
          <g key={tk.s}>
            {chs.map((ch, ci) => {
              const h = hs[ci];
              const bx = lerp(cx + (ci === 0 ? -10 : 10), cx, u2) - 6;
              const yBot = ci === 0 ? lerp(Yb, Yb - (sum - h), u2) : Yb;
              const col = EC[Math.floor(ch.e / Ed)];
              return (
                <g key={ci}>
                  <rect x={bx} y={yBot - h} width={12} height={h} rx={2} fill={col} fillOpacity={0.85} stroke={T.ink} strokeOpacity={0.5} />
                  <text x={bx + 6} y={Yb - HM - 5} textAnchor="middle" fontFamily={mono} fontSize="8.5" fill={T.soft} opacity={1 - u2}>×{lerp(1, ch.w, u1).toFixed(2)}</text>
                </g>
              );
            })}
            <text x={cx} y={Yb + 12} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.accent} opacity={u2}>out t{tk.s}</text>
          </g>
        );
      })}
    </g>
  );
}

function Scene({ step, sel, cfg }) {
  const p = useProgress(step.id); // 0 → 1 each time a new picture arrives
  const e = ease(p);
  const { round, fan, detail, phase, ffn, back } = step;
  const sid = step.sid;
  const { Ed, tag } = cfg;
  const done = round >= D - 1 && phase !== "send";
  const landed = phase === "send" ? round - 1 : round;

  // position of the bundle with ring distance r on its route, before / after this line
  const uPair = (r) => {
    if (sid === "keep") return r === 0 ? [0, 1] : [0, 0];
    if (phase === "send" && r === round) return [0, 0.5]; // leaves the device, rides the wire
    if (phase === "recv" && r === round) return [0.5, 1]; // lands in the receive buffer
    const v = r <= landed ? 1 : 0;
    return [v, v];
  };
  const wPair = (r) => (back == null ? [0, 0] : sid === "combine" ? [1, 1] : [r < back ? 1 : 0, r <= back ? 1 : 0]);
  const qPair = (r) => {
    if (back == null) return uPair(r);
    const w = wPair(r);
    return [1 - w[0], 1 - w[1]]; // the return trip is the same route run backwards
  };

  let hb = 18, act = 0;
  if (sid === "ffn_up") { hb = lerp(18, 44, ease(clamp01(p / 0.55))); act = clamp01((p - 0.55) / 0.45); }
  else if (sid === "ffn_dn") { hb = lerp(44, 18, ease(clamp01((p - 0.45) / 0.55))); act = 1 - clamp01(p / 0.45); }

  const arcs = [];
  if (fan) for (let t = 0; t < D; t++) if (t !== sel) arcs.push([sel, t]);
  if (phase) for (let s = 0; s < D; s++) arcs.push([s, mod(s + round, D)]);
  if (back > 0 && sid !== "combine") for (let s = 0; s < D; s++) arcs.push([mod(s + back, D), s]);
  const bundles = [];
  let total = 0;
  for (let s = 0; s < D; s++) for (let t = 0; t < D; t++) {
    const r = mod(t - s, D);
    const [qb, qa] = qPair(r);
    const w = wPair(r);
    const retNow = back != null && (w[0] === 1 || (w[1] === 1 && e >= 0.5));
    total += back == null ? lerp(qb, qa, e) : 1 - lerp(qb, qa, e);
    bundles.push(<Bundle key={`${s}-${t}`} cfg={cfg} s={s} t={t} qb={qb} qa={qa} p={p} e={e} sel={sel} detail={detail} retNow={retNow} hb={hb} act={act} />);
  }
  const fanOn = (s, t) => fan && s === sel && t !== s;
  const flashP = hump((p - 0.55) / 0.45);
  const slotGlow = (d, j, recv) => {
    if (recv) {
      if (sid === "keep") return d === j ? flashP : 0;
      return phase === "recv" && mod(d - j, D) === round ? flashP : 0;
    }
    return back != null && sid !== "combine" && mod(j - d, D) === back ? flashP : 0;
  };
  const ffnLabel = ffn === "up" ? (p < 0.55 ? "x → W_up x : M → H" : "act( · ) : negatives → 0") : ffn === "dn" ? "h → W_down h : H → M" : null;
  const t0 = groupsFirst(cfg, sel);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        <marker id="a2a-arrow" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={T.wire} />
        </marker>
      </defs>

      {Array.from({ length: D }, (_, d) => (
        <g key={d}>
          <rect x={DX(d)} y={BY} width={DW} height={BH} rx={10} fill={T.panel} stroke={d === sel ? T.accent : T.rule} strokeWidth={d === sel ? 1.8 : 1.2} />
          <text x={DX(d) + 12} y={BY + 22} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.ink}>Device {d}{d === sel ? " · following" : ""}</text>
          <Experts d={d} ffn={ffn} Ed={Ed} />
          {[[CY1, back != null ? `home: results y from ${tag}_j` : `send: tokens for ${tag}_j`], [CY2, ffnLabel ?? "recv: tokens from G_j"]].map(([cy, label]) => (
            <g key={cy}>
              <text x={DX(d) + 10} y={cy - 32} fontFamily={mono} fontSize="9.5" fill={(cy === CY2 && done) || (cy === CY2 && ffn) ? T.accent : T.dim}>{label}</text>
              {Array.from({ length: D }, (_, j) => {
                const g = slotGlow(d, j, cy === CY2);
                return (
                  <rect
                    key={j} x={DX(d) + 10 + j * 44} y={cy - 24} width={40} height={48} rx={7} fill={g > 0 ? T.accent : "none"} fillOpacity={0.22 * g}
                    stroke={g > 0 ? T.accent : T.rule} strokeWidth={1 + 2 * g} strokeDasharray={g > 0.2 ? "none" : "3 3"}
                  />
                );
              })}
            </g>
          ))}
        </g>
      ))}

      {sid === "combine" && <Combine cfg={cfg} sel={sel} e={e} />}

      {arcs.map(([s, t], i) => {
        const x0 = DX(s) + DW / 2, x1 = DX(t) + DW / 2;
        const h = arcH(s, t);
        const grow = fan ? clamp01(e * 1.4) : 1;
        return (
          <path
            key={i} d={`M${x0},${BY - 4} Q${(x0 + x1) / 2},${BY - 4 - h * 1.6} ${x1},${BY - 4}`} fill="none" stroke={T.wire} strokeWidth={2}
            opacity={0.8 * (fan ? clamp01(e * 3) : 1)} className={fan ? undefined : "lane"} markerEnd="url(#a2a-arrow)"
            pathLength={1} strokeDasharray={fan ? `${grow} 1` : undefined}
          />
        );
      })}
      {fan && arcs.map(([s, t], i) => {
        const x0 = DX(s) + DW / 2, x1 = DX(t) + DW / 2;
        const h = arcH(s, t);
        const a = (1 - e) * (1 - e), b = 2 * e * (1 - e), c2 = e * e;
        return <circle key={`dot${i}`} cx={a * x0 + b * (x0 + x1) / 2 + c2 * x1} cy={a * (BY - 4) + b * (BY - 4 - h * 1.6) + c2 * (BY - 4)} r={4} fill={T.ink} opacity={0.9 * (1 - clamp01((e - 0.9) / 0.1))} />;
      })}

      {ffn && Array.from({ length: D }, (_, d) => Array.from({ length: Ed }, (_, i) => {
        const g = pairGeo(d, i, Ed);
        const ux = g.upX + g.mw / 2;
        const up = ffn === "up";
        return (
          <g key={`${d}-${i}`}>
            <path
              className="lane" fill="none" stroke={T.wire} strokeWidth={2} opacity={0.85} markerEnd="url(#a2a-arrow)"
              d={up ? `M${ux},${CY2 + 28} L${ux},${MY - 28}` : `M${g.upX + g.mw + 1},${MY} L${g.dnX - 1},${MY}`}
            />
            <circle cx={up ? ux : lerp(g.upX + g.mw + 1, g.dnX - 1, e)} cy={up ? lerp(CY2 + 28, MY - 28, e) : MY} r={3.5} fill={T.ink} opacity={0.9} />
          </g>
        );
      }))}

      {bundles}

      <text x={20} y={46} fontFamily={mono} fontSize="12" fontWeight="700" fill={T.accent}>
        {sid === "combine" && t0
          ? `out[t${t0.s}] = ${[t0.t1, t0.t2].filter((c) => c.ok).map((c) => `${lerp(1, c.w, e).toFixed(2)}·y(E${c.e})`).join(" + ")}`
          : ffn
            ? null
            : back != null
              ? `home ${Math.round(total)} / ${D * D} bundles`
              : `delivered ${Math.round(total)} / ${D * D} bundles`}
      </text>

      {ffn && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          {Ed === 1 ? "device d holds only expert d's weights, so the tokens had to come to it" : "each device holds only its own experts' weights, so the tokens had to come to them"}
        </text>
      )}
      {back != null && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          {sid === "combine" ? "gate weights g1, g2 scale each expert's y before the sum" : "same pattern as dispatch, direction reversed: results come home to the group that owns the token"}
        </text>
      )}
      {phase && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          each device sends 1 token bundle and receives 1 — no link is shared
        </text>
      )}
    </svg>
  );
}
const groupsFirst = (cfg, sel) => cfg.groups[sel].toks[0];


// Where is each token of the followed group right now? One row per token, one
// cell per expert it was routed to (top-1 / top-2), each with its gate weight.
function Inspector({ cfg, step, sel }) {
  const { Ed, C, groups } = cfg;
  const g = groups[sel];
  const landed = step.phase === "send" ? step.round - 1 : step.round;
  const where = (ch) => {
    const dev = Math.floor(ch.e / Ed);
    const r = mod(dev - sel, D);
    if (step.back != null && r <= step.back) return ["result is home", T.accent];
    if (r <= landed) {
      if (step.ffn) return [step.ffn === "up" ? `E${ch.e} · W_up` : `E${ch.e} · W_down`, T.ink];
      if (step.back != null || ["ffn_up", "ffn_dn"].includes(step.sid)) return [`y waits on D${dev}`, T.soft];
      return [`arrived at D${dev}`, T.soft];
    }
    return [dev === sel ? "stays on this device" : `still on D${sel}, bound for D${dev}`, T.dim];
  };
  const cell = (ch, why) => {
    if (!ch.ok) return <span style={{ color: T.bad }}>dropped · {why}</span>;
    const dev = Math.floor(ch.e / Ed);
    const [txt, col] = where(ch);
    return (
      <span>
        <b style={{ color: EC[dev] }}>E{ch.e}</b> <span style={{ color: T.dim }}>on D{dev}</span> · <b>g = {ch.w.toFixed(2)}</b>
        <div style={{ color: col, fontSize: 10.5 }}>{txt}</div>
      </span>
    );
  };
  const combined = step.sid === "combine";
  const ce = ease(useProgress(step.id));
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 8 }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>TOKENS OF DEVICE {sel} (GROUP G{sel}) · digit in a bar = token id</div>
      <div style={{ fontSize: 11.5, color: T.dim, lineHeight: 1.5 }}>
        The router gave every token two experts. <b style={{ color: T.soft }}>g</b> is that expert's gate weight (the two add up to 1). A bar carries the token's vector to the expert, so a token with two experts appears in two bars.
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "44px 1fr 1fr", gap: "6px 12px", fontFamily: mono, fontSize: 11.5, alignItems: "start" }}>
        <span style={{ color: T.dim }}>token</span><span style={{ color: T.dim }}>1st choice</span><span style={{ color: T.dim }}>2nd choice</span>
        {g.toks.map((t) => (
          <div key={t.s} style={{ display: "contents" }}>
            <b style={{ color: T.ink }}>t{t.s}</b>
            {cell(t.t1, "over capacity")}
            {cell(t.t2, t.t2.capOk ? "random routing" : "over capacity")}
            {combined && (
              <div style={{ gridColumn: "1 / -1", color: T.accent, fontSize: 11, margin: "-2px 0 2px 56px" }}>
                out[t{t.s}] = {t.t1.ok ? `${lerp(1, t.t1.w, ce).toFixed(2)}·y(E${t.t1.e})` : ""}{t.t1.ok && t.t2.ok ? " + " : ""}{t.t2.ok ? `${lerp(1, t.t2.w, ce).toFixed(2)}·y(E${t.t2.e})` : ""}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function A2AView() {
  const [sel, setSel] = useState(0);
  const [Ed, setEd] = useState(1);
  const cfg = makeCfg(Ed);
  return (
    <Stepper
      title="All-to-all collective"
      subtitle={`D=${D} devices · ${Ed} expert${Ed > 1 ? "s" : ""} per device (E=${cfg.E}) · every device sends its tokens for each device's experts to that device`}
      blocks={cfg.BLOCKS}
      steps={cfg.STEPS}
      header={(step) => (
        <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>experts / device</span>
            {[1, 2, 3, 4].map((v) => (
              <button key={v} onClick={() => setEd(v)} style={{ ...btn(v === Ed, false), height: 28, padding: "0 10px" }}>{v}</button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }} title="the bundles that device 's' started with stay bright in every step, including the return trip">
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow device</span>
            {Array.from({ length: D }, (_, k) => (
              <button key={k} onClick={() => setSel(k)} style={{ ...btn(k === sel, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
        </div>
      )}
      scene={(step) => <Scene step={step} sel={sel} cfg={cfg} />}
      extra={(step) => <Inspector cfg={cfg} step={step} sel={sel} />}
      glossary={cfg.GLOSSARY}
      shapes={(step) => cfg.SHAPES[step.sid]}
      legend={
        <>
          <span>bright = bundles that started on the followed device · bar = one token's vector (a vector of M numbers, not a probability) · dashed = padding{Ed > 1 ? " · one shade per local expert" : ""} · box <b style={{ color: T.soft }}>→{cfg.tag} t</b> = tokens for {Ed === 1 ? "expert" : "the experts on device"} t</span>
          {EC.map((c, e) => <span key={e} style={{ color: c }}>■ for device {e}</span>)}
        </>
      }
    />
  );
}
