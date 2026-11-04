import { useState } from "react";
import { T, mono, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";
import Stepper from "./Stepper";

// All-to-all as a stepper. Device s holds D token bundles; bundle t is addressed to
// device t. After the collective, device t holds the bundles from every s:
// the D×D chunk grid is transposed. Chunk (s→t) is drawn at device s / slot t
// before it moves and at device t / slot s after it arrives. Schedule shown:
// D-1 pairwise rounds, round r sends to (d+r) mod D, one bundle per link.

const D = CFG.G;
const W = 860, H = 430;
const DX = (d) => 20 + d * 210, DW = 190, BY = 92, BH = 300;
const slotCX = (d, j) => DX(d) + 10 + j * 44 + 20;
const CY1 = BY + 84, CY2 = BY + 160; // send row, receive row
const MY = BY + BH - 54; // centre line of the expert's W_up / W_down
const UPX = (d) => DX(d) + 14, DNX = (d) => DX(d) + 106, MW = 70;
const mod = (a, n) => ((a % n) + n) % n;

// tokens group s sends to expert t (every chunk is padded to C slots)
const COUNT = Array.from({ length: D }, (_, s) => {
  const row = Array(D).fill(0);
  MODEL.groups[s].toks.forEach((k) => k.final.forEach((f) => (row[f.e] += 1)));
  return row;
});

const BLOCKS = [
  { id: "start", label: "x[d] = D token bundles", sub: "bundle t is for expert t", code: ["# every device d holds one bundle of tokens per expert", "x[d] = [tokens_for_E0, …, tokens_for_E(D-1)]   # [Ddst, C, M] on each device"] },
  { id: "keep", label: "keep own bundle", sub: "no network", code: ["out[d][d] = x[d][d]   # one bundle [C, M], copied locally"] },
  {
    id: "loop", label: "for r in 1..D-1", sub: "one bundle per link", loop: true,
    code: ["for r in 1 .. D-1:", "  send x[d][(d+r) % D]  → device (d+r) % D   # [C, M]", "  recv from device (d-r) % D → out[d][(d-r) % D]   # [C, M]"],
  },
  { id: "done", label: "rows → columns", sub: "a transpose of the bundle grid", code: ["# one library call does all of it:", "dist.all_to_all_single(out, x)   # x [Dsrc,Ddst,C,M] → out [Ddst,Dsrc,C,M]", "# out[j] = x_on_device_j[d]"] },
  { id: "moe", label: "in MoE", sub: "bundle = tokens for one expert", code: ["# bundle s→e = [C, M]: tokens of group s   (Dsrc = G, Ddst = E)", "# for expert e, padded to C slots"] },
  { id: "ffn", label: "expert FFN", sub: "device d applies only expert d", code: ["h = act(tokens @ W_up[d])   # [Dsrc·C, M] @ [M, H] → [Dsrc·C, H]", "y = h @ W_down[d]   # [Dsrc·C, H] @ [H, M] → [Dsrc·C, M]", "# a second all-to-all returns y to the token's home group"] },
  {
    id: "back", label: "return all-to-all", sub: "the dispatch, run backwards",
    code: [
      "# same collective, reversed: results go back where the tokens came from",
      "dist.all_to_all_single(y_home, y)   # y [Ddst,Dsrc,C,M] → y_home [Dsrc,Ddst,C,M]",
      "for r in 1 .. D-1:   send y[(d-r) % D] → device (d-r) % D   # [C, M]",
      "out[s,c] = Σ_e gate[s,c,e] · y_home[s,e,slot]   # combine: gate-weighted sum, [S, M]",
    ],
  },
];

const sendsTo = (r) => Array.from({ length: D }, (_, d) => `${d}→${mod(d + r, D)}`).join(" ");
const round = (r, vars) => [
  { id: `r${r}a`, block: "loop", lines: [0, 1], visAt: 1, round: r, phase: "send", caption: `round ${r} · every device sends a token bundle to d+${r}`, vars: () => [["r", `${r}`], ["d sends to", sendsTo(r)]] },
  { id: `r${r}b`, block: "loop", lines: [2], round: r, phase: "recv", caption: `round ${r} · the tokens land in the receive buffers`, vars: () => [["r", `${r}`], ["d receives from", Array.from({ length: D }, (_, d) => `${mod(d - r, D)}→${d}`).join(" ")]] },
];
const STEPS = [
  { id: "start", block: "start", lines: [1], round: -2, caption: "each device holds a bundle of tokens for every expert", vars: () => [["D", `${D}`], ["bundles / device", `${D}`]] },
  { id: "addr", block: "start", lines: [], round: -2, fan: true, caption: "the bundle for expert t is addressed to device t", vars: () => [["D", `${D}`], ["bundles / device", `${D}`]] },
  { id: "keep", block: "keep", lines: [0], round: 0, caption: "tokens for the local expert stay home", vars: () => [["stays", Array.from({ length: D }, (_, d) => `${d}→${d}`).join(" ")]] },
  ...[1, 2, 3].flatMap((r) => round(r)),
  { id: "done", block: "done", lines: [1], round: 3, caption: "rows became columns: every expert has its tokens", vars: () => [["moved", `${D * (D - 1)} bundles`], ["kept", `${D}`]] },
  { id: "moe", block: "moe", lines: [0], round: 3, detail: true, caption: "in MoE: bundle s→e = group s's tokens for expert e", vars: () => [["C", `${CFG.C}`], ["unused slots", `${D * D * CFG.C - COUNT.flat().reduce((a, b) => a + b, 0)} (padding)`]] },
  { id: "ffn_up", block: "ffn", lines: [0], round: 3, ffn: "up", caption: "each expert multiplies its tokens by its own W_up", vars: () => [["tokens / expert", `${D * CFG.C}`], ["W_up", "[M, H]"]] },
  { id: "ffn_dn", block: "ffn", lines: [1, 2], visAt: 0, round: 3, ffn: "dn", caption: "then by W_down; a reverse all-to-all sends results home", vars: () => [["W_down", "[H, M]"], ["output", "[Dsrc·C, M]"]] },
  { id: "back0", block: "back", lines: [0, 1], visAt: 1, round: 3, back: 0, caption: "the dispatch runs backwards: results return to their home group", vars: () => [["y", "[Ddst, Dsrc, C, M]"], ["stays", Array.from({ length: D }, (_, d) => `${d}→${d}`).join(" ")]] },
  ...[1, 2, 3].map((r) => ({ id: `back${r}`, block: "back", lines: [2], round: 3, back: r, caption: `return round ${r} · every device sends results to d−${r}`, vars: () => [["r", `${r}`], ["d sends to", Array.from({ length: D }, (_, d) => `${d}→${mod(d - r, D)}`).join(" ")]] })),
  { id: "combine", block: "back", lines: [3], round: 3, back: 3, caption: "each token sums its expert outputs, weighted by the gates", vars: () => [["per token", "≤ 2 outputs"], ["out", "[S, M]"]] },
];

const GLOSSARY = {
  Dsrc: ["sending device (row of the grid)", D],
  Ddst: ["receiving device (column of the grid)", D],
  C: ["slots in one token bundle", CFG.C],
  M: ["model dimension of a token vector", null],
  S: ["tokens per group", CFG.S],
  H: ["expert hidden width (W_up out / W_down in)", null],
};
const row = (name, dims, what, io, note) => ({ name, dims, what, io, note });
const XALL = row("x", ["Dsrc", "Ddst", "C", "M"], "x[s,t] = the tokens device s wants to give expert t; device s holds only its own row x[s]", "in", "split on Dsrc");
const SHAPES = {
  start: [XALL],
  addr: [XALL, row("x[s,t]", ["C", "M"], "one bundle: C token slots of M numbers, addressed to expert t", "in")],
  keep: [row("out[d,d]", ["C", "M"], "the diagonal bundle never leaves device d", "out")],
  r1a: [row("bundle", ["C", "M"], "every link carries exactly one token bundle this round; all D links are busy at once", "in"), row("out[d,(d−1)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  r2a: [row("bundle", ["C", "M"], "same size each round, so D−1 rounds move D−1 bundles per device", "in"), row("out[d,(d−2)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  r3a: [row("bundle", ["C", "M"], "last round: after it every receive slot is filled", "in"), row("out[d,(d−3)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  done: [XALL, row("out", ["Ddst", "Dsrc", "C", "M"], "same data with the first two dims swapped: now device t holds column t, every bundle addressed to it", "out", "split on Ddst")],
  moe: [row("dispatched", ["Ddst", "Dsrc", "C", "M"], "read Dsrc as G and Ddst as E: expert e's buffer holds C slots from each of the G groups; unused slots are padding", "out", "= [E,G,C,M]")],
};

SHAPES.ffn_up = [
  row("tokens", ["Dsrc", "C", "M"], "everything device d received: C slots from each of the D sources, all meant for expert d", "in"),
  row("W_up[d]", ["M", "H"], "expert d's up projection: only device d stores it, so tokens had to travel here", "in"),
  row("h", ["Dsrc", "C", "H"], "wider activations after act(tokens @ W_up)", "out"),
];
SHAPES.ffn_dn = [
  row("h", ["Dsrc", "C", "H"], "hidden activations on device d", "in"),
  row("W_down[d]", ["H", "M"], "projects back to the model dimension", "in"),
  row("y", ["Dsrc", "C", "M"], "same shape as the tokens that arrived; a second all-to-all returns slot (s, c) to group s", "out"),
];

SHAPES.back0 = [
  row("y", ["Ddst", "Dsrc", "C", "M"], "expert outputs, laid out as the tokens arrived: row e on device e", "in"),
  row("y_home", ["Dsrc", "Ddst", "C", "M"], "the transpose undone: device s holds y for every token it sent, one bundle per expert", "out", "split on Dsrc"),
];
SHAPES.back1 = [
  row("bundle", ["C", "M"], "the same size as on the way in; D−1 rounds again, one link per device", "in"),
  row("y_home[s,e]", ["C", "M"], "lands in the slot the tokens were sent from, so their original positions are known", "out"),
];
SHAPES.back2 = SHAPES.back1;
SHAPES.back3 = SHAPES.back1;
SHAPES.combine = [
  row("y_home", ["Dsrc", "Ddst", "C", "M"], "outputs of the ≤2 experts each token visited (padding slots are ignored)", "in"),
  row("out", ["S", "M"], "per token: g1·y(e1) + g2·y(e2), the combine tensor of Algorithm 2 applied to y", "out"),
];

[1, 2, 3].forEach((r) => { SHAPES[`r${r}b`] = SHAPES[`r${r}a`]; });

// A bundle = the tokens one group sends to one expert. Each token is a bar (a
// vector of M numbers); dashed bars are padding up to the capacity C.
function Chunk({ s, t, round, detail, sel, moving, back }) {
  const r = mod(t - s, D);
  const ret = back != null && r <= back; // result has gone home
  const arrived = r <= round && !ret;
  const dev = arrived ? t : s;
  const slot = arrived ? s : t;
  const x = slotCX(dev, slot);
  const home = s === t;
  const c = EC[t];
  const n = Math.min(COUNT[s][t], CFG.C);
  return (
    <g
      style={{
        transform: `translate(${x}px, ${arrived ? CY2 : CY1}px)`,
        transition: `transform .9s cubic-bezier(.3,.7,.3,1) ${(s * 60) % 240}ms`,
      }}
    >
      <rect
        x={-20} y={-22} width={40} height={44} rx={7} fill={`${c}${ret ? "44" : "22"}`} stroke={moving ? T.ink : c}
        strokeWidth={moving ? 2.6 : home && round === 0 ? 2.4 : 1.4} opacity={sel === s || sel == null ? 1 : 0.6}
      />
      <text y={-9} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={T.ink}>
        {detail ? `${n}/${CFG.C}` : ret ? `y E${t}` : arrived ? `G${s}` : `→E${t}`}
      </text>
      {Array.from({ length: CFG.C }, (_, k) => (
        <rect key={k} x={-16 + k * 11} y={-1} width={9} height={18} rx={2} fill={k < n ? c : "transparent"} stroke={c} strokeDasharray={k < n ? "none" : "2 2"} />
      ))}
    </g>
  );
}

// expert d's two weight matrices, drawn as in the Dense → MoE tab: W_up widens
// M → H, W_down narrows H → M
function Expert({ d, ffn }) {
  const c = EC[d];
  const lo = 14, hi = 44;
  const poly = (x0, l, r) => `${x0},${MY - l / 2} ${x0 + MW},${MY - r / 2} ${x0 + MW},${MY + r / 2} ${x0},${MY + l / 2}`;
  const lit = (k) => ffn === k;
  return (
    <g>
      <polygon points={poly(UPX(d), lo, hi)} fill={`${c}${lit("up") ? "66" : "22"}`} stroke={lit("up") ? T.ink : c} strokeWidth={lit("up") ? 2.4 : 1.3} style={{ transition: "all .3s" }} />
      <polygon points={poly(DNX(d), hi, lo)} fill={`${c}${lit("dn") ? "66" : "22"}`} stroke={lit("dn") ? T.ink : c} strokeWidth={lit("dn") ? 2.4 : 1.3} style={{ transition: "all .3s" }} />
      <text x={UPX(d) + MW / 2} y={MY + 4} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={T.ink}>W_up</text>
      <text x={DNX(d) + MW / 2} y={MY + 4} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={T.ink}>W_down</text>
      <text x={UPX(d) + MW / 2} y={MY - 30} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>M→H</text>
      <text x={DNX(d) + MW / 2} y={MY - 30} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>H→M</text>
      <text x={DX(d) + DW / 2} y={MY + 40} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={c}>Expert {d}</text>
    </g>
  );
}

function Scene({ step, sel }) {
  const { round, fan, detail, phase, ffn, back } = step;
  const done = round >= D - 1 && phase !== "send";
  const landed = phase === "send" ? round - 1 : round; // chunks of this round only move on the recv line
  const arcs = [];
  if (fan) for (let t = 0; t < D; t++) if (t !== sel) arcs.push([sel, t]);
  if (phase) for (let s = 0; s < D; s++) arcs.push([s, mod(s + round, D)]);
  if (back > 0 && step.sid !== "combine") for (let s = 0; s < D; s++) arcs.push([mod(s + back, D), s]);
  const chunks = [];
  for (let s = 0; s < D; s++) for (let t = 0; t < D; t++) chunks.push([s, t]);
  const movingR = phase ? round : -1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        <marker id="a2a-arrow" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={T.wire} />
        </marker>
      </defs>

      {Array.from({ length: D }, (_, d) => (
        <g key={d}>
          <rect x={DX(d)} y={BY} width={DW} height={BH} rx={10} fill={T.panel} stroke={d === sel ? T.soft : T.rule} strokeWidth={d === sel ? 1.8 : 1.2} />
          <text x={DX(d) + 12} y={BY + 22} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.ink}>Device {d}</text>
          <Expert d={d} ffn={ffn} />
          {[[CY1, back != null ? "home: results y from E_j" : "send: tokens for E_j"], [CY2, "recv: tokens from G_j"]].map(([cy, label]) => (
            <g key={cy}>
              <text x={DX(d) + 10} y={cy - 32} fontFamily={mono} fontSize="9.5" fill={cy === CY2 && done ? T.accent : T.dim}>{label}</text>
              {Array.from({ length: D }, (_, j) => (
                <rect key={j} x={DX(d) + 10 + j * 44} y={cy - 24} width={40} height={48} rx={7} fill="none" stroke={T.rule} strokeDasharray="3 3" />
              ))}
            </g>
          ))}
        </g>
      ))}

      {arcs.map(([s, t], i) => {
        const x0 = DX(s) + DW / 2, x1 = DX(t) + DW / 2;
        const h = 28 + Math.abs(t - s) * 16;
        return (
          <path
            key={i} d={`M${x0},${BY - 4} Q${(x0 + x1) / 2},${BY - 4 - h * 1.6} ${x1},${BY - 4}`} fill="none" stroke={T.wire} strokeWidth={2}
            opacity={0.8} className="lane" markerEnd="url(#a2a-arrow)"
          />
        );
      })}

      {ffn && Array.from({ length: D }, (_, d) => (
        <path
          key={d} className="lane" fill="none" stroke={T.wire} strokeWidth={2} opacity={0.85} markerEnd="url(#a2a-arrow)"
          d={ffn === "up" ? `M${UPX(d) + MW / 2},${CY2 + 28} L${UPX(d) + MW / 2},${MY - 28}` : `M${UPX(d) + MW + 2},${MY} L${DNX(d) - 2},${MY}`}
        />
      ))}

      {chunks.map(([s, t]) => (
        <Chunk key={`${s}-${t}`} s={s} t={t} round={landed} detail={detail} sel={fan ? sel : null} moving={mod(t - s, D) === movingR || (back > 0 && step.sid !== "combine" && mod(t - s, D) === back) || (fan && s === sel && t !== s)}
          back={back} />
      ))}

      {ffn && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          device d holds only expert d's weights, so the tokens had to come to it
        </text>
      )}
      {back != null && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          {step.sid === "combine" ? "gate weights g1, g2 scale each expert's y before the sum" : "same pattern as dispatch, direction reversed: results come home to the group that owns the token"}
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

export default function A2AView() {
  const [sel, setSel] = useState(0);
  return (
    <Stepper
      title="All-to-all collective"
      subtitle={`D=${D} devices · each holds one expert · every device sends its tokens for each expert to that expert's device`}
      blocks={BLOCKS}
      steps={STEPS}
      header={(step) =>
        step.fan && (
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow device</span>
            {Array.from({ length: D }, (_, k) => (
              <button key={k} onClick={() => setSel(k)} style={{ ...btn(k === sel, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
        )
      }
      scene={(step) => <Scene step={step} sel={sel} />}
      glossary={GLOSSARY}
      shapes={(step) => SHAPES[step.sid]}
      legend={
        <>
          <span>bar = one token (M numbers) · dashed = padding · box <b style={{ color: T.soft }}>→E t</b> = tokens for expert t</span>
          {EC.map((c, e) => <span key={e} style={{ color: c }}>■ for device {e}</span>)}
        </>
      }
    />
  );
}
