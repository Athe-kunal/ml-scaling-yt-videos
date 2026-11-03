import { useState } from "react";
import { T, mono, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";
import Stepper from "./Stepper";

// All-to-all as a stepper. Device s holds D chunks; chunk t is addressed to
// device t. After the collective, device t holds the chunks from every s:
// the D×D chunk grid is transposed. Chunk (s→t) is drawn at device s / slot t
// before it moves and at device t / slot s after it arrives. Schedule shown:
// D-1 pairwise rounds, round r sends to (d+r) mod D, one chunk per link.

const D = CFG.G;
const W = 860, H = 350;
const DX = (d) => 20 + d * 210, DW = 190, BY = 92, BH = 215;
const slotCX = (d, j) => DX(d) + 10 + j * 44 + 20;
const CY1 = BY + 84, CY2 = BY + 176; // send row, receive row
const mod = (a, n) => ((a % n) + n) % n;

// tokens group s sends to expert t (every chunk is padded to C slots)
const COUNT = Array.from({ length: D }, (_, s) => {
  const row = Array(D).fill(0);
  MODEL.groups[s].toks.forEach((k) => k.final.forEach((f) => (row[f.e] += 1)));
  return row;
});

const BLOCKS = [
  { id: "start", label: "x[d] = D chunks", sub: "chunk t is for device t", code: ["# every device d holds", "x[d] = [chunk_0, …, chunk_D-1]   # [Ddst, C, M] on each device"] },
  { id: "keep", label: "keep own chunk", sub: "no network", code: ["out[d][d] = x[d][d]   # one chunk [C, M], copied locally"] },
  {
    id: "loop", label: "for r in 1..D-1", sub: "one chunk per link", loop: true,
    code: ["for r in 1 .. D-1:", "  send x[d][(d+r) % D]  → device (d+r) % D   # [C, M]", "  recv from device (d-r) % D → out[d][(d-r) % D]   # [C, M]"],
  },
  { id: "done", label: "rows → columns", sub: "a transpose of the chunk grid", code: ["# one library call does all of it:", "dist.all_to_all_single(out, x)   # x [Dsrc,Ddst,C,M] → out [Ddst,Dsrc,C,M]", "# out[j] = x_on_device_j[d]"] },
  { id: "moe", label: "in MoE", sub: "chunk = expert buffer slice", code: ["# chunk s→e = [C, M]: tokens of group s   (Dsrc = G, Ddst = E)", "# for expert e, padded to C slots"] },
];

const sendsTo = (r) => Array.from({ length: D }, (_, d) => `${d}→${mod(d + r, D)}`).join(" ");
const round = (r, vars) => [
  { id: `r${r}a`, block: "loop", lines: [0, 1], visAt: 1, round: r, phase: "send", caption: `round ${r} · every device sends a chunk to d+${r}`, vars: () => [["r", `${r}`], ["d sends to", sendsTo(r)]] },
  { id: `r${r}b`, block: "loop", lines: [2], round: r, phase: "recv", caption: `round ${r} · the chunks land in the receive buffers`, vars: () => [["r", `${r}`], ["d receives from", Array.from({ length: D }, (_, d) => `${mod(d - r, D)}→${d}`).join(" ")]] },
];
const STEPS = [
  { id: "start", block: "start", lines: [1], round: -2, caption: "each device holds one chunk per destination", vars: () => [["D", `${D}`], ["chunks / device", `${D}`]] },
  { id: "addr", block: "start", lines: [], round: -2, fan: true, caption: "chunk t is addressed to device t", vars: () => [["D", `${D}`], ["chunks / device", `${D}`]] },
  { id: "keep", block: "keep", lines: [0], round: 0, caption: "chunk d→d is already home", vars: () => [["stays", Array.from({ length: D }, (_, d) => `${d}→${d}`).join(" ")]] },
  ...[1, 2, 3].flatMap((r) => round(r)),
  { id: "done", block: "done", lines: [1], round: 3, caption: "rows became columns: a transpose", vars: () => [["moved", `${D * (D - 1)} chunks`], ["kept", `${D}`]] },
  { id: "moe", block: "moe", lines: [0], round: 3, detail: true, caption: "in MoE: chunk s→e = group s's tokens for expert e", vars: () => [["C", `${CFG.C}`], ["unused slots", `${D * D * CFG.C - COUNT.flat().reduce((a, b) => a + b, 0)} (padding)`]] },
];

const GLOSSARY = {
  Dsrc: ["sending device (row of the grid)", D],
  Ddst: ["receiving device (column of the grid)", D],
  C: ["slots in one chunk", CFG.C],
  M: ["model dimension of a token vector", null],
};
const row = (name, dims, what, io, note) => ({ name, dims, what, io, note });
const XALL = row("x", ["Dsrc", "Ddst", "C", "M"], "x[s,t] = the chunk device s wants to give device t; device s holds only its own row x[s]", "in", "split on Dsrc");
const SHAPES = {
  start: [XALL],
  addr: [XALL, row("x[s,t]", ["C", "M"], "one chunk: C token slots of M numbers, addressed to device t", "in")],
  keep: [row("out[d,d]", ["C", "M"], "the diagonal chunk never leaves device d", "out")],
  r1a: [row("chunk", ["C", "M"], "every link carries exactly one chunk this round; all D links are busy at once", "in"), row("out[d,(d−1)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  r2a: [row("chunk", ["C", "M"], "same size each round, so D−1 rounds move D−1 chunks per device", "in"), row("out[d,(d−2)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  r3a: [row("chunk", ["C", "M"], "last round: after it every receive slot is filled", "in"), row("out[d,(d−3)]", ["C", "M"], "lands in the receive buffer slot of the sender", "out")],
  done: [XALL, row("out", ["Ddst", "Dsrc", "C", "M"], "same data with the first two dims swapped: now device t holds column t, every chunk addressed to it", "out", "split on Ddst")],
  moe: [row("dispatched", ["Ddst", "Dsrc", "C", "M"], "read Dsrc as G and Ddst as E: expert e's buffer holds C slots from each of the G groups; unused slots are padding", "out", "= [E,G,C,M]")],
};

[1, 2, 3].forEach((r) => { SHAPES[`r${r}b`] = SHAPES[`r${r}a`]; });

function Chunk({ s, t, round, detail, sel, moving }) {
  const r = mod(t - s, D);
  const arrived = r <= round;
  const dev = arrived ? t : s;
  const slot = arrived ? s : t;
  const x = slotCX(dev, slot);
  const home = s === t;
  const c = EC[t];
  return (
    <g
      style={{
        transform: `translate(${x}px, ${arrived ? CY2 : CY1}px)`,
        transition: `transform .9s cubic-bezier(.3,.7,.3,1) ${(s * 60) % 240}ms`,
      }}
    >
      <rect
        x={-20} y={-22} width={40} height={44} rx={7} fill={`${c}${detail ? "1c" : "55"}`} stroke={moving ? T.ink : c}
        strokeWidth={moving ? 2.6 : home && round === 0 ? 2.4 : 1.4} opacity={sel === s || sel == null ? 1 : 0.6}
      />
      <text y={detail ? -8 : 4} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={T.ink}>{s}→{t}</text>
      {detail &&
        Array.from({ length: CFG.C }, (_, k) => (
          <rect key={k} x={-16 + k * 11} y={2} width={9} height={14} rx={2} fill={k < Math.min(COUNT[s][t], CFG.C) ? c : "transparent"} stroke={c} strokeDasharray={k < COUNT[s][t] ? "none" : "2 2"} />
        ))}
    </g>
  );
}

function Scene({ step, sel }) {
  const { round, fan, detail, phase } = step;
  const done = round >= D - 1 && phase !== "send";
  const landed = phase === "send" ? round - 1 : round; // chunks of this round only move on the recv line
  const arcs = [];
  if (fan) for (let t = 0; t < D; t++) if (t !== sel) arcs.push([sel, t]);
  if (phase) for (let s = 0; s < D; s++) arcs.push([s, mod(s + round, D)]);
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
          {[[CY1, "send x[j]  → device j"], [CY2, "recv out[j]  ← device j"]].map(([cy, label]) => (
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

      {chunks.map(([s, t]) => (
        <Chunk key={`${s}-${t}`} s={s} t={t} round={landed} detail={detail} sel={fan ? sel : null} moving={mod(t - s, D) === movingR || (fan && s === sel && t !== s)} />
      ))}

      {phase && (
        <text x={W / 2} y={H - 6} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.wire}>
          each device sends 1 chunk and receives 1 chunk — no link is shared
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
      subtitle={`D=${D} devices · every device sends a different chunk to every device`}
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
          <span>chunk <b style={{ color: T.soft }}>s→t</b> = from device s, for device t</span>
          {EC.map((c, e) => <span key={e} style={{ color: c }}>■ for device {e}</span>)}
        </>
      }
    />
  );
}
