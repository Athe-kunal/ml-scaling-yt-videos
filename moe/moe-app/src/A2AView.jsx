import { useState } from "react";
import { f2 } from "./charts";
import { T, mono, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, SEED, buildModel } from "./moeModel";
import Stepper from "./Stepper";
import MoEBlockMap from "./MoEBlockMap";

// All-to-all as a stepper. D devices each own Ed experts (E = D·Ed in total).
// Device s holds D token bundles; bundle t carries the tokens bound for the Ed
// experts living on device t. After the collective, device t holds the bundles
// from every s: the D×D bundle grid is transposed. Each token's journey is set by
// the router: token chips at home, straight-line moves into the slot of its expert.
// Schedule shown: D-1 pairwise rounds, round r sends to (d+r) mod D.

const D = CFG.G;
const mod = (a, n) => ((a % n) + n) % n;

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

const devOf = (cfg, e) => Math.floor(e / cfg.Ed);
// ids of the tokens of group s that have at least one surviving expert on device t
const why = (ch, second) => (!ch.capOk && second ? "capacity" : second && !ch.diceOk ? "dice" : "capacity");
// ---- the router's table for the followed device, with where each choice is now ------
function Inspector({ cfg, step, sel }) {
  const { Ed, groups } = cfg;
  const g = groups[sel];
  const pid = step.id;
  const landed = step.phase === "send" ? step.round - 1 : step.round;
  const where = (ch) => {
    const dev = devOf(cfg, ch.e);
    const r = mod(dev - sel, D);
    if (step.back != null && r <= step.back) return ["result home", T.accent];
    if (r <= landed) {
      if (step.ffn) return [step.ffn === "up" ? "in W_up" : "in W_down", T.ink];
      if (step.back != null || pid === "ffn_up" || pid === "ffn_dn") return [`y waits on D${dev}`, T.soft];
      return [`arrived at D${dev}`, T.soft];
    }
    return [dev === sel ? "stays" : `→ D${dev}`, T.dim];
  };
  const cell = (ch, second) => {
    if (!ch.ok) return <span style={{ color: T.bad }}>dropped · {why(ch, second)}</span>;
    const dev = devOf(cfg, ch.e);
    const [txt, col] = where(ch);
    return (
      <span>
        <b style={{ color: EC[dev] }}>E{ch.e}</b> <span style={{ color: T.dim }}>D{dev}</span> · g {f2(ch.w)} <span style={{ color: col }}>· {txt}</span>
      </span>
    );
  };
  const combined = pid === "combine";
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 8 }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.12em", color: T.dim }}>ROUTER DECISIONS · TOKENS OF DEVICE {sel} · g = gate weight</div>
      <div style={{ display: "grid", gridTemplateColumns: "36px 1fr 1fr", gap: "6px 12px", fontFamily: mono, fontSize: 11.5, alignItems: "start" }}>
        <span style={{ color: T.dim }}>token</span><span style={{ color: T.dim }}>1st choice</span><span style={{ color: T.dim }}>2nd choice</span>
        {g.toks.map((t) => (
          <div key={t.s} style={{ display: "contents" }}>
            <b style={{ color: T.ink }}>t{t.s}</b>
            {cell(t.t1, false)}
            {cell(t.t2, true)}
            {combined && (
              <div style={{ gridColumn: "1 / -1", color: T.accent, fontSize: 11, margin: "-2px 0 2px 48px" }}>
                out[t{t.s}] = {t.final.length ? t.final.map((f) => `${f2(f.w)}·y(E${f.e})`).join(" + ") : "0 (dropped)"}
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
      subtitle={`D=${D} devices · ${Ed} expert${Ed > 1 ? "s" : ""} per device (E=${cfg.E}) · the router decides which tokens each device sends where`}
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
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow device</span>
            {Array.from({ length: D }, (_, k) => (
              <button key={k} onClick={() => setSel(k)} style={{ ...btn(k === sel, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
        </div>
      )}
      scene={(step) => <MoEBlockMap step={step} Ed={Ed} E={cfg.E} sel={sel} cfg={cfg} />}
      extra={(step) => (
        <div style={{ display: "grid", gap: 12 }}>
          <Inspector cfg={cfg} step={step} sel={sel} />
        </div>
      )}
      glossary={cfg.GLOSSARY}
      shapes={(step) => cfg.SHAPES[step.sid]}
      legend={
        <>
          <span>bright = followed device · digit in a slot = token id · dashed red = dropped (capacity or random routing){Ed > 1 ? " · one shade per local expert" : ""}</span>
          {EC.map((c, e) => <span key={e} style={{ color: c }}>■ device {e}</span>)}
        </>
      }
    />
  );
}
