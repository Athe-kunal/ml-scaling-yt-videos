import { T, mono } from "./theme";
import { CFG, EXPERT_COLORS as EC } from "./moeModel";

// Tensor lens: the einsum of the current step with each tensor drawn as a row
// of dimension boxes. A sharded dimension is cut into one stripe per device
// (device k = the color of expert k). Gating steps also show the S×E matrix
// that combine_weights collapses to, with the slot c each token landed in.

const SIZE = { G: CFG.G, S: CFG.S, E: CFG.E, C: CFG.C };

function Dim({ d, sharded }) {
  const n = SIZE[d];
  return (
    <div style={{ width: 54, height: 46, borderRadius: 6, border: `1px solid ${sharded ? T.soft : T.rule}`, background: T.well, position: "relative", overflow: "hidden", display: "grid", placeItems: "center" }}>
      {sharded && (
        <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: `repeat(${CFG.G}, 1fr)` }}>
          {EC.map((c) => <div key={c} style={{ background: `${c}38`, borderRight: `1px solid ${T.bg}` }} />)}
        </div>
      )}
      <div style={{ position: "relative", textAlign: "center", fontFamily: mono, lineHeight: 1.25 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: T.ink }}>{d}</div>
        <div style={{ fontSize: 9.5, color: T.dim }}>{n ?? "·"}</div>
      </div>
    </div>
  );
}

function Tensor({ x }) {
  return (
    <div style={{ display: "grid", gap: 6, justifyItems: "start" }}>
      {x.comm && (
        <div style={{ fontFamily: mono, fontSize: 11, fontWeight: 700, letterSpacing: "0.12em", color: T.wire, border: `1px solid ${T.wire}66`, background: `${T.wire}14`, borderRadius: 6, padding: "3px 10px" }}>
          {x.comm}
        </div>
      )}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontFamily: mono, fontSize: 12.5, fontWeight: 700, color: T.accent, minWidth: 92 }}>{x.name}</span>
        <div style={{ display: "flex", gap: 5 }}>
          {x.dims.length ? x.dims.map((d) => <Dim key={d} d={d} sharded={x.shard === d} />) : <span style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>scalar</span>}
        </div>
        {x.repl && (
          <span style={{ display: "inline-flex" }}>
            {EC.map((c, i) => <span key={c} style={{ width: 14, height: 30, borderRadius: 3, border: `1px solid ${c}`, background: `${c}30`, marginLeft: i ? -6 : 0 }} />)}
            <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginLeft: 8, alignSelf: "center" }}>copy on each</span>
          </span>
        )}
        {x.shard && <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>split {x.shard}</span>}
      </div>
      {x.note && <div style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginLeft: 102 }}>{x.note}</div>}
    </div>
  );
}

function Matrix({ gm, stage }) {
  const accepted = (k, e) => {
    if (stage >= 3 && k.t1.ok && k.e1 === e) return k.t1.pos;
    if (stage >= 5 && k.t2.ok && k.e2 === e) return k.t2.pos;
    return null;
  };
  const refused = (k, e) => (stage >= 3 && !k.t1.ok && k.e1 === e) || (stage >= 5 && !k.t2.ok && k.e2 === e);
  return (
    <div style={{ display: "grid", gap: 4, justifyContent: "start" }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: "0.1em", color: T.dim, marginBottom: 2 }}>THIS GROUP · [S, E] SLICE  ·  CELL = SLOT c IN THE EXPERT BUFFER</div>
      <div style={{ display: "grid", gridTemplateColumns: `38px repeat(${CFG.E}, 62px)`, gap: 4 }}>
        <span />
        {EC.map((c, e) => <span key={e} style={{ textAlign: "center", fontFamily: mono, fontSize: 11, fontWeight: 700, color: c }}>E{e}</span>)}
        {gm.toks.map((k) => [
          <span key={`l${k.s}`} style={{ fontFamily: mono, fontSize: 11, color: T.soft, alignSelf: "center" }}>x{k.s}</span>,
          ...gm.gates[k.s].map((g, e) => {
            const slot = accepted(k, e);
            const bad = refused(k, e);
            const first = stage >= 2 && e === k.e1;
            const second = stage >= 2 && e === k.e2;
            return (
              <div
                key={e}
                style={{
                  height: 30, borderRadius: 5, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 11, fontWeight: 700,
                  background: `color-mix(in srgb, ${EC[e]} ${Math.round(g * 70)}%, ${T.well})`,
                  border: `${first || second ? 2 : 1}px ${second ? "dashed" : "solid"} ${bad ? T.bad : first || second ? T.ink : T.rule}`,
                  color: bad ? T.bad : slot != null ? "#0A0E14" : "transparent", transition: "all .4s",
                }}
              >
                {bad ? "✕" : slot != null ? `c${slot}` : "·"}
              </div>
            );
          }),
        ])}
      </div>
    </div>
  );
}

export default function TensorView({ step, gm }) {
  const tn = step.tensors;
  return (
    <div style={{ padding: "18px 20px", display: "grid", gap: 18, minHeight: 440, alignContent: "start" }}>
      <div style={{ display: "grid", gap: 14 }}>
        {tn.ins.map((x) => <Tensor key={x.name} x={x} />)}
      </div>
      <div style={{ fontFamily: mono, fontSize: 12.5, color: T.accent2, display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 18 }}>↓</span>{tn.op}
      </div>
      {tn.outs && (
        <div style={{ display: "grid", gap: 14 }}>
          {tn.outs.map((x, i) => <Tensor key={`${x.name}${i}`} x={x} />)}
        </div>
      )}
      {step.scene === "group" && step.stage >= 0 && (
        <div style={{ borderTop: `1px solid ${T.rule}`, paddingTop: 16 }}>
          <Matrix gm={gm} stage={step.stage} />
        </div>
      )}
    </div>
  );
}
