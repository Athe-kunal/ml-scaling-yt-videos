import { useState } from "react";
import { T, mono, sans } from "./theme";
import { PatchImage, Row } from "./draw";
import { SUBJECTS, renderSubject, vitDims, fmtN } from "./model";

// "ViT-B/16": the letter = how big the transformer is, the number = patch side.
// One page: pick a model, see both halves of the name, compare all three.

const SIZES = {
  B: { name: "Base", D: 768, L: 12, h: 12, M: 3072, K: 1000 },
  L: { name: "Large", D: 1024, L: 24, h: 16, M: 4096, K: 1000 },
};
const MODELS = [
  { id: "B/32", size: "B", P: 32 },
  { id: "B/16", size: "B", P: 16 },
  { id: "L/14", size: "L", P: 14 },
];
const RES = 224;
const SIZE_C = "#C79BFF", PATCH_C = "#FF8AC6";
const img = renderSubject(SUBJECTS[0], RES);

const stats = (m) => {
  const cfg = SIZES[m.size], d = vitDims(RES, m.P, cfg);
  return { ...d, enc: d.params - (d.D * d.K + d.K) }; // encoder only, no classifier head
};

function Big({ m }) {
  return (
    <div style={{ fontFamily: mono, fontSize: 64, fontWeight: 700, letterSpacing: "-0.02em", lineHeight: 1 }}>
      <span style={{ color: T.soft }}>ViT-</span>
      <span style={{ color: SIZE_C }}>{m.size}</span>
      <span style={{ color: T.soft }}>/</span>
      <span style={{ color: PATCH_C }}>{m.P}</span>
    </div>
  );
}

// the transformer: one bar per block, bar width ∝ hidden width D
function Stack({ m }) {
  const s = SIZES[m.size], w = (s.D / 1024) * 300;
  return (
    <div style={{ display: "grid", gap: 10, justifyItems: "start" }}>
      <div style={{ fontFamily: sans, fontSize: 22, color: SIZE_C, fontWeight: 600 }}>{m.size} = {s.name}</div>
      <div style={{ display: "grid", gap: 2 }}>
        {Array.from({ length: s.L }, (_, i) => (
          <div key={i} style={{ width: w, height: s.L > 12 ? 9 : 19, borderRadius: 2, background: `${SIZE_C}${i % 2 ? "88" : "bb"}`, transition: "all .4s" }} />
        ))}
      </div>
      <div style={{ display: "grid", gap: 4, fontFamily: mono, fontSize: 17, color: T.ink }}>
        <span><b style={{ color: SIZE_C }}>{s.L}</b> layers</span>
        <span>width <b style={{ color: SIZE_C }}>{s.D}</b></span>
        <span><b style={{ color: SIZE_C }}>{s.h}</b> heads</span>
      </div>
    </div>
  );
}

function Patches({ m, d }) {
  return (
    <div style={{ display: "grid", gap: 10, justifyItems: "start" }}>
      <div style={{ fontFamily: sans, fontSize: 22, color: PATCH_C, fontWeight: 600 }}>/{m.P} = {m.P}×{m.P} px patches</div>
      <PatchImage img={img} P={m.P} size={300} grid={1} />
      <div style={{ fontFamily: mono, fontSize: 17, color: T.ink }}>
        {d.g}×{d.g} = {d.N} patches + CLS = <b style={{ color: PATCH_C }}>{d.Tn} tokens</b>
      </div>
    </div>
  );
}

// all three models side by side, one bar chart per quantity
function Compare({ sel, setSel }) {
  const rows = MODELS.map((m) => ({ m, s: stats(m) }));
  const charts = [
    { k: "tokens", get: (s) => s.Tn, fmt: (v) => `${v}` },
    { k: "params", get: (s) => s.enc, fmt: (v) => fmtN(v) },
    { k: "compute", get: (s) => s.macs, fmt: (v) => `${(v / 1e9).toFixed(1)} G` },
  ];
  return (
    <Row gap={30} style={{ alignItems: "flex-start" }}>
      {charts.map((c) => {
        const mx = Math.max(...rows.map((r) => c.get(r.s)));
        return (
          <div key={c.k} style={{ display: "grid", gap: 8, minWidth: 230 }}>
            <div style={{ fontFamily: sans, fontSize: 18, color: T.soft, fontWeight: 600 }}>{c.k}</div>
            {rows.map(({ m, s }) => {
              const on = m.id === sel.id, v = c.get(s);
              return (
                <div key={m.id} onClick={() => setSel(m)} style={{ display: "grid", gridTemplateColumns: "64px 1fr 66px", gap: 8, alignItems: "center", cursor: "pointer", opacity: on ? 1 : 0.5 }}>
                  <span style={{ fontFamily: mono, fontSize: 15, color: on ? T.ink : T.soft, fontWeight: on ? 700 : 400 }}>{m.id}</span>
                  <div style={{ height: 18, borderRadius: 3, background: T.well }}>
                    <div style={{ height: "100%", width: `${(v / mx) * 100}%`, borderRadius: 3, background: on ? T.accent : T.soft, transition: "width .4s" }} />
                  </div>
                  <span style={{ fontFamily: mono, fontSize: 15, color: on ? T.ink : T.soft, textAlign: "right" }}>{c.fmt(v)}</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </Row>
  );
}

export default function SizesView() {
  const [sel, setSel] = useState(MODELS[1]);
  const d = stats(sel);
  return (
    <div style={{ maxWidth: 1480, margin: "0 auto", padding: "20px clamp(16px, 3vw, 36px) 40px", display: "grid", gap: 22 }}>
      <Row gap={10}>
        {MODELS.map((m) => {
          const on = m.id === sel.id;
          return (
            <button key={m.id} onClick={() => setSel(m)} style={{
              fontFamily: mono, fontSize: 20, fontWeight: 700, padding: "10px 18px", borderRadius: 10, cursor: "pointer",
              border: `1.5px solid ${on ? T.accent : T.rule}`, background: on ? `${T.accent}18` : T.panel, color: on ? T.accent : T.soft,
            }}>ViT-{m.id}</button>
          );
        })}
        <span style={{ fontFamily: mono, fontSize: 15, color: T.dim, marginLeft: 8 }}>all at {RES}×{RES}</span>
      </Row>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 14, padding: "26px 28px", display: "grid", gap: 26 }}>
        <Big m={sel} />
        <Row gap={60} style={{ alignItems: "flex-start" }}>
          <Stack m={sel} />
          <Patches m={sel} d={d} />
        </Row>
      </div>

      <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 14, padding: "22px 28px", display: "grid", gap: 14 }}>
        <Compare sel={sel} setSel={setSel} />
        <div style={{ fontFamily: mono, fontSize: 15, color: T.dim }}>B/32 → B/16: same network, smaller patches → 4× the tokens</div>
      </div>
    </div>
  );
}
