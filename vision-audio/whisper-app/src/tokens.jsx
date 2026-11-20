import { T, mono } from "./theme";
import { TOK, VOCAB_REGIONS, tokText, kindOf } from "./model";

// token chips, a token tape, and the vocabulary bar with masked (−∞) ranges

export const KIND_COL = { text: "#8DB4FF", special: "#FF8AC6", ts: "#FFC94D" };

export function TokChip({ id, state, showId = true, big, sub }) {
  const k = kindOf(id), c = KIND_COL[k];
  const lit = state === "new" || state === "active";
  return (
    <div style={{ display: "grid", justifyItems: "center", gap: 2, opacity: state === "dim" ? 0.4 : 1, transition: "all .3s" }}>
      <span style={{
        fontFamily: mono, fontSize: big ? 17 : 15, padding: big ? "5px 10px" : "3px 8px", borderRadius: 6, whiteSpace: "pre",
        border: `1.5px solid ${lit ? T.ink : c}`, background: lit ? `${c}55` : `${c}1c`, color: c,
        boxShadow: state === "new" ? `0 0 0 4px ${c}33` : "none", textDecoration: state === "struck" ? "line-through" : "none",
      }}>{tokText(id).replace(/^ /, "·")}</span>
      {showId && <span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>{id}</span>}
      {sub && <span style={{ fontFamily: mono, fontSize: 12, color: c }}>{sub}</span>}
    </div>
  );
}

export function Tape({ ids, from = 0, newAt, dimBefore, label, wrap = true, showId = true, subs }) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {label && <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, letterSpacing: "0.1em" }}>{label}</span>}
      <div style={{ display: "flex", gap: 5, flexWrap: wrap ? "wrap" : "nowrap", alignItems: "flex-start" }}>
        {ids.map((id, i) => <TokChip key={i} id={id} showId={showId} sub={subs?.[from + i]} state={from + i === newAt ? "new" : dimBefore != null && from + i < dimBefore ? "dim" : undefined} />)}
      </div>
    </div>
  );
}

// the 51865-wide vocabulary, drawn with three regions at readable widths.
// masks: [{ a, b, label, color }] id ranges set to −∞ (inclusive). pick: one id to point at.
const posOf = (id) => {
  let x = 0;
  for (const r of VOCAB_REGIONS) {
    if (id <= r.b) return x + ((id - r.a + 0.5) / (r.b - r.a + 1)) * r.w;
    x += r.w;
  }
  return 1;
};
export function VocabStrip({ masks = [], pick, marks = [], h = 34 }) {
  return (
    <div style={{ display: "grid", gap: 4, fontFamily: mono }}>
      <div style={{ position: "relative", height: h + 34, marginTop: 4 }}>
        <div style={{ position: "absolute", top: 18, left: 0, right: 0, height: h, display: "flex", borderRadius: 6, overflow: "hidden", border: `1px solid ${T.rule}` }}>
          {VOCAB_REGIONS.map((r) => <div key={r.name} style={{ width: `${r.w * 100}%`, background: `${r.color}30`, borderRight: `1px solid ${T.bg}` }} />)}
        </div>
        {VOCAB_REGIONS.reduce((acc, r) => { acc.items.push(
          <span key={r.name} style={{ position: "absolute", top: 0, left: `${acc.x * 100}%`, width: `${r.w * 100}%`, fontSize: 12, color: r.color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", paddingLeft: 4 }}>
            {r.name}
          </span>); acc.x += r.w; return acc; }, { x: 0, items: [] }).items}
        {masks.map((m, i) => {
          const a = posOf(m.a) - 0.0005, b = posOf(m.b) + 0.0005;
          return (
            <div key={i} title={m.label} style={{
              position: "absolute", top: 18, height: h, left: `${a * 100}%`, width: `${Math.max(0.004, b - a) * 100}%`,
              background: `repeating-linear-gradient(135deg, ${T.bg} 0 5px, ${m.color ?? T.bad}66 5px 7px)`, borderLeft: `1px solid ${m.color ?? T.bad}`, borderRight: `1px solid ${m.color ?? T.bad}`, transition: "all .4s",
            }} />
          );
        })}
        {marks.map((m, i) => <div key={i} style={{ position: "absolute", top: 14, height: h + 8, left: `${posOf(m.id) * 100}%`, width: 2, background: m.color ?? T.soft }} />)}
        {pick != null && (
          <div style={{ position: "absolute", top: 14, left: `${posOf(pick) * 100}%`, transform: "translateX(-50%)", display: "grid", justifyItems: "center" }}>
            <div style={{ width: 3, height: h + 8, background: T.ink, borderRadius: 2 }} />
            <span style={{ fontSize: 13, color: T.ink, whiteSpace: "pre" }}>▲ {tokText(pick).replace(/^ /, "·")}</span>
          </div>
        )}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.dim, marginTop: 14 }}>
        <span>0</span><span>{TOK.eot}</span><span>{TOK.ts0}</span><span>51864</span>
      </div>
    </div>
  );
}

export function MaskLegend({ masks }) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {masks.map((m, i) => (
        <div key={i} style={{ display: "flex", gap: 10, alignItems: "baseline", fontFamily: mono, fontSize: 15 }}>
          <span style={{ width: 14, height: 14, flexShrink: 0, borderRadius: 3, background: `repeating-linear-gradient(135deg, ${T.bg} 0 3px, ${m.color ?? T.bad} 3px 5px)`, border: `1px solid ${m.color ?? T.bad}` }} />
          <span style={{ color: m.color ?? T.bad, minWidth: 150 }}>{m.rule}</span>
          <span style={{ color: T.soft }}>{m.label}</span>
        </div>
      ))}
    </div>
  );
}
