import { useState } from "react";
import { T, mono, sans, btn, FIELD } from "./theme";
import { FORMATS, decode, fields, bitsOf, hex, exactValue, toNumber, encode, normalized, exactDecimal, shortValue } from "./float";

// which field a bit index (0 = leftmost) belongs to
export const fieldOf = (f, i) => (i === 0 ? "sign" : i <= f.e ? "exp" : "mant");

// A row of bits coloured by field. onFlip(i) makes it clickable. hl = fields to glow.
export function BitStrip({ f, raw, onFlip, hl = [], size, showIdx = false, dimFrom }) {
  const bits = bitsOf(f, raw);
  const cell = size ?? (f.bits === 32 ? 22 : 34);
  const glow = (fl) => hl.length === 0 || hl.includes(fl);
  return (
    <div style={{ display: "flex", gap: 2, alignItems: "flex-end" }}>
      {[...bits].map((b, i) => {
        const fl = fieldOf(f, i), c = FIELD[fl], on = b === "1";
        const gap = i === 1 || i === f.e + 1 ? cell * 0.4 : 0;
        const faded = !glow(fl) || (dimFrom != null && i >= dimFrom);
        return (
          <div key={i} style={{ marginLeft: gap, display: "grid", justifyItems: "center", gap: 2 }}>
            {showIdx && <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{f.bits - 1 - i}</span>}
            <button onClick={onFlip ? () => onFlip(i) : undefined} title={onFlip ? `flip bit ${f.bits - 1 - i}` : undefined}
              style={{
                width: cell, height: cell * 1.3, padding: 0, borderRadius: 4, fontFamily: mono, fontWeight: 700,
                fontSize: cell * 0.62, cursor: onFlip ? "pointer" : "default", transition: "all .25s",
                border: `1.5px solid ${c}${faded ? "44" : ""}`,
                background: on ? `${c}${faded ? "22" : "33"}` : T.well,
                color: on ? (faded ? `${c}77` : c) : faded ? T.rule : T.dim,
                boxShadow: !faded && hl.includes(fl) ? `0 0 10px ${c}55` : "none",
              }}>{b}</button>
          </div>
        );
      })}
    </div>
  );
}

const sup = (n) => String(n).replace(/-/g, "−");

// (−1)^s × 1.xxx₂ × 2^(exp − bias), coloured by field
export function Formula({ f, d, size = 18 }) {
  const s = { fontFamily: mono, fontSize: size };
  if (d.kind === "inf" || d.kind === "nan")
    return <div style={{ ...s, color: T.dim }}>exponent all 1s · significand {d.kind === "inf" ? "= 0 → ∞" : "≠ 0 → NaN"}</div>;
  const mant = d.mant.toString(2).padStart(f.m, "0");
  return (
    <div style={{ ...s, color: T.ink, display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 6 }}>
      <span style={{ color: FIELD.sign }}>{d.sign ? "−" : "+"}</span>
      <span>
        <span style={{ color: d.lead ? T.ink : T.bad, opacity: 0.8 }}>{d.lead}.</span>
        <span style={{ color: FIELD.mant }}>{mant}</span>
        <sub style={{ fontSize: size * 0.6, color: T.dim }}>2</sub>
      </span>
      <span style={{ color: T.dim }}>×</span>
      <span>2<sup style={{ color: FIELD.exp, fontSize: size * 0.7 }}>
        {d.exp === 0 ? `${sup(1 - f.bias)}` : `${d.exp}−${f.bias}`}
      </sup></span>
      {d.exp !== 0 && <span style={{ color: T.dim, fontSize: size * 0.8 }}>= 2<sup style={{ color: FIELD.exp }}>{sup(d.E)}</sup></span>}
      {d.kind === "subnormal" && <span style={{ color: T.bad, fontSize: size * 0.75 }}>subnormal</span>}
    </div>
  );
}

function Readout({ k, v, c = T.ink, big }) {
  return (
    <div style={{ display: "grid", gap: 2, minWidth: 0 }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>{k}</span>
      <span style={{ fontFamily: mono, fontSize: big ? 22 : 16, fontWeight: 700, color: c, overflowWrap: "anywhere" }}>{v}</span>
    </div>
  );
}

// float.exposed-style editor: click bits, ±1 the raw integer, type a value, switch format
export function BitEditor({ fmt, raw, setFmt, setRaw, hl, onReset }) {
  const f = FORMATS[fmt];
  const d = decode(f, raw);
  const { sign, exp, mant } = fields(f, raw);
  const [text, setText] = useState("");
  const [full, setFull] = useState(false);
  const ex = exactValue(d).replace("-", "−");
  const long = ex.length > 24;
  const top = (1n << BigInt(f.bits)) - 1n;
  const flip = (i) => setRaw(BigInt(raw) ^ (1n << BigInt(f.bits - 1 - i)));
  const step = (dx) => setRaw((BigInt(raw) + BigInt(dx) + top + 1n) & top);
  const switchFmt = (id) => { const g = FORMATS[id]; setFmt(id); setRaw(d.kind === "nan" ? encode(g, NaN) : encode(g, toNumber(d))); };
  const submit = () => {
    const t = text.trim().toLowerCase();
    if (!t) return;
    if (t.startsWith("0x")) { try { setRaw(BigInt(t) & top); } catch { /* ignore */ } }
    else {
      const x = t === "inf" ? Infinity : t === "-inf" ? -Infinity : t === "nan" ? NaN : Number(t);
      if (!Number.isNaN(x) || t === "nan") setRaw(encode(f, x));
    }
    setText("");
  };
  const finite = d.kind !== "inf" && d.kind !== "nan";
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "12px 16px", display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        {Object.keys(FORMATS).map((id) => (
          <button key={id} onClick={() => switchFmt(id)} style={{ ...btn(id === fmt), height: 28, fontSize: 14 }}>{id}</button>
        ))}
        <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, marginLeft: 4 }}>click bits</span>
        <div style={{ flex: 1 }} />
        <button onClick={() => step(-1)} style={{ ...btn(false), height: 28, fontSize: 14 }}>raw − 1</button>
        <button onClick={() => step(1)} style={{ ...btn(false), height: 28, fontSize: 14 }}>raw + 1</button>
        <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="0.1 or 0x44c0 ⏎"
          style={{ fontFamily: mono, fontSize: 14, height: 28, width: 130, padding: "0 8px", borderRadius: 7, border: `1px solid ${T.rule}`, background: T.well, color: T.ink }} />
        <button onClick={onReset} style={{ ...btn(false), height: 28, fontSize: 14 }}>reset</button>
      </div>

      <div style={{ display: "grid", justifyItems: "center", gap: 6, overflowX: "auto" }}>
        <BitStrip f={f} raw={raw} onFlip={flip} hl={hl} />
        <div style={{ display: "flex", gap: 22, fontFamily: mono, fontSize: 14 }}>
          <span style={{ color: FIELD.sign }}>sign {sign}</span>
          <span style={{ color: FIELD.exp }}>exponent {exp}</span>
          <span style={{ color: FIELD.mant }}>significand {mant.toString()}</span>
        </div>
      </div>

      <Formula f={f} d={d} />
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2.2fr) repeat(3, minmax(0, 1fr))", gap: 14 }}>
        <div style={{ display: "grid", gap: 2, minWidth: 0 }}>
          <Readout k="exact value" v={long ? shortValue(d).replace("-", "−") : ex} c={T.accent} big />
          {long && <button onClick={() => setFull(!full)} style={{ ...btn(full), height: 22, fontSize: 12, padding: "0 8px", justifySelf: "start" }}>{full ? "hide" : `all ${ex.replace(/[^0-9]/g, "").length} digits`}</button>}
        </div>
        <Readout k="bits (hex)" v={hex(f, raw)} />
        <Readout k="raw integer" v={BigInt(raw).toString()} />
        <Readout k="gap to next" v={finite ? <>2<sup>{sup(d.k)}</sup></> : "—"} c={T.wire} />
      </div>
      {long && full && <div style={{ fontFamily: mono, fontSize: 14, color: T.accent, overflowWrap: "anywhere", lineHeight: 1.5 }}>{ex}</div>}
    </div>
  );
}

export { normalized, exactDecimal };
