import { T, mono, FIELD } from "./theme";
import { BitStrip } from "./Bits";
import {
  FORMATS, decode, encode, exactValue, toNumber, binString, normalized, fracBits,
  exactSub, flushToZero, hex, shortValue, doubleParts, fields, exactDecimal,
} from "./float";

const { fp32, fp16, bf16 } = FORMATS;
const D32 = (raw) => decode(fp32, raw);
const neg = (s) => String(s).replace(/^-/, "−");
const sup = (n) => String(n).replace(/-/g, "−");

// ---------- tiny layout pieces (big type, few words) ----------
const Stage = ({ children, gap = 20 }) => (
  <div style={{ display: "grid", justifyItems: "center", alignContent: "center", gap, minHeight: 320, padding: "12px 0" }}>{children}</div>
);
const Big = ({ children, c = T.ink, size = 30, style }) => (
  <div style={{ fontFamily: mono, fontSize: size, fontWeight: 700, color: c, ...style }}>{children}</div>
);
const Note = ({ children, c = T.dim, size = 16 }) => <div style={{ fontFamily: mono, fontSize: size, color: c }}>{children}</div>;
const Row = ({ children, gap = 18, align = "center" }) => (
  <div style={{ display: "flex", gap, alignItems: align, flexWrap: "wrap", justifyContent: "center" }}>{children}</div>
);
const Chip = ({ children, c = T.accent }) => (
  <span style={{ fontFamily: mono, fontSize: 16, color: c, border: `1px solid ${c}66`, background: `${c}14`, borderRadius: 14, padding: "3px 12px" }}>{children}</span>
);
const P2 = ({ e, c = FIELD.exp }) => <>2<sup style={{ color: c, fontSize: "0.65em" }}>{sup(e)}</sup></>;
const Sub2 = () => <sub style={{ fontSize: "0.55em", color: T.dim }}>2</sub>;

// digits with their place value underneath; ones glow in binary
function PlaceRow({ s, base }) {
  const point = s.indexOf(".");
  const digits = [...s.replace(".", "")];
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
      {digits.map((d, i) => {
        const p = point - 1 - i;
        const one = base === 2 && d === "1";
        const val = one ? exactValue(D32(encode(fp32, 2 ** p))) : "";
        return (
          <div key={i} style={{ display: "flex", alignItems: "flex-start" }}>
            {i === point && <span style={{ fontFamily: mono, fontSize: 44, fontWeight: 700, color: T.wire, margin: "0 2px" }}>.</span>}
            <div style={{ display: "grid", justifyItems: "center", gap: 6, minWidth: 54 }}>
              <span style={{ fontFamily: mono, fontSize: 44, fontWeight: 700, color: base === 2 ? (one ? T.accent : T.rule) : T.ink }}>{d}</span>
              <span style={{ fontFamily: mono, fontSize: 16, color: T.soft }}>{base}<sup>{sup(p)}</sup></span>
              <span style={{ fontFamily: mono, fontSize: 14, color: T.accent, minHeight: 18 }}>{val}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const Labelled = ({ label, c = T.soft, f, raw, value, hl, size }) => (
  <div style={{ display: "grid", gridTemplateColumns: "96px auto", alignItems: "center", gap: "4px 14px" }}>
    <span style={{ fontFamily: mono, fontSize: 16, color: c, textAlign: "right" }}>{label}</span>
    <BitStrip f={f} raw={raw} hl={hl} size={size ?? (f.bits === 32 ? 16 : 24)} />
    <span />
    <span style={{ fontFamily: mono, fontSize: 20, fontWeight: 700, color: c }}>{value}</span>
  </div>
);

// ---------- shared numbers ----------
const R_9 = encode(fp32, 9.3125), d9 = D32(R_9), n9 = normalized(d9.M, d9.k);
const R_02 = encode(fp32, 0.2);
const X = -2343.53125, R_X = encode(fp32, X), dX = D32(R_X), nX = normalized(dX.M, dX.k);
const sub_x = (3n << 23n) | 0b01100001111101010000101n, sub_y = (3n << 23n) | 0b01100000011001011100001n;
const R_DIFF = exactSub(fp32, sub_x, sub_y), dDiff = D32(R_DIFF), nDiff = normalized(dDiff.M, dDiff.k);
const NARROW = -282960.039306640625;
const NARROW_S = (() => { const { M, k } = doubleParts(NARROW); return "−" + exactDecimal(M, k); })();

// ---------- scenes ----------
function Decimal() {
  return (
    <Stage>
      <PlaceRow s="327.849" base={10} />
      <Big size={22} c={T.soft}>300 + 20 + 7 + 0.8 + 0.04 + 0.009</Big>
    </Stage>
  );
}

function Binary() {
  const ones = [3, 0, -2, -4].map((p) => exactValue(D32(encode(fp32, 2 ** p))));
  return (
    <Stage>
      <PlaceRow s="1001.0101" base={2} />
      <Big size={26}>{ones.join(" + ")} = <span style={{ color: T.accent }}>{exactValue(d9)}</span></Big>
    </Stage>
  );
}

function Normalise() {
  return (
    <Stage gap={16}>
      <Big size={40}>{binString(d9.M, d9.k)}<Sub2 /></Big>
      <Note size={18}>point moves 3 places ←</Note>
      <Big size={40}>1.{n9.frac}<Sub2 /> × <P2 e={n9.e} /></Big>
      <Chip>always starts with 1</Chip>
    </Stage>
  );
}

function PointTwo() {
  const raw = fracBits(1, 5, 40); // 0.2 = 0.001100110011…
  const sig = raw.slice(raw.indexOf("1")); // 1 1001 1001 …  (starts at 2^-3)
  const kept = sig.slice(0, 24), cut = sig.slice(24, 32);
  const up = D32(R_02), down = D32(R_02 - 1n);
  const off = (d) => Math.abs(toNumber(d) - 0.2).toExponential(1);
  return (
    <Stage gap={14}>
      <Big size={26}>0.2 = 0.{raw.slice(0, 16)}…<Sub2 /></Big>
      <Big size={26}>= {kept[0]}.<span style={{ color: FIELD.mant }}>{kept.slice(1)}</span><span style={{ color: T.bad }}>{cut}…</span> × <P2 e={-3} /></Big>
      <Row gap={10}><Chip c={T.accent}>24 bits kept</Chip><Chip c={T.bad}>rest cut</Chip></Row>
      <div style={{ display: "grid", gap: 8, marginTop: 6 }}>
        <Row gap={14}><Big size={18} c={T.accent}>↑ {exactValue(up)}</Big><Note>off {off(up)} ✓ nearer</Note></Row>
        <Row gap={14}><Big size={18} c={T.soft}>↓ {exactValue(down)}</Big><Note>off {off(down)}</Note></Row>
      </div>
    </Stage>
  );
}

function ToBinary() {
  return (
    <Stage gap={16}>
      <Big size={30}><span style={{ color: FIELD.sign }}>−</span>2343.53125</Big>
      <Row gap={40}>
        <Note size={20}>2343 → {binString(BigInt(2343), 0)}</Note>
        <Note size={20}>.53125 → .{binString(17n, -5).split(".")[1]}</Note>
      </Row>
      <Big size={32}><span style={{ color: FIELD.sign }}>−</span>1.{nX.frac}<Sub2 /> × <P2 e={nX.e} /></Big>
    </Stage>
  );
}

function SignBit() {
  return (
    <Stage>
      <Row gap={30}>
        <Big size={64} c={FIELD.sign}>−</Big><Big size={40} c={T.dim}>→</Big><Big size={64} c={FIELD.sign}>1</Big>
      </Row>
      <Note size={20}>+ → 0</Note>
    </Stage>
  );
}

function DropOne() {
  const stored = dX.mant.toString(2).padStart(23, "0");
  const used = nX.frac.length;
  return (
    <Stage gap={18}>
      <Big size={36}><span style={{ color: T.bad, textDecoration: "line-through" }}>1.</span><span style={{ color: FIELD.mant }}>{nX.frac}</span></Big>
      <Note size={18}>the leading 1 is free: always 1</Note>
      <Big size={28} style={{ letterSpacing: 2 }}>
        <span style={{ color: FIELD.mant }}>{stored.slice(0, used)}</span><span style={{ color: T.rule }}>{stored.slice(used)}</span>
      </Big>
      <Chip>23 stored bits</Chip>
    </Stage>
  );
}

function Bias() {
  const segs = [["0", "zero · subnormal", T.bad, 1], ["1 … 254", "2⁻¹²⁶ … 2¹²⁷", FIELD.exp, 12], ["255", "∞ · NaN", T.bad, 1]];
  return (
    <Stage gap={22}>
      <Big size={34} c={FIELD.exp}>{nX.e} + 127 = {dX.exp} = {dX.exp.toString(2)}<Sub2 /></Big>
      <div style={{ display: "flex", width: "min(640px, 90%)", gap: 4 }}>
        {segs.map(([a, b, c, w]) => (
          <div key={a} style={{ flex: w, display: "grid", justifyItems: "center", gap: 6 }}>
            <div style={{ width: "100%", height: 26, borderRadius: 5, background: `${c}33`, border: `1.5px solid ${c}` }} />
            <span style={{ fontFamily: mono, fontSize: 16, color: c }}>{a}</span>
            <span style={{ fontFamily: mono, fontSize: 14, color: T.soft, whiteSpace: "nowrap" }}>{b}</span>
          </div>
        ))}
      </div>
      <Note>stored = exponent + 127</Note>
    </Stage>
  );
}

function Assemble() {
  const b = dX.mant.toString(2).padStart(23, "0");
  return (
    <Stage gap={22}>
      <Big size={30} style={{ letterSpacing: 1 }}>
        <span style={{ color: FIELD.sign }}>1</span>{" "}
        <span style={{ color: FIELD.exp }}>{dX.exp.toString(2)}</span>{" "}
        <span style={{ color: FIELD.mant }}>{b}</span>
      </Big>
      <Row gap={30}><Note c={FIELD.sign}>sign</Note><Note c={FIELD.exp}>exponent</Note><Note c={FIELD.mant}>significand</Note></Row>
      <Big size={40} c={T.accent}>{hex(fp32, R_X)}</Big>
    </Stage>
  );
}

function Zero() {
  return (
    <Stage>
      <div style={{ display: "grid", gap: 12 }}>
        <Labelled label="+0" f={fp32} raw={0n} value="0" />
        <Labelled label="−0" f={fp32} raw={0x80000000n} value="−0" />
      </div>
      <Row gap={40}><Big size={22}>+0 == −0 → <span style={{ color: T.accent }}>{String(0 === -0)}</span></Big><Big size={22}>1 / −0 → {neg(String(1 / -0)).replace("Infinity", "∞")}</Big></Row>
    </Stage>
  );
}

const fr = Math.fround;
const show = (v) => (Number.isNaN(v) ? "NaN" : v === Infinity ? "∞" : v === -Infinity ? "−∞" : String(v));

function Infinity_() {
  const max = toNumber(D32(0x7f7fffffn));
  const rows = [["max × 2", fr(max * 2)], ["1 / 0", 1 / 0], ["1 / ∞", 1 / Infinity], ["−1 / 0", -1 / 0]];
  return (
    <Stage>
      <Labelled label="+∞" f={fp32} raw={0x7f800000n} value="∞" hl={["exp"]} />
      <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: "8px 24px" }}>
        {rows.flatMap(([a, v]) => [<Big key={a} size={22} c={T.soft} style={{ textAlign: "right" }}>{a}</Big>, <Big key={a + "v"} size={22} c={T.accent}>→ {show(v)}</Big>])}
      </div>
    </Stage>
  );
}

function NaN_() {
  const rows = [["0 / 0", 0 / 0], ["∞ − ∞", Infinity - Infinity], ["0 × ∞", 0 * Infinity], ["√−1", Math.sqrt(-1)]];
  return (
    <Stage gap={16}>
      <Labelled label="NaN" f={fp32} raw={0x7fc00000n} value="NaN" hl={["exp", "mant"]} />
      <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: "6px 24px" }}>
        {rows.flatMap(([a, v]) => [<Big key={a} size={22} c={T.soft} style={{ textAlign: "right" }}>{a}</Big>, <Big key={a + "v"} size={22} c={T.bad}>→ {show(v)}</Big>])}
      </div>
      <Row gap={30}><Big size={22}>NaN == NaN → <span style={{ color: T.bad }}>{String(NaN === NaN)}</span></Big><Note size={18}>2²³ − 1 NaNs per sign</Note></Row>
    </Stage>
  );
}

function MaxMin() {
  return (
    <Stage gap={26}>
      <div style={{ display: "grid", gap: 6, justifyItems: "center" }}>
        <Labelled label="max" f={fp32} raw={0x7f7fffffn} value={shortValue(D32(0x7f7fffffn))} />
        <Note size={18}>(2 − 2⁻²³) × 2¹²⁷</Note>
      </div>
      <div style={{ display: "grid", gap: 6, justifyItems: "center" }}>
        <Labelled label="min normal" f={fp32} raw={0x00800000n} value={shortValue(D32(0x00800000n))} />
        <Note size={18}>1.0 × 2⁻¹²⁶</Note>
      </div>
    </Stage>
  );
}

function Subnormal() {
  const flushed = flushToZero(fp32, R_DIFF);
  return (
    <Stage gap={14}>
      <Row gap={12}><Chip c={T.bad}>exponent 0</Chip><Big size={22}>→ <span style={{ color: T.bad }}>0.</span>xxx × <P2 e={-126} /></Big></Row>
      <div style={{ display: "grid", gap: 8, marginTop: 6 }}>
        <Labelled label="x" f={fp32} raw={sub_x} value={shortValue(D32(sub_x), 6)} />
        <Labelled label="y" f={fp32} raw={sub_y} value={shortValue(D32(sub_y), 6)} />
        <Labelled label="x − y" c={T.accent} f={fp32} raw={R_DIFF} value={<>1.{nDiff.frac.slice(0, 6)}… × <P2 e={nDiff.e} /></>} />
      </div>
      <Row gap={14}><Note c={T.bad} size={18}>no subnormals → x − y = {exactValue(D32(flushed))}</Note><Chip c={T.bad}>x ≠ y ✗</Chip></Row>
    </Stage>
  );
}

function Gaps() {
  const W = 520, span = 0.25;
  const rows = [1, 2, 4].map((b) => {
    const ticks = [];
    for (let r = encode(bf16, b); ; r++) {
      const v = toNumber(decode(bf16, r));
      if (v > b + span + 1e-12) break;
      ticks.push(v);
    }
    return { b, ticks, gap: -7 + Math.log2(b) };
  });
  const tbl = [1, 2, 1024, 2 ** 24].map((v) => [v === 2 ** 24 ? "2²⁴" : String(v), decode(fp32, encode(fp32, v)).k]);
  return (
    <Stage gap={14}>
      <Note>bf16 ticks, same zoom</Note>
      <svg width={W + 200} height={rows.length * 56} style={{ maxWidth: "100%" }}>
        {rows.map(({ b, ticks, gap }, i) => (
          <g key={b} transform={`translate(70, ${i * 56 + 20})`}>
            <text x={-12} y={6} textAnchor="end" fill={T.soft} fontFamily={mono} fontSize={16}>{b}</text>
            <line x1={0} x2={W} y1={0} y2={0} stroke={T.rule} />
            {ticks.map((v) => <line key={v} x1={((v - b) / span) * W} x2={((v - b) / span) * W} y1={-10} y2={10} stroke={T.accent} strokeWidth={2} />)}
            <text x={W + 12} y={6} fill={FIELD.exp} fontFamily={mono} fontSize={16}>gap 2<tspan dy={-7} fontSize={12}>{sup(gap)}</tspan></text>
          </g>
        ))}
      </svg>
      <Note>fp32 gap at:</Note>
      <Row gap={26}>{tbl.map(([v, k]) => <Big key={v} size={20}>{v}: <span style={{ color: FIELD.exp }}><P2 e={k} /></span></Big>)}</Row>
    </Stage>
  );
}

function Gapless() {
  const ints = [16777214, 16777215, 16777216, 16777217, 16777218, 16777219, 16777220];
  return (
    <Stage gap={18}>
      <Row gap={8}>
        {ints.map((n) => {
          const ok = toNumber(D32(encode(fp32, n))) === n;
          return (
            <div key={n} style={{ display: "grid", justifyItems: "center", gap: 6, width: 96 }}>
              <div style={{ width: 22, height: 22, borderRadius: 11, border: `2px solid ${ok ? T.accent : T.bad}`, background: ok ? T.accent : "transparent" }} />
              <span style={{ fontFamily: mono, fontSize: 13, color: ok ? T.ink : T.bad }}>…{String(n).slice(-3)}</span>
            </div>
          );
        })}
      </Row>
      <Big size={24}>16777217 → <span style={{ color: T.accent }}>{exactValue(D32(encode(fp32, 16777217)))}</span></Big>
      <Row gap={12}><Chip c={T.wire}>tie → even</Chip><Chip>gap 2 above 2²⁴</Chip></Row>
    </Stage>
  );
}

function RawPlusOne() {
  const a = 0x49ffffffn, b = a + 1n;
  return (
    <Stage gap={16}>
      <Labelled label={hex(fp32, a)} f={fp32} raw={a} value={exactValue(D32(a))} />
      <Big size={22} c={T.wire}>+ 1</Big>
      <Labelled label={hex(fp32, b)} c={T.accent} f={fp32} raw={b} value={exactValue(D32(b))} />
      <Row gap={10}><Chip c={FIELD.mant}>1s → 0s</Chip><Chip c={FIELD.exp}>carry into exponent</Chip></Row>
      <Note size={18}>next float = raw integer + 1</Note>
    </Stage>
  );
}

function Layouts() {
  const fmts = [fp32, fp16, bf16];
  const stat = (f) => {
    const max = decode(f, ((BigInt((1 << f.e) - 2)) << BigInt(f.m)) | ((1n << BigInt(f.m)) - 1n));
    return { max: shortValue(max, 3), min: <P2 e={1 - f.bias} />, eps: <P2 e={-f.m} c={FIELD.mant} />, dig: (Math.log10(2) * (f.m + 1)).toFixed(1) };
  };
  return (
    <Stage gap={14}>
      {fmts.map((f) => (
        <div key={f.id} style={{ display: "grid", gridTemplateColumns: "60px 1fr", gap: 14, alignItems: "center", width: "min(720px, 95%)" }}>
          <Big size={20}>{f.id}</Big>
          <div style={{ display: "flex", gap: 3, width: `${(f.bits / 32) * 100}%` }}>
            {[["sign", 1], ["exp", f.e], ["mant", f.m]].map(([k, n]) => (
              <div key={k} style={{ flex: n, height: 34, borderRadius: 5, background: `${FIELD[k]}33`, border: `1.5px solid ${FIELD[k]}`, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 15, color: FIELD[k], fontWeight: 700 }}>{n}</div>
            ))}
          </div>
        </div>
      ))}
      <table style={{ fontFamily: mono, fontSize: 17, color: T.ink, borderSpacing: "22px 6px", marginTop: 6 }}>
        <thead><tr style={{ color: T.dim, fontSize: 14 }}><td /><td>max</td><td>min normal</td><td>gap at 1</td><td>≈ digits</td></tr></thead>
        <tbody>
          {fmts.map((f) => { const s = stat(f); return (
            <tr key={f.id}><td style={{ color: T.soft }}>{f.id}</td><td style={{ color: FIELD.exp }}>{s.max}</td><td style={{ color: FIELD.exp }}>{s.min}</td><td style={{ color: FIELD.mant }}>{s.eps}</td><td style={{ color: FIELD.mant }}>{s.dig}</td></tr>
          ); })}
        </tbody>
      </table>
    </Stage>
  );
}

function SameBits() {
  return (
    <Stage gap={22}>
      <Big size={34} c={T.wire}>0x44C0</Big>
      <div style={{ display: "grid", gap: 14 }}>
        <Labelled label="bf16" c={T.accent} f={bf16} raw={0x44c0n} value={exactValue(decode(bf16, 0x44c0n))} size={26} />
        <Labelled label="fp16" c={T.accent2} f={fp16} raw={0x44c0n} value={exactValue(decode(fp16, 0x44c0n))} size={26} />
      </div>
      <Row gap={30}>
        <Note size={18}>bf16: 1.1₂ × 2<sup>137−127</sup></Note>
        <Note size={18}>fp16: 1.0011₂ × 2<sup>17−15</sup></Note>
      </Row>
    </Stage>
  );
}

function ThreeFormats() {
  return (
    <Stage gap={14}>
      <Big size={26}>{neg(X)}</Big>
      <div style={{ display: "grid", gap: 10 }}>
        {[fp32, fp16, bf16].map((f) => {
          const r = encode(f, X), d = decode(f, r), err = Math.abs(toNumber(d) - X);
          return <Labelled key={f.id} label={f.id} c={err ? T.wire : T.accent} f={f} raw={r} value={<>{neg(exactValue(d))} <span style={{ fontSize: 15, color: T.dim }}>{err ? `off ${err}` : "exact"}</span></>} />;
        })}
      </div>
      <Note size={18}>fewer significand bits → coarser</Note>
    </Stage>
  );
}

function Nearest() {
  const { M } = doubleParts(NARROW);
  const s = M.toString(2); // 53 bits
  const r32 = encode(fp32, NARROW);
  return (
    <Stage gap={16}>
      <Note>double</Note>
      <Big size={24}>{NARROW_S}</Big>
      <Big size={17} style={{ letterSpacing: 1 }}>1.<span style={{ color: FIELD.mant }}>{s.slice(1, 24)}</span><span style={{ color: T.bad }}>{s.slice(24).replace(/0+$/, "")}</span></Big>
      <Row gap={10}><Chip c={FIELD.mant}>23 kept</Chip><Chip c={T.bad}>rounded off</Chip></Row>
      <Big size={26} c={T.accent}>fp32 → {neg(exactValue(D32(r32)))}</Big>
      <Row gap={34}>
        <div style={{ display: "grid", gap: 4, justifyItems: "center" }}><Note>half up</Note><Big size={18} c={T.bad}>72.5→73, 27.5→28 = 101%</Big></div>
        <div style={{ display: "grid", gap: 4, justifyItems: "center" }}><Note>half to even</Note><Big size={18} c={T.accent}>72.5→72, 27.5→28 = 100%</Big></div>
      </Row>
    </Stage>
  );
}

function Overflow() {
  return (
    <Stage gap={16}>
      <Big size={24}>{NARROW_S} ≈ −1.08 × <P2 e={18} /></Big>
      <div style={{ display: "grid", gap: 10 }}>
        {[fp32, bf16, fp16].map((f) => {
          const d = decode(f, encode(f, NARROW));
          return <Labelled key={f.id} label={f.id} c={d.kind === "inf" ? T.bad : T.accent} f={f} raw={encode(f, NARROW)} value={neg(exactValue(d)).replace("inf", "∞")} />;
        })}
      </div>
      <Row gap={10}><Chip c={T.bad}>fp16 max exponent 15 &lt; 18</Chip><Chip c={T.accent}>bf16 keeps fp32's range</Chip></Row>
    </Stage>
  );
}

// ---------- the lecture: chapter → steps ----------
export const CHAPTERS = [
  { id: "binary", label: "1 · Binary", sub: "place values · 2^e" },
  { id: "encode", label: "2 · Encoding", sub: "sign · exponent · significand" },
  { id: "special", label: "3 · Specials & spacing", sub: "0 · ∞ · NaN · gaps" },
  { id: "formats", label: "4 · Formats", sub: "fp32 · fp16 · bf16" },
];

const s = (ch, id, label, caption, Scene, fmt, raw, hl = []) => ({ ch, id, label, caption, Scene, fmt, raw: BigInt(raw), hl });

export const STEPS = [
  s("binary", "decimal", "decimal places", "Each digit × a power of 10", Decimal, "fp32", encode(fp32, 327.849)),
  s("binary", "binary", "binary places", "Binary: powers of 2", Binary, "fp32", R_9),
  s("binary", "normalise", "1.xxx × 2^e", "Move the point: one leading 1", Normalise, "fp32", R_9),
  s("binary", "point2", "0.2 never ends", "0.2 repeats forever, gets rounded", PointTwo, "fp32", R_02, ["mant"]),
  s("encode", "tobinary", "−2343.53125", "Write it in binary first", ToBinary, "fp32", R_X),
  s("encode", "sign", "sign bit", "Sign: one bit", SignBit, "fp32", R_X, ["sign"]),
  s("encode", "drop1", "drop the 1", "Store only the bits after 1.", DropOne, "fp32", R_X, ["mant"]),
  s("encode", "bias", "exponent + 127", "Exponent: add the bias 127", Bias, "fp32", R_X, ["exp"]),
  s("encode", "assemble", "32 bits", "Sign | exponent | significand", Assemble, "fp32", R_X),
  s("special", "zero", "±0", "Exponent 0, significand 0 → zero", Zero, "fp32", 0x80000000n),
  s("special", "inf", "±∞", "Exponent all 1s, significand 0 → ∞", Infinity_, "fp32", 0x7f800000n, ["exp", "mant"]),
  s("special", "nan", "NaN", "Exponent all 1s, significand ≠ 0 → NaN", NaN_, "fp32", 0x7fc00000n, ["exp", "mant"]),
  s("special", "maxmin", "max · min", "Largest and smallest normal", MaxMin, "fp32", 0x7f7fffffn),
  s("special", "subnormal", "subnormals", "Exponent 0: no hidden 1", Subnormal, "fp32", R_DIFF, ["exp", "mant"]),
  s("special", "gaps", "gaps double", "Gaps double at every power of 2", Gaps, "bf16", encode(bf16, 1)),
  s("special", "gapless", "16777217", "Past 2²⁴, odd integers vanish", Gapless, "fp32", encode(fp32, 16777216)),
  s("special", "rawplus", "raw + 1", "Next float = raw integer + 1", RawPlusOne, "fp32", 0x49ffffffn),
  s("formats", "layouts", "three layouts", "Same idea, different bit budgets", Layouts, "bf16", encode(bf16, 1)),
  s("formats", "samebits", "0x44c0", "Same 16 bits, two numbers", SameBits, "bf16", 0x44c0n),
  s("formats", "three", "one value, 3 formats", "Fewer bits, bigger rounding", ThreeFormats, "fp16", encode(fp16, X)),
  s("formats", "nearest", "round to even", "Narrowing rounds to nearest, ties to even", Nearest, "fp32", encode(fp32, NARROW), ["mant"]),
  s("formats", "overflow", "fp16 overflow", "fp16 runs out of exponent", Overflow, "fp16", encode(fp16, NARROW), ["exp"]),
];

export { fields };
