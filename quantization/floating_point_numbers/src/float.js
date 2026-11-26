// Exact IEEE-754 binary floats (fp32 / fp16 / bf16) with BigInt — the single source of truth
// for every number on screen. A finite value is held exactly as  (−1)^sign · M · 2^k  (M integer).

export const FORMATS = {
  fp32: { id: "fp32", bits: 32, e: 8, m: 23, bias: 127 },
  fp16: { id: "fp16", bits: 16, e: 5, m: 10, bias: 15 },
  bf16: { id: "bf16", bits: 16, e: 8, m: 7, bias: 127 },
};

const B = BigInt;
const bitLen = (M) => (M === 0n ? 0 : M.toString(2).length);
const mask = (n) => (1n << B(n)) - 1n;

export function fields(f, raw) {
  raw = B(raw);
  return {
    sign: Number((raw >> B(f.m + f.e)) & 1n),
    exp: Number((raw >> B(f.m)) & mask(f.e)),
    mant: raw & mask(f.m),
  };
}

export const compose = (f, sign, exp, mant) =>
  (B(sign) << B(f.m + f.e)) | (B(exp) << B(f.m)) | B(mant);

// kind: zero | subnormal | normal | inf | nan.  E = unbiased exponent, value = M·2^k
export function decode(f, raw) {
  const { sign, exp, mant } = fields(f, raw);
  const top = (1 << f.e) - 1;
  if (exp === top) return { sign, exp, mant, kind: mant === 0n ? "inf" : "nan", lead: null };
  if (exp === 0) {
    const E = 1 - f.bias;
    return { sign, exp, mant, kind: mant === 0n ? "zero" : "subnormal", E, M: mant, k: E - f.m, lead: 0 };
  }
  const E = exp - f.bias;
  return { sign, exp, mant, kind: "normal", E, M: (1n << B(f.m)) | mant, k: E - f.m, lead: 1 };
}

// exact decimal of M·2^k (M ≥ 0): M·2^-n = M·5^n / 10^n
export function exactDecimal(M, k) {
  if (M === 0n) return "0";
  if (k >= 0) return (M << B(k)).toString();
  const n = -k;
  const s = (M * 5n ** B(n)).toString().padStart(n + 1, "0");
  const int = s.slice(0, s.length - n), frac = s.slice(s.length - n).replace(/0+$/, "");
  return frac ? `${int}.${frac}` : int;
}

export function exactValue(d) {
  if (d.kind === "nan") return "NaN";
  if (d.kind === "inf") return d.sign ? "-inf" : "inf";
  return (d.sign ? "-" : "") + exactDecimal(d.M, d.k);
}

export function toNumber(d) {
  if (d.kind === "nan") return NaN;
  if (d.kind === "inf") return d.sign ? -Infinity : Infinity;
  const v = Number(d.M) * 2 ** d.k;
  return d.sign ? -v : v;
}

// exact parts of a JS double
export function doubleParts(x) {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, x);
  const bits = dv.getBigUint64(0);
  const neg = bits >> 63n === 1n;
  const exp = Number((bits >> 52n) & 0x7ffn);
  const mant = bits & mask(52);
  return exp === 0 ? { neg, M: mant, k: -1074 } : { neg, M: mant | (1n << 52n), k: exp - 1075 };
}

// round the exact value (−1)^neg · M · 2^k into format f, round-half-to-even → raw bits
export function roundParts(f, neg, M, k) {
  const sign = neg ? 1 : 0;
  if (M === 0n) return compose(f, sign, 0, 0n);
  const emin = 1 - f.bias, emax = f.bias;
  const e = k + bitLen(M) - 1; // value in [2^e, 2^(e+1))
  let q = Math.max(e, emin) - f.m; // exponent of the last kept bit
  const shift = q - k;
  let R;
  if (shift > 0) {
    const s = B(shift);
    R = M >> s;
    const rem = M & mask(shift), half = 1n << (s - 1n);
    if (rem > half || (rem === half && (R & 1n) === 1n)) R += 1n;
  } else R = M << B(-shift);
  if (R === 1n << B(f.m + 1)) { R >>= 1n; q += 1; } // rounding carried into a new power of two
  if (R === 0n) return compose(f, sign, 0, 0n);
  if (R < 1n << B(f.m)) return compose(f, sign, 0, R); // subnormal (q = emin − m)
  const E = q + f.m;
  if (E > emax) return compose(f, sign, (1 << f.e) - 1, 0n); // overflow → inf
  return compose(f, sign, E + f.bias, R - (1n << B(f.m)));
}

export function encode(f, x) {
  if (Number.isNaN(x)) return compose(f, 0, (1 << f.e) - 1, 1n << B(f.m - 1));
  if (!Number.isFinite(x)) return compose(f, x < 0 ? 1 : 0, (1 << f.e) - 1, 0n);
  const { neg, M, k } = doubleParts(x);
  return roundParts(f, neg, M, k);
}

// exact x − y for two finite floats of format f, rounded once
export function exactSub(f, rx, ry) {
  const a = decode(f, rx), b = decode(f, ry);
  const k = Math.min(a.k, b.k);
  const va = (a.sign ? -a.M : a.M) << B(a.k - k), vb = (b.sign ? -b.M : b.M) << B(b.k - k);
  const d = va - vb;
  return roundParts(f, d < 0n, d < 0n ? -d : d, k);
}

// a machine without subnormals: anything with stored exponent 0 becomes zero
export const flushToZero = (f, raw) => (fields(f, raw).exp === 0 ? compose(f, fields(f, raw).sign, 0, 0n) : B(raw));

// gap to the next float up (finite values): 2^k
export const ulpExp = (d) => d.k;

// "1001.0101" — the exact binary expansion of M·2^k
export function binString(M, k) {
  if (k >= 0) return (M << B(k)).toString(2);
  const s = M.toString(2).padStart(-k + 1, "0");
  const int = s.slice(0, s.length + k), frac = s.slice(s.length + k).replace(/0+$/, "");
  return frac ? `${int}.${frac}` : int;
}

// M·2^k = 1.frac × 2^e
export function normalized(M, k) {
  const s = M.toString(2);
  return { frac: s.slice(1).replace(/0+$/, ""), e: k + s.length - 1 };
}

// first n binary digits after the point of p/q
export function fracBits(p, q, n) {
  let out = "", r = p % q;
  for (let i = 0; i < n; i++) { r *= 2; out += r >= q ? "1" : "0"; if (r >= q) r -= q; }
  return out;
}

export const hex = (f, raw) => "0x" + B(raw).toString(16).toUpperCase().padStart(f.bits / 4, "0");
export const bitsOf = (f, raw) => B(raw).toString(2).padStart(f.bits, "0");

// short human value: exact if short, else 9 significant digits
export function shortValue(d, digits = 9) {
  const ex = exactValue(d);
  if (ex.replace(/[-.]/g, "").length <= 12) return ex;
  const v = toNumber(d);
  return Number.isFinite(v) ? v.toPrecision(digits).replace(/\.?0+(e|$)/, "$1") : ex;
}
