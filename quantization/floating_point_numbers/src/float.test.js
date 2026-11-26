// Every number the app shows comes from float.js; these pin it to the article's values.
//   run: npm test   (node --test, no extra deps)
import test from "node:test";
import assert from "node:assert/strict";
import {
  FORMATS, decode, encode, exactValue, toNumber, binString, normalized, fracBits,
  exactSub, flushToZero, ulpExp, fields,
} from "./float.js";

const { fp32, fp16, bf16 } = FORMATS;

test("binary place values: 1001.0101₂ = 9.3125, normalised 1.0010101 × 2^3", () => {
  const raw = encode(fp32, 9.3125);
  const d = decode(fp32, raw);
  assert.equal(exactValue(d), "9.3125");
  assert.equal(binString(d.M, d.k), "1001.0101");
  assert.deepEqual(normalized(d.M, d.k), { frac: "0010101", e: 3 });
});

test("0.2 repeats in binary and rounds up to 0.20000000298…", () => {
  assert.equal(fracBits(1, 5, 12), "001100110011");
  const raw = encode(fp32, 0.2);
  assert.equal(raw, 0x3e4ccccdn);
  assert.equal(exactValue(decode(fp32, raw)), "0.20000000298023223876953125");
  assert.equal(exactValue(decode(fp32, raw - 1n)), "0.199999988079071044921875");
});

test("−2343.53125 encodes to 1 10001010 00100100111100010000000", () => {
  const raw = encode(fp32, -2343.53125);
  assert.equal(raw, 0xc5127880n);
  assert.deepEqual(fields(fp32, raw), { sign: 1, exp: 138, mant: 0b00100100111100010000000n });
  const d = decode(fp32, raw);
  assert.equal(binString(d.M, d.k), "100100100111.10001");
  assert.deepEqual(normalized(d.M, d.k), { frac: "0010010011110001", e: 11 });
});

test("special values", () => {
  assert.equal(encode(fp32, 0), 0x00000000n);
  assert.equal(encode(fp32, -0), 0x80000000n);
  assert.equal(encode(fp32, Infinity), 0x7f800000n);
  assert.equal(encode(fp32, -Infinity), 0xff800000n);
  assert.equal(decode(fp32, encode(fp32, NaN)).kind, "nan");
  assert.equal(decode(fp32, 0x7f800001n).kind, "nan");
  assert.equal(decode(fp32, 0x7f800000n).kind, "inf");
  // overflow: max × 2 → inf
  assert.equal(encode(fp32, 3.4028234663852886e38 * 2), 0x7f800000n);
});

test("max and smallest normal fp32", () => {
  const max = decode(fp32, 0x7f7fffffn);
  assert.equal(toNumber(max), 2 ** 128 - 2 ** 104);
  assert.equal(toNumber(max).toPrecision(9), "3.40282347e+38");
  const min = decode(fp32, 0x00800000n);
  assert.equal(toNumber(min), 2 ** -126);
  assert.equal(toNumber(min).toPrecision(9), "1.17549435e-38");
});

test("subnormal: exponent 0 means 0.xxx × 2^-126", () => {
  const d = decode(fp32, 0x00000c69n); // stored 00000000000110001101001
  assert.equal(d.kind, "subnormal");
  assert.deepEqual(normalized(d.M, d.k), { frac: "10001101001", e: -138 });
});

test("x ≠ y but x − y = 0 without subnormals", () => {
  // x = 1.01100001111101010000101 × 2^-124, y = 1.01100000011001011100001 × 2^-124
  const x = (3n << 23n) | 0b01100001111101010000101n;
  const y = (3n << 23n) | 0b01100000011001011100001n;
  const diff = exactSub(fp32, x, y);
  const d = decode(fp32, diff);
  assert.equal(d.kind, "subnormal");
  assert.deepEqual(normalized(d.M, d.k), { frac: "1000111101001", e: -132 });
  assert.equal(flushToZero(fp32, diff), 0n);
});

test("gaps: 2^-23 at 1.0, 2^-24 below; 16777217 does not fit", () => {
  assert.equal(ulpExp(decode(fp32, encode(fp32, 1))), -23);
  assert.equal(ulpExp(decode(fp32, encode(fp32, 0.75))), -24);
  const r = encode(fp32, 16777217); // tie → even
  assert.equal(r, 0x4b800000n);
  assert.equal(exactValue(decode(fp32, r)), "16777216");
  assert.equal(exactValue(decode(fp32, r + 1n)), "16777218");
});

test("next float = raw integer + 1", () => {
  assert.equal(exactValue(decode(fp32, 0x49ffffffn)), "2097151.875");
  assert.equal(exactValue(decode(fp32, 0x4a000000n)), "2097152");
});

test("0x44c0 is 1536 in bf16 but 4.75 in fp16", () => {
  assert.equal(exactValue(decode(bf16, 0x44c0n)), "1536");
  assert.equal(exactValue(decode(fp16, 0x44c0n)), "4.75");
});

test("narrowing −282960.039306640625: fp32 rounds, fp16 overflows", () => {
  const x = -282960.039306640625;
  const r32 = encode(fp32, x);
  assert.equal(r32, 0xc88a2a01n);
  assert.equal(exactValue(decode(fp32, r32)), "-282960.03125");
  assert.equal(decode(fp16, encode(fp16, x)).kind, "inf");
  assert.equal(encode(fp16, x), 0xfc00n);
});

test("round half to even", () => {
  assert.equal(exactValue(decode(fp16, encode(fp16, 2049))), "2048"); // tie, 2048 even
  assert.equal(exactValue(decode(fp16, encode(fp16, 2051))), "2052"); // tie, 2052 even
  assert.equal(exactValue(decode(fp16, encode(fp16, -2343.53125))), "-2344");
  assert.equal(exactValue(decode(bf16, encode(bf16, -2343.53125))), "-2336");
});
