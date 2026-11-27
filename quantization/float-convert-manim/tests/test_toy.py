"""Every number in the lecture, checked against the slide (17.125 → 0 10000011 0001001…) and struct."""
import struct
from fractions import Fraction as F

import toy


def test_integer_ladder_17():
    rows = toy.int_ladder(17)
    assert [(r.n, r.q, r.r) for r in rows] == [(17, 8, 1), (8, 4, 0), (4, 2, 0), (2, 1, 0), (1, 0, 1)]
    assert toy.read_up(rows) == "10001"


def test_fraction_rows_0125_stop_at_zero():
    rows, why = toy.frac_rows(F(1, 8))
    assert [(r.x, r.y, r.bit) for r in rows] == [(F(1, 8), F(1, 4), 0), (F(1, 4), F(1, 2), 0), (F(1, 2), F(1), 1)]
    assert why == "zero"
    assert toy.read_down(rows) == "001"


def test_fixed_point_and_normalise():
    # the slide's middle line says 10001.100 — the ×2 table gives .001 (its own final bits agree)
    assert toy.FIXED == "10001.001"
    assert (toy.LEAD, toy.MANT, toy.E) == ("1", "0001001", 4)


def test_bias_and_131_ladder():
    assert toy.BIASED == 131
    rows = toy.int_ladder(131)
    assert [(r.n, r.r) for r in rows] == [(131, 1), (65, 1), (32, 0), (16, 0), (8, 0), (4, 0), (2, 0), (1, 1)]
    assert toy.read_up(rows) == toy.EXP_BITS == "10000011"


def test_final_bits_match_struct():
    bits = toy.SIGN_BIT + toy.EXP_BITS + toy.MANT23
    assert bits == "0" + "10000011" + "0001001" + "0" * 16
    assert int(bits, 2) == struct.unpack(">I", struct.pack(">f", 17.125))[0] == 0x41890000
    assert toy.HEX == "0x41890000"
    assert toy.NIBBLES == ["0100", "0001", "1000", "1001", "0000", "0000", "0000", "0000"]


def test_verify_back():
    assert toy.decode(toy.SIGN_BIT, toy.EXP_BITS, toy.MANT23) == F(137, 8)  # 17.125
    assert toy.PARTS == [16, 1, F(1, 8)]


def test_point_one_repeats():
    rows, why = toy.frac_rows(F(1, 10), max_rows=12)
    assert why == "repeat"
    assert [(r.x, r.y, r.bit, r.rest) for r in rows] == [
        (F(1, 10), F(1, 5), 0, F(1, 5)), (F(1, 5), F(2, 5), 0, F(2, 5)), (F(2, 5), F(4, 5), 0, F(4, 5)),
        (F(4, 5), F(8, 5), 1, F(3, 5)), (F(3, 5), F(6, 5), 1, F(1, 5))]
    assert toy.repeat_from(rows) == 1  # next input 0.2 = row 1's input


def test_point_one_runs_out_of_bits():
    r = toy.POINT_ONE
    assert r.e == -4
    assert r.kept == "10011001100110011001100"          # 23 mantissa bits
    assert r.cut.startswith("11")                       # rest > half → round up
    assert r.rounded == "10011001100110011001101"
    assert r.hex == "0x3DCCCCCD" == "0x%08X" % struct.unpack(">I", struct.pack(">f", 0.1))[0]
    assert r.stored == "0.100000001490116119384765625"


# ---------------------------------------------------------------- subnormals
def _raw(x):
    return struct.unpack(">I", struct.pack(">f", x))[0]


def test_subnormal_example_does_not_fit_normal():
    s = toy.SUB
    assert s.x == F(5, 4) * F(1, 2 ** 129)
    assert (s.lead, s.frac, s.e) == ("1", "01", -129)
    assert s.e + toy.BIAS == -2 < 1          # no stored exponent ≥ 1 is left for it


def test_subnormal_shift_and_bits():
    s = toy.SUB
    assert s.shift == 3                       # −129 → −126: point moves 3 left
    assert s.shifted == "00101"               # 0.00101 × 2^-126
    assert s.bits == "0" + "00000000" + "00101" + "0" * 18
    assert s.hex == "0x00140000" == "0x%08X" % _raw(float(s.x))
    assert toy.decode_sub(s.mant23) == s.x


def test_fp32_encoder_matches_struct():
    for x in [F(137, 8), F(1, 10), F(1, 2 ** 126), F(1, 2 ** 127), F(1, 2 ** 130), F(1, 2 ** 149),
              F(1, 2 ** 150), F(3, 2 ** 151), F(5, 4) * F(1, 2 ** 129)]:
        assert int(toy.fp32_bits(x), 2) == _raw(float(x)), x


def test_fading_rows():
    rows = toy.FADING
    assert [r.label for r in rows] == ["2⁻¹²⁶", "2⁻¹²⁷", "2⁻¹³⁰", "2⁻¹⁴⁹", "2⁻¹⁵⁰"]
    assert [r.kind for r in rows] == ["normal", "subnormal", "subnormal", "subnormal", "zero"]
    assert [r.sig for r in rows] == [24, 23, 20, 1, 0]
    assert rows[3].bits[9:] == "0" * 22 + "1"            # smallest subnormal: last bit only
    assert rows[4].bits == "0" * 32                      # tie with 0 → round to even → 0


# ---------------------------------------------------------------- fp32 vs fp16 vs bf16
def _half(x):
    return struct.unpack(">H", struct.pack(">e", x))[0]


def _bf16(x):
    raw = _raw(x)                                    # fp32 → bf16: round to nearest even on the top 16 bits
    return (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16


def test_formats_table():
    f = toy.FMTS
    assert [(r.name, r.ebits, r.mbits, r.bias) for r in f] == [("fp32", 8, 23, 127), ("fp16", 5, 10, 15), ("bf16", 8, 7, 127)]


def test_17125_in_three_formats():
    r32, r16, rbf = toy.CMP
    assert (r32.biased, r32.exp_bits, r32.mant, r32.hex) == (131, "10000011", "0001001" + "0" * 16, "0x41890000")
    assert (r16.biased, r16.exp_bits, r16.mant, r16.hex) == (19, "10011", "0001001000", "0x4C48")
    assert (rbf.biased, rbf.exp_bits, rbf.mant, rbf.hex) == (131, "10000011", "0001001", "0x4189")
    assert int(r16.bits, 2) == _half(17.125) and int(rbf.bits, 2) == _bf16(17.125)
    assert all(r.value == toy.X and r.exact for r in toy.CMP)
    assert r32.hex.startswith(rbf.hex)                # bf16 = top half of fp32


def test_point_one_in_three_formats():
    r32, r16, rbf = toy.CMP_01
    assert int(r32.bits, 2) == _raw(0.1) and int(r16.bits, 2) == _half(0.1) and int(rbf.bits, 2) == _bf16(0.1)
    assert (r32.hex, r16.hex, rbf.hex) == ("0x3DCCCCCD", "0x2E66", "0x3DCD")
    assert toy.dec(r16.value) == "0.0999755859375" and toy.dec(rbf.value) == "0.10009765625"
    assert abs(r32.value - F(1, 10)) < abs(r16.value - F(1, 10)) < abs(rbf.value - F(1, 10))
