"""Single source of truth: 17.125 → IEEE-754 fp32 by hand, with exact fractions.

Lessons only draw these rows; tests/test_toy.py checks them against the slide and struct.
"""
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60

X = F(137, 8)            # 17.125
INT, FRAC = 17, F(1, 8)  # 17 and 0.125
BIAS, MBITS = 127, 23


# ---------------------------------------------------------------- integer part: ÷ 2, keep remainders
@dataclass
class DivRow:
    n: int   # number divided
    q: int   # quotient (goes to the next row)
    r: int   # remainder = one bit


def int_ladder(n):
    rows = []
    while n > 0:
        rows.append(DivRow(n, n // 2, n % 2))
        n //= 2
    return rows


def read_up(rows):
    """Remainders read bottom-up give the binary digits, most significant first."""
    return "".join(str(r.r) for r in reversed(rows))


# ---------------------------------------------------------------- fraction part: × 2, keep the integer part
@dataclass
class MulRow:
    x: F     # fraction in
    y: F     # x × 2
    bit: int # integer part of y = one bit
    rest: F  # fractional part of y (the next row's input)


def frac_rows(x, max_rows=40):
    """Rows until the fraction is 0 ('zero'), an input repeats ('repeat'), or max_rows ('bits')."""
    rows, seen = [], set()
    while True:
        seen.add(x)
        y = 2 * x
        bit = int(y >= 1)
        rest = y - bit
        rows.append(MulRow(x, y, bit, rest))
        if rest == 0:
            return rows, "zero"
        if rest in seen:
            return rows, "repeat"
        if len(rows) >= max_rows:
            return rows, "bits"
        x = rest


def read_down(rows):
    """Integer parts read top-down give the digits after the point."""
    return "".join(str(r.bit) for r in rows)


def repeat_from(rows):
    """Index of the row whose input comes back as the last row's rest."""
    return next(i for i, r in enumerate(rows) if r.x == rows[-1].rest)


def dec(x, digits=None):
    """Exact decimal of a fraction with a power-of-2 (or 5·2^k) denominator, e.g. 0.125."""
    d = Decimal(x.numerator) / Decimal(x.denominator)
    s = format(d.normalize(), "f")
    return s if digits is None else format(round(d, digits), "f")


# ---------------------------------------------------------------- 17.125
INT_ROWS = int_ladder(INT)
FRAC_ROWS, FRAC_STOP = frac_rows(FRAC)
INT_BITS, FRAC_BITS = read_up(INT_ROWS), read_down(FRAC_ROWS)
FIXED = f"{INT_BITS}.{FRAC_BITS}"                 # 10001.001

# normalise: move the point left until one 1 is in front of it
E = len(INT_BITS) - 1                             # 4 places
LEAD, MANT = INT_BITS[0], (INT_BITS[1:] + FRAC_BITS).rstrip("0") or "0"   # 1 . 0001001
BIASED = E + BIAS                                 # 131
EXP_ROWS = int_ladder(BIASED)
EXP_BITS = read_up(EXP_ROWS).rjust(8, "0")        # 10000011
SIGN_BIT = "0" if X >= 0 else "1"
MANT23 = MANT.ljust(MBITS, "0")
BITS = SIGN_BIT + EXP_BITS + MANT23
NIBBLES = [BITS[i:i + 4] for i in range(0, 32, 4)]
HEX = "0x%08X" % int(BITS, 2)


def decode(sign, exp_bits, mant_bits):
    m = 1 + sum(F(int(b), 2 ** (i + 1)) for i, b in enumerate(mant_bits))
    return (-1) ** int(sign) * m * F(2) ** (int(exp_bits, 2) - BIAS)


# 10001.001 = 16 + 1 + 0.125: the place values of its 1s
PARTS = [F(2) ** (len(INT_BITS) - 1 - i) for i, b in enumerate(INT_BITS) if b == "1"] + \
        [F(1, 2 ** (i + 1)) for i, b in enumerate(FRAC_BITS) if b == "1"]


# ---------------------------------------------------------------- 0.1: never reaches 0 → fill 23 bits, round
@dataclass
class RunOut:
    rows: list       # the ×2 rows until the first repeat
    first: int       # row index the cycle returns to
    bits: str        # fraction bits after the point (enough for normalise + 23 + rounding)
    e: int           # exponent after normalising
    kept: str        # 23 mantissa bits before rounding
    cut: str         # the bits that don't fit (shown in red)
    rounded: str     # 23 bits after round-to-nearest-even
    hex: str
    stored: str      # exact decimal of what fp32 stores


def frac_bits(x, n):
    """First n bits after the point: keep doubling, even through repeats."""
    out = ""
    for _ in range(n):
        x *= 2
        out += str(int(x >= 1))
        x -= int(x >= 1)
    return out


def run_out(x, cut_show=8):
    rows, _ = frac_rows(x, max_rows=12)
    bits = frac_bits(x, 64)
    lead = bits.index("1")
    e = -(lead + 1)
    after = bits[lead + 1:]
    kept, cut = after[:MBITS], after[MBITS:]
    m = int(kept, 2)
    first_cut, rest = cut[0], cut[1:]
    if first_cut == "1" and ("1" in rest or m & 1):   # above half, or exactly half and odd
        m += 1
    rounded = format(m, f"0{MBITS}b")
    exp_bits = format(e + BIAS, "08b")
    word = int("0" + exp_bits + rounded, 2)
    stored = dec(decode("0", exp_bits, rounded))
    return RunOut(rows, repeat_from(rows), bits, e, kept, cut[:cut_show], rounded, "0x%08X" % word, stored)


POINT_ONE = run_out(F(1, 10))


# ---------------------------------------------------------------- subnormals
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(n):
    return str(n).translate(SUP)


def fmt_bits(x, ebits=8, mbits=MBITS):
    """Exact encoding of a Fraction in any binary format (round half to even, subnormals, overflow → inf)."""
    bias = 2 ** (ebits - 1) - 1
    sign = "1" if x < 0 else "0"
    a = abs(F(x))
    if a == 0:
        return sign + "0" * (ebits + mbits)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < F(2) ** e:
        e -= 1
    q = max(e, 1 - bias) - mbits                  # place value of the last kept bit
    m = a / F(2) ** q
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > F(1, 2) or (rem == F(1, 2) and fl % 2):
        fl += 1
    if fl == 2 ** (mbits + 1):
        fl, q = fl // 2, q + 1
    if fl >= 2 ** mbits:
        E = q + mbits
        if E > bias:
            return sign + "1" * ebits + "0" * mbits
        return sign + format(E + bias, f"0{ebits}b") + format(fl - 2 ** mbits, f"0{mbits}b")
    return sign + "0" * ebits + format(fl, f"0{mbits}b")


def fp32_bits(x):
    return fmt_bits(x, 8, MBITS)


def decode_sub(mant_bits):
    """Exponent field 0: value = 0.mantissa × 2^-126 (no hidden 1)."""
    return sum(F(int(b), 2 ** (i + 1)) for i, b in enumerate(mant_bits)) * F(1, 2 ** (BIAS - 1))


@dataclass
class Subnormal:
    x: F
    lead: str       # "1" of the normalised form
    frac: str       # bits after the point (normalised)
    e: int          # normalised exponent (−129)
    shift: int      # places the point moves left to reach 2^-126
    shifted: str    # bits after "0." once the exponent is −126
    mant23: str
    bits: str
    hex: str


def subnormal(lead, frac, e):
    x = (1 + sum(F(int(b), 2 ** (i + 1)) for i, b in enumerate(frac))) * F(2) ** e
    shift = (1 - BIAS) - e
    shifted = "0" * (shift - 1) + lead + frac
    mant23 = shifted.ljust(MBITS, "0")[:MBITS]
    bits = "0" + "0" * 8 + mant23
    return Subnormal(x, lead, frac, e, shift, shifted, mant23, bits, "0x%08X" % int(bits, 2))


SUB = subnormal("1", "01", -129)                 # 1.01₂ × 2^-129 ≈ 9.18e-40


@dataclass
class Fade:
    label: str      # 2⁻¹³⁰
    x: F
    bits: str       # all 32 bits
    kind: str       # normal | subnormal | zero
    sig: int        # significant bits left


def fade_row(k):
    x = F(1, 2 ** k)
    bits = fp32_bits(x)
    exp, mant = bits[1:9], bits[9:]
    if exp != "00000000":
        kind, sig = "normal", MBITS + 1
    elif "1" in mant:
        kind, sig = "subnormal", MBITS - mant.index("1")
    else:
        kind, sig = "zero", 0
    return Fade(f"2{sup(-k)}", x, bits, kind, sig)


FADING = [fade_row(k) for k in (126, 127, 130, 149, 150)]


# ---------------------------------------------------------------- fp32 vs fp16 vs bf16
@dataclass
class Fmt:
    name: str
    ebits: int
    mbits: int
    bias: int


FMTS = [Fmt("fp32", 8, 23, 127), Fmt("fp16", 5, 10, 15), Fmt("bf16", 8, 7, 127)]


@dataclass
class Encoded:
    name: str
    ebits: int
    mbits: int
    bias: int
    biased: int      # stored exponent (as an integer)
    exp_bits: str
    mant: str
    bits: str
    hex: str
    value: F         # what the format stores
    exact: bool


def bits_value(bits, ebits, mbits):
    bias = 2 ** (ebits - 1) - 1
    e, m = int(bits[1:1 + ebits], 2), bits[1 + ebits:]
    frac = sum(F(int(b), 2 ** (i + 1)) for i, b in enumerate(m))
    v = (frac * F(2) ** (1 - bias)) if e == 0 else (1 + frac) * F(2) ** (e - bias)
    return -v if bits[0] == "1" else v


def encode_in(x, f):
    bits = fmt_bits(x, f.ebits, f.mbits)
    v = bits_value(bits, f.ebits, f.mbits)
    eb = bits[1:1 + f.ebits]
    return Encoded(f.name, f.ebits, f.mbits, f.bias, int(eb, 2), eb, bits[1 + f.ebits:], bits,
                   "0x%0*X" % (len(bits) // 4, int(bits, 2)), v, v == x)


CMP = [encode_in(X, f) for f in FMTS]            # 17.125: exact in all three
CMP_01 = [encode_in(F(1, 10), f) for f in FMTS]  # 0.1: rounded in all three
