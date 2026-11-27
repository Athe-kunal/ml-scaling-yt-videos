"""Lesson 6 · subnormals: too small for 2^-126 → exponent bits 0, value 0.xxx × 2^-126."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
import toy

S = toy.SUB
EMIN = 1 - toy.BIAS                     # −126


def mn(n):
    """a real minus sign in plain text"""
    return str(n).replace("-", "−")


def sci(x):
    """9.18 × 10⁻⁴⁰"""
    m, e = f"{float(x):.2e}".split("e")
    return f"{m} × 10{toy.sup(int(e))}"


class L6_Subnormal(Lesson):
    NUM, TITLE, SUB = 6, "Subnormal numbers", "smaller than 2⁻¹²⁶: no hidden 1"
    SECTIONS = ["title", "too_small", "convert", "fading"]
    TOPICS = {"too_small": "too small for 2⁻¹²⁶", "convert": f"convert {sci(S.x)}",
              "fading": "precision fades, then 0"}

    def too_small(self):
        lo = M(r"1.0_2 \times 2^{" + str(EMIN) + "}", 60, INK, t2c={str(EMIN): EXP_C}).move_to([-2.2, 2.5, 0])
        lo_bits = VGroup(num("exponent bits", 30, DIM), bitrow("00000001", 36, EXP_C)).arrange(RIGHT, buff=0.25)
        lo_bits.next_to(lo, RIGHT, buff=0.8)
        self.play(Write(lo), FadeIn(lo_bits))
        self.say(f"smallest normal: 2{toy.sup(EMIN)}")
        self.beat()

        x = M(r"x = " + S.lead + "." + S.frac + r"_2 \times 2^{" + str(S.e) + "}", 60, INK, t2c={str(S.e): BAD}).move_to([-2.2, 1.1, 0])
        approx = num(f"≈ {sci(S.x)}", 40, DIM).next_to(x, RIGHT, buff=0.6)
        hand = num(f"by hand: ×2 table starts with {-S.e} zero bits", 32, DIM).next_to(x, DOWN, buff=0.35).align_to(x, LEFT)
        self.play(Write(x), FadeIn(approx))
        self.play(FadeIn(hand))
        self.say("x is smaller than that")
        self.beat()

        b = M(f"{S.e} + {toy.BIAS} = {S.e + toy.BIAS}", 60, INK, t2c={str(S.e + toy.BIAS): BAD}).move_to([-2.2, -0.6, 0])
        bad = num("< 1  ✗  no room", 40, BAD).next_to(b, RIGHT, buff=0.6)
        self.play(Write(b))
        self.play(FadeIn(bad))
        self.say(f"stored exponent would be {mn(S.e + toy.BIAS)}", BAD)
        self.beat()

        rule = VGroup(num("exponent bits", 36, DIM), bitrow("00000000", 44, EXP_C), num("→", 40, DIM),
                      M(r"0.\text{xxx} \times 2^{" + str(EMIN) + "}", 56, INK, t2c={str(EMIN): EXP_C}))
        rule.arrange(RIGHT, buff=0.3).move_to([0, -2.2, 0])
        box = SurroundingRectangle(rule, stroke_color=ACCENT, buff=0.2)
        self.play(FadeIn(rule), ShowCreation(box))
        self.say(f"exponent 0 → 0.xxx × 2{toy.sup(EMIN)}, no hidden 1", ACCENT)
        self.beat()
        self.clear_all()

    def convert(self):
        before = M(S.lead + "." + S.frac + r"_2 \times 2^{" + str(S.e) + "}", 64, INK).move_to([-3.4, 2.5, 0])
        after = M("0." + S.shifted + r"_2 \times 2^{" + str(EMIN) + "}", 64, INK,
                  t2c={S.shifted: MANT_C, str(EMIN): EXP_C}).move_to([3.4, 2.5, 0])
        go = arrow(before.get_right() + 0.2 * RIGHT, after.get_left() + 0.2 * LEFT, SOFT)
        why = num(f"point moves {S.shift} left · {mn(S.e)} + {S.shift} = {mn(EMIN)}", 34, DIM).next_to(VGroup(before, after), DOWN, buff=0.35)
        self.play(Write(before))
        self.play(GrowArrow(go), TransformFromCopy(before, after), FadeIn(why), run_time=1.2)
        self.say(f"shift until the exponent is {mn(EMIN)}")
        self.beat()

        def src(left, bits, color):
            return VGroup(num(left, 32, SOFT), num("→", 32, DIM), bitrow(bits, 38, color)).arrange(RIGHT, buff=0.2)
        s_sign = src("+", "0", SIGN_C)
        s_exp = src(f"2{toy.sup(EMIN)}, no 1", "00000000", EXP_C)
        s_man = src(f"0.{S.shifted}", S.shifted, MANT_C)
        srcs = VGroup(s_sign, s_exp, s_man).arrange(RIGHT, buff=0.8).move_to([0, 0.8, 0])
        bar = BitBar(y=-0.5)
        slots = [bar.slot(i, b) for i, b in enumerate(S.bits)]
        self.play(FadeIn(srcs), LaggedStartMap(FadeIn, bar.cells, lag_ratio=0.02), FadeIn(bar.labels), run_time=1.0)
        self.play(TransformFromCopy(s_sign[2][0], slots[0]),
                  LaggedStart(*[TransformFromCopy(s_exp[2][i], slots[1 + i]) for i in range(8)], lag_ratio=0.1), run_time=1.2)
        self.say("sign 0 · exponent bits all 0", EXP_C)
        self.beat()

        n = len(S.shifted)
        self.play(LaggedStart(*[TransformFromCopy(s_man[2][i], slots[9 + i]) for i in range(n)], lag_ratio=0.12), run_time=1.0)
        pad = VGroup(*slots[9 + n:]).set_color(DIM)
        self.play(LaggedStartMap(FadeIn, pad, lag_ratio=0.04), run_time=0.8)
        self.say(f"mantissa {S.shifted}, pad {23 - n} zeros", MANT_C)
        self.beat()

        hx = tag(S.hex, GOLD, 56).move_to([-3.0, -2.2, 0])
        back = M("0." + S.shifted + r"_2 \times 2^{" + str(EMIN) + r"} = x\ \checkmark", 48, INK, t2c={S.shifted: MANT_C}).move_to([3.0, -2.2, 0])
        self.play(FadeOut(bar.labels), Write(hx))
        self.play(Write(back))
        self.say(f"{S.hex} · reads back to x", ACCENT)
        self.beat()
        self.clear_all()

    def fading(self):
        X_LAB, X_EXP, X_MAN, X_SIG = -5.6, -3.4, 0.9, 5.0
        head = VGroup(
            num("x", 32, DIM).move_to([X_LAB, 2.9, 0]),
            num("exponent", 32, EXP_C).move_to([X_EXP, 2.9, 0]),
            num("hidden · mantissa (23)", 32, MANT_C).move_to([X_MAN, 2.9, 0]),
            num("bits left", 32, DIM).move_to([X_SIG, 2.9, 0]),
        )
        self.play(FadeIn(head))
        caps = {"normal": "normal: hidden 1, 24 bits", "zero": "halfway to 0 → rounds to 0"}
        for i, r in enumerate(toy.FADING):
            y = 2.1 - i * 0.85
            exp, man = r.bits[1:9], r.bits[9:]
            hidden = "1." if r.kind == "normal" else "0."
            lab = num(r.label, 46, INK).move_to([X_LAB, y, 0])
            e_row = bitrow(exp, 33, EXP_C, buff=0.02).move_to([X_EXP, y, 0])
            m_row = VGroup(num(hidden, 36, INK if r.kind == "normal" else BAD), bitrow(man, 33, DIM, buff=0.02)).arrange(RIGHT, buff=0.08)
            m_row.move_to([X_MAN, y, 0])
            for g, b in zip(m_row[1], man):
                if b == "1":
                    g.set_color(MANT_C)
            sig = num(str(r.sig), 46, ACCENT if r.sig > 1 else BAD).move_to([X_SIG, y, 0])
            note = num(r.kind, 28, DIM if r.kind != "zero" else BAD).next_to(sig, RIGHT, buff=0.35)
            self.play(FadeIn(lab), FadeIn(e_row), FadeIn(m_row, shift=0.1 * RIGHT), FadeIn(sig), FadeIn(note), run_time=0.8)
            self.say(caps.get(r.kind, f"{r.label}: {r.sig} bit{'s' if r.sig != 1 else ''} left"),
                     BAD if r.kind == "zero" else SOFT)
            self.beat()
        self.say("smaller → fewer bits → then 0", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L6_Subnormal, globals())
