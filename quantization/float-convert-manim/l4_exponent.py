"""Lesson 4 · normalise 10001.001 → 1.0001001 × 2⁴, bias 4 + 127 = 131, 131 → 10000011."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
from l1_integer import ladder_rows
import toy


class L4_Exponent(Lesson):
    NUM, TITLE, SUB = 4, "Normalise + exponent", "1.xxx × 2ᵉ, then add the bias"
    SECTIONS = ["title", "normalise", "bias"]
    TOPICS = {"normalise": "move the point: 1.xxx × 2⁴", "bias": f"{toy.E} + {toy.BIAS} = {toy.BIASED} → binary"}

    def normalise(self):
        digits = toy.FIXED.replace(".", "")
        gap = 0.62
        glyphs = VGroup(*[Text(d, font="Menlo", font_size=80).set_color(MANT_C) for d in digits])
        for i, g in enumerate(glyphs):
            g.move_to([(i - (len(digits) - 1) / 2) * gap - 1.0, 1.4, 0])
        point = len(toy.INT_BITS)
        dot_at = lambda k: (glyphs[k - 1].get_center() + glyphs[k].get_center()) / 2 + 0.33 * DOWN  # noqa: E731
        dot = Dot(dot_at(point), radius=0.08, fill_color=GOLD)
        lab = num("fixed point", 30, DIM).next_to(glyphs, UP, buff=0.5)
        self.play(FadeIn(glyphs), FadeIn(dot), FadeIn(lab))
        self.say(f"{toy.FIXED}: one 1 must lead")
        self.beat()

        counts = VGroup()
        for k in range(point - 1, 0, -1):
            c = num(str(point - k), 30, GOLD).next_to(glyphs[k], UP, buff=0.15)
            counts.add(c)
            self.play(dot.animate.move_to(dot_at(k)), FadeIn(c), run_time=0.45)
        pw = M(r"\times 2^{" + str(toy.E) + "}", 80, INK, t2c={str(toy.E): EXP_C}).next_to(glyphs, RIGHT, buff=0.4)
        self.play(Write(pw))
        self.say(f"point moves {toy.E} left → × 2⁴", ACCENT)
        self.beat()

        self.play(glyphs[0].animate.set_color(DIM), FadeOut(counts), FadeOut(lab))
        under = Line(glyphs[1].get_corner(DL) + 0.12 * DOWN, glyphs[-1].get_corner(DR) + 0.12 * DOWN, stroke_color=MANT_C, stroke_width=4)
        hid = num("always 1 → not stored", 32, DIM).next_to(glyphs[0], UP, buff=0.45).shift(0.6 * RIGHT)
        man = num(f"mantissa = {toy.MANT}", 38, MANT_C).next_to(under, DOWN, buff=0.4)
        self.play(ShowCreation(under), FadeIn(hid), FadeIn(man))
        self.say(f"mantissa = {toy.MANT}", MANT_C)
        self.beat()

        ex = num(f"exponent = {toy.E}", 40, EXP_C).move_to([0, -1.9, 0])
        self.play(TransformFromCopy(pw, ex))
        self.say(f"exponent = {toy.E}", EXP_C)
        self.beat()
        self.clear_all()

    def bias(self):
        b = M(f"{toy.E} + {toy.BIAS} = {toy.BIASED}", 64, INK, t2c={str(toy.BIASED): EXP_C}).move_to([-4.6, 2.5, 0])
        self.play(Write(b))
        why = VGroup(num("stores −126 … 127", 32, DIM), num("as 1 … 254", 32, DIM)).arrange(DOWN, buff=0.12).next_to(b, DOWN, buff=0.4)
        self.play(FadeIn(why))
        self.say(f"add the bias {toy.BIAS}", EXP_C)
        self.beat()

        src = b[-len(str(toy.BIASED)):]
        L = Ladder(toy.EXP_ROWS, x0=-1.3, top=2.95, row_h=0.67, size=42)
        work = ladder_rows(self, L, toy.EXP_ROWS, src, lambda r: f"{r.n} ÷ 2 = {r.q}, remainder {r.r}")
        self.play(FadeIn(L.zero, shift=0.1 * DOWN), work.animate.set_opacity(0.35))
        self.say("quotient is 0 → stop")
        self.beat()

        rems = [L.rem(i) for i in range(len(toy.EXP_ROWS))]
        bits = bitrow(toy.EXP_BITS, 60).move_to([4.7, -0.9, 0])
        label = num(f"{toy.BIASED} =", 46).next_to(bits, UP, buff=0.3)
        self.play(GrowArrow(read_arrow(rems, up=True)), FadeIn(label))
        fly_bits(self, rems[::-1], bits, EXP_C)
        self.say(f"read bottom-up: {toy.EXP_BITS}", EXP_C)
        self.beat()
        self.clear_all()


split_sections(L4_Exponent, globals())
