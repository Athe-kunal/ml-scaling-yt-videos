"""Lesson 5 · sign | exponent | mantissa → 32 bits → 0x41890000, then read it back."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
import toy


class L5_Assemble(Lesson):
    NUM, TITLE, SUB = 5, "Assemble + verify", "sign | exponent | mantissa"
    SECTIONS = ["title", "assemble", "verify"]
    TOPICS = {"assemble": "32 bits: 1 + 8 + 23", "verify": "read it back: 17.125?"}

    def assemble(self):
        def src(left, bits, color):
            r = bitrow(bits, 40, color)
            return VGroup(num(left, 34, SOFT), num("→", 34, DIM), r).arrange(RIGHT, buff=0.22)
        s_sign = src("+", toy.SIGN_BIT, SIGN_C)
        s_exp = src(str(toy.BIASED), toy.EXP_BITS, EXP_C)
        s_man = src(f"1.{toy.MANT}", toy.MANT, MANT_C)
        srcs = VGroup(s_sign, s_exp, s_man).arrange(RIGHT, buff=0.9).move_to([0, 2.5, 0])
        heads = VGroup(*[num(t, 26, c).next_to(s, UP, buff=0.2) for t, c, s in
                         (("sign", SIGN_C, s_sign), ("exponent", EXP_C, s_exp), ("mantissa", MANT_C, s_man))])
        bar = BitBar(y=0.3)
        self.play(FadeIn(srcs), FadeIn(heads))
        self.play(LaggedStartMap(FadeIn, bar.cells, lag_ratio=0.02), FadeIn(bar.labels), run_time=1.0)
        self.say("32 cells: 1 + 8 + 23")
        self.beat()

        slots = [bar.slot(i, b) for i, b in enumerate(toy.BITS)]
        self.play(TransformFromCopy(s_sign[2][0], slots[0]))
        self.say(f"positive → sign {toy.SIGN_BIT}", SIGN_C)
        self.beat()

        self.play(LaggedStart(*[TransformFromCopy(s_exp[2][i], slots[1 + i]) for i in range(8)], lag_ratio=0.12), run_time=1.3)
        self.say(f"exponent {toy.EXP_BITS}", EXP_C)
        self.beat()

        n = len(toy.MANT)
        self.play(LaggedStart(*[TransformFromCopy(s_man[2][i], slots[9 + i]) for i in range(n)], lag_ratio=0.12), run_time=1.2)
        pad = VGroup(*slots[9 + n:]).set_color(DIM)
        self.play(LaggedStartMap(FadeIn, pad, lag_ratio=0.04), run_time=0.9)
        self.say(f"mantissa {toy.MANT}, pad {23 - n} zeros", MANT_C)
        self.beat()

        hexes = VGroup()
        for k, nib in enumerate(toy.NIBBLES):
            cells = bar.cells[4 * k:4 * k + 4]
            br = Brace(VGroup(*cells), DOWN, buff=0.75).set_color(DIM)
            h = tag("%X" % int(nib, 2), GOLD, 34).next_to(br, DOWN, buff=0.1)
            hexes.add(VGroup(br, h))
        self.play(FadeOut(bar.labels), LaggedStartMap(FadeIn, hexes, lag_ratio=0.1), run_time=1.2)
        big = tag(toy.HEX, GOLD, 60).move_to([0, -2.4, 0])
        self.play(Write(big))
        self.say(f"4 bits → 1 hex digit: {toy.HEX}", GOLD)
        self.beat()
        self.clear_all()

    def verify(self):
        e = int(toy.EXP_BITS, 2) - toy.BIAS
        tc = {toy.MANT: MANT_C, str(toy.BIASED): EXP_C, "2^{" + str(e) + "}": EXP_C}
        lines = [
            M(r"(-1)^{" + toy.SIGN_BIT + r"} \times 1." + toy.MANT + r"_2 \times 2^{" + str(toy.BIASED) + "-" + str(toy.BIAS) + "}", 54, INK, t2c=tc),
            M(r"= 1." + toy.MANT + r"_2 \times 2^{" + str(e) + "}", 54, INK, t2c=tc),
            M(r"= " + toy.FIXED + r"_2", 54, INK),
            M(r"= " + " + ".join(toy.dec(p) for p in toy.PARTS) + " = " + toy.dec(toy.decode(toy.SIGN_BIT, toy.EXP_BITS, toy.MANT23)), 54, INK),
        ]
        VGroup(*lines).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, 0.4, 0])
        caps = [("bits → formula", SOFT), (f"{toy.BIASED} − {toy.BIAS} = {e}", EXP_C),
                (f"point moves {e} right", SOFT), (f"{toy.dec(toy.X)} ✓ same number back", ACCENT)]
        for m, (c, col) in zip(lines, caps):
            self.play(Write(m), run_time=1.0)
            self.say(c, col)
            self.beat()
        box = SurroundingRectangle(lines[-1], stroke_color=ACCENT, buff=0.15)
        self.play(ShowCreation(box))
        self.say(f"{toy.dec(toy.X)} = {toy.HEX}", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L5_Assemble, globals())
