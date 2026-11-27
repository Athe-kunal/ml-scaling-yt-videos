"""Lesson 2 · 0.125 → binary by repeated × 2 (integer parts read top-down)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
import toy


def mul_rows(scene, tab, rows, first_src):
    """One → per row: the fraction arrives, × 2, its integer part drops out as a bit."""
    for i, r in enumerate(rows):
        src = first_src if i == 0 else tab.mobs[i - 1][2]
        m = tab.mobs[i]
        scene.play(TransformFromCopy(src, m[0]), run_time=0.6)
        scene.play(FadeIn(m[1]), FadeIn(m[2], shift=0.1 * RIGHT), run_time=0.6)
        scene.play(FadeIn(m[3]), TransformFromCopy(m[2], m[4]), run_time=0.6)
        rest = f", keep {toy.dec(r.rest)}" if r.bit and r.rest else ""
        scene.say(f"{toy.dec(r.x)} × 2 = {toy.dec(r.y)} → bit {r.bit}{rest}")
        scene.beat()


class L2_Fraction(Lesson):
    NUM, TITLE, SUB = 2, "The fraction part", "0.125 → binary by multiplying by 2"
    SECTIONS = ["title", "doubling"]
    TOPICS = {"doubling": "0.125 → binary: × 2, keep integer parts"}

    def doubling(self):
        recap = VGroup(num(f"{toy.INT} =", 40, DIM), bitrow(toy.INT_BITS, 40, MANT_C)).arrange(RIGHT, buff=0.25)
        recap.move_to([-4.7, 3.0, 0])
        f = num(toy.dec(toy.FRAC), 72).move_to([-4.7, 1.6, 0])
        self.play(FadeIn(recap), Write(f))
        self.say("now the part after the point")
        self.beat()

        tab = MulTable(toy.FRAC_ROWS, x0=-1.6, top=2.4)
        mul_rows(self, tab, toy.FRAC_ROWS, f)

        zero = tab.next_input(toy.FRAC_ROWS[-1].rest, ACCENT)
        stop = chip("fraction = 0 → stop", ACCENT, 30, h=0.6).next_to(zero, RIGHT, buff=0.5)
        self.play(TransformFromCopy(tab.mobs[-1][2], zero), FadeIn(stop))
        self.say("nothing left after the point → stop")
        self.beat()

        bits_src = [tab.bit(i) for i in range(len(toy.FRAC_ROWS))]
        frac = bitrow("." + toy.FRAC_BITS, 60).move_to([-4.7, -0.3, 0])
        self.play(GrowArrow(read_arrow(bits_src, up=False)))
        self.add(frac[0].set_color(MANT_C))
        fly_bits(self, bits_src, frac[1:], MANT_C)
        self.say("read integer parts top-down", ACCENT)
        self.beat()

        whole = bitrow(toy.FIXED, 72, MANT_C).move_to([0, -2.1, 0])
        eq = num(f"{toy.dec(toy.X)} =", 64).next_to(whole, LEFT, buff=0.35)
        two = num("2", 36, DIM).next_to(whole, RIGHT, buff=0.08).shift(0.28 * DOWN)
        n = len(toy.INT_BITS)
        self.play(FadeIn(eq), TransformFromCopy(recap[1], whole[:n]), TransformFromCopy(frac, whole[n:]), FadeIn(two), run_time=1.2)
        self.say(f"fixed point: {toy.FIXED}", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L2_Fraction, globals())
