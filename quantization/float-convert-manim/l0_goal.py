"""Lesson 0 · the goal: convert 17.125 into a 32-bit IEEE-754 float (shown before any method)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
from l1_integer import L1_Integer
from l2_fraction import L2_Fraction
from l3_repeat import L3_Repeat
from l4_exponent import L4_Exponent
from l5_assemble import L5_Assemble
import toy

STEPS = [L1_Integer, L2_Fraction, L3_Repeat, L4_Exponent, L5_Assemble]


class L0_Goal(Lesson):
    NUM, TITLE, SUB = 0, f"Converting {toy.dec(toy.X)}", "into a 32-bit IEEE-754 float"
    SECTIONS = ["title", "goal"]
    TOPICS: dict = {}

    def title(self):
        n = num(toy.dec(toy.X), 140, ACCENT)
        t = T("→ IEEE-754 float, 32 bits", 56)
        s = T("step by step, by hand", 32, DIM)
        VGroup(n, t, s).arrange(DOWN, buff=0.4)
        self.play(FadeIn(n, scale=0.7), run_time=0.9)
        self.play(Write(t), FadeIn(s, shift=0.2 * UP))
        self.beat()
        self.clear_all()

    def goal(self):
        x = num(toy.dec(toy.X), 96).move_to([0, 2.3, 0])
        bar = BitBar(y=0.0)
        q = VGroup(*[num("?", 22, DIM).move_to(c) for c in bar.cells])
        self.play(FadeIn(x, scale=0.8))
        down = arrow(x.get_bottom() + 0.15 * DOWN, bar.cells.get_top() + 0.25 * UP, SOFT)
        self.play(GrowArrow(down),
                  LaggedStartMap(FadeIn, bar.cells, lag_ratio=0.02), FadeIn(bar.labels), FadeIn(q), run_time=1.2)
        self.say(f"we're converting {toy.dec(toy.X)} into these 32 bits", ACCENT)
        self.beat()

        self.play(FadeOut(q), VGroup(x, down, bar.cells, bar.labels).animate.scale(0.75).shift(1.2 * UP), run_time=0.8)
        rows = VGroup(*[
            VGroup(tag(str(L.NUM), ACCENT, 34), T(L.TITLE, 38), T(L.SUB, 28, DIM)).arrange(RIGHT, buff=0.35)
            for L in STEPS
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -1.3, 0])
        self.play(LaggedStartMap(FadeIn, rows, shift=0.15 * RIGHT, lag_ratio=0.15), run_time=1.4)
        self.say("five steps to fill them")
        self.beat()
        self.clear_all()


split_sections(L0_Goal, globals())
