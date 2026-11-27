"""Lesson 3 · 0.1 never reaches 0: the rows repeat, so we stop when 23 mantissa bits are full."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
from l2_fraction import mul_rows
import toy

P = toy.POINT_ONE


class L3_Repeat(Lesson):
    NUM, TITLE, SUB = 3, "When it never stops", "0.1 repeats → fill 23 bits, round"
    SECTIONS = ["title", "repeat", "runout"]
    TOPICS = {"repeat": "0.1: the rows come back", "runout": "out of mantissa bits → round"}

    def repeat(self):
        f = num("0.1", 72).move_to([-4.9, 2.4, 0])
        self.play(Write(f))
        self.say("same method, a harder number")
        self.beat()

        tab = MulTable(P.rows, x0=-2.6, top=2.7, row_h=0.78)
        mul_rows(self, tab, P.rows, f)

        again = tab.next_input(P.rows[-1].rest, PINK)
        self.play(TransformFromCopy(tab.mobs[-1][2], again))
        b1 = SurroundingRectangle(tab.input(P.first), stroke_color=PINK, buff=0.1)
        b2 = SurroundingRectangle(again, stroke_color=PINK, buff=0.1)
        link = dashed_arrow(b2.get_left() + 0.1 * LEFT, b1.get_left() + 0.1 * LEFT, PINK)
        self.play(ShowCreation(b1), ShowCreation(b2), FadeIn(link))
        self.say(f"{toy.dec(P.rows[-1].rest)} again → loops forever", PINK)
        self.beat()

        n = len(P.rows)
        cyc = P.bits[P.first:n]
        shown = P.bits[:n] + cyc + cyc
        bits = VGroup(num("0.1 =", 48), bitrow("0." + shown, 48, MANT_C), num("…", 48, PINK)).arrange(RIGHT, buff=0.2)
        bits.move_to([0, -2.45, 0])
        self.play(FadeIn(bits))
        self.say(f"bits repeat {cyc} forever", PINK)
        self.beat()
        self.clear_all()

    def runout(self):
        e = P.e
        top = VGroup(num("0.1 =", 44), bitrow("0." + P.bits[:-e + 12], 44, MANT_C), num("…", 44, PINK)).arrange(RIGHT, buff=0.2)
        top.move_to([0, 2.7, 0])
        norm = M(r"= 1.1001100\ldots_2 \times 2^{" + str(e) + "}", 52).next_to(top, DOWN, buff=0.45)
        self.play(FadeIn(top))
        self.play(Write(norm))
        self.say(f"normalise: point moves {-e} right → × 2⁻{-e}")
        self.beat()

        kept = bitrow(P.kept, 36, MANT_C, buff=0.03)
        cut = bitrow(P.cut, 36, BAD, buff=0.03)
        lead = num("1.", 40, DIM)
        row = VGroup(lead, kept, cut, num("…", 36, BAD)).arrange(RIGHT, buff=0.12).move_to([0, 0.2, 0])
        bk = Brace(kept, DOWN, buff=0.12).set_color(MANT_C)
        bc = Brace(cut, DOWN, buff=0.12).set_color(BAD)
        lk = num(f"{len(P.kept)} mantissa bits", 28, MANT_C).next_to(bk, DOWN, buff=0.1)
        lc = num("don't fit", 28, BAD).next_to(bc, DOWN, buff=0.1)
        self.play(FadeIn(row[:2]), GrowFromCenter(bk), FadeIn(lk))
        self.play(FadeIn(row[2:]), GrowFromCenter(bc), FadeIn(lc))
        self.say("23 bits full → stop", BAD)
        self.beat()

        tail_k, tail_r = P.kept[-4:], P.rounded[-4:]
        rnd = VGroup(
            num(f"cut starts {P.cut[:2]}… > half → round up", 32, SOFT),
            VGroup(bitrow("…" + tail_k, 40, MANT_C), num("+ 1 →", 36, DIM), bitrow("…" + tail_r, 40, ACCENT)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, buff=0.25).move_to([0, -1.7, 0])
        self.play(FadeIn(rnd[0]))
        self.play(FadeIn(rnd[1]))
        self.say(f"…{tail_k} rounds up to …{tail_r}")
        self.beat()

        self.play(FadeOut(rnd), FadeOut(VGroup(bk, bc, lk, lc)))
        stored = VGroup(num("stored:", 36, DIM), num(P.stored, 40, ACCENT)).arrange(RIGHT, buff=0.3).move_to([0, -1.2, 0])
        ne = VGroup(num("≠ 0.1", 40, BAD), tag(P.hex, GOLD, 34)).arrange(RIGHT, buff=0.8).next_to(stored, DOWN, buff=0.35)
        self.play(FadeIn(stored, shift=0.1 * UP))
        self.play(FadeIn(ne))
        self.say("0.1 can't be stored exactly", BAD)
        self.beat()
        self.clear_all()


split_sections(L3_Repeat, globals())
