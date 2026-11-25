"""Lesson 1 · Why pack: a 2-bit weight in a uint8 wastes 6 bits; four fit in one byte; when it fails."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from bits import Byte, TensorRow, mono
import toy


def one_per_byte(values, top=1.3, gap=1.0):
    """Each weight in its own uint8: only slot 0 is used."""
    rows = VGroup()
    for k, v in enumerate(values):
        rows.add(Byte(v, occupied={0}, label=f"w[{k}]", index=(k == 0)))
    for k, r in enumerate(rows):
        r.move_to([0.6, top - k * gap, 0])
    return rows


class L1_Why(Lesson):
    NUM, TITLE, SUB = 1, "Why pack?", "2-bit weights in 8-bit boxes"
    SECTIONS = ["title", "waste", "four_x", "shape"]
    TOPICS = {"waste": "6 of 8 bits wasted", "four_x": "four weights, one byte", "shape": "when packing fails"}

    # ------------------------------------------------------------ one weight per byte
    def waste(self):
        w = TensorRow(toy.EX1, label="2-bit weights").move_to([0, 2.9, 0])
        self.play(FadeIn(w))
        self.say("values 0…3 need only 2 bits")
        self.beat()

        rows = one_per_byte(toy.EX1)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * DOWN) for r in rows], lag_ratio=0.15), run_time=1.2)
        self.say("PyTorch's smallest int: uint8")
        self.beat()

        marks = VGroup()
        for r in rows:
            cells = VGroup(*r.frame[:6])
            marks.add(Rectangle(width=cells.get_width() + 0.06, height=cells.get_height() + 0.06,
                                stroke_color=BAD, stroke_width=2.5, fill_color=BAD, fill_opacity=0.12).move_to(cells))
        lab = T("wasted", 30, BAD).next_to(marks[0], UP, buff=0.35)
        used = T("used", 30, ACCENT).next_to(rows[0].slot_cells(0), UP, buff=0.35)
        self.play(LaggedStart(*[FadeIn(m) for m in marks], lag_ratio=0.1), FadeIn(lab), FadeIn(used), run_time=1.0)
        self.say("6 of every 8 bits wasted", BAD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ four weights → one byte
    def four_x(self):
        rows = one_per_byte(toy.EX1, top=2.4, gap=0.95)
        self.play(FadeIn(rows))
        self.say("4 weights = 4 bytes")
        self.beat()

        packed = Byte(0, occupied=set(), label="packed").move_to([0.6, -2.0, 0])
        self.play(FadeIn(packed))
        movers = []
        for j, r in enumerate(rows):
            src = VGroup(r.tiles[6], r.tiles[7]).copy()
            dst = VGroup(packed.tiles[6 - 2 * j], packed.tiles[7 - 2 * j])
            movers.append(src)
            self.add(src)
        self.play(LaggedStart(*[m.animate.move_to(VGroup(packed.tiles[6 - 2 * j], packed.tiles[7 - 2 * j]))
                                for j, m in enumerate(movers)], lag_ratio=0.25),
                  *[r.dim_anim() for r in rows], run_time=2.0)
        full = Byte(177, label="packed").move_to(packed)
        self.remove(*movers, packed)
        self.add(full)
        self.say("4 weights = 1 byte", ACCENT)
        self.beat()

        br = VGroup(*[full.slot_brace(j, f"w[{j}]") for j in range(4)])
        self.play(LaggedStart(*[FadeIn(b) for b in br], lag_ratio=0.15))
        self.say("w[0] sits in the lowest bits")
        self.beat()

        self.play(FadeOut(rows), FadeOut(br), full.animate.move_to([0.6, 1.8, 0]))
        a = VGroup(T("7B weights as uint8", 36, SOFT), mono("7.00 GB", 40, BAD)).arrange(RIGHT, buff=0.5)
        b = VGroup(T("7B weights packed", 36, SOFT), mono("1.75 GB", 40, ACCENT)).arrange(RIGHT, buff=0.5)
        g = VGroup(a, b).arrange(DOWN, buff=0.45, aligned_edge=RIGHT).move_to([0, -0.6, 0])
        self.play(FadeIn(a))
        self.play(FadeIn(b))
        self.say("4× less memory", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the shape check
    def shape(self):
        code = mono("if uint8tensor.shape[0] * bits % 8 != 0:", 28, SOFT).move_to([0, 3.0, 0])
        f = M(r"\text{num\_values} = \frac{n \cdot \text{bits}}{8}", 46).move_to([0, 1.9, 0])
        self.play(FadeIn(code), Write(f))
        self.say("total bits must fill whole bytes")
        self.beat()

        cases = [(toy.EX1, "4 · 2 = 8 bits", "1 byte", ACCENT), (toy.EX2, "8 · 2 = 16 bits", "2 bytes", ACCENT),
                 (toy.BAD, "5 · 2 = 10 bits", "1.25 bytes", BAD)]
        rows = VGroup()
        for vals, bits_txt, res, col in cases:
            t = TensorRow(vals, size=26, box=0.5, index=False)
            rows.add(VGroup(t, mono(bits_txt, 28, SOFT), mono("→ " + res, 28, col)))
        for r in rows:
            r[0].move_to(ORIGIN, aligned_edge=LEFT)
        y = [0.7, -0.4, -1.5]
        for r, yy in zip(rows, y):
            r[0].move_to([-5.2, yy, 0], aligned_edge=LEFT)
            r[1].move_to([1.5, yy, 0], aligned_edge=LEFT)
            r[2].move_to([4.7, yy, 0], aligned_edge=LEFT)
        for r, (_, _, _, col) in zip(rows[:2], cases[:2]):
            self.play(FadeIn(r))
            self.beat()
        self.play(FadeIn(rows[2]))
        err = chip("ValueError", BAD, size=26, h=0.6).next_to(rows[2][2], DOWN, buff=0.35)
        self.play(FadeIn(err, scale=0.8))
        self.say("5 values: 2 bits left over", BAD)
        self.beat()
        self.clear_all()


split_sections(L1_Why, globals())
