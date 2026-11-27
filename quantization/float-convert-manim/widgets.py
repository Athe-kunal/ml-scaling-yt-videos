"""Drawing pieces for the hand conversion: the ÷2 ladder, the ×2 table, bit rows, the 32-bit bar."""
from style import *  # noqa: F401,F403
import toy

MONO = "Menlo"


def num(s, size=40, color=INK):
    return Text(str(s), font=FONT, font_size=size).set_color(color)


def bitrow(bits, size=44, color=INK, buff=0.04):
    """A string of 0/1 as separate glyphs so each can fly on its own."""
    # Menlo draws "." as a centred dot — use the text font for the point
    return VGroup(*[Text(b, font=FONT if b == "." else MONO, font_size=size).set_color(color) for b in bits]).arrange(
        RIGHT, buff=buff, aligned_edge=DOWN)


def working(text, color=INK, size=42):
    """The current calculation, in a box ("17 ÷ 2 = 8  r 1")."""
    t = num(text, size, color)
    box = SurroundingRectangle(t, buff=0.22, stroke_color=RULE, stroke_width=2, fill_color=PANEL, fill_opacity=1)
    return VGroup(box, t)


class Ladder:
    """Repeated ÷ 2 drawn like the slide:   2 | 17   1
                                                 ‾‾‾‾
    Row i holds n_i; its remainder sits right of it; the quotient becomes the next row."""

    def __init__(self, rows, x0, top, row_h=0.82, size=50, width=1.8):
        self.rows, self.x0, self.top, self.h, self.size, self.w = rows, x0, top, row_h, size, width
        self.mobs = [self._row(i) for i in range(len(rows))]
        self.zero = num("0", size, DIM).move_to([x0 + 0.62 + 0.2, self.y(len(rows)) , 0])

    def y(self, i):
        return self.top - i * self.h

    def _row(self, i):
        r, y, x0, h = self.rows[i], self.y(i), self.x0, self.h
        two = num("2", self.size * 0.85, DIM).move_to([x0, y, 0])
        bar = Line([x0 + 0.32, y + h * 0.42, 0], [x0 + 0.32, y - h * 0.45, 0], stroke_color=SOFT, stroke_width=3)
        n = num(r.n, self.size).next_to([x0 + 0.5, y, 0], RIGHT, buff=0)
        line = Line([x0 + 0.32, y - h * 0.45, 0], [x0 + 0.32 + self.w, y - h * 0.45, 0], stroke_color=SOFT, stroke_width=3)
        rem = num(r.r, self.size, BIT_C).move_to([x0 + self.w + 0.75, y, 0])
        return VGroup(two, bar, n, line, rem)

    def rem(self, i):
        return self.mobs[i][4]

    def frame(self, i):
        """Everything of row i except its remainder."""
        return VGroup(*self.mobs[i][:4])


class MulTable:
    """Repeated × 2:   0.125   × 2   = 0.25   |  0"""

    COLS = (0.0, 2.05, 3.3, 5.3, 5.8)

    def __init__(self, rows, x0, top, row_h=0.85, size=48):
        self.rows, self.x0, self.top, self.h, self.size = rows, x0, top, row_h, size
        self.mobs = [self._row(i) for i in range(len(rows))]

    def y(self, i):
        return self.top - i * self.h

    def _at(self, mob, col, y):
        return mob.next_to([self.x0 + self.COLS[col], y, 0], RIGHT, buff=0)

    def _row(self, i):
        r, y = self.rows[i], self.y(i)
        x = self._at(num(toy.dec(r.x), self.size), 0, y)
        times = self._at(num("× 2", self.size * 0.85, DIM), 1, y)
        eq = self._at(num(f"= {toy.dec(r.y)}", self.size), 2, y)
        bar = self._at(num("|", self.size, DIM), 3, y)
        bit = self._at(num(r.bit, self.size, BIT_C), 4, y)
        return VGroup(x, times, eq, bar, bit)

    def bit(self, i):
        return self.mobs[i][4]

    def input(self, i):
        return self.mobs[i][0]

    def next_input(self, value, color=DIM):
        """The fraction carried to the row below the last one."""
        return self._at(num(toy.dec(value), self.size, color), 0, self.y(len(self.rows)))


def read_arrow(mobs, up=True, color=ACCENT, side=0.55):
    """Arrow along a column of bits showing the reading order."""
    xs = max(m.get_right()[0] for m in mobs) + side
    lo, hi = min(m.get_bottom()[1] for m in mobs), max(m.get_top()[1] for m in mobs)
    a, b = ([xs, lo, 0], [xs, hi, 0]) if up else ([xs, hi, 0], [xs, lo, 0])
    return arrow(a, b, color, thick=5, tip=0.12)


def fly_bits(scene, sources, target, color, run_time=1.4):
    """Copies of `sources` (in reading order) fly into the glyphs of `target`."""
    scene.play(LaggedStart(*[TransformFromCopy(s, t) for s, t in zip(sources, target)], lag_ratio=0.18),
               run_time=run_time)
    scene.play(target.animate.set_color(color), run_time=0.3)


class BitBar:
    """The final 32 bits: 1 sign | 8 exponent | 23 mantissa cells, slide colours."""

    def __init__(self, y=0.0, cell=0.36, gap=0.035, field_gap=0.16, size=26):
        cells, digits = VGroup(), VGroup()
        x = -(32 * cell + 31 * gap + 2 * field_gap) / 2
        for i in range(32):
            col = SIGN_C if i == 0 else EXP_C if i <= 8 else MANT_C
            if i in (1, 9):
                x += field_gap
            sq = Square(cell, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.12)
            sq.move_to([x + cell / 2, y, 0])
            cells.add(sq)
            digits.add(Text("0", font=MONO, font_size=size).set_color(INK).move_to(sq).set_opacity(0))
            x += cell + gap
        self.cells, self.digits = cells, digits
        self.labels = VGroup(
            num("sign", 24, SIGN_C).next_to(cells[0], DOWN, buff=0.2),
            num("exponent", 24, EXP_C).next_to(cells[1:9], DOWN, buff=0.2),
            num("mantissa", 24, MANT_C).next_to(cells[9:], DOWN, buff=0.2),
        )

    def slot(self, i, bit, size=26):
        return Text(bit, font=MONO, font_size=size).set_color(INK).move_to(self.cells[i])


class FmtBar:
    """Bit bar for any format, left-aligned so lengths compare: 1 sign | e exponent | m mantissa."""

    def __init__(self, ebits, mbits, x_left=-5.3, y=0.0, cell=0.31, gap=0.03, field_gap=0.14, size=20):
        n = 1 + ebits + mbits
        self.ebits, self.mbits, self.size = ebits, mbits, size
        cells = VGroup()
        x = x_left
        for i in range(n):
            col = SIGN_C if i == 0 else EXP_C if i <= ebits else MANT_C
            if i in (1, 1 + ebits):
                x += field_gap
            cells.add(Square(cell, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.12).move_to([x + cell / 2, y, 0]))
            x += cell + gap
        self.cells = cells
        self.sign, self.exp, self.mant = cells[:1], cells[1:1 + ebits], cells[1 + ebits:]
        self.counts = VGroup(*[num(str(len(g)), 24, c).next_to(g, UP, buff=0.1)
                               for g, c in ((self.sign, SIGN_C), (self.exp, EXP_C), (self.mant, MANT_C))])

    def slot(self, i, bit, color=INK):
        return Text(bit, font=MONO, font_size=self.size).set_color(color).move_to(self.cells[i])
