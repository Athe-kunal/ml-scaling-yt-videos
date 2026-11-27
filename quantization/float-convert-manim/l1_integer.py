"""Lesson 1 · 17 → binary by repeated ÷ 2 (remainders read bottom-up)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
import toy

WORK = np.array([4.55, 2.1, 0])


def q_glyphs(w, r):
    """The quotient's glyphs inside a working box 'n ÷ 2 = q r b' (Text drops spaces)."""
    i = len(str(r.n)) + 3
    return w[1][i:i + len(str(r.q))]


def ladder_rows(scene, L, rows, first_src, caption):
    """One → per row: the number arrives, the division is worked, the remainder drops out."""
    work, prev = None, None
    for i, r in enumerate(rows):
        w = working(f"{r.n} ÷ 2 = {r.q}   r {r.r}").move_to(WORK)
        src = first_src if i == 0 else q_glyphs(prev, rows[i - 1])
        anims = [FadeIn(L.frame(i)[:2]), ShowCreation(L.frame(i)[3]), TransformFromCopy(src, L.frame(i)[2])]
        if work is None:
            work = w
            anims.append(FadeIn(work))
        else:
            anims.append(Transform(work, w))
        scene.play(*anims, run_time=0.9)
        scene.play(TransformFromCopy(w[1][-1], L.rem(i)), run_time=0.6)
        scene.say(caption(r))
        scene.beat()
        prev = w
    return work


class L1_Integer(Lesson):
    NUM, TITLE, SUB = 1, "The integer part", "17 → binary by dividing by 2"
    SECTIONS = ["title", "ladder"]
    TOPICS = {"ladder": "17 → binary: ÷ 2, keep remainders"}

    def ladder(self):
        x = num(toy.dec(toy.X), 72).move_to([-4.6, 2.5, 0])
        self.play(Write(x))
        self.say("one number, two parts")
        self.beat()

        ip, fp = num(toy.INT, 60), num(toy.dec(toy.FRAC), 60, DIM)
        VGroup(ip, fp).arrange(RIGHT, buff=0.9).move_to([-4.6, 1.2, 0])
        self.play(TransformFromCopy(x, ip), TransformFromCopy(x, fp))
        self.say(f"integer {toy.INT} first; {toy.dec(toy.FRAC)} later")
        self.beat()

        L = Ladder(toy.INT_ROWS, x0=-0.8, top=2.5)
        work = ladder_rows(self, L, toy.INT_ROWS, ip, lambda r: f"{r.n} ÷ 2 = {r.q}, remainder {r.r}")

        self.play(FadeIn(L.zero, shift=0.1 * DOWN), work.animate.set_opacity(0.35))
        self.say("quotient is 0 → stop")
        self.beat()

        rems = [L.rem(i) for i in range(len(toy.INT_ROWS))]
        bits = bitrow(toy.INT_BITS, 64).move_to([5.3, -1.0, 0])
        label = num(f"{toy.INT} =", 52).next_to(bits, LEFT, buff=0.3)
        sub = num("2", 30, DIM).next_to(bits, RIGHT, buff=0.08).shift(0.22 * DOWN)
        self.play(GrowArrow(read_arrow(rems, up=True)))
        self.play(FadeIn(label))
        fly_bits(self, rems[::-1], bits, MANT_C)
        self.play(FadeIn(sub))
        self.say("read remainders bottom-up", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L1_Integer, globals())
