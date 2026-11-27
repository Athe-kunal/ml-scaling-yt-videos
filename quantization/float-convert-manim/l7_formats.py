"""Lesson 7 · the same 17.125 in fp32, fp16 and bf16 — computed side by side, differences highlighted."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from widgets import *  # noqa: F401,F403
import toy

ROW_Y = [1.7, 0.1, -1.5]
X_NAME, X_LEFT = -6.25, -5.3
DIFF = PINK          # anything that differs between the formats


def sci(v):
    m, e = f"{float(v):.1e}".split("e")
    return f"{m} × 10{toy.sup(int(e))}"


class L7_Formats(Lesson):
    NUM, TITLE, SUB = 7, "fp32 vs fp16 vs bf16", f"the same {toy.dec(toy.X)}, three layouts"
    SECTIONS = ["title", "build", "precision"]
    TOPICS = {"build": f"{toy.dec(toy.X)} in fp32 · fp16 · bf16", "precision": "0.1: where they differ"}

    def build(self):
        top = M(toy.dec(toy.X) + r" = 1." + toy.MANT + r"_2 \times 2^{" + str(toy.E) + "}", 56, INK,
                t2c={toy.MANT: MANT_C, "2^{" + str(toy.E) + "}": EXP_C}).move_to([0, 3.15, 0])
        self.play(Write(top))
        self.say("same binary for all three")
        self.beat()

        bars, names = [], VGroup()
        for r, y in zip(toy.CMP, ROW_Y):
            b = FmtBar(r.ebits, r.mbits, X_LEFT, y)
            bars.append(b)
            names.add(num(r.name, 36, INK).move_to([X_NAME, y, 0]))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(n), FadeIn(b.cells), FadeIn(b.counts)) for n, b in zip(names, bars)],
                              lag_ratio=0.3), run_time=1.5)
        self.say("1 + 8 + 23 · 1 + 5 + 10 · 1 + 8 + 7")
        self.beat()

        hl = VGroup(SurroundingRectangle(bars[1].exp, stroke_color=DIFF, buff=0.06),
                    SurroundingRectangle(bars[1].mant, stroke_color=DIFF, buff=0.06),
                    SurroundingRectangle(bars[2].mant, stroke_color=DIFF, buff=0.06))
        self.play(ShowCreation(hl), bars[1].counts[1].animate.set_color(DIFF), bars[1].counts[2].animate.set_color(DIFF),
                  bars[2].counts[2].animate.set_color(DIFF))
        self.say("fp16: short exponent · bf16: short mantissa", DIFF)
        self.beat()

        # exponent: same E, different bias
        exp_txt = []
        for r, b, y in zip(toy.CMP, bars, ROW_Y):
            col = DIFF if r.bias != toy.BIAS else INK
            t = VGroup(num(f"{toy.E} +", 28, SOFT), num(str(r.bias), 28, col), num(f"= {r.biased} →", 28, SOFT),
                       bitrow(r.exp_bits, 28, EXP_C, buff=0.02)).arrange(RIGHT, buff=0.12)
            t.move_to([0, y - 0.55, 0]).align_to(b.cells, LEFT)
            exp_txt.append(t)
        self.play(FadeOut(hl), *[FadeIn(t) for t in exp_txt])
        self.play(*[LaggedStart(*[TransformFromCopy(t[3][i], b.slot(1 + i, bit)) for i, bit in enumerate(r.exp_bits)], lag_ratio=0.08)
                    for r, b, t in zip(toy.CMP, bars, exp_txt)],
                  *[TransformFromCopy(Text("0").move_to(n), b.slot(0, "0")) for n, b in zip(names, bars)], run_time=1.4)
        self.say(f"exponent: bias 127 · 15 · 127", EXP_C)
        self.beat()

        # mantissa: same 7 bits, different padding
        man_txt = []
        anims = []
        for r, b, t in zip(toy.CMP, bars, exp_txt):
            pad = r.mbits - len(toy.MANT)
            note = f"+ {pad} zeros" if pad else "exact fit"
            m = VGroup(num("mantissa", 28, SOFT), bitrow(toy.MANT, 28, MANT_C, buff=0.02),
                       num(note, 28, DIFF if pad == 0 else DIM)).arrange(RIGHT, buff=0.15)
            m.next_to(t, RIGHT, buff=0.6)
            man_txt.append(m)
            base = 1 + r.ebits
            anims.append(LaggedStart(*[FadeIn(b.slot(base + i, bit, MANT_C if i < len(toy.MANT) else DIM), shift=0.15 * DOWN)
                                       for i, bit in enumerate(r.mant)], lag_ratio=0.03))
        self.play(*[FadeIn(m) for m in man_txt])
        self.play(*anims, run_time=1.4)
        self.say(f"mantissa: same {toy.MANT}, padded to 23 · 10 · 7", MANT_C)
        self.beat()

        # hex: bf16 is the top half of fp32
        hexes = VGroup(*[tag(r.hex, GOLD, 34).move_to([0, y - 0.55, 0]).to_edge(RIGHT, buff=0.35) for r, y in zip(toy.CMP, ROW_Y)])
        self.play(LaggedStartMap(FadeIn, hexes, lag_ratio=0.2))
        top16 = VGroup(*bars[0].cells[:16])
        br = Brace(top16, UP, buff=0.42).set_color(DIFF)
        brl = num("first 16 bits = bf16", 26, DIFF).next_to(br, UP, buff=0.05)
        n = len(toy.CMP[2].hex)
        boxes = VGroup(SurroundingRectangle(hexes[0][:n], stroke_color=DIFF, buff=0.05),
                       SurroundingRectangle(hexes[2], stroke_color=DIFF, buff=0.05))
        self.play(FadeOut(bars[0].counts), FadeOut(top), GrowFromCenter(br), FadeIn(brl), ShowCreation(boxes))
        self.say(f"bf16 = top half of fp32: {toy.CMP[2].hex}", DIFF)
        self.beat()

        ok = VGroup(*[num("✓", 40, ACCENT).next_to(nm, LEFT, buff=0.1) for nm in names])
        self.play(LaggedStartMap(FadeIn, ok, lag_ratio=0.2))
        self.say(f"all three store {toy.dec(toy.X)} exactly", ACCENT)
        self.beat()
        self.clear_all()

    def precision(self):
        x = toy.F(1, 10)
        title = num("0.1 doesn't fit any of them", 44).move_to([0, 3.0, 0])
        cols = [-5.7, -4.0, -2.1, 1.4, 5.0]
        head = VGroup(*[num(h, 28, DIM).move_to([cx, 2.2, 0]) for h, cx in
                        zip(["format", "mantissa", "bits", "stored", "error"], cols)])
        self.play(FadeIn(title), FadeIn(head))
        self.say("now a number that doesn't fit")
        self.beat()

        errs = [abs(r.value - x) for r in toy.CMP_01]
        worst = max(errs)
        for i, (r, err) in enumerate(zip(toy.CMP_01, errs)):
            y = 1.3 - i * 0.9
            s = toy.dec(r.value)
            s = s if len(s) <= 15 else s[:14] + "…"
            col = DIFF if err == worst else (ACCENT if err == min(errs) else INK)
            row = VGroup(num(r.name, 38).move_to([cols[0], y, 0]),
                         num(f"{r.mbits} bits", 32, MANT_C).move_to([cols[1], y, 0]),
                         tag(r.hex, GOLD, 30).move_to([cols[2], y, 0]),
                         num(s, 36).move_to([cols[3], y, 0]),
                         num(sci(err), 34, col).move_to([cols[4], y, 0]))
            self.play(LaggedStartMap(FadeIn, row, lag_ratio=0.12), run_time=0.9)
            self.say(f"{r.name}: {r.mbits} mantissa bits → off by {sci(err)}", col if col != INK else SOFT)
            self.beat()

        def fmax(f):
            return (2 - toy.F(1, 2 ** f.mbits)) * toy.F(2) ** f.bias
        rng = VGroup(*[num(f"{f.name} max " + (f"≈ {sci(fmax(f))}" if fmax(f) > 10 ** 6 else f"= {int(fmax(f))}"), 32,
                           DIFF if f.name == "fp16" else SOFT) for f in toy.FMTS]).arrange(RIGHT, buff=0.8).move_to([0, -1.9, 0])
        self.play(LaggedStartMap(FadeIn, rng, lag_ratio=0.2))
        self.say("fp16: more precise, tiny range · bf16: fp32's range", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L7_Formats, globals())
