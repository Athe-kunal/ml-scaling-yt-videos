"""Lesson 1 · Waveforms and sampling."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

SR = dsp.SAMPLE_RATE


def axes(w=12, h=3.2, x=(0, 1, 0.25), y=(-1, 1, 1)):
    return Axes(x_range=x, y_range=y, width=w, height=h,
                axis_config=dict(stroke_color=RULE, stroke_width=2, include_tip=False, include_ticks=False))


class L1_Waveform(Lesson):
    NUM, TITLE, SUB = 1, "Waveforms & sampling", "from air pressure to numbers"
    SECTIONS = ["title", "sine_knobs", "the_melody", "sampling", "rate_vs_freq", "nyquist",
                "quantize", "mono_resample", "pad_30s"]
    TOPICS = {"sine_knobs": "a sine's three knobs", "the_melody": "our melody", "sampling": "sampling: dots in time",
              "rate_vs_freq": "sample rate ≠ frequency", "nyquist": "Nyquist & aliasing",
              "quantize": "bit depth: quantization", "mono_resample": "mono & resampling", "pad_30s": "pad to 30 s"}

    def prepare(self):
        y = dsp.melody()
        s0, n = int(0.1 * SR), int(0.02 * SR)  # 20 ms of the first note (C4)
        self.zoom_y = y[s0:s0 + n]

    # ------------------------------------------------------------ a sine has 3 knobs
    def sine_knobs(self):
        ax = axes(w=12, h=3.6, x=(0, 1, 0.25), y=(-1.2, 1.2, 1)).shift(0.3 * UP)
        A, f, ph = ValueTracker(0.8), ValueTracker(2.0), ValueTracker(0.0)
        curve = always_redraw(lambda: graph(
            ax, lambda t: A.get_value() * np.sin(TAU * f.get_value() * t + ph.get_value()), (0, 1), ACCENT, 4))
        formula = M(r"x(t) = A \sin(2\pi f t + \varphi)", 52, isolate=["A", "f", r"\varphi"]).to_edge(UP, buff=0.5)
        self.play(ShowCreation(ax), Write(formula))
        self.play(ShowCreation(curve), run_time=1.5)
        self.say("a pure tone")
        self.beat()

        for sym, trk, vals, word, col in [
            ("A", A, [1.15, 0.35, 0.8], "amplitude → loudness", GOLD),
            ("f", f, [6.0, 1.0, 3.0], "frequency → pitch", PINK),
        ]:
            self.play(formula.select_part(sym).animate.set_color(col), run_time=0.4)
            self.say(word, col)
            for v in vals:
                self.play(trk.animate.set_value(v), run_time=1.0)
            self.beat()
            self.play(formula.select_part(sym).animate.set_color(INK), run_time=0.3)

        # phase: keep the φ = 0 curve as "before", shift a copy to φ = π/2 ("after")
        A0, f0 = A.get_value(), f.get_value()  # 0.8, 3 cycles
        before = DashedVMobject(graph(ax, lambda t: A0 * np.sin(TAU * f0 * t), (0, 1), DIM, 3), num_dashes=90)
        lab_b = T("before: φ = 0", 28, DIM)
        lab_a = T("after: φ = π/2", 28, BLUE)
        VGroup(lab_b, lab_a).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ax, UP, buff=0.1).to_edge(RIGHT, buff=0.6)
        self.play(formula.select_part(r"\varphi").animate.set_color(BLUE), run_time=0.4)
        self.add(before)
        self.play(FadeIn(lab_b))
        self.say("phase → where it starts", BLUE)
        self.beat()
        self.play(ph.animate.set_value(PI / 2), FadeIn(lab_a), run_time=1.5)
        # matching peaks: before at t = (1/4 + 1)/f, after a quarter cycle earlier at t = 1/f
        tb, ta, yp = 1.25 / f0, 1.0 / f0, A0 + 0.22
        arrow = Arrow(ax.c2p(tb, yp), ax.c2p(ta, yp), buff=0, stroke_color=BLUE, stroke_width=5)
        pk = VGroup(Dot(ax.c2p(tb, A0), radius=0.08, fill_color=DIM), Dot(ax.c2p(ta, A0), radius=0.08, fill_color=BLUE))
        dt = M(r"\Delta t = \frac{\varphi}{2\pi f} = \tfrac{1}{4}\ \text{cycle}", 38, BLUE).next_to(arrow, UP, buff=0.12)
        self.play(FadeIn(pk), GrowArrow(arrow), Write(dt))
        self.say("same wave, shifted in time", BLUE)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the melody we use everywhere
    def the_melody(self):
        y = dsp.melody()
        t = np.arange(len(y)) / SR
        ax = axes(w=12.5, h=3.4, x=(0, 4, 1), y=(-0.6, 0.6, 0.3)).shift(0.4 * UP)
        env = envelope(ax, t, y)
        ticks = VGroup(*[T(f"{s} s", 24, DIM).next_to(ax.c2p(s, -0.6), DOWN, buff=0.15) for s in range(5)])
        self.play(ShowCreation(ax), FadeIn(ticks))
        self.play(ShowCreation(env), run_time=2.5, rate_func=linear)
        lbl = T("our melody · 8 notes · 4 s", 32, INK).next_to(ax, UP, buff=0.3)
        self.play(FadeIn(lbl))
        self.say("audio clip")
        self.beat()

        # zoom into 20 ms of the first note: harmonics make the shape
        s0, n = int(0.1 * SR), int(0.02 * SR)
        box = Rectangle(width=ax.c2p(0.12, 0)[0] - ax.c2p(0.1, 0)[0] + 0.06, height=3.4,
                        stroke_color=GOLD).move_to(ax.c2p(0.11, 0))
        self.play(ShowCreation(box))
        ax2 = axes(w=12.5, h=3.4, x=(0, 20, 5), y=(-0.6, 0.6, 0.3)).shift(0.4 * UP)
        zt = np.arange(n) / SR * 1000
        zoom = wave(ax2, zt, y[s0:s0 + n], ACCENT, 3.5)
        ms = VGroup(*[T(f"{v} ms", 24, DIM).next_to(ax2.c2p(v, -0.6), DOWN, buff=0.15) for v in (0, 10, 20)])
        self.play(FadeOut(env), FadeOut(ticks), FadeOut(box), ReplacementTransform(ax, ax2),
                  FadeTransform(lbl, T("20 ms of C4 · 262 Hz", 32).next_to(ax2, UP, buff=0.3)))
        self.play(ShowCreation(zoom), FadeIn(ms), run_time=1.5)
        self.say("one note = a sum of harmonics")
        self.beat()

    # ------------------------------------------------------------ sampling
    def sampling(self):
        y = self.zoom_y
        # show the first 4 ms (64 samples) so individual samples are visible
        n = 64  # 4 ms ≈ one period of C4
        raw = y
        y = y / np.abs(y[:n]).max() * 0.9  # display scale only; the array below shows the real values
        ax3 = axes(w=12.5, h=3.4, x=(0, n, 16), y=(-1, 1, 0.5)).shift(0.4 * UP)
        fine = np.linspace(0, n - 1, 800)
        cont = wave(ax3, fine, np.interp(fine, np.arange(len(y)), y), ACCENT, 3)
        self.fade_all(keep=(self.caption,))
        self.play(ShowCreation(ax3), ShowCreation(cont))
        self.say("measure every T seconds")
        st = stems(ax3, np.arange(n), y[:n])
        self.play(LaggedStartMap(FadeIn, st, lag_ratio=0.02), run_time=1.5)
        f = VGroup(M(r"t_n = n\,T", 48), M(r"s_r = \tfrac{1}{T} = 16\,000\ \text{Hz}", 48, GOLD)).arrange(RIGHT, buff=1.0)
        f.to_edge(UP, buff=0.5)
        self.play(Write(f))
        brace = Brace(Line(ax3.c2p(10, 0), ax3.c2p(11, 0)), DOWN).shift(1.9 * DOWN)
        bt = M(r"T = 62.5\,\mu s", 34, GOLD).next_to(brace, DOWN, buff=0.1)
        self.play(GrowFromCenter(brace), FadeIn(bt))
        self.beat()
        self.play(FadeOut(cont), FadeOut(brace), FadeOut(bt))
        self.say("the computer keeps only the dots")
        arr = tag(f"x = [{', '.join(f'{v:.3f}' for v in raw[:6])}, …]", ACCENT, 28).next_to(ax3, DOWN, buff=0.5)
        self.play(TransformFromCopy(st[:12], arr))  # values are rescaled for display
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ samples/s vs frequencies in the sound
    def rate_vs_freq(self):
        rows = VGroup()
        n = 64  # 4 ms at 16 kHz
        for f, col in [(262, ACCENT), (1047, PINK)]:
            ax = axes(w=11, h=1.8, x=(0, n, 16), y=(-1.1, 1.1, 1))
            t = np.linspace(0, n, 600)
            c = wave(ax, t, np.sin(TAU * f * t / SR), col, 3)
            d = VGroup(*[Dot(ax.c2p(k, np.sin(TAU * f * k / SR)), radius=0.05, fill_color=GOLD) for k in range(n)])
            lab = T(f"{f} Hz", 30, col).next_to(ax, LEFT, buff=0.3)
            rows.add(VGroup(ax, c, d, lab))
        rows.arrange(DOWN, buff=0.6).shift(0.3 * UP + 0.5 * RIGHT)
        for r in rows:
            self.play(ShowCreation(r[0]), ShowCreation(r[1]), FadeIn(r[3]), run_time=0.8)
            self.play(LaggedStartMap(FadeIn, r[2], lag_ratio=0.01), run_time=0.8)
        hd = VGroup(T("same 16 000 dots / s", 34, GOLD), T("different Hz inside", 34, PINK)).arrange(RIGHT, buff=1.2)
        hd.to_edge(UP, buff=0.4)
        self.play(FadeIn(hd))
        self.say("sample rate ≠ frequency of the sound")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ Nyquist / aliasing
    def nyquist(self):
        n = 16  # 1 ms of samples: few enough to see every dot
        ax = axes(w=12, h=2.5, x=(0, n, 4), y=(-1.25, 1.25, 1)).shift(1.15 * UP)
        f, vis = ValueTracker(2000), ValueTracker(1.0)
        t = np.linspace(0, n, 2400)
        k = np.arange(n + 1)

        def alias():
            v = f.get_value() % SR
            return min(v, SR - v)

        # cos, so 8 kHz gives clean ±1 dots, and the alias has no sign flip: cos(2π(sr−f)k/sr) = cos(2πfk/sr)
        true = always_redraw(lambda: wave(ax, t, np.cos(TAU * f.get_value() * t / SR), ACCENT, 2.5)
                             .set_stroke(opacity=vis.get_value()))
        dots = always_redraw(lambda: VGroup(*[Dot(p, radius=0.075, fill_color=GOLD)
                                                for p in c2p(ax, k, np.cos(TAU * f.get_value() * k / SR))]))
        ghost = always_redraw(lambda: wave(ax, t, np.cos(TAU * alias() * t / SR), PINK, 4.5)
                              .set_stroke(opacity=0.0 if f.get_value() <= SR / 2 + 1 else 0.95))

        # readout rebuilt only when the rounded numbers change
        def build_readout(fr):
            parts = [T("tone", 30, ACCENT), Text(f"{fr:,.0f} Hz", font="Menlo", font_size=32, color=ACCENT),
                     T("·", 30, DIM), Text(f"{SR / fr:.1f}", font="Menlo", font_size=32, color=GOLD), T("dots / cycle", 30, GOLD)]
            return VGroup(*parts).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.35)

        readout = build_readout(2000)
        readout.key = 2000

        def upd_readout(m):
            key = round(f.get_value() / 100) * 100
            if key != m.key:
                m.key = key
                m.set_submobjects(build_readout(key).submobjects)

        readout.add_updater(upd_readout)

        # frequency line 0 – 16 kHz, folding at 8 kHz
        line = NumberLine(x_range=(0, SR, 2000), width=12, stroke_color=RULE, include_ticks=False).shift(1.05 * DOWN)
        upper = Rectangle(width=6, height=0.36, stroke_width=0, fill_color=BAD, fill_opacity=0.12).move_to(line.n2p(12000))
        ny_tick = Line(line.n2p(8000) + 0.3 * DOWN, line.n2p(8000) + 0.3 * UP, stroke_color=GOLD, stroke_width=3)
        ticks = VGroup(T("0", 24, DIM).next_to(line.n2p(0), DOWN, buff=0.2),
                       T("8 kHz", 24, GOLD).next_to(line.n2p(8000), DOWN, buff=0.35),
                       T("16 kHz", 24, DIM).next_to(line.n2p(SR), DOWN, buff=0.2))
        tone_mk = always_redraw(lambda: Triangle(fill_color=ACCENT, fill_opacity=1, stroke_width=0).scale(0.13)
                                .rotate(PI).next_to(line.n2p(f.get_value()), UP, buff=0.05))
        alias_mk = always_redraw(lambda: Triangle(fill_color=PINK, fill_opacity=1, stroke_width=0).scale(0.13)
                                 .rotate(PI).next_to(line.n2p(alias()), UP, buff=0.05)
                                 .set_opacity(0 if f.get_value() <= SR / 2 + 1 else 1))
        fold = always_redraw(lambda: ArcBetweenPoints(line.n2p(f.get_value()) + 0.4 * UP, line.n2p(alias()) + 0.4 * UP,
                                                      angle=0.6, stroke_color=PINK, stroke_width=2.5)
                             .set_stroke(opacity=0 if f.get_value() <= SR / 2 + 50 else 0.9))

        self.play(ShowCreation(ax), ShowCreation(true), FadeIn(dots), FadeIn(readout))
        self.play(ShowCreation(line), FadeIn(upper), ShowCreation(ny_tick), FadeIn(ticks), FadeIn(tone_mk))
        self.say("2 kHz: plenty of dots per cycle")
        self.beat()

        self.play(f.animate.set_value(8000), run_time=3)
        self.say("8 kHz: just 2 dots per cycle", GOLD)
        self.beat()

        self.add(ghost, alias_mk)
        self.play(f.animate.set_value(10000), run_time=2)
        self.say("go past it: 10 kHz")
        self.beat()
        self.play(vis.animate.set_value(0.12), run_time=1)
        self.say("hide the tone: dots trace 6 kHz", PINK)
        self.beat()

        self.add(fold)
        fa = M(r"f_{\text{alias}} = 16\,000 - f", 44, PINK).next_to(line, DOWN, buff=0.95)
        self.play(Write(fa), Indicate(alias_mk, color=PINK))
        self.say("it folds back around 8 kHz", PINK)
        self.beat()

        self.play(vis.animate.set_value(0.6), run_time=0.5)
        self.play(f.animate.set_value(15000), run_time=3)
        self.say("15 kHz looks like 1 kHz", PINK)
        self.beat()

        ny = M(r"f_{\text{Nyquist}} = s_r / 2 = 8\,000\ \text{Hz}", 44, GOLD).move_to(fa)
        self.play(FadeTransform(fa, ny))
        self.say("Whisper hears 0 – 8 kHz only")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ quantization
    def quantize(self):
        y = 0.93 * np.sin(TAU * np.arange(24) / 24 + 0.3)
        n = len(y)
        ax = axes(w=11, h=4.6, x=(0, n, 4), y=(-1, 1, 0.25)).shift(0.2 * UP + 0.6 * RIGHT)
        bits = ValueTracker(3)

        def levels():
            return 2 ** int(round(bits.get_value()))

        orig = stems(ax, np.arange(n), y, DIM, 0.035)

        # the picture only changes when the level count does: rebuild then, not every frame
        def build(L):
            centres = (np.arange(L) + 0.5) / L * 2 - 1
            grid = VGroup(*[Line(ax.c2p(0, v), ax.c2p(n, v), stroke_color=RULE, stroke_width=1.2) for v in centres])
            q = (np.clip(np.floor((y + 1) / 2 * L), 0, L - 1) + 0.5) / L * 2 - 1
            dots = VGroup(*[Dot(ax.c2p(k, v), radius=0.07, fill_color=GOLD) for k, v in enumerate(q)])
            txt = Text(f"{int(np.log2(L))} bits → {L:,} levels", font="Menlo", font_size=34, color=GOLD).to_edge(UP, buff=0.45)
            return [grid, dots, txt]

        live = VGroup(*build(levels()))
        live.L = levels()

        def refresh(m):
            if levels() != m.L:
                m.L = levels()
                m.set_submobjects(build(m.L))

        live.add_updater(refresh)
        self.play(ShowCreation(ax), FadeIn(orig))
        self.play(FadeIn(live))
        self.say("each dot snaps to a level")
        self.beat()
        self.play(bits.animate.set_value(6), run_time=2)
        self.beat()
        fin = VGroup(
            T("16 bits → 65 536 levels", 40, GOLD),
            M(r"x = \frac{q}{32\,768}, \quad q \in \{-32\,768, \dots, 32\,767\}", 46),
        ).arrange(DOWN, buff=0.4)
        live.clear_updaters()
        self.play(*[FadeOut(m) for m in (live, orig, ax)])
        self.play(FadeIn(fin[0]), Write(fin[1]))
        self.say("int16 → float in [−1, 1)")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ stereo → mono, 44.1 kHz → 16 kHz
    def mono_resample(self):
        n = 44
        t = np.linspace(0, n, 600)
        L = lambda x: 0.8 * np.sin(TAU * x / 22) + 0.15 * np.sin(TAU * x / 5)
        R = lambda x: 0.6 * np.sin(TAU * x / 22 + 0.6) - 0.15 * np.sin(TAU * x / 7)
        axL, axR, axM = [axes(w=5.2, h=1.5, x=(0, n, 11), y=(-1, 1, 1)) for _ in range(3)]
        VGroup(axL, axR).arrange(DOWN, buff=0.5).to_edge(LEFT, buff=1.4).shift(0.4 * UP)
        axM.next_to(VGroup(axL, axR), RIGHT, buff=2.0)
        cL, cR = wave(axL, t, L(t), BLUE, 3), wave(axR, t, R(t), PINK, 3)
        cM = wave(axM, t, (L(t) + R(t)) / 2, ACCENT, 3.5)
        labs = VGroup(T("left", 28, BLUE).next_to(axL, LEFT), T("right", 28, PINK).next_to(axR, LEFT),
                      T("mono", 28, ACCENT).next_to(axM, UP))
        avg = M(r"\tfrac{L + R}{2}", 44).move_to(VGroup(axL, axR, axM)).shift(0.3 * RIGHT)
        self.play(*map(ShowCreation, (axL, axR, cL, cR)), FadeIn(labs[:2]))
        self.say("stereo → mono: average channels")
        self.play(Write(avg), TransformFromCopy(VGroup(cL, cR), cM), ShowCreation(axM), FadeIn(labs[2]))
        self.beat()
        self.clear_all()

        # resample: 44.1 kHz dots → 16 kHz dots on the same curve (1 ms window)
        ax = axes(w=12, h=3.2, x=(0, 1, 0.25), y=(-1.1, 1.1, 1)).shift(0.4 * UP)
        g = lambda x: 0.7 * np.sin(TAU * 1.3 * x) + 0.25 * np.sin(TAU * 4.1 * x)
        tt = np.linspace(0, 1, 500)
        c = wave(ax, tt, g(tt), DIM, 2)
        hi = VGroup(*[Dot(ax.c2p(x, g(x)), radius=0.045, fill_color=BLUE) for x in np.arange(0, 1, 1 / 44.1)])
        lo = VGroup(*[Dot(ax.c2p(x, g(x)), radius=0.07, fill_color=GOLD) for x in np.arange(0, 1, 1 / 16)])
        r1 = T("44 100 / s", 34, BLUE).to_edge(UP, buff=0.5)
        r2 = T("16 000 / s", 34, GOLD).to_edge(UP, buff=0.5)
        self.play(ShowCreation(ax), ShowCreation(c), FadeIn(hi), FadeIn(r1))
        self.say("resample: same sound, fewer dots")
        self.beat()
        self.play(FadeOut(hi), FadeIn(lo), FadeTransform(r1, r2))
        note = T("keeps 0 – 8 kHz (Nyquist)", 30, DIM).next_to(ax, DOWN, buff=0.4)
        self.play(FadeIn(note))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ pad to 30 s
    def pad_30s(self):
        W, S = 12.0, 40  # timeline 0 – 40 s across 12 units
        x0, y0 = -6.0, 0.55
        sx = lambda sec: x0 + W * sec / S
        box = Rectangle(width=sx(30) - x0, height=1.5, stroke_color=GOLD, stroke_width=3)
        box.move_to([(x0 + sx(30)) / 2, y0, 0])
        outside = Rectangle(width=sx(40) - sx(30), height=1.5, stroke_width=0, fill_color=RULE, fill_opacity=0.25)
        outside.move_to([(sx(30) + sx(40)) / 2, y0, 0])
        ticks = VGroup(*[T(f"{v} s", 24, GOLD if v == 30 else DIM).move_to([sx(v), y0 - 1.05, 0]) for v in (0, 10, 20, 30, 40)])
        head = VGroup(T("Whisper's window", 32, GOLD), tag("30 s × 16 000 = 480 000 samples", GOLD, 28))
        head.arrange(DOWN, buff=0.12).next_to(box, UP, buff=0.35)
        self.play(ShowCreation(box), FadeIn(outside), FadeIn(ticks), FadeIn(head))
        self.say("the input is always 30 s")
        self.beat()

        # case 1: our 4 s melody → pad with zeros
        y = dsp.melody()
        ax = axes(w=sx(4) - x0, h=1.3, x=(0, 4, 1), y=(-0.6, 0.6, 0.3)).move_to([(x0 + sx(4)) / 2, y0, 0])
        env = envelope(ax, np.arange(len(y)) / SR, y, bins=200)
        b_real = Brace(Line([x0, 0, 0], [sx(4), 0, 0]), DOWN).move_to([(x0 + sx(4)) / 2, y0 - 1.55, 0])
        t_real = tag("64 000 real", ACCENT, 24).next_to(b_real, DOWN, buff=0.08)
        self.play(ShowCreation(env), GrowFromCenter(b_real), FadeIn(t_real))
        self.say("our clip: only 4 s")
        self.beat()
        zeros = Rectangle(width=sx(30) - sx(4), height=1.5, stroke_width=0, fill_color=DIM, fill_opacity=0.25)
        zeros.move_to([(sx(4) + sx(30)) / 2, y0, 0])
        zt = T("zeros", 32, DIM).move_to(zeros)
        b_zero = Brace(Line([sx(4), 0, 0], [sx(30), 0, 0]), DOWN).move_to([(sx(4) + sx(30)) / 2, y0 - 1.55, 0])
        t_zero = tag("416 000 zeros", DIM, 24).next_to(b_zero, DOWN, buff=0.08)
        eq = M(r"64\,000 + 416\,000 = 480\,000", 44, GOLD).move_to(2.35 * DOWN)
        self.play(GrowFromEdge(zeros, LEFT), FadeIn(zt), GrowFromCenter(b_zero), FadeIn(t_zero), run_time=1.2)
        self.play(Write(eq))
        self.say("shorter → pad with zeros")
        self.beat()

        # case 2: a 40 s clip → keep the first 30 s
        self.play(*[FadeOut(m) for m in (env, zeros, zt, b_real, t_real, b_zero, t_zero, eq)])
        long = np.tile(y, 10)  # 40 s
        ax40 = axes(w=sx(40) - x0, h=1.3, x=(0, 40, 10), y=(-0.6, 0.6, 0.3)).move_to([0, y0, 0])
        env40 = envelope(ax40, np.arange(len(long)) / SR, long, bins=600, color=ACCENT)
        cut = DashedLine([sx(30), y0 + 1.0, 0], [sx(30), y0 - 0.85, 0], stroke_color=BAD, stroke_width=4)
        self.play(ShowCreation(env40), run_time=1.5)
        self.say("a 40 s clip?")
        self.beat()
        tail = Rectangle(width=sx(40) - sx(30), height=1.5, stroke_width=0, fill_color=BAD, fill_opacity=0.3).move_to(outside)
        ct = T("cut", 30, BAD).next_to(cut, UP, buff=0.1)
        eq2 = M(r"\text{first } 30\text{ s} = 480\,000", 44, GOLD).move_to(2.35 * DOWN)
        self.play(ShowCreation(cut), FadeIn(ct), FadeIn(tail))
        self.play(Write(eq2))
        self.say("longer → cut at 30 s", BAD)
        self.beat()
        self.clear_all()


split_sections(L1_Waveform, globals())
