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
            (r"\varphi", ph, [PI, 2 * PI], "phase → where it starts", BLUE),
        ]:
            self.play(formula.select_part(sym).animate.set_color(col), run_time=0.4)
            self.say(word, col)
            for v in vals:
                self.play(trk.animate.set_value(v), run_time=1.0)
            self.beat()
            self.play(formula.select_part(sym).animate.set_color(INK), run_time=0.3)
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
        self.say("the same clip all lecture")
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
        n = 32
        ax = axes(w=12, h=3.4, x=(0, n, 4), y=(-1.2, 1.2, 1)).shift(0.4 * UP)
        f = ValueTracker(1000)
        t = np.linspace(0, n, 2400)
        true = always_redraw(lambda: wave(ax, t, np.sin(TAU * f.get_value() * t / SR), ACCENT, 2.5))
        dots = always_redraw(lambda: VGroup(*[Dot(ax.c2p(k, np.sin(TAU * f.get_value() * k / SR)), radius=0.06,
                                                   fill_color=GOLD) for k in range(n + 1)]))

        def alias_f():
            v = f.get_value() % SR
            return min(v, SR - v)

        ghost = always_redraw(lambda: wave(
            ax, t, np.sin(TAU * f.get_value() * 0 + TAU * alias_f() * t / SR) * (1 if (f.get_value() % SR) <= SR / 2 else -1),
            PINK, 4.5).set_fill(opacity=0).set_stroke(opacity=0.0 if f.get_value() <= 8000 else 0.95))
        readout = always_redraw(lambda: VGroup(
            T("tone", 30, ACCENT), Text(f"{f.get_value():5.0f} Hz", font="Menlo", font_size=32, color=ACCENT),
            T("dots look like", 30, PINK), Text(f"{alias_f():5.0f} Hz", font="Menlo", font_size=32, color=PINK),
        ).arrange(RIGHT, buff=0.3).to_edge(UP, buff=0.5))
        self.play(ShowCreation(ax), ShowCreation(true), FadeIn(dots), FadeIn(readout))
        self.say("raise the pitch, keep 16 000 dots / s")
        self.play(f.animate.set_value(6000), run_time=3)
        self.beat()
        self.add(ghost)
        self.play(f.animate.set_value(15000), run_time=4)
        self.say("above 8 kHz the dots lie: aliasing", PINK)
        self.beat()
        ny = M(r"f_{\text{Nyquist}} = \frac{s_r}{2} = 8\,000\ \text{Hz}", 54, GOLD).next_to(ax, DOWN, buff=0.4)
        self.play(Write(ny))
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
        y = dsp.melody()
        bar = Rectangle(width=12.5, height=1.6, stroke_color=RULE).shift(0.5 * UP)
        x0 = bar.get_left()[0]
        w4 = 12.5 * 4 / 30
        ax = axes(w=w4, h=1.5, x=(0, 4, 1), y=(-0.6, 0.6, 0.3)).move_to([x0 + w4 / 2, 0.5, 0])
        env = envelope(ax, np.arange(len(y)) / SR, y, bins=300)
        zeros = Rectangle(width=12.5 - w4, height=1.6, stroke_width=0, fill_color=RULE, fill_opacity=0.35)
        zeros.move_to([x0 + w4 + (12.5 - w4) / 2, 0.5, 0])
        self.play(ShowCreation(bar), ShowCreation(env))
        t4 = tag("4 s · 64 000 samples", ACCENT, 26).next_to(ax, UP, buff=0.25)
        self.play(FadeIn(t4))
        self.say("Whisper always takes 30 s")
        self.beat()
        self.play(GrowFromEdge(zeros, LEFT), run_time=1.2)
        zt = T("zeros", 30, DIM).move_to(zeros)
        b = Brace(bar, DOWN)
        tot = tag("[480 000] = 30 s × 16 000", GOLD, 34).next_to(b, DOWN, buff=0.2)
        self.play(FadeIn(zt), GrowFromCenter(b), Write(tot))
        self.say("pad (or cut) to 480 000 samples")
        self.beat()
        self.clear_all()


split_sections(L1_Waveform, globals())
