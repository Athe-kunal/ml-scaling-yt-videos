"""Lesson 2 · Fourier transform → STFT."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

SR, N = dsp.SAMPLE_RATE, dsp.N_FFT
F0 = dsp.midi_hz(60)  # C4 ≈ 261.6 Hz, the first note of the melody
HARM = [(1, 1.0), (2, 0.55), (3, 0.3)]  # a simple 3-harmonic note for the intuition beats


def axes(w=12, h=3.2, x=(0, 1, 0.25), y=(-1, 1, 1)):
    return Axes(x_range=x, y_range=y, width=w, height=h,
                axis_config=dict(stroke_color=RULE, stroke_width=2, include_tip=False, include_ticks=False))


def note(t):
    return sum(a * np.sin(TAU * F0 * h * t) for h, a in HARM)


def stft_generic(y, n_fft, hop):
    yp = np.pad(y, n_fft // 2, mode="reflect")
    n = 1 + (len(yp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    w = 0.5 - 0.5 * np.cos(TAU * np.arange(n_fft) / n_fft)
    return np.abs(np.fft.rfft(yp[idx] * w, axis=1).T) ** 2


def db(p):
    return 10 * np.log10(p + 1e-10)


class L2_Fourier(Lesson):
    NUM, TITLE, SUB = 2, "Fourier → STFT", "which frequencies are inside?"
    SECTIONS = ["title", "decompose", "probe", "winding", "dft", "framing", "leakage",
                "frame_count", "stft", "tradeoff"]

    # ------------------------------------------------------------ a note = sum of sines
    def decompose(self):
        dur = 0.015
        t = np.linspace(0, dur, 900)
        rows = VGroup()
        cols = [ACCENT, PINK, GOLD]
        for (h, a), c in zip(HARM, cols):
            ax = axes(w=5.6, h=1.1, x=(0, dur, dur / 3), y=(-1, 1, 1))
            rows.add(VGroup(ax, wave(ax, t, a * np.sin(TAU * F0 * h * t), c, 3),
                            T(f"{F0 * h:.0f} Hz", 26, c)))
        for r in rows:
            r[2].next_to(r[0], LEFT, buff=0.3)
        rows.arrange(DOWN, buff=0.35).to_edge(LEFT, buff=1.6).shift(0.4 * UP)
        axs = axes(w=4.4, h=3.2, x=(0, dur, dur / 3), y=(-1.9, 1.9, 1)).next_to(rows, RIGHT, buff=1.3)
        sumc = wave(axs, t, note(t), INK, 3.5)
        plus = M(r"\sum", 70).next_to(axs, LEFT, buff=0.35)
        self.play(LaggedStart(*[AnimationGroup(ShowCreation(r[0]), ShowCreation(r[1]), FadeIn(r[2])) for r in rows],
                              lag_ratio=0.3), run_time=2)
        self.say("a note is a sum of sines")
        self.play(ShowCreation(axs), FadeIn(plus), TransformFromCopy(VGroup(*[r[1] for r in rows]), sumc), run_time=1.5)
        self.beat()
        q = T("given only the sum… find the sines?", 36, GOLD).to_edge(UP, buff=0.45)
        self.play(FadeOut(rows), FadeOut(plus), FadeIn(q))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ multiply by a probe and measure the area
    def probe(self):
        dur = 0.025  # one 25 ms window, like Whisper's 400 samples
        t = np.linspace(0, dur, 1500)
        g = note(t)
        f = ValueTracker(150.0)

        ax_g = axes(w=6.2, h=1.2, x=(0, dur, dur / 5), y=(-2, 2, 1))
        ax_p = axes(w=6.2, h=1.2, x=(0, dur, dur / 5), y=(-1.1, 1.1, 1))
        ax_m = axes(w=6.2, h=1.4, x=(0, dur, dur / 5), y=(-2, 2, 1))
        left = VGroup(ax_g, ax_p, ax_m).arrange(DOWN, buff=0.55).to_edge(LEFT, buff=0.9).shift(0.35 * UP)
        lab = VGroup(T("signal", 24, INK).next_to(ax_g, UP, buff=0.05).align_to(ax_g, LEFT),
                     T("probe", 24, PINK).next_to(ax_p, UP, buff=0.05).align_to(ax_p, LEFT),
                     T("product", 24, GOLD).next_to(ax_m, UP, buff=0.05).align_to(ax_m, LEFT))

        cg = wave(ax_g, t, g, INK, 2.5)
        cp = always_redraw(lambda: wave(ax_p, t, np.sin(TAU * f.get_value() * t), PINK, 2.5))

        def area_poly():
            prod = g * np.sin(TAU * f.get_value() * t)
            step = 3
            tt, pp = t[::step], prod[::step]
            xs = np.concatenate([[tt[0]], tt, [tt[-1]]])
            pos = c2p(ax_m, xs, np.concatenate([[0], np.maximum(pp, 0), [0]]))
            neg = c2p(ax_m, xs, np.concatenate([[0], np.minimum(pp, 0), [0]]))
            return VGroup(Polygon(*pos, stroke_width=0, fill_color=GOLD, fill_opacity=0.55),
                          Polygon(*neg, stroke_width=0, fill_color=BLUE, fill_opacity=0.55),
                          wave(ax_m, t, prod, GOLD, 2))

        cm = always_redraw(area_poly)

        def area(fr):
            return np.trapezoid(g * np.sin(TAU * fr * t), t) / dur

        ax_s = Axes(x_range=(0, 1000, 250), y_range=(-0.1, 0.6, 0.5), width=5.6, height=3.6,
                    axis_config=dict(stroke_color=RULE, include_tip=False, include_ticks=False)).to_edge(RIGHT, buff=0.7).shift(0.3 * UP)
        xs = VGroup(*[T(f"{v}", 22, DIM).next_to(ax_s.c2p(v, -0.1), DOWN, buff=0.1) for v in (0, 262, 523, 785)])
        hz = T("probe Hz", 24, DIM).next_to(ax_s, DOWN, buff=0.55)
        ys = T("area", 24, GOLD).next_to(ax_s, UP, buff=0.1).align_to(ax_s, LEFT)
        trace = TracedPath(lambda: ax_s.c2p(f.get_value(), area(f.get_value())), stroke_color=GOLD, stroke_width=3)
        dot = always_redraw(lambda: Dot(ax_s.c2p(f.get_value(), area(f.get_value())), radius=0.07, fill_color=GOLD))
        fr = always_redraw(lambda: Text(f"{f.get_value():4.0f} Hz", font="Menlo", font_size=30, color=PINK)
                           .to_edge(UP, buff=0.35))
        formula = M(r"\int g(t)\,\sin(2\pi f t)\,dt", 44).next_to(ax_s, UP, buff=0.55)

        self.play(*map(ShowCreation, (ax_g, ax_p, ax_m, cg)), FadeIn(lab))
        self.add(cp, cm, fr)
        self.say("multiply by a probe sine")
        self.beat()
        self.play(ShowCreation(ax_s), FadeIn(xs), FadeIn(hz), FadeIn(ys), Write(formula))
        self.add(trace, dot)
        self.say("sum the product: the area")
        self.play(f.animate.set_value(F0), run_time=2.5, rate_func=linear)
        self.beat()
        self.say("big area ⇒ that frequency is inside", GOLD)
        self.play(f.animate.set_value(950), run_time=7, rate_func=linear)
        self.beat()
        # phase problem: the same 262 Hz tone shifted by 90° gives zero area
        self.say("but shift the phase…", PINK)
        self.clear_all()
        self.phase_problem()

    def phase_problem(self):
        dur = 0.025
        t = np.linspace(0, dur, 1200)
        ax = axes(w=11, h=2.2, x=(0, dur, dur / 5), y=(-1.2, 1.2, 1)).to_edge(UP, buff=0.5)
        g = np.cos(TAU * F0 * t)  # same tone, 90° later
        p = np.sin(TAU * F0 * t)
        self.play(ShowCreation(ax), ShowCreation(wave(ax, t, g, INK, 3)), ShowCreation(wave(ax, t, p, PINK, 2.5)))
        val = M(r"\int \cos \cdot \sin\,dt \approx 0", 50, BAD).next_to(ax, DOWN, buff=0.4)
        self.play(Write(val))
        self.say("right frequency, zero area")
        self.beat()
        fix = VGroup(M(r"\text{probe with }\cos\text{ and }\sin", 46),
                     M(r"e^{-i2\pi f t} = \cos(2\pi f t) - i \sin(2\pi f t)", 52, ACCENT)).arrange(DOWN, buff=0.4)
        fix.next_to(ax, DOWN, buff=0.4)
        self.play(FadeOut(val), FadeIn(fix[0]))
        self.play(Write(fix[1]))
        self.say("use both at once: a complex probe")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ winding around the complex plane
    def winding(self):
        dur = 0.025
        t = np.linspace(0, dur, 2400)
        g = note(t) / 1.85 + 0.0
        f = ValueTracker(60.0)
        plane = ComplexPlane(x_range=(-1.2, 1.2, 0.5), y_range=(-1.2, 1.2, 0.5), width=5.6, height=5.6,
                             background_line_style=dict(stroke_color=RULE, stroke_width=1, stroke_opacity=0.6),
                             axis_config=dict(stroke_color=DIM))
        plane.to_edge(LEFT, buff=1.0).shift(0.2 * UP)

        def z():
            return g * np.exp(-1j * TAU * f.get_value() * t)

        def curve():
            zz = z()[::2]
            return polyline(c2p(plane, zz.real, zz.imag), ACCENT, 2)

        def com():
            return complex(z().mean())

        wound = always_redraw(curve)
        cdot = always_redraw(lambda: Dot(plane.n2p(com()), radius=0.1, fill_color=GOLD))
        ax_s = Axes(x_range=(0, 1000, 250), y_range=(0, 0.35, 0.1), width=5.6, height=3.6,
                    axis_config=dict(stroke_color=RULE, include_tip=False, include_ticks=False)).to_edge(RIGHT, buff=0.7).shift(0.2 * DOWN)
        xs = VGroup(*[T(f"{v}", 22, DIM).next_to(ax_s.c2p(v, 0), DOWN, buff=0.1) for v in (0, 262, 523, 785)])
        ys = T("|center of mass|", 24, GOLD).next_to(ax_s, UP, buff=0.1).align_to(ax_s, LEFT)
        trace = TracedPath(lambda: ax_s.c2p(f.get_value(), abs(com())), stroke_color=GOLD, stroke_width=3)
        fr = always_redraw(lambda: Text(f"winding at {f.get_value():4.0f} Hz", font="Menlo", font_size=28, color=PINK)
                           .next_to(plane, UP, buff=0.2))
        formula = M(r"\hat g(f) = \int g(t)\, e^{-i 2\pi f t}\, dt", 48).to_edge(UP, buff=0.35).shift(2.6 * RIGHT)

        self.play(ShowCreation(plane), FadeIn(fr))
        self.play(ShowCreation(wound), run_time=2)
        self.say("wrap the signal around a circle")
        self.beat()
        self.play(FadeIn(cdot), ShowCreation(ax_s), FadeIn(xs), FadeIn(ys), Write(formula))
        self.add(trace)
        self.say("track its center of mass", GOLD)
        self.play(f.animate.set_value(F0), run_time=4, rate_func=linear)
        self.beat()
        self.say("it jumps out at each harmonic", GOLD)
        self.play(f.animate.set_value(900), run_time=9, rate_func=linear)
        self.beat()
        polar = VGroup(M(r"|\hat g(f)|", 44, GOLD), T("= how much", 30, DIM),
                       M(r"\angle \hat g(f)", 44, PINK), T("= phase", 30, DIM)).arrange(RIGHT, buff=0.25)
        polar.next_to(xs, DOWN, buff=0.35)
        self.play(FadeIn(polar))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the DFT with Whisper's numbers
    def dft(self):
        A = dsp.whisper_input()
        y = A["y30"]
        t0 = 30  # frame 30 = 0.30 s, inside the first note
        fr = dsp.frames(y)[t0] * dsp.hann()
        X = np.fft.fft(fr)
        mag = np.abs(X)

        f1 = M(r"\hat x(k) = \sum_{n=0}^{N-1} x(n)\, e^{-i 2\pi n k / N}", 50).to_edge(UP, buff=0.4)
        nn = VGroup(M(r"N = 400", 44, GOLD), M(r"F(k) = k \cdot \tfrac{s_r}{N} = 40k\ \text{Hz}", 44, ACCENT)).arrange(RIGHT, buff=1.0)
        nn.next_to(f1, DOWN, buff=0.35)
        self.play(Write(f1))
        self.play(FadeIn(nn))
        self.say("integral → sum over 400 samples")
        self.beat()

        ax = Axes(x_range=(0, 400, 100), y_range=(0, mag.max() * 1.1, 1), width=12, height=3.2,
                  axis_config=dict(stroke_color=RULE, include_tip=False, include_ticks=False)).shift(1.1 * DOWN)
        bars = VGroup(*[Line(ax.c2p(k, 0), ax.c2p(k, mag[k]), stroke_width=2.2,
                             stroke_color=ACCENT if k <= 200 else DIM) for k in range(400)])
        ticks = VGroup(*[T(lbl, 22, DIM).next_to(ax.c2p(k, 0), DOWN, buff=0.12)
                         for k, lbl in [(0, "0"), (200, "8 kHz"), (399, "16 kHz")]])
        tg = tag("400 samples → 400 complex numbers", GOLD, 26).next_to(ax, UP, buff=0.1)
        self.play(ShowCreation(ax), FadeIn(ticks), LaggedStartMap(GrowFromEdge, bars, edge=DOWN, lag_ratio=0.004), run_time=2)
        self.play(FadeIn(tg))
        self.say("|x̂(k)| of one 25 ms frame")
        self.beat()
        mirror = SurroundingRectangle(VGroup(*bars[201:]), stroke_color=BAD, buff=0.05)
        self.play(ShowCreation(mirror))
        self.say("above 8 kHz: a mirror image", BAD)
        self.beat()
        tg2 = tag("keep k = 0 … 200 → 201 bins", ACCENT, 30).move_to(tg)
        self.play(FadeOut(VGroup(*bars[201:])), FadeOut(mirror), FadeTransform(tg, tg2))
        self.say("real input ⇒ keep N/2 + 1 = 201")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ frames slide over the melody
    def framing(self):
        A = dsp.whisper_input()
        y = A["y"]
        sec = 0.08  # first 80 ms so 25 ms boxes are visible
        n = int(sec * SR)
        ax = axes(w=12.5, h=2.8, x=(0, n, 400), y=(-0.6, 0.6, 0.3)).shift(0.8 * UP)
        c = wave(ax, np.arange(n), y[:n] + 0, ACCENT, 2)
        self.play(ShowCreation(ax), ShowCreation(c))
        unit = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        boxes = VGroup()
        for i in range(4):
            r = Rectangle(width=400 * unit, height=3.0, stroke_color=[GOLD, PINK, BLUE, ACCENT][i], stroke_width=3)
            r.move_to(ax.c2p(i * 160 + 200, 0))
            boxes.add(r)
        self.play(ShowCreation(boxes[0]))
        b1 = Brace(boxes[0], DOWN)
        t1 = VGroup(tag("400 samples", GOLD, 26), T("= 25 ms", 26, DIM)).arrange(RIGHT, buff=0.2).next_to(b1, DOWN, buff=0.1)
        self.play(GrowFromCenter(b1), FadeIn(t1))
        self.say("cut a short frame")
        self.beat()
        self.play(FadeOut(b1), FadeOut(t1))
        for i in range(1, 4):
            self.play(TransformFromCopy(boxes[i - 1], boxes[i]), run_time=0.6)
        a, b = boxes[0].get_corner(DL) + 0.4 * DOWN, boxes[1].get_corner(DL) + 0.4 * DOWN
        hop = VGroup(Arrow(a, b, buff=0, stroke_color=PINK), Arrow(b, a, buff=0, stroke_color=PINK))
        ht = VGroup(tag("hop 160", PINK, 26), T("= 10 ms", 26, DIM)).arrange(RIGHT, buff=0.2).next_to(hop, DOWN, buff=0.1)
        self.play(GrowFromCenter(hop), FadeIn(ht))
        self.say("slide by a hop · frames overlap")
        self.beat()
        why = T("speech ≈ steady over 20–30 ms", 34, GOLD).to_edge(UP, buff=0.3)
        self.play(FadeIn(why))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ leakage and the Hann window
    def leakage(self):
        n = np.arange(N)
        k0 = 10.5  # a tone between two bins
        x = np.sin(TAU * k0 * n / N)
        w = dsp.hann()
        ax1 = axes(w=6, h=2.2, x=(0, N, 100), y=(-1.1, 1.1, 1))
        ax2 = Axes(x_range=(0, 40, 10), y_range=(-80, 0, 20), width=5.6, height=2.6,
                   axis_config=dict(stroke_color=RULE, include_tip=False, include_ticks=False))
        VGroup(ax1, ax2).arrange(RIGHT, buff=1.2).shift(0.7 * UP)

        def spec(sig):
            m = np.abs(np.fft.rfft(sig))[:41]
            return 20 * np.log10(m / np.abs(np.fft.rfft(x)).max() + 1e-4)

        def bars(s, col):
            return VGroup(*[Line(ax2.c2p(k, -80), ax2.c2p(k, max(v, -80)), stroke_color=col, stroke_width=5)
                            for k, v in enumerate(s)])

        raw = wave(ax1, n, x, ACCENT, 2.5)
        edge = VGroup(Dot(ax1.c2p(0, x[0]), fill_color=BAD), Dot(ax1.c2p(N - 1, x[-1]), fill_color=BAD))
        b_raw = bars(spec(x), BAD)
        lbl1 = T("frame", 26, DIM).next_to(ax1, UP, buff=0.1)
        lbl2 = T("spectrum (dB)", 26, DIM).next_to(ax2, UP, buff=0.1)
        self.play(ShowCreation(ax1), ShowCreation(raw), FadeIn(lbl1))
        self.play(FadeIn(edge, scale=2))
        self.say("hard cut: ends don't match", BAD)
        self.beat()
        self.play(ShowCreation(ax2), FadeIn(lbl2), LaggedStartMap(GrowFromEdge, b_raw, edge=DOWN, lag_ratio=0.02))
        self.say("energy leaks into every bin", BAD)
        self.beat()
        hw = wave(ax1, n, w, GOLD, 3)
        f = M(r"w(n) = 0.5 - 0.5\cos\big(\tfrac{2\pi n}{N}\big)", 46, GOLD).to_edge(UP, buff=0.35)
        self.play(ShowCreation(hw), Write(f))
        self.say("Hann window: fade in, fade out", GOLD)
        self.beat()
        win = wave(ax1, n, x * w, ACCENT, 2.5)
        b_win = bars(spec(x * w), ACCENT)
        self.play(Transform(raw, win), FadeOut(edge), FadeOut(hw), Transform(b_raw, b_win), run_time=1.5)
        self.say("leakage drops far more than peak")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ how many frames: 3001 → 3000
    def frame_count(self):
        bar = Rectangle(width=11, height=0.8, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.15)
        padL = Rectangle(width=0.6, height=0.8, stroke_color=PINK, fill_color=PINK, fill_opacity=0.3).next_to(bar, LEFT, buff=0)
        padR = padL.copy().next_to(bar, RIGHT, buff=0)
        lab = tag("480 000 samples", ACCENT, 28).move_to(bar)
        self.play(ShowCreation(bar), FadeIn(lab))
        self.play(GrowFromEdge(padL, RIGHT), GrowFromEdge(padR, LEFT))
        pt = VGroup(tag("200", PINK, 24).next_to(padL, UP), tag("200", PINK, 24).next_to(padR, UP))
        self.play(FadeIn(pt))
        self.say("reflect-pad 200 so every frame is centered")
        self.beat()
        eq = VGroup(
            M(r"\text{frames} = 1 + \frac{480\,000}{160} = 3001", 52),
            M(r"\text{drop last} \;\Rightarrow\; 3000\ \text{frames} = 30\text{ s} \times 100", 46, GOLD),
        ).arrange(DOWN, buff=0.5).shift(1.6 * DOWN)
        self.play(Write(eq[0]))
        self.beat()
        self.play(Write(eq[1]))
        self.say("one frame every 10 ms")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ stack the spectra: the spectrogram
    def stft(self):
        A = dsp.whisper_input()
        P = A["power"]
        C1 = 420  # first 4.2 s
        f = M(r"S(m, k) = \sum_{n} x\big(n + mH - \tfrac{N}{2}\big)\, w(n)\, e^{-i 2\pi n k / N}", 46).to_edge(UP, buff=0.35)
        self.play(Write(f))
        sub = VGroup(T("m: frame", 28, PINK), T("k: frequency bin", 28, ACCENT), T("H = 160", 28, GOLD)).arrange(RIGHT, buff=0.8)
        sub.next_to(f, DOWN, buff=0.25)
        self.play(FadeIn(sub))
        self.say("one DFT per frame")
        self.beat()

        D = db(P[:, :C1])
        img = heat(D, "l2_spec", 11.5, 3.9, vmin=D.max() - 80, vmax=D.max()).shift(0.35 * DOWN)
        border = SurroundingRectangle(img, buff=0, stroke_color=RULE, stroke_width=1.5)
        # columns grow left → right
        cover = Rectangle(width=11.5, height=3.9, stroke_width=0, fill_color=BG, fill_opacity=1).move_to(img)
        self.add(img, cover, border)
        lk = VGroup(T("0", 22, DIM).next_to(img.get_corner(DL), LEFT, buff=0.15),
                    T("8 kHz", 22, DIM).next_to(img.get_corner(UL), LEFT, buff=0.15),
                    T("4.2 s", 22, DIM).next_to(img.get_corner(DR), DOWN, buff=0.1))
        self.play(FadeIn(lk), FadeOut(sub))
        self.play(cover.animate.stretch(0.001, 0, about_edge=RIGHT), run_time=3, rate_func=linear)
        self.remove(cover)
        shp = tag("[201, 3000]  power |S|²", GOLD, 30).next_to(border, UP, buff=0.12).align_to(border, RIGHT)
        self.play(FadeIn(shp))
        self.say("each column = one frame's spectrum")
        self.beat()
        ax = VGroup(Arrow(border.get_corner(DL) + 0.25 * DOWN, border.get_corner(DR) + 0.25 * DOWN, buff=0,
                          stroke_color=PINK, stroke_width=3),
                    T("time →", 26, PINK).next_to(border.get_corner(DL), DOWN, buff=0.45).shift(1.0 * RIGHT))
        self.play(FadeIn(ax))
        self.say("harmonic stacks = the 8 notes")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ window size trade-off
    def tradeoff(self):
        y = dsp.whisper_input()["y30"][: int(4.2 * SR)]
        imgs = Group()
        for n_fft, lab, col in [(100, "6 ms window", PINK), (400, "25 ms · Whisper", ACCENT), (1600, "100 ms window", BLUE)]:
            D = db(stft_generic(y, n_fft, min(160, n_fft // 2)))  # frames must overlap
            im = heat(D, f"l2_trade_{n_fft}", 3.9, 3.8, vmin=D.max() - 80, vmax=D.max())
            b = SurroundingRectangle(im, buff=0, stroke_color=col, stroke_width=2)
            t = T(lab, 26, col).next_to(im, UP, buff=0.15)
            sh = tag(f"[{n_fft // 2 + 1}, …]", DIM, 22).next_to(im, DOWN, buff=0.1)
            imgs.add(Group(im, b, t, sh))
        imgs.arrange(RIGHT, buff=0.4).shift(0.3 * UP)
        for g in imgs:
            self.play(FadeIn(g), run_time=0.8)
        self.say("short: sharp time · long: sharp pitch")
        self.beat()
        dt = M(r"\Delta t \cdot \Delta f \geq \text{const}", 52, GOLD).to_edge(UP, buff=0.25)
        self.play(Write(dt))
        self.beat()
        self.clear_all()


split_sections(L2_Fourier, globals())
