"""Lesson 3 · Mel → log-mel spectrogram."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

C1 = 420  # show the first 4.2 s of the 30 s (the rest is padding)


def db(p):
    return 10 * np.log10(p + 1e-10)


def plain_axes(x, y, w, h):
    return Axes(x_range=x, y_range=y, width=w, height=h,
                axis_config=dict(stroke_color=RULE, stroke_width=2, include_tip=False, include_ticks=False))


class L3_LogMel(Lesson):
    NUM, TITLE, SUB = 3, "Mel → log-mel", "how ears hear pitch"
    SECTIONS = ["title", "start_power", "pitch_is_log", "mel_curve", "filterbank", "matmul", "log_steps"]
    TOPICS = {"start_power": "the power spectrogram", "pitch_is_log": "pitch is logarithmic", "mel_curve": "the mel scale",
              "filterbank": "building the mel filterbank", "matmul": "filters @ power", "log_steps": "log, floor, scale"}

    def prepare(self):
        self.A = dsp.whisper_input()

    # ------------------------------------------------------------ where we are
    def start_power(self):
        P = self.A["power"][:, :C1]
        img = heat(db(P), "l3_power_db", 11, 4.0, vmin=db(P).max() - 80, vmax=db(P).max()).shift(0.2 * UP)
        b = SurroundingRectangle(img, buff=0, stroke_color=RULE, stroke_width=1.5)
        t = tag("power  [201, 3000]", GOLD, 32).next_to(b, UP, buff=0.15)
        ax = VGroup(T("0", 24, DIM).next_to(img.get_corner(DL), LEFT, buff=0.15),
                    T("8 kHz", 24, DIM).next_to(img.get_corner(UL), LEFT, buff=0.15))
        self.play(FadeIn(img), ShowCreation(b), FadeIn(t), FadeIn(ax))
        self.say("201 bins, each 40 Hz wide")
        self.beat()
        top = Rectangle(width=img.get_width(), height=img.get_height() / 2, stroke_color=BAD, stroke_width=3,
                        fill_color=BAD, fill_opacity=0.12).align_to(img, UP).match_x(img)
        self.play(ShowCreation(top))
        self.say("half the bins: 4–8 kHz", BAD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ octaves double in Hz
    def pitch_is_log(self):
        notes = [("C3", 131), ("C4", 262), ("C5", 523), ("C6", 1047), ("C7", 2093)]
        ax = NumberLine(x_range=(0, 2200, 500), width=10.5, stroke_color=RULE, include_ticks=False).shift(0.6 * UP)
        dots = VGroup(*[Dot(ax.n2p(f), radius=0.1, fill_color=ACCENT) for _, f in notes])
        labs = VGroup(*[VGroup(T(n, 28, INK), T(f"{f}", 24, DIM)).arrange(DOWN, buff=0.08).next_to(ax.n2p(f), UP, buff=0.2)
                        for n, f in notes])
        hz = T("Hz", 26, DIM).next_to(ax, RIGHT)
        self.play(ShowCreation(ax), FadeIn(hz))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(d, scale=2), FadeIn(l)) for d, l in zip(dots, labs)], lag_ratio=0.25))
        self.say("each octave sounds the same step")
        self.beat()
        ax2 = NumberLine(x_range=(0, 5, 1), width=10.5, stroke_color=RULE, include_ticks=False).shift(1.6 * DOWN)
        d2 = VGroup(*[Dot(ax2.n2p(i + 0.5), radius=0.1, fill_color=GOLD) for i in range(5)])
        l2 = VGroup(*[T(n, 28, GOLD).next_to(ax2.n2p(i + 0.5), DOWN, buff=0.2) for i, (n, _) in enumerate(notes)])
        p2 = T("perceived", 26, GOLD).next_to(ax2, RIGHT)
        self.play(ShowCreation(ax2), FadeIn(p2), TransformFromCopy(dots, d2), FadeIn(l2))
        self.say("but in Hz each step doubles", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the mel curve
    def mel_curve(self):
        ax = plain_axes((0, 8000, 1000), (0, 3000, 500), 7.2, 5.0).shift(0.4 * UP + 2.9 * LEFT)
        xs = VGroup(*[T(f"{v // 1000} k", 24, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in range(0, 8001, 2000)])
        xl = T("Hz", 26, DIM).next_to(ax, DOWN, buff=0.55)
        yl = T("mel", 26, ACCENT).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        htk = graph(ax, dsp.hz_to_mel_htk, (0, 8000), ACCENT, 5)
        f1 = M(r"m = 2595 \log_{10}\Big(1 + \frac{f}{700}\Big)", 44, ACCENT).move_to(3.6 * RIGHT + 2.2 * UP)
        self.play(ShowCreation(ax), FadeIn(xs), FadeIn(xl), FadeIn(yl))
        self.play(ShowCreation(htk), Write(f1), run_time=1.5)
        self.say("mel: equal steps sound equal")
        self.beat()
        lin = DashedLine(ax.c2p(0, 0), ax.c2p(1000, float(dsp.hz_to_mel_htk(1000))), stroke_color=GOLD, stroke_width=3)
        k1 = DashedLine(ax.c2p(1000, 0), ax.c2p(1000, 3000), stroke_color=RULE)
        n1 = T("≈ linear < 1 kHz", 28, GOLD).next_to(ax.c2p(1000, 1000), RIGHT, buff=0.2)
        n2 = T("≈ log above", 28, PINK).next_to(ax.c2p(5000, 2300), UP, buff=0.2)
        self.play(ShowCreation(k1), ShowCreation(lin), FadeIn(n1))
        self.play(FadeIn(n2))
        self.beat()
        # Whisper's mel (Slaney) — rescale to the same end point so the shapes compare
        s_end = float(dsp.hz_to_mel(8000)); h_end = float(dsp.hz_to_mel_htk(8000))
        sl = graph(ax, lambda f: dsp.hz_to_mel(f) / s_end * h_end, (0, 8000), PINK, 4)
        sl_lab = T("Whisper's mel (Slaney, rescaled)", 30, PINK).next_to(f1, DOWN, buff=0.5).align_to(f1, LEFT)
        f2 = M(r"\text{linear} < 1\text{ kHz},\ \log \text{ above}", 36, PINK).next_to(sl_lab, DOWN, buff=0.2).align_to(sl_lab, LEFT)
        self.play(FadeOut(n1), FadeOut(n2), FadeOut(lin), ShowCreation(sl), FadeIn(sl_lab), FadeIn(f2), run_time=1.5)
        self.say("same idea, slightly different curve", PINK)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ building the filterbank in 5 steps
    def filterbank(self):
        nm = 10  # build with 10 bands so every point is visible, then show Whisper's 80
        m_max = float(dsp.hz_to_mel(8000))
        mel_pts = np.linspace(0, m_max, nm + 2)
        hz_pts = dsp.mel_to_hz(mel_pts)

        mel_ax = NumberLine(x_range=(0, m_max, 10), width=12, stroke_color=ACCENT, include_ticks=False).shift(2.3 * UP)
        hz_ax = NumberLine(x_range=(0, 8000, 1000), width=12, stroke_color=GOLD, include_ticks=False).shift(0.3 * UP)
        ml = T("mel", 28, ACCENT).next_to(mel_ax, LEFT)
        hl = T("Hz", 28, GOLD).next_to(hz_ax, LEFT)
        ends = VGroup(T("0", 24, DIM).next_to(mel_ax.n2p(0), UP), T(f"{m_max:.1f}", 24, DIM).next_to(mel_ax.n2p(m_max), UP),
                      T("0", 24, DIM).next_to(hz_ax.n2p(0), DOWN), T("8 000", 24, DIM).next_to(hz_ax.n2p(8000), DOWN))

        step = lambda i, s: T(f"{i}. {s}", 34, INK).to_edge(UP, buff=0.3)
        st = step(1, "0 – 8 kHz → mel")
        self.play(FadeIn(st), ShowCreation(mel_ax), ShowCreation(hz_ax), FadeIn(ml), FadeIn(hl), FadeIn(ends))
        self.beat()
        md = VGroup(*[Dot(mel_ax.n2p(m), radius=0.08, fill_color=ACCENT) for m in mel_pts])
        self.play(Transform(st, step(2, f"{nm + 2} points, evenly spaced in mel")), LaggedStartMap(FadeIn, md, lag_ratio=0.1))
        self.beat()
        hd = VGroup(*[Dot(hz_ax.n2p(f), radius=0.08, fill_color=GOLD) for f in hz_pts])
        lines = VGroup(*[Line(mel_ax.n2p(m), hz_ax.n2p(f), stroke_color=RULE, stroke_width=1.5) for m, f in zip(mel_pts, hz_pts)])
        self.play(Transform(st, step(3, "back to Hz: bunched low")), ShowCreation(lines), TransformFromCopy(md, hd), run_time=1.5)
        self.beat()
        bins = VGroup(*[Line(hz_ax.n2p(k * 40) + 0.08 * DOWN, hz_ax.n2p(k * 40) + 0.08 * UP, stroke_color=DIM, stroke_width=1)
                        for k in range(0, 201)])
        bt = tag("201 bins · 40 Hz apart", DIM, 24).next_to(hz_ax, DOWN, buff=0.55)
        self.play(Transform(st, step(4, "read triangles at 201 bins")), FadeIn(bins), FadeIn(bt))
        self.beat()

        fb = plain_axes((0, 8000, 1000), (0, 1, 0.5), 12, 2.2).next_to(hz_ax, DOWN, buff=1.0)
        cols = [ACCENT, PINK, GOLD, BLUE]
        tris = VGroup(*[Polygon(fb.c2p(hz_pts[i], 0), fb.c2p(hz_pts[i + 1], 1), fb.c2p(hz_pts[i + 2], 0),
                                stroke_color=cols[i % 4], stroke_width=3, fill_color=cols[i % 4], fill_opacity=0.15)
                        for i in range(nm)])
        self.play(Transform(st, step(5, "a triangle per band")), FadeOut(bt),
                  LaggedStartMap(ShowCreation, tris, lag_ratio=0.15), run_time=2)
        self.say("narrow low, wide high")
        self.beat()

        # Whisper's 80 (Slaney-normalised: equal area → shorter as they widen)
        W, pts = self.A["W"], self.A["pts"]
        fb2 = plain_axes((0, 8000, 1000), (0, float(W.max()) * 1.05, 0.01), 12, 4.6).shift(0.3 * DOWN)
        f = dsp.bin_hz(np.arange(dsp.N_FREQ))
        tri80 = VGroup(*[polyline(np.array([fb2.c2p(x, v) for x, v in zip(f, W[i])]), cols[i % 4], 2) for i in range(80)])
        t80 = tag("mel_filters  W  [80, 201]", GOLD, 32).to_edge(UP, buff=0.35)
        self.fade_all(keep=(self.caption,))
        self.play(FadeIn(t80), ShowCreation(fb2), LaggedStartMap(ShowCreation, tri80, lag_ratio=0.02), run_time=2.5)
        xs = VGroup(*[T(f"{v // 1000} k", 24, DIM).next_to(fb2.c2p(v, 0), DOWN, buff=0.12) for v in range(0, 8001, 2000)])
        self.play(FadeIn(xs))
        self.say("Whisper: 80 bands, equal area")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ W @ P
    def matmul(self):
        W, P = self.A["W"], self.A["power"]
        Wi = heat(np.sqrt(W / W.max()), "l3_W", 2.6, 1.8, vmin=0, vmax=1)
        Pd = db(P[:, :C1])
        Pi = heat(Pd, "l3_P", 3.8, 3.4, vmin=Pd.max() - 80, vmax=Pd.max())
        Md = db(self.A["mel"][:, :C1])
        Mi = heat(Md, "l3_M", 3.8, 1.8, vmin=Md.max() - 80, vmax=Md.max())
        at, eq = M(r"@", 60), M(r"=", 60)
        row = Group(Wi, at, Pi, eq, Mi).arrange(RIGHT, buff=0.45).shift(0.2 * UP)
        boxes = VGroup(*[SurroundingRectangle(m, buff=0, stroke_color=RULE, stroke_width=1.5) for m in (Wi, Pi, Mi)])
        tags = VGroup(tag("W [80, 201]", PINK, 26).next_to(Wi, DOWN, buff=0.2),
                      tag("P [201, 3000]", ACCENT, 26).next_to(Pi, DOWN, buff=0.2),
                      tag("mel [80, 3000]", GOLD, 26).next_to(Mi, DOWN, buff=0.2))
        self.play(FadeIn(Wi), FadeIn(Pi), ShowCreation(boxes[:2]), FadeIn(tags[:2]), FadeIn(at))
        self.say("each band = weighted sum of bins")
        self.beat()
        hl1 = SurroundingRectangle(tags[0][-5:-1], stroke_color=GOLD, buff=0.04)
        hl2 = SurroundingRectangle(tags[1][2:5], stroke_color=GOLD, buff=0.04)
        self.play(ShowCreation(hl1), ShowCreation(hl2))
        self.play(FadeIn(eq), FadeIn(Mi), ShowCreation(boxes[2]), FadeIn(tags[2]))
        self.say("201 contracted → 80 rows", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ log10, max − 8, (x + 4) / 4
    def log_steps(self):
        mel, log, floor, scaled = (self.A[k][:, :C1] for k in ("mel", "log", "floor", "scaled"))
        img_kw = dict(width=11, height=3.2)

        def show(arr, name, vmin, vmax):
            im = heat(arr, name, img_kw["width"], img_kw["height"], vmin=vmin, vmax=vmax).shift(0.5 * UP)
            return Group(im, SurroundingRectangle(im, buff=0, stroke_color=RULE, stroke_width=1.5))

        lin = show(mel, "l3_lin", 0, mel.max())
        f0 = tag("mel  [80, 3000]", GOLD, 30).next_to(lin, UP, buff=0.15)
        self.play(FadeIn(lin), FadeIn(f0))
        self.say("linear power: almost all dark")
        self.beat()

        lg = show(log, "l3_log", -10, log.max())
        f1 = M(r"\log_{10}\big(\max(\text{mel},\ 10^{-10})\big)", 44).next_to(lg, UP, buff=0.15)
        r1 = tag(f"range  [{log.min():.1f}, {log.max():.1f}]  · −10 = silent padding", DIM, 26).next_to(lg, DOWN, buff=0.2)
        self.play(FadeOut(lin), FadeIn(lg), FadeTransform(f0, f1), FadeIn(r1))
        self.say("log: loud and quiet both visible")
        self.beat()

        mx = self.A["max"]
        fl = show(floor, "l3_floor", mx - 8, mx)
        f2 = M(r"\max\big(x,\ x_{\max} - 8\big)", 44).next_to(fl, UP, buff=0.15)
        r2 = tag(f"range  [{mx - 8:.1f}, {mx:.1f}]  = 8 decades = 80 dB", DIM, 26).next_to(fl, DOWN, buff=0.2)
        self.play(FadeOut(lg), FadeIn(fl), FadeTransform(f1, f2), FadeTransform(r1, r2))
        self.say("floor: keep the top 80 dB")
        self.beat()

        sc = show(scaled, "l3_scaled", scaled.min(), scaled.max())
        f3 = M(r"\frac{x + 4}{4}", 50, ACCENT).next_to(sc, UP, buff=0.15)
        r3 = tag(f"range  [{scaled.min():.2f}, {scaled.max():.2f}]", ACCENT, 28).next_to(sc, DOWN, buff=0.2)
        self.play(FadeOut(fl), FadeIn(sc), FadeTransform(f2, f3), FadeTransform(r2, r3))
        self.say("shift and scale: span of 2")
        self.beat()
        fin = tag("log-mel  [80, 3000]  →  encoder", GOLD, 34).move_to(f3)
        self.play(FadeTransform(f3, fin))
        self.say("100 frames per second · 30 s")
        self.beat()
        self.clear_all()


split_sections(L3_LogMel, globals())
