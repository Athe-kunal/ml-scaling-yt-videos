"""Shared look for every lesson: dark palette, big type, few words, one colour code."""
from pathlib import Path

import numpy as np
from PIL import Image
from manimlib import *  # noqa: F401,F403
from manim_slides import Slide

BG = "#0A0E14"
PANEL = "#161D29"
INK = "#F5F8FB"
SOFT = "#C7D1DD"
DIM = "#94A1B2"
RULE = "#334053"
ACCENT = "#6EF3DD"   # teal
PINK = "#FF8AC6"
GOLD = "#FFC94D"
BLUE = "#8DB4FF"
BAD = "#FF8A8A"
GREEN = "#8BE48B"
PURPLE = "#C9A0FF"

# ---------------------------------------------------------------- the colour code (same in every lesson)
C_MTP = PINK     # MTP head: h^(k), p^(k)
C_NTP = BLUE     # NTP head: h^(0), q = aligned p^(0)
C_X = GOLD       # data target x (one-hot)
C_W = PURPLE     # shared output head W
C_KL = ACCENT    # g_KL   keep
C_RES = BAD      # g_res  remove
C_HARD = INK     # g_hard = what standard MTP sends
C_SG = DIM       # stop-gradient / detach

# LaTeX pieces that always get their colour (longest first so nothing is split)
TEX_COLORS = {
    r"p^{(k)}_t": C_MTP, r"p^{(1)}_3": C_MTP, r"p^{(2)}_3": C_MTP, r"h^{(k)}_t": C_MTP, r"h^{(1)}_3": C_MTP,
    r"h^{(2)}_3": C_MTP, r"h^{(k)}_{t-k}": C_MTP, r"z^{(k)}_t": C_MTP,
    r"q^{(k)}_t": C_NTP, r"p^{(0)}_{t+k}": C_NTP, r"h^{(0)}_{t+k}": C_NTP, r"p^{(0)}_t": C_NTP,
    r"h^{(0)}_t": C_NTP, r"p^{(0)}_4": C_NTP, r"h^{(0)}_4": C_NTP, r"p^{(0)}_3": C_NTP, r"p^{(0)}_5": C_NTP,
    r"\tilde h_{t+k}": C_NTP, r"\tilde h_t": C_NTP, r"\tilde h_4": C_NTP,
    r"x_{t+k+1}": C_X, r"x_{t+1}": C_X, r"x_5": C_X,
    r"W_\theta": C_W,
    r"g_{\text{KL}}": C_KL, r"g_{\text{res}}": C_RES, r"g_{\text{hard}}": C_HARD,
    r"\mathrm{sg}": C_SG,
    r"\bar w_p": C_MTP, r"\bar w_q": C_NTP, r"\bar w_\pi": C_X, r"w_{\text{cat}}": C_X,
}

FONT = "Helvetica Neue"
SAFE_X, SAFE_Y = FRAME_X_RADIUS - 0.1, FRAME_Y_RADIUS - 0.05  # stay inside the rendered frame
CACHE = Path(__file__).parent / "media" / "img"
CACHE.mkdir(parents=True, exist_ok=True)


def T(s, size=34, color=INK, **kw):
    """Plain label (keep ≤ 6 words)."""
    return Text(s, font=FONT, font_size=size, **kw).set_color(color)  # Text ignores color= here


def M(s, size=40, color=INK, t2c=None, **kw):
    """LaTeX math; every symbol in TEX_COLORS gets its colour automatically."""
    cmap = {k: v for k, v in TEX_COLORS.items() if k in s}
    cmap.update(t2c or {})
    return Tex(s, font_size=size, color=color, t2c=cmap, **kw)


def tag(s, color=GOLD, size=26):
    """Tensor shape written on a diagram, e.g. [80, 3000]."""
    return Text(s, font="Menlo", font_size=size).set_color(color)


def polyline(points, color=ACCENT, width=2.5):
    v = VMobject(stroke_color=color, stroke_width=width)
    v.set_points_as_corners(points)
    return v


def c2p(axes, x, y):
    """Vectorised axes.c2p for linear axes / planes: origin + x·ux + y·uy.

    Calling axes.c2p once per point from Python was ~75 % of render time
    (curves rebuilt every frame by always_redraw)."""
    o = np.asarray(axes.c2p(0, 0))
    ux = np.asarray(axes.c2p(1, 0)) - o
    uy = np.asarray(axes.c2p(0, 1)) - o
    x = np.asarray(x, np.float64)[:, None]
    y = np.asarray(y, np.float64)[:, None]
    return o + x * ux + y * uy


def wave(axes, t, y, color=ACCENT, width=2.5, max_pts=2500):
    """Fast sampled curve on axes."""
    if len(t) > max_pts:
        step = len(t) // max_pts
        t, y = t[::step], y[::step]
    return polyline(c2p(axes, t, y), color, width)


def graph(axes, f, x_range, color=ACCENT, width=4, n=400):
    """Vectorised replacement for axes.get_graph (f takes a numpy array)."""
    x = np.linspace(*x_range[:2], n)
    return polyline(c2p(axes, x, f(x)), color, width)


def envelope(axes, t, y, bins=900, color=ACCENT):
    """Min/max envelope for long signals (so 64 000 samples look like a waveform)."""
    n = len(y) // bins
    yy = y[: n * bins].reshape(bins, n)
    tt = t[: n * bins].reshape(bins, n)[:, 0]
    xs = np.repeat(tt, 2)
    ys = np.stack([yy.min(1), yy.max(1)], axis=1).ravel()
    return polyline(c2p(axes, xs, ys), color, 1.6)


def stems(axes, t, y, color=GOLD, r=0.045):
    g = VGroup()
    for a, b in zip(t, y):
        g.add(Line(axes.c2p(a, 0), axes.c2p(a, b), stroke_color=color, stroke_width=2))
        g.add(Dot(axes.c2p(a, b), radius=r, fill_color=color))
    return g


# ---------------------------------------------------------------- heatmaps
_STOPS = np.array([  # dark → teal → gold → white (matches palette)
    [10, 14, 20], [22, 40, 70], [40, 110, 140], [110, 243, 221], [255, 201, 77], [255, 250, 235],
]) / 255.0


def colormap(x):
    x = np.clip(x, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(x.astype(int), len(_STOPS) - 2)
    f = (x - i)[..., None]
    return _STOPS[i] * (1 - f) + _STOPS[i + 1] * f


def heat(arr, name, width, height, vmin=None, vmax=None, flip=True, upscale=1):
    """2-D array → ImageMobject (row 0 at the bottom when flip, like a spectrogram)."""
    a = np.asarray(arr, np.float64)
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    rgb = colormap((a - vmin) / max(vmax - vmin, 1e-12))
    if flip:
        rgb = rgb[::-1]
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    p = CACHE / f"{name}.png"
    img.save(p)
    m = ImageMobject(str(p))
    m.set_width(width, stretch=True)
    m.set_height(height, stretch=True)
    return m


def framed(mob, color=RULE, buff=0.0):
    return VGroup(mob, SurroundingRectangle(mob, buff=buff, stroke_color=color, stroke_width=1.5))


def block(label, w=3.2, h=0.9, color=ACCENT, size=26, sub=None):
    box = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=color, stroke_width=2.5,
                           fill_color=color, fill_opacity=0.10)
    txt = T(label, size, INK)
    g = VGroup(box, txt)
    if sub:
        s = tag(sub, DIM, 20)
        VGroup(txt, s).arrange(DOWN, buff=0.08).move_to(box)
        g.add(s)
    else:
        txt.move_to(box)
    return g


def chip(text, color=PINK, size=20, h=0.5, min_w=0.6, fill=0.15):
    """A token chip: rounded box with monospace text (e.g. <|AUDIO|>, ' And')."""
    t = tag(text, INK, size)
    box = RoundedRectangle(width=max(min_w, t.get_width() + 0.25), height=h, corner_radius=0.08,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=fill)
    t.move_to(box)
    return VGroup(box, t)


def vec_rows(values, w=0.34, h=0.34, name="rows"):
    """A column of feature vectors drawn as small heat squares (one per row)."""
    v = np.asarray(values, np.float64)
    cols = colormap((v - v.min()) / max(v.max() - v.min(), 1e-9))
    return VGroup(*[Square(w, stroke_color=BG, stroke_width=1, fill_color=rgb_to_hex(c), fill_opacity=1) for c in cols])


class Lesson(Slide):
    """A lecture split into SECTIONS; `beat()` = one press of →.

    Each section can also render as its own scene (see `split_sections` / render.py),
    so editing one section re-renders only that section. To make section scenes join
    seamlessly, `beat()` and `clear_all()` are lazy: they run when the next animation
    starts. Inside a lesson nothing changes; at the end of a section the trailing pause
    and fade are simply dropped (no empty keypress), and the next section starts clean.
    """

    default_camera_config = dict(background_color=BG)
    skip_reversing = True  # reversed clips only serve the V key; ← replays the previous beat forward
    NUM, TITLE, SUB = 0, "", ""
    SECTIONS: list = []
    TOPICS: dict = {}          # section → short label for the roadmap slide
    ALL_SECTIONS = None        # set on per-section scenes so the roadmap still shows the whole lesson

    def setup(self):
        super().setup()
        self.caption = None
        self._pending_beat = False
        self._pending_clear = None

    def prepare(self):
        """Shared data every section needs (runs before any section)."""

    def construct(self):
        self.prepare()
        for name in self.SECTIONS:
            if name in self.TOPICS:
                self.roadmap(name)
            getattr(self, name)()

    def roadmap(self, current):
        """Where we are in the lesson: done topics dim, the next one highlighted."""
        order = [s for s in (self.ALL_SECTIONS or self.SECTIONS) if s in self.TOPICS]
        idx = order.index(current)
        head = T(f"{self.NUM} · {self.TITLE}", 34, DIM)
        rows = VGroup()
        for i, s in enumerate(order):
            col = ACCENT if i == idx else (RULE if i < idx else SOFT)
            num = tag(f"{i + 1}", col, 32)
            rows.add(VGroup(num, T(self.TOPICS[s], 42 if i == idx else 36, col)).arrange(RIGHT, buff=0.4))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        VGroup(head, rows).arrange(DOWN, buff=0.45).move_to(0.15 * DOWN)
        head.align_to(rows, LEFT)
        box = SurroundingRectangle(rows[idx], stroke_color=ACCENT, buff=0.14, fill_color=ACCENT, fill_opacity=0.08)
        self.play(FadeIn(head), LaggedStartMap(FadeIn, rows, shift=0.15 * RIGHT, lag_ratio=0.06), run_time=0.9)
        self.play(ShowCreation(box), run_time=0.4)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ lazy beat / clear
    def _flush(self):
        if self._pending_beat:
            self._pending_beat = False
            self.next_slide()
        if self._pending_clear is not None:
            keep, self._pending_clear = self._pending_clear, None
            self.fade_all(keep)

    def play(self, *args, **kwargs):
        self._flush()
        super().play(*args, **kwargs)

    def beat(self):
        self.check_safe_area()
        self._flush()
        self._pending_beat = True

    def clear_all(self, keep=()):
        """Fade everything out — when the next animation starts."""
        self._pending_clear = tuple(keep)

    def fade_all(self, keep=()):
        """Fade everything out now."""
        self._flush()
        mobs = [m for m in self.mobjects if m not in keep and m is not self.frame]
        for m in mobs:
            m.clear_updaters()  # freeze always_redraw / TracedPath before fading
        if mobs:
            super().play(*[FadeOut(m) for m in mobs], run_time=0.6)
        if self.caption not in keep:
            self.caption = None

    def fade_in_live(self, *mobs, **kw):
        """FadeIn for mobjects with updaters (always_redraw, live numbers): fade a frozen
        snapshot, then swap in the live one (a fade can't follow changing geometry)."""
        snaps = [m.copy().clear_updaters() for m in mobs]
        self.play(*[FadeIn(s) for s in snaps], **kw)
        self.remove(*snaps)
        self.add(*mobs)

    # ------------------------------------------------------------ building blocks
    def title(self):
        num = T(f"{self.NUM}", 120, ACCENT)
        t = T(self.TITLE, 64)
        s = T(self.SUB, 32, DIM)
        VGroup(num, t, s).arrange(DOWN, buff=0.35)
        self.play(FadeIn(num, scale=0.6), Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=0.2 * UP))
        self.beat()
        self.clear_all()

    def say(self, text, color=SOFT):
        """Swap the step caption at the bottom."""
        self._flush()
        new = T(text, 44, color).to_edge(DOWN, buff=0.35)
        if self.caption is None:
            self.play(FadeIn(new, shift=0.15 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self.caption, shift=0.15 * UP), FadeIn(new, shift=0.15 * UP), run_time=0.5)
        self.caption = new

    def check_safe_area(self):
        """Warn when anything on screen sticks out of the rendered frame (a clipped label).

        Fitting the frame to a screen of any shape is the player's job (letterbox), so the
        only rule here is that the frame itself shows everything."""
        self._beat_no = getattr(self, "_beat_no", 0) + 1
        for m in self.mobjects:
            if m is self.frame:
                continue
            try:
                if m.get_num_points() == 0 and not m.submobjects and not isinstance(m, ImageMobject):
                    continue
                l, r = m.get_left()[0], m.get_right()[0]
                b, t = m.get_bottom()[1], m.get_top()[1]
            except Exception:
                continue
            if l < -SAFE_X or r > SAFE_X or b < -SAFE_Y or t > SAFE_Y:
                cap = self.caption.text if self.caption is not None and hasattr(self.caption, "text") else ""
                what = getattr(m, "text", None) or getattr(m, "tex_string", None) or type(m).__name__
                print(f"SAFE-AREA {type(self).__name__} beat {self._beat_no} [{cap}] "
                      f"{what!s:.50} x[{l:.2f},{r:.2f}] y[{b:.2f},{t:.2f}]", flush=True)


def section_scene_name(lesson, i, name):
    return f"{lesson.__name__}_{i:02d}_{name}"


def split_sections(lesson, namespace):
    """Add one scene per section, e.g. L2_Fourier_02_probe, to the lesson's module."""
    for i, name in enumerate(lesson.SECTIONS):
        cls_name = section_scene_name(lesson, i, name)
        namespace[cls_name] = type(cls_name, (lesson,), {"SECTIONS": [name], "ALL_SECTIONS": lesson.SECTIONS, "__module__": lesson.__module__})


# ---------------------------------------------------------------- arrows (ManimGL Arrow is a filled shape)
def arrow(a, b, color=SOFT, thick=4.0, buff=0.05, tip=0.3):
    """Solid arrow a → b (Arrow ignores stroke_color; it needs fill_color)."""
    return Arrow(np.array(a, float), np.array(b, float), buff=buff, fill_color=color, thickness=thick,
                 max_tip_length_to_length_ratio=tip)


def tip_at(point, direction, color, size=0.18):
    d = np.array(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    t = ArrowTip(angle=np.arctan2(d[1], d[0]), width=size, length=size, fill_color=color)
    t.move_to(np.array(point, float) - d * size / 2)
    return t


def dashed_arrow(a, b, color=DIM, width=2.5, size=0.16, both=False, dash=0.07):
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    line = DashedLine(a + (d * size if both else 0), b - d * size, dash_length=dash,
                      stroke_color=color, stroke_width=width)
    g = VGroup(line, tip_at(b, d, color, size))
    if both:
        g.add(tip_at(a, -d, color, size))
    return g
