"""Pictures shared by several lessons: the MTP figure, the toy 2-D plane, probability bars."""
from style import *  # noqa: F401,F403
import toy


# ---------------------------------------------------------------- the MTP figure (K = 2, t = 3)
def node(tex, color, r=0.42, size=34):
    """A distribution / hidden state drawn as a circle with its symbol."""
    c = Circle(radius=r, stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.12)
    lab = M(tex, size).set_color(color)
    lab.move_to(c)
    return VGroup(c, lab)


def dotted_up(a, b, color=DIM):
    return dashed_arrow(a, b, color, width=2.2, size=0.14, dash=0.06)


def mtp_figure(center=ORIGIN, scale=1.0):
    """The paper's picture: tokens → shared trunk → NTP row; MTP heads stacked diagonally at t = 3.

    Returns a dict of named parts so lessons can animate them one by one."""
    xs = [-3.2, -1.6, 0.0, 1.6, 3.2]
    toks = VGroup(*[VGroup(RoundedRectangle(width=0.8, height=0.8, corner_radius=0.1, stroke_color=SOFT, stroke_width=2),
                           M(f"x_{i + 1}", 34).set_color(SOFT)) for i in range(5)])
    for i, t in enumerate(toks):
        t.move_to([xs[i], -2.6, 0])
        t[1].move_to(t[0])
    trunk = RoundedRectangle(width=7.6, height=0.8, corner_radius=0.15, stroke_color=SOFT, stroke_width=2,
                             fill_color=PANEL, fill_opacity=1).move_to([0, -1.35, 0])
    trunk_l = T("Shared Trunk", 30, INK).move_to(trunk)
    tok_arrows = VGroup(*[dotted_up(t.get_top(), [t.get_x(), trunk.get_bottom()[1], 0]) for t in toks])

    ntp = {i: node(rf"p^{{(0)}}_{i}", C_NTP) for i in (3, 4, 5)}
    for i, n in ntp.items():
        n.move_to([xs[i - 1], 0.0, 0])
    up_arrows = VGroup(*[dotted_up([xs[i - 1], trunk.get_top()[1], 0], ntp[i].get_bottom()) for i in (3, 4, 5)])
    mtp = {1: node(r"p^{(1)}_3", C_MTP), 2: node(r"p^{(2)}_3", C_MTP)}
    mtp[1].move_to([xs[3], 1.45, 0])
    mtp[2].move_to([xs[4], 2.9, 0])
    band = Polygon(*[np.array(p) for p in [(xs[2] - 0.6, -0.45, 0), (xs[2] + 0.45, -0.6, 0),
                                            (xs[4] + 0.6, 2.75, 0), (xs[4] - 0.45, 3.45, 0)]],
                   stroke_width=0, fill_color=SOFT, fill_opacity=0.10)
    head_l = VGroup(T("NTP head", 24, C_NTP).next_to(ntp[3], LEFT, buff=0.25),
                    T("MTP head 1", 24, C_MTP).next_to(mtp[1], LEFT, buff=0.25),
                    T("MTP head 2", 24, C_MTP).next_to(mtp[2], LEFT, buff=0.25))
    align = VGroup(dashed_arrow(ntp[4].get_top(), mtp[1].get_bottom(), C_NTP, 3, both=True),
                   dashed_arrow(ntp[5].get_top(), mtp[2].get_bottom(), C_NTP, 3, both=True))
    parts = dict(toks=toks, tok_arrows=tok_arrows, trunk=VGroup(trunk, trunk_l), up=up_arrows,
                 ntp=VGroup(*ntp.values()), ntp3=ntp[3], ntp4=ntp[4], ntp5=ntp[5],
                 mtp1=mtp[1], mtp2=mtp[2], band=band, heads=head_l, align=align)
    allg = VGroup(band, toks, tok_arrows, parts["trunk"], up_arrows, parts["ntp"], mtp[1], mtp[2], head_l)
    allg.scale(scale, about_point=ORIGIN).shift(center)
    align.scale(scale, about_point=ORIGIN).shift(center)
    parts["all"] = allg
    return parts


# ---------------------------------------------------------------- toy 2-D plane
class Plane(VGroup):
    """A faint grid; p(v) maps a 2-D toy vector to the screen."""

    def __init__(self, center=ORIGIN, unit=1.0, lim=2.4, xr=None, yr=None):
        """center = where the toy origin (0, 0) lands on screen; xr/yr crop the grid (for zoomed views)."""
        super().__init__()
        self.o, self.u = np.array(center, float), unit
        xr, yr = xr or (-lim, lim), yr or (-lim, lim)
        for i in range(int(np.ceil(xr[0])), int(np.floor(xr[1])) + 1):
            self.add(Line(self.p((i, yr[0])), self.p((i, yr[1])), stroke_color=SOFT if i == 0 else RULE,
                          stroke_width=1.6 if i == 0 else 1, stroke_opacity=0.7))
        for i in range(int(np.ceil(yr[0])), int(np.floor(yr[1])) + 1):
            self.add(Line(self.p((xr[0], i)), self.p((xr[1], i)), stroke_color=SOFT if i == 0 else RULE,
                          stroke_width=1.6 if i == 0 else 1, stroke_opacity=0.7))

    def p(self, v):
        return self.o + self.u * np.array([v[0], v[1], 0.0])

    def arrow(self, a, b, color, thick=5.0, buff=0.0):
        return arrow(self.p(a), self.p(b), color, thick=thick, buff=buff, tip=0.25)

    def dot(self, v, color, r=0.09):
        return Dot(self.p(v), radius=r, fill_color=color)


def token_rows(plane, size=26, highlight=None, only=None):
    """W's four rows drawn as faint arrows from the origin, one per token."""
    g = VGroup()
    for i, (w, name) in enumerate(zip(toy.W, toy.VOCAB)):
        if only is not None and i not in only:
            continue
        col = C_X if i == highlight else C_W
        a = arrow(plane.p((0, 0)), plane.p(w), col, thick=3.0, buff=0, tip=0.15).set_opacity(0.85)
        lab = T(name, size, col).move_to(plane.p(w * 1.17))
        g.add(VGroup(a, lab))
    return g


# ---------------------------------------------------------------- probability bars
def bars(values, color, h=1.8, w=0.5, gap=0.22, labels=True, size=22, signed=False, colors=None):
    """Bar chart over the 4 toy tokens; values printed on the bars.  signed: baseline in the middle."""
    g, rects, nums, names = VGroup(), VGroup(), VGroup(), VGroup()
    scale = h / 2 if signed else h
    for i, v in enumerate(values):
        c = colors[i] if colors else color
        hh = max(abs(v) * scale, 0.02)
        r = Rectangle(width=w, height=hh, stroke_width=0, fill_color=c, fill_opacity=0.85)
        x = i * (w + gap)
        r.move_to([x, (hh / 2) * (1 if v >= 0 else -1), 0])
        rects.add(r)
        n = tag(toy.f2(v), c, size)
        n.next_to(r, UP if v >= 0 else DOWN, buff=0.06)
        nums.add(n)
        if labels:
            names.add(T(toy.VOCAB[i], size, SOFT).move_to([x, -0.25 if not signed else -h / 2 - 0.25, 0]))
    base = Line([-w / 2 - 0.1, 0, 0], [3 * (w + gap) + w / 2 + 0.1, 0, 0], stroke_color=RULE, stroke_width=2)
    g.add(base, rects, nums, names)
    g.rects, g.nums, g.names = rects, nums, names
    return g


def legend(items, size=24):
    """Small colour key: [(tex, color, words)]."""
    rows = VGroup()
    for tex, col, words in items:
        rows.add(VGroup(M(tex, size + 6).set_color(col), T(words, size, col)).arrange(RIGHT, buff=0.18))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    box = SurroundingRectangle(rows, buff=0.18, stroke_color=RULE, stroke_width=1.5, fill_color=PANEL, fill_opacity=0.9)
    return VGroup(box, rows)
