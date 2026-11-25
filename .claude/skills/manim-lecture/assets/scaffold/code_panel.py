"""Optional: source code on screen with a moving line highlight + token box, and a live
variable readout. Use only when the user chose "code on screen".

    code = CodePanel(SRC).move_to([-3.55, 0.3, 0])
    self.play(code.highlight("|=", token="|="))          # animate to the line containing "|="
    code.set_highlight("|=", token="|=")                  # same end state, instantly (section restore)
    vb = VarBox(["i", "j"], size=24, cols=2, i=0)         # restored values are teal, like updated ones
    self.play(vb.update_anim(i=1))

Keep source lines ≤ ~48 chars (2-space indent, wrap long lines) so the panel fits the left half.
Markers/tokens are looked up by substring, so they can't drift when the code text changes.
"""
import re

from style import *  # noqa: F401,F403

MONO = "Menlo"
KEYWORDS = ["def", "for", "in", "if", "else", "while", "raise", "return", "range", "import", "class"]


def mono(s, size=28, color=INK):
    return Text(s, font=MONO, font_size=size).set_color(color)


# ---------------------------------------------------------------- the code, with a moving highlight
class CodePanel(VGroup):
    def __init__(self, src, size=22, line_h=0.4, width=None):
        super().__init__()
        self.src_lines = src.split("\n")
        self.cw = Text("M" * 40, font=MONO, font_size=size).get_width() / 40
        self.line_h = line_h
        self.lines = VGroup()
        self.rows = []  # (row index, Text|None)
        for r, line in enumerate(self.src_lines):
            body = line.lstrip()
            if not body:
                continue
            t = Text(body, font=MONO, font_size=size).set_color(SOFT)
            cmap = {}
            for kw in KEYWORDS:
                if re.search(rf"\b{kw}\b", body):
                    cmap[re.compile(rf"\b{kw}\b")] = ACCENT
            if "#" in body:
                cmap[re.compile(r"#.*")] = DIM
            for op in ("|=", "&=", "<<", ">>"):
                if op in body:
                    cmap[op] = INK
            if cmap:
                t.set_color_by_text_to_color_map(cmap)
            indent = len(line) - len(body)
            t.move_to(self.cell(r, indent), aligned_edge=LEFT)
            self.lines.add(t)
        w = width or (max(len(l) for l in self.src_lines) * self.cw + 0.5)
        h = len(self.src_lines) * line_h + 0.4
        self.bg = RoundedRectangle(width=w, height=h, corner_radius=0.12, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=RULE, stroke_width=1.5)
        self.bg.move_to(self.cell(0, 0) + RIGHT * (w / 2 - 0.25) + DOWN * (h / 2 - 0.2 - line_h / 2))
        self.bar = Rectangle(width=w - 0.1, height=line_h * 0.98, stroke_width=0, fill_color=ACCENT,
                             fill_opacity=0).move_to([self.bg.get_center()[0], 0, 0])
        self.box = Rectangle(width=0.1, height=line_h, stroke_color=GOLD, stroke_width=0).move_to(self.bar)
        self.add(self.bg, self.bar, self.lines, self.box)

    def cell(self, row, col):
        """Left edge, vertical centre of character `col` on `row` (before any moving of the panel)."""
        return np.array([col * self.cw, -row * self.line_h, 0])

    def _origin(self):
        # where cell(0, 0) is now: bg was placed relative to it, so recover from bg
        w, h = self.bg.get_width(), self.bg.get_height()
        return self.bg.get_center() - RIGHT * (w / 2 - 0.25) - DOWN * (h / 2 - 0.2 - self.line_h / 2)

    def find(self, marker):
        for r, line in enumerate(self.src_lines):
            if marker in line:
                return r, line.index(marker)
        raise KeyError(marker)

    def set_highlight(self, marker, token=None, color=GOLD):
        """Same end state as highlight(), without animating (for rebuilding a section's start)."""
        anim = self.highlight(marker, token, color)
        for a in anim.animations:
            a.begin()
            a.interpolate(1)
            a.finish()
        return self

    def highlight(self, marker, token=None, color=GOLD, run_time=0.5):
        """Bar on the first line containing `marker`; box around `token` on that line (if given)."""
        r, _ = self.find(marker)
        o = self._origin()
        y = (o + self.cell(r, 0))[1]
        anims = [self.bar.animate.move_to([self.bg.get_center()[0], y, 0]).set_fill(opacity=0.13)]
        if token:
            c = self.src_lines[r].index(token)
            left = (o + self.cell(r, c))[0]
            rect = Rectangle(width=len(token) * self.cw + 0.12, height=self.line_h * 0.95,
                             stroke_color=color, stroke_width=3)
            rect.move_to([left + len(token) * self.cw / 2, y, 0])
            anims.append(Transform(self.box, rect))
        else:
            anims.append(self.box.animate.set_stroke(width=0))
        return AnimationGroup(*anims, run_time=run_time)


# ---------------------------------------------------------------- loop variables  i = 0   j = 2 ...
class VarBox(VGroup):
    def __init__(self, names, size=28, cols=None, **vals):
        super().__init__()
        self.names, self.size = names, size
        self.vals = {n: vals.get(n, "–") for n in names}
        self.items = VGroup(*[self._item(n, self.vals[n], ACCENT if n in vals else SOFT) for n in names])
        if cols and cols < len(names):
            self.items.arrange_in_grid(n_cols=cols, h_buff=0.6, v_buff=0.25, aligned_edge=LEFT)
        else:
            self.items.arrange(RIGHT, buff=0.55)
        self.add(self.items)

    def _item(self, name, v, col):
        return VGroup(mono(f"{name} =", self.size, DIM), mono(str(v), self.size, col)).arrange(RIGHT, buff=0.15)

    def update_anim(self, **vals):
        anims = []
        for k, (n, item) in enumerate(zip(self.names, self.items)):
            if n in vals and vals[n] != self.vals[n]:
                self.vals[n] = vals[n]
                new = mono(str(vals[n]), self.size, ACCENT).move_to(item[1], aligned_edge=LEFT)
                anims.append(Transform(item[1], new))
        return AnimationGroup(*anims, run_time=0.5) if anims else Wait(0.01)


