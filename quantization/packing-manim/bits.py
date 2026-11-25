"""Reusable pieces: a byte of 8 bit-cells, the code panel, the loop variables, a tensor row.

Byte layout: bit 7 (MSB) on the left, bit 0 on the right; cells are grouped in 2-bit
slots (small gap between slots). Slot k = bits (2k+1, 2k), so slot 0 is the RIGHTMOST
pair. An occupied slot is filled with its value's colour (style.VALUE_COLOR).
"""
import re

from style import *  # noqa: F401,F403
import toy

CELL, GAP = 0.52, 0.12
MONO = "Menlo"

# notebook code, 2-space indent and the two longest lines wrapped so it fits half the frame
PACK_SRC = """def pack_weights(uint8tensor, bits):
  if uint8tensor.shape[0] * bits % 8 != 0:
    raise ValueError("not a multiple of 8/bits")
  num_values = uint8tensor.shape[0] * bits // 8
  num_steps = 8 // bits
  unpacked_idx = 0
  packed_tensor = torch.zeros((num_values),
                              dtype=torch.uint8)
  for i in range(num_values):
    for j in range(num_steps):
      packed_tensor[i] |= \\
        uint8tensor[unpacked_idx] << (bits * j)
      unpacked_idx += 1
  return packed_tensor"""

UNPACK_SRC = """def unpack_weights(uint8tensor, bits):
  num_values = uint8tensor.shape[0] * 8 // bits
  num_steps = 8 // bits
  unpacked_tensor = torch.zeros((num_values),
                    dtype=torch.uint8)
  unpacked_idx = 0
  mask = 2 ** bits - 1
  for i in range(uint8tensor.shape[0]):
    for j in range(num_steps):
      unpacked_tensor[unpacked_idx] |= \\
        uint8tensor[i] >> (bits * j)
      unpacked_idx += 1
  unpacked_tensor &= mask
  return unpacked_tensor"""

KEYWORDS = ["def", "for", "in", "if", "raise", "return", "range"]


def mono(s, size=28, color=INK):
    return Text(s, font=MONO, font_size=size).set_color(color)


def _x(p):
    """Offset of cell p (0 = leftmost = bit 7) from cell 0; works for p outside 0..7 too."""
    return p * CELL + (p // 2) * GAP


# ---------------------------------------------------------------- one byte
class Byte(VGroup):
    def __init__(self, n=0, occupied=None, label=None, index=True, label_color=SOFT, digit_size=30, label_size=28):
        super().__init__()
        self.n = n
        self.occ = set(range(4) if occupied is None else occupied)
        self.digit_size = digit_size
        self.frame = VGroup(*[Square(CELL, stroke_color=RULE, stroke_width=2).move_to(RIGHT * _x(p))
                              for p in range(8)])
        self.index = VGroup(*[mono(str(7 - p), 16, DIM).next_to(self.frame[p], UP, buff=0.08)
                              for p in range(8)]) if index else VGroup()
        self.tiles = VGroup()
        self.add(self.frame, self.index)
        self.tiles = self.make_tiles(n, self.occ)
        self.add(self.tiles)
        self.label = T(label, label_size, label_color).next_to(self.frame, LEFT, buff=0.3) if label else VGroup()
        self.add(self.label)

    # -------------------------------------------------------- geometry
    def place(self, frame_left, y):
        """Put the 8 cells at a fixed left edge, so stacked bytes line up bit for bit."""
        self.shift(np.array([frame_left - self.frame.get_left()[0], y - self.frame.get_y(), 0]))
        return self

    def pos(self, p):
        return self.frame[0].get_center() + RIGHT * _x(p)

    def slot_cells(self, k):
        """Frame cells of slot k (slot 0 = rightmost pair)."""
        return VGroup(self.frame[7 - 2 * k - 1], self.frame[7 - 2 * k])

    def slot_brace(self, k, text, color=None, direction=DOWN, size=26):
        color = color or VALUE_COLOR[toy.slots(self.n)[k]]
        b = Brace(self.slot_cells(k), direction, buff=0.08).set_color(color)
        t = T(text, size, color).next_to(b, direction, buff=0.08)
        return VGroup(b, t)

    def dim_anim(self, o=0.25):
        """Fade the byte back without filling its empty cells."""
        tiles = [a for sq, d in self.tiles for a in (sq.animate.set_fill(opacity=float(sq.get_fill_opacity()) * o),
                                                     d.animate.set_opacity(o))]
        return AnimationGroup(*tiles, self.frame.animate.set_stroke(opacity=o),
                              self.index.animate.set_opacity(o), self.label.animate.set_opacity(o))

    def digits(self):
        return VGroup(*[t[1] for t in self.tiles])

    # -------------------------------------------------------- state
    def make_tiles(self, n, occ):
        bits, sl = toy.bits8(n), toy.slots(n)
        tiles = VGroup()
        for p in range(8):
            k = (7 - p) // 2
            on = k in occ
            col = VALUE_COLOR[sl[k]]
            sq = Square(CELL * 0.86, stroke_width=0, fill_color=col, fill_opacity=0.38 if on else 0)
            dcol = (INK if bits[p] else SOFT) if on else RULE
            d = mono(str(bits[p]), self.digit_size, dcol)
            tiles.add(VGroup(sq, d).move_to(self.pos(p)))
        return tiles

    def set_value_anim(self, n, occupied=None, run_time=0.8):
        occ = set(range(4) if occupied is None else occupied)
        new = self.make_tiles(n, occ)
        self.n, self.occ = n, occ
        return AnimationGroup(*[Transform(a, b) for a, b in zip(self.tiles, new)], run_time=run_time)

    def play_shift(self, scene, k, direction=LEFT, run_time=1.3):
        """Play `n << k` (direction=LEFT) or `n >> k` (RIGHT). Bits that fall off turn red."""
        left = direction is LEFT or np.allclose(direction, LEFT)
        if k == 0:
            return
        new_n = ((self.n << k) & 255) if left else (self.n >> k)
        ds = k // 2
        new_occ = {s + ds for s in self.occ if s + ds < 4} if left else {s - ds for s in self.occ if s - ds >= 0}
        lost = [p for p in range(8) if (p < k if left else p >= 8 - k)]
        if any(toy.bits8(self.n)[p] for p in lost):
            scene.play(*[self.tiles[p][1].animate.set_color(BAD) for p in lost], run_time=0.35)
        step = -1 if left else 1
        anims, incoming = [], VGroup()
        for p, tile in enumerate(self.tiles):
            d = self.pos(p + step * k) - self.pos(p)
            if p in lost:
                anims.append(tile.animate.shift(d).set_opacity(0))
            else:
                anims.append(tile.animate.shift(d))
        zero = self.make_tiles(0, set())
        for q in range(k):
            p = 8 - k + q if left else q            # where the new zero ends up
            t = zero[p].copy()  # FadeIn(shift=v) starts it at pos(p) − v by itself
            incoming.add(t)
            anims.append(FadeIn(t, shift=self.pos(p) - self.pos(p - step * k)))
        scene.play(*anims, run_time=run_time)
        scene.remove(incoming)      # stand-alone copies, safe to drop
        self.n, self.occ = new_n, new_occ
        # reset the SAME 8 tile objects in place: swapping in new objects would split the
        # byte's group in scene.mobjects (ManimGL restructures on remove) and hide it
        for tile, fresh in zip(self.tiles, self.make_tiles(new_n, new_occ)):
            tile.become(fresh)
        # render the clean state once, so a beat ending here shows it. A counted play, not
        # scene.wait(): manim-slides splits slides by play count, and wait() isn't counted
        scene.play(Animation(self.tiles), run_time=0.1)


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


# ---------------------------------------------------------------- a 1-D tensor of 2-bit values
class TensorRow(VGroup):
    """[ 1 0 3 2 ] as coloured chips with index labels; None = empty slot (filled later)."""

    def __init__(self, values, label=None, size=32, box=0.62, index=True):
        super().__init__()
        self.boxes, self.vals = VGroup(), VGroup()
        for v in values:
            col = VALUE_COLOR[v] if v is not None else RULE
            b = RoundedRectangle(width=box, height=box, corner_radius=0.08, stroke_color=col, stroke_width=2,
                                 fill_color=col, fill_opacity=0.18 if v is not None else 0)
            self.boxes.add(b)
        self.boxes.arrange(RIGHT, buff=0.12)
        for v, b in zip(values, self.boxes):
            self.vals.add(mono(str(v), size, VALUE_COLOR[v]).move_to(b) if v is not None else VectorizedPoint(b.get_center()))
        lb, rb = mono("[", size + 8, SOFT), mono("]", size + 8, SOFT)
        lb.next_to(self.boxes, LEFT, buff=0.1)
        rb.next_to(self.boxes, RIGHT, buff=0.1)
        self.brackets = VGroup(lb, rb)
        self.index = VGroup(*[mono(str(i), 16, DIM).next_to(b, DOWN, buff=0.08) for i, b in enumerate(self.boxes)]) \
            if index else VGroup()
        self.add(self.boxes, self.vals, self.brackets, self.index)
        self.label = T(label, 28, SOFT).next_to(self.brackets[0], LEFT, buff=0.25) if label else VGroup()
        self.add(self.label)
        self.pointer = Triangle(fill_color=ACCENT, fill_opacity=1, stroke_width=0).rotate(PI).scale(0.12)
        self.pointer.next_to(self.boxes[0], UP, buff=0.08)

    def pointer_anim(self, idx):
        return self.pointer.animate.next_to(self.boxes[idx], UP, buff=0.08)

    def set_anim(self, i, v):
        col = VALUE_COLOR[v]
        new = mono(str(v), 32, col).move_to(self.boxes[i])
        return AnimationGroup(Transform(self.vals[i], new),
                              self.boxes[i].animate.set_stroke(col).set_fill(col, opacity=0.18))


# ---------------------------------------------------------------- the whole pack_weights stage
def or_caption(s):
    """Last caption of a pack step (also restored at the start of the next section)."""
    return f"{toy.bstr(s.before)}  |  {toy.bstr(s.shifted)}  =  {toy.bstr(s.after)}"


PX = 3.65      # right-column centre
BYTE_X = 1.75  # left edge of every byte's cells on the stage


class PackStage:
    """Code left; tensor, loop variables, `value` scratch byte and packed byte(s) right.

    `restore(scene, k)` rebuilds the picture as it is after trace step k-1 (k=0: before the
    loop), so each lesson section can start exactly where the previous one ended."""

    def __init__(self, values):
        self.values = values
        self.trace = toy.pack_trace(values)
        self.n_bytes = len(self.trace) // 4

    def build(self, k):
        last = self.trace[k - 1] if k > 0 else None
        self.code = CodePanel(PACK_SRC).move_to([-3.55, 0.3, 0])
        box = 0.62 if len(self.values) <= 4 else 0.46
        self.tensor = TensorRow(self.values, size=30 if box > 0.55 else 26, box=box).move_to([PX + (0.3 if len(self.values) <= 4 else 0.65), 2.75, 0])
        self.tname = mono("uint8tensor", 20, DIM).next_to(self.tensor, LEFT, buff=0.2)
        self.tensor.pointer.next_to(self.tensor.boxes[last.idx if last else 0], UP, buff=0.08)
        v = dict(i=last.i, j=last.j, unpacked_idx=last.idx, **{"bits*j": last.shift}) if last else {}
        self.vars = VarBox(["i", "j", "unpacked_idx", "bits*j"], size=24, cols=2, **v).move_to([PX, 1.45, 0])
        self.value = Byte(last.shifted if last else 0, occupied={last.j} if last else set(), label="value")
        self.value.place(BYTE_X, 0.05)
        ys = [-1.55] if self.n_bytes == 1 else [-1.35, -2.55]
        self.packed = []
        for i in range(self.n_bytes):
            done = [s for s in self.trace[:k] if s.i == i]
            n = done[-1].after if done else 0
            occ = {s.j for s in done}
            b = Byte(n, occupied=occ, label=f"packed[{i}]", index=(i == 0))
            b.place(BYTE_X, ys[i])
            self.packed.append(b)
        return self

    def mobjects(self):
        return [self.code, self.tensor, self.tname, self.tensor.pointer, self.vars, self.value, *self.packed]

    def restore(self, scene, k, caption=None, animate=False):
        self.build(k)
        mobs = self.mobjects()
        if k > 0:  # every step ends with the bar and box on |=
            self.code.set_highlight("|=", token="|=")
        if animate:
            scene.play(*[FadeIn(m) for m in mobs], run_time=0.8)
        else:
            scene.add(*mobs)
        if caption:
            scene.caption = T(caption, 44, SOFT).to_edge(DOWN, buff=0.35)
            scene.add(scene.caption)

    # -------------------------------------------------------- one inner-loop step: load → shift → OR
    def iteration(self, scene, s, fast=False):
        col = VALUE_COLOR[s.value]
        byte = self.packed[s.i]
        rt = 0.45 if fast else 0.8
        # load
        scene.play(self.code.highlight("<< (bits", token="uint8tensor[unpacked_idx]"),
                   self.tensor.pointer_anim(s.idx),
                   self.vars.update_anim(i=s.i, j=s.j, unpacked_idx=s.idx, **{"bits*j": s.shift}),
                   self.value.set_value_anim(s.value, {0}, run_time=rt), run_time=rt)
        if not fast:
            scene.say(f"load uint8tensor[{s.idx}] = {s.value}", col)
            scene.beat()
        # shift
        if not fast:
            scene.play(self.code.highlight("<< (bits", token="<< (bits * j)"))
            scene.say(f"shift left by 2·{s.j} = {s.shift}" if s.shift else "j = 0: no shift", col)
        if s.shift:
            self.value.play_shift(scene, s.shift, run_time=0.6 if fast else 1.3)
        if not fast:
            scene.beat()
        # OR
        cells = VGroup(self.value.tiles[6 - 2 * s.j], self.value.tiles[7 - 2 * s.j]).copy()
        target = VGroup(byte.tiles[6 - 2 * s.j], byte.tiles[7 - 2 * s.j])
        scene.play(self.code.highlight("|=", token="|="), cells.animate.move_to(target), run_time=rt + 0.2)
        scene.play(byte.set_value_anim(s.after, byte.occ | {s.j}, run_time=rt), FadeOut(cells), run_time=rt)
        if not fast:
            scene.say(or_caption(s))
            scene.beat()
