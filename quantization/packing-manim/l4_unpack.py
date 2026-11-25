"""Lesson 4 · unpack_weights (course order): shift each slot down inside the loop, mask once at the end."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from bits import BYTE_X, UNPACK_SRC, Byte, CodePanel, TensorRow, VarBox, mono
import toy

PACKED = toy.pack_weights(toy.EX1, 2)        # [177]
TRACE = toy.unpack_trace(PACKED)
ROW_Y = [0.75, -0.2, -1.15, -2.1]             # unpacked_tensor[0..3]


def shift_caption(s):
    """Last caption of an unpack step (also restored at the start of the next section)."""
    if s.shift:
        return f"{s.before} >> {s.shift} = {s.shifted}: w[{s.idx}] now in bits 1–0", VALUE_COLOR[s.masked]
    return "j = 0: no shift — w[0] already in bits 1–0", VALUE_COLOR[s.masked]


class UnpackStage:
    """Code left; the packed byte on top right; the four output bytes below it."""

    def build(self, k, masked=False):
        last = TRACE[k - 1] if k > 0 else None
        self.code = CodePanel(UNPACK_SRC).move_to([-3.55, 0.3, 0])
        self.src = Byte(PACKED[0], index=True).place(BYTE_X, 2.75)
        self.src_name = mono("uint8tensor[0]", 18, DIM).next_to(self.src.frame, LEFT, buff=0.15)
        v = dict(i=last.i, j=last.j, unpacked_idx=last.idx, **{"bits*j": last.shift}) if last else {}
        self.vars = VarBox(["i", "j", "unpacked_idx", "bits*j"], size=22, cols=2, **v)
        self.vars.move_to([BYTE_X + 2.26, 1.72, 0])
        self.out, self.reads = [], []
        for j in range(4):
            if j < k:
                n = TRACE[j].masked if masked else TRACE[j].shifted
                occ = {0} if masked else toy.unpack_occupancy(j)
            else:
                n, occ = 0, set()
            b = Byte(n, occupied=occ, index=False, label=f"unpacked[{j}]", label_size=24).place(BYTE_X, ROW_Y[j])
            self.out.append(b)
            self.reads.append(self.read(b, n, j < k))
        return self

    def read(self, b, n, show=True):
        col = (VALUE_COLOR[n] if n < 4 else SOFT) if show else RULE
        return mono(f"{n}" if show else "", 26, col).next_to(b.frame, RIGHT, buff=0.18)

    def mobjects(self):
        return [self.code, self.src, self.src_name, self.vars, *self.out, *[r for r in self.reads]]

    def restore(self, scene, k, masked=False, animate=False):
        """Rebuild the picture after step k-1. animate=False: appear exactly as the previous
        section left it (bar on the >> line, same caption), so the sections join seamlessly."""
        self.build(k, masked)
        if k > 0:
            self.code.set_highlight("uint8tensor[i] >>", token=">> (bits * j)")
        if animate:
            scene.play(*[FadeIn(m) for m in self.mobjects()], run_time=0.8)
            return
        scene.add(*self.mobjects())
        if k > 0:
            text, col = shift_caption(TRACE[k - 1])
            scene.caption = T(text, 44, col).to_edge(DOWN, buff=0.35)
            scene.add(scene.caption)

    def new_read(self, scene, j, n):
        new = self.read(self.out[j], n)
        scene.play(Transform(self.reads[j], new), run_time=0.4)

    # -------------------------------------------------------- one inner-loop step: copy → shift right
    def step(self, scene, s):
        b = self.out[s.j]
        scene.play(self.code.highlight("uint8tensor[i] >>", token="uint8tensor[i]"),
                   self.vars.update_anim(i=s.i, j=s.j, unpacked_idx=s.idx, **{"bits*j": s.shift}))
        copy = self.src.tiles.copy()
        scene.play(copy.animate.move_to(VGroup(*b.tiles)), run_time=0.9)
        scene.play(b.set_value_anim(s.before, set(range(4)), run_time=0.4), FadeOut(copy), run_time=0.4)
        self.new_read(scene, s.j, s.before)
        scene.say(f"take uint8tensor[{s.i}] = {s.before}")
        scene.beat()

        scene.play(self.code.highlight("uint8tensor[i] >>", token=">> (bits * j)"))
        if s.shift:
            scene.say(f"shift right by 2·{s.j} = {s.shift}", VALUE_COLOR[s.masked])
            b.play_shift(scene, s.shift, RIGHT)
            self.new_read(scene, s.j, s.shifted)
        scene.say(*shift_caption(s))
        scene.beat()


class L4_Unpack(Lesson):
    NUM, TITLE, SUB = 4, "Unpack", "177  →  [1, 0, 3, 2]"
    SECTIONS = ["title", "before_loop", "j0", "j1", "j2", "j3", "mask", "roundtrip"]
    TOPICS = {"before_loop": "four empty output bytes", "j0": "shift each slot to the bottom",
              "mask": "one mask at the end", "roundtrip": "pack → unpack"}

    def stage_at(self, k):
        if getattr(self, "stage", None) is None:
            self.stage = UnpackStage()
            self.stage.restore(self, k, animate=(k == 0))  # j0 follows a roadmap slide
        return self.stage

    def before_loop(self):
        st = UnpackStage().build(0)
        self.play(FadeIn(st.code), FadeIn(st.src), FadeIn(st.src_name))
        self.say("input: the packed byte 177")
        self.beat()
        self.play(st.code.highlight("num_values =", token="num_values"),
                  LaggedStart(*[FadeIn(b) for b in st.out], lag_ratio=0.15))
        self.say("1 · 8 // 2 = 4 output bytes, all 0")
        self.beat()
        self.play(st.code.highlight("mask =", token="mask = 2 ** bits - 1"), FadeIn(st.vars))
        self.say("mask = 2² − 1 = 3 = 0000 0011", ACCENT)
        self.beat()
        self.add(*st.reads)
        self.stage = st

    def j0(self):
        self.stage_at(0).step(self, TRACE[0])

    def j1(self):
        self.stage_at(1).step(self, TRACE[1])

    def j2(self):
        self.stage_at(2).step(self, TRACE[2])

    def j3(self):
        self.stage_at(3).step(self, TRACE[3])
        self.say("w[j] is at the bottom — junk above it", BAD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ unpacked_tensor &= mask
    def mask(self):
        st = UnpackStage()
        st.restore(self, 4, animate=True)  # follows a roadmap slide
        m = Byte(3, occupied={0}, index=True, label="mask", label_size=24, label_color=ACCENT)
        m.place(BYTE_X, 2.75)
        self.play(st.code.highlight("&= mask", token="unpacked_tensor &= mask"),
                  FadeOut(VGroup(st.src, st.src_name, st.vars)), FadeIn(m))
        self.say("after the loop: AND every byte with 3", ACCENT)
        self.beat()

        hi = [b.tiles[p][1] for b in st.out for p in range(6)]
        self.play(*[d.animate.set_color(BAD) for d in hi], run_time=0.5)
        self.play(*[b.set_value_anim(s.masked, {0}, run_time=0.9) for b, s in zip(st.out, TRACE)], run_time=0.9)
        self.play(*[Transform(st.reads[j], st.read(st.out[j], TRACE[j].masked)) for j in range(4)], run_time=0.5)
        self.say("bits 7–2 → 0 · only w[j] is left")
        self.beat()
        out = TensorRow(toy.unpack_weights(PACKED, 2), label="unpacked_tensor =").move_to([3.4, 2.55, 0])
        self.play(FadeOut(m), FadeIn(out))
        self.say("unpack_weights(177) = [1, 0, 3, 2]", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the round trip
    def roundtrip(self):
        a = TensorRow(toy.EX1, index=False)
        p = Byte(PACKED[0], label=None)
        c = TensorRow(toy.unpack_weights(PACKED, 2), index=False)
        row = VGroup(a, p, c).arrange(RIGHT, buff=1.3).move_to([0, 0.4, 0])
        if row.get_width() > 13.4:
            row.set_width(13.4)
        ar1 = arrow(a.get_right() + 0.15 * RIGHT, p.frame.get_left() + 0.15 * LEFT, ACCENT)
        ar2 = arrow(p.frame.get_right() + 0.15 * RIGHT, c.get_left() + 0.15 * LEFT, ACCENT)
        l1 = mono("pack", 24, ACCENT).next_to(ar1, UP, buff=0.12)
        l2 = mono("unpack", 24, ACCENT).next_to(ar2, UP, buff=0.12)
        self.play(FadeIn(a))
        self.play(GrowArrow(ar1), FadeIn(l1), FadeIn(p))
        self.play(GrowArrow(ar2), FadeIn(l2), FadeIn(c))
        n = VGroup(mono("pack:   << 2j, then |=", 28, SOFT), mono("unpack: >> 2j, then & 3", 28, SOFT))
        n.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0, -1.8, 0])
        self.play(FadeIn(n))
        self.say("same weights back, 4× less memory", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L4_Unpack, globals())
