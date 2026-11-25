"""Lesson 3 · eight weights → two bytes: the outer loop, and unpacked_idx that keeps counting."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from bits import PackStage, mono
import toy


def active(byte):
    return SurroundingRectangle(byte.frame, buff=0.1, stroke_color=ACCENT, stroke_width=3)


class L3_TwoBytes(Lesson):
    NUM, TITLE, SUB = 3, "Two bytes", "[1, 0, 3, 2, 3, 3, 3, 3]  →  [177, 255]"
    SECTIONS = ["title", "outer", "byte0_fast", "byte1", "result"]
    TOPICS = {"outer": "the outer loop i", "byte0_fast": "byte 0: a quick replay",
              "byte1": "byte 1 starts at index 4", "result": "two packed bytes"}

    def prepare(self):
        self.trace = toy.pack_trace(toy.EX2)

    def stage_at(self, k, i_active):
        st = PackStage(toy.EX2)
        st.restore(self, k, animate=True)
        self.box = active(st.packed[i_active])
        self.play(ShowCreation(self.box), run_time=0.4)
        return st

    def outer(self):
        st = PackStage(toy.EX2).build(0)
        self.play(*[FadeIn(m) for m in (st.code, st.tensor, st.tname)])
        self.play(st.code.highlight("num_values =", token="num_values"), FadeIn(st.packed[0]), FadeIn(st.packed[1]))
        self.say("8 · 2 // 8 = 2 packed bytes")
        self.beat()
        box = active(st.packed[0])
        self.play(st.code.highlight("for i", token="for i in range(num_values)"), ShowCreation(box),
                  FadeIn(st.vars), FadeIn(st.value), FadeIn(st.tensor.pointer))
        self.play(st.vars.update_anim(i=0))
        self.say("outer loop: one byte per i", ACCENT)
        self.beat()
        self.clear_all()

    def byte0_fast(self):
        st = self.stage_at(0, 0)
        self.say("i = 0: same four steps as before")
        for s in self.trace[:4]:
            st.iteration(self, s, fast=True)
        self.say("packed[0] = 1011 0001 = 177", ACCENT)
        self.beat()
        self.clear_all()

    def byte1(self):
        st = self.stage_at(4, 0)
        nxt = active(st.packed[1])
        self.play(st.code.highlight("for i", token="for i in range(num_values)"), st.vars.update_anim(i=1, unpacked_idx=4),
                  Transform(self.box, nxt), st.tensor.pointer_anim(4))
        self.say("i = 1 · unpacked_idx is now 4", ACCENT)
        self.beat()
        for s in self.trace[4:]:
            st.iteration(self, s)
        self.clear_all()

    def result(self):
        st = PackStage(toy.EX2).build(8)
        g = VGroup(st.tensor, *st.packed)
        self.add(g)
        self.play(st.tensor.animate.move_to([0, 2.4, 0]),
                  st.packed[0].animate.place(-2.2, 0.6), st.packed[1].animate.place(-2.2, -0.8))
        vals = VGroup(*[mono(f"= {b.n}", 34, ACCENT).next_to(b.frame, RIGHT, buff=0.4) for b in st.packed])
        self.play(FadeIn(vals))
        self.say("pack_weights(...) → [177, 255]", ACCENT)
        self.beat()
        a = mono("8 bytes → 2 bytes", 34, SOFT).move_to([0, -2.2, 0])
        self.play(FadeIn(a))
        self.say("still 4× smaller")
        self.beat()
        self.clear_all()


split_sections(L3_TwoBytes, globals())
