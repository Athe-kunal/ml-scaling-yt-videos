"""Lesson 2 · pack_weights on [1, 0, 3, 2], one operation per press: load → shift → OR."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from bits import PackStage, mono, or_caption
import toy



class L2_Pack(Lesson):
    NUM, TITLE, SUB = 2, "Pack one byte", "[1, 0, 3, 2]  →  177"
    SECTIONS = ["title", "before_loop", "iter0", "iter1", "iter2", "iter3", "reveal"]
    TOPICS = {"before_loop": "the loop variables", "iter0": "four steps: load, shift, OR", "reveal": "read the byte back"}

    def prepare(self):
        self.trace = toy.pack_trace(toy.EX1)

    def stage_at(self, k, animate=False):
        """Reuse the stage when sections run back to back; rebuild it in a section scene."""
        if getattr(self, "stage", None) is None:
            self.stage = PackStage(toy.EX1)
            cap = or_caption(self.trace[k - 1]) if (k > 0 and not animate) else None
            self.stage.restore(self, k, caption=cap, animate=animate)
        return self.stage

    # ------------------------------------------------------------ before the loop
    def before_loop(self):
        st = PackStage(toy.EX1).build(0)
        self.play(FadeIn(st.code), FadeIn(st.tensor), FadeIn(st.tname))
        self.say("the notebook's pack_weights")
        self.beat()

        nv = mono("num_values = 4 * 2 // 8 = 1", 24, SOFT)
        ns = mono("num_steps  = 8 // 2     = 4", 24, SOFT)
        calc = VGroup(nv, ns).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([st.vars.get_x(), 1.3, 0])
        self.play(st.code.highlight("num_values =", token="num_values"), FadeIn(nv))
        self.say("4 weights → 1 packed byte")
        self.beat()
        self.play(st.code.highlight("num_steps =", token="num_steps"), FadeIn(ns))
        self.say("4 slots of 2 bits per byte")
        self.beat()

        self.play(st.code.highlight("torch.zeros", token="torch.zeros"), FadeIn(st.packed[0]))
        self.say("start from 0000 0000")
        self.beat()

        self.play(st.code.highlight("for j", token="for j in range(num_steps)"), FadeOut(calc),
                  FadeIn(st.vars), FadeIn(st.value), FadeIn(st.tensor.pointer))
        self.say("each step fills one slot", ACCENT)
        self.beat()
        self.stage = st

    # ------------------------------------------------------------ the four inner-loop steps
    def iter0(self):
        st = self.stage_at(0, animate=True)
        st.iteration(self, self.trace[0])

    def iter1(self):
        self.stage_at(1).iteration(self, self.trace[1])

    def iter2(self):
        self.stage_at(2).iteration(self, self.trace[2])

    def iter3(self):
        self.stage_at(3).iteration(self, self.trace[3])
        self.clear_all()

    # ------------------------------------------------------------ the slots come out reversed
    def reveal(self):
        self.stage = None
        st = self.stage_at(4, animate=True)
        b = st.packed[0]
        self.play(FadeOut(st.code), FadeOut(st.vars), FadeOut(st.value), FadeOut(st.tensor.pointer), FadeOut(st.tname))
        self.play(st.tensor.animate.scale(1.5).move_to([0, 2.5, 0]), b.animate.scale(1.25).move_to([0.5, -1.0, 0]))
        self.say("where did each weight go?")
        self.beat()

        lines = VGroup()
        for k, v in enumerate(toy.EX1):
            a = st.tensor.index[k].get_bottom() + 0.12 * DOWN
            z = b.slot_cells(k).get_top() + 0.45 * UP
            lines.add(Line(a, z, stroke_color=VALUE_COLOR[v], stroke_width=4))
        self.play(LaggedStart(*[ShowCreation(l) for l in lines], lag_ratio=0.3), run_time=1.6)
        self.say("w[0] lands in the lowest bits")
        self.beat()

        br = VGroup(*[b.slot_brace(k, f"w[{k}]") for k in range(4)])
        self.play(FadeIn(br))
        self.say("read left→right: 2 3 0 1 — reversed", GOLD)
        self.beat()

        s = M(r"1011\,0001_2 = 128 + 32 + 16 + 1 = 177", 46).move_to([0, -2.5, 0])
        self.play(FadeOut(br), Write(s))
        self.say("one uint8 holds all four", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L2_Pack, globals())
