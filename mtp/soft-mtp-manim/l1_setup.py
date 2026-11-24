"""Lesson 1 · Multi-token prediction: heads, one shared W, position-aligned NTP distributions."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from fig import mtp_figure, Plane, token_rows, bars, node
import toy


class L1_Setup(Lesson):
    NUM, TITLE, SUB = 1, "Multi-token prediction", "extra heads · one shared W"
    SECTIONS = ["title", "colors", "ntp", "mtp", "aligned", "toy_model"]
    TOPICS = {"colors": "the colour code", "ntp": "next-token prediction", "mtp": "add K prediction heads",
              "aligned": "two heads, same target", "toy_model": "a 4-token, 2-D toy"}

    # ------------------------------------------------------------ colour code used in every lesson
    def colors(self):
        rows = [
            (r"h^{(k)}_t,\ p^{(k)}_t", C_MTP, "MTP head (the student)"),
            (r"h^{(0)}_t,\ q^{(k)}_t", C_NTP, "NTP head (the teacher)"),
            (r"x_{t+k+1}", C_X, "data target (one-hot)"),
            (r"W_\theta", C_W, "shared output head"),
            (r"g_{\text{hard}}", C_HARD, "what standard MTP sends"),
            (r"g_{\text{KL}}", C_KL, "the part we keep"),
            (r"g_{\text{res}}", C_RES, "the part we remove"),
            (r"\mathrm{sg}(\cdot)", C_SG, "stop-gradient (detach)"),
        ]
        syms = [M(tex, 44).set_color(col) for tex, col, _ in rows]
        words = [T(w, 34, col) for _, col, w in rows]
        sym_w = max(m.get_width() for m in syms)
        g = VGroup()
        for i, (sm, wd) in enumerate(zip(syms, words)):
            y = 2.3 - i * 0.68
            sm.move_to([-4.2 + sm.get_width() / 2, y, 0])
            wd.move_to([-4.2 + sym_w + 0.6 + wd.get_width() / 2, y, 0])
            g.add(VGroup(sm, wd))
        g.move_to(0.3 * DOWN)
        head = T("colour code", 36, DIM).next_to(g, UP, buff=0.35).align_to(g, LEFT)
        self.play(FadeIn(head))
        self.play(LaggedStartMap(FadeIn, g[:4], shift=0.15 * RIGHT, lag_ratio=0.15))
        self.say("who is who")
        self.beat()
        self.play(LaggedStartMap(FadeIn, g[4:], shift=0.15 * RIGHT, lag_ratio=0.15))
        self.say("gradients get their own colours")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ plain next-token prediction
    def ntp(self):
        f = mtp_figure(center=3.0 * LEFT + 0.35 * DOWN, scale=0.82)
        self.play(LaggedStartMap(FadeIn, f["toks"], lag_ratio=0.1), FadeIn(f["tok_arrows"]))
        self.play(FadeIn(f["trunk"]))
        self.play(FadeIn(f["up"]), LaggedStartMap(FadeIn, f["ntp"], scale=0.7, lag_ratio=0.15))
        self.say("one prediction per position")
        self.beat()

        e1 = M(r"p^{(0)}_t = \mathrm{softmax}\big(W_\theta\, h^{(0)}_t\big)", 46)
        s1 = VGroup(tag("W [V, d]", C_W, 26), tag("h [d]", C_NTP, 26), tag("p [V]", C_NTP, 26)).arrange(RIGHT, buff=0.5)
        e2 = M(r"\mathcal{L}_{\text{NTP}} = -\sum_{t} \log p^{(0)}_t\big(x_{t+1}\big)", 46)
        col = VGroup(e1, s1, e2).arrange(DOWN, buff=0.35).move_to([3.9, 0.9, 0])
        e2.shift(0.4 * DOWN)
        self.play(Write(e1))
        self.play(FadeIn(s1))
        self.say("hidden state → shared head → softmax")
        self.beat()

        tgt = M(r"\to x_4", 36, C_X).next_to(f["ntp3"], UP, buff=0.15)
        self.play(Write(e2), FadeIn(tgt, shift=0.1 * UP))
        self.say("position 3 predicts token 4", C_X)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ K extra heads, same W
    def mtp(self):
        f = mtp_figure(center=3.0 * LEFT + 0.35 * DOWN, scale=0.82)
        self.play(FadeIn(VGroup(f["toks"], f["tok_arrows"], f["trunk"], f["up"], f["ntp"])), run_time=0.6)
        self.play(FadeIn(f["band"]), FadeIn(f["mtp1"], shift=0.3 * UP), FadeIn(f["mtp2"], shift=0.3 * UP),
                  FadeIn(f["heads"]))
        t1 = M(r"\to x_5", 34, C_X).next_to(f["mtp1"], RIGHT, buff=0.15)
        t2 = M(r"\to x_6", 34, C_X).next_to(f["mtp2"], RIGHT, buff=0.15)
        t0 = M(r"\to x_4", 34, C_X).next_to(f["ntp3"], UP, buff=0.12).shift(0.35 * LEFT)
        self.play(FadeIn(t0), FadeIn(t1), FadeIn(t2))
        self.say("head k looks k+1 tokens ahead")
        self.beat()

        e1 = M(r"p^{(k)}_t = \mathrm{softmax}\big(W_\theta\, h^{(k)}_t\big)", 46)
        same = T("same W as the NTP head", 28, C_W)
        e2 = M(r"\mathcal{L}^{(k)}_{\text{MTP}} = -\sum_{t} \log p^{(k)}_t\big(x_{t+k+1}\big)", 44)
        e3 = M(r"\mathcal{L} = \mathcal{L}_{\text{NTP}} + \frac{\lambda}{K}\sum_{k=1}^{K}\mathcal{L}^{(k)}_{\text{MTP}}", 44)
        col = VGroup(e1, same, e2, e3).arrange(DOWN, buff=0.4).move_to([4.1, 0.4, 0])
        self.play(Write(e1))
        self.play(FadeIn(same, shift=0.1 * UP))
        self.say("every head shares one W", C_W)
        self.beat()
        self.play(Write(e2))
        self.play(Write(e3))
        self.say("one weighted total loss")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ position-aligned NTP distributions
    def aligned(self):
        f = mtp_figure(center=2.6 * LEFT + 0.2 * DOWN, scale=0.9)
        self.play(FadeIn(f["all"]))
        self.say("look along a column")
        self.beat()

        # column t+k = 4: MTP head 1 at t=3 and the NTP head at t=4 both predict x5
        hl1 = SurroundingRectangle(VGroup(f["mtp1"], f["ntp4"]), buff=0.15, stroke_color=C_X)
        lab1 = M(r"x_5", 38, C_X).next_to(hl1, UP, buff=0.12)
        self.play(GrowFromCenter(f["align"][0]))
        self.play(ShowCreation(hl1), FadeIn(lab1))
        self.say("same target x5, two predictors", C_X)
        self.beat()

        e = M(r"q^{(k)}_t := p^{(0)}_{t+k}", 54).move_to([4.2, 1.2, 0])
        d = VGroup(T("aligned NTP distribution", 28, C_NTP), M(r"\text{both predict } x_{t+k+1}", 36)).arrange(DOWN, buff=0.2)
        d.next_to(e, DOWN, buff=0.35)
        self.play(Write(e))
        self.play(FadeIn(d))
        self.beat()

        hl2 = SurroundingRectangle(VGroup(f["mtp2"], f["ntp5"]), buff=0.15, stroke_color=C_X)
        lab2 = M(r"x_6", 38, C_X).next_to(hl2, RIGHT, buff=0.2)
        self.play(GrowFromCenter(f["align"][1]), ShowCreation(hl2), FadeIn(lab2))
        self.say("k = 2: column 5, target x6")
        self.beat()

        why = VGroup(T("NTP sees more context", 30, C_NTP), T("→ a natural teacher", 30, C_NTP)).arrange(DOWN, buff=0.15)
        why.next_to(d, DOWN, buff=0.6)
        self.play(FadeIn(why, shift=0.1 * UP))
        self.say("q is the teacher, p the student")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ toy numbers used everywhere
    def toy_model(self):
        pl = Plane(center=[-3.6, -0.15, 0], unit=1.25, lim=2.6)
        rows = token_rows(pl, 28)
        head = VGroup(tag("V = 4 tokens", C_W, 30), tag("d = 2 dims", INK, 30)).arrange(RIGHT, buff=0.6)
        head.to_edge(UP, buff=0.3).shift(3.6 * LEFT)
        Wm = M(r"W_\theta = \begin{bmatrix} 2.0 & 0.4 \\ 0.2 & 2.0 \\ -1.9 & 0.5 \\ -0.3 & -1.9 \end{bmatrix}", 40)
        names = VGroup(*[T(n, 26, C_W) for n in toy.VOCAB])
        Wm.move_to([2.6, 0.5, 0])
        for i, n in enumerate(names):
            n.next_to(Wm, RIGHT, buff=0.25)
            n.set_y(Wm.get_top()[1] - (i + 0.5) * Wm.get_height() / 4)
        self.play(FadeIn(pl), FadeIn(head))
        self.play(LaggedStartMap(GrowArrow, VGroup(*[r[0] for r in rows]), lag_ratio=0.15),
                  LaggedStartMap(FadeIn, VGroup(*[r[1] for r in rows]), lag_ratio=0.15))
        self.play(Write(Wm), FadeIn(names))
        self.say("each row of W is a token")
        self.beat()

        self.play(FadeOut(Wm), FadeOut(names))
        hm = pl.arrow((0, 0), toy.H_MTP, C_MTP, thick=6)
        hn = pl.arrow((0, 0), toy.H_NTP, C_NTP, thick=6)
        lm = M(r"h^{(1)}_3", 34).set_color(C_MTP).next_to(pl.p(toy.H_MTP), DOWN + RIGHT, buff=0.08)
        ln = M(r"h^{(0)}_4", 34).set_color(C_NTP).next_to(pl.p(toy.H_NTP), LEFT, buff=0.35)
        vm = M(r"h^{(1)}_3 = " + toy.vec(toy.H_MTP), 36)
        vn = M(r"h^{(0)}_4 = " + toy.vec(toy.H_NTP), 36)
        lz = M(r"z = W_\theta h \quad\text{(dot with each row)}", 32)
        VGroup(vm, vn, lz).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([3.4, 2.6, 0])
        self.play(GrowArrow(hm), GrowArrow(hn), FadeIn(lm), FadeIn(ln))
        self.play(FadeIn(vm), FadeIn(vn), FadeIn(lz))
        self.say("two hidden states, same target")
        self.beat()

        bp = bars(toy.p, C_MTP, h=2.1, w=0.62, size=26, colors=[C_MTP, C_X, C_MTP, C_MTP]).move_to([1.55, -0.1, 0])
        bq = bars(toy.q, C_NTP, h=2.1, w=0.62, size=26, colors=[C_NTP, C_X, C_NTP, C_NTP]).move_to([4.95, -0.1, 0])
        bq.align_to(bp, DOWN)
        lp = M(r"p^{(1)}_3", 36).next_to(bp, DOWN, buff=0.2)
        lq = M(r"q^{(1)}_3 = p^{(0)}_4", 36).next_to(bq, DOWN, buff=0.2)
        self.play(FadeIn(bp), FadeIn(lp))
        self.play(FadeIn(bq), FadeIn(lq))
        cat = M(r"x_5 = \text{cat}", 40, C_X).next_to(VGroup(lp, lq), DOWN, buff=0.4)
        self.play(FadeIn(cat), rows[1][0].animate.set_fill(C_X), rows[1][1].animate.set_color(C_X))
        self.say("MTP says 'the', NTP says 'cat'")
        self.beat()
        self.clear_all()


split_sections(L1_Setup, globals())
