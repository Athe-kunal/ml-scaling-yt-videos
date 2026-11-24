"""Lesson 3 · The KL gradient-recovery trick: straight-through routing and the full backward pass."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from fig import Plane, node, mtp_figure
import toy

V = toy.vec


def graph():
    """The backward-pass picture for k = 1, t = 3 (MTP side left, aligned NTP side right)."""
    g = {}
    g["trunk"] = block("Shared Trunk", w=7.0, h=0.75, color=SOFT).move_to([-1.9, -3.05, 0])
    g["h30"] = node(r"h^{(0)}_3", C_NTP, r=0.42, size=30).move_to([-4.4, -1.85, 0])
    g["head"] = block("MTP head 1", w=2.2, h=0.62, color=C_MTP, size=22).move_to([-4.4, -0.65, 0])
    g["h31"] = node(r"h^{(1)}_3", C_MTP, r=0.46, size=32).move_to([-4.4, 0.7, 0])
    g["p31"] = node(r"p^{(1)}_3", C_MTP, r=0.46, size=32).move_to([-4.4, 2.55, 0])
    g["h40"] = node(r"h^{(0)}_4", C_NTP, r=0.42, size=30).move_to([0.6, -1.85, 0])
    plus = Circle(radius=0.42, stroke_color=C_NTP, stroke_width=3, fill_color=C_NTP, fill_opacity=0.12).move_to([0.6, 0.7, 0])
    g["plus"] = VGroup(plus, M(r"+", 40, C_NTP).move_to(plus))
    g["plus_l"] = M(r"\tilde h_4", 34).next_to(plus, RIGHT, buff=0.15)
    g["p40"] = node(r"p^{(0)}_4", C_NTP, r=0.46, size=32).move_to([0.6, 2.55, 0])
    g["ce1"] = M(r"\mathrm{CE}(x_5,\cdot)", 28, C_X).next_to(g["p31"], UP, buff=0.12)
    g["ce4"] = M(r"\mathrm{CE}(x_5,\cdot)", 28, C_X).next_to(g["p40"], UP, buff=0.12)
    f = VGroup()
    for a, b in [("h30", "head"), ("head", "h31"), ("h31", "p31"), ("h40", "plus"), ("plus", "p40")]:
        f.add(arrow(g[a].get_top(), g[b].get_bottom(), DIM, 2.5, buff=0.04))
    f.add(arrow([g["h30"].get_x(), g["trunk"].get_top()[1], 0], g["h30"].get_bottom(), DIM, 2.5, buff=0.04))
    f.add(arrow([g["h40"].get_x(), g["trunk"].get_top()[1], 0], g["h40"].get_bottom(), DIM, 2.5, buff=0.04))
    g["fwd"] = f
    # two edges h31 → plus: +sg(h) (dashed) and −h (solid)
    a, b = g["h31"].get_right() + 0.02 * RIGHT, plus.get_left() + 0.02 * LEFT
    g["e_sg"] = dashed_arrow(a + 0.22 * UP, b + 0.22 * UP, C_SG, 2.5, 0.14)
    g["e_neg"] = arrow(a + 0.22 * DOWN, b + 0.22 * DOWN, C_MTP, 2.5, buff=0.0, tip=0.08)
    g["l_sg"] = M(r"+\,\mathrm{sg}(h^{(1)}_3)", 28).next_to(g["e_sg"], UP, buff=0.08)
    g["l_neg"] = M(r"-\,h^{(1)}_3", 28).next_to(g["e_neg"], DOWN, buff=0.08)
    VGroup(*g.values()).scale(0.86, about_point=ORIGIN).shift(0.2 * UP + 0.4 * LEFT)
    return g


class L3_Trick(Lesson):
    NUM, TITLE, SUB = 3, "The KL gradient-recovery trick", "same forward, new backward"
    SECTIONS = ["title", "goal", "augment", "forward", "jacobian", "backward", "toy_check", "all_heads",
                "head_grad", "scaling"]
    TOPICS = {"goal": "what we need: −g_res", "augment": "the straight-through edit",
              "forward": "forward: nothing changes", "jacobian": "backward: values cancel, grads don't",
              "backward": "the full backward pass", "toy_check": "check with the toy",
              "all_heads": "every head, every position", "head_grad": "W still learns from data",
              "scaling": "loss weights: the factor s_k"}

    # ------------------------------------------------------------ the target
    def goal(self):
        e1 = M(r"g_{\text{hard}} - g_{\text{res}} = g_{\text{KL}}", 60).move_to(2.4 * UP)
        e2 = VGroup(T("need an extra", 34, INK), M(r"-\,g_{\text{res}}", 50), T("at", 34, INK), M(r"h^{(k)}_t", 50)).arrange(RIGHT, buff=0.2)
        e2.next_to(e1, DOWN, buff=0.5)
        self.play(Write(e1))
        self.play(FadeIn(e2))
        self.say("cancel the residual at the MTP state")
        self.beat()

        # option A: explicit KL — costs a [T, V] matrix for teacher and student
        lab_a = T("explicit KL", 32, BAD)
        mats = VGroup(Rectangle(width=3.6, height=1.0, stroke_color=C_NTP, fill_color=C_NTP, fill_opacity=0.2),
                      Rectangle(width=3.6, height=1.0, stroke_color=C_MTP, fill_color=C_MTP, fill_opacity=0.2)).arrange(DOWN, buff=0.2)
        ta = VGroup(tag("q logits [T, V]", C_NTP, 22).move_to(mats[0]), tag("p logits [T, V]", C_MTP, 22).move_to(mats[1]))
        num = tag("2048 × 50 304 ≈ 103M floats each", DIM, 22)
        A = VGroup(lab_a, VGroup(mats, ta), num).arrange(DOWN, buff=0.2).move_to([-3.4, -1.3, 0])
        self.play(FadeIn(A))
        self.say("explicit KL: two huge logit tensors", BAD)
        self.beat()
        # option B: reuse what NTP already computes
        lab_b = T("reuse the NTP gradient", 32, ACCENT)
        e3 = M(r"g_{\text{res}} = \nabla_{h^{(0)}_{t+k}}\, \mathcal{L}_{\text{NTP}}", 44)
        e4 = VGroup(T("already computed — just", 28, SOFT), T("route it, flipped", 28, ACCENT)).arrange(DOWN, buff=0.12)
        B = VGroup(lab_b, e3, e4).arrange(DOWN, buff=0.3).move_to([3.4, -1.3, 0])
        self.play(FadeIn(B))
        self.say("route NTP's gradient, with a minus sign", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ h~ = h0 + sg(hk) − hk
    def augment(self):
        e1 = M(r"\tilde h_t = h^{(0)}_t + \sum_{k=1}^{K} \underbrace{\Big[\mathrm{sg}\big(h^{(k)}_{t-k}\big) - h^{(k)}_{t-k}\Big]}_{=\,0}", 52)
        e1.move_to(2.2 * UP)
        self.play(Write(e1))
        self.say("add zero, in a special way")
        self.beat()
        e2 = M(r"\tilde h_4 = h^{(0)}_4 + \mathrm{sg}\big(h^{(1)}_3\big) - h^{(1)}_3", 48).next_to(e1, DOWN, buff=0.7)
        n2 = T("k = 1: NTP slot 4 takes MTP head 1 from position 3", 28, SOFT).next_to(e2, DOWN, buff=0.2)
        self.play(Write(e2), FadeIn(n2))
        self.say("the aligned pair from lesson 1", C_NTP)
        self.beat()
        e3 = M(r"\tilde{\mathcal{L}}_{\text{NTP}} = \sum_t \mathrm{CE}\big(x_{t+1},\, \mathrm{softmax}(W_\theta \tilde h_t)\big)", 46)
        e3.next_to(n2, DOWN, buff=0.6)
        self.play(Write(e3))
        self.say("the NTP loss now reads h~")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ forward pass with toy numbers
    def forward(self):
        e = M(r"\tilde h_4 = h^{(0)}_4 + \mathrm{sg}\big(h^{(1)}_3\big) - h^{(1)}_3", 50).move_to(2.6 * UP)
        n = M(r"= " + V(toy.H_NTP) + r" + " + V(toy.H_MTP) + r" - " + V(toy.H_MTP), 46).next_to(e, DOWN, buff=0.35)
        r = M(r"= " + V(toy.H_NTP) + r" = h^{(0)}_4", 50).next_to(n, DOWN, buff=0.35)
        self.play(FadeIn(e))
        self.play(Write(n))
        self.play(Write(r))
        self.say("numerically identical", C_NTP)
        self.beat()
        loss = -np.log(toy.q[toy.TARGET])
        c1 = M(r"\mathrm{softmax}(W_\theta \tilde h_4) = q^{(1)}_3 \quad\Rightarrow\quad \mathrm{CE} = -\log " + toy.f2(toy.q[toy.TARGET]) + " = " + toy.f2(loss), 44)
        c1.next_to(r, DOWN, buff=0.7)
        c2 = T("same logits · same loss · same NTP update", 30, SOFT).next_to(c1, DOWN, buff=0.35)
        self.play(Write(c1))
        self.play(FadeIn(c2))
        self.say("forward pass: nothing changes")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ Jacobians of h~
    def jacobian(self):
        rows = VGroup(
            M(r"\frac{\partial \tilde h_t}{\partial h^{(0)}_t} = I", 52),
            M(r"\frac{\partial \tilde h_t}{\partial\, \mathrm{sg}(\cdot)} = 0", 52),
            M(r"\frac{\partial \tilde h_t}{\partial h^{(k)}_{t-k}} = -I", 52),
        ).arrange(RIGHT, buff=1.1).move_to(1.4 * UP)
        notes = VGroup(T("NTP path untouched", 28, C_NTP), T("detached: blocked", 28, C_SG), T("minus sign gets through", 28, C_MTP))
        for n, r in zip(notes, rows):
            n.next_to(r, DOWN, buff=0.4)
        for r, n in zip(rows, notes):
            self.play(Write(r), FadeIn(n), run_time=0.9)
            self.beat()
        G = M(r"G_{t+k} := \nabla_{\tilde h_{t+k}} \mathrm{CE}\big(x_{t+k+1}, \mathrm{softmax}(W_\theta \tilde h_{t+k})\big)", 44)
        G.move_to(1.3 * DOWN)
        res = M(r"\nabla_{h^{(k)}_t} \tilde{\mathcal{L}}_{\text{NTP}} = -\,G_{t+k}", 50).next_to(G, DOWN, buff=0.4)
        self.play(Write(G))
        self.play(Write(res))
        self.say("values cancel, gradients don't")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the animated backward pass on the figure
    def backward(self):
        g = graph()
        base = VGroup(g["trunk"], g["h30"], g["head"], g["h31"], g["p31"], g["h40"], g["plus"], g["plus_l"], g["p40"],
                      g["ce1"], g["ce4"], g["fwd"])
        self.play(FadeIn(base), run_time=0.8)
        self.play(FadeIn(g["e_sg"]), FadeIn(g["l_sg"]), GrowArrow(g["e_neg"]), FadeIn(g["l_neg"]))
        self.say("forward graph with the extra edges")
        self.beat()

        panel = VGroup().move_to([4.6, 2.6, 0])

        def add_line(tex, size=31):
            m = M(tex, size)
            if len(panel):
                m.next_to(panel, DOWN, buff=0.3, aligned_edge=LEFT)
            else:
                m.move_to([1.55 + m.get_width() / 2, 3.1, 0])
            panel.add(m)
            return m

        # 1. MTP cross-entropy → g_hard at h31
        d = 0.32 * LEFT
        a_hard = arrow(g["p31"].get_bottom() + d, g["h31"].get_top() + d, C_HARD, 5, buff=0.02)
        l_hard = M(r"g_{\text{hard}}", 32).next_to(a_hard, LEFT, buff=0.1)
        m1 = add_line(r"\text{MTP CE:}\ \ g_{\text{hard}} \to h^{(1)}_3")
        self.play(GrowArrow(a_hard), FadeIn(l_hard), FadeIn(m1))
        self.say("① MTP loss sends g_hard")
        self.beat()

        # 2. NTP cross-entropy at position 4 → G4 at h~4
        d2 = 0.32 * RIGHT
        a_G = arrow(g["p40"].get_bottom() + d2, g["plus"].get_top() + d2, C_NTP, 5, buff=0.02)
        l_G = M(r"G_4", 32, C_NTP).next_to(a_G, RIGHT, buff=0.1)
        m2 = add_line(r"G_4 = W_\theta^\top\big(p^{(0)}_4 - x_5\big) = g_{\text{res}}")
        self.play(GrowArrow(a_G), FadeIn(l_G), FadeIn(m2))
        self.say("② NTP loss at t+k sends G4 = g_res", C_NTP)
        self.beat()

        # 3. the + node fans G4 out along its three inputs
        a_I = arrow(g["plus"].get_bottom() + d2, g["h40"].get_top() + d2, C_NTP, 5, buff=0.02)
        l_I = M(r"\times I", 28, C_NTP).next_to(a_I, RIGHT, buff=0.1)
        self.play(GrowArrow(a_I), FadeIn(l_I))
        m3 = add_line(r"h^{(0)}_4\ \text{gets } G_4\ \text{(unchanged)}")
        self.play(FadeIn(m3))
        self.say("NTP path: ordinary gradient", C_NTP)
        self.beat()
        block_x = Cross(g["e_sg"], stroke_color=C_SG).scale(0.35)
        l0 = M(r"\times 0", 28, C_SG).next_to(g["l_sg"], UP, buff=0.05)
        self.play(ShowCreation(block_x), FadeIn(l0))
        self.say("sg edge: blocked", C_SG)
        self.beat()
        # the gradient runs back along the same −h edge, opposite to the forward arrow
        e = g["e_neg"]
        y = e.get_y() - 0.05
        a_neg = arrow([e.get_right()[0], y, 0], [e.get_left()[0], y, 0], C_RES, 6, buff=0.0, tip=0.1)
        l_neg = M(r"-\,g_{\text{res}}\ (\times -I)", 30).next_to(g["l_neg"], DOWN, buff=0.12)
        self.play(GrowArrow(a_neg), FadeIn(l_neg))
        m4 = add_line(r"h^{(1)}_3\ \text{gets } -G_4 = -\,g_{\text{res}}")
        self.play(FadeIn(m4))
        self.say("③ the −h edge delivers −g_res", C_RES)
        self.beat()

        # 4. sum at h31 → only g_KL flows down
        hl = Circle(radius=0.62, stroke_color=C_KL, stroke_width=4).move_to(g["h31"])
        m5 = add_line(r"g_{\text{hard}} - g_{\text{res}} = g_{\text{KL}}", 36)
        self.play(ShowCreation(hl), FadeIn(m5))
        k1 = arrow(g["h31"].get_bottom() + d, g["head"].get_top() + d, C_KL, 6, buff=0.02)
        k2 = arrow(g["head"].get_bottom() + d, g["h30"].get_top() + d, C_KL, 6, buff=0.02)
        k3 = arrow(g["h30"].get_bottom() + d, [g["h30"].get_x() + d[0], g["trunk"].get_top()[1], 0], C_KL, 6, buff=0.02)
        lk = M(r"g_{\text{KL}}", 34).next_to(k1, LEFT, buff=0.1)
        self.play(GrowArrow(k1), FadeIn(lk))
        self.play(GrowArrow(k2))
        self.play(GrowArrow(k3))
        self.say("④ only g_KL reaches the trunk", C_KL)
        self.beat()
        eq7 = M(r"\nabla_{h^{(k)}_t}\big(\tilde{\mathcal{L}}_{\text{NTP}} + \mathcal{L}^{(k)}_{\text{MTP}}\big) = g_{\text{KL}}", 36)
        eq7.next_to(panel, DOWN, buff=0.5, aligned_edge=LEFT)
        box = SurroundingRectangle(eq7, buff=0.15, stroke_color=C_KL)
        self.play(Write(eq7), ShowCreation(box))
        self.say("exact KL gradient, no KL computed", C_KL)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ toy numbers: g_hard + (−g_res) = g_KL
    def toy_check(self):
        pl = Plane(center=[-4.6, 1.5, 0], unit=2.4, xr=(-0.4, 1.6), yr=(-1.7, 0.4))
        o = (0, 0)
        gh = pl.arrow(o, toy.g_hard, C_HARD, thick=6)
        gr = pl.arrow(toy.g_hard, toy.g_hard - toy.g_res, C_RES, thick=6)
        gk = pl.arrow(o, toy.g_kl, C_KL, thick=6)
        lh = M(r"g_{\text{hard}}", 40).next_to(gh.get_center(), LEFT, buff=0.25)
        lr = M(r"-\,g_{\text{res}}", 40).next_to(gr.get_center(), RIGHT, buff=0.2)
        lk = M(r"g_{\text{KL}}", 40).next_to(gk.get_center(), UP + RIGHT, buff=0.1)
        self.play(FadeIn(pl), GrowArrow(gh), FadeIn(lh))
        e1 = M(r"g_{\text{hard}} = " + V(toy.g_hard), 46)
        e2 = M(r"-\,g_{\text{res}} = " + V(-toy.g_res), 46)
        e3 = M(r"g_{\text{KL}} = " + V(toy.g_hard - toy.g_res), 46)
        col = VGroup(e1, e2, e3).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to([3.2, 1.0, 0])
        self.play(FadeIn(e1))
        self.say("what standard MTP would send")
        self.beat()
        self.play(GrowArrow(gr), FadeIn(lr), FadeIn(e2))
        self.say("add the routed −g_res", C_RES)
        self.beat()
        self.play(GrowArrow(gk), FadeIn(lk), FadeIn(e3))
        chk = M(r"= \bar w_p - \bar w_q = \nabla D_{\text{KL}}\ \checkmark", 44).next_to(e3, DOWN, buff=0.4).align_to(e3, LEFT)
        self.play(FadeIn(chk))
        self.say("exactly the KL gradient", C_KL)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the same routing for every k
    def all_heads(self):
        f = mtp_figure(center=2.8 * LEFT + 0.25 * DOWN, scale=0.9)
        self.play(FadeIn(f["all"]))
        r1 = ArcBetweenPoints(f["ntp4"].get_right() + 0.05 * RIGHT, f["mtp1"].get_right() + 0.05 * RIGHT, angle=PI / 2.2)
        r2 = ArcBetweenPoints(f["ntp5"].get_right() + 0.05 * RIGHT, f["mtp2"].get_right() + 0.05 * RIGHT, angle=PI / 2.2)
        arcs = VGroup()
        for r in (r1, r2):
            r.set_stroke(C_RES, 4)
            arcs.add(VGroup(r, tip_at(r.get_end(), r.get_end() - r.point_from_proportion(0.95), C_RES, 0.2)))
        l1 = M(r"-\,g^{(1)}_{\text{res}}", 30, C_RES).next_to(r1, RIGHT, buff=0.08)
        l2 = M(r"-\,g^{(2)}_{\text{res}}", 30, C_RES).next_to(r2, RIGHT, buff=0.08)
        self.play(ShowCreation(arcs[0]), FadeIn(l1))
        self.play(ShowCreation(arcs[1]), FadeIn(l2))
        self.say("each aligned NTP slot feeds its head", C_RES)
        self.beat()
        e = M(r"\nabla_{h^{(k)}_t} \tilde{\mathcal{L}}_{\text{NTP}} = -\,G_{t+k}", 44)
        e2 = M(r"t \le T-k-1", 40, SOFT)
        n = T("NTP target exists ⇔ MTP target exists", 26, SOFT)
        VGroup(e, e2, n).arrange(DOWN, buff=0.35).move_to([4.2, 0.6, 0])
        self.play(Write(e))
        self.play(FadeIn(e2), FadeIn(n))
        self.say("the positions line up exactly")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the output head still gets CE
    def head_grad(self):
        e1 = M(r"\nabla_{W_\theta}\, \mathrm{CE}\big(x_{t+1}, \mathrm{softmax}(W_\theta \tilde h_t)\big) = \big(p^{(0)}_t - x_{t+1}\big)\, \tilde h_t^{\top}", 42)
        e1b = M(r"= \big(p^{(0)}_t - x_{t+1}\big)\, h^{(0)\top}_t", 42, t2c={r"h^{(0)\top}_t": C_NTP})
        e2 = M(r"\nabla_{W_\theta}\, \mathcal{L}^{(k)}_{\text{MTP}} \ni \big(p^{(k)}_t - x_{t+k+1}\big)\, h^{(k)\top}_t", 42, t2c={r"h^{(k)\top}_t": C_MTP})
        VGroup(e1, e1b, e2).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(1.9 * UP)
        e2.shift(0.25 * DOWN)
        self.play(Write(e1))
        self.play(Write(e1b))
        self.say("W's NTP gradient: unchanged", C_W)
        self.beat()
        self.play(Write(e2))
        self.say("W's MTP gradient: still CE vs data", C_X)
        self.beat()

        hdr = ["", "output head W", "MTP heads + trunk"]
        data = [("standard MTP", "CE", "CE"), ("KL-MTP", "KL", "KL"), ("Soft-MTP", "CE", "KL")]
        colc = {"CE": C_X, "KL": C_KL}
        cells = VGroup()
        xs, y0 = [-3.4, 0.4, 4.0], -0.9
        for j, h in enumerate(hdr):
            cells.add(T(h, 28, DIM).move_to([xs[j], y0, 0]))
        for i, (name, a, b) in enumerate(data):
            y = y0 - 0.7 * (i + 1)
            cells.add(T(name, 32, INK if name != "Soft-MTP" else ACCENT).move_to([xs[0], y, 0]))
            cells.add(T(a, 32, colc[a]).move_to([xs[1], y, 0]))
            cells.add(T(b, 32, colc[b]).move_to([xs[2], y, 0]))
        line = Line([-5.0, y0 - 0.35, 0], [5.6, y0 - 0.35, 0], stroke_color=RULE)
        self.play(FadeIn(cells[:3]), ShowCreation(line))
        self.play(LaggedStartMap(FadeIn, cells[3:], lag_ratio=0.08))
        box = SurroundingRectangle(VGroup(*cells[-3:]), stroke_color=ACCENT, buff=0.12)
        self.play(ShowCreation(box))
        self.say("Soft-MTP: CE at W, KL below it", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ λ/K and per-head means → s_k
    def scaling(self):
        e1 = M(r"\mathcal{L} = \frac{1}{N_0}\sum_t \mathrm{CE}\big(x_{t+1}, p^{(0)}_t\big)"
               r" + \frac{\lambda}{K}\sum_{k}\frac{1}{N_k}\sum_t \mathrm{CE}\big(x_{t+k+1}, p^{(k)}_t\big)", 40)
        e1.move_to(3.0 * UP)
        self.play(Write(e1))
        self.say("real losses are means, weighted")
        self.beat()
        a = M(r"\text{MTP term at } h^{(k)}_t:\quad \frac{\lambda}{K N_k}\, g_{\text{hard}}", 40)
        b = M(r"\text{plain routing gives}:\quad -\frac{1}{N_0}\, g_{\text{res}}", 40)
        VGroup(a, b).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(e1, DOWN, buff=0.5)
        self.play(Write(a))
        self.play(Write(b))
        mism = T("scales don't match", 30, BAD).next_to(b, RIGHT, buff=0.4)
        self.play(FadeIn(mism))
        self.say("the cancellation would be off", BAD)
        self.beat()
        c = M(r"\tilde h_t = h^{(0)}_t + \sum_k s_k \Big[\mathrm{sg}\big(h^{(k)}_{t-k}\big) - h^{(k)}_{t-k}\Big]", 40)
        d = M(r"-\frac{s_k}{N_0}\, g_{\text{res}} = -\frac{\lambda}{K N_k}\, g_{\text{res}}"
              r"\ \Rightarrow\ s_k = \frac{\lambda\, N_0}{K\, N_k}", 42,
              isolate=[r"s_k = \frac{\lambda\, N_0}{K\, N_k}"])
        VGroup(c, d).arrange(DOWN, buff=0.35).next_to(b, DOWN, buff=0.5)
        self.play(FadeOut(mism), Write(c))
        self.play(Write(d))
        fr = SurroundingRectangle(d[r"s_k = \frac{\lambda\, N_0}{K\, N_k}"], stroke_color=GOLD, buff=0.08)
        self.play(ShowCreation(fr))
        self.say("scale the edit by s_k", GOLD)
        self.beat()
        self.fade_all()

        e = M(r"\nabla_{h^{(k)}_t} \mathcal{L}_{\text{Soft-MTP}} = \frac{\lambda}{K N_k}\big(g_{\text{hard}} - g_{\text{res}}\big) = \frac{\lambda}{K N_k}\, g_{\text{KL}}", 46)
        e.move_to(2.3 * UP)
        f = M(r"N_0 = T,\quad N_k = T - k \quad\Rightarrow\quad s_k = \frac{\lambda\, T}{K\,(T-k)}", 46).next_to(e, DOWN, buff=0.6)
        vals = VGroup(*[tag(f"s_{k} = 2048 / (3·{2048 - k}) = {2048 / (3 * (2048 - k)):.4f}", GOLD, 28) for k in (1, 2, 3)])
        vals.arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(f, DOWN, buff=0.5)
        sub = T("T = 2048, K = 3, λ = 1", 26, DIM).next_to(vals, UP, buff=0.12).align_to(vals, LEFT)
        vals.shift(0.25 * DOWN)
        self.play(Write(e))
        self.say("matches the weighted KL exactly", C_KL)
        self.beat()
        self.play(Write(f))
        self.play(FadeIn(sub), FadeIn(vals))
        self.say("with sum-reduction: s_k = λ/K", GOLD)
        self.beat()
        self.clear_all()


split_sections(L3_Trick, globals())
