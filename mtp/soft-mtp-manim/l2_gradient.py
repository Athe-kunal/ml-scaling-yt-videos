"""Lesson 2 · The MTP gradient splits exactly into a KL part and a residual."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from fig import Plane, token_rows, bars, node
import toy

V = toy.vec


def stack(*eqs, buff=0.4, x=-6.4, top=3.2):
    """Left-aligned derivation lines starting at the top-left."""
    g = VGroup(*eqs).arrange(DOWN, buff=buff, aligned_edge=LEFT)
    g.move_to([x + g.get_width() / 2, top - g.get_height() / 2, 0])
    return g


class L2_Gradient(Lesson):
    NUM, TITLE, SUB = 2, "Splitting the MTP gradient", "hard = KL + residual"
    SECTIONS = ["title", "ce_logits", "chain", "centroid", "split", "kl", "upstream"]
    TOPICS = {"ce_logits": "CE gradient w.r.t. logits", "chain": "back to the hidden state",
              "centroid": "the picture: average row", "split": "add and subtract q",
              "kl": "the KL part is self-distillation", "upstream": "the split survives backprop"}

    # ------------------------------------------------------------ dCE/dz = p − y
    def ce_logits(self):
        e1 = M(r"\mathrm{CE}(y, p) = -\sum_v y_v \log p_v", 42)
        e2 = M(r"\log p_v = z_v - \log \sum_u e^{z_u}", 42)
        e3 = M(r"\mathrm{CE} = -\sum_v y_v z_v + \Big(\sum_v y_v\Big) \log \sum_u e^{z_u}", 42, isolate=[r"\Big(\sum_v y_v\Big)"])
        e4 = M(r"\mathrm{CE} = -\,y^\top z + \log \sum_u e^{z_u}", 42)
        e5 = M(r"\frac{\partial\, \mathrm{CE}}{\partial z_v} = -\,y_v + \frac{e^{z_v}}{\sum_u e^{z_u}} = p_v - y_v", 42)
        g = stack(e1, e2, e3, e4, e5, buff=0.32, top=3.55)
        side = VGroup(M(r"z = \text{logits}", 32, SOFT), M(r"p = \mathrm{softmax}(z)", 32, SOFT),
                      M(r"y = \text{target},\ \textstyle\sum_v y_v = 1", 32, C_X)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        side.to_corner(UR, buff=0.4)
        self.play(FadeIn(side))
        self.play(Write(e1))
        self.say("cross-entropy against a target y")
        self.beat()
        self.play(Write(e2))
        self.say("log of a softmax")
        self.beat()
        self.play(Write(e3))
        one = SurroundingRectangle(e3[r"\Big(\sum_v y_v\Big)"], stroke_color=C_X, buff=0.06)
        eq1 = M(r"= 1", 36, C_X).next_to(one, DOWN, buff=0.1)
        self.play(ShowCreation(one), FadeIn(eq1))
        self.say("the target sums to 1", C_X)
        self.beat()
        self.play(FadeOut(one), FadeOut(eq1), Write(e4))
        self.say("so CE = −y·z + logsumexp")
        self.beat()
        self.play(Write(e5))
        self.say("differentiate one logit")
        self.beat()

        box_eq = M(r"\nabla_z\, \mathrm{CE}(y, \mathrm{softmax}(z)) = p - y", 56)
        self.play(FadeOut(VGroup(e1, e2, e3, e4)), FadeOut(side), ReplacementTransform(e5, box_eq.move_to(2.4 * UP)))
        frame = SurroundingRectangle(box_eq, buff=0.2, stroke_color=ACCENT)
        soft = T("y can be one-hot or any distribution", 30, C_X).next_to(frame, DOWN, buff=0.25)
        self.play(ShowCreation(frame), FadeIn(soft))
        self.say("prediction minus target", ACCENT)
        self.beat()

        # toy: p^(1)_3 − x_5
        bp = bars(toy.p, C_MTP, h=1.6, w=0.6, size=24).move_to([-4.3, -1.0, 0])
        bx = bars(toy.x, C_X, h=1.6, w=0.6, size=24).move_to([-0.6, -1.0, 0]).align_to(bp, DOWN)
        bd = bars(toy.p - toy.x, C_HARD, h=2.4, w=0.6, size=24, signed=True,
                  colors=[C_MTP, C_X, C_MTP, C_MTP]).move_to([4.0, -1.05, 0])
        minus = M("-", 60).move_to([-2.45, -1.0, 0])
        eqs = M("=", 60).move_to([1.6, -1.0, 0])
        lp = M(r"p^{(1)}_3", 36).next_to(bp, DOWN, buff=0.15)
        lx = M(r"x_5", 36).next_to(bx, DOWN, buff=0.15)
        ld = M(r"\nabla_z \mathrm{CE}", 36).next_to(bd, DOWN, buff=0.1)
        self.play(FadeIn(bp), FadeIn(lp))
        self.play(FadeIn(minus), FadeIn(bx), FadeIn(lx))
        self.play(FadeIn(eqs), FadeIn(bd), FadeIn(ld))
        self.say("push 'cat' up, everything else down")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ chain rule through z = W h
    def chain(self):
        e1 = M(r"z^{(k)}_t = W_\theta\, h^{(k)}_t \quad\Rightarrow\quad \frac{\partial z^{(k)}_t}{\partial h^{(k)}_t} = W_\theta", 48)
        e2 = M(r"\nabla_{h^{(k)}_t} \mathrm{CE} = \Big(\frac{\partial z^{(k)}_t}{\partial h^{(k)}_t}\Big)^{\!\top} \nabla_{z} \mathrm{CE}", 48)
        e3 = M(r"g_{\text{hard}} = W_\theta^\top \big(p^{(k)}_t - x_{t+k+1}\big)", 58)
        g = VGroup(e1, e2, e3).arrange(DOWN, buff=0.55).move_to(0.9 * UP)
        self.play(Write(e1))
        self.say("logits are linear in h", C_W)
        self.beat()
        self.play(Write(e2))
        self.say("chain rule: transpose the Jacobian")
        self.beat()
        self.play(Write(e3))
        fr = SurroundingRectangle(e3, buff=0.18, stroke_color=C_HARD)
        shp = VGroup(tag("[d]", C_HARD, 28), tag("=", SOFT, 28), tag("[d, V]", C_W, 28), tag("[V]", SOFT, 28)).arrange(RIGHT, buff=0.3)
        shp.next_to(fr, DOWN, buff=0.25)
        self.play(ShowCreation(fr), FadeIn(shp))
        self.say("the hard MTP gradient")
        self.beat()

        ex = M(r"W_\theta^\top (p - x) = \sum_v (p_v - x_v)\, w_v = " + V(toy.g_hard), 44)
        ex.next_to(shp, DOWN, buff=0.5)
        self.play(Write(ex))
        self.say("toy: mix of W's rows")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ W^T p = probability-weighted average row
    def centroid(self):
        pl = Plane(center=[-3.4, -0.2, 0], unit=1.35, lim=2.5)
        rows = token_rows(pl, 28, highlight=toy.TARGET)
        self.play(FadeIn(pl), FadeIn(rows))
        e1 = M(r"W_\theta^\top p = \sum_v p_v\, w_v =: \bar w_p", 46)
        e2 = M(r"W_\theta^\top x_5 = w_{\text{cat}}", 46)
        e3 = M(r"g_{\text{hard}} = \bar w_p - w_{\text{cat}}", 50)
        col = VGroup(e1, e2, e3).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to([3.4, 1.2, 0])
        self.play(Write(e1))
        dp = pl.dot(toy.mu_p, C_MTP, 0.11)
        lp = M(r"\bar w_p", 40).next_to(dp, RIGHT, buff=0.12)
        ghosts = VGroup(*[Line(pl.p(w), pl.p(toy.mu_p), stroke_color=C_MTP, stroke_width=1 + 6 * pv, stroke_opacity=0.6)
                          for w, pv in zip(toy.W, toy.p)])
        self.play(ShowCreation(ghosts, lag_ratio=0.1))
        self.play(FadeIn(dp, scale=0.5), FadeIn(lp), FadeOut(ghosts))
        self.say("p averages the rows", C_MTP)
        self.beat()
        dx = pl.dot(toy.w_y, C_X, 0.11)
        self.play(Write(e2), FadeIn(dx, scale=0.5))
        self.say("one-hot x picks one row", C_X)
        self.beat()
        gh = pl.arrow(toy.w_y, toy.mu_p, C_HARD, thick=6)
        lg = M(r"g_{\text{hard}}", 40).next_to(gh.get_center(), RIGHT, buff=0.15)
        self.play(Write(e3), GrowArrow(gh), FadeIn(lg))
        num = VGroup(M(r"= " + V(toy.mu_p) + r" - " + V(toy.w_y), 36), M(r"= " + V(toy.g_hard), 36, C_HARD))
        num.arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(e3, DOWN, buff=0.25).align_to(e3, LEFT).shift(0.6 * RIGHT)
        self.play(FadeIn(num))
        self.say("gradient: from target to average")
        self.beat()
        step = VGroup(T("descent moves h along", 28, SOFT), M(r"-g_{\text{hard}}", 36)).arrange(RIGHT, buff=0.15)
        step.next_to(num, DOWN, buff=0.45).align_to(e3, LEFT)
        self.play(FadeIn(step))
        self.say("so the average slides toward 'cat'", C_X)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ ± q: KL part + residual
    def split(self):
        e1 = M(r"g_{\text{hard}} = W_\theta^\top \big(p^{(k)}_t - x_{t+k+1}\big)", 50)
        e2 = M(r"= W_\theta^\top \big(p^{(k)}_t \;{-\, q^{(k)}_t + q^{(k)}_t}\; - x_{t+k+1}\big)", 50)
        e3 = M(r"= \underbrace{W_\theta^\top \big(p^{(k)}_t - q^{(k)}_t\big)}_{g_{\text{KL}}}"
               r" + \underbrace{W_\theta^\top \big(q^{(k)}_t - x_{t+k+1}\big)}_{g_{\text{res}}}", 50,
               isolate=[r"W_\theta^\top \big(p^{(k)}_t - q^{(k)}_t\big)", r"W_\theta^\top \big(q^{(k)}_t - x_{t+k+1}\big)"])
        g = VGroup(e1, e2, e3).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(0.8 * UP)
        self.play(Write(e1))
        self.beat()
        self.play(Write(e2))
        zero = M(r"= 0", 36, GOLD).next_to(e2, RIGHT, buff=0.3)
        self.play(FadeIn(zero))
        self.say("add and subtract the teacher q", C_NTP)
        self.beat()
        self.play(FadeOut(zero), Write(e3))
        self.say("an exact split, no approximation")
        self.beat()
        kl_b = SurroundingRectangle(e3[r"W_\theta^\top \big(p^{(k)}_t - q^{(k)}_t\big)"], stroke_color=C_KL, buff=0.08)
        rs_b = SurroundingRectangle(e3[r"W_\theta^\top \big(q^{(k)}_t - x_{t+k+1}\big)"], stroke_color=C_RES, buff=0.08)
        n1 = T("model vs model: no data", 30, C_KL).next_to(kl_b, DOWN, buff=0.7)
        n2 = T("depends on the sampled x", 30, C_RES).next_to(rs_b, DOWN, buff=0.7)
        self.play(ShowCreation(kl_b), FadeIn(n1))
        self.play(ShowCreation(rs_b), FadeIn(n2))
        self.say("two very different pieces")
        self.beat()
        self.fade_all()

        # the triangle in the toy plane
        pl = Plane(center=[-3.7, -0.1, 0], unit=1.3, lim=2.4)
        rows = token_rows(pl, 26, highlight=toy.TARGET)
        dp, dq, dx = pl.dot(toy.mu_p, C_MTP, 0.11), pl.dot(toy.mu_q, C_NTP, 0.11), pl.dot(toy.w_y, C_X, 0.11)
        lp = M(r"\bar w_p", 38).next_to(dp, DOWN, buff=0.12)
        lq = M(r"\bar w_q", 38).next_to(dq, LEFT, buff=0.15)
        gh = pl.arrow(toy.w_y, toy.mu_p, C_HARD, thick=5)
        self.play(FadeIn(pl), FadeIn(rows), FadeIn(dp), FadeIn(dx), FadeIn(lp), GrowArrow(gh))
        self.play(FadeIn(dq, scale=0.5), FadeIn(lq))
        self.say("teacher's average sits in between", C_NTP)
        self.beat()
        # zoom: origin moves to the bottom-left so the triangle fills the left half
        zu, mid = 2.4, np.array([0.8, 1.3])
        zo = np.array([-3.4, 0.3, 0]) - zu * np.array([*mid, 0])
        zp = Plane(center=zo, unit=zu, xr=(-0.4, 2.1), yr=(-0.1, 2.45))
        zrows = token_rows(zp, 30, highlight=toy.TARGET, only=(0, 1))
        zdp, zdq, zdx = zp.dot(toy.mu_p, C_MTP, 0.13), zp.dot(toy.mu_q, C_NTP, 0.13), zp.dot(toy.w_y, C_X, 0.13)
        zlp = M(r"\bar w_p", 44).next_to(zdp, DOWN, buff=0.15)
        zlq = M(r"\bar w_q", 44).next_to(zdq, DOWN + LEFT, buff=0.12)
        zgh = zp.arrow(toy.w_y, toy.mu_p, C_HARD, thick=5)
        self.play(FadeOut(pl), FadeOut(VGroup(rows[0], rows[2], rows[3])),
                  ReplacementTransform(rows[1], zrows[1]), FadeIn(zp), FadeIn(zrows[0]),
                  *[ReplacementTransform(a, b) for a, b in [(dp, zdp), (dq, zdq), (dx, zdx), (lp, zlp), (lq, zlq), (gh, zgh)]],
                  run_time=1.4)
        pl, gr, gk = zp, zp.arrow(toy.w_y, toy.mu_q, C_RES, thick=7), zp.arrow(toy.mu_q, toy.mu_p, C_KL, thick=7)
        r1 = M(r"g_{\text{res}} = \bar w_q - w_{\text{cat}} = " + V(toy.g_res), 40)
        r2 = M(r"g_{\text{KL}} = \bar w_p - \bar w_q = " + V(toy.g_kl), 40)
        r3 = M(r"g_{\text{hard}} = " + V(toy.g_hard), 40)
        col = VGroup(r1, r2, r3).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([3.6, 1.0, 0])
        self.play(GrowArrow(gr), Write(r1))
        self.play(GrowArrow(gk), Write(r2))
        self.play(Write(r3))
        chk = M(r"g_{\text{res}} + g_{\text{KL}} = g_{\text{hard}}\ \checkmark", 40).next_to(col, DOWN, buff=0.55).align_to(col, LEFT)
        self.play(FadeIn(chk))
        self.say("tip to tail: res + KL = hard")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ why g_KL is the KL gradient; what g_res is
    def kl(self):
        e1 = M(r"D_{\text{KL}}\big(\mathrm{sg}(q^{(k)}_t)\,\Vert\,p^{(k)}_t\big) = \sum_v q_v \log q_v - \sum_v q_v \log p_v", 46)
        e2 = M(r"= -H\big(q^{(k)}_t\big) + \mathrm{CE}\big(q^{(k)}_t, p^{(k)}_t\big)", 46, isolate=[r"-H\big(q^{(k)}_t\big)"])
        e3 = M(r"\nabla_{h^{(k)}_t} D_{\text{KL}} = \nabla_{h^{(k)}_t} \mathrm{CE}\big(q^{(k)}_t, p^{(k)}_t\big) = W_\theta^\top \big(p^{(k)}_t - q^{(k)}_t\big) = g_{\text{KL}}", 44)
        g = stack(e1, e2, e3, buff=0.65, x=-6.5, top=3.4)
        self.play(Write(e1))
        self.say("teacher frozen with stop-gradient", C_SG)
        self.beat()
        self.play(Write(e2))
        ent = e2[r"-H\big(q^{(k)}_t\big)"]
        note = T("constant in h → 0", 26, C_SG).next_to(ent, DOWN, buff=0.1)
        self.play(ent.animate.set_opacity(0.3), FadeIn(note))
        self.say("entropy of a frozen q: no gradient")
        self.beat()
        self.play(Write(e3))
        self.say("same p − y rule, soft target y = q", C_KL)
        self.beat()
        fr = SurroundingRectangle(e3[r"g_{\text{KL}}"], stroke_color=C_KL, buff=0.08)
        self.play(ShowCreation(fr))
        self.say("g_KL pulls the student to the teacher", C_KL)
        self.beat()

        # the residual is exactly the NTP gradient at position t+k
        r1 = M(r"g_{\text{res}} = W_\theta^\top \big(q^{(k)}_t - x_{t+k+1}\big) = W_\theta^\top \big(p^{(0)}_{t+k} - x_{t+k+1}\big)", 46)
        r2 = M(r"= \nabla_{h^{(0)}_{t+k}} \mathrm{CE}\big(x_{t+k+1},\, p^{(0)}_{t+k}\big)", 46)
        rr = VGroup(r1, r2).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(g, DOWN, buff=0.55).align_to(g, LEFT)
        self.play(Write(r1))
        self.play(Write(r2))
        fr2 = SurroundingRectangle(r2, stroke_color=C_RES, buff=0.12)
        self.play(ShowCreation(fr2))
        self.say("g_res = NTP's own gradient at t+k", C_RES)
        self.beat()
        hint = T("the NTP loss already computes it", 30, GOLD).next_to(fr2, RIGHT, buff=0.4)
        self.play(FadeIn(hint, shift=0.1 * LEFT))
        self.say("remember this for the trick", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ backprop is linear in the incoming gradient
    def upstream(self):
        trunk = block("Shared Trunk", w=4.2, h=0.85, color=SOFT).move_to([-3.4, -2.6, 0])
        head = block("MTP head k", w=2.8, h=0.8, color=C_MTP).move_to([-3.4, -0.9, 0])
        h = node(r"h^{(k)}_t", C_MTP, r=0.5, size=34).move_to([-3.4, 0.7, 0])
        pk = node(r"p^{(k)}_t", C_MTP, r=0.5, size=34).move_to([-3.4, 2.5, 0])
        fw = VGroup(arrow(trunk.get_top(), head.get_bottom(), DIM, 3), arrow(head.get_top(), h.get_bottom(), DIM, 3),
                    arrow(h.get_top(), pk.get_bottom(), DIM, 3))
        self.play(FadeIn(trunk), FadeIn(head), FadeIn(h), FadeIn(pk), FadeIn(fw))
        ce = M(r"\mathrm{CE}\big(x_{t+k+1},\cdot\big)", 34, C_X).next_to(pk, RIGHT, buff=0.25)
        self.play(FadeIn(ce))
        gh = arrow(pk.get_bottom() + 0.45 * LEFT + 0.05 * DOWN, h.get_top() + 0.45 * LEFT, C_HARD, 5)
        lgh = M(r"g_{\text{hard}}", 36).next_to(gh, LEFT, buff=0.1)
        self.play(GrowArrow(gh), FadeIn(lgh))
        self.say("g_hard arrives at the MTP state")
        self.beat()

        def down(a, b, col, dx):
            return arrow(a + dx * RIGHT, b + dx * RIGHT, col, 5)
        k1 = down(h.get_bottom(), head.get_top(), C_KL, -0.35)
        r1 = down(h.get_bottom(), head.get_top(), C_RES, 0.35)
        k2 = down(head.get_bottom(), trunk.get_top(), C_KL, -0.35)
        r2 = down(head.get_bottom(), trunk.get_top(), C_RES, 0.35)
        self.play(GrowArrow(k1), GrowArrow(r1))
        self.play(GrowArrow(k2), GrowArrow(r2))
        e1 = M(r"\nabla_\phi = J^\top g_{\text{hard}} = J^\top g_{\text{KL}} + J^\top g_{\text{res}}", 46)
        e2 = M(r"J = \frac{\partial h^{(k)}_t}{\partial \phi}\quad \phi \in \text{MTP head, trunk}", 38, SOFT)
        VGroup(e1, e2).arrange(DOWN, buff=0.35).move_to([2.9, 0.6, 0])
        self.play(Write(e1), FadeIn(e2))
        self.say("backprop is linear: the split survives")
        self.beat()
        goal = VGroup(T("goal: send only", 34, INK), M(r"g_{\text{KL}}", 44), T("into the trunk", 34, INK)).arrange(RIGHT, buff=0.2)
        goal.next_to(VGroup(e1, e2), DOWN, buff=0.8)
        self.play(FadeIn(goal), r1.animate.set_opacity(0.2), r2.animate.set_opacity(0.2))
        self.say("drop the red part — but how?", C_RES)
        self.beat()
        self.clear_all()


split_sections(L2_Gradient, globals())
