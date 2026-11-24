"""Lesson 4 · Why dropping the residual helps: bias vs. variance, drafts, and why W keeps CE."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
from fig import Plane, token_rows, bars
import toy

V = toy.vec
C_BIAS = GOLD   # NTP error q − π


class L4_Why(Lesson):
    NUM, TITLE, SUB = 4, "Why drop the residual?", "bias · variance · drafts"
    SECTIONS = ["title", "bias_noise", "sampling", "scale", "drafts", "shortcut", "recap"]
    TOPICS = {"bias_noise": "the residual = error + noise", "sampling": "watch the noise",
              "scale": "what shrinks with scale", "drafts": "the bias is the goal",
              "shortcut": "why W keeps cross-entropy", "recap": "the whole story"}

    # ------------------------------------------------------------ g_res = W^T(q − π) + W^T(π − x)
    def bias_noise(self):
        d = VGroup(M(r"\pi^{(k)}_t", 46, C_X), T("= true distribution of the next token", 30, SOFT)).arrange(RIGHT, buff=0.2)
        d.move_to(2.8 * UP)
        e = M(r"g_{\text{res}} = \underbrace{W_\theta^\top\big(q^{(k)}_t - \pi^{(k)}_t\big)}_{\text{NTP error}}"
              r" + \underbrace{W_\theta^\top\big(\pi^{(k)}_t - x_{t+k+1}\big)}_{\text{one-hot noise}}", 50,
              t2c={r"\pi^{(k)}_t": C_X})
        e.move_to(1.0 * UP)
        self.play(FadeIn(d))
        self.play(Write(e))
        self.say("add and subtract π this time", C_X)
        self.beat()
        m = M(r"x_{t+k+1} \sim \pi^{(k)}_t \ \Rightarrow\ \mathbb{E}\big[x_{t+k+1}\big] = \pi^{(k)}_t", 44, t2c={r"\pi^{(k)}_t": C_X})
        m.next_to(e, DOWN, buff=0.8)
        z = T("noise has zero mean, but never goes away", 30, C_RES).next_to(m, DOWN, buff=0.3)
        self.play(Write(m))
        self.play(FadeIn(z))
        self.say("unbiased — but noisy", C_RES)
        self.beat()
        k = VGroup(M(r"g_{\text{KL}}", 46), T("contains no sampled x at all", 30, C_KL)).arrange(RIGHT, buff=0.3)
        k.next_to(z, DOWN, buff=0.45)
        self.play(FadeIn(k))
        self.say("g_KL has no sampling noise", C_KL)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ resample the target, watch g_res jump
    def sampling(self):
        pl = Plane(center=[-4.3, -0.6, 0], unit=1.0, lim=0, xr=(-2.1, 2.6), yr=(-1.1, 3.6))
        gk = pl.arrow((0, 0), toy.g_kl, C_KL, thick=6)
        lk = M(r"g_{\text{KL}}", 38).next_to(gk.get_end(), DOWN, buff=0.1)
        bp = bars(toy.PI, C_X, h=1.6, w=0.6, size=24).move_to([3.4, 1.6, 0])
        lp = M(r"\pi\ \text{(true next-token dist.)}", 32, C_X).next_to(bp, DOWN, buff=0.15)
        self.play(FadeIn(pl), GrowArrow(gk), FadeIn(lk), FadeIn(bp), FadeIn(lp))
        self.say("g_KL: the same arrow every time", C_KL)
        self.beat()

        seq = [1, 1, 0, 1, 2, 1, 1, 3, 0, 1, 1, 1, 2, 1, 0, 1]   # draws from π (cat-heavy)
        lab = T("sample 1: x = cat", 30, C_RES).move_to([3.4, -0.6, 0])
        g = pl.arrow((0, 0), toy.g_res_for(seq[0]), C_RES, thick=6)
        mean = toy.g_res_for(seq[0]).copy()
        md = pl.dot(mean, INK, 0.1)
        ml = T("running mean", 24, INK).next_to(md, RIGHT, buff=0.1)
        self.play(GrowArrow(g), FadeIn(lab), FadeIn(md), FadeIn(ml))
        for n, i in enumerate(seq[1:], start=2):
            mean = mean + (toy.g_res_for(i) - mean) / n
            ng = pl.arrow((0, 0), toy.g_res_for(i), C_RES, thick=6)
            nl = T(f"sample {n}: x = {toy.VOCAB[i]}", 30, C_RES).move_to(lab)
            nd = pl.dot(mean, INK, 0.1)
            self.play(Transform(g, ng), Transform(lab, nl), Transform(md, nd),
                      ml.animate.next_to(nd, RIGHT, buff=0.1), run_time=0.35)
        self.say("g_res jumps with every sample", C_RES)
        self.beat()
        e = M(r"\mathbb{E}[g_{\text{res}}] = W_\theta^\top(q - \pi) = " + V(toy.ntp_error), 36)
        e.move_to([3.4, -1.6, 0])
        self.play(FadeOut(g), Transform(md, pl.dot(toy.ntp_error, C_BIAS, 0.12)), FadeOut(ml), FadeIn(e))
        self.say("its average is tiny: the NTP error", C_BIAS)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ bias shrinks, noise doesn't
    def scale(self):
        ax = Axes(x_range=(0, 10, 1), y_range=(0, 1.1, 0.5), width=7.5, height=4.2,
                  axis_config=dict(stroke_color=SOFT, include_ticks=False)).move_to([-2.0, -0.2, 0])
        xl = T("model scale / training →", 26, SOFT).next_to(ax, DOWN, buff=0.15)
        yl = T("size", 26, SOFT).next_to(ax.y_axis, UP, buff=0.1)
        bias = graph(ax, lambda x: 0.85 * np.exp(-0.32 * x) + 0.04, (0, 10), C_BIAS, 5)
        noise = graph(ax, lambda x: 0.72 + 0 * x, (0, 10), C_RES, 5)
        lb = T("NTP error (bias)", 28, C_BIAS).next_to(ax.c2p(6.2, 0.2), UP, buff=0.1)
        ln = T("one-hot noise", 28, C_RES).next_to(ax.c2p(7.5, 0.72), UP, buff=0.1)
        self.play(ShowCreation(ax), FadeIn(xl), FadeIn(yl))
        self.play(ShowCreation(noise), FadeIn(ln))
        self.say("sampling noise: fixed by the data", C_RES)
        self.beat()
        self.play(ShowCreation(bias), FadeIn(lb))
        self.say("the teacher's error shrinks", C_BIAS)
        self.beat()
        r1 = VGroup(T("standard MTP", 36, INK), T("no bias, full noise", 30, C_RES)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        r2 = VGroup(T("Soft-MTP", 36, ACCENT), T("small bias, no noise", 30, C_BIAS)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        VGroup(r1, r2).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to([4.7, 0.3, 0])
        self.play(FadeIn(r1), FadeIn(r2))
        self.say("the trade gets better with scale", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ speculative decoding wants p ≈ q
    def drafts(self):
        top = VGroup(T("MTP head drafts a token", 30, C_MTP), T("→", 30, SOFT),
                     T("NTP head verifies it", 30, C_NTP)).arrange(RIGHT, buff=0.3).move_to(2.9 * UP)
        self.play(FadeIn(top))
        self.say("drafts must agree with the NTP head")
        self.beat()
        bp = bars(toy.p, C_MTP, h=2.0, w=0.62, size=24).move_to([-2.6, 0.0, 0])
        bq = bars(toy.q, C_NTP, h=2.0, w=0.62, size=24).move_to([2.6, 0.0, 0]).align_to(bp, DOWN)
        lp = M(r"p^{(1)}_3\ \text{(draft)}", 36).next_to(bp, DOWN, buff=0.2)
        lq = M(r"q^{(1)}_3\ \text{(verifier)}", 36).next_to(bq, DOWN, buff=0.2)
        self.play(FadeIn(bp), FadeIn(lp), FadeIn(bq), FadeIn(lq))
        self.beat()
        # a few KL steps on h: p slides toward q
        h = toy.H_MTP.copy()
        for _ in range(3):
            for _ in range(25):
                h = h - 0.4 * toy.W.T @ (toy.softmax(toy.W @ h) - toy.q)
            nb = bars(toy.softmax(toy.W @ h), C_MTP, h=2.0, w=0.62, size=24).move_to(bp).align_to(bp, DOWN)
            self.play(Transform(bp, nb), run_time=0.7)
        e = M(r"\text{descent on } g_{\text{KL}} \Rightarrow p^{(k)}_t \to q^{(k)}_t", 40).next_to(VGroup(lp, lq), DOWN, buff=0.45)
        self.play(FadeIn(e))
        self.say("more drafts accepted: the bias helps", C_KL)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ KL at W has a shortcut; CE doesn't
    def shortcut(self):
        pl = Plane(center=[-3.8, -0.3, 0], unit=1.2, lim=2.5)
        rows = token_rows(pl, 26)
        hm = pl.arrow((0, 0), toy.H_MTP, C_MTP, thick=5)
        hn = pl.arrow((0, 0), toy.H_NTP, C_NTP, thick=5)
        dvec = toy.H_MTP - toy.H_NTP
        dd = pl.arrow(toy.H_NTP, toy.H_MTP, GOLD, thick=4)
        self.play(FadeIn(pl), FadeIn(rows), GrowArrow(hm), GrowArrow(hn))
        e1 = M(r"\text{KL depends on } W_\theta\, \delta", 42).move_to([3.4, 2.9, 0])
        ld = M(r"\delta = h^{(1)}_3 - h^{(0)}_4", 38, GOLD).next_to(e1, DOWN, buff=0.25)
        self.play(GrowArrow(dd), FadeIn(ld), Write(e1))
        self.say("p and q differ only through Wδ")
        self.beat()

        u = dvec / np.linalg.norm(dvec)
        Wc = toy.W - np.outer(toy.W @ u, u)          # rows with the δ direction removed
        bp = bars(toy.softmax(toy.W @ toy.H_MTP), C_MTP, h=1.6, w=0.5, size=22).move_to([1.9, -0.6, 0])
        bq = bars(toy.softmax(toy.W @ toy.H_NTP), C_NTP, h=1.6, w=0.5, size=22).move_to([4.9, 0.0, 0]).align_to(bp, DOWN)
        lp = M(r"p", 34, C_MTP).next_to(bp, DOWN, buff=0.1)
        lq = M(r"q", 34, C_NTP).next_to(bq, DOWN, buff=0.1)
        self.play(FadeIn(bp), FadeIn(bq), FadeIn(lp), FadeIn(lq))
        self.say("KL-MTP's gradient can also move W", BAD)
        self.beat()
        new_rows = VGroup()
        for i, w in enumerate(Wc):
            a = arrow(pl.p((0, 0)), pl.p(w), C_W, thick=3.0, buff=0, tip=0.15).set_opacity(0.85)
            new_rows.add(VGroup(a, VectorizedPoint(pl.p(w))))  # labels pile up on one line: drop them
        nbp = bars(toy.softmax(Wc @ toy.H_MTP), C_MTP, h=1.6, w=0.5, size=22).move_to(bp).align_to(bp, DOWN)
        nbq = bars(toy.softmax(Wc @ toy.H_NTP), C_NTP, h=1.6, w=0.5, size=22).move_to(bq).align_to(bq, DOWN)
        self.play(Transform(rows, new_rows), Transform(bp, nbp), Transform(bq, nbq), run_time=1.6)
        e2 = M(r"W_\theta \delta = 0 \Rightarrow D_{\text{KL}} = 0", 40, BAD).next_to(ld, DOWN, buff=0.25)
        coll = T("rows collapse onto one line", 26, C_W).move_to(pl.p((-0.4, 2.3)))
        self.play(FadeIn(e2), FadeIn(coll))
        self.say("shortcut: blind W, KL = 0, h unchanged", BAD)
        self.beat()
        e3 = VGroup(T("CE target x does not depend on W", 28, C_X),
                    T("→ Soft-MTP keeps CE at W", 28, ACCENT)).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        e3.move_to([3.4, -2.5, 0])
        self.play(FadeIn(e3))
        self.say("no shortcut: W stays fitted to data", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ everything on one slide
    def recap(self):
        lines = [
            (r"g_{\text{hard}} = W_\theta^\top\big(p^{(k)}_t - x_{t+k+1}\big)", "standard MTP sends this"),
            (r"= g_{\text{KL}} + g_{\text{res}}", "exact split by ± q"),
            (r"g_{\text{KL}} = \nabla D_{\text{KL}}\big(\mathrm{sg}(q^{(k)}_t)\,\Vert\,p^{(k)}_t\big)", "self-distillation: keep"),
            (r"g_{\text{res}} = \nabla_{h^{(0)}_{t+k}} \mathcal{L}_{\text{NTP}}", "noisy, NTP's job: remove"),
            (r"\tilde h_{t+k} = h^{(0)}_{t+k} + s_k\big[\mathrm{sg}(h^{(k)}_t) - h^{(k)}_t\big]", "same forward, routes −g_res"),
            (r"\text{trunk} \leftarrow g_{\text{KL}},\quad W_\theta \leftarrow \mathrm{CE}", "Soft-MTP"),
        ]
        g = VGroup()
        for tex, words in lines:
            g.add(VGroup(M(tex, 38), T(words, 26, DIM)))
        for i, row in enumerate(g):
            row[0].move_to([-6.4 + row[0].get_width() / 2, 3.0 - i * 1.02, 0])
            row[1].move_to([3.2 + row[1].get_width() / 2, row[0].get_y(), 0])
        for row in g:
            self.play(Write(row[0]), FadeIn(row[1]), run_time=0.9)
        box = SurroundingRectangle(g[-1], stroke_color=ACCENT, buff=0.12)
        self.play(ShowCreation(box))
        self.say("no extra params, no extra logits", ACCENT)
        self.beat()
        self.clear_all()


split_sections(L4_Why, globals())
