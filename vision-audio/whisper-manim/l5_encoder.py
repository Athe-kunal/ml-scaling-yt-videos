"""Lesson 5 · Whisper's audio encoder (+ a brief look at the decoder that trains it)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

D, H, L = dsp.D_MODEL, dsp.N_HEAD, dsp.N_ENC_LAYERS
C1 = 420  # visible frames (4.2 s of the 30 s)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def conv1d(x, w, stride=1):
    """x [Cin, T], w [Cout, Cin, 3], padding 1 → [Cout, T // stride]."""
    xp = np.pad(x, ((0, 0), (1, 1)))
    T = x.shape[1]
    out = sum(np.einsum("oc,ct->ot", w[:, :, j], xp[:, j:j + T]) for j in range(3))
    return out[:, ::stride]


def sinusoids(length, channels, max_timescale=10000):
    """whisper.model.sinusoids — the encoder's fixed positions."""
    inc = np.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-inc * np.arange(channels // 2))
    t = np.arange(length)[:, None] * inv[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], axis=1)


def toy_stem(mel):
    """Real shapes, fixed random weights: pictures come from the actual log-mel."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 1 / np.sqrt(80 * 3), (D, 80, 3))
    w2 = rng.normal(0, 1 / np.sqrt(D * 3), (D, D, 3))
    h1 = gelu(conv1d(mel, w1))
    h2 = gelu(conv1d(h1, w2, stride=2))
    return h1, h2


class L5_Encoder(Lesson):
    NUM, TITLE, SUB = 5, "Whisper's audio encoder", "log-mel → 1500 audio tokens"
    SECTIONS = ["title", "overview", "conv1", "conv2", "positions", "blocks", "decoder", "training", "summary"]

    def prepare(self):
        self.mel = dsp.whisper_input()["scaled"]
        self.h1, self.h2 = toy_stem(self.mel)

    # ------------------------------------------------------------ the whole model in one picture
    def overview(self):
        enc = block("encoder", 3.6, 1.2, ACCENT, 36, sub=f"{L} blocks · D {D}")
        dec = block("decoder", 3.6, 1.2, PINK, 36, sub=f"{dsp.N_DEC_LAYERS} blocks · D {D}")
        VGroup(enc, dec).arrange(RIGHT, buff=2.4).shift(0.3 * UP)
        mel = tag("log-mel [80, 3000]", GOLD, 28).next_to(enc, DOWN, buff=0.9)
        txt = VGroup(tag("<|startoftranscript|>", DIM, 22), tag("<|en|><|transcribe|>", DIM, 22)).arrange(DOWN, buff=0.08)
        txt.next_to(dec, DOWN, buff=0.9)
        out = tag("next token", PINK, 28).next_to(dec, UP, buff=0.9)
        arrs = VGroup(Arrow(mel.get_top(), enc.get_bottom(), buff=0.12, stroke_color=GOLD),
                      Arrow(txt.get_top(), dec.get_bottom(), buff=0.12, stroke_color=DIM),
                      Arrow(enc.get_right(), dec.get_left(), buff=0.12, stroke_color=ACCENT),
                      Arrow(dec.get_top(), out.get_bottom(), buff=0.12, stroke_color=PINK))
        kv = tag("[1500, 384]", ACCENT, 24).next_to(arrs[2], UP, buff=0.1)
        self.play(FadeIn(enc), FadeIn(mel), GrowArrow(arrs[0]))
        self.say("encoder: sound → features")
        self.beat()
        self.play(GrowArrow(arrs[2]), FadeIn(kv), FadeIn(dec), FadeIn(txt), GrowArrow(arrs[1]), GrowArrow(arrs[3]), FadeIn(out))
        self.say("decoder: features → text")
        self.beat()
        foc = SurroundingRectangle(VGroup(enc, mel), stroke_color=ACCENT, buff=0.25)
        self.play(ShowCreation(foc), VGroup(dec, txt, out, arrs[1], arrs[3]).animate.set_opacity(0.25))
        self.say("first: the encoder", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ conv1: 80 → 384, kernel 3
    def conv1(self):
        mi = heat(self.mel[:, :C1], "l5_mel", 5.4, 2.4, vmin=-1, vmax=self.mel.max())
        h1 = self.h1[:, :C1]
        oi = heat(h1, "l5_h1", 5.4, 3.6, vmin=0, vmax=np.percentile(h1, 99))
        Group(mi, oi).arrange(RIGHT, buff=1.6).shift(0.3 * UP)
        mb = SurroundingRectangle(mi, buff=0, stroke_color=RULE, stroke_width=1.5)
        ob = SurroundingRectangle(oi, buff=0, stroke_color=RULE, stroke_width=1.5)
        t_in = tag("[80, 3000]", GOLD, 28).next_to(mi, DOWN, buff=0.2)
        t_out = VGroup(tag("[384, 3000]", ACCENT, 28), tag("toy weights", DIM, 20)).arrange(RIGHT, buff=0.3).next_to(oi, DOWN, buff=0.2)
        op = VGroup(T("Conv1d", 32, ACCENT), tag("80 → 384 · k 3 · GELU", DIM, 24)).arrange(DOWN, buff=0.1).to_edge(UP, buff=0.4)
        self.play(FadeIn(mi), ShowCreation(mb), FadeIn(t_in), FadeIn(op))
        self.beat()

        colw = mi.get_width() / C1
        k = ValueTracker(20)
        kern = always_redraw(lambda: Rectangle(width=max(3 * colw, 0.08), height=mi.get_height(), stroke_color=PINK, stroke_width=3)
                             .move_to([mi.get_left()[0] + (k.get_value() + 0.5) * colw, mi.get_center()[1], 0]))
        ocol = always_redraw(lambda: Rectangle(width=max(colw * oi.get_width() / mi.get_width(), 0.05), height=oi.get_height(),
                                               stroke_color=PINK, stroke_width=3)
                             .move_to([oi.get_left()[0] + (k.get_value() + 0.5) * oi.get_width() / C1, oi.get_center()[1], 0]))
        link = always_redraw(lambda: Line(kern.get_right(), ocol.get_left(), stroke_color=PINK, stroke_width=2))
        cover = Rectangle(width=oi.get_width(), height=oi.get_height(), stroke_width=0, fill_color=BG, fill_opacity=1).move_to(oi)
        self.add(oi, cover, ob)
        self.play(ShowCreation(kern), FadeIn(ocol), ShowCreation(link), FadeIn(t_out))
        self.say("3 frames × 80 mels → 384 numbers")
        self.beat()
        cover.add_updater(lambda c: c.become(
            Rectangle(width=max(oi.get_right()[0] - (oi.get_left()[0] + (k.get_value() + 1) * oi.get_width() / C1), 0.001),
                      height=oi.get_height(), stroke_width=0, fill_color=BG, fill_opacity=1).align_to(oi, RIGHT).match_y(oi)))
        self.play(k.animate.set_value(C1 - 2), run_time=4, rate_func=linear)
        cover.clear_updaters()
        self.remove(cover)
        self.say("stride 1: still 3000 frames")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ conv2: stride 2, 3000 → 1500
    def conv2(self):
        n = 16
        cw = 0.6
        top = VGroup(*[Square(cw, stroke_color=ACCENT, stroke_width=2, fill_color=ACCENT, fill_opacity=0.15) for _ in range(n)])
        top.arrange(RIGHT, buff=0.08).shift(1.4 * UP)
        bot = VGroup(*[Square(cw, stroke_color=GOLD, stroke_width=2, fill_color=GOLD, fill_opacity=0.25) for _ in range(n // 2)])
        for i, b in enumerate(bot):
            b.move_to([top[2 * i].get_x(), -1.0, 0])
        lt = tag("[384, 3000]  · 10 ms each", ACCENT, 26).next_to(top, UP, buff=0.25)
        lb = tag("[384, 1500]  · 20 ms each", GOLD, 26).next_to(bot, DOWN, buff=0.3).align_to(lt, LEFT)
        op = VGroup(T("Conv1d · stride 2", 32, GOLD), tag("384 → 384 · k 3 · GELU", DIM, 24)).arrange(DOWN, buff=0.1).to_edge(UP, buff=0.25)
        self.play(FadeIn(op), LaggedStartMap(FadeIn, top, lag_ratio=0.03), FadeIn(lt))
        win = SurroundingRectangle(VGroup(*top[0:2]), stroke_color=PINK, buff=0.05)
        win.stretch_to_fit_width(cw * 3 + 0.16 + 0.1).align_to(top[0], LEFT).shift(0.05 * LEFT + (cw + 0.08) * LEFT)
        self.play(ShowCreation(win))
        for i in range(n // 2):
            tgt = win.copy().move_to(top[2 * i])
            ln = Line(tgt.get_bottom(), bot[i].get_top(), stroke_color=PINK, stroke_width=2)
            self.play(win.animate.move_to(top[2 * i]), FadeIn(bot[i]), ShowCreation(ln), run_time=0.35 if i else 0.6)
        self.play(FadeIn(lb))
        self.say("jump 2 frames: half as many", GOLD)
        self.beat()
        eq = M(r"3000 / 2 = 1500 \text{ tokens} = 30\,\text{s} \,/\, 20\,\text{ms}", 46, GOLD).shift(2.3 * DOWN)
        self.play(FadeOut(lb), Write(eq))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ transpose + sinusoids
    def positions(self):
        h2 = self.h2.T  # [1500, 384]
        vis = 200
        xi = heat(h2[:vis], "l5_x", 3.2, 4.4, vmin=0, vmax=np.percentile(h2, 99), flip=False)
        pe = sinusoids(dsp.N_AUDIO_CTX, D)
        pi_ = heat(pe[:vis], "l5_pe", 3.2, 4.4, vmin=-1, vmax=1, flip=False)
        s = h2[:vis] + pe[:vis]
        si = heat(s, "l5_sum", 3.2, 4.4, flip=False)
        plus, eq = M("+", 64), M("=", 64)
        row = Group(xi, plus, pi_, eq, si).arrange(RIGHT, buff=0.45).shift(0.2 * UP)
        boxes = VGroup(*[SurroundingRectangle(m, buff=0, stroke_color=RULE, stroke_width=1.5) for m in (xi, pi_, si)])
        tags = VGroup(tag("x [1500, 384]", GOLD, 24).next_to(xi, DOWN, buff=0.15),
                      tag("PE [1500, 384]", PINK, 24).next_to(pi_, DOWN, buff=0.15),
                      tag("[1500, 384]", ACCENT, 24).next_to(si, DOWN, buff=0.15))
        rows = T("rows = tokens (time)", 26, DIM).rotate(PI / 2).next_to(xi, LEFT, buff=0.2)
        self.play(FadeIn(xi), ShowCreation(boxes[0]), FadeIn(tags[0]), FadeIn(rows))
        self.say("transpose: one row per 20 ms")
        self.beat()
        f = M(r"PE_{t} = \big[\sin(t\,\omega_i),\ \cos(t\,\omega_i)\big],\quad \omega_i = 10000^{-i/191}", 40, PINK)
        f.to_edge(UP, buff=0.3)
        self.play(FadeIn(plus), FadeIn(pi_), ShowCreation(boxes[1]), FadeIn(tags[1]), Write(f))
        self.say("fixed sinusoids: where in time", PINK)
        self.beat()
        self.play(FadeIn(eq), FadeIn(si), ShowCreation(boxes[2]), FadeIn(tags[2]))
        self.say("no learned positions in the encoder")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the 4 Transformer blocks
    def blocks(self):
        ln1 = block("LayerNorm", 3.6, 0.7, DIM, 26)
        att = block("self-attention", 3.6, 0.95, ACCENT, 30, sub=f"{H} heads · no mask")
        ln2 = block("LayerNorm", 3.6, 0.7, DIM, 26)
        mlp = block("MLP", 3.6, 0.95, GOLD, 30, sub=f"{D} → {4 * D} → {D} · GELU")
        stack = VGroup(ln1, att, ln2, mlp).arrange(UP, buff=0.35).shift(3.4 * LEFT + 0.25 * DOWN)
        arrows = VGroup(*[Arrow(a.get_top(), b.get_bottom(), buff=0.04, stroke_color=RULE, stroke_width=2)
                          for a, b in zip(stack[:-1], stack[1:])])
        frame = SurroundingRectangle(stack, buff=0.3, stroke_color=ACCENT, stroke_width=2)
        times = T(f"× {L}", 48, ACCENT).next_to(frame, LEFT, buff=0.3)
        res = VGroup(*[T("+ residual", 22, DIM).next_to(m, RIGHT, buff=0.35) for m in (att, mlp)])
        io = VGroup(tag("[1500, 384]", ACCENT, 24).next_to(frame, DOWN, buff=0.15),
                    tag("[1500, 384]", ACCENT, 24).next_to(frame, UP, buff=0.15))
        self.play(LaggedStartMap(FadeIn, stack, lag_ratio=0.2), LaggedStartMap(GrowArrow, arrows), ShowCreation(frame),
                  FadeIn(times), FadeIn(res), FadeIn(io))
        self.say("pre-norm Transformer blocks")
        self.beat()

        # all-to-all attention among a few tokens (no causal mask)
        n = 10
        toks = VGroup(*[Square(0.42, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.2) for _ in range(n)])
        toks.arrange(RIGHT, buff=0.12).shift(3.0 * RIGHT + 1.2 * UP)
        toks2 = toks.copy().shift(2.6 * DOWN)
        edges = VGroup(*[Line(a.get_bottom(), b.get_top(), stroke_color=ACCENT, stroke_width=1, stroke_opacity=0.45)
                         for a in toks for b in toks2])
        lab = tag("every token ↔ all 1500", ACCENT, 26).next_to(toks, UP, buff=0.25)
        self.play(Indicate(att), FadeIn(toks), FadeIn(toks2))
        self.play(ShowCreation(edges, lag_ratio=0.005), FadeIn(lab), run_time=1.5)
        self.say("whole 30 s visible at once")
        self.beat()
        attn = M(r"\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{64}}\Big)V", 40).next_to(toks2, DOWN, buff=0.35)
        sh = tag("[1500, 1500] per head", DIM, 22).next_to(attn, DOWN, buff=0.1)
        self.play(Write(attn), FadeIn(sh))
        self.beat()
        self.play(*[FadeOut(m) for m in (toks, toks2, edges, lab, attn, sh)])
        post = block("ln_post", 3.6, 0.7, DIM, 26).next_to(io[1], UP, buff=0.3)
        self.play(FadeIn(post))
        feat = tag("audio features  [1500, 384]", ACCENT, 34).move_to(3.2 * RIGHT + 0.5 * UP)
        self.play(TransformFromCopy(post, feat))
        self.say("output: 1500 vectors of 384")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the decoder, briefly
    def decoder(self):
        audio = VGroup(*[Square(0.34, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.25) for _ in range(14)])
        audio.arrange(RIGHT, buff=0.08).shift(2.3 * UP)
        al = tag("audio features [1500, 384]", ACCENT, 24).next_to(audio, UP, buff=0.15)
        words = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", " And", " so", " my", " fellow"]
        toks = VGroup(*[VGroup(RoundedRectangle(width=max(0.9, 0.17 * len(w) + 0.3), height=0.6, corner_radius=0.08,
                                                stroke_color=PINK, fill_color=PINK, fill_opacity=0.15),
                               Text(w, font="Menlo", font_size=20, color=INK)) for w in words])
        for t in toks:
            t[1].move_to(t[0])
        toks.arrange(RIGHT, buff=0.12).shift(1.8 * DOWN)
        tl = tag("text tokens so far", PINK, 24).next_to(toks, DOWN, buff=0.2)
        self.play(FadeIn(audio), FadeIn(al), LaggedStartMap(FadeIn, toks, lag_ratio=0.1), FadeIn(tl))
        self.say("decoder reads text so far")
        self.beat()
        # masked self-attention: token i sees ≤ i
        arcs = VGroup()
        i = len(toks) - 1
        for j in range(i):
            arcs.add(ArcBetweenPoints(toks[i].get_top(), toks[j].get_top(), angle=PI / 3, stroke_color=PINK, stroke_width=2))
        self.play(ShowCreation(arcs, lag_ratio=0.1))
        self.say("masked self-attn: only the past", PINK)
        self.beat()
        self.play(FadeOut(arcs))
        cross = VGroup(*[Line(toks[i].get_top(), a.get_bottom(), stroke_color=GOLD, stroke_width=1.6, stroke_opacity=0.7) for a in audio])
        q = M(r"Q \leftarrow \text{text},\quad K, V \leftarrow \text{audio}", 40, GOLD).shift(0.3 * UP + 3.6 * LEFT)
        self.play(ShowCreation(cross, lag_ratio=0.05), FadeIn(q))
        self.say("cross-attn: text asks the audio", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ how the encoder learns: text loss
    def training(self):
        spec = [("convs", GOLD, None), ("encoder ×4", ACCENT, None), ("decoder ×4", PINK, "self + cross-attn"), ("logits", PINK, None)]
        bl = VGroup(*[block(n, 2.8, 1.05, c, 26, sub=s) for n, c, s in spec]).arrange(RIGHT, buff=0.45).shift(0.6 * UP)
        fw = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.05, stroke_color=RULE, stroke_width=2) for a, b in zip(bl[:-1], bl[1:])])
        mel = tag("log-mel", GOLD, 24).next_to(bl[0], DOWN, buff=0.35)
        shp = tag("[n, 51865]", PINK, 22).next_to(bl[-1], DOWN, buff=0.15)
        self.play(LaggedStartMap(FadeIn, bl, lag_ratio=0.15), LaggedStartMap(GrowArrow, fw), FadeIn(mel), FadeIn(shp))
        loss = M(r"\mathcal{L} = -\sum_i \log p(\text{token}_{i+1} \mid \text{audio}, \text{token}_{\le i})", 42, BAD)
        loss.to_edge(UP, buff=0.3)
        self.play(Write(loss))
        self.say("predict the next text token")
        self.beat()
        back = VGroup(*[Arrow(b.get_top() + 0.15 * UP, a.get_top() + 0.15 * UP, buff=0.05,
                              stroke_color=BAD, stroke_width=4, path_arc=0.8)
                        for a, b in zip(bl[:-1], bl[1:])])
        grad = M(r"\nabla \mathcal{L}", 40, BAD).next_to(bl[-1], UP, buff=0.5)
        self.play(FadeIn(grad), LaggedStart(*[ShowCreation(a) for a in reversed(back)], lag_ratio=0.4), run_time=2.5)
        self.play(*[Indicate(b, color=BAD) for b in bl[:2]])
        self.say("text loss trains the encoder", BAD)
        self.beat()
        data = T("680 000 h of audio + transcripts", 36, GOLD).shift(2.2 * DOWN)
        self.play(FadeIn(data))
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the whole path, with shapes
    def summary(self):
        items = [("waveform", "[480 000]", INK), ("log-mel", "[80, 3000]", GOLD), ("conv1", "[384, 3000]", ACCENT),
                 ("conv2 ↓2", "[384, 1500]", GOLD), ("+ PE · ×4 blocks", "[1500, 384]", ACCENT)]
        g = VGroup(*[VGroup(T(a, 30, c), tag(b, DIM, 24)).arrange(DOWN, buff=0.12) for a, b, c in items])
        g.arrange(RIGHT, buff=0.9).shift(0.4 * UP)
        arr = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.15, stroke_color=RULE) for a, b in zip(g[:-1], g[1:])])
        self.play(LaggedStart(*[FadeIn(x, shift=0.2 * RIGHT) for x in g], lag_ratio=0.3), LaggedStartMap(GrowArrow, arr), run_time=2.5)
        self.say("next: the code in whisper-app")
        self.beat()
        self.clear_all()


split_sections(L5_Encoder, globals())
