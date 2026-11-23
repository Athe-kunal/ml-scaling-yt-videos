"""Lesson 6 · Connecting audio features to an LLM (compress → project → splice → causal LM)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

# our 4 s melody through a Qwen2-Audio-style encoder
MEL_FRAMES = len(dsp.melody()) // dsp.HOP_LENGTH            # 400 real mel frames
AFTER_CONV = (MEL_FRAMES - 1) // 2 + 1                       # 200 (conv2, stride 2)
N_AUDIO = (AFTER_CONV - 2) // 2 + 1                          # 100 (avg-pool, stride 2)
D_ENC, D_LLM = 1280, 4096


def token_col(values, cell=0.3, colors=None):
    """One token's vector as a vertical stack of coloured cells."""
    v = np.asarray(values, np.float64)
    cols = colors or [rgb_to_hex(c) for c in colormap((v + 1) / 2)]
    return VGroup(*[Square(cell, stroke_color=BG, stroke_width=1.5, fill_color=c, fill_opacity=1) for c in cols]).arrange(DOWN, buff=0)


class L6_AudioToLLM(Lesson):
    NUM, TITLE, SUB = 6, "Audio features → an LLM", "compress · project · splice"
    SECTIONS = ["title", "two_ways", "rate", "project", "splice", "causal", "backprop"]
    TOPICS = {"two_ways": "two ways to read audio", "rate": "too many tokens: pool",
              "project": "project to the LLM width", "splice": "fill the <|AUDIO|> slots",
              "causal": "a causal LLM reads it", "backprop": "text loss trains the encoder"}

    # ------------------------------------------------------------ cross-attention vs one sequence
    def two_ways(self):
        div = DashedLine(3.3 * UP, 2.9 * DOWN, stroke_color=RULE)
        # left: Whisper keeps two streams
        lt = T("Whisper: cross-attention", 32, PINK).move_to([-3.6, 3.2, 0])
        audio = VGroup(*[Square(0.36, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.3) for _ in range(9)])
        audio.arrange(RIGHT, buff=0.08).move_to([-3.6, 1.7, 0])
        al = tag("audio [1500, D]", ACCENT, 22).next_to(audio, UP, buff=0.12)
        words = VGroup(*[chip(w, PINK) for w in ["<|en|>", " And", " so", " my"]]).arrange(RIGHT, buff=0.1).move_to([-3.6, -1.0, 0])
        wl = tag("text so far", PINK, 22).next_to(words, DOWN, buff=0.12)
        q = words[-1]
        looks = VGroup(*[Line(q.get_top(), a.get_bottom(), stroke_color=GOLD, stroke_width=1.6, stroke_opacity=0.75) for a in audio])
        qkv = M(r"Q \leftarrow \text{text}\quad K,V \leftarrow \text{audio}", 32, GOLD).move_to([-3.6, 0.35, 0]).shift(1.2 * LEFT)
        self.play(ShowCreation(div), FadeIn(lt))
        self.play(FadeIn(audio), FadeIn(al), FadeIn(words), FadeIn(wl))
        self.play(ShowCreation(looks, lag_ratio=0.05), FadeIn(qkv))
        self.say("Whisper: text looks up the audio", PINK)
        self.beat()

        # right: an audio LLM writes the audio INTO the token sequence
        rt = T("audio LLM: one sequence", 32, BLUE).move_to([3.6, 3.2, 0])
        seq = VGroup(chip("user", PINK), *[chip("▦", ACCENT, min_w=0.42) for _ in range(4)], chip(" what's", PINK), chip(" this?", PINK))
        seq.arrange(RIGHT, buff=0.07).move_to([3.6, 0.75, 0])
        llm = RoundedRectangle(width=seq.get_width() + 0.6, height=2.7, corner_radius=0.15, stroke_color=BLUE,
                               fill_color=BLUE, fill_opacity=0.06).move_to([3.6, 0.6, 0])
        ll = T("causal LLM", 28, BLUE).next_to(llm.get_bottom(), UP, buff=0.18)
        brace = Brace(seq[1:5], UP, buff=0.08)
        bl = tag("audio rows", ACCENT, 22).next_to(brace, UP, buff=0.06)
        nxt = VGroup(Arrow(llm.get_bottom(), llm.get_bottom() + 0.9 * DOWN, buff=0.05, stroke_color=BLUE), )
        nt = chip(" a melody", PINK).next_to(nxt, DOWN, buff=0.1)
        self.play(FadeIn(rt), FadeIn(llm), FadeIn(ll))
        self.play(LaggedStartMap(FadeIn, seq, shift=0.2 * DOWN, lag_ratio=0.1), GrowFromCenter(brace), FadeIn(bl))
        self.play(GrowArrow(nxt[0]), FadeIn(nt))
        self.say("audio LLM: audio becomes tokens", BLUE)
        self.beat()
        why = T("reuse an LLM that already knows language", 30, GOLD).to_edge(UP, buff=0.15)
        self.play(FadeOut(lt), FadeOut(rt), FadeIn(why))
        self.say("no decoder to train from scratch")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ 50 tokens/s is a lot → average pairs
    def rate(self):
        # one second of audio vs one second of text
        line = NumberLine(x_range=(0, 1, 0.25), width=11, stroke_color=RULE, include_ticks=False).shift(2.2 * UP)
        ticks = VGroup(*[Line(line.n2p(i / 50) + 0.18 * DOWN, line.n2p(i / 50) + 0.18 * UP, stroke_color=ACCENT, stroke_width=2)
                         for i in range(50)])
        l1 = VGroup(T("1 s of audio", 28, INK), tag("= 50 encoder tokens (20 ms)", ACCENT, 24)).arrange(RIGHT, buff=0.3)
        l1.next_to(line, UP, buff=0.35)
        words = VGroup(*[chip(w, PINK, 22) for w in [" ask", " not", " what"]]).arrange(RIGHT, buff=1.6).next_to(line, DOWN, buff=0.55)
        l2 = tag("≈ 3 text tokens", PINK, 24).next_to(words, DOWN, buff=0.15)
        self.play(ShowCreation(line), FadeIn(l1))
        self.play(LaggedStartMap(ShowCreation, ticks, lag_ratio=0.03), run_time=1.5)
        self.play(FadeIn(words), FadeIn(l2))
        self.say("audio is far denser than text")
        self.beat()
        cost = VGroup(tag("30 s → 1500 tokens", ACCENT, 30), M(r"\text{attention cost} \propto n^2", 40, GOLD)).arrange(RIGHT, buff=1.0)
        cost.move_to(1.9 * DOWN)
        self.play(FadeIn(cost))
        self.say("so shorten the sequence first", GOLD)
        self.beat()
        self.fade_all(keep=(self.caption,))

        # average neighbouring tokens: each token is a column vector
        rng = np.random.default_rng(3)
        vals = np.tanh(np.cumsum(rng.normal(0, 0.6, (8, 6)), axis=0) * 0.5)  # smooth-ish in time
        cols = VGroup(*[token_col(v) for v in vals]).arrange(RIGHT, buff=0.18).move_to(1.25 * UP)
        tl = tag("[1500, 1280] · 20 ms each", ACCENT, 26).next_to(cols, UP, buff=0.25)
        pairs = VGroup(*[SurroundingRectangle(VGroup(cols[2 * i], cols[2 * i + 1]), stroke_color=GOLD, buff=0.06) for i in range(4)])
        pooled = VGroup(*[token_col((vals[2 * i] + vals[2 * i + 1]) / 2) for i in range(4)]).arrange(RIGHT, buff=0.66).move_to(1.6 * DOWN)
        pl = tag("[750, 1280] · 40 ms each", GOLD, 26).next_to(pooled, DOWN, buff=0.25)
        f = M(r"h'_i = \tfrac{1}{2}\,(h_{2i} + h_{2i+1})", 42, GOLD).next_to(pooled, RIGHT, buff=0.8)
        self.play(FadeIn(cols, lag_ratio=0.05), FadeIn(tl))
        self.play(ShowCreation(pairs, lag_ratio=0.2))
        self.say("average pooling, stride 2")
        self.beat()
        self.play(*[TransformFromCopy(VGroup(cols[2 * i], cols[2 * i + 1]), pooled[i]) for i in range(4)], run_time=1.5)
        self.play(FadeIn(pl), Write(f))
        self.say("half the tokens: 1500 → 750", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ LN → Linear: encoder width → LLM width
    def project(self):
        # widths drawn to scale: 1280 vs 4096
        u = 3.4 / D_LLM
        X = Rectangle(width=D_ENC * u, height=2.2, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.2)
        W = Rectangle(width=D_LLM * u, height=D_ENC * u, stroke_color=GOLD, fill_color=GOLD, fill_opacity=0.15)
        Y = Rectangle(width=D_LLM * u, height=2.2, stroke_color=BLUE, fill_color=BLUE, fill_opacity=0.2)
        at, eq = M("@", 52), M("=", 52)
        row = VGroup(X, at, W, eq, Y).arrange(RIGHT, buff=0.4).shift(0.4 * UP)
        W.align_to(X, UP)
        tx = tag("[N, 1280]", ACCENT, 26).next_to(X, DOWN, buff=0.15)
        tw = tag("W [1280, 4096]", GOLD, 26).next_to(W, DOWN, buff=0.15)
        ty = tag("[N, 4096]", BLUE, 26).next_to(Y, DOWN, buff=0.15)
        nx = T("audio rows", 26, ACCENT).next_to(X, UP, buff=0.15)
        ny = T("LLM-sized rows", 26, BLUE).next_to(Y, UP, buff=0.15)
        head = VGroup(T("encoder width", 28, ACCENT), tag("1280", ACCENT, 30), T("≠", 32, BAD),
                      T("LLM width", 28, BLUE), tag("4096", BLUE, 30)).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.4)
        self.play(FadeIn(head))
        self.play(FadeIn(X), FadeIn(tx), FadeIn(nx))
        self.say("widths don't match")
        self.beat()
        self.play(FadeIn(at), FadeIn(W), FadeIn(tw))
        k1 = SurroundingRectangle(tx[3:7], stroke_color=GOLD, buff=0.04)
        k2 = SurroundingRectangle(tw[2:6], stroke_color=GOLD, buff=0.04)
        self.play(FadeIn(eq), FadeIn(Y), FadeIn(ty), FadeIn(ny), ShowCreation(k1), ShowCreation(k2))
        ln = tag("LayerNorm → Linear (one layer)", DIM, 24).next_to(VGroup(tx, ty), DOWN, buff=0.45)
        self.play(FadeIn(ln))
        self.say("a projector: 1280 → 4096", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ placeholders in the prompt, filled in order
    def splice(self):
        words = ["user", "<|audio_bos|>", "<|AUDIO|>", "<|audio_eos|>", " what's", " this?"]
        cols = [PINK, ACCENT, ACCENT, ACCENT, PINK, PINK]
        row = VGroup(*[chip(w, c, 24, h=0.6) for w, c in zip(words, cols)]).arrange(RIGHT, buff=0.1).move_to(2.2 * UP)
        self.play(LaggedStartMap(FadeIn, row, shift=0.15 * DOWN, lag_ratio=0.08))
        self.say("the prompt has one audio placeholder")
        self.beat()

        # expand <|AUDIO|> into N copies (draw 6 + dots)
        k = 4
        reps = VGroup(*[chip("<|AUDIO|>", ACCENT, 20, h=0.6) for _ in range(k)])
        dots = T("…", 30, ACCENT)
        new = VGroup(row[0].copy(), row[1].copy(), *reps, dots, row[3].copy(), row[4].copy(), row[5].copy())
        new.arrange(RIGHT, buff=0.08).move_to(2.2 * UP)
        if new.get_width() > 13.4:
            new.set_width(13.4)
        cnt = tag(f"× {N_AUDIO}  (our 4 s clip)", ACCENT, 24).next_to(VGroup(*reps, dots), UP, buff=0.1)
        self.play(ReplacementTransform(row[:2], new[:2]), ReplacementTransform(row[3:], new[2 + k + 1:]),
                  ReplacementTransform(row[2], VGroup(*reps, dots)), run_time=1.2)
        self.play(FadeIn(cnt))
        self.say(f"one slot per audio token: {N_AUDIO}")
        self.beat()

        # embedding lookup: one column per token; placeholders get a dummy embedding
        rng = np.random.default_rng(5)
        emb = VGroup()
        for c in new:
            if isinstance(c, Text) or c is dots:
                emb.add(T("…", 30, DIM).move_to([c.get_x(), 0.3, 0]))
                continue
            is_slot = any(c is r for r in reps)
            colors = [DIM] * 4 if is_slot else [rgb_to_hex(x) for x in colormap(rng.uniform(0.15, 0.8, 4))]
            emb.add(token_col(np.zeros(4), cell=0.32, colors=colors).move_to([c.get_x(), 0.3, 0]))
        el = tag("embed(ids)  [S, 4096]", INK, 24).next_to(emb, LEFT, buff=0.3).shift(0.0 * UP)
        if el.get_left()[0] < -6.9:
            el.next_to(emb, DOWN, buff=0.2)
        self.play(LaggedStartMap(FadeIn, emb, shift=0.2 * DOWN, lag_ratio=0.04), FadeIn(el))
        self.say("every id → an embedding row")
        self.beat()

        # audio features arrive and overwrite the placeholder rows, in order
        slots = [e for e, c in zip(emb, new) if any(c is r for r in reps)]
        audio = VGroup(*[token_col(np.tanh(rng.normal(0, 1, 4)), cell=0.32) for _ in range(k)]).arrange(RIGHT, buff=0.4)
        audio.move_to(2.0 * DOWN)
        al = tag(f"audio features  [{N_AUDIO}, 4096]", ACCENT, 24).next_to(audio, DOWN, buff=0.15)
        self.play(FadeIn(audio), FadeIn(al))
        self.play(LaggedStart(*[Transform(s, a.copy().move_to(s)) for s, a in zip(slots, audio)], lag_ratio=0.12),
                  LaggedStart(*[a.animate.move_to(s).set_opacity(0) for s, a in zip(slots, audio)], lag_ratio=0.12), run_time=1.8)
        self.remove(audio)
        rule = tag(f"{N_AUDIO} slots = {N_AUDIO} rows, else error", GOLD, 24).move_to(al)
        self.play(FadeTransform(al, rule))
        self.say("masked_scatter: fill slots in order", ACCENT)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ causal attention over text + audio
    def causal(self):
        labels = ["user", "▦", "▦", "▦", "▦", " what's", " this?", " a", " melody"]
        kinds = ["t", "a", "a", "a", "a", "t", "t", "o", "o"]
        col = {"t": PINK, "a": ACCENT, "o": GOLD}
        n = len(labels)
        cell = 0.46
        grid = VGroup()
        for i in range(n):
            for j in range(n):
                on = j <= i
                c = col[kinds[j]] if on else BG
                sq = Square(cell, stroke_color=RULE, stroke_width=1, fill_color=c, fill_opacity=0.55 if on else 1)
                sq.move_to([(j - n / 2) * cell, (n / 2 - i) * cell, 0])
                grid.add(sq)
        grid.move_to(0.2 * UP + 1.2 * RIGHT)
        top = VGroup(*[Square(cell * 0.7, stroke_width=0, fill_color=col[k], fill_opacity=0.9).next_to(grid[j], UP, buff=0.12)
                       for j, k in enumerate(kinds)])
        legend = VGroup(*[VGroup(Square(0.25, stroke_width=0, fill_color=c, fill_opacity=0.9), T(t, 24, c)).arrange(RIGHT, buff=0.15)
                          for c, t in [(PINK, "prompt text"), (ACCENT, "audio rows"), (GOLD, "answer")]])
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(UR, buff=0.5).shift(0.3 * DOWN)
        left = VGroup(*[chip(l, col[k], 15, h=0.38, min_w=0.42) for l, k in zip(labels, kinds)])
        for i, c in enumerate(left):
            c.next_to(grid[i * n], LEFT, buff=0.12)
        kl = tag("keys →", DIM, 22).next_to(top, UP, buff=0.1)
        ql = tag("queries ↓", DIM, 22).next_to(left, UP, buff=0.12).align_to(left, RIGHT)
        self.play(FadeIn(grid, lag_ratio=0.005), FadeIn(top), FadeIn(left), FadeIn(kl), FadeIn(ql), FadeIn(legend), run_time=1.5)
        self.say("causal: each token sees its past")
        self.beat()
        row = n - 2  # " a"
        hl = SurroundingRectangle(VGroup(*grid[row * n: row * n + row + 1]), stroke_color=GOLD, buff=0.02, stroke_width=3)
        aud = SurroundingRectangle(VGroup(*[grid[row * n + j] for j in range(1, 5)]), stroke_color=ACCENT, buff=0.0, stroke_width=4)
        self.play(ShowCreation(hl))
        self.play(ShowCreation(aud))
        self.say("the answer attends to the audio rows", ACCENT)
        self.beat()
        nt = VGroup(Arrow(LEFT, RIGHT, stroke_color=GOLD), chip(" melody", GOLD, 22)).arrange(RIGHT, buff=0.15)
        nt.next_to(grid, RIGHT, buff=0.3).match_y(grid[row * n])
        if nt.get_right()[0] > 6.9:
            nt.shift((nt.get_right()[0] - 6.9) * LEFT)
        self.play(FadeIn(nt, shift=0.2 * RIGHT))
        self.say("predict the next text token", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the text loss trains the audio path
    def backprop(self):
        spec = [("log-mel", GOLD, "[128, 3000]"), ("encoder", ACCENT, "[1500, 1280]"), ("pool · LN", GREEN, "[750, 1280]"),
                ("projector", GREEN, "[N, 4096]"), ("LLM", BLUE, "causal")]
        bl = VGroup(*[block(n, 2.2, 1.0, c, 26, sub=s) for n, c, s in spec]).arrange(RIGHT, buff=0.42).shift(0.5 * UP)
        fw = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.05, stroke_color=RULE) for a, b in zip(bl[:-1], bl[1:])])
        self.play(LaggedStartMap(FadeIn, bl, lag_ratio=0.15), LaggedStartMap(GrowArrow, fw), run_time=1.5)
        obj = M(r"\max\ P_\theta\big(x_t \mid x_{<t},\ \mathrm{Enc}_\phi(a)\big)", 44).to_edge(UP, buff=0.45)
        self.play(Write(obj))
        self.say("objective: the next text token")
        self.beat()
        loss = M(r"\mathcal{L} = -\sum_t \log P(x_t \mid \dots)", 38, BAD).next_to(bl[-1], DOWN, buff=0.9).shift(0.6 * LEFT)
        back = VGroup(*[Arrow(b.get_bottom() + 0.15 * DOWN, a.get_bottom() + 0.15 * DOWN, buff=0.05,
                              stroke_color=BAD, stroke_width=4, path_arc=0.7) for a, b in zip(bl[1:-1], bl[2:])])
        self.play(Write(loss))
        self.play(LaggedStart(*[ShowCreation(a) for a in reversed(back)], lag_ratio=0.4), run_time=2)
        self.play(*[Indicate(b, color=BAD) for b in bl[1:4]])
        self.say("gradients flow back to the encoder", BAD)
        self.beat()
        new = VGroup(T("new: pool · projector", 28, GREEN), T("from Whisper: encoder", 28, ACCENT),
                     T("from a text LLM: LLM", 28, BLUE)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        new.to_corner(DL, buff=0.6).shift(0.6 * UP)
        self.play(FadeIn(new, lag_ratio=0.2))
        self.say("old parts, new glue")
        self.beat()
        self.clear_all()


split_sections(L6_AudioToLLM, globals())
