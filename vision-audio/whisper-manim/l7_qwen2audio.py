"""Lesson 7 · Qwen2-Audio: Whisper morphs into an audio LLM, then how it is trained."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import dsp

MEL_FRAMES = len(dsp.melody()) // dsp.HOP_LENGTH            # 400 real mel frames (4 s)
AFTER_CONV = (MEL_FRAMES - 1) // 2 + 1                       # 200
N_AUDIO = (AFTER_CONV - 2) // 2 + 1                          # 100 audio tokens
Y = 0.55                                                     # the pipeline row
X = dict(mel=-6.0, conv=-4.0, enc=-1.75, dec=2.2, pool=0.55, proj=2.75, llm=5.0)


def blk(name, sub, color, x, w=1.8, size=27):
    return block(name, w, 1.25, color, size, sub=sub).move_to([x, Y, 0])


def link(a, b, color=RULE):
    return Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_color=color, stroke_width=3)


def bar(total, real, width, color=ACCENT, h=0.42):
    """A length-`total` sequence drawn to scale: the first `real` cells are audio, the rest padding."""
    full = Rectangle(width=width, height=h, stroke_color=RULE, stroke_width=1.5, fill_color=DIM, fill_opacity=0.18)
    fill = Rectangle(width=width * real / total, height=h, stroke_width=0, fill_color=color, fill_opacity=0.85)
    fill.align_to(full, LEFT)
    return VGroup(full, fill)


class L7_Qwen2Audio(Lesson):
    NUM, TITLE, SUB = 7, "Qwen2-Audio", "Whisper, rebuilt as an audio LLM"
    SECTIONS = ["title", "morph", "token_count", "modes", "training", "compare"]
    TOPICS = {"morph": "Whisper → Qwen2-Audio, step by step", "token_count": "how many audio tokens",
              "modes": "voice chat & audio analysis", "training": "pretrain → SFT → DPO",
              "compare": "Whisper vs Qwen2-Audio"}

    # ------------------------------------------------------------ the morph, one change per beat
    def morph(self):
        head = T("Whisper tiny", 36, PINK).to_edge(UP, buff=0.45)
        mel = blk("log-mel", "[80, 3000]", GOLD, X["mel"])
        conv = blk("conv × 2", "3000 → 1500", GOLD, X["conv"])
        enc = blk("encoder × 4", "384 · 6 heads", ACCENT, X["enc"], w=2.3)
        dec = blk("decoder × 4", "self + cross", PINK, X["dec"], w=2.3)
        out = chip(" text", PINK, 24, h=0.6).move_to([X["llm"], Y, 0])
        a1, a2 = link(mel, conv), link(conv, enc)
        cross = Arrow(enc.get_right(), dec.get_left(), buff=0.06, stroke_color=PINK, stroke_width=4)
        cl = tag("cross-attn", PINK, 20).next_to(cross, UP, buff=0.08)
        a4 = link(dec, out, PINK)
        legend = VGroup(*[VGroup(Square(0.22, stroke_width=0, fill_color=c, fill_opacity=1), T(t, 24, c)).arrange(RIGHT, buff=0.12)
                          for c, t in [(GOLD, "changed"), (BAD, "removed"), (GREEN, "added")]]).arrange(RIGHT, buff=0.6)
        legend.to_edge(DOWN, buff=1.25)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(m) for m in (mel, a1, conv, a2, enc, cross, cl, dec, a4, out)], lag_ratio=0.12), run_time=2)
        self.play(FadeIn(legend))
        self.say("start: Whisper")
        self.beat()

        # 1. scale up: Whisper-large-v3 (128 mels, D 1280, 32 layers, 20 heads)
        mel2 = blk("log-mel", "[128, 3000]", GOLD, X["mel"])
        enc2 = blk("encoder × 32", "1280 · 20 heads", ACCENT, X["enc"], w=2.3)
        head2 = T("Whisper-large-v3 encoder", 36, GOLD).move_to(head)
        self.play(Transform(mel, mel2), Transform(enc, enc2), Transform(head, head2), run_time=1.2)
        self.play(Indicate(mel[2], color=GOLD), Indicate(enc[1], color=GOLD), Indicate(enc[2], color=GOLD))
        self.say("swap in the large-v3 encoder", GOLD)
        self.beat()

        # 2. remove Whisper's text decoder and its cross-attention
        gone = VGroup(cross, cl, dec, a4, out)
        self.play(gone.animate.set_color(BAD))
        x_mark = Cross(dec, stroke_color=BAD, stroke_width=6)
        self.play(ShowCreation(x_mark))
        self.play(FadeOut(VGroup(gone, x_mark), shift=0.6 * DOWN), run_time=0.8)
        self.say("drop the decoder and cross-attn", BAD)
        self.beat()

        # 3. padding mask: our 4 s clip fills only 200 of the 1500 encoder positions
        mb = bar(1500, AFTER_CONV, 2.3, ACCENT).next_to(enc, DOWN, buff=0.55)
        ml = VGroup(tag(f"{AFTER_CONV} real", ACCENT, 20), tag(f"{1500 - AFTER_CONV} padding", DIM, 20)).arrange(RIGHT, buff=0.3)
        ml.next_to(mb, DOWN, buff=0.12)
        mt = T("mask padded keys", 26, GREEN).next_to(ml, DOWN, buff=0.12)
        self.play(FadeIn(mb), FadeIn(ml))
        self.play(FadeIn(mt), Indicate(mb[0], color=GREEN))
        self.say("attend only to real audio", GREEN)
        self.beat()

        # 4. average-pool ×2, then the LayerNorm
        pool = blk("pool ÷2 · LN", "1500 → 750", GREEN, X["pool"], w=1.9)
        a3 = link(enc, pool, GREEN)
        rate = tag("40 ms / token", GREEN, 22).next_to(pool, UP, buff=0.15)
        self.play(GrowArrow(a3), FadeIn(pool, scale=0.8), FadeIn(rate))
        self.say("average pairs: 1500 → 750", GREEN)
        self.beat()

        # 5. the projector into the LLM's width
        proj = blk("projector", "1280 → 4096", GREEN, X["proj"], w=1.9)
        a5 = link(pool, proj, GREEN)
        self.play(GrowArrow(a5), FadeIn(proj, scale=0.8))
        self.say("one Linear to the LLM width", GREEN)
        self.beat()

        # 6. the LLM: audio rows go into <|AUDIO|> slots of the prompt
        llm = blk("Qwen LLM", "7B · causal", BLUE, X["llm"], w=2.0)
        a6 = link(proj, llm, GREEN)
        prompt = VGroup(chip("<|audio_bos|>", ACCENT, 20, h=0.5), chip(f"<|AUDIO|> × {N_AUDIO}", ACCENT, 20, h=0.5),
                        chip("<|audio_eos|>", ACCENT, 20, h=0.5), chip(" what's this?", PINK, 20, h=0.5)).arrange(RIGHT, buff=0.08)
        prompt.next_to(VGroup(proj, llm), DOWN, buff=0.75).align_to(llm, RIGHT)
        up = Arrow(prompt.get_top(), llm.get_bottom(), buff=0.08, stroke_color=BLUE, stroke_width=3)
        ans = VGroup(Arrow(llm.get_top(), llm.get_top() + 0.75 * UP, buff=0.05, stroke_color=BLUE), )
        at = chip(" a melody", PINK, 20, h=0.5).next_to(ans, UP, buff=0.05)
        self.play(FadeOut(VGroup(mb, ml, mt)), GrowArrow(a6), FadeIn(llm, scale=0.8))
        self.play(FadeIn(prompt, shift=0.2 * UP), GrowArrow(up))
        self.play(GrowArrow(ans[0]), FadeIn(at))
        self.say(f"{N_AUDIO} audio rows fill the prompt", BLUE)
        self.beat()

        # 7. name the result
        br = Brace(VGroup(mel, llm), DOWN, buff=0.2)
        bt = T("Qwen2-Audio · 8.2 B parameters", 32, INK).next_to(br, DOWN, buff=0.1)
        head3 = T("Qwen2-Audio", 36, BLUE).move_to(head)
        self.play(Transform(head, head3), GrowFromCenter(br), FadeIn(bt), FadeOut(prompt), FadeOut(up))
        self.say("Whisper's encoder + glue + an LLM")
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ our clip: 400 → 200 → 100 tokens
    def token_count(self):
        W = 9.0
        rows = [("mel frames", 3000, MEL_FRAMES, GOLD, r"4\,\text{s} \times 100 = 400"),
                ("after conv2", 1500, AFTER_CONV, ACCENT, r"\lfloor (400-1)/2 \rfloor + 1 = 200"),
                ("after pool", 750, N_AUDIO, GREEN, r"\lfloor (200-2)/2 \rfloor + 1 = 100")]
        grp = VGroup()
        for name, total, real, col, f in rows:
            b = bar(total, real, W * total / 3000, col)
            lab = T(name, 28, col)
            cnt = tag(f"{real} of {total}", col, 24)
            fm = M(f, 34, col)
            grp.add(VGroup(lab, b, cnt, fm))
        for i, g in enumerate(grp):
            lab, b, cnt, fm = g
            b.move_to([-5.0 + b.get_width() / 2 + 0.3, 2.1 - i * 1.5, 0])
            lab.next_to(b, UP, buff=0.12).align_to(b, LEFT)
            cnt.next_to(lab, RIGHT, buff=0.35)
            fm.next_to(b, DOWN, buff=0.12).align_to(b, LEFT)
        for g in grp:
            self.play(FadeIn(g, shift=0.15 * DOWN), run_time=0.8)
        self.say("each stride-2 step halves it")
        self.beat()
        pad = grp[2][1][0]
        keep = SurroundingRectangle(grp[2][1][1], stroke_color=GREEN, buff=0.05)
        drop = T("padded tail dropped", 28, BAD).next_to(grp[2][1], RIGHT, buff=0.4)
        self.play(ShowCreation(keep), pad.animate.set_fill(BAD, 0.25), FadeIn(drop))
        self.say(f"keep the first {N_AUDIO} tokens = 4 s", GREEN)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ two ways to talk to it, one model
    def modes(self):
        def panel(title, color, chips, x):
            t = T(title, 32, color)
            row = VGroup(*chips).arrange(RIGHT, buff=0.1)
            box = SurroundingRectangle(row, buff=0.3, stroke_color=color)
            return VGroup(t, VGroup(box, row)).arrange(DOWN, buff=0.35).move_to([x, 0.6, 0])

        left = panel("audio analysis", ACCENT, [chip("audio clip", ACCENT, 24, h=0.6), chip(" What is this sound?", PINK, 24, h=0.6)], -3.5)
        right = panel("voice chat", BLUE, [chip("spoken question", ACCENT, 24, h=0.6)], 3.5)
        ln = tag("audio + text instruction", DIM, 22).next_to(left, DOWN, buff=0.25)
        rn = tag("the instruction is IN the audio", DIM, 22).next_to(right, DOWN, buff=0.25)
        self.play(FadeIn(left, shift=0.2 * UP), FadeIn(ln))
        self.say("analysis: audio plus a question", ACCENT)
        self.beat()
        self.play(FadeIn(right, shift=0.2 * UP), FadeIn(rn))
        self.say("chat: just talk to it", BLUE)
        self.beat()
        same = T("one model · no system prompt to switch", 32, GOLD).to_edge(UP, buff=0.5)
        self.play(FadeIn(same))
        self.say("both modes trained jointly", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ the three training stages
    def training(self):
        stages = [("1 · pre-train", "natural-language prompts", ACCENT), ("2 · SFT", "chat + analysis data", BLUE),
                  ("3 · DPO", "prefer good answers", GOLD)]
        boxes = VGroup(*[block(n, 3.5, 1.15, c, 30, sub=s) for n, s, c in stages]).arrange(RIGHT, buff=0.7).to_edge(UP, buff=0.6)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_color=RULE) for a, b in zip(boxes[:-1], boxes[1:])])
        self.play(LaggedStartMap(FadeIn, boxes, lag_ratio=0.25), LaggedStartMap(GrowArrow, arrows), run_time=1.5)
        self.say("three stages")
        self.beat()

        # stage 1: prompts instead of Whisper-style task tags
        hl = SurroundingRectangle(boxes[0], stroke_color=ACCENT, buff=0.08)
        tags = VGroup(*[chip(t, DIM, 20, h=0.5) for t in ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]]).arrange(RIGHT, buff=0.08)
        tags.move_to(0.5 * UP)
        xt = Cross(tags, stroke_color=BAD, stroke_width=5)
        nl = chip(' "Detect the language and recognize the speech:"', ACCENT, 22, h=0.6).move_to(0.8 * DOWN)
        obj = M(r"\max\ P_\theta\big(x_t \mid x_{<t},\ \mathrm{Enc}_\phi(a)\big)", 40).move_to(2.1 * DOWN)
        self.play(ShowCreation(hl), FadeIn(tags))
        self.play(ShowCreation(xt))
        self.play(FadeIn(nl, shift=0.2 * UP))
        self.play(Write(obj))
        self.say("tags → plain-language prompts", ACCENT)
        self.beat()
        both = tag("θ (LLM) and ϕ (encoder) both trained", GOLD, 24).next_to(obj, DOWN, buff=0.2)
        self.play(FadeIn(both))
        self.beat()
        self.play(*[FadeOut(m) for m in (tags, xt, nl, obj, both)], hl.animate.move_to(boxes[1]).set_color(BLUE))

        # stage 2: instruction data, both modes together
        sft = VGroup(chip("audio + question -> answer", ACCENT, 24, h=0.6), chip("spoken question -> text reply", BLUE, 24, h=0.6))
        sft.arrange(DOWN, buff=0.35).move_to(0.2 * DOWN)
        self.play(FadeIn(sft, lag_ratio=0.3))
        self.say("SFT: curated instruction data", BLUE)
        self.beat()
        self.play(FadeOut(sft), hl.animate.move_to(boxes[2]).set_color(GOLD))

        # stage 3: DPO on (good, bad) answer pairs
        good = VGroup(M(r"y_w", 40, GREEN), chip(" a short piano melody", GREEN, 22, h=0.55)).arrange(RIGHT, buff=0.2)
        bad = VGroup(M(r"y_l", 40, BAD), chip(" someone is speaking", BAD, 22, h=0.55)).arrange(RIGHT, buff=0.2)
        pair = VGroup(good, bad).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to(0.35 * UP)
        dpo = M(r"-\log\sigma\Big(\beta\log\frac{P_\theta(y_w\mid x)}{P_{\text{ref}}(y_w\mid x)}"
                r"-\beta\log\frac{P_\theta(y_l\mid x)}{P_{\text{ref}}(y_l\mid x)}\Big)", 36).move_to(1.85 * DOWN)
        self.play(FadeIn(pair, lag_ratio=0.3))
        self.play(Write(dpo))
        self.say("DPO: prefer the good answer", GOLD)
        self.beat()
        self.clear_all()

    # ------------------------------------------------------------ side by side
    def compare(self):
        rows = [("", "Whisper large-v3", "Qwen2-Audio"),
                ("input", "128 mels · 30 s", "same"),
                ("encoder", "32 layers · 1280", "same, initialised from it"),
                ("audio tokens", "1500 · 20 ms", "750 · 40 ms (pooled)"),
                ("text reads audio", "cross-attention", "<|AUDIO|> tokens"),
                ("text model", "own 32-layer decoder", "Qwen 7B LLM"),
                ("training", "multitask ASR", "pre-train → SFT → DPO")]
        cols = [-4.6, -0.9, 3.6]
        lines = VGroup()
        for i, r in enumerate(rows):
            y = 2.9 - i * 0.82
            for j, (x, text) in enumerate(zip(cols, r)):
                if not text:
                    continue
                c = DIM if j == 0 else (PINK if j == 1 else BLUE)
                if i == 0:
                    c = PINK if j == 1 else BLUE
                t = T(text, 34 if i == 0 else 28, c)
                if j > 0 and i > 0 and text.startswith("same"):
                    t.set_color(ACCENT)
                lines.add(t.move_to([x, y, 0]))
            if i == 0:
                lines.add(Line([-6.6, y - 0.42, 0], [6.6, y - 0.42, 0], stroke_color=RULE))
        self.play(LaggedStartMap(FadeIn, lines, lag_ratio=0.05), run_time=2)
        self.say("same ears, new brain")
        self.beat()
        self.clear_all()


split_sections(L7_Qwen2Audio, globals())
