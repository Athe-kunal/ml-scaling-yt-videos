# Whisper from scratch

One problem covers the basics of the whole package, with no training. It's in the style of the ViT → CLIP → SigLIP pack: an independent loop-based reference in `tests.py` and many named mutants. The starter is template-heavy, so you fill in only the core blocks.

## The paper in 5 lines

1. **Claim:** a plain encoder-decoder Transformer, trained on 680k hours of weakly labelled audio, transfers zero-shot and robustly (§1, §3).
2. **Input:** 16 kHz audio → 30 s window → 80-channel log-Mel (25 ms window, 10 ms hop), globally scaled to about [-1, 1] (§2.2).
3. **Encoder:** two width-3 convs + GELU (the second has stride 2) → sinusoidal positions → pre-LN blocks → final LN.
4. **Decoder:** learned positions, causal self-attention, cross-attention to the audio, tied input/output embedding.
5. **Everything is tokens:** `<|startoftranscript|> <|en|> <|transcribe|> …` tells one decoder which task to do (§2.3). Here we decode greedily from such a prefix.

## The problem

**01 · Whisper from scratch** (hard). `audio.py` + `model.py` + a greedy `decoding.py` loop. Your `state_dict()` uses the real checkpoint names. The reference solution loads a real `Whisper` state dict with `strict=True` and gives identical logits.

What it kills (each one is a mutant that a test catches):

| Part | Misconceptions |
|---|---|
| `pad_or_trim` | padding on the left; keeping the tail |
| log-Mel | magnitude instead of power; keeping 3001 frames; `ln` instead of `log10`; per-frame clamp instead of the global `max − 8`; forgetting `(x+4)/4` |
| sinusoids | interleaved sin/cos; `/(C/2)` instead of `/(C/2 − 1)` |
| attention | a key bias (softmax-invisible, but it shows up in the raw `qk`); scaling by `1/d` |
| encoder | stride on conv1; no GELU after conv2; no `ln_post`; post-LN blocks |
| decoder | cross-attention before self-attention; masked cross-attention; mask hiding the diagonal; no causal mask; untied output head; no final `ln` |
| greedy | logits from position 0; scoring padding EOTs; re-encoding at every step; stopping when *any* row finishes |

Source: `paper.pdf`, and the full repo in `refs/whisper-src/whisper/` (read `audio.py` and `model.py` after you solve it). Natural follow-ups if you want more later: KV cache, the timestamp logit rules, beam search, word timestamps via DTW.

## 02 · Qwen2-Audio & Qwen3-Omni: the audio → LLM bridge (hard)

Whisper's decoder **cross-attends** to the audio. Audio LLMs instead **splice the encoder output into the token sequence**: each `<|AUDIO|>` placeholder is expanded to *n* copies, and their embedding rows are replaced (`masked_scatter`) by projected audio features. A plain causal LLM then decodes from `inputs_embeds`.

| | Qwen2-Audio | Qwen3-Omni (AuT) |
|---|---|---|
| Encoder | Whisper-large-v3 (128 mels), always 30 s, padded keys masked | AuT, un-padded audio cut into 1 s chunks |
| Downsampling | conv2 s2 + **AvgPool1d(2)** → 40 ms/token | **3× Conv2d s2** per chunk → 13 tokens/s (80 ms) |
| Positions | one sinusoid table over 1500 frames | sinusoids **restart in every chunk** |
| Attention | full (over real keys) | **block-diagonal windows** (8 s), never across audios |
| Into the LLM | LN → `Linear` projector; keep the first `n` tokens | `ln_post → proj1 → GELU → proj2` (already the Thinker width), packed |

What it kills: wrong pooled-length arithmetic; no padding mask (or one based on mel length instead of conv length); LN before the pool; pooling over channels; AuT lengths without chunking; global positions; `(f, c)` flatten; full or per-audio attention instead of windows; no GELU between projections; keeping padding tokens; no expansion or count check; in-place merge; decoding from position 0; rebuilding the input from ids (which loses the audio); splicing padded audio tokens.

Sources: `refs/qwen/` (both papers + the HF modeling/processing files).
