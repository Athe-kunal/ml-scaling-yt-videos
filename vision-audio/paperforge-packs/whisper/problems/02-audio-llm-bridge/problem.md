---
id: 02-audio-llm-bridge
title: "Qwen2-Audio & Qwen3-Omni: how the audio encoder feeds a decoder-only LLM"
difficulty: hard
runtime: torch
timeout_s: 60
entrypoint: qwen2_audio_generate
paper_refs: ["Qwen2-Audio §2", "Qwen3-Omni §2.2 (AuT)", "Qwen3-Omni §2.4 (TM-RoPE)", "HF modeling_qwen2_audio.py", "HF modeling_qwen3_omni_moe.py"]
---

## Context

In problem 01, Whisper's decoder reads the audio through **cross-attention**: the text stream and the audio stream stay separate. Audio LLMs take a different route. The encoder output is **projected to the LLM's width and spliced into the token sequence**, in place of `<|AUDIO|>` placeholder tokens. A plain causal LLM then reads it like a prefix of text.

```
text:   <|im_start|>user  <|audio_bos|> <|AUDIO|> <|audio_eos|> Transcribe.
ids:    ...               29            30 30 30 … 30 (n copies)  28        …
embeds: embed(ids)        embed(29)     [ audio feature rows ]   embed(28) …   → causal LLM → next token
```

**Qwen2-Audio** (arXiv 2407.10759, §2) initialises its audio encoder from **Whisper-large-v3** (128 mels). It adds *"a pooling layer with a stride of two"*, so *"each frame of the encoder output approximately corresponds to a 40ms segment"*. In the HF code:
- The encoder always sees **30 s** of mel (3000 frames), padded.
- It runs the Whisper encoder with the padded positions masked as keys.
- Then `AvgPool1d(2, 2)` over time, then `LayerNorm`, then a `Linear` projector into the 7B LLM.
- Only the first `n` pooled tokens are real audio.

**Qwen3-Omni** (arXiv 2509.17765, §2.2) replaces Whisper with **AuT**, which runs at a **12.5 Hz** token rate (80 ms per token):
- The mel is cut into **1 s chunks** (100 frames).
- Each chunk goes through three stride-2 `Conv2d`s over (mel, time), so 100 frames become 13 tokens.
- Sinusoidal positions **restart at 0 inside every chunk**.
- Attention is **block-diagonal over windows** (`n_window_infer`, 8 s by default) so the Thinker can cache prefill window by window.
- Finally `ln_post → proj1 → GELU → proj2` maps straight into the Thinker's width.

Under TM-RoPE (§2.4), audio tokens get one temporal ID per 80 ms, which is one per token. For audio-only input, that is ordinary 1-D positions, and `TinyLM` uses exactly those.

## Task

Given (don't change): `sinusoids`, `AudioAttention` / `AudioEncoderLayer` (Whisper-style, HF names; AuT's `k_proj` **has** a bias, Whisper's doesn't), `TinyLM` (stands in for the LLM / Thinker, `lm(inputs_embeds) -> logits`), every `__init__`, and `Qwen2AudioProjector`.

Fill in, top-down (the starter is in the same order: L0 architecture → L1 bridge → L2 encoders → L3 given blocks):

| function | spec |
|---|---|
| `qwen2_audio_generate(...)`, `qwen3_omni_generate(...)` | the end-to-end bridges |
| `expand_audio_placeholders(ids, audio_token_id, n_tokens)` | the processor step: the k-th `<|AUDIO|>` becomes `n_tokens[k]` copies; `ValueError` on a count mismatch |
| `merge_audio_embeddings(inputs_embeds, input_ids, audio_token_id, audio_features)` | the HF `masked_scatter` step: placeholders, in reading order, get the packed feature rows. `ValueError` if the counts differ. Must not modify the input. |
| `greedy_generate(lm, inputs_embeds, max_new_tokens, eos_id)` | batch 1; argmax of the last position; append the new token's **embedding**; stop after EOS |
| `Qwen2AudioEncoder.forward(input_features, feature_lens)` | input `(B, n_mels, 2·P)` padded; conv/positions given; mask padded **keys** → layers → pool over time → LN. Returns `(B, P//2, d)`. |
| `qwen2_audio_output_lengths(mel_lens)` | `(after_conv, after_pool)` real lengths; conv2 is k3/s2/p1, pool k2/s2 |
| `AuTEncoder.forward(features)` | list of `(n_mels, L_i)` → **packed** `(Σ tokens, output_dim)`. Chunking and the conv stem are given. Write the flatten (`c`-major), per-chunk positions, valid-token selection, windowed attention (window = `13 · n_window_infer // 100` tokens, counted per audio), `ln_post → proj1 → GELU → proj2`. |
| `aut_output_lengths(mel_lens, n_window=50)` | tokens per audio: 13 per full 100-frame chunk, plus the tail chunk through 3 convs. Vectorised. |

## Example

Qwen2-Audio, 30 s window but only 15 real mel frames (out of `2·P = 24` in the tests):
`after_conv = (15−1)//2+1 = 8`, `after_pool = (8−2)//2+1 = 4`. The encoder returns 6 pooled tokens, of which only the first **4** are spliced into the prompt.

AuT, 250 frames: chunks 100 + 100 + 50 → 13 + 13 + 7 = **33** tokens.

## Constraints

- No `transformers` and no new parameters.
- `greedy_generate` must never rebuild the LLM input from token ids. The audio only exists as embeddings.

## What the tests check

- `test_qwen2_output_lengths_match_conv_arithmetic`: compared against actually running the convs and the pool.
- `test_qwen2_encoder_matches_reference`: a per-head reference where every query sees only the real keys.
- `test_qwen2_padding_frames_cannot_leak_into_real_tokens`
- `test_aut_output_lengths_match_chunked_conv_arithmetic`
- `test_aut_encoder_matches_reference`: chunk by chunk, an explicit `(c, f)` flatten, window by window; three configs, including two window sizes.
- `test_aut_attention_is_windowed_and_never_crosses_audios`
- `test_expand_audio_placeholders`, `test_merge_scatters_audio_in_reading_order_and_checks_count`
- `test_greedy_generate_keeps_the_audio_prefix`: a spy LM records what you feed it at every step.
- `test_qwen2_audio_pipeline_end_to_end`, `test_qwen3_omni_pipeline_end_to_end`: the exact prefix embeddings the LLM receives.
