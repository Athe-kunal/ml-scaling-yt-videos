---
id: 01-whisper-from-scratch
title: "Whisper from scratch: log-Mel, encoder, decoder, greedy decode"
difficulty: hard
runtime: torch
timeout_s: 60
entrypoint: Whisper
paper_refs: ["§2.2", "Fig. 1", "audio.py", "model.py", "decoding.py GreedyDecoder"]
---

## Context

Whisper (Radford et al. 2022) deliberately uses an **off-the-shelf** encoder-decoder Transformer (§2.2). The paper's results come from the data, not the architecture, so what's left to learn is the exact details. Every one of them shows up when you load a real checkpoint:

```
audio (16 kHz) ─pad_or_trim→ 480000 samples ─log-Mel→ (80, 3000) ─conv stem→ (1500, D) ─+sinusoids→ encoder blocks → ln_post ─┐
tokens ─embed + learned pos→ decoder blocks [self-attn (causal) → cross-attn (to audio) → MLP] → ln → @ token_embedding.T → logits
```

From the paper (§2.2): *"80-channel log-magnitude Mel spectrogram … on 25-millisecond windows with a stride of 10 milliseconds. … globally scale the input to be between -1 and 1 … two convolution layers with a filter width of 3 and the GELU activation … the second convolution layer has a stride of two. Sinusoidal position embeddings … pre-activation residual blocks … a final layer normalization … The decoder uses learned position embeddings and tied input-output token representations."*

The paper says "log-magnitude". The code is what defines the model, so follow `audio.py`. The tests are written against it.

## Task

Fill in the `YOUR CODE` blocks of the starter. All the module and parameter names are given, and your `state_dict()` has to match Whisper's exactly.

The starter is ordered **top-down**: L0 `Whisper` + `greedy_decode` → L1 encoder/decoder → L2 block → L3 attention/sinusoids → L4 the audio front end. The tables below follow the original file order (`audio.py`, then `model.py`).

**audio.py**

| function | spec |
|---|---|
| `pad_or_trim(audio, length=480000)` | Make the last axis exactly `length`: zero-pad on the **right**, or keep the **first** `length` samples. |
| `log_mel_spectrogram(audio, filters)` | `torch.stft` is given (Hann 400, hop 160, `center=True`). **Drop the last frame**, take the **power** `|X|²`, apply `filters @ power`, then `log10(clamp(·, 1e-10))`, then `max(·, M − 8)` where `M` is the max over the **entire** output tensor, then `(·+4)/4`. |

`filters` is Whisper's own `mel_80` matrix `(80, 201)` (Slaney mel scale, from `librosa`). The tests load it from `files/mel_filters.npz`.

**model.py**

| piece | spec |
|---|---|
| `sinusoids(L, C)` | `inv_i = exp(-i·ln(10000)/(C/2 − 1))` for `i = 0..C/2−1`; output `[sin(t·inv) ‖ cos(t·inv)]`, **concatenated** |
| `MultiHeadAttention` | `query, key, value, out`. `key` has **no bias**. Scale `q` and `k` each by `(D/H)^{-1/4}`, add the mask `mask[:Tq, :Tq]` if given, softmax over keys. Return `(out, qk)` where `qk` is the detached pre-softmax scores `(B, H, Tq, Tk)`. |
| `ResidualAttentionBlock` | pre-LN: `x += attn(attn_ln(x), mask)`; if cross: `x += cross_attn(cross_attn_ln(x), xa)` with **no mask**; `x += mlp(mlp_ln(x))` |
| `AudioEncoder` | `gelu(conv1)` (k3, p1) → `gelu(conv2)` (k3, p1, **stride 2**) → `(B, T, D)` → assert `T == n_ctx` → `+ positional_embedding` → blocks → `ln_post` |
| `TextDecoder` | `token_embedding(tokens) + positional_embedding[:T]` → blocks(`xa`, causal mask) → `ln` → `x @ token_embedding.weight.T`. The mask is `-inf` above the diagonal, registered with `persistent=False`. |

**decoding.py**: `greedy_decode(model, mel, sot_sequence, eot, max_new_tokens) -> (tokens, sum_logprobs)`. Encode once. At every step take the argmax of the logits at the **last** position. Add that token's `log_softmax` to `sum_logprobs` unless the row had **already** ended with EOT. Rows that have finished keep appending EOT. Stop after `max_new_tokens` steps, or as soon as **every** row ends with EOT. This is `GreedyDecoder.update` at temperature 0, with no KV cache.

## Examples

- 30 s of audio → `log_mel_spectrogram` → `(80, 3000)`. `torch.stft` gives 3001 frames, and Whisper drops the last one. `conv2` (stride 2) turns 3000 frames into **1500 tokens**, so one audio token is **20 ms**. That is also the timestamp resolution.
- A silent first half: every mel bin there ends up at exactly `(M − 8 + 4)/4 = M_scaled − 2`.

## Constraints

- Keep the given names. No `F.scaled_dot_product_attention` and no `nn.MultiheadAttention`: `qk` has to be your own scores.
- The tests run everything in float64.

## What the tests check

- `test_pad_or_trim_pads_right_and_trims_tail`
- `test_thirty_seconds_give_exactly_3000_frames`
- `test_log_mel_matches_explicit_reference`: compared against an explicit reflect-pad + framing + `rfft` reference, on three lengths, with a quiet section.
- `test_dynamic_range_clamp_is_global_not_per_frame`
- `test_sinusoids_match_reference`: compared against a double loop.
- `test_state_dict_names_and_shapes_match_whisper`: the exact key set and shapes of a real checkpoint.
- `test_attention_returns_raw_scaled_scores`: both `qk` and the output, against a per-head loop.
- `test_encoder_matches_reference`, `test_encoder_rejects_wrong_length`
- `test_decoder_matches_reference`: T = 1, 4, 7, against a per-token loop.
- `test_decoder_is_causal_but_sees_all_audio`
- `test_logits_use_the_tied_embedding`
- `test_whisper_forward_is_decoder_of_encoder`
- `test_greedy_decode_scripted`: a scripted fake model. Checks EOT handling, which position's logits you use, the log-prob bookkeeping, and that the encoder is called once.
- `test_greedy_decode_matches_full_model_reference`
