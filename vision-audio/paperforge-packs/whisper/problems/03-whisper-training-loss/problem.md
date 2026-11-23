---
id: 03-whisper-training-loss
title: "Whisper's training loss: multitask sequence, teacher forcing, masked NLL"
difficulty: hard
runtime: torch
timeout_s: 60
entrypoint: whisper_loss
paper_refs: ["§2.3 Multitask Format", "Fig. 1"]
---

## Context

Problem 01 built Whisper and decoded with it. This problem is how Whisper is **trained**: plain next-token prediction on one long token sequence. Every decoder position predicts the next token, and the loss is the **negative log-likelihood** of the true next token.

Two details from §2.3 decide which tokens count:
- A sequence can start with the **previous segment's text** after `<|startofprev|>`, as conditioning. *"We only mask out the training loss over the previous context text, and train the model to predict all other tokens."*
- So the language token **is** a target. That's how Whisper learns language ID: at inference, `detect_language` reads the logits at the `<|startoftranscript|>` position.

The decoder's causal mask is what makes **teacher forcing** valid. One forward pass over the true sequence gives every position's prediction at once, and position $i$ only ever sees tokens $\le i$. The gradient of this text-only loss flows through cross-attention into the **audio encoder**. Nothing else trains it.

## Task

The model from problem 01 is given (`ModelDimensions`, `Whisper`, ...). Write three functions.

**`make_training_example(prev_ids, sot_sequence, text_ids, eot, sot_prev)`**

Build the sequence

$$[\,\texttt{<|startofprev|>},\ \text{prev}\ldots\,]\ (\text{only if prev is non-empty}) \;+\; [\,\texttt{<|sot|>},\ \texttt{<|lang|>},\ \texttt{<|task|>}\ldots\,] \;+\; [\,\text{text}\ldots,\ \texttt{<|endoftext|>}\,]$$

Return `(input_ids, target_ids, loss_mask)`, all 1-D, each of length `len(sequence) − 1`:
- `input_ids = sequence[:-1]`
- `target_ids = sequence[1:]`
- `loss_mask` is `False` exactly where the **target** is previous-context text.

Dtypes are long, long and bool.

**`nll_loss(logits, target_ids, loss_mask)`**

Takes `(B, T, V)`, `(B, T)` and `(B, T)`, and returns a scalar:

$$\mathcal{L} = \frac{1}{\sum m}\sum_{b,t} m_{bt}\,\Big(\log\sum_v e^{z_{btv}} - z_{bt\,y_{bt}}\Big)$$

This is a mean over **all counted tokens in the batch**. Compute it stably.

**`whisper_loss(model, mel, input_ids, target_ids, loss_mask)`**

One teacher-forced forward pass `model(mel, input_ids)`, then `nll_loss`. Don't detach anything.

## Example

With `prev = [3, 4]`, `sot_sequence = [8, 9, 10]`, `text = [5, 6, 7]`, `eot = 11` and `sot_prev = 12`:

```
sequence  = [12, 3, 4, 8, 9, 10, 5, 6, 7, 11]
input_ids = [12, 3, 4, 8, 9, 10, 5, 6, 7]
targets   = [ 3, 4, 8, 9, 10, 5, 6, 7, 11]
loss_mask = [ F, F, T, T, T,  T, T, T, T]
```

The position holding `8` (`<|startoftranscript|>`) predicts `9`, the language token, and that target counts.

## Constraints

- `nll_loss` must stay finite for logits around ±1e4. `log(softmax(z))` underflows there.
- Padded positions (mask `False`) must not change the loss.

## What the tests check

- `test_example_layout_with_previous_text`: the exact inputs, targets and mask, and their dtypes.
- `test_no_startofprev_without_previous_text`
- `test_language_token_is_a_trained_target`: only the context *text* is masked.
- `test_shift_by_one_and_eot_is_only_a_target`
- `test_nll_matches_loop_reference`: an explicit loop over tokens.
- `test_nll_averages_over_tokens_not_sequences`
- `test_masked_targets_do_not_matter`
- `test_nll_is_stable_for_huge_logits`
- `test_uniform_logits_give_log_vocab`: a uniform guess costs $\log V$ nats.
- `test_gradient_is_softmax_minus_onehot_over_count`
- `test_teacher_forcing_equals_stepwise_scoring`: one pass equals scoring each next token from the true prefix.
- `test_text_loss_reaches_the_encoder`: `encoder.conv1` and the cross-attention keys both get gradient.
- `test_padded_batch_is_token_weighted`
