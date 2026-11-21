---
id: 01-llava-bridge
title: "LLaVA: from ViT patch features to LM input tokens"
difficulty: hard
runtime: python
timeout_s: 30
entrypoint: llava_forward
paper_refs: ["LLaVA §4.1", "LLaVA-1.5 §3.2", "LLaVA-1.5 Fig. 1"]
---

## Context

LLaVA does not change the ViT or the language model. It adds a bridge between them:

$$X_v = \text{ViT}(I),\qquad H_v = \text{MLP}(X_v),\qquad \text{LM}([\,w_{<p};\ H_v;\ w_{>p}\,])$$

Four small choices decide what the LM actually sees:

1. **Which ViT output.** It is not CLIP's image embedding. It is the **penultimate** layer's hidden state (`mm_vision_select_layer = -2`), with **no `ln_post`**, and **only the patch tokens** (`select_feature = "patch"`, so CLS is dropped). CLIP's last layer is shaped for the contrastive loss, while the layer before it keeps more local detail.
2. **The connector.** LLaVA-1 used one `Linear`. LLaVA-1.5 uses a 2-layer MLP, `Linear → GELU → Linear`, applied to each patch on its own.
3. **Where the tokens go.** The prompt contains one `<image>` id. That one id is **replaced** by N vectors, so the sequence grows from T to **T − 1 + N**. The image sits where the placeholder was, not at the front.
4. **How the LM treats them.** They are ordinary positions: RoPE positions `0..L−1` over the spliced sequence, the same causal mask, and labels of −100, so the model is never trained to predict an image.

The ViT block and the LLaMA decoder block are **given** in the starter, already solved in the ViT pack. Your code is only the bridge.

## Task

```python
vision_features(image, vt)                                            -> (B, N, Dv)
projector(f, proj)                                                    -> (B, N, D)
merge_image_tokens(input_ids, image_token_id, embed, image_tokens, labels)
                                                                      -> (embeds, labels, vis_mask)
llava_forward(image, input_ids, labels, image_token_id, params)      -> (logits, loss)
```

**Vision tower** `vt` (CLIP-ViT, as in the CLIP problem):

| key | shape | |
|---|---|---|
| `patch`, `heads` | int | |
| `W_patch` | `(P·P·C, Dv)` | no bias; patches are row-major, each flattened as `(row, col, channel)` |
| `cls`, `pos` | `(Dv,)`, `(N+1, Dv)` | CLS at index 0 |
| `blocks` | list of dicts | run each with `clip_block(x, heads, blk)` |

Use HF's `hidden_states` convention: `hs[0] = ln_pre(concat(cls, patches) + pos)`, and `hs[i]` is the output of block `i`. `layer_norm` is the `ln_pre`.

**Projector** `proj`: `W1 (Dv, D)`, `b1 (D,)`, `W2 (D, D)`, `b2 (D,)`. Use the given exact `gelu`.

**LM** `params["lm"]`: `heads`, `embed (V, D)`, `blocks` (run each with `decoder_block(x, positions, heads, blk)`), and `lm_head (D, V)`. Apply `rms_norm` before the head. `params = {"vision": vt, "proj": proj, "lm": lm}`.

**Ids and labels**: `input_ids (B, T)`. Each row has **exactly one** `image_token_id`, at a different place in each row. `labels (B, T)` uses −100 for ignored positions. The loss is next-token cross-entropy: `logits[:, t]` predicts `labels[:, t+1]`, averaged over all targets that are not −100 in the batch.

## Example

`T = 6`, `ids = [12, 7, 9, 15, 11, 10]` with `<image> = 7`, and N = 4 patches. The LM sees 9 vectors:

```
pos:   0     1    2    3    4    5    6    7    8
      e12 | v0   v1   v2   v3 | e9  e15  e11  e10
lab:  -100 -100 -100 -100 -100 -100 -100  11   10
```

## Constraints

NumPy only. Do not edit the given block.

## What the tests check

- `test_vision_features_are_penultimate_patch_tokens`: the correct layer, no `ln_post`, CLS dropped, over 3 configs.
- `test_projector_is_two_layer_gelu_mlp`: the MLP form, and that it acts per patch.
- `test_merge_puts_image_tokens_at_the_placeholder`: placement, length T − 1 + N, `vis_mask`, image labels, and text labels moving with the text.
- `test_forward_matches_reference`: logits and loss against a loop-based reference.
- `test_text_before_image_cannot_see_the_image`: causality through the splice.
