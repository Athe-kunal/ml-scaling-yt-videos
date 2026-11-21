---
id: 02-deepstack
title: "DeepStack: stack high-res tokens into the LM's early layers"
difficulty: hard
runtime: python
timeout_s: 30
entrypoint: deepstack_forward
paper_refs: ["DeepStack §3.2", "DeepStack Alg. 1", "DeepStack Eq. 4-5", "DeepStack Fig. 2"]
---

## Context

LLaVA puts every visual token **into the input sequence**. With 4× the resolution there are 4× the tokens, and the context gets 4× longer. DeepStack keeps the sequence from problem 01 (the **low-res** global view, N tokens at the placeholder) and sends the extra detail **through the depth instead of the length**:

$$X_{stack}=\{X^1_{stack},\dots,X^s_{stack}\}=\text{Sampling2D}\big(M(F^v(I_{hires}))\big)\qquad\text{(Eq. 4)}$$

```python
for idx, layer in enumerate(self.layers):             # Algorithm 1
    if idx >= l_start and (idx - l_start) % n == 0:
        H[vis_pos] += X_stack[(idx - l_start) // n]
    H = layer(H)
```

- **The high-res stream.** The image at 2× resolution is cut into a 2×2 grid of tiles. Each tile is encoded by the **same** ViT, with the same LLaVA feature (penultimate layer, patches only), and the tiles are stitched back into one `(2hp, 2wp)` feature grid. That grid goes through the **same** projector `M`.
- **Sampling2D (dilation 2).** Split that grid into s = 4 stacks, each the size of X. Stack `(di, dj)` takes `grid[di::2, dj::2]`, so token `(i, j)` of every stack is one of the 4 high-res neighbours **inside low-res cell (i, j)**. That cell is exactly the one the visual token at that position describes. A residual add is meaningful only because of this alignment.
- **Injection.** Adding happens **before** layer `idx`, only at the visual positions, and only until the stacks run out. Nothing is concatenated, so L = T − 1 + N, the same as LLaVA.

(Qwen3-VL also calls its version "DeepStack". It uses features from several *ViT layers* instead of high-res tiles, but the LM side is the same residual add at `vis_pos` in the first layers.)

## Task

Copy your problem 01 bridge (`vision_features`, `projector`, `merge_image_tokens`) into the marked block, then write:

```python
deepstack_tokens(image_hires, vt, proj)                                          -> (4, B, N, D)
deepstack_forward(image, image_hires, input_ids, image_token_id, params, l_start, n) -> (B, T-1+N, V)
```

- `image (B, H, W, C)`: the global view. `image_hires (B, 2H, 2W, C)`.
- Tiles are TL, TR, BL, BR, each `(H, W)`. Inside a tile the patches are row-major, as in 01.
- Stacks are ordered `(di, dj) = (0,0), (0,1), (1,0), (1,1)`.
- `params` is the same as in 01. There are no new weights.
- Return logits `rms_norm(H) @ lm_head`. There is no loss.

## Example

Low-res grid 2×2 (N = 4), so the high-res grid is 4×4. Write the high-res tokens as `g[r, c]`:

```
stack (0,0): g[0,0] g[0,2] g[2,0] g[2,2]      stack (0,1): g[0,1] g[0,3] g[2,1] g[2,3]
stack (1,0): g[1,0] g[1,2] g[3,0] g[3,2]      stack (1,1): g[1,1] g[1,3] g[3,1] g[3,3]
```

With `l_start = 1, n = 2` and 6 layers, stacks 0, 1, 2 go in before layers 1, 3, 5. Stack 3 is never used.

## Constraints

NumPy only. Do not edit the given block. Do not add tokens to the sequence.

## What the tests check

- `test_stacks_are_dilated_high_res_neighbours`: stack shapes, tile stitching, the dilated sampling and the stack order, over 3 configs including a non-square grid.
- `test_forward_matches_algorithm_1`: logits against a loop-based reference over 4 `(l_start, n)` schedules × 3 configs.
- `test_reduces_to_llava_when_no_layer_is_selected`: with `l_start` past the last layer, the output equals plain LLaVA.
- `test_text_before_image_is_untouched_by_the_stacks`
