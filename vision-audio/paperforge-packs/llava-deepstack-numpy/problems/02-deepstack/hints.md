The high-res tokens and the low-res tokens both come out of the same functions from 01: run `vision_features` on each of the 4 tiles, reshape each one to `(B, hp, wp, Dv)`, and write them into a `(B, 2hp, 2wp, Dv)` grid at `[ti*hp:(ti+1)*hp, tj*wp:(tj+1)*wp]`.
---
A stack is not a tile. Tile 0 covers only the top-left quarter of the image, while each stack must cover the whole image at low-res density. Project the grid first, then take `g[:, di::2, dj::2].reshape(B, N, D)`.
---
Get `vis_mask` from `merge_image_tokens` (pass dummy labels). In the layer loop, *before* calling `decoder_block`: `k = (idx - l_start) // n`, and if `idx >= l_start`, `(idx - l_start) % n == 0` and `k < 4`, do `H[vis] += stacks[k].reshape(-1, D)`. Boolean indexing reads row by row, which matches stacking the batch.
