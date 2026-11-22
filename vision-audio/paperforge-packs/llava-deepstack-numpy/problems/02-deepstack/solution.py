# ---------------------------------------------------------------------------------------------
# GIVEN: already-solved pieces (ViT pack 01/02 + a LLaMA-style decoder). You do not edit these.
# ---------------------------------------------------------------------------------------------
import math

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):                       # no gain / bias, population variance
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(-1, keepdims=True) + eps)


def rms_norm(x, eps=1e-6):                         # no gain
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def quick_gelu(x):                                 # CLIP's activation
    return x / (1.0 + np.exp(-1.702 * x))


_erf = np.vectorize(math.erf)


def gelu(x):                                       # exact (erf) GELU = torch nn.GELU()
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def _split(t, H):                                  # (B, N, D) -> (B, H, N, dh)
    B, N, D = t.shape
    return t.reshape(B, N, H, D // H).transpose(0, 2, 1, 3)


def _merge(t):                                     # (B, H, N, dh) -> (B, N, D)
    B, H, N, dh = t.shape
    return t.transpose(0, 2, 1, 3).reshape(B, N, H * dh)


def clip_block(x, num_heads, blk):
    """CLIP-ViT block: pre-LN, bidirectional MHA, QuickGELU MLP.  x: (B, N, Dv)"""
    h = layer_norm(x)
    Q, K, V = (_split(h @ blk[k], num_heads) for k in ("Wq", "Wk", "Wv"))
    a = softmax(Q @ K.transpose(0, 1, 3, 2) / math.sqrt(Q.shape[-1])) @ V
    x = x + _merge(a) @ blk["Wo"]
    return x + quick_gelu(layer_norm(x) @ blk["W1"]) @ blk["W2"]


def rope(t, positions, theta=10000.0):
    """Rotary embedding, HF 'rotate_half' layout.  t: (B, H, L, dh), positions: int (B, L)."""
    dh = t.shape[-1]
    inv = theta ** (-np.arange(0, dh, 2) / dh)                     # (dh/2,)
    ang = positions[:, None, :, None].astype(np.float64) * inv     # (B, 1, L, dh/2)
    cos, sin = np.cos(ang), np.sin(ang)
    cos, sin = np.concatenate([cos, cos], -1), np.concatenate([sin, sin], -1)
    t1, t2 = t[..., : dh // 2], t[..., dh // 2:]
    return t * cos + np.concatenate([-t2, t1], -1) * sin


def decoder_block(x, positions, num_heads, blk):
    """LLaMA block: pre-RMSNorm, causal MHA with RoPE, SwiGLU MLP.  x: (B, L, D)."""
    L = x.shape[1]
    h = rms_norm(x)
    Q, K, V = (_split(h @ blk[k], num_heads) for k in ("Wq", "Wk", "Wv"))
    Q, K = rope(Q, positions), rope(K, positions)
    s = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(Q.shape[-1])
    s = np.where(np.tril(np.ones((L, L), bool)), s, -np.inf)       # causal
    x = x + _merge(softmax(s) @ V) @ blk["Wo"]
    h = rms_norm(x)
    return x + (silu(h @ blk["Wg"]) * (h @ blk["Wu"])) @ blk["Wd"]


# ---------------------------------------------------------------------------------------------
# FROM PROBLEM 01: your LLaVA bridge (paste your solution of 01 here).
# ---------------------------------------------------------------------------------------------
SELECT_LAYER = -2          # LLaVA: mm_vision_select_layer = -2  (penultimate hidden state)


def vision_features(image, vt):
    """CLIP-ViT tower -> patch features (B, N, Dv) handed to the projector."""
    B, H, W, C = image.shape
    P = vt["patch"]
    hp, wp = H // P, W // P
    x = image.reshape(B, hp, P, wp, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, hp * wp, P * P * C)
    x = x @ vt["W_patch"]                                            # CLIP: no patch bias
    cls = np.broadcast_to(vt["cls"], (B, 1, x.shape[-1]))
    x = layer_norm(np.concatenate([cls, x], 1) + vt["pos"])          # ln_pre

    hidden_states = [x]                                              # hs[0] = embeddings (after ln_pre)
    for blk in vt["blocks"]:
        x = clip_block(x, vt["heads"], blk)
        hidden_states.append(x)                                      # hs[i] = output of block i

    f = hidden_states[SELECT_LAYER]        # penultimate block output; ln_post is NOT applied
    return f[:, 1:]                        # select_feature='patch': drop CLS -> (B, N, Dv)


def projector(f, proj):
    """LLaVA-1.5 'mlp2x_gelu': Linear -> GELU -> Linear, applied to every patch independently."""
    return gelu(f @ proj["W1"] + proj["b1"]) @ proj["W2"] + proj["b2"]


def merge_image_tokens(input_ids, image_token_id, embed, image_tokens, labels):
    """Replace the single <image> placeholder in each row by the N projected image tokens.

    Returns embeds (B, T-1+N, D), labels (B, T-1+N) with -100 on image slots, vis_mask (B, T-1+N) bool.

    Worked example (one row, <image> = 7, N = 4):
        input_ids = [12, 7, 9, 15, 11, 10]              T = 6
        labels    = [-100, -100, -100, -100, 11, 10]    only the answer "11 10" is trained

        position:  0     1    2    3    4    5    6    7    8
        embeds:   e12 | v0   v1   v2   v3 | e9  e15  e11  e10      [9, D] = T - 1 + N
        labels:   -100| -100 -100 -100 -100| -100 -100  11   10
        vis_mask:  F  |  T    T    T    T  |  F    F    F    F

    The same cut (:p | insert N | p+1:) is applied to embeds, labels and the mask, so every label
    stays next to its token. DeepStack needs vis_mask: the high-res stacks are added at exactly
    these positions (H[vis] += stack) before the chosen LM layers.
    """
    B, T = input_ids.shape                      # e.g. 1, 6
    N = image_tokens.shape[1]                   # e.g. 4   (image_tokens: [B, N, D] from the projector)
    rows_e, rows_l, rows_m = [], [], []
    # row by row: each row may have its <image> at a different index, so no single slice works
    for b in range(B):
        # where is the placeholder?  ids == 7 -> [F, T, F, F, F, F] -> flatnonzero -> [1] -> p = 1
        p = int(np.flatnonzero(input_ids[b] == image_token_id)[0])
        # split around it, dropping the placeholder itself:  pre = [12],  post = [9, 15, 11, 10]
        pre, post = input_ids[b, :p], input_ids[b, p + 1:]
        # what the LM sees: embed(pre) [1, D] + image vectors [4, D] + embed(post) [4, D] -> [9, D]
        # (embed[7], the placeholder's own embedding, is never used)
        rows_e.append(np.concatenate([embed[pre], image_tokens[b], embed[post]]))
        # same cut for labels; image slots get -100 so cross-entropy never asks the LM to predict pixels
        rows_l.append(np.concatenate([labels[b, :p], np.full(N, -100), labels[b, p + 1:]]))
        # which positions are image: p False, N True, T-p-1 False  ->  [F, T, T, T, T, F, F, F, F]
        rows_m.append(np.concatenate([np.zeros(p, bool), np.ones(N, bool), np.zeros(T - p - 1, bool)]))
    # every row grew by the same N - 1, so the rows stack back into a batch: [B, 9, D], [B, 9], [B, 9]
    # positions are computed AFTER this (0..L-1), so text after the image sits at 5, not 2
    return np.stack(rows_e), np.stack(rows_l), np.stack(rows_m)


# ---------------------------------------------------------------------------------------------
# YOUR PART: DeepStack (Algorithm 1).
# ---------------------------------------------------------------------------------------------
def hires_feature_grid(image_hires, vt):
    """(B, 2H, 2W, C) -> (B, 2hp, 2wp, Dv): encode the 2x2 tiles with the SAME ViT, stitch them back."""
    B, H2, W2, C = image_hires.shape
    H, W = H2 // 2, W2 // 2
    P = vt["patch"]
    hp, wp = H // P, W // P
    grid = np.zeros((B, 2 * hp, 2 * wp, vt["W_patch"].shape[1]))
    for ti in range(2):
        for tj in range(2):
            tile = image_hires[:, ti * H:(ti + 1) * H, tj * W:(tj + 1) * W]
            f = vision_features(tile, vt).reshape(B, hp, wp, -1)          # row-major patches inside a tile
            grid[:, ti * hp:(ti + 1) * hp, tj * wp:(tj + 1) * wp] = f
    return grid


def deepstack_tokens(image_hires, vt, proj):
    """-> X_stack (4, B, N, D). Stack k = (di, dj) in row-major order (0,0),(0,1),(1,0),(1,1).

    Sampling2D with dilation 2: stack (di, dj) at low-res cell (i, j) is the high-res token
    (2i+di, 2j+dj), i.e. one of the 4 high-res neighbours that sit inside that low-res cell.
    """
    g = projector(hires_feature_grid(image_hires, vt), proj)               # same connector M
    B, H2, W2, D = g.shape
    return np.stack([g[:, di::2, dj::2].reshape(B, -1, D) for di in range(2) for dj in range(2)])


def deepstack_forward(image, image_hires, input_ids, image_token_id, params, l_start, n):
    """-> logits (B, L, V), L = T - 1 + N (the same context length as plain LLaVA)."""
    img = projector(vision_features(image.astype(np.float64), params["vision"]), params["proj"])
    stacks = deepstack_tokens(image_hires.astype(np.float64), params["vision"], params["proj"])
    dummy = np.full(input_ids.shape, -100)
    H, _, vis = merge_image_tokens(input_ids, image_token_id, params["lm"]["embed"], img, dummy)

    B, L, D = H.shape
    positions = np.broadcast_to(np.arange(L), (B, L))
    for idx, blk in enumerate(params["lm"]["blocks"]):
        k = (idx - l_start) // n
        if idx >= l_start and (idx - l_start) % n == 0 and k < len(stacks):
            H = H.copy()
            H[vis] += stacks[k].reshape(-1, D)          # residual add at the visual positions, BEFORE layer idx
        H = decoder_block(H, positions, params["lm"]["heads"], blk)
    return rms_norm(H) @ params["lm"]["lm_head"]
