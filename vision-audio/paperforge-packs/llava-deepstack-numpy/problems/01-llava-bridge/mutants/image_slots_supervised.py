"""Labels the image slots with the placeholder id instead of -100."""
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
# YOUR PART: the bridge from ViT embeddings into the language model.
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
    """
    B, T = input_ids.shape
    N = image_tokens.shape[1]
    rows_e, rows_l, rows_m = [], [], []
    for b in range(B):
        p = int(np.flatnonzero(input_ids[b] == image_token_id)[0])
        pre, post = input_ids[b, :p], input_ids[b, p + 1:]
        rows_e.append(np.concatenate([embed[pre], image_tokens[b], embed[post]]))
        rows_l.append(np.concatenate([labels[b, :p], np.full(N, image_token_id), labels[b, p + 1:]]))
        rows_m.append(np.concatenate([np.zeros(p, bool), np.ones(N, bool), np.zeros(T - p - 1, bool)]))
    return np.stack(rows_e), np.stack(rows_l), np.stack(rows_m)


def llava_forward(image, input_ids, labels, image_token_id, params):
    """-> (logits (B, L, V), loss: float)  with L = T - 1 + N."""
    f = vision_features(image.astype(np.float64), params["vision"])
    img = projector(f, params["proj"])                               # (B, N, D) in the LM's space
    x, lab, _ = merge_image_tokens(input_ids, image_token_id, params["lm"]["embed"], img, labels)

    B, L, _ = x.shape
    positions = np.broadcast_to(np.arange(L), (B, L))                # positions AFTER the splice
    for blk in params["lm"]["blocks"]:
        x = decoder_block(x, positions, params["lm"]["heads"], blk)
    logits = rms_norm(x) @ params["lm"]["lm_head"]

    # next-token loss: position t predicts label t+1; image and prompt slots carry -100
    lg, tgt = logits[:, :-1], lab[:, 1:]
    keep = tgt != -100
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    nll = -np.take_along_axis(logp, np.where(keep, tgt, 0)[..., None], -1)[..., 0]
    return logits, float(nll[keep].mean())
