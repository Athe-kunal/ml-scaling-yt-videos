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
def vision_features(image, vt):
    """image (B, H, W, C) -> the patch features (B, N, Dv) that LLaVA hands to the projector."""
    raise NotImplementedError


def projector(f, proj):
    """(B, N, Dv) -> (B, N, D): LLaVA-1.5's connector."""
    raise NotImplementedError


def merge_image_tokens(input_ids, image_token_id, embed, image_tokens, labels):
    """-> embeds (B, T-1+N, D), labels (B, T-1+N), vis_mask (B, T-1+N) bool"""
    raise NotImplementedError


def llava_forward(image, input_ids, labels, image_token_id, params):
    """-> (logits (B, T-1+N, V), loss: float)"""
    raise NotImplementedError
