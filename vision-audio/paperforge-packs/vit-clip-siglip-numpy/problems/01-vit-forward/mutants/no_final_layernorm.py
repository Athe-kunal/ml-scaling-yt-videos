"""Skips the LayerNorm before the classification head."""
import math

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # subtract the max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                   # population variance (ddof=0)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):  # tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def vit_encoder_block(x, num_heads, Wq, Wk, Wv, Wo, W1, W2):
    x = x.astype(np.float64)
    B, N, D = x.shape
    H = num_heads
    dh = D // H

    h = layer_norm(x)                                                 # pre-LN
    split = lambda t: t.reshape(B, N, H, dh).transpose(0, 2, 1, 3)   # (B, H, N, dh)
    Q, K, V = split(h @ Wq), split(h @ Wk), split(h @ Wv)

    attn = softmax(Q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh), axis=-1) @ V
    attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)               # merge heads

    y = x + attn @ Wo                                                 # residual 1
    return y + gelu(layer_norm(y) @ W1) @ W2                          # residual 2 (pre-LN MLP)


def classification_head(encoder_output, W_head):
    h = encoder_output[:, 0, :].astype(np.float64)                   # CLS token (B, D)
    return h @ W_head.astype(np.float64)                 # (B, C)


def vit_forward(image, patch_size, num_heads, W_patch, patch_bias,
                cls_token, pos_embed, encoder_weights, W_head):
    B, H, W, C = image.shape
    P = patch_size
    hp, wp = H // P, W // P

    # patchify + embed
    x = image[:, :hp * P, :wp * P, :].astype(np.float64)
    x = x.reshape(B, hp, P, wp, P, C).transpose(0, 1, 3, 2, 4, 5)   # (B, hp, wp, P, P, C)
    x = x.reshape(B, hp * wp, P * P * C)
    x = x @ W_patch + patch_bias                                     # (B, N, D)
    D = x.shape[-1]

    # prepend CLS, then add positions for all N+1 tokens
    cls = np.broadcast_to(cls_token.reshape(1, 1, D), (B, 1, D))
    x = np.concatenate((cls, x), axis=1) + pos_embed                 # (B, N+1, D)

    # encoder stack: each block consumes the previous output
    for blk in encoder_weights:
        x = vit_encoder_block(x, num_heads,
                              blk['Wq'], blk['Wk'], blk['Wv'],
                              blk['Wo'], blk['W1'], blk['W2'])

    return classification_head(x, W_head)                            # (B, C)
