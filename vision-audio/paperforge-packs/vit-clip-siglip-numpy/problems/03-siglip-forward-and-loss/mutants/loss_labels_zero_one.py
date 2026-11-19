"""Uses labels eye(n) (0/1) as if it were BCE: negatives contribute a constant."""
import math

import numpy as np


# ----------------------------------------------------------------------------- building blocks
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):  # tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def l2_normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)                                    # log(1/(1+e^-x)), never overflows


def multihead_attention(q_in, kv_in, num_heads, Wq, Wk, Wv, Wo):
    """q_in: (B, Nq, D), kv_in: (B, Nk, D) -> (B, Nq, D)."""
    B, Nq, D = q_in.shape
    Nk = kv_in.shape[1]
    H, dh = num_heads, D // num_heads
    Q = (q_in @ Wq).reshape(B, Nq, H, dh).transpose(0, 2, 1, 3)
    K = (kv_in @ Wk).reshape(B, Nk, H, dh).transpose(0, 2, 1, 3)
    V = (kv_in @ Wv).reshape(B, Nk, H, dh).transpose(0, 2, 1, 3)
    attn = softmax(Q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh), axis=-1) @ V
    return attn.transpose(0, 2, 1, 3).reshape(B, Nq, D) @ Wo


def encoder_block(x, num_heads, blk):
    """Pre-LN block, no mask (both SigLIP towers are bidirectional)."""
    h = layer_norm(x)
    y = x + multihead_attention(h, h, num_heads, blk['Wq'], blk['Wk'], blk['Wv'], blk['Wo'])
    return y + gelu(layer_norm(y) @ blk['W1']) @ blk['W2']


# ----------------------------------------------------------------------------- image tower
def map_head(x, num_heads, probe, Wq, Wk, Wv, Wo, W1, W2):
    """Multihead Attention Pooling: one LEARNED query (the probe) attends over all tokens."""
    B, N, D = x.shape
    q = np.broadcast_to(probe.reshape(1, 1, D), (B, 1, D))          # (B, 1, D), same probe for every image
    y = multihead_attention(q, x, num_heads, Wq, Wk, Wv, Wo)        # (B, 1, D)
    y = y + gelu(layer_norm(y) @ W1) @ W2                            # residual MLP
    return y[:, 0, :]                                                # (B, D)


def siglip_image_encoder(image, patch_size, num_heads, W_patch, patch_bias, pos_embed,
                         encoder_weights, map_weights, W_proj):
    B, H, W, C = image.shape
    P = patch_size
    hp, wp = H // P, W // P

    x = image[:, :hp * P, :wp * P, :].astype(np.float64)
    x = x.reshape(B, hp, P, wp, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, hp * wp, P * P * C)
    x = x @ W_patch + patch_bias + pos_embed                         # (B, N, D): NO class token

    for blk in encoder_weights:
        x = encoder_block(x, num_heads, blk)
    x = layer_norm(x)                                                # encoder_norm

    m = map_weights
    pooled = map_head(x, num_heads, m['probe'], m['Wq'], m['Wk'], m['Wv'], m['Wo'], m['W1'], m['W2'])
    return pooled @ W_proj                                           # (B, E)


# ----------------------------------------------------------------------------- text tower
def siglip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj):
    B, L = tokens.shape
    x = token_embed[tokens] + pos_embed[:L]
    for blk in encoder_weights:
        x = encoder_block(x, num_heads, blk)                         # bidirectional: no causal mask
    x = layer_norm(x)
    return x[:, -1, :] @ W_proj                                      # the LAST position (it saw every token)


# ----------------------------------------------------------------------------- loss (Algorithm 1)
def siglip_loss(image_emb, text_emb, t_prime, b):
    zimg = l2_normalize(image_emb)
    ztxt = l2_normalize(text_emb)
    logits = zimg @ ztxt.T * np.exp(t_prime) + b                    # (N, N)
    n = logits.shape[0]
    labels = np.eye(n)                         # +1 on the diagonal, -1 elsewhere
    loss = -np.sum(log_sigmoid(labels * logits)) / n                 # N^2 independent binary problems, / N
    return loss, logits
