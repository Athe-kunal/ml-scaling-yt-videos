"""Only the image->text cross-entropy."""
import math

import numpy as np


# ----------------------------------------------------------------------------- building blocks
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def layer_norm(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def quick_gelu(x):  # CLIP's activation: x * sigmoid(1.702 x)
    return x / (1.0 + np.exp(-1.702 * x))


def l2_normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def encoder_block(x, num_heads, blk, mask=None):
    """Pre-LN transformer block. mask: (N, N) additive (0 or -inf), or None."""
    B, N, D = x.shape
    H, dh = num_heads, D // num_heads

    h = layer_norm(x)
    split = lambda t: t.reshape(B, N, H, dh).transpose(0, 2, 1, 3)       # (B, H, N, dh)
    Q, K, V = split(h @ blk['Wq']), split(h @ blk['Wk']), split(h @ blk['Wv'])

    scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh)                 # (B, H, N, N)
    if mask is not None:
        scores = scores + mask                                           # -inf above the diagonal
    attn = (softmax(scores, axis=-1) @ V).transpose(0, 2, 1, 3).reshape(B, N, D)

    y = x + attn @ blk['Wo']
    return y + quick_gelu(layer_norm(y) @ blk['W1']) @ blk['W2']


# ----------------------------------------------------------------------------- image tower
def clip_image_encoder(image, patch_size, num_heads, W_patch, cls_token, pos_embed, encoder_weights, W_proj):
    B, H, W, C = image.shape
    P = patch_size
    hp, wp = H // P, W // P

    x = image[:, :hp * P, :wp * P, :].astype(np.float64)
    x = x.reshape(B, hp, P, wp, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, hp * wp, P * P * C)
    x = x @ W_patch                                                  # CLIP's patch conv has NO bias
    D = x.shape[-1]

    cls = np.broadcast_to(cls_token.reshape(1, 1, D), (B, 1, D))
    x = np.concatenate((cls, x), axis=1) + pos_embed                 # (B, N+1, D)
    x = layer_norm(x)                                                # ln_pre: CLIP's one change to ViT

    for blk in encoder_weights:
        x = encoder_block(x, num_heads, blk)                         # no mask: images are bidirectional

    return layer_norm(x[:, 0, :]) @ W_proj                           # ln_post on CLS, then linear projection


# ----------------------------------------------------------------------------- text tower
def clip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj):
    B, L = tokens.shape
    x = token_embed[tokens] + pos_embed[:L]                          # (B, L, D)

    mask = np.triu(np.full((L, L), -np.inf), k=1)                    # causal: token i sees j <= i
    for blk in encoder_weights:
        x = encoder_block(x, num_heads, blk, mask)

    x = layer_norm(x)                                                # ln_final
    eos = tokens.argmax(axis=-1)                                     # [EOS] has the largest id
    return x[np.arange(B), eos] @ W_proj                             # (B, E)


# ----------------------------------------------------------------------------- loss
TEMPERATURE = 0.07     # tau at initialisation (CLIP §2.5, from Wu et al. 2018)
MAX_SCALE = 100.0      # "clipped to prevent scaling the logits by more than 100"


def clip_loss(image_emb, text_emb):
    # CLIP stores the temperature as a LOG-scale parameter t = log(1/tau) and learns t during training.
    # The logits are multiplied by exp(t) = 1/tau, clipped so it never exceeds 100.
    logit_scale = np.log(1.0 / TEMPERATURE)                          # t = log(1/0.07) ≈ 2.659
    scale = min(np.exp(logit_scale), MAX_SCALE)                      # exp(t) = 1/0.07 ≈ 14.29

    I = l2_normalize(image_emb)                                      # (N, E) unit vectors
    T = l2_normalize(text_emb)
    logits = scale * I @ T.T                                         # (N, N); row i = image i vs all texts

    n = logits.shape[0]
    idx = np.arange(n)                                               # the true pairs sit on the diagonal
    loss_i2t = -log_softmax(logits, axis=1)[idx, idx].mean()         # each image picks its text (rows)
    loss_t2i = -log_softmax(logits, axis=0)[idx, idx].mean()         # each text picks its image (columns)
    return loss_i2t, logits
