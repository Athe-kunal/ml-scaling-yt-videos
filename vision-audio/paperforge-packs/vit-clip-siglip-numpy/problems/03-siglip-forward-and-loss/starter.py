import math

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`."""
    raise NotImplementedError


def layer_norm(x, eps=1e-6):
    """LayerNorm over the last axis, no learnable gain/bias, population variance."""
    raise NotImplementedError


def gelu(x):
    """GELU, tanh approximation."""
    raise NotImplementedError


def l2_normalize(x, axis=-1):
    """Divide each vector along `axis` by its L2 norm."""
    raise NotImplementedError


def log_sigmoid(x):
    """log(sigmoid(x)) that never overflows."""
    raise NotImplementedError


def multihead_attention(q_in, kv_in, num_heads, Wq, Wk, Wv, Wo):
    """q_in: (B, Nq, D), kv_in: (B, Nk, D) -> (B, Nq, D)"""
    raise NotImplementedError


def encoder_block(x, num_heads, blk):
    """Pre-LN transformer block (see statement)."""
    raise NotImplementedError


def map_head(x, num_heads, probe, Wq, Wk, Wv, Wo, W1, W2):
    """x: (B, N, D) -> pooled (B, D)"""
    raise NotImplementedError


def siglip_image_encoder(image, patch_size, num_heads, W_patch, patch_bias, pos_embed,
                         encoder_weights, map_weights, W_proj):
    """image: (B, H, W, C) -> image embedding (B, E)"""
    raise NotImplementedError


def siglip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj):
    """tokens: int (B, L) -> text embedding (B, E)"""
    raise NotImplementedError


def siglip_loss(image_emb, text_emb, t_prime, b):
    """Returns (loss: float, logits: (N, N)). Algorithm 1 of the paper."""
    raise NotImplementedError
