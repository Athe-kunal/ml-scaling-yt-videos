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


def vit_encoder_block(x, num_heads, Wq, Wk, Wv, Wo, W1, W2):
    """Pre-LN block: x + MHA(LN(x)) @ Wo, then + GELU(LN(.) @ W1) @ W2.  x: (B, N, D)"""
    raise NotImplementedError


def classification_head(encoder_output, W_head):
    """LN(CLS token) @ W_head -> (B, num_classes)"""
    raise NotImplementedError


def vit_forward(image, patch_size, num_heads, W_patch, patch_bias,
                cls_token, pos_embed, encoder_weights, W_head):
    """image: (B, H, W, C) -> logits (B, num_classes)"""
    raise NotImplementedError
