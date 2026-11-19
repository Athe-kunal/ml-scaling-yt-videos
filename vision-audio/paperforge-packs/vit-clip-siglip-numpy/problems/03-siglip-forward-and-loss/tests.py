import math

import numpy as np
import pytest

from solution import map_head, siglip_image_encoder, siglip_loss, siglip_text_encoder

V, CTX = 20, 8


def _blocks(r, D, layers):
    bl = []
    for _ in range(layers):
        b = {k: r.normal(0, D ** -0.5, (D, D)) for k in ("Wq", "Wk", "Wv", "Wo")}
        b["W1"] = r.normal(0, D ** -0.5, (D, 4 * D))
        b["W2"] = r.normal(0, (4 * D) ** -0.5, (4 * D, D))
        bl.append(b)
    return bl


def _map_weights(r, D):
    m = _blocks(r, D, 1)[0]
    m["probe"] = r.normal(size=D)
    return m


# ---- loop-based reference --------------------------------------------------------------------
def _ln(v, eps=1e-6):
    return (v - v.mean()) / math.sqrt(v.var() + eps)


def _gelu(v):
    return 0.5 * v * (1 + np.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def _ref_mha(xq, xkv, heads, b):
    D = xq.shape[1]
    dh = D // heads
    Q, K, Vv = xq @ b["Wq"], xkv @ b["Wk"], xkv @ b["Wv"]
    out = np.zeros((len(xq), D))
    for hd in range(heads):
        s = slice(hd * dh, (hd + 1) * dh)
        for i in range(len(xq)):
            sc = np.array([Q[i, s] @ K[j, s] / math.sqrt(dh) for j in range(len(xkv))])
            w = np.exp(sc - sc.max())
            w /= w.sum()
            out[i, s] = w @ Vv[:, s]
    return out @ b["Wo"]


def _ref_block(x, heads, b):
    h = np.stack([_ln(t) for t in x])
    y = x + _ref_mha(h, h, heads, b)
    return y + _gelu(np.stack([_ln(t) for t in y]) @ b["W1"]) @ b["W2"]


def _ref_map(x, heads, m):
    y = _ref_mha(m["probe"][None], x, heads, m)[0]
    return y + _gelu(_ln(y) @ m["W1"]) @ m["W2"]


def _ref_image(p, P, heads):
    outs = []
    for img in p["image"]:
        hp, wp = img.shape[0] // P, img.shape[1] // P
        x = np.stack([img[i * P:(i + 1) * P, j * P:(j + 1) * P].reshape(-1) for i in range(hp) for j in range(wp)])
        x = x @ p["W_patch"] + p["bias"] + p["pos"]
        for b in p["blocks"]:
            x = _ref_block(x, heads, b)
        x = np.stack([_ln(t) for t in x])
        outs.append(_ref_map(x, heads, p["map"]) @ p["W_proj"])
    return np.stack(outs)


def _ref_text(p, tokens, heads):
    outs = []
    for row in tokens:
        x = p["tok"][row] + p["pos"][:len(row)]
        for b in p["blocks"]:
            x = _ref_block(x, heads, b)
        outs.append(_ln(x[-1]) @ p["W_proj"])
    return np.stack(outs)


def _ref_loss(I, T, tp, b):
    I = I / np.sqrt((I ** 2).sum(1, keepdims=True))
    T = T / np.sqrt((T ** 2).sum(1, keepdims=True))
    n = len(I)
    total = 0.0
    for i in range(n):
        for j in range(n):
            z = 1.0 if i == j else -1.0
            logit = math.exp(tp) * float(I[i] @ T[j]) + b
            total += math.log1p(math.exp(-z * logit)) if -z * logit < 30 else -z * logit
    return total / n


# ---- fixtures --------------------------------------------------------------------------------
IMG_CFGS = [(2, 8, 8, 3, 4, 8, 2, 2, 6), (3, 12, 8, 1, 4, 12, 3, 1, 5)]   # (B,H,W,C,P,D,heads,layers,E)
TXT_CFGS = [(8, 2, 2, 6), (12, 3, 1, 5)]                                   # (D, heads, layers, E)


def _img_params(cfg, seed):
    B, H, W, C, P, D, heads, layers, E = cfg
    r = np.random.default_rng(seed)
    N = (H // P) * (W // P)
    return dict(image=r.normal(size=(B, H, W, C)), W_patch=r.normal(0, (P * P * C) ** -0.5, (P * P * C, D)),
                bias=r.normal(0, 0.1, D), pos=r.normal(0, 0.5, (N, D)), blocks=_blocks(r, D, layers),
                map=_map_weights(r, D), W_proj=r.normal(0, D ** -0.5, (D, E)))


def _txt_params(cfg, seed):
    D, heads, layers, E = cfg
    r = np.random.default_rng(seed)
    return dict(tok=r.normal(size=(V, D)), pos=r.normal(0, 0.5, (CTX, D)), blocks=_blocks(r, D, layers),
                W_proj=r.normal(0, D ** -0.5, (D, E)))


def _img(p, cfg):
    return siglip_image_encoder(p["image"], cfg[4], cfg[6], p["W_patch"], p["bias"], p["pos"], p["blocks"],
                                p["map"], p["W_proj"])


def _txt(p, tokens, cfg):
    return siglip_text_encoder(tokens, cfg[1], p["tok"], p["pos"], p["blocks"], p["W_proj"])


def _map(x, heads, m):
    return map_head(x, heads, m["probe"], m["Wq"], m["Wk"], m["Wv"], m["Wo"], m["W1"], m["W2"])


# ---- MAP head --------------------------------------------------------------------------------
@pytest.mark.parametrize("D,heads,N", [(8, 2, 5), (12, 3, 7), (6, 1, 1)])
def test_map_head_matches_reference(D, heads, N):
    r = np.random.default_rng(D + N)
    m = _map_weights(r, D)
    x = r.normal(size=(3, N, D))
    out = np.asarray(_map(x, heads, m))
    ref = np.stack([_ref_map(xi, heads, m) for xi in x])
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape} (one pooled vector per image)"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. MAP: the learned probe (projected by Wq) is the only query; keys and "
        f"values are all tokens; then a residual MLP: y + gelu(LN(y) @ W1) @ W2"
    )


def test_map_head_is_permutation_invariant_and_uses_the_probe():
    r = np.random.default_rng(20)
    m = _map_weights(r, 8)
    x = r.normal(size=(2, 6, 8))
    a = np.asarray(_map(x, 2, m))
    b = np.asarray(_map(x[:, r.permutation(6)], 2, m))
    assert np.allclose(a, b, atol=1e-12), "MAP pools a SET of tokens: shuffling them must not change the output"
    m2 = dict(m, probe=m["probe"] + 1.0)
    assert not np.allclose(a, np.asarray(_map(x, 2, m2))), "changing the probe did nothing: the probe must be the query"


def test_map_head_is_not_mean_pooling():
    r = np.random.default_rng(21)
    m = _map_weights(r, 8)
    x = r.normal(size=(1, 6, 8))
    d = r.normal(size=(1, 6, 8))
    d -= d.mean(1, keepdims=True)                                 # same token mean, different tokens
    assert np.abs(np.asarray(_map(x, 2, m)) - np.asarray(_map(x + 2 * d, 2, m))).max() > 1e-4, (
        "output depends only on the token mean: that is mean pooling, not attention pooling"
    )


# ---- towers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", IMG_CFGS)
def test_image_encoder_matches_reference(cfg):
    p = _img_params(cfg, 0)
    out, ref = np.asarray(_img(p, cfg)), _ref_image(p, cfg[4], cfg[6])
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. SigLIP-ViT: patches @ W_patch + bias + pos (no CLS token) -> blocks "
        f"-> LN -> MAP head -> @ W_proj"
    )


@pytest.mark.parametrize("cfg", TXT_CFGS)
def test_text_encoder_matches_reference(cfg):
    p = _txt_params(cfg, 1)
    tokens = np.random.default_rng(2).integers(1, V, (3, CTX))
    out, ref = np.asarray(_txt(p, tokens, cfg)), _ref_text(p, tokens, cfg[1])
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. SigLIP text: bidirectional (no mask) blocks, final LN, the LAST "
        f"position, @ W_proj"
    )


def test_text_last_position_sees_the_first_token():
    cfg = TXT_CFGS[0]
    p = _txt_params(cfg, 3)
    tokens = np.random.default_rng(4).integers(1, V, (1, CTX))
    base = np.asarray(_txt(p, tokens, cfg))
    t2 = tokens.copy()
    t2[0, 0] = 1 + t2[0, 0] % (V - 2)
    assert np.abs(np.asarray(_txt(p, t2, cfg)) - base).max() > 1e-8, "the first token has no influence"


# ---- loss (Algorithm 1) ----------------------------------------------------------------------
@pytest.mark.parametrize("n,e,tp,b", [(6, 5, math.log(10), -10.0), (5, 3, 1.0, 0.5), (9, 8, math.log(30), -4.0)])
def test_loss_matches_reference(n, e, tp, b):
    r = np.random.default_rng(n)
    I, T = r.normal(size=(n, e)), r.normal(size=(n, e))
    loss, logits = siglip_loss(I, T, tp, b)
    In = I / np.linalg.norm(I, axis=1, keepdims=True)
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    assert np.allclose(logits, math.exp(tp) * In @ Tn.T + b, atol=1e-10), "logits = exp(t') * cos + b"
    ref = _ref_loss(I, T, tp, b)
    assert abs(float(loss) - ref) < 1e-9, (
        f"loss {float(loss):.6f} != {ref:.6f}: -sum over ALL N^2 pairs of log sigmoid(z_ij * logit_ij), divided by N, "
        f"with z = +1 on the diagonal and -1 elsewhere"
    )


def test_loss_has_no_global_normalisation():
    """Adding pair N only adds the terms in row N and column N; the other pairs' terms are untouched."""
    r = np.random.default_rng(30)
    I, T = r.normal(size=(6, 4)), r.normal(size=(6, 4))
    tp, b = 1.5, -2.0
    full, L = siglip_loss(I, T, tp, b)
    part, _ = siglip_loss(I[:5], T[:5], tp, b)
    added = 6 * float(full) - 5 * float(part)
    ls = lambda x: -math.log1p(math.exp(-x))
    expect = -(ls(L[5, 5]) + sum(ls(-L[5, j]) + ls(-L[j, 5]) for j in range(5)))
    assert abs(added - expect) < 1e-9, (
        "N*loss(N) - (N-1)*loss(N-1) should be exactly the new row and column of binary terms. "
        "A softmax-normalised (CLIP-style) loss or a mean over N^2 breaks this."
    )


def test_loss_is_symmetric_and_stable():
    r = np.random.default_rng(31)
    I, T = r.normal(size=(7, 4)), r.normal(size=(7, 4))
    a, _ = siglip_loss(I, T, 2.0, -3.0)
    b, _ = siglip_loss(T, I, 2.0, -3.0)
    assert abs(float(a) - float(b)) < 1e-12, "the sigmoid loss is symmetric in images and texts"
    big, _ = siglip_loss(I, -I, math.log(2000.0), 0.0)
    assert np.isfinite(big), "log_sigmoid overflowed at |logit| ~ 2000: use -logaddexp(0, -x)"
    exp_big = _ref_loss(I, -I, math.log(2000.0), 0.0)
    assert abs(float(big) - exp_big) / exp_big < 1e-9, "large-logit loss is wrong"


def test_init_bias_makes_negatives_cheap():
    """With t'=log 10, b=-10 (the paper's init), almost every off-diagonal pair is already 'correct'."""
    r = np.random.default_rng(32)
    n = 16
    I, T = r.normal(size=(n, 8)), r.normal(size=(n, 8))
    loss, logits = siglip_loss(I, T, math.log(10), -10.0)
    assert np.all(logits < 0), "with b = -10 and |t·cos| <= 10 every logit is <= 0"
    ref = _ref_loss(I, T, math.log(10), -10.0)
    assert abs(float(loss) - ref) < 1e-9
