import math

import numpy as np
import pytest

from solution import llava_forward, merge_image_tokens, projector, vision_features

IMG = 7                                  # <image> placeholder id
# (B, H, W, C, P, Dv, vheads, vlayers, D, heads, layers, V, T)
CONFIGS = [
    (2, 8, 8, 3, 4, 8, 2, 3, 12, 3, 2, 20, 6),
    (1, 12, 8, 1, 4, 6, 3, 2, 8, 2, 3, 16, 5),     # non-square, 1 channel, 6 patches
    (3, 8, 12, 3, 4, 8, 4, 4, 16, 4, 2, 24, 7),
]


def make(cfg, seed):
    B, H, W, C, P, Dv, vh, vl, D, h, nl, V, T = cfg
    r = np.random.default_rng(seed)
    N = (H // P) * (W // P)
    n = lambda *s: r.normal(0, s[0] ** -0.5, s)
    vblocks = [dict(Wq=n(Dv, Dv), Wk=n(Dv, Dv), Wv=n(Dv, Dv), Wo=n(Dv, Dv), W1=n(Dv, 4 * Dv), W2=n(4 * Dv, Dv))
               for _ in range(vl)]
    lblocks = [dict(Wq=n(D, D), Wk=n(D, D), Wv=n(D, D), Wo=n(D, D), Wg=n(D, 2 * D), Wu=n(D, 2 * D), Wd=n(2 * D, D))
               for _ in range(nl)]
    params = dict(
        vision=dict(patch=P, heads=vh, W_patch=n(P * P * C, Dv), cls=r.normal(size=Dv),
                    pos=r.normal(0, 0.5, (N + 1, Dv)), blocks=vblocks),
        proj=dict(W1=n(Dv, D) * 2, b1=r.normal(0, 0.3, D), W2=n(D, D), b2=r.normal(0, 0.3, D)),
        lm=dict(heads=h, embed=r.normal(size=(V, D)), blocks=lblocks, lm_head=n(D, V)),
    )
    ids = r.integers(8, V, size=(B, T))
    for b in range(B):                                  # one placeholder per row, at different places
        ids[b, (b + 1) % (T - 1)] = IMG
    labels = ids.copy()
    for b in range(B):                                  # supervise only the last 2 tokens (the "answer")
        labels[b, : T - 2] = -100
    return r.normal(size=(B, H, W, C)), ids, labels, params


# ---------------- independent, loop-based reference ------------------------------------------
def _ln(v):
    return (v - v.mean(-1, keepdims=True)) / np.sqrt(v.var(-1, keepdims=True) + 1e-6)


def _rms(v):
    return v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-6)


def _sm(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _attn(Q, K, V, heads, causal, pos=None):
    N, D = Q.shape
    dh = D // heads
    out = np.zeros_like(Q)
    for hd in range(heads):
        s = slice(hd * dh, (hd + 1) * dh)
        q, k = Q[:, s].copy(), K[:, s].copy()
        if pos is not None:
            for t in range(N):
                for i in range(dh // 2):
                    a = pos[t] * 10000.0 ** (-2 * i / dh)
                    c, si = math.cos(a), math.sin(a)
                    for m in (q, k):
                        x1, x2 = m[t, i], m[t, i + dh // 2]
                        m[t, i], m[t, i + dh // 2] = x1 * c - x2 * si, x2 * c + x1 * si
        sc = q @ k.T / math.sqrt(dh)
        if causal:
            sc[np.triu_indices(N, 1)] = -np.inf
        out[:, s] = _sm(sc) @ V[:, s]
    return out


def ref_vision(img, vt):
    P = vt["patch"]
    hp, wp = img.shape[0] // P, img.shape[1] // P
    toks = [img[i * P:(i + 1) * P, j * P:(j + 1) * P].reshape(-1) for i in range(hp) for j in range(wp)]
    x = _ln(np.vstack([vt["cls"], np.stack(toks) @ vt["W_patch"]]) + vt["pos"])
    hs = [x]
    for b in vt["blocks"]:
        h = _ln(x)
        x = x + _attn(h @ b["Wq"], h @ b["Wk"], h @ b["Wv"], vt["heads"], False) @ b["Wo"]
        hh = _ln(x) @ b["W1"]
        x = x + (hh / (1 + np.exp(-1.702 * hh))) @ b["W2"]
        hs.append(x)
    return hs


def ref_gelu(v):
    return 0.5 * v * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))


def ref_proj(f, p):
    return ref_gelu(f @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def ref_lm(x, lm):
    pos = list(range(len(x)))
    for b in lm["blocks"]:
        h = _rms(x)
        x = x + _attn(h @ b["Wq"], h @ b["Wk"], h @ b["Wv"], lm["heads"], True, pos) @ b["Wo"]
        h = _rms(x)
        g = h @ b["Wg"]
        x = x + (g / (1 + np.exp(-g)) * (h @ b["Wu"])) @ b["Wd"]
    return _rms(x) @ lm["lm_head"]


def ref_forward(image, ids, labels, params, inject=None):
    outs, total, count = [], 0.0, 0
    for b in range(len(ids)):
        img_tok = ref_proj(ref_vision(image[b], params["vision"])[-2][1:], params["proj"])
        seq, lab = [], []
        for t, tok in enumerate(ids[b]):
            if tok == IMG:
                seq += list(img_tok); lab += [-100] * len(img_tok)
            else:
                seq.append(params["lm"]["embed"][tok]); lab.append(labels[b, t])
        lg = ref_lm(np.stack(seq), params["lm"])
        for t in range(len(lab) - 1):
            if lab[t + 1] != -100:
                z = lg[t] - lg[t].max()
                total += -(z[lab[t + 1]] - math.log(np.exp(z).sum())); count += 1
        outs.append(lg)
    return np.stack(outs), total / count


def run(cfg, seed=0):
    image, ids, labels, params = make(cfg, seed)
    return image, ids, labels, params, llava_forward(image, ids, labels, IMG, params)


# ---------------- tests ------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS)
def test_vision_features_are_penultimate_patch_tokens(cfg):
    image, _, _, params = make(cfg, 0)
    f = np.asarray(vision_features(image, params["vision"]))
    hs = [ref_vision(im, params["vision"]) for im in image]
    N = (cfg[1] // cfg[4]) * (cfg[2] // cfg[4])
    assert f.shape == (cfg[0], N, cfg[5]), f"features shape {f.shape}, want (B, N={N}, Dv): drop CLS"
    if np.allclose(f, np.stack([h[-1][1:] for h in hs]), atol=1e-9):
        pytest.fail("these are the LAST block's outputs; LLaVA takes hidden_states[-2] (penultimate)")
    if np.allclose(f, np.stack([_ln(h[-2][1:]) for h in hs]), atol=1e-9):
        pytest.fail("ln_post was applied: LLaVA reads the raw hidden state, before ln_post")
    assert np.allclose(f, np.stack([h[-2][1:] for h in hs]), atol=1e-9), (
        f"max err {np.abs(f - np.stack([h[-2][1:] for h in hs])).max():.2e}: want hidden_states[-2][:, 1:], "
        "where hidden_states[0] = ln_pre(embeddings) and hidden_states[i] = output of block i")


def test_projector_is_two_layer_gelu_mlp():
    _, _, _, params = make(CONFIGS[0], 1)
    f = np.random.default_rng(5).normal(size=(2, 4, 8))
    out = np.asarray(projector(f, params["proj"]))
    assert np.allclose(out, ref_proj(f, params["proj"]), atol=1e-9), (
        "projector must be Linear(W1,b1) -> exact GELU -> Linear(W2,b2) (LLaVA-1.5 mlp2x_gelu)")
    # tokens are projected independently: permuting patches permutes outputs
    perm = [3, 0, 2, 1]
    assert np.allclose(np.asarray(projector(f[:, perm], params["proj"])), out[:, perm], atol=1e-12), \
        "the projector mixed patches; it acts on each patch vector on its own"


def test_merge_puts_image_tokens_at_the_placeholder():
    _, ids, labels, params = make(CONFIGS[2], 2)
    B, T = ids.shape
    N, D = 5, params["lm"]["embed"].shape[1]
    img = np.random.default_rng(3).normal(size=(B, N, D))
    e, lab, m = (np.asarray(a) for a in merge_image_tokens(ids, IMG, params["lm"]["embed"], img, labels))
    assert e.shape == (B, T - 1 + N, D), f"embeds {e.shape}: the placeholder is REPLACED, so L = T - 1 + N"
    for b in range(B):
        p = int(np.flatnonzero(ids[b] == IMG)[0])
        assert np.allclose(e[b, p:p + N], img[b]), f"row {b}: image tokens must sit at the placeholder index {p}"
        assert np.allclose(e[b, :p], params["lm"]["embed"][ids[b, :p]]), f"row {b}: text before <image> changed"
        assert np.allclose(e[b, p + N:], params["lm"]["embed"][ids[b, p + 1:]]), f"row {b}: text after <image>"
        assert m[b].sum() == N and m[b, p:p + N].all(), f"row {b}: vis_mask should mark exactly the {N} image slots"
        assert (lab[b, p:p + N] == -100).all(), f"row {b}: image slots must be ignored in the loss (-100)"
        assert np.array_equal(np.delete(lab[b], range(p, p + N)), np.delete(labels[b], p)), \
            f"row {b}: the text labels must shift along with the text"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_matches_reference(cfg):
    image, ids, labels, params, (logits, loss) = run(cfg)
    ref_logits, ref_loss = ref_forward(image, ids, labels, params)
    logits = np.asarray(logits)
    assert logits.shape == ref_logits.shape, f"logits {logits.shape} != {ref_logits.shape}"
    assert np.allclose(logits, ref_logits, atol=1e-8), (
        f"max err {np.abs(logits - ref_logits).max():.2e}. Check RoPE positions: they are 0..L-1 over the "
        "spliced sequence, so text after the image is shifted by N-1")
    assert abs(loss - ref_loss) < 1e-8, (
        f"loss {loss:.6f} vs {ref_loss:.6f}: logits[t] predicts labels[t+1], mean over non -100 targets only")


def test_text_before_image_cannot_see_the_image():
    cfg = CONFIGS[0]
    image, ids, labels, params, (a, _) = run(cfg, seed=4)
    image2 = np.random.default_rng(11).normal(size=image.shape)
    b_, _ = llava_forward(image2, ids, labels, IMG, params)
    a, b_ = np.asarray(a), np.asarray(b_)
    for b in range(len(ids)):
        p = int(np.flatnonzero(ids[b] == IMG)[0])
        assert np.allclose(a[b, :p], b_[b, :p], atol=1e-12), \
            f"row {b}: tokens before the image changed with the image: the image must be spliced in place, causally"
        assert not np.allclose(a[b, -1], b_[b, -1]), f"row {b}: the last token ignores the image"
