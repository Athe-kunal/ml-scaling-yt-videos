import math

import numpy as np
import pytest

from solution import deepstack_forward, deepstack_tokens

IMG = 7
# (B, H, W, C, P, Dv, vheads, vlayers, D, heads, layers, V, T)
CONFIGS = [
    (2, 8, 8, 3, 4, 8, 2, 3, 12, 3, 5, 20, 6),
    (1, 8, 12, 1, 4, 6, 3, 2, 8, 2, 6, 16, 5),     # non-square: low-res 2x3 grid, high-res 4x6
    (3, 8, 8, 3, 4, 8, 4, 2, 16, 4, 5, 24, 7),
]
SCHEDULES = [(0, 1), (1, 1), (1, 2), (2, 1)]      # (l_start, n)


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




def ref_lm_stack(x, lm, vis, stacks, l_start, n):
    pos = list(range(len(x)))
    for idx, b in enumerate(lm["blocks"]):
        if idx >= l_start and (idx - l_start) % n == 0 and (idx - l_start) // n < len(stacks):
            x = x.copy(); x[vis] = x[vis] + stacks[(idx - l_start) // n]
        h = _rms(x)
        x = x + _attn(h @ b["Wq"], h @ b["Wk"], h @ b["Wv"], lm["heads"], True, pos) @ b["Wo"]
        h = _rms(x)
        g = h @ b["Wg"]
        x = x + (g / (1 + np.exp(-g)) * (h @ b["Wu"])) @ b["Wd"]
    return _rms(x) @ lm["lm_head"]


def ref_stacks(img_hi, params):
    """Loop version: stack (di,dj) at low-res cell (i,j) = high-res token (2i+di, 2j+dj)."""
    vt, P = params["vision"], params["vision"]["patch"]
    H, W = img_hi.shape[0] // 2, img_hi.shape[1] // 2
    hp, wp = H // P, W // P
    tiles = {(ti, tj): ref_proj(ref_vision(img_hi[ti * H:(ti + 1) * H, tj * W:(tj + 1) * W], vt)[-2][1:],
                                params["proj"]) for ti in range(2) for tj in range(2)}
    out = []
    for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        toks = []
        for i in range(hp):
            for j in range(wp):
                r, c = 2 * i + di, 2 * j + dj
                toks.append(tiles[(r // hp, c // wp)][(r % hp) * wp + (c % wp)])
        out.append(np.stack(toks))
    return out


def ref_forward(image, img_hi, ids, params, l_start, n):
    outs = []
    for b in range(len(ids)):
        img_tok = ref_proj(ref_vision(image[b], params["vision"])[-2][1:], params["proj"])
        stacks = ref_stacks(img_hi[b], params) if l_start is not None else []
        seq, vis = [], []
        for tok in ids[b]:
            if tok == IMG:
                vis = list(range(len(seq), len(seq) + len(img_tok))); seq += list(img_tok)
            else:
                seq.append(params["lm"]["embed"][tok])
        outs.append(ref_lm_stack(np.stack(seq), params["lm"], vis, stacks, l_start or 0, n))
    return np.stack(outs)


def inputs(cfg, seed):
    image, ids, _, params = make(cfg, seed)
    hi = np.random.default_rng(seed + 100).normal(size=(cfg[0], 2 * cfg[1], 2 * cfg[2], cfg[3]))
    return image, hi, ids, params


# ---------------- tests ------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS)
def test_stacks_are_dilated_high_res_neighbours(cfg):
    image, hi, ids, params = inputs(cfg, 0)
    st = np.asarray(deepstack_tokens(hi, params["vision"], params["proj"]))
    N = (cfg[1] // cfg[4]) * (cfg[2] // cfg[4])
    assert st.shape == (4, cfg[0], N, cfg[8]), (
        f"stacks {st.shape}, want (4, B, N={N}, D): 4x the high-res tokens, split into 4 stacks the size of X")
    ref = np.stack([np.stack(ref_stacks(h, params)) for h in hi], 1)
    for k, (di, dj) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        assert np.allclose(st[k], ref[k], atol=1e-9), (
            f"stack {k}=(di={di}, dj={dj}) wrong (max err {np.abs(st[k] - ref[k]).max():.2e}). Stitch the 4 tiles "
            "into one (2hp, 2wp) grid (tile order TL, TR, BL, BR), project it with the same MLP, then take "
            "grid[di::2, dj::2]: token (i, j) of every stack must lie inside low-res cell (i, j)")


@pytest.mark.parametrize("sched", SCHEDULES)
@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_matches_algorithm_1(cfg, sched):
    image, hi, ids, params = inputs(cfg, 1)
    out = np.asarray(deepstack_forward(image, hi, ids, IMG, params, *sched))
    ref = ref_forward(image, hi, ids, params, *sched)
    assert out.shape == ref.shape, f"logits {out.shape} != {ref.shape}: DeepStack must NOT lengthen the context"
    assert np.allclose(out, ref, atol=1e-8), (
        f"(l_start, n)={sched}: max err {np.abs(out - ref).max():.2e}. Before layer idx (not after), if "
        "idx >= l_start and (idx - l_start) % n == 0, add X_stack[(idx - l_start) // n] at the visual positions only")


def test_reduces_to_llava_when_no_layer_is_selected():
    cfg = CONFIGS[0]
    image, hi, ids, params = inputs(cfg, 2)
    out = np.asarray(deepstack_forward(image, hi, ids, IMG, params, l_start=cfg[10], n=1))
    ref = ref_forward(image, hi, ids, params, None, 1)
    assert np.allclose(out, ref, atol=1e-8), (
        "with l_start past the last layer nothing is stacked, so the output must equal plain LLaVA "
        "(low-res tokens at the placeholder, high-res image unused)")


def test_text_before_image_is_untouched_by_the_stacks():
    cfg = CONFIGS[2]
    image, hi, ids, params = inputs(cfg, 3)
    a = np.asarray(deepstack_forward(image, hi, ids, IMG, params, 1, 1))
    b_ = np.asarray(deepstack_forward(image, hi * 0.3 + 1.0, ids, IMG, params, 1, 1))
    for b in range(len(ids)):
        p = int(np.flatnonzero(ids[b] == IMG)[0])
        assert np.allclose(a[b, :p], b_[b, :p], atol=1e-12), \
            f"row {b}: text before the image changed with the high-res image: stacks go to visual positions only"
        assert not np.allclose(a[b, -1], b_[b, -1]), f"row {b}: the high-res image has no effect on the answer"
