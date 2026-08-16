"""Visualizer for Questions 5-9 from https://jax-ml.github.io/scaling-book/sharding/

Q5: minimum-latency matmul on TPU v4p 4x4x4 (shard non-contracting dim + AllGather
    at the end, vs shard contracting dim + AllReduce, vs fully replicated).
Q6: mixed I_x/J_y sharding scenarios on TPU v5e 4x4 (training vs inference tensor
    parallelism).
Q7: transformer-block matmul under a memory budget on TPU v5e 2x2 — FSDP vs tensor
    parallelism.
Q8: the four communication primitives (AllGather / AllReduce / ReduceScatter /
    AllToAll) — semantics reference + code template.
Q9: AllGather-then-matmul vs matmul-then-AllReduce, the two ways to handle a
    single contracting-dim-sharded operand — the core comparison, worked out with
    real TPU v5e / v4p numbers.

Hardware numbers (bf16 FLOPs/s, HBM bandwidth, ICI bandwidth per link, mesh
topology) are taken from jax-ml.github.io/scaling-book/tpus/.
"""

import colorsys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, FancyBboxPatch

st.set_page_config(page_title="Scaling Book — Questions 5-9", page_icon="⨯", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0A0E14"
PANEL_BG = "#111826"
CARD_BG = "#161F2E"
INK = "#EDF1F7"
INK_SOFT = "#8E9BAF"
RULE = "#26314A"
ACCENT = "#5EEAD4"
ACCENT2 = "#F472B6"
AXIS_COLOR = {"x": "#5EEAD4", "y": "#F472B6", "z": "#FBBF24"}
COMM_COLOR = "#FBBF24"
BAD_COLOR = "#FB7185"
GOOD_COLOR = "#5EEAD4"
SHADOW = "#00000055"
FIG_DPI = 210

plt.rcParams["font.family"] = "monospace"
plt.rcParams["axes.edgecolor"] = RULE
plt.rcParams["savefig.facecolor"] = BG

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; -webkit-font-smoothing: antialiased; }}
    .stApp {{
        background:
            radial-gradient(1100px 550px at 12% -8%, rgba(94,234,212,0.07), transparent),
            radial-gradient(900px 500px at 100% 0%, rgba(244,114,182,0.05), transparent),
            {BG};
        color: {INK};
    }}
    .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
    section[data-testid="stSidebar"] {{ background: {PANEL_BG}; border-right: 1px solid {RULE}; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; letter-spacing: 0.02em; }}

    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
        font-size: clamp(26px, 3vw, 38px); letter-spacing: 0.01em; margin: 0;
        background: linear-gradient(100deg, {INK} 30%, {ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK_SOFT}; margin-top: 6px; }}

    .case-box, .step-desc, .hw-card, .eq-box {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -12px {SHADOW};
    }}
    .case-box {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {INK}; line-height: 1.55; }}
    .case-box b {{ color: {ACCENT}; }}
    .step-desc {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 14.5px; color: {INK}; line-height: 1.6; }}
    .step-desc b {{ color: {ACCENT}; }}
    .eq-box .katex {{ color: {INK} !important; font-size: 1.05em; }}
    .eq-box [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}

    .comm-tag {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
        font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
        padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
    }}
    .comm-none {{ background: rgba(94,234,212,0.10); color: {ACCENT}; border: 1px solid rgba(94,234,212,0.32); }}
    .comm-yes {{ background: rgba(251,191,36,0.12); color: {COMM_COLOR}; border: 1px solid rgba(251,191,36,0.35); }}
    .comm-invalid {{ background: rgba(251,113,133,0.12); color: {BAD_COLOR}; border: 1px solid rgba(251,113,133,0.38); }}

    .hw-card {{ font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: {INK_SOFT}; line-height: 1.7; }}
    .hw-card b {{ color: {INK}; }}
    .hw-card .v {{ color: {ACCENT}; font-weight: 600; }}

    .winner-badge {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        background: rgba(94,234,212,0.14); color: {ACCENT}; border: 1px solid rgba(94,234,212,0.4);
    }}
    .fit-ok {{ color: {GOOD_COLOR}; font-weight: 700; }}
    .fit-bad {{ color: {BAD_COLOR}; font-weight: 700; }}

    .mono-table {{ width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace;
                    font-size: 12.5px; color: {INK}; margin-top: 12px; }}
    .mono-table th, .mono-table td {{ padding: 8px 12px; border-bottom: 1px solid {RULE}; text-align: left; }}
    .mono-table th {{ color: {INK_SOFT}; font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.05em; }}
    .mono-table tr.current td {{ color: {ACCENT}; }}

    div[data-testid="stButton"] button {{
        font-family: 'IBM Plex Mono', monospace; font-weight: 500; border-radius: 10px;
        border: 1px solid {RULE}; background: {CARD_BG}; transition: border-color 0.15s ease, color 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover:not(:disabled) {{ border-color: {ACCENT}; color: {ACCENT}; }}
    div[data-testid="stButton"] button:disabled {{ opacity: 0.35; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {RULE}; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: {INK_SOFT};
        padding: 10px 18px; border-radius: 10px 10px 0 0; background: transparent;
    }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; background: {CARD_BG}; box-shadow: inset 0 -2px 0 {ACCENT}; }}
    div[data-testid="stImage"] img {{ border-radius: 16px; }}
    code {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Scaling Book — Sharded Matmuls, Questions 5-9</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Real TPU v5e / v4p numbers plugged into the AllGather-then-matmul vs '
    'matmul-then-AllReduce/ReduceScatter cost model. Based on jax-ml.github.io/scaling-book/sharding/</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- hardware ---
# bf16 FLOPs/s/chip, HBM bandwidth (B/s), HBM capacity (GB), ICI bandwidth per
# link — one-way and bidirectional (B/s) — and torus dimensionality.
# Source: jax-ml.github.io/scaling-book/tpus/
HW = {
    "v5e": dict(name="TPU v5e", flops=1.97e14, hbm_bw=8.2e11, hbm_gb=16,
                ici_uni=4.5e10, ici_bidi=9.0e10, max_axes=2, topology="2D torus (4x4)"),
    "v4p": dict(name="TPU v4p", flops=2.75e14, hbm_bw=1.2e12, hbm_gb=32,
                ici_uni=4.5e10, ici_bidi=9.0e10, max_axes=3, topology="3D torus (4x4x4)"),
}
DTYPE_BYTES = {"bf16": 2, "fp32": 4, "int8": 1}


def hw_card(hw):
    st.markdown(
        f'<div class="hw-card"><b>{hw["name"]}</b> &nbsp;({hw["topology"]})<br>'
        f'peak bf16 &nbsp;<span class="v">{hw["flops"]:.2e}</span> FLOPs/s/chip &nbsp;|&nbsp; '
        f'HBM &nbsp;<span class="v">{hw["hbm_bw"]:.2e}</span> B/s, {hw["hbm_gb"]}GB &nbsp;|&nbsp; '
        f'ICI/link &nbsp;<span class="v">{hw["ici_uni"]:.2e}</span> B/s one-way, '
        f'<span class="v">{hw["ici_bidi"]:.2e}</span> B/s bidi</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------- formulas ---
def flops_matmul(I, J, K):
    return 2 * I * J * K


def bytes_of(n_elems, dtype):
    return n_elems * DTYPE_BYTES[dtype]


def t_math(per_device_flops, hw):
    return per_device_flops / hw["flops"]


def t_gather_or_scatter(full_bytes, hw, n_axes):
    """AllGather / ReduceScatter of a `full_bytes`-sized array over `n_axes`
    ICI axes used in parallel — matches the book's V / (bw * n_axes) model."""
    n_axes = max(1, min(n_axes, hw["max_axes"]))
    return full_bytes / (hw["ici_bidi"] * n_axes)


def t_allreduce(full_bytes, hw, n_axes):
    """AllReduce = ReduceScatter + AllGather, i.e. 2x the single-pass cost."""
    return 2 * t_gather_or_scatter(full_bytes, hw, n_axes)


def fmt_t(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_bytes(n):
    for unit, scale in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n:.0f}B"


def comm_tag(comm):
    if comm is None:
        return '<span class="comm-tag comm-none">no communication</span>'
    if comm == "INVALID":
        return '<span class="comm-tag comm-invalid">invalid sharding</span>'
    return f'<span class="comm-tag comm-yes">{comm}</span>'


def render_equations(lines):
    st.markdown('<div class="eq-box">', unsafe_allow_html=True)
    st.latex(r"\begin{aligned}" + r"\\[4pt] ".join(lines) + r"\end{aligned}")
    st.markdown("</div>", unsafe_allow_html=True)


def timing_bar(labels, math_times, comm_times, title, ylabel="time"):
    """Grouped bar chart: compute time vs comms time (not stacked — the book
    treats them as overlappable, so seeing both bars side by side shows which
    one is the bottleneck)."""
    fig, ax = plt.subplots(figsize=(max(5.5, 1.9 * len(labels)), 3.6), dpi=FIG_DPI, facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w / 2, [m * 1e3 for m in math_times], w, label="compute (T_math)", color=ACCENT, zorder=3)
    b2 = ax.bar(x + w / 2, [c * 1e3 for c in comm_times], w, label="comms (T_comms)", color=COMM_COLOR, zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, fmt_t(h / 1e3), ha="center", va="bottom",
                    fontsize=7.6, color=INK_SOFT, fontfamily="monospace")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK, fontfamily="monospace")
    ax.set_ylabel(f"{ylabel} (ms)", fontsize=9, color=INK_SOFT, fontfamily="monospace")
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(RULE)
    top = max([*math_times, *comm_times], default=0) * 1e3
    if top > 0:
        ax.set_ylim(0, top * 1.15)  # headroom so the value-label text never clips the axes top
    # Title and legend are figure-level (not axes-level), stacked in the margin
    # reserved above the axes by tight_layout's rect — this keeps them clear of
    # the bars no matter how tall a bar gets or how few bars are plotted (e.g.
    # Q6's single-scenario chart, where an axes-level legend used to overlap it).
    fig.suptitle(title, fontsize=10.5, color=INK, fontfamily="monospace", y=0.99)
    handles, leg_labels = ax.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


# --------------------------------------------------------- device-panel viz --
def chunk_color(k, n):
    hue = (k / max(n, 1)) * 0.80 + 0.03
    return colorsys.hls_to_rgb(hue % 1.0, 0.60, 0.62)


def rrect(ax, x0, y0, w, h, r=None, shadow=False, **kw):
    if r is None:
        r = max(min(w, h) * 0.14, 0.02)
    r = min(r, min(w, h) / 2 - 1e-4) if min(w, h) > 2e-4 else 0
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.035, y0 - 0.035), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor="black", alpha=0.22, zorder=kw.get("zorder", 2) - 0.2,
        ))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw))


def step_nav(key, n_steps):
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = max(0, min(st.session_state[key], n_steps - 1))
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("◀ Previous", key=key + "_prev", disabled=st.session_state[key] == 0, use_container_width=True):
            st.session_state[key] -= 1
    with c3:
        if st.button("Next ▶", key=key + "_next", disabled=st.session_state[key] == n_steps - 1, use_container_width=True):
            st.session_state[key] += 1
    st.session_state[key] = max(0, min(st.session_state[key], n_steps - 1))
    with c2:
        dots = "&nbsp;".join(
            f'<span style="color:{ACCENT};">●</span>' if i == st.session_state[key] else f'<span style="color:{RULE};">●</span>'
            for i in range(n_steps)
        )
        st.markdown(
            f'<p style="text-align:center; font-size:16px; margin-top:8px;">{dots}</p>'
            f'<p style="text-align:center; font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:{INK_SOFT};">'
            f'step {st.session_state[key] + 1} / {n_steps}</p>',
            unsafe_allow_html=True,
        )
    return st.session_state[key]


def draw_block(ax, x0, y0, w, h, spec, title):
    state = spec["state"]
    PAD = 0.05
    if state == "empty":
        rrect(ax, x0, y0, w, h, facecolor=PANEL_BG, alpha=0.5, edgecolor=RULE,
              linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(x0 + w / 2, y0 + h / 2, "—", ha="center", va="center", color=INK_SOFT, fontsize=13)
    elif state == "full_replicated":
        rrect(ax, x0, y0, w, h, facecolor=ACCENT, edgecolor="none", alpha=0.90, zorder=2, shadow=True)
        shape = spec.get("shape", "")
        ax.text(x0 + w / 2, y0 + h / 2, shape, ha="center", va="center",
                color="#08131A", fontsize=11, fontweight="bold", fontfamily="monospace")
    elif state == "partial":
        rrect(ax, x0, y0, w, h, facecolor=COMM_COLOR, alpha=0.20, edgecolor=COMM_COLOR,
              linewidth=1.6, hatch="////", zorder=2, shadow=True)
        shape = spec.get("shape", "")
        ax.text(x0 + w / 2, y0 + h / 2, f"{shape}\npartial (unreduced)", ha="center", va="center",
                color=COMM_COLOR, fontsize=8.2, fontweight="bold", fontfamily="monospace")
    elif state in ("ghost_chunked", "full_gathered", "final_chunked"):
        axis, n, idx = spec["axis"], spec["n_chunks"], spec["chunk_idx"]
        border = spec.get("border_axis", AXIS_COLOR["x"])
        for k in range(n):
            if axis == "col":
                cw = w / n
                cx0, cy0, chw, chh = x0 + k * cw, y0, cw, h
            else:
                chh = h / n
                cx0, cy0, chw, chh = x0, y0 + (n - 1 - k) * chh, w, chh
            pad = min(chw, chh) * PAD
            cx, cy, tw, th = cx0 + pad, cy0 + pad, chw - 2 * pad, chh - 2 * pad
            fillc = chunk_color(k, n)
            owned = k == idx
            if state == "ghost_chunked":
                if owned:
                    rrect(ax, cx, cy, tw, th, facecolor=fillc, edgecolor=border, linewidth=2.2, zorder=3, shadow=True)
                else:
                    rrect(ax, cx, cy, tw, th, facecolor=RULE, alpha=0.30, edgecolor="none", zorder=2)
            elif state == "full_gathered":
                ec = border if owned else "none"
                lw = 2.4 if owned else 0
                z = 3 if owned else 2
                rrect(ax, cx, cy, tw, th, facecolor=fillc, alpha=0.92, edgecolor=ec, linewidth=lw, zorder=z, shadow=owned)
            elif state == "final_chunked" and owned:
                rrect(ax, cx, cy, tw, th, facecolor=fillc, edgecolor=border, linewidth=2.2, zorder=3, shadow=True)
    rrect(ax, x0, y0, w, h, facecolor="none", edgecolor=RULE, linewidth=1.1, zorder=6)
    ax.text(x0 + w / 2, y0 + h + 0.22, title, ha="center", va="bottom", fontsize=10.5,
            color=INK_SOFT, fontfamily="monospace", fontweight="bold")


def draw_operator(ax, cx, cy, symbol, r=0.16):
    ax.add_patch(Circle((cx, cy), r, facecolor=CARD_BG, edgecolor=RULE, linewidth=1.1, zorder=6))
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=12.5, color=INK_SOFT, zorder=7)


def draw_device_panel(ax, specs, label):
    bs, gap = 1.3, 0.55
    x = 0.0
    draw_block(ax, x, 0.0, bs, bs, specs["A"], "A")
    x += bs + gap
    draw_operator(ax, x - gap / 2, bs / 2, "×")
    draw_block(ax, x, 0.0, bs, bs, specs["B"], "B")
    x += bs + gap
    draw_operator(ax, x - gap / 2, bs / 2, "=")
    draw_block(ax, x, 0.0, bs, bs, specs["C"], "C")
    x_end = x + bs
    pad_l, pad_r, pad_t, pad_b = 0.22, 0.22, 0.55, 0.22
    rrect(ax, -pad_l, -pad_b, x_end + pad_l + pad_r, bs + pad_t + pad_b,
          r=0.14, facecolor=PANEL_BG, edgecolor=RULE, linewidth=1.0, zorder=0, shadow=True)
    ax.set_xlim(-pad_l - 0.08, x_end + pad_r + 0.08)
    ax.set_ylim(-pad_b - 0.08, bs + pad_t + 0.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(BG)
    ax.set_title(label, fontsize=9.5, color=INK_SOFT, fontfamily="monospace", pad=12)


def render_grid(panels_by_pos, nrows, ncols, panel_w_in=3.2, panel_h_in=2.3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_w_in, nrows * panel_h_in),
                              dpi=FIG_DPI, squeeze=False, facecolor=BG)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if (r, c) in panels_by_pos:
                specs, label = panels_by_pos[(r, c)]
                draw_device_panel(ax, specs, label)
            else:
                ax.axis("off")
    fig.subplots_adjust(wspace=0.35, hspace=0.55)
    fig.tight_layout()
    return fig


def draw_shape_box(ax, x0, y0, w, h, title, shape_label, kind):
    """One labeled A/B/C box for a single representative chip. kind:
    'full' (replicated, solid accent), 'sharded' (this chip's own slice,
    amber-bordered inset), 'partial' (unreduced partial sum, hatched)."""
    if kind == "full":
        rrect(ax, x0, y0, w, h, facecolor=ACCENT, edgecolor="none", alpha=0.90, zorder=2, shadow=True)
        tcolor = "#08131A"
    elif kind == "sharded":
        rrect(ax, x0, y0, w, h, facecolor="#3A4B63", alpha=0.55, edgecolor="none", zorder=2)
        pad = min(w, h) * 0.08
        rrect(ax, x0 + pad, y0 + pad, w - 2 * pad, h - 2 * pad, facecolor=AXIS_COLOR["y"], alpha=0.90,
              edgecolor=AXIS_COLOR["y"], linewidth=2.2, zorder=3, shadow=True)
        tcolor = "#08131A"
    else:  # partial
        rrect(ax, x0, y0, w, h, facecolor=COMM_COLOR, alpha=0.20, edgecolor=COMM_COLOR,
              linewidth=1.6, hatch="////", zorder=2, shadow=True)
        tcolor = COMM_COLOR
    ax.text(x0 + w / 2, y0 + h / 2, shape_label, ha="center", va="center", color=tcolor,
            fontsize=9.5, fontweight="bold", fontfamily="monospace")
    rrect(ax, x0, y0, w, h, facecolor="none", edgecolor=RULE, linewidth=1.1, zorder=6)
    ax.text(x0 + w / 2, y0 + h + 0.22, title, ha="center", va="bottom", fontsize=10.5,
            color=INK_SOFT, fontfamily="monospace", fontweight="bold")


def render_single_panel(a, b, c):
    """a, b, c = (shape_label, kind) tuples — draws one representative-chip A x B = C diagram."""
    fig, ax = plt.subplots(figsize=(6.4, 2.5), dpi=FIG_DPI, facecolor=BG)
    ax.set_facecolor(BG)
    bs, gap = 1.5, 0.6
    x = 0.0
    draw_shape_box(ax, x, 0, bs, bs, "A", a[0], a[1])
    x += bs + gap
    draw_operator(ax, x - gap / 2, bs / 2, "×")
    draw_shape_box(ax, x, 0, bs, bs, "B", b[0], b[1])
    x += bs + gap
    draw_operator(ax, x - gap / 2, bs / 2, "=")
    draw_shape_box(ax, x, 0, bs, bs, "C", c[0], c[1])
    x_end = x + bs
    pad_l, pad_r, pad_t, pad_b = 0.25, 0.25, 0.6, 0.25
    rrect(ax, -pad_l, -pad_b, x_end + pad_l + pad_r, bs + pad_t + pad_b, r=0.14, facecolor=PANEL_BG,
          edgecolor=RULE, linewidth=1.0, zorder=0, shadow=True)
    ax.set_xlim(-pad_l - 0.1, x_end + pad_r + 0.1)
    ax.set_ylim(-pad_b - 0.1, bs + pad_t + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ===========================================================================
tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Q5 — min latency (v4p)",
    "Q6 — mixed sharding (v5e)",
    "Q7 — transformer block memory",
    "Q8 — comm primitives",
    "Q9 — AllGather vs AllReduce",
])

# =========================================================================== Q5
with tab5:
    st.markdown(
        '<div class="case-box"><b>A[I, J] &middot; B[J, K] &rarr; C[I, K]</b> on TPU v4p 4x4x4, lowest possible '
        "latency, result must end up fully replicated. Compare the four candidate shardings.</div>",
        unsafe_allow_html=True,
    )
    hw5 = HW["v4p"]
    hw_card(hw5)

    c1, c2, c3 = st.columns(3)
    with c1:
        I5 = st.number_input("I", 1024, 131072, 16384, step=1024, key="I5")
    with c2:
        J5 = st.number_input("J", 1024, 131072, 16384, step=1024, key="J5")
    with c3:
        K5 = st.number_input("K", 1024, 131072, 16384, step=1024, key="K5")
    dtype5 = st.radio("dtype", ["bf16", "fp32"], horizontal=True, key="dtype5")

    n_devices5 = 4 * 4 * 4  # v4p 4x4x4
    n_axes5 = 3
    total_flops5 = flops_matmul(I5, J5, K5)
    c_bytes_IK = bytes_of(I5 * K5, dtype5)

    t_math_shared = t_math(total_flops5 / n_devices5, hw5)  # options (1)-(3): compute divides across 64 chips
    t_math_full = t_math(total_flops5, hw5)                 # option (4): every chip redoes the whole thing

    options5 = [
        dict(
            name="(1) shard I on XYZ + AllGather C",
            eq=r"\mathbf{A}[I_{XYZ},J]\cdot\mathbf{B}[J,K]\rightarrow\mathbf{C}[I_{XYZ},K]\xrightarrow{\text{AllGather}_{XYZ}}\mathbf{C}[I,K]",
            desc="A is sharded along its non-contracting dim I across all 3 mesh axes. Every chip already has "
                 "the full J and K it needs, so the local matmul needs no communication at all — the only comms "
                 "cost is AllGathering C at the end to make it fully replicated, as the problem requires.",
            comm="AllGather (C, 3 axes)",
            tm=t_math_shared, tc=t_gather_or_scatter(c_bytes_IK, hw5, n_axes5),
            a=("I/64 × J", "sharded"), b=("J × K", "full"), c=("I × K", "full"),
        ),
        dict(
            name="(2) shard K on XYZ + AllGather C",
            eq=r"\mathbf{A}[I,J]\cdot\mathbf{B}[J,K_{XYZ}]\rightarrow\mathbf{C}[I,K_{XYZ}]\xrightarrow{\text{AllGather}_{XYZ}}\mathbf{C}[I,K]",
            desc="Symmetric to option (1) — B is sharded along its non-contracting dim K instead of A along I. "
                 "Same FLOPs division, and the exact same AllGather cost, since C is the same size either way.",
            comm="AllGather (C, 3 axes)",
            tm=t_math_shared, tc=t_gather_or_scatter(c_bytes_IK, hw5, n_axes5),
            a=("I × J", "full"), b=("J × K/64", "sharded"), c=("I × K", "full"),
        ),
        dict(
            name="(3) shard J on XYZ + AllReduce C",
            eq=r"\mathbf{A}[I,J_{XYZ}]\cdot\mathbf{B}[J_{XYZ},K]\rightarrow\mathbf{C}[I,K]\{U_{XYZ}\}\xrightarrow{\text{AllReduce}_{XYZ}}\mathbf{C}[I,K]",
            desc="Both operands are sharded on the contracting dim J instead. Every chip's local matmul covers "
                 "only 1/64th of the contraction, producing an unreduced partial sum that must be AllReduced — "
                 "2x the bytes of an AllGather of the same-size array.",
            comm="AllReduce (C, 3 axes)",
            tm=t_math_shared, tc=t_allreduce(c_bytes_IK, hw5, n_axes5),
            a=("I × J/64", "sharded"), b=("J/64 × K", "sharded"), c=("I × K", "full"),
        ),
        dict(
            name="(4) fully replicated — no sharding",
            eq=r"\mathbf{A}[I,J]\cdot\mathbf{B}[J,K]\rightarrow\mathbf{C}[I,K]\quad\text{(every chip, redundantly)}",
            desc="Neither operand is sharded at all. Zero communication, but every one of the 64 chips "
                 "redundantly computes the entire matmul — no parallelism gain whatsoever.",
            comm=None,
            tm=t_math_full, tc=0.0,
            a=("I × J", "full"), b=("J × K", "full"), c=("I × K", "full"),
        ),
    ]
    all_totals5 = [max(o["tm"], o["tc"]) for o in options5]
    best5 = options5[int(np.argmin(all_totals5))]

    step5 = step_nav("q5_step", len(options5))
    opt5 = options5[step5]
    render_equations([opt5["eq"]])
    st.markdown(
        f'<div class="step-desc">{comm_tag(opt5["comm"])}<br><b>{opt5["name"]}</b><br>{opt5["desc"]}</div>',
        unsafe_allow_html=True,
    )
    st.pyplot(render_single_panel(opt5["a"], opt5["b"], opt5["c"]), use_container_width=False)
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:11.5px; color:{INK_SOFT}; margin-top:-6px;">'
        f'<span style="color:{ACCENT};">&#9632;</span> full / replicated &nbsp;&nbsp; '
        f'<span style="color:{AXIS_COLOR["y"]};">&#9632;</span> this chip\'s shard &nbsp;&nbsp; '
        f'shapes shown are what <i>one representative chip</i> (of 64) holds</p>',
        unsafe_allow_html=True,
    )

    this_total5 = max(opt5["tm"], opt5["tc"])
    cheapest_note = (
        '<span class="winner-badge">cheapest of the four</span>' if opt5 is best5 else
        f'cheapest overall is <b style="color:{ACCENT};">{best5["name"]}</b> at {fmt_t(max(best5["tm"], best5["tc"]))}'
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
        f'T_math = {fmt_t(opt5["tm"])} &nbsp;|&nbsp; T_comms = {fmt_t(opt5["tc"])} &nbsp;|&nbsp; '
        f'total ≈ {fmt_t(this_total5)} (overlapped) / {fmt_t(opt5["tm"] + opt5["tc"])} (serial) &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'{cheapest_note}</p>',
        unsafe_allow_html=True,
    )

    rows5 = "".join(
        f'<tr class="{"current" if o is opt5 else ""}"><td>{o["name"]}</td><td>{fmt_t(o["tm"])}</td>'
        f'<td>{fmt_t(o["tc"])}</td><td>{fmt_t(max(o["tm"], o["tc"]))}</td>'
        f'<td>{"◀ cheapest" if o is best5 else ""}</td></tr>'
        for o in options5
    )
    st.markdown(
        f'<table class="mono-table"><tr><th>option</th><th>T_math</th><th>T_comms</th><th>total (overlapped)</th><th></th></tr>{rows5}</table>',
        unsafe_allow_html=True,
    )

# =========================================================================== Q6
with tab6:
    st.markdown(
        '<div class="case-box">Mixed <b>I_x / J_y</b> sharding scenarios on TPU v5e 4x4 (16 chips, axes x, y '
        "each of size 4). Pick a scenario to see what communication it forces.</div>",
        unsafe_allow_html=True,
    )
    hw6 = HW["v5e"]
    hw_card(hw6)

    scenario = st.radio(
        "Scenario",
        [
            "A[I_x, J_y] · B[J_y, K] → C[I_x, K]",
            "A[I_x, J] · B[J_x, K_y] → C[I_x, K_y]   (training: data + tensor + ZeRO)",
            "A[I_x, J] · B[J, K_y] → C[I_x, K_y]   (inference: pure tensor parallel)",
        ],
        key="q6_scenario",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        I6 = st.number_input("I", 1024, 131072, 8192, step=1024, key="I6")
    with c2:
        J6 = st.number_input("J", 1024, 131072, 8192, step=1024, key="J6")
    with c3:
        K6 = st.number_input("K", 1024, 131072, 8192, step=1024, key="K6")
    dtype6 = st.radio("dtype", ["bf16", "fp32"], horizontal=True, key="dtype6")

    x_size6 = y_size6 = 4
    n_devices6 = x_size6 * y_size6
    total_flops6 = flops_matmul(I6, J6, K6)
    per_device_flops6 = total_flops6 / n_devices6  # I and K are always divided across x,y in all 3 scenarios
    tm6 = t_math(per_device_flops6, hw6)

    if scenario.startswith("A[I_x, J_y]"):
        # J sharded on y only -> local partial-sum matmul, AllReduce over y (1 axis)
        c_bytes = bytes_of(I6 * K6, dtype6)  # per-(x) replica of the full I,K block held on that x-row
        tc6 = t_allreduce(c_bytes / x_size6, hw6, 1)  # I is already sharded on x, so the array being reduced is I/x * K
        comm_label = "AllReduce over y (J is contracting & sharded there)"
        eq = r"\mathbf{A}[I_X,J_Y]\cdot_J\mathbf{B}[J_Y,K]\rightarrow\mathbf{C}[I_X,K]\{U_Y\}\xrightarrow{\text{AllReduce}_Y}\mathbf{C}[I_X,K]"
        note = "x only carries a (free) batch/output split; all the actual communication is confined to the y axis."
    elif scenario.startswith("A[I_x, J] · B[J_x"):
        # B sharded on J via x -> AllGather B along x, then local matmul, output lands I_x,K_y for free
        b_bytes = bytes_of(J6 * K6, dtype6) / y_size6  # B's K is already sharded on y; only x-chunk of J needs gathering
        tc6 = t_gather_or_scatter(b_bytes, hw6, 1)
        comm_label = "AllGather B along x (contracting dim only sharded on B)"
        eq = r"\text{AllGather}_X\ \mathbf{B}[J_X,K_Y]\rightarrow\mathbf{B}[J,K_Y]\ \ ;\ \ \mathbf{A}[I_X,J]\cdot\mathbf{B}[J,K_Y]\rightarrow\mathbf{C}[I_X,K_Y]"
        note = "This is the ZeRO/FSDP-style pattern: x shards the weight's contracting dim for memory, then gathers it back before the matmul."
    else:
        # J unsharded on both operands -> Case 1, no communication at all
        tc6 = 0.0
        comm_label = None
        eq = r"\mathbf{A}[I_X,J]\cdot\mathbf{B}[J,K_Y]\rightarrow\mathbf{C}[I_X,K_Y]\quad(\text{no communication})"
        note = "J (contracting) is unsharded on both operands, so every chip already has everything it needs — pure tensor parallelism for inference is communication-free here."

    render_equations([eq])
    st.markdown(
        f'<div class="step-desc">{comm_tag(comm_label)}<br>{note}</div>',
        unsafe_allow_html=True,
    )
    st.pyplot(timing_bar(["this scenario"], [tm6], [tc6],
               f"per-chip compute vs comms — {fmt_bytes(bytes_of(I6 * J6 + J6 * K6, dtype6))} total operand bytes"),
               use_container_width=True)
    bound = "compute-bound" if tm6 >= tc6 else "comms-bound"
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
        f'T_math = {fmt_t(tm6)}, T_comms = {fmt_t(tc6)} → <b style="color:{ACCENT};">{bound}</b> at this problem size '
        "(if the two are overlapped, total latency ≈ max(T_math, T_comms); if not overlapped, ≈ their sum).</p>",
        unsafe_allow_html=True,
    )

# =========================================================================== Q7
with tab7:
    st.markdown(
        '<div class="case-box"><b>In[B,D] &middot; W_in[D,F] &middot; W_out[F,D] &rarr; Out[B,D]</b> on TPU v5e '
        "2x2 (4 chips), with a per-chip memory budget. Compare FSDP vs tensor parallelism.</div>",
        unsafe_allow_html=True,
    )
    hw7 = HW["v5e"]
    hw_card(hw7)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        D7 = st.number_input("D", 1024, 65536, 8192, step=512, key="D7")
    with c2:
        F7 = st.number_input("F", 1024, 131072, 32768, step=512, key="F7")
    with c3:
        B7 = st.number_input("B (batch)", 8, 8192, 128, step=8, key="B7")
    with c4:
        mem_limit_mb = st.slider("per-chip memory budget (MB)", 50, 2000, 300, step=10, key="mem7")

    dtype7 = "bf16"
    mesh7 = 2 * 2  # v5e 2x2
    n_axes7 = 2

    w_bytes_full = bytes_of(D7 * F7, dtype7)  # each of W_in, W_out
    in_bytes_full = bytes_of(B7 * D7, dtype7)
    out_bytes_full = in_bytes_full
    total_flops7 = 2 * flops_matmul(B7, D7, F7)  # two matmuls, same FLOPs each direction

    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
        f"W_in = W_out = {fmt_bytes(w_bytes_full)} each (full, unsharded) &nbsp;|&nbsp; "
        f"In = Out = {fmt_bytes(in_bytes_full)} (full) &nbsp;|&nbsp; mesh = {mesh7} chips over 2 axes</p>",
        unsafe_allow_html=True,
    )

    render_equations([
        r"\text{FSDP: }\ In[B_X,D]\cdot W_{in}[D_{XY},F]\cdot W_{out}[F,D_{XY}]\rightarrow Out[B_X,D]",
        r"\text{Tensor-parallel: }\ In[B,D_{XY}]\cdot W_{in}[D,F_{XY}]\cdot W_{out}[F_{XY},D]\rightarrow Out[B,D_{XY}]",
    ])

    # --- FSDP: must AllGather both full weight matrices onto every chip before the matmul.
    fsdp_gather_bytes = 2 * w_bytes_full  # W_in + W_out, each gathered fully over XY
    fsdp_peak_mem = 2 * w_bytes_full + in_bytes_full  # both full weights resident + activations
    fsdp_tc = t_gather_or_scatter(fsdp_gather_bytes, hw7, n_axes7)
    fsdp_tm = t_math(total_flops7 / mesh7, hw7)
    fsdp_fits = fsdp_peak_mem <= mem_limit_mb * 1e6

    # --- Tensor parallel: AllGather In along XY (tiny), local matmuls with F sharded XY,
    # ReduceScatter Out along XY at the end.
    tp_gather_bytes = in_bytes_full
    tp_scatter_bytes = out_bytes_full
    tp_peak_mem = 2 * (w_bytes_full / mesh7) + in_bytes_full  # W_in, W_out shards + activations
    tp_tc = t_gather_or_scatter(tp_gather_bytes, hw7, n_axes7) + t_gather_or_scatter(tp_scatter_bytes, hw7, n_axes7)
    tp_tm = t_math(total_flops7 / mesh7, hw7)
    tp_fits = tp_peak_mem <= mem_limit_mb * 1e6

    strategies7 = [
        ("FSDP", fsdp_tm, fsdp_tc, fsdp_peak_mem, fsdp_fits,
         "AllGather W_in and W_out (full matrices) onto every chip before each matmul."),
        ("Tensor-parallel", tp_tm, tp_tc, tp_peak_mem, tp_fits,
         "AllGather the small activation In, keep weights sharded on F, ReduceScatter Out at the end."),
    ]

    cols = st.columns(2)
    for col, (label, tm, tc, mem, fits, desc) in zip(cols, strategies7):
        with col:
            fit_html = f'<span class="fit-ok">fits ({fmt_bytes(mem)} ≤ {mem_limit_mb}MB)</span>' if fits else \
                       f'<span class="fit-bad">EXCEEDS budget ({fmt_bytes(mem)} &gt; {mem_limit_mb}MB)</span>'
            st.markdown(
                f'<div class="step-desc"><b>{label}</b><br>{desc}<br><br>'
                f"peak per-chip memory: {fit_html}<br>"
                f"T_math = {fmt_t(tm)} &nbsp;|&nbsp; T_comms = {fmt_t(tc)}</div>",
                unsafe_allow_html=True,
            )

    st.pyplot(
        timing_bar([s[0] for s in strategies7], [s[1] for s in strategies7], [s[2] for s in strategies7],
                   "compute vs comms per strategy"),
        use_container_width=True,
    )
    winner7 = "Tensor-parallel" if (tp_fits and not fsdp_fits) or (tp_fits and (tp_tm + tp_tc) < (fsdp_tm + fsdp_tc)) else \
              ("FSDP" if fsdp_fits and not tp_fits else min(strategies7, key=lambda s: s[1] + s[2])[0])
    st.markdown(
        f'<p><span class="winner-badge">recommended: {winner7}</span>&nbsp;&nbsp;'
        f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
        "tensor parallelism only ever gathers/scatters the small activations, so its comms are cheap and its "
        "per-chip memory stays low; FSDP has to materialize the full weight matrices on every chip to do the "
        "matmul, which typically blows the memory budget first.</span></p>",
        unsafe_allow_html=True,
    )

# =========================================================================== Q8
with tab8:
    st.markdown(
        '<div class="case-box">The four core collectives, and the JAX primitive each one maps to. No TPU numbers '
        "here — this is a semantics reference for the benchmarking exercise (run these yourself with "
        "<code>pmap</code>/<code>shard_map</code> to get real timings).</div>",
        unsafe_allow_html=True,
    )

    prims = [
        ("AllGather", "jax.lax.all_gather", "V[i]", "V (full, on every device)",
         "Every device sends its shard to every other device. Shape grows: sharded → replicated."),
        ("AllReduce", "jax.lax.psum", "V[i]", "sum(V) (full, on every device)",
         "Every device's local (same-shape) value is summed across all devices. Shape unchanged."),
        ("ReduceScatter", "jax.lax.psum_scatter", "V[i] (full per device)", "sum(V)[i] (shard)",
         "Sums full-size unreduced values across devices, but each device keeps only its own shard of the result."),
        ("AllToAll", "jax.lax.all_to_all", "V[i][j]", "V[j][i]", "Devices exchange data pairwise — each device sends a different shard to each other device, transposing the sharded axis."),
    ]
    cols = st.columns(4)
    for col, (name, fn, before, after, desc) in zip(cols, prims):
        with col:
            st.markdown(
                f'<div class="step-desc"><b>{name}</b><br>'
                f'<code>{fn}</code><br><br>'
                f'in:&nbsp; <code>{before}</code><br>out: <code>{after}</code><br><br>{desc}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Benchmark template")
    st.code(
        '''import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np, time

mesh = Mesh(np.array(jax.devices()).reshape(4, 4), ("x", "y"))
x = jax.device_put(jnp.ones((8192, 8192)), NamedSharding(mesh, P("x", None)))

@jax.jit
def all_gather(x):
    return jax.lax.all_gather(x, "x", axis=0, tiled=True)

with mesh:
    y = jax.jit(all_gather, in_shardings=NamedSharding(mesh, P("x", None)))(x)
    y.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(20):
        y = all_gather(x)
    y.block_until_ready()
    print((time.perf_counter() - t0) / 20, "s / call")
# Swap all_gather's body for jax.lax.psum(x, "x"), jax.lax.psum_scatter(x, "x"),
# or jax.lax.all_to_all(x, "x", split_axis=0, concat_axis=1) to benchmark the others.''',
        language="python",
    )

# =========================================================================== Q9
with tab9:
    st.markdown(
        '<div class="case-box"><b>A[I, J_X] &middot;<sub>J</sub> B[J, K] &rarr; C[I, K]</b> — only A is sharded '
        "on the contracting dim. Compare the book's default (AllGather A, then matmul) against the alternative "
        "(local matmul first, then AllReduce C).</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        hw_name9 = st.radio("Hardware", ["v5e", "v4p"], horizontal=True, key="hw9")
    hw9 = HW[hw_name9]
    with c2:
        max_axes9 = hw9["max_axes"]
        n_axes9 = st.slider(f"Mesh axes used for the X sharding (max {max_axes9} on {hw9['name']})", 1, max_axes9, 1, key="axes9")
    hw_card(hw9)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        devs_per_axis9 = st.slider("devices per axis", 2, 8, 4, key="dpa9")
    with c2:
        I9 = st.number_input("I", 512, 131072, 8192, step=512, key="I9")
    with c3:
        J9 = st.number_input("J", 512, 131072, 8192, step=512, key="J9")
    with c4:
        K9 = st.number_input("K", 512, 131072, 8192, step=512, key="K9")
    dtype9 = st.radio("dtype", ["bf16", "fp32"], horizontal=True, key="dtype9")

    n9 = devs_per_axis9 ** n_axes9
    total_flops9 = flops_matmul(I9, J9, K9)
    a_bytes9 = bytes_of(I9 * J9, dtype9)
    c_bytes9 = bytes_of(I9 * K9, dtype9)

    # Strategy A: AllGather A, then every chip redundantly does the full matmul.
    ag_tc = t_gather_or_scatter(a_bytes9, hw9, n_axes9)
    ag_tm = t_math(total_flops9, hw9)  # NOT divided by n9 — B was already fully replicated, no parallel work created

    # Strategy B: local matmul on the J-chunk (parallel!), then AllReduce the partial C.
    ar_tm = t_math(total_flops9 / n9, hw9)
    ar_tc = t_allreduce(c_bytes9, hw9, n_axes9)

    render_equations([
        r"\text{AllGather-then-matmul: }\ \text{AllGather}_X\ \mathbf{A}[I,J_X]\rightarrow\mathbf{A}[I,J]\ ;\ \mathbf{A}[I,J]\cdot\mathbf{B}[J,K]\rightarrow\mathbf{C}[I,K]",
        r"\text{Matmul-then-AllReduce: }\ \mathbf{A}[I,J_X]\cdot\mathbf{B}[J_X,K]\rightarrow\mathbf{C}[I,K]\{U_X\}\ ;\ \text{AllReduce}_X\ \mathbf{C}[I,K]\{U_X\}\rightarrow\mathbf{C}[I,K]",
    ])

    steps9 = [
        dict(title="Initial sharding", desc="A[I, J_x] is sharded on the contracting dimension; B[J, K] is fully replicated everywhere.", comm=None),
        dict(title="Two ways to finish it", desc="From here the two strategies diverge — flip between them below to see the difference in what moves and how much compute each chip repeats.", comm=None),
    ]
    step9 = step_nav("q9_step", len(steps9))
    s = steps9[step9]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    # Cap how many chips we actually draw — n9 can reach 8**3 = 512, and nobody
    # needs 512 tiny subplots. The panel count is just a legible sample; every
    # T_math / T_comms number above still reflects the true n9 chip count.
    MAX_SHOWN9 = 4
    n_shown9 = min(n9, MAX_SHOWN9)
    vis_n9 = min(n9, 8)  # cap the visual chunk-count too, so slices stay legible
    ncols9 = n_shown9
    nrows9 = 1
    sample_note9 = f" (showing {n_shown9} of {n9} chips)" if n9 > n_shown9 else ""

    colL, colR = st.columns(2)
    with colL:
        st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{ACCENT}; font-weight:600;">AllGather A → matmul{sample_note9}</p>', unsafe_allow_html=True)
        panels = {}
        for k in range(n_shown9):
            a_state = "ghost_chunked" if step9 == 0 else "full_gathered"
            c_state = "empty" if step9 == 0 else "full_replicated"
            specs = {
                "A": dict(state=a_state, axis="col", n_chunks=vis_n9, chunk_idx=min(k, vis_n9 - 1), border_axis=AXIS_COLOR["x"]),
                "B": dict(state="full_replicated", shape="J × K"),
                "C": dict(state=c_state, shape="I × K"),
            }
            panels[divmod(k, ncols9)] = (specs, f"chip {k}")
        st.pyplot(render_grid(panels, nrows9, ncols9, panel_w_in=2.6, panel_h_in=2.0), use_container_width=True)
        st.markdown(
            f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT};">'
            f"moves A ({fmt_bytes(a_bytes9)}) &nbsp;|&nbsp; every chip then does the <b>full</b> matmul redundantly<br>"
            f"T_comms = {fmt_t(ag_tc)} &nbsp;|&nbsp; T_math = {fmt_t(ag_tm)} &nbsp;|&nbsp; "
            f"<b style='color:{ACCENT};'>total ≈ {fmt_t(max(ag_tc, ag_tm))}</b> (overlapped) / {fmt_t(ag_tc + ag_tm)} (serial)</p>",
            unsafe_allow_html=True,
        )
    with colR:
        st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{COMM_COLOR}; font-weight:600;">matmul → AllReduce C{sample_note9}</p>', unsafe_allow_html=True)
        panels = {}
        for k in range(n_shown9):
            specs = {
                "A": dict(state="ghost_chunked", axis="col", n_chunks=vis_n9, chunk_idx=min(k, vis_n9 - 1), border_axis=AXIS_COLOR["x"]),
                "B": dict(state="ghost_chunked", axis="row", n_chunks=vis_n9, chunk_idx=min(k, vis_n9 - 1), border_axis=AXIS_COLOR["x"]),
                "C": dict(state="empty") if step9 == 0 else dict(state="partial", shape="I × K") if step9 == 1 else dict(state="full_replicated", shape="I × K"),
            }
            panels[divmod(k, ncols9)] = (specs, f"chip {k}")
        st.pyplot(render_grid(panels, nrows9, ncols9, panel_w_in=2.6, panel_h_in=2.0), use_container_width=True)
        st.markdown(
            f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT};">'
            f"each chip matmuls only its own J-chunk (parallel!), then moves C twice ({fmt_bytes(c_bytes9)} × 2 for the AllReduce)<br>"
            f"T_comms = {fmt_t(ar_tc)} &nbsp;|&nbsp; T_math = {fmt_t(ar_tm)} &nbsp;|&nbsp; "
            f"<b style='color:{COMM_COLOR};'>total ≈ {fmt_t(max(ar_tc, ar_tm))}</b> (overlapped) / {fmt_t(ar_tc + ar_tm)} (serial)</p>",
            unsafe_allow_html=True,
        )

    st.pyplot(
        timing_bar(["AllGather + matmul", "matmul + AllReduce"], [ag_tm, ar_tm], [ag_tc, ar_tc],
                   f"{hw9['name']}, {n_axes9} axis(es) × {devs_per_axis9} devices = {n9} chips"),
        use_container_width=True,
    )
    ag_total, ar_total = max(ag_tc, ag_tm), max(ar_tc, ar_tm)
    winner9 = "AllGather + matmul" if ag_total < ar_total else "matmul + AllReduce"
    st.markdown(
        f'<p><span class="winner-badge">wins here: {winner9}</span>&nbsp;&nbsp;'
        f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
        "AllGather moves A once (I·J elements) but repeats the <i>entire</i> matmul on every chip; AllReduce "
        "divides the matmul work by n but moves C twice (2·I·K elements). AllGather wins when A is small "
        "relative to C (e.g. J ≪ K) or when n is large enough that redundant compute is cheap; AllReduce wins "
        "when C is small relative to A (e.g. K ≪ J) — try dragging J and K in opposite directions to feel the "
        "crossover.</span></p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Full code — both strategies")
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT};">'
        f"Runnable with the current I={I9}, J={J9}, K={K9}, n={devs_per_axis9} settings above (single ICI axis "
        '"x" — for multi-axis X sharding, combine axis names in the PartitionSpec entry instead, e.g. '
        '<code>P(None, ("x", "y"))</code>, and pass the same tuple to <code>all_gather</code> / <code>psum</code> / '
        "<code>axis_index</code>).</p>",
        unsafe_allow_html=True,
    )
    st.code(
        f'''import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
import numpy as np, time

I, J, K = {I9}, {J9}, {K9}
n = {devs_per_axis9}                                   # devices sharding J
mesh = Mesh(np.array(jax.devices()[:n]).reshape(n,), ("x",))

A = jax.device_put(jnp.ones((I, J), jnp.bfloat16), NamedSharding(mesh, P(None, "x")))
B = jax.device_put(jnp.ones((J, K), jnp.bfloat16), NamedSharding(mesh, P(None, None)))


# --- Strategy 1: AllGather A, then every chip does the FULL matmul ---------
def allgather_then_matmul(a_shard, b_full):
    a_full = jax.lax.all_gather(a_shard, "x", axis=1, tiled=True)   # [I, J] on every chip
    return a_full @ b_full                                          # [I, K], replicated


# --- Strategy 2: local matmul on the J-chunk, then AllReduce the result ----
def matmul_then_allreduce(a_shard, b_full):
    j = a_shard.shape[1]
    b_shard = jax.lax.dynamic_slice_in_dim(b_full, jax.lax.axis_index("x") * j, j, axis=0)
    partial = a_shard @ b_shard        # [I, K], unreduced partial sum (only this chip's J-slice)
    return jax.lax.psum(partial, "x")  # AllReduce -> [I, K], replicated


def bench(fn, name, n_iter=20):
    f = jax.jit(shard_map(fn, mesh=mesh, in_specs=(P(None, "x"), P(None, None)), out_specs=P(None, None)))
    out = f(A, B)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = f(A, B)
    out.block_until_ready()
    print(f"{{name}}: {{(time.perf_counter() - t0) / n_iter * 1e3:.3f}} ms/call")


with mesh:
    bench(allgather_then_matmul, "AllGather + matmul")
    bench(matmul_then_allreduce, "matmul + AllReduce")''',
        language="python",
    )

st.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT}; margin-top:18px;">'
    "T_comms formulas follow the book's model: AllGather / ReduceScatter of a V-byte array over n ICI axes "
    "used in parallel costs V / (bidi_bw × n); AllReduce = ReduceScatter + AllGather ≈ 2× that. "
    "Overlapped total ≈ max(T_math, T_comms); non-overlapped total ≈ their sum.</p>",
    unsafe_allow_html=True,
)
