"""Sharding notation visualizer — step through I,J -> I_x,J -> ... -> I,J_xy.

Matches the device-mesh sharding notation from the JAX scaling book
(https://jax-ml.github.io/scaling-book/sharding/): a matrix A with logical
dimensions (I, J) is placed on a 2D device mesh with axes (x, y). A subscript
on a dimension says which mesh axis(es) partition it; no subscript means the
dimension is fully replicated across that axis.
"""

import colorsys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyBboxPatch, Rectangle

st.set_page_config(page_title="Sharding Notation — I, J on mesh (x, y)", page_icon="◫", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0B0F14"
PANEL_BG = "#121821"
DEVICE_BG = "#1B2530"
INK = "#E7EDF2"
INK_SOFT = "#8B98A5"
RULE = "#232B35"
ACCENT = "#5EEAD4"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{
        background: radial-gradient(1200px 600px at 15% -10%, rgba(94,234,212,0.06), transparent), {BG};
        color: {INK};
    }}
    section[data-testid="stSidebar"] {{ background: {PANEL_BG}; border-right: 1px solid {RULE}; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; letter-spacing: 0.02em; }}
    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
        font-size: clamp(24px, 3vw, 36px); letter-spacing: 0.03em; margin: 0;
        background: linear-gradient(90deg, {INK}, {ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK_SOFT}; margin-top: 4px; }}
    .notation-box {{
        margin-top: 10px; padding: 14px 18px; border-radius: 12px; border: 1px solid {RULE};
        background: {PANEL_BG}; font-family: 'IBM Plex Mono', monospace; font-size: 15px; color: {INK_SOFT};
    }}
    .notation-box b {{ color: {ACCENT}; }}
    div[role="radiogroup"] label, .stSelectbox label, .stSlider label, .stMultiSelect label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Sharding Notation Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A[I, J] placed on device mesh (x, y) — subscripts show which mesh axis '
    'partitions which logical dimension. Based on jax-ml.github.io/scaling-book/sharding/</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- presets ---
PRESETS = [
    ("I, J", set(), set()),
    ("I_x, J", {"x"}, set()),
    ("I, J_x", set(), {"x"}),
    ("I_y, J", {"y"}, set()),
    ("I_xy, J", {"x", "y"}, set()),
    ("I_y, J_x", {"y"}, {"x"}),
    ("I, J_y", set(), {"y"}),
    ("I_x, J_y", {"x"}, {"y"}),
    ("I, J_xy", set(), {"x", "y"}),
]

# ------------------------------------------------------------- sidebar ---
R, C = 4, 4

with st.sidebar:
    st.markdown("### Device mesh")
    x_size = st.slider("Mesh axis x — # devices", 1, 4, 2, step=1)
    y_size = st.slider("Mesh axis y — # devices", 1, 4, 2, step=1)

    st.markdown("### Stage")
    stage_labels = [p[0] for p in PRESETS] + ["Custom"]
    stage = st.select_slider("Step through the build-up", options=stage_labels, value="I, J")

    if stage == "Custom":
        st.markdown("Pick which mesh axes shard each dimension. An axis can shard **either** I or J, not both.")
        shard_I_pick = st.multiselect("Axes sharding I (rows)", ["x", "y"], default=[])
        remaining = [a for a in ["x", "y"] if a not in shard_I_pick]
        shard_J_pick = st.multiselect("Axes sharding J (cols)", remaining, default=[])
        shard_I, shard_J = set(shard_I_pick), set(shard_J_pick)
        notation = f"I{'_' + ''.join(sorted(shard_I)) if shard_I else ''}, J{'_' + ''.join(sorted(shard_J)) if shard_J else ''}"
    else:
        notation, shard_I, shard_J = next(p for p in PRESETS if p[0] == stage)

# ------------------------------------------------------------ notation ---
def tex_label(dim, shard):
    return f"{dim}_{{{''.join(sorted(shard))}}}" if shard else dim


st.markdown(
    f'<div class="notation-box">Sharding: <b>${tex_label("I", shard_I)}, {tex_label("J", shard_J)}$</b> '
    f'&nbsp;&nbsp;|&nbsp;&nbsp; mesh = ({x_size}, {y_size}) &rarr; {x_size * y_size} devices '
    f'&nbsp;&nbsp;|&nbsp;&nbsp; matrix = ({R}, {C})</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:#8B98A5; margin-top:6px;">'
    '<span style="color:#5EEAD4;">&#9632;</span> border = cut by x axis &nbsp;&nbsp; '
    '<span style="color:#F472B6;">&#9632;</span> border = cut by y axis &nbsp;&nbsp; '
    '<span style="color:#FBBF24;">&#9632;</span> border = cut by both x and y</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- colors ---
AXIS_COLOR = {"x": "#5EEAD4", "y": "#F472B6"}
BOTH_COLOR = "#FBBF24"


def edge_color(shard_set):
    if not shard_set:
        return None
    if len(shard_set) == 1:
        return AXIS_COLOR[next(iter(shard_set))]
    return BOTH_COLOR


def cell_color(i, j, R, C):
    hue = 0.32 + 0.55 * (j / max(C - 1, 1))
    hue %= 1.0
    lightness = 0.68 - 0.10 * (i / max(R - 1, 1))
    return colorsys.hls_to_rgb(hue, lightness, 0.55)


def axis_slice(n_items, shard_set, dx, dy, x_size, y_size):
    if shard_set == {"x", "y"}:
        n_chunks, idx = x_size * y_size, dx * y_size + dy
    elif shard_set == {"x"}:
        n_chunks, idx = x_size, dx
    elif shard_set == {"y"}:
        n_chunks, idx = y_size, dy
    else:
        n_chunks, idx = 1, 0
    chunks = np.array_split(np.arange(n_items), n_chunks)
    return chunks[idx]


def fmt_label(i, j, R, C):
    return f"{i}{j}" if R <= 10 and C <= 10 else f"{i},{j}"


# ---------------------------------------------------------- mesh figure ---
def render_mesh(x_size, y_size):
    fig_m, ax_m = plt.subplots(figsize=(y_size * 0.8 + 0.9, x_size * 0.8 + 0.9), facecolor=BG)
    ax_m.set_facecolor(BG)
    for dx in range(x_size):
        for dy in range(y_size):
            ax_m.add_patch(Rectangle((dy + 0.07, dx + 0.07), 0.86, 0.86, facecolor=DEVICE_BG, edgecolor=RULE, linewidth=1))
            ax_m.text(dy + 0.5, dx + 0.5, f"({dx},{dy})", ha="center", va="center",
                       fontsize=8, color=INK_SOFT, fontfamily="monospace")
    ax_m.annotate("", xy=(y_size + 0.15, -0.25), xytext=(-0.05, -0.25),
                  arrowprops=dict(arrowstyle="->", color=AXIS_COLOR["y"], lw=2))
    ax_m.text(y_size / 2, -0.5, "y axis", color=AXIS_COLOR["y"], fontsize=9, ha="center", fontfamily="monospace")
    ax_m.annotate("", xy=(-0.25, x_size + 0.15), xytext=(-0.25, -0.05),
                  arrowprops=dict(arrowstyle="->", color=AXIS_COLOR["x"], lw=2))
    ax_m.text(-0.5, x_size / 2, "x axis", color=AXIS_COLOR["x"], fontsize=9, va="center", ha="center",
               rotation=90, fontfamily="monospace")
    ax_m.set_xlim(-0.75, y_size + 0.2)
    ax_m.set_ylim(x_size + 0.2, -0.75)
    ax_m.set_aspect("equal")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for spine in ax_m.spines.values():
        spine.set_visible(False)
    ax_m.set_title("physical device mesh", fontsize=10, color=INK_SOFT, fontfamily="monospace", pad=8)
    fig_m.tight_layout()
    return fig_m


# ------------------------------------------------------------ data axes ---
def chunk_boundaries(n_items, shard_set, x_size, y_size):
    """Row/col cut positions implied by a shard set, matching axis_slice's chunking."""
    if shard_set == {"x", "y"}:
        n_chunks = x_size * y_size
    elif shard_set == {"x"}:
        n_chunks = x_size
    elif shard_set == {"y"}:
        n_chunks = y_size
    else:
        n_chunks = 1
    chunks = np.array_split(np.arange(n_items), n_chunks)
    return np.cumsum([len(c) for c in chunks])[:-1]


def render_data_axes(R, C, shard_I, shard_J, x_size, y_size):
    """The logical (I, J) tensor on its own — the data-side counterpart to the
    physical device mesh: same cell colors as the per-device panels, with colored
    cut-lines showing where each mesh axis slices the data."""
    fig_d, ax_d = plt.subplots(figsize=(C * 0.5 + 0.9, R * 0.5 + 0.9), facecolor=BG)
    ax_d.set_facecolor(BG)

    panel = FancyBboxPatch(
        (-0.06, -0.06), C + 0.12, R + 0.12,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0, facecolor=DEVICE_BG, zorder=0,
    )
    ax_d.add_patch(panel)

    for i in range(R):
        for j in range(C):
            color = cell_color(i, j, R, C)
            ax_d.add_patch(Rectangle((j, i), 1, 1, facecolor=color, alpha=0.85, edgecolor=BG, linewidth=1.5, zorder=1))

    row_edge, col_edge = edge_color(shard_I), edge_color(shard_J)
    if row_edge:
        for rc in chunk_boundaries(R, shard_I, x_size, y_size):
            ax_d.plot([0, C], [rc, rc], color=row_edge, linewidth=3, zorder=3, solid_capstyle="round")
    if col_edge:
        for cc in chunk_boundaries(C, shard_J, x_size, y_size):
            ax_d.plot([cc, cc], [0, R], color=col_edge, linewidth=3, zorder=3, solid_capstyle="round")

    ax_d.annotate("", xy=(C + 0.15, 0), xytext=(C + 0.15, R),
                  arrowprops=dict(arrowstyle="<-", color=INK_SOFT, lw=1.6))
    ax_d.text(C + 0.32, R / 2, "I axis", color=INK_SOFT, fontsize=9, va="center", ha="left",
               rotation=-90, fontfamily="monospace")
    ax_d.annotate("", xy=(C, R + 0.15), xytext=(0, R + 0.15),
                  arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.6))
    ax_d.text(C / 2, R + 0.32, "J axis", color=INK_SOFT, fontsize=9, ha="center", fontfamily="monospace")

    ax_d.set_xlim(-0.1, C + 0.75)
    ax_d.set_ylim(R + 0.55, -0.1)
    ax_d.set_aspect("equal")
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_title("logical data axes (I, J)", fontsize=10, color=INK_SOFT, fontfamily="monospace", pad=8)
    fig_d.tight_layout()
    return fig_d


# --------------------------------------------------------------- figure ---
cell = 0.5
pad_between = 0.55
fig_w = y_size * (C * cell + pad_between) + 0.5
fig_h = x_size * (R * cell + pad_between) + 0.5
fig, axes = plt.subplots(x_size, y_size, figsize=(fig_w, fig_h), squeeze=False, facecolor=BG)

for dx in range(x_size):
    for dy in range(y_size):
        ax = axes[dx][dy]
        ax.set_facecolor(BG)
        rows_owned = set(axis_slice(R, shard_I, dx, dy, x_size, y_size).tolist())
        cols_owned = set(axis_slice(C, shard_J, dx, dy, x_size, y_size).tolist())

        panel = FancyBboxPatch(
            (-0.06, -0.06), C + 0.12, R + 0.12,
            boxstyle="round,pad=0,rounding_size=0.12",
            linewidth=0, facecolor=DEVICE_BG, zorder=0,
        )
        ax.add_patch(panel)

        for i in range(R):
            for j in range(C):
                owned = i in rows_owned and j in cols_owned
                color = cell_color(i, j, R, C)
                if owned:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor=BG, linewidth=2, zorder=1))
                    ax.text(
                        j + 0.5, i + 0.5, fmt_label(i, j, R, C),
                        ha="center", va="center", fontsize=9, fontfamily="monospace",
                        color="#111318", zorder=2,
                    )
                else:
                    ax.add_patch(Rectangle(
                        (j, i), 1, 1, facecolor=color, alpha=0.10,
                        edgecolor=RULE, linewidth=1, linestyle="--", zorder=1,
                    ))
                    ax.text(
                        j + 0.5, i + 0.5, fmt_label(i, j, R, C),
                        ha="center", va="center", fontsize=9, fontfamily="monospace",
                        color=INK_SOFT, alpha=0.35, zorder=2,
                    )

        row_min, row_max = min(rows_owned), max(rows_owned)
        col_min, col_max = min(cols_owned), max(cols_owned)
        row_edge, col_edge = edge_color(shard_I), edge_color(shard_J)
        if row_edge:
            ax.plot([col_min, col_max + 1], [row_min, row_min], color=row_edge, linewidth=3, zorder=3, solid_capstyle="round")
            ax.plot([col_min, col_max + 1], [row_max + 1, row_max + 1], color=row_edge, linewidth=3, zorder=3, solid_capstyle="round")
        if col_edge:
            ax.plot([col_min, col_min], [row_min, row_max + 1], color=col_edge, linewidth=3, zorder=3, solid_capstyle="round")
            ax.plot([col_max + 1, col_max + 1], [row_min, row_max + 1], color=col_edge, linewidth=3, zorder=3, solid_capstyle="round")

        ax.set_xlim(-0.06, C + 0.06)
        ax.set_ylim(R + 0.06, -0.06)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"x={dx}, y={dy}", fontsize=9, color=INK_SOFT, fontfamily="monospace", pad=6)

fig.tight_layout()

col_mesh, col_data, col_devices = st.columns([1, 1, 4])
with col_mesh:
    st.pyplot(render_mesh(x_size, y_size), use_container_width=True)
with col_data:
    st.pyplot(render_data_axes(R, C, shard_I, shard_J, x_size, y_size), use_container_width=True)
with col_devices:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT};">'
    "The mesh on the far left is the physical layout of devices along axes x and y. Next to it, the "
    "logical data axes (I, J) show the tensor on its own, with colored cut-lines marking where each "
    "mesh axis slices it — same colors as the per-device borders on the right. Each device panel "
    "corresponds one-to-one to a mesh cell, at the same (x, y) coordinate. "
    "A subscript on a dimension splits it across that mesh axis; devices that differ only along an "
    "axis <i>not</i> in the subscript hold identical (replicated) data. The colored border on each "
    "device panel marks which mesh axis produced that cut.</p>",
    unsafe_allow_html=True,
)
