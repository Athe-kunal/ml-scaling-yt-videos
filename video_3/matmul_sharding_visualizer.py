"""Sharded matrix-multiplication visualizer — the four cases from
https://jax-ml.github.io/scaling-book/sharding/

Case 1: A[I_x, J] . B[J, K_y] -> C[I_x, K_y]                  (no communication)
Case 2: A[I, J_x] . B[J, K]   -> AllGather A -> C[I, K]        (AllGather before matmul)
Case 3: A[I, J_x] . B[J_x, K] -> C[I,K]{U_x} -> AllReduce      (AllReduce / ReduceScatter after matmul)
Case 4: A[I_x, J] . B[J, K_x] -> INVALID -> AllGather one side -> C[I,K_x] or C[I_x,K]
"""

import colorsys

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle, FancyBboxPatch

st.set_page_config(page_title="Sharded Matmul — 4 Cases", page_icon="⨯", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0A0E14"
PANEL_BG = "#111826"
CARD_BG = "#161F2E"
DEVICE_BG = "#1A2436"
INK = "#EDF1F7"
INK_SOFT = "#8E9BAF"
RULE = "#26314A"
ACCENT = "#5EEAD4"
AXIS_COLOR = {"x": "#5EEAD4", "y": "#F472B6"}
COMM_COLOR = "#FBBF24"
INVALID_COLOR = "#FB7185"
SHADOW = "#00000055"
FIG_DPI = 220

plt.rcParams["font.family"] = "monospace"
plt.rcParams["text.antialiased"] = True
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

    .case-box {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -12px {SHADOW};
        font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {INK}; line-height: 1.55;
    }}
    .case-box b {{ color: {ACCENT}; }}

    .step-desc {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -14px {SHADOW};
        font-family: 'IBM Plex Sans', sans-serif; font-size: 14.5px; color: {INK}; line-height: 1.6;
    }}
    .step-desc b {{ color: {ACCENT}; }}

    .comm-tag {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
        font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
        padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
    }}
    .comm-none {{ background: rgba(94,234,212,0.10); color: {ACCENT}; border: 1px solid rgba(94,234,212,0.32); }}
    .comm-yes {{ background: rgba(251,191,36,0.12); color: {COMM_COLOR}; border: 1px solid rgba(251,191,36,0.35); }}
    .comm-invalid {{ background: rgba(251,113,133,0.12); color: {INVALID_COLOR}; border: 1px solid rgba(251,113,133,0.38); }}

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
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important; background: {CARD_BG};
        box-shadow: inset 0 -2px 0 {ACCENT};
    }}
    div[data-testid="stImage"] img {{ border-radius: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Sharded Matmul — Four Cases</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A[I, J] &middot; B[J, K] &rarr; C[I, K] under different device-mesh shardings. '
    'Based on jax-ml.github.io/scaling-book/sharding/</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- helpers ---
def chunk_color(k, n):
    hue = (k / max(n, 1)) * 0.80 + 0.03
    return colorsys.hls_to_rgb(hue % 1.0, 0.60, 0.62)


def rrect(ax, x0, y0, w, h, r=None, shadow=False, **kw):
    """Rounded rectangle. Optionally draws a soft drop-shadow copy behind it."""
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
    PAD = 0.05  # gap between adjacent chunk tiles, for a clean bento-grid look

    if state == "empty":
        rrect(ax, x0, y0, w, h, facecolor=PANEL_BG, alpha=0.5, edgecolor=RULE,
              linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(x0 + w / 2, y0 + h / 2, "—", ha="center", va="center", color=INK_SOFT, fontsize=13)

    elif state in ("full_replicated", "final_full"):
        color = ACCENT if state == "final_full" else "#3A4B63"
        alpha = 0.92 if state == "final_full" else 0.65
        rrect(ax, x0, y0, w, h, facecolor=color, edgecolor="none", alpha=alpha, zorder=2, shadow=(state == "final_full"))
        shape = spec.get("shape", "")
        if state == "final_full":
            ax.text(x0 + w / 2, y0 + h / 2, shape, ha="center", va="center",
                    color="#08131A", fontsize=11, fontweight="bold", fontfamily="monospace")
        else:
            ax.text(x0 + w / 2, y0 + h / 2, f"{shape}\nreplicated", ha="center", va="center",
                    color=INK_SOFT, fontsize=7.8, fontfamily="monospace")

    elif state == "partial":
        rrect(ax, x0, y0, w, h, facecolor=COMM_COLOR, alpha=0.20, edgecolor=COMM_COLOR,
              linewidth=1.6, hatch="////", zorder=2, shadow=True)
        shape = spec.get("shape", "")
        ax.text(x0 + w / 2, y0 + h / 2, f"{shape}\npartial (unreduced)", ha="center", va="center",
                color=COMM_COLOR, fontsize=8.2, fontweight="bold", fontfamily="monospace")

    elif state == "grid_chunked":
        n_row, n_col = spec["n_row"], spec["n_col"]
        row_idx, col_idx = spec["row_idx"], spec["col_idx"]
        cw, chh = w / n_col, h / n_row
        pad = min(cw, chh) * PAD
        for r in range(n_row):
            for c in range(n_col):
                cx, cy = x0 + c * cw + pad, y0 + (n_row - 1 - r) * chh + pad
                tw, th = cw - 2 * pad, chh - 2 * pad
                owned = r == row_idx and c == col_idx
                if owned:
                    rrect(ax, cx, cy, tw, th, facecolor=ACCENT, alpha=0.92, edgecolor="none", zorder=3, shadow=True)
                    rrect(ax, cx, cy, tw, th, facecolor="none", edgecolor=AXIS_COLOR["y"], linewidth=2.2, zorder=4)
                    ax.plot([cx, cx + tw], [cy, cy], color=AXIS_COLOR["x"], linewidth=2.6, zorder=5, solid_capstyle="round")
                    ax.plot([cx, cx + tw], [cy + th, cy + th], color=AXIS_COLOR["x"], linewidth=2.6, zorder=5, solid_capstyle="round")
                else:
                    rrect(ax, cx, cy, tw, th, facecolor=RULE, alpha=0.35, edgecolor="none", zorder=2)

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
            elif state == "final_chunked":
                if owned:
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


def render_outer_product_explainer():
    """One generic diagram (not per-device) showing why a chunked matmul over J still
    produces a dense, full-shape I x K result instead of touching only part of it."""
    col_w, col_h = 0.42, 1.7
    row_w, row_h = 1.7, 0.42
    blk = 1.7
    gap = 0.6

    fig, ax = plt.subplots(figsize=(9.4, 2.9), dpi=FIG_DPI, facecolor=BG)
    ax.set_facecolor(BG)

    x = 0.0
    rrect(ax, x, 0, col_w, col_h, facecolor=chunk_color(0, 3), edgecolor="none", zorder=2, shadow=True)
    ax.text(x + col_w / 2, col_h + 0.16, "one column of\nA's chunk (I × 1)", ha="center", va="bottom",
            fontsize=8.6, color=INK_SOFT, fontfamily="monospace")

    x += col_w + gap
    draw_operator(ax, x - gap / 2, col_h / 2, "⊗", r=0.17)

    row_y = col_h / 2 - row_h / 2
    rrect(ax, x, row_y, row_w, row_h, facecolor=chunk_color(1, 3), edgecolor="none", zorder=2, shadow=True)
    ax.text(x + row_w / 2, row_y - 0.20, "one row of\nB's chunk (1 × K)", ha="center", va="top",
            fontsize=8.6, color=INK_SOFT, fontfamily="monospace")

    x += row_w + gap
    draw_operator(ax, x - gap / 2, col_h / 2, "=", r=0.17)

    by0 = col_h / 2 - blk / 2
    rrect(ax, x, by0, blk, blk, facecolor=COMM_COLOR, alpha=0.28, edgecolor=COMM_COLOR,
          linewidth=1.6, hatch="////", zorder=2, shadow=True)
    ax.text(x + blk / 2, col_h / 2, "outer product\ndense I × K\n(every entry touched)",
            ha="center", va="center", fontsize=8.6, color=COMM_COLOR, fontweight="bold", fontfamily="monospace")

    x_end = x + blk
    arrow_len = 1.5
    x_acc = x_end + gap + arrow_len + gap
    ax.annotate(
        "", xy=(x_acc - 0.08, col_h / 2), xytext=(x_end + gap * 0.7, col_h / 2),
        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.5, mutation_scale=14),
    )
    ax.text(x_end + gap + arrow_len / 2, col_h / 2 + 0.24,
            "Σ over every j\nin this device's chunk",
            ha="center", va="bottom", fontsize=8.2, color=INK_SOFT, fontfamily="monospace")

    acc_by0 = col_h / 2 - blk / 2
    rrect(ax, x_acc, acc_by0, blk, blk, facecolor=COMM_COLOR, alpha=0.28, edgecolor=COMM_COLOR,
          linewidth=1.6, hatch="////", zorder=2, shadow=True)
    ax.text(x_acc + blk / 2, col_h / 2, "one dense\npartial C[I,K]",
            ha="center", va="center", fontsize=8.6, color=COMM_COLOR, fontweight="bold", fontfamily="monospace")

    x_end2 = x_acc + blk
    ax.set_xlim(-0.2, x_end2 + 0.2)
    ax.set_ylim(-0.65, col_h + 0.65)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


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


def comm_tag(comm):
    if comm is None:
        return '<span class="comm-tag comm-none">no communication</span>'
    if comm == "INVALID":
        return '<span class="comm-tag comm-invalid">invalid sharding</span>'
    return f'<span class="comm-tag comm-yes">{comm}</span>'


# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Case 1 — no contracting-dim sharding",
    "Case 2 — one operand sharded on J",
    "Case 3 — both operands sharded on J",
    "Case 4 — invalid: same axis, non-contracting",
])

# --------------------------------------------------------------- CASE 1 ---
with tab1:
    st.markdown(
        '<div class="case-box"><b>A[I_x, J] &middot; B[J, K_y] &rarr; C[I_x, K_y]</b> — J (the contracting '
        "dimension) is unsharded on both operands, so the computation is valid and needs no communication at all.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        x_size = st.slider("Devices along mesh axis x (shards I)", 1, 4, 2, key="c1_x")
    with c2:
        y_size = st.slider("Devices along mesh axis y (shards K)", 1, 4, 2, key="c1_y")

    steps1 = [
        dict(
            title="Initial sharding",
            desc="Each device holds one row-chunk of A (sharded by x) and one column-chunk of B (sharded by y). "
                 "The contracting dimension J is fully present on every device for both operands.",
            comm=None,
        ),
        dict(
            title="Local matmul — done",
            desc="Because J is unsharded, every device already has everything it needs. Each computes its own "
                 "output block locally: <b>C[I_x, K_y] = A[I_x, J] · B[J, K_y]</b>. No data ever moves between devices.",
            comm=None,
        ),
    ]
    step1 = step_nav("c1_step", len(steps1))
    s = steps1[step1]
    st.markdown(
        f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>',
        unsafe_allow_html=True,
    )

    panels = {}
    for dx in range(x_size):
        for dy in range(y_size):
            specs = {
                "A": dict(state="ghost_chunked", axis="row", n_chunks=x_size, chunk_idx=dx, border_axis=AXIS_COLOR["x"]),
                "B": dict(state="ghost_chunked", axis="col", n_chunks=y_size, chunk_idx=dy, border_axis=AXIS_COLOR["y"]),
                "C": dict(state="empty") if step1 == 0 else dict(state="grid_chunked", n_row=x_size, row_idx=dx, n_col=y_size, col_idx=dy),
            }
            panels[(dx, dy)] = (specs, f"device (x={dx}, y={dy})")
    st.pyplot(render_grid(panels, x_size, y_size), use_container_width=True)

# --------------------------------------------------------------- CASE 2 ---
with tab2:
    st.markdown(
        '<div class="case-box"><b>A[I, J_x] &middot; B[J, K] &rarr; C[I, K]</b> — A is sharded on the contracting '
        "dimension J, B is fully replicated. Fix it with an AllGather on A <i>before</i> the matmul.</div>",
        unsafe_allow_html=True,
    )
    n2 = st.slider("Number of devices (mesh axis x)", 2, 8, 4, key="c2_n")

    steps2 = [
        dict(title="Initial sharding",
             desc="A[I, J_x] — each device holds only one chunk of the contracting dimension J. "
                  "B[J, K] is fully replicated: every device already has the whole thing.",
             comm=None),
        dict(title="AllGather (before matmul)",
             desc="<b>AllGather_x A[I, J_x] &rarr; A[I, J]</b> — every device exchanges its J-chunk with all "
                  "the others over ICI, so each one ends up holding the full A matrix.",
             comm="AllGather"),
        dict(title="Local matmul",
             desc="Now that every device holds the full A[I,J] and B[J,K], each computes the identical, "
                  "full result locally: <b>C[I, K] = A[I,J] · B[J,K]</b>. The output is fully replicated.",
             comm=None),
    ]
    step2 = step_nav("c2_step", len(steps2))
    s = steps2[step2]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    a_state = "ghost_chunked" if step2 == 0 else "full_gathered"
    c_state = "empty" if step2 < 2 else "final_full"
    panels = {}
    ncols = min(n2, 4)
    for k in range(n2):
        specs = {
            "A": dict(state=a_state, axis="col", n_chunks=n2, chunk_idx=k, border_axis=AXIS_COLOR["x"]),
            "B": dict(state="full_replicated", shape="J × K"),
            "C": dict(state=c_state, shape="I × K"),
        }
        panels[divmod(k, ncols)] = (specs, f"device {k}")
    nrows = (n2 + ncols - 1) // ncols
    st.pyplot(render_grid(panels, nrows, ncols), use_container_width=True)

# --------------------------------------------------------------- CASE 3 ---
with tab3:
    st.markdown(
        '<div class="case-box"><b>A[I, J_x] &middot; B[J_x, K] &rarr; C[I, K] {U_x}</b> — both operands are '
        "sharded on the same contracting dimension. Local matmuls give partial sums that must be reduced.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        n3 = st.slider("Number of devices (mesh axis x)", 2, 8, 4, key="c3_n")
    with c2:
        use_rs = st.checkbox("Use ReduceScatter instead of AllReduce (sharded output)", key="c3_rs")

    steps3 = [
        dict(title="Initial sharding",
             desc="A[I, J_x] and B[J_x, K] are both sharded along the same contracting dimension J, on the same mesh axis x.",
             comm=None),
        dict(title="Local matmul — partial sums",
             desc="Each device multiplies its own chunks, but this only covers part of the J dimension: the result is a "
                  "<b>partial sum C[I,K]{U_x}</b> — every device has a different, incomplete value, same shape.",
             comm=None),
        dict(
            title="ReduceScatter (sum + shard)" if use_rs else "AllReduce (sum across devices)",
            desc=(
                "<b>ReduceScatter_x C[I,K]{U_x} &rarr; C[I,K_x]</b> — the partial sums are summed across devices and the "
                "result is scattered, so each device ends up holding only its own final shard of K."
                if use_rs else
                "<b>AllReduce_x C[I,K]{U_x} &rarr; C[I,K]</b> — the partial sums are summed across every device "
                "(ReduceScatter + AllGather under the hood), so every device ends up with the identical, complete result."
            ),
            comm="ReduceScatter" if use_rs else "AllReduce",
        ),
    ]
    step3 = step_nav("c3_step", len(steps3))
    s = steps3[step3]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    if step3 == 1:
        st.markdown(
            f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT}; margin:12px 0 -4px;">'
            "How the multiplication actually works, per J-value in the chunk:</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(render_outer_product_explainer(), use_container_width=True)

    if step3 == 0:
        c_spec_fn = lambda k: dict(state="empty")
    elif step3 == 1:
        c_spec_fn = lambda k: dict(state="partial", shape="I × K")
    else:
        c_spec_fn = (lambda k: dict(state="final_chunked", axis="col", n_chunks=n3, chunk_idx=k, border_axis=AXIS_COLOR["x"])) if use_rs \
            else (lambda k: dict(state="final_full", shape="I × K"))

    panels = {}
    ncols = min(n3, 4)
    for k in range(n3):
        specs = {
            "A": dict(state="ghost_chunked", axis="col", n_chunks=n3, chunk_idx=k, border_axis=AXIS_COLOR["x"]),
            "B": dict(state="ghost_chunked", axis="row", n_chunks=n3, chunk_idx=k, border_axis=AXIS_COLOR["x"]),
            "C": c_spec_fn(k),
        }
        panels[divmod(k, ncols)] = (specs, f"device {k}")
    nrows = (n3 + ncols - 1) // ncols
    st.pyplot(render_grid(panels, nrows, ncols), use_container_width=True)

# --------------------------------------------------------------- CASE 4 ---
with tab4:
    st.markdown(
        '<div class="case-box"><b>A[I_x, J] &middot; B[J, K_x] &rarr; C[I_x, K_x]</b> — <b>invalid.</b> The same mesh '
        "axis x shards two non-contracting dimensions at once, so the output would only have diagonal blocks of C. "
        "Fix it by AllGathering one of the two operands first.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        n4 = st.slider("Number of devices (mesh axis x)", 2, 8, 4, key="c4_n")
    with c2:
        gather_choice = st.radio("Which operand do we AllGather to resolve it?", ["A", "B"], horizontal=True, key="c4_choice")

    if gather_choice == "A":
        step2_desc = "<b>AllGather_x A[I_x, J] &rarr; A[I, J]</b> — every device gathers the missing row-chunks of A, so it now holds the full matrix. B stays sharded on K."
        step3_desc = "Local matmul: <b>A[I,J] · B[J,K_x] &rarr; C[I,K_x]</b>. The output ends up sharded along K, matching B's sharding."
    else:
        step2_desc = "<b>AllGather_x B[J, K_x] &rarr; B[J, K]</b> — every device gathers the missing column-chunks of B, so it now holds the full matrix. A stays sharded on I."
        step3_desc = "Local matmul: <b>A[I_x,J] · B[J,K] &rarr; C[I_x,K]</b>. The output ends up sharded along I, matching A's sharding."

    steps4 = [
        dict(title="Initial sharding — INVALID", desc="A[I_x, J] and B[J, K_x] both use mesh axis x on their "
             "non-contracting dimensions. A given shard along x would only produce the (i, i)-th block of C — "
             "the diagonal — leaving the rest of the output undefined.", comm="INVALID"),
        dict(title=f"AllGather {gather_choice} (resolve the conflict)", desc=step2_desc, comm="AllGather"),
        dict(title="Local matmul", desc=step3_desc, comm=None),
    ]
    step4 = step_nav("c4_step", len(steps4))
    s = steps4[step4]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    a_axis, b_axis = "row", "col"
    if step4 == 0:
        a_state, b_state, c_spec = "ghost_chunked", "ghost_chunked", lambda k: dict(state="empty")
    elif step4 == 1:
        if gather_choice == "A":
            a_state, b_state = "full_gathered", "ghost_chunked"
        else:
            a_state, b_state = "ghost_chunked", "full_gathered"
        c_spec = lambda k: dict(state="empty")
    else:
        if gather_choice == "A":
            a_state, b_state = "full_gathered", "ghost_chunked"
            c_spec = lambda k: dict(state="final_chunked", axis="col", n_chunks=n4, chunk_idx=k, border_axis=AXIS_COLOR["x"])
        else:
            a_state, b_state = "ghost_chunked", "full_gathered"
            c_spec = lambda k: dict(state="final_chunked", axis="row", n_chunks=n4, chunk_idx=k, border_axis=AXIS_COLOR["x"])

    panels = {}
    ncols = min(n4, 4)
    for k in range(n4):
        specs = {
            "A": dict(state=a_state, axis=a_axis, n_chunks=n4, chunk_idx=k, border_axis=AXIS_COLOR["x"]),
            "B": dict(state=b_state, axis=b_axis, n_chunks=n4, chunk_idx=k, border_axis=AXIS_COLOR["x"]),
            "C": c_spec(k),
        }
        panels[divmod(k, ncols)] = (specs, f"device {k}")
    nrows = (n4 + ncols - 1) // ncols
    st.pyplot(render_grid(panels, nrows, ncols), use_container_width=True)

st.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT}; margin-top:18px;">'
    "In every panel, the highlighted chunk is what that device physically holds; faded dashed chunks are data it "
    "does not have (yet). A colored border marks which mesh axis produced a cut — "
    f'<span style="color:{AXIS_COLOR["x"]};">teal</span> for x, '
    f'<span style="color:{AXIS_COLOR["y"]};">pink</span> for y.</p>',
    unsafe_allow_html=True,
)
