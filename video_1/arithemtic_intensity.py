"""Arithmetic Intensity — matmul roofline explorer (Streamlit)."""

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ---------------------------------------------------------------- config ---
@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = load_config()
ACCELERATORS = CFG["accelerators"]
PRECISIONS = CFG["precisions"]
DEFAULTS = CFG["defaults"]
SHAPE = CFG["matmul_shape"]
DOT_SHAPE = CFG["dot_product_shape"]
BMAX = SHAPE["b_max"]

# ------------------------------------------------------------------ page ---
st.set_page_config(
    page_title="Arithmetic Intensity — matmul",
    page_icon="◈",
    layout="wide",
)

ACCENT = "#5EEAD4"      # teal
ACCENT_2 = "#F472B6"    # pink (ridge)
ACCENT_3 = "#FBBF24"    # amber (break-even star)
MEMBOUND = "rgba(94, 234, 212, 0.10)"
COMPBOUND = "rgba(251, 191, 36, 0.10)"
BG = "#0B0F14"
PANEL = "#121821"
INK = "#E7EDF2"
INK_SOFT = "#8B98A5"
RULE = "#232B35"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .stApp {{
        background: radial-gradient(1200px 600px at 15% -10%, rgba(94,234,212,0.06), transparent),
                    radial-gradient(1000px 500px at 100% 0%, rgba(244,114,182,0.05), transparent),
                    {BG};
        color: {INK};
    }}
    section[data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {RULE};
    }}
    h1, h2, h3 {{
        font-family: 'IBM Plex Sans Condensed', sans-serif !important;
        letter-spacing: 0.02em;
    }}
    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif;
        font-weight: 700;
        font-size: clamp(28px, 3.2vw, 44px);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0;
        background: linear-gradient(90deg, {INK}, {ACCENT});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-shape {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        color: {INK_SOFT};
        margin-top: 4px;
    }}
    .hero-shape b {{ color: {ACCENT}; font-weight: 600; }}
    .grp-h {{
        font-family: 'IBM Plex Sans Condensed', sans-serif;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {INK_SOFT};
        border-bottom: 1px solid {RULE};
        padding-bottom: 6px;
        margin: 18px 0 10px;
    }}
    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {RULE};
        border-radius: 12px;
        padding: 14px 16px 10px;
    }}
    div[data-testid="stMetric"] label {{
        font-family: 'IBM Plex Sans Condensed', sans-serif !important;
        font-size: 11px !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {INK_SOFT} !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        color: {ACCENT} !important;
    }}
    .verdict-box {{
        margin-top: 6px;
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid {RULE};
        background: {PANEL};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: {INK_SOFT};
    }}
    .verdict-box b {{ color: {ACCENT}; }}
    .util-meter {{
        height: 10px;
        border-radius: 6px;
        background: {RULE};
        overflow: hidden;
        margin-top: 6px;
    }}
    .util-meter i {{
        display: block;
        height: 100%;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_3});
    }}
    .spec-note {{
        font-size: 11px;
        color: {INK_SOFT};
        line-height: 1.5;
        margin-top: 4px;
    }}
    .stepper-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        color: {INK_SOFT};
        margin: 2px 0 3px;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 18px;
        color: {ACCENT};
        background: {BG};
        border: 1px solid {RULE};
        border-radius: 8px;
        padding: 0.15rem 0;
        min-height: 38px;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
        border-color: {ACCENT};
        color: {ACCENT};
    }}
    section[data-testid="stSidebar"] .stNumberInput input {{
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }}
    hr {{ border-color: {RULE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- helpers ---
def stepper(label, min_v, max_v, default, step, key):
    """Number input flanked by -/+ buttons that jump by `step` — built for live demos."""
    nkey = f"{key}_num"
    if nkey not in st.session_state:
        st.session_state[nkey] = default

    def _dec():
        st.session_state[nkey] = max(min_v, st.session_state[nkey] - step)

    def _inc():
        st.session_state[nkey] = min(max_v, st.session_state[nkey] + step)

    st.markdown(f'<div class="stepper-label">{label}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.button("−", key=f"{key}_dec", on_click=_dec, use_container_width=True)
    with c2:
        st.number_input(label, min_v, max_v, step=step, key=nkey, label_visibility="collapsed")
    with c3:
        st.button("+", key=f"{key}_inc", on_click=_inc, use_container_width=True)
    return st.session_state[nkey]

def intensity(B, D, F, s):
    return (2 * B * D * F) / (s * (B * D + D * F + B * F))

def fmt(v, p=3):
    if not math.isfinite(v):
        return "∞"
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{float(f'{v:.{p}g}'):g}"

def fmt_time(seconds):
    if not math.isfinite(seconds):
        return "∞"
    if seconds < 1e-6:
        return f"{seconds*1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds*1e3:.1f} ms"
    return f"{seconds:.3f} s"

# ------------------------------------------------------------------- ui ----
st.markdown('<p class="hero-title">Arithmetic Intensity</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="grp-h">Matmul shape</div>', unsafe_allow_html=True)
    B = stepper("B — tokens", SHAPE["b_slider_min"], SHAPE["b_slider_max"],
                DEFAULTS["B"], SHAPE["b_slider_step"], "B")

    D = stepper("D — model dim", SHAPE["d_slider_min"], SHAPE["d_slider_max"],
                DEFAULTS["D"], SHAPE["d_slider_step"], "D")

    F = stepper("F — hidden dim", SHAPE["f_slider_min"], SHAPE["f_slider_max"],
                DEFAULTS["F"], SHAPE["f_slider_step"], "F")

    st.markdown('<div class="grp-h">Dot product length</div>', unsafe_allow_html=True)
    D_dot = stepper("D — vector length", DOT_SHAPE["d_slider_min"], DOT_SHAPE["d_slider_max"],
                     DEFAULTS["D_dot"], DOT_SHAPE["d_slider_step"], "D_dot")

    st.markdown('<div class="grp-h">Precision</div>', unsafe_allow_html=True)
    prec_labels = [p["label"] for p in PRECISIONS]
    prec_choice = st.selectbox("dtype", prec_labels, index=DEFAULTS["dtype_index"], label_visibility="collapsed")
    prec = next(p for p in PRECISIONS if p["label"] == prec_choice)
    s_bytes = prec["bytes"]
    compute_mult = prec["compute_multiplier"]

    st.markdown('<div class="grp-h">Accelerator</div>', unsafe_allow_html=True)
    acc_names = [a["name"] for a in ACCELERATORS] + ["Custom…"]
    acc_choice = st.selectbox("Preset", acc_names, index=DEFAULTS["accelerator_index"])

    if acc_choice == "Custom…":
        peak = st.number_input("Peak compute — TFLOP/s", min_value=1.0, value=989.0, step=1.0)
        bw = st.number_input("Memory bandwidth — TB/s", min_value=0.01, value=3.35, step=0.01)
    else:
        acc = next(a for a in ACCELERATORS if a["name"] == acc_choice)
        peak_default = acc["peak_bf16_tflops"] * compute_mult
        peak = st.number_input(
            "Peak compute — TFLOP/s", min_value=1.0, value=float(peak_default), step=1.0,
            help=f"{acc['peak_bf16_tflops']:.0f} TFLOP/s bf16 baseline × {compute_mult:g}× for {prec_choice}",
        )
        bw = st.number_input("Memory bandwidth — TB/s", min_value=0.01, value=float(acc["bandwidth_tbs"]), step=0.01)

    st.markdown(
        f"<p class='spec-note'>Peak compute auto-scales with precision from each chip's bf16 "
        f"baseline (int8/fp8 ≈2× denser, fp32 ≈0.5×). Edit the number above to override.</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------- compute ---
peak_flops = peak * 1e12
bw_bytes = bw * 1e12
ridge = peak_flops / bw_bytes

tab_matmul, tab_dot = st.tabs(["Matmul", "Dot product"])

with tab_matmul:
    st.markdown(
        f'<p class="hero-shape">FLOPs = 2·B·D·F&nbsp;&nbsp;/&nbsp;&nbsp;bytes = s·(BD + DF + BF)'
        f'&nbsp;&nbsp;→&nbsp;&nbsp;<b>[{B}, {D}] × [{D}, {F}]</b></p>',
        unsafe_allow_html=True,
    )

    xlo, xhi = 1, BMAX
    i_max = intensity(BMAX, D, F, s_bytes)
    ylo = 0.4
    yhi = max(ridge * 3, i_max * 1.5, 50)

    xs = np.logspace(np.log10(xlo), np.log10(xhi), 400)
    ys = intensity(xs, D, F, s_bytes)

    den2 = 2 * D * F - ridge * s_bytes * (D + F)
    Bs = (ridge * s_bytes * D * F) / den2 if den2 > 0 else float("inf")

    I_cur = intensity(B, D, F, s_bytes)
    achieved = min(peak_flops, bw_bytes * I_cur)
    util = achieved / peak_flops

    # -------------------------------------------------------------- plot ---
    fig = go.Figure()

    # roofline bands
    fig.add_shape(type="rect", x0=xlo, x1=xhi, y0=ylo, y1=ridge,
                  fillcolor=MEMBOUND, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=xlo, x1=xhi, y0=ridge, y1=yhi,
                  fillcolor=COMPBOUND, line_width=0, layer="below")

    fig.add_annotation(x=math.log10(xlo) + 0.05, y=math.log10(ylo) + 0.06,
                        xref="x", yref="y", text="MEMORY-BOUND", showarrow=False,
                        font=dict(family="IBM Plex Sans Condensed", size=12, color="#7DD3C0"),
                        xanchor="left", yanchor="bottom")
    fig.add_annotation(x=math.log10(xlo) + 0.05, y=math.log10(yhi) - 0.06,
                        xref="x", yref="y", text="COMPUTE-BOUND", showarrow=False,
                        font=dict(family="IBM Plex Sans Condensed", size=12, color="#E8C168"),
                        xanchor="left", yanchor="top")

    # curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Intensity",
        line=dict(color=ACCENT, width=3),
        hovertemplate="B=%{x:.0f}<br>I=%{y:.3g} FLOP/byte<extra></extra>",
    ))

    # ridge line
    fig.add_shape(type="line", x0=xlo, x1=xhi, y0=ridge, y1=ridge,
                  line=dict(color=ACCENT_2, width=2, dash="dash"))
    fig.add_annotation(x=math.log10(xhi), y=math.log10(ridge), xref="x", yref="y",
                        text=f"ridge {fmt(ridge)}", showarrow=False, xanchor="left",
                        font=dict(family="IBM Plex Mono", size=12, color=ACCENT_2))

    # break-even star
    if math.isfinite(Bs) and xlo <= Bs <= xhi:
        fig.add_shape(type="line", x0=Bs, x1=Bs, y0=ylo, y1=ridge,
                      line=dict(color=ACCENT_3, width=1.5, dash="dot"))
        fig.add_trace(go.Scatter(
            x=[Bs], y=[ridge], mode="markers", name="Break-even",
            marker=dict(color=ACCENT_3, size=9, symbol="star"),
            hovertemplate=f"B* = {fmt(Bs,4)}<extra></extra>",
        ))

    # current point
    fig.add_trace(go.Scatter(
        x=[B], y=[I_cur], mode="markers+text", name="Current",
        marker=dict(color=ACCENT, size=12, line=dict(color=BG, width=2)),
        text=[f"B={B}  I={fmt(I_cur)}"], textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=13, color=ACCENT),
        hovertemplate=f"B={B}<br>I={fmt(I_cur)} FLOP/byte<extra></extra>",
    ))

    fig.update_xaxes(
        type="log", range=[math.log10(xlo), math.log10(xhi)],
        title=dict(text="B — TOKENS IN THE BATCH", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
        gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
    )
    fig.update_yaxes(
        type="log", range=[math.log10(ylo), math.log10(yhi)],
        title=dict(text="INTENSITY — FLOP / BYTE", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
        gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=110, t=20, b=10),
        plot_bgcolor=PANEL,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="IBM Plex Sans", color=INK),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------- stats --
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.3])
    c1.metric("Intensity at B", f"{fmt(I_cur)}", "FLOP/byte")
    c2.metric("Ridge point", f"{fmt(ridge)}", "FLOP/byte")
    c3.metric("Break-even B*", fmt(Bs, 4) if math.isfinite(Bs) else "never", "tokens")
    c4.metric("Achieved", f"{fmt(achieved/1e12)}", "TFLOP/s")

    with c5:
        st.metric("Utilization of peak", f"{util*100:.1f}", "%")
        st.markdown(f'<div class="util-meter"><i style="width:{util*100:.1f}%"></i></div>', unsafe_allow_html=True)
        verdict = (
            f"HBM-limited — {fmt(ridge/I_cur)}× short of the ridge"
            if I_cur < ridge
            else f"Compute-limited — {fmt(I_cur/ridge)}× past the ridge"
        )
        st.markdown(f'<div class="verdict-box"><b>Verdict</b> · {verdict}</div>', unsafe_allow_html=True)

with tab_dot:
    st.markdown(
        f'<p class="hero-shape">FLOPs = 2·D − 1 ≈ 2·D&nbsp;&nbsp;/&nbsp;&nbsp;bytes = s·2·D'
        f'&nbsp;&nbsp;→&nbsp;&nbsp;<b>x · w,  x, w ∈ ℝ^{D_dot}</b></p>',
        unsafe_allow_html=True,
    )

    # a plain dot product's intensity is constant — it never depends on D
    I_dot = 2 * D_dot / (s_bytes * 2 * D_dot)  # == 1 / s_bytes

    Tcompute_dot = 2 * D_dot / peak_flops
    Tmem_dot = (s_bytes * 2 * D_dot) / bw_bytes
    achieved_dot = min(peak_flops, bw_bytes * I_dot)
    util_dot = achieved_dot / peak_flops
    dot_compute_bound = I_dot >= ridge

    d_lo, d_hi = DOT_SHAPE["d_slider_min"], DOT_SHAPE["d_slider_max"]
    d_arr = np.logspace(np.log10(d_lo), np.log10(d_hi), 400)
    Tcompute_arr = 2 * d_arr / peak_flops
    Tmem_arr = (s_bytes * 2 * d_arr) / bw_bytes

    fig_dot = go.Figure()

    fig_dot.add_trace(go.Scatter(
        x=d_arr, y=Tcompute_arr, mode="lines", name="Tcompute",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="D=%{x:.0f}<br>Tcompute=%{y:.3e} s<extra></extra>",
    ))
    fig_dot.add_trace(go.Scatter(
        x=d_arr, y=Tmem_arr, mode="lines", name="Tmem",
        line=dict(color=ACCENT_2, width=2.5),
        hovertemplate="D=%{x:.0f}<br>Tmem=%{y:.3e} s<extra></extra>",
    ))
    fig_dot.add_trace(go.Scatter(
        x=[D_dot], y=[Tcompute_dot], mode="markers", name="current Tcompute",
        marker=dict(color=ACCENT, size=11, line=dict(color=BG, width=2)),
        hovertemplate=f"Tcompute at D={D_dot}: {fmt_time(Tcompute_dot)}<extra></extra>",
    ))
    fig_dot.add_trace(go.Scatter(
        x=[D_dot], y=[Tmem_dot], mode="markers", name="current Tmem",
        marker=dict(color=ACCENT_2, size=11, line=dict(color=BG, width=2)),
        hovertemplate=f"Tmem at D={D_dot}: {fmt_time(Tmem_dot)}<extra></extra>",
    ))

    fig_dot.update_xaxes(
        type="log", title=dict(text="D — VECTOR LENGTH", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
        gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
    )
    fig_dot.update_yaxes(
        type="log", title=dict(text="TIME — SECONDS", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
        gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
    )
    fig_dot.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor=PANEL,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(family="IBM Plex Mono", size=12, color=INK_SOFT), bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig_dot, use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Intensity", f"{fmt(I_dot)}", "FLOP/byte")
    d2.metric("Ridge point", f"{fmt(ridge)}", "FLOP/byte")
    d3.metric("Tcompute", fmt_time(Tcompute_dot))
    d4.metric("Tmem", fmt_time(Tmem_dot))

    dot_verdict = (
        f"Compute-bound — intensity is {fmt(I_dot/ridge)}× the ridge"
        if dot_compute_bound
        else f"Memory-bound — intensity is {fmt(ridge/I_dot)}× short of the ridge"
    )
    st.markdown(
        f'<div class="verdict-box"><b>Verdict</b> · {dot_verdict} · utilization {util_dot*100:.2f}%<br>'
        f"A dot product's intensity is <b>1/s = {fmt(I_dot)}</b> FLOP/byte — constant, "
        f"independent of D. Growing the vector never crosses the ridge; the bound is fixed "
        f"by precision and hardware alone.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------- spec sheet -----
st.markdown('<div class="grp-h" style="margin-top:28px">Vendor peak spec sheet</div>', unsafe_allow_html=True)

rows_html = ""
for a in ACCELERATORS:
    is_active = a["name"] == acc_choice
    row_bg = "background:rgba(94,234,212,0.08);" if is_active else ""
    name_color = ACCENT if is_active else INK
    peak_at_prec = a["peak_bf16_tflops"] * compute_mult
    rows_html += (
        f"<tr style='{row_bg}border-bottom:1px solid {RULE}'>"
        f"<td style='padding:7px 10px;color:{name_color};font-weight:{600 if is_active else 400}'>{a['name']}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{fmt(a['peak_bf16_tflops'])}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{fmt(peak_at_prec)}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{a['bandwidth_tbs']:.2f}</td>"
        f"<td style='padding:7px 10px;text-align:right;color:{ACCENT_2};font-weight:600'>"
        f"{fmt(peak_at_prec/a['bandwidth_tbs'])}</td>"
        f"</tr>"
    )

st.markdown(
    f"""
    <div style="background:{PANEL};border:1px solid {RULE};border-radius:12px;padding:14px 6px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:{INK_SOFT}">
        <thead>
          <tr style="border-bottom:1px solid {RULE}">
            <th style="text-align:left;padding:0 10px 8px;font-family:'IBM Plex Sans Condensed',sans-serif;
                       font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{INK_SOFT}">Chip</th>
            <th style="text-align:right;padding:0 10px 8px;font-family:'IBM Plex Sans Condensed',sans-serif;
                       font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{INK_SOFT}">TFLOP/s bf16</th>
            <th style="text-align:right;padding:0 10px 8px;font-family:'IBM Plex Sans Condensed',sans-serif;
                       font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{INK_SOFT}">TFLOP/s @ {prec_choice}</th>
            <th style="text-align:right;padding:0 10px 8px;font-family:'IBM Plex Sans Condensed',sans-serif;
                       font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{INK_SOFT}">TB/s</th>
            <th style="text-align:right;padding:0 10px 8px;font-family:'IBM Plex Sans Condensed',sans-serif;
                       font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:{INK_SOFT}">Ridge</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="spec-note" style="padding:8px 10px 0">Ridge = peak ÷ bandwidth, in FLOP per byte — dense, no sparsity.
      Real kernels land below peak. Highlighted row is the currently selected accelerator.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
