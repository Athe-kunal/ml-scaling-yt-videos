"""TPU Tmath vs Tcomms visualizer — compute-bound vs comms-bound matmul explorer."""

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).parent / "tpu_config.yaml"

# ---------------------------------------------------------------- config ---
@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = load_config()
TPUS = CFG["tpus"]
PCIE_BW = float(CFG["pcie"]["bandwidth_bytes_s"])
PRECISIONS = CFG["precisions"]
DEFAULTS = CFG["defaults"]
SHAPE = CFG["matmul_shape"]

# ------------------------------------------------------------------ page ---
st.set_page_config(page_title="Tmath vs Tcomms — TPU", page_icon="◈", layout="wide")

ACCENT = "#5EEAD4"      # teal — Tmath
ACCENT_2 = "#F472B6"    # pink — Tcomms
ACCENT_3 = "#FBBF24"    # amber — crossover / total
BG = "#0B0F14"
PANEL = "#121821"
INK = "#E7EDF2"
INK_SOFT = "#8B98A5"
RULE = "#232B35"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{
        background: radial-gradient(1200px 600px at 15% -10%, rgba(94,234,212,0.06), transparent),
                    radial-gradient(1000px 500px at 100% 0%, rgba(244,114,182,0.05), transparent),
                    {BG};
        color: {INK};
    }}
    section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {RULE}; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; letter-spacing: 0.02em; }}
    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
        font-size: clamp(26px, 3vw, 40px); letter-spacing: 0.04em; text-transform: uppercase; margin: 0;
        background: linear-gradient(90deg, {INK}, {ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-shape {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {INK_SOFT}; margin-top: 4px; }}
    .hero-shape b {{ color: {ACCENT}; font-weight: 600; }}
    .grp-h {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 600; font-size: 11px;
        letter-spacing: 0.14em; text-transform: uppercase; color: {INK_SOFT};
        border-bottom: 1px solid {RULE}; padding-bottom: 6px; margin: 18px 0 10px;
    }}
    div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 12px; padding: 14px 16px 10px; }}
    div[data-testid="stMetric"] label {{
        font-family: 'IBM Plex Sans Condensed', sans-serif !important; font-size: 11px !important;
        letter-spacing: 0.1em; text-transform: uppercase; color: {INK_SOFT} !important;
    }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {ACCENT} !important; }}
    .verdict-box {{
        margin-top: 6px; padding: 10px 14px; border-radius: 10px; border: 1px solid {RULE};
        background: {PANEL}; font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK_SOFT};
    }}
    .verdict-box b {{ color: {ACCENT}; }}
    .verdict-box.bound-comms b {{ color: {ACCENT_2}; }}
    .spec-note {{ font-size: 11px; color: {INK_SOFT}; line-height: 1.5; margin-top: 4px; }}
    div[role="radiogroup"] label {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
    hr {{ border-color: {RULE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- helpers ---
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

def synced_slider(label, min_v, max_v, default, step, key):
    """Slider paired with a number input, kept in sync via session_state."""
    skey, nkey = f"{key}_slider", f"{key}_num"
    if skey not in st.session_state:
        st.session_state[skey] = default
        st.session_state[nkey] = default

    def _from_slider():
        st.session_state[nkey] = st.session_state[skey]

    def _from_num():
        st.session_state[skey] = st.session_state[nkey]

    col1, col2 = st.columns([2.2, 1])
    with col1:
        st.slider(label, min_v, max_v, step=step, key=skey, on_change=_from_slider)
    with col2:
        st.number_input(label, min_v, max_v, step=step, key=nkey, on_change=_from_num,
                         label_visibility="collapsed")
    return st.session_state[skey]

# ------------------------------------------------------------------- ui ----
header_l, header_r = st.columns([2, 3])
with header_l:
    st.markdown('<p class="hero-title">Tmath vs Tcomms</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="grp-h">Matmul shape</div>', unsafe_allow_html=True)
    B = synced_slider("B — batch", SHAPE["b_slider_min"], SHAPE["b_slider_max"],
                       DEFAULTS["B"], SHAPE["b_slider_step"], "B")

    D = synced_slider("D — model dim", SHAPE["d_slider_min"], SHAPE["d_slider_max"],
                       DEFAULTS["D"], SHAPE["d_slider_step"], "D")

    F = synced_slider("F — hidden dim", SHAPE["f_slider_min"], SHAPE["f_slider_max"],
                       DEFAULTS["F"], SHAPE["f_slider_step"], "F")

    st.markdown('<div class="grp-h">TPU chip</div>', unsafe_allow_html=True)
    tpu_names = [t["name"] for t in TPUS]
    tpu_choice = st.selectbox("Model", tpu_names, index=DEFAULTS["tpu_index"])
    tpu = next(t for t in TPUS if t["name"] == tpu_choice)

    prec_names = [p["name"] for p in PRECISIONS]
    prec_choice = st.selectbox("Precision", prec_names, index=DEFAULTS["precision_index"])
    prec = next(p for p in PRECISIONS if p["name"] == prec_choice)
    peak_flops = float(tpu[prec["flops_field"]])
    s_bytes = prec["bytes"]

    st.markdown('<div class="grp-h">Communication channel</div>', unsafe_allow_html=True)
    channel = st.selectbox(
        "Link",
        ["PCIe — host ↔ chip", "ICI one-way — chip ↔ chip", "ICI bidi — chip ↔ chip"],
        index=0,
    )
    if channel.startswith("PCIe"):
        bw = PCIE_BW
    elif channel.startswith("ICI one-way"):
        bw = float(tpu["ici_bw_oneway_bytes_s"])
    else:
        bw = float(tpu["ici_bw_bidi_bytes_s"])

    st.markdown('<div class="grp-h">Overlap mode</div>', unsafe_allow_html=True)
    overlap_mode = st.radio(
        "How compute and transfer combine",
        ["Overlapping — max(Tmath, Tcomms)", "Non-overlapping — Tmath + Tcomms"],
        index=0,
        label_visibility="collapsed",
    )
    overlapping = overlap_mode.startswith("Overlapping")

# --------------------------------------------------------------- compute ---
with header_r:
    st.markdown(
        f'<p class="hero-shape">FLOPs = 2·B·D·F&nbsp;&nbsp;/&nbsp;&nbsp;bytes moved = s·2·(BD + DF + BF)'
        f'&nbsp;&nbsp;→&nbsp;&nbsp;<b>A[{D}, {F}] · x[{B}, {D}]</b> on {tpu_choice} ({prec_choice})</p>',
        unsafe_allow_html=True,
    )

bytes_moved = s_bytes * (B * D + D * F + B * F)

Tmath = 2 * B * D * F / peak_flops
Tcomms = bytes_moved / bw

Ttotal = max(Tmath, Tcomms) if overlapping else Tmath + Tcomms

# break-even B*: solve Tmath(B) = Tcomms(B) for B, given D, F, peak, bw, s
den = 2 * bw * D * F - peak_flops * s_bytes * (D + F)
Bs = (peak_flops * s_bytes * D * F) / den if den > 0 else float("inf")

bound_compute = Tmath >= Tcomms

# ------------------------------------------------------------------ plot ---
Bs_arr = np.logspace(math.log10(max(1, SHAPE["b_slider_min"])), math.log10(SHAPE["b_slider_max"]), 400)
Tmath_arr = 2 * Bs_arr * D * F / peak_flops
Tcomms_arr = (s_bytes * (Bs_arr * D + D * F + Bs_arr * F)) / bw
Ttotal_arr = np.maximum(Tmath_arr, Tcomms_arr) if overlapping else Tmath_arr + Tcomms_arr

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=Bs_arr, y=Tmath_arr, mode="lines", name="Tmath",
    line=dict(color=ACCENT, width=2.5),
    hovertemplate="B=%{x:.0f}<br>Tmath=%{y:.3e} s<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=Bs_arr, y=Tcomms_arr, mode="lines", name="Tcomms",
    line=dict(color=ACCENT_2, width=2.5),
    hovertemplate="B=%{x:.0f}<br>Tcomms=%{y:.3e} s<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=Bs_arr, y=Ttotal_arr, mode="lines", name="Ttotal",
    line=dict(color=ACCENT_3, width=2, dash="dot"),
    hovertemplate="B=%{x:.0f}<br>Ttotal=%{y:.3e} s<extra></extra>",
))

if math.isfinite(Bs) and SHAPE["b_slider_min"] <= Bs <= SHAPE["b_slider_max"]:
    y_at_Bs = 2 * Bs * D * F / peak_flops
    fig.add_shape(type="line", x0=Bs, x1=Bs, y0=min(Tmath_arr.min(), Tcomms_arr.min()), y1=y_at_Bs,
                  line=dict(color=INK_SOFT, width=1.5, dash="dash"))
    fig.add_trace(go.Scatter(
        x=[Bs], y=[y_at_Bs], mode="markers+text", name="Break-even B*",
        marker=dict(color=INK, size=9, symbol="star", line=dict(color=BG, width=1)),
        text=[f"B* = {fmt(Bs,4)}"], textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=12, color=INK),
        hovertemplate=f"B* = {fmt(Bs,4)}<extra></extra>",
    ))

fig.add_trace(go.Scatter(
    x=[B], y=[Tmath], mode="markers", name="current Tmath",
    marker=dict(color=ACCENT, size=11, line=dict(color=BG, width=2)),
    hovertemplate=f"Tmath at B={B}: {fmt_time(Tmath)}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[B], y=[Tcomms], mode="markers", name="current Tcomms",
    marker=dict(color=ACCENT_2, size=11, line=dict(color=BG, width=2)),
    hovertemplate=f"Tcomms at B={B}: {fmt_time(Tcomms)}<extra></extra>",
))

fig.update_xaxes(
    type="log", title=dict(text="B — BATCH SIZE", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
    gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
)
fig.update_yaxes(
    type="log", title=dict(text="TIME — SECONDS", font=dict(family="IBM Plex Sans Condensed", size=12, color=INK_SOFT)),
    gridcolor=RULE, linecolor=RULE, tickfont=dict(family="IBM Plex Mono", size=11, color=INK_SOFT),
)
fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor=PANEL,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color=INK),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(family="IBM Plex Mono", size=12, color=INK_SOFT), bgcolor="rgba(0,0,0,0)"),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- stats ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tmath", fmt_time(Tmath))
c2.metric("Tcomms", fmt_time(Tcomms))
c3.metric("Ttotal", fmt_time(Ttotal), "overlapping" if overlapping else "non-overlapping")
c4.metric("Break-even B*", fmt(Bs, 4) if math.isfinite(Bs) else "never")

verdict_class = "" if bound_compute else "bound-comms"
verdict_text = (
    f"Compute-bound — Tmath is {fmt(Tmath/Tcomms) if Tcomms > 0 else '∞'}× Tcomms"
    if bound_compute
    else f"Comms-bound — Tcomms is {fmt(Tcomms/Tmath) if Tmath > 0 else '∞'}× Tmath"
)
st.markdown(
    f'<div class="verdict-box {verdict_class}"><b>Verdict</b> · {verdict_text} '
    f'&nbsp;·&nbsp; channel: {channel} &nbsp;·&nbsp; bandwidth: {bw:.3g} B/s</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------- break-even derivation
with st.expander("How is the break-even B* derived?"):
    st.markdown(
        "We want the batch size where compute time equals transfer time — "
        "the point where the workload flips from comms-bound to compute-bound."
    )
    st.latex(r"T_{\text{math}} = \frac{2BDF}{\text{peak}} \qquad "
             r"T_{\text{comms}} = \frac{s\,(BD + DF + BF)}{\text{bw}}")
    st.markdown("Set them equal and solve for $B$:")
    st.latex(r"\frac{2BDF}{\text{peak}} = \frac{s\,(BD + DF + BF)}{\text{bw}}")
    st.latex(r"2 \cdot \text{bw} \cdot BDF = \text{peak} \cdot s \cdot (BD + DF + BF)")
    st.latex(
        r"B\left(2\cdot\text{bw}\cdot DF - \text{peak}\cdot s\cdot D - \text{peak}\cdot s\cdot F\right)"
        r" = \text{peak}\cdot s\cdot DF"
    )
    st.latex(
        r"B^{*} = \frac{\text{peak} \cdot s \cdot D F}"
        r"{2\cdot\text{bw}\cdot DF - \text{peak}\cdot s\,(D + F)}"
    )
    st.markdown("Plugging in the current sidebar values:")
    bs_latex = fmt(Bs, 6) if math.isfinite(Bs) else "\\infty"
    st.latex(
        rf"B^{{*}} = \frac{{{peak_flops:.3g} \times {s_bytes} \times {D} \times {F}}}"
        rf"{{2 \times {bw:.3g} \times {D} \times {F} - {peak_flops:.3g} \times {s_bytes} \times ({D} + {F})}}"
        rf" \;\simeq\; {bs_latex}"
    )
    st.markdown(
        "If the denominator is $\\le 0$, bandwidth can never keep up with compute at any "
        "batch size for this shape — the workload stays comms-bound everywhere, so $B^{*} = \\infty$."
    )

# ---------------------------------------------------------- spec sheet -----
st.markdown('<div class="grp-h" style="margin-top:28px">TPU spec sheet</div>', unsafe_allow_html=True)

rows_html = ""
for t in TPUS:
    is_active = t["name"] == tpu_choice
    row_bg = "background:rgba(94,234,212,0.08);" if is_active else ""
    name_color = ACCENT if is_active else INK
    rows_html += (
        f"<tr style='{row_bg}border-bottom:1px solid {RULE}'>"
        f"<td style='padding:7px 10px;color:{name_color};font-weight:{600 if is_active else 400}'>{t['name']}</td>"
        f"<td style='padding:7px 10px'>{t['pod_size']}</td>"
        f"<td style='padding:7px 10px'>{t['host_size']}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{t['hbm_capacity_gb']} GB</td>"
        f"<td style='padding:7px 10px;text-align:right'>{fmt(t['hbm_bw_bytes_s'])}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{fmt(t['flops_bf16'])}</td>"
        f"<td style='padding:7px 10px;text-align:right'>{fmt(t['flops_int8'])}</td>"
        f"<td style='padding:7px 10px;text-align:right;color:{ACCENT_2}'>{fmt(t['ici_bw_oneway_bytes_s'])}</td>"
        f"<td style='padding:7px 10px;text-align:right;color:{ACCENT_2}'>{fmt(t['ici_bw_bidi_bytes_s'])}</td>"
        f"</tr>"
    )

headers = ["Model", "Pod size", "Host size", "HBM/chip", "HBM BW/chip (B/s)",
           "FLOPs/s bf16", "FLOPs/s int8", "ICI one-way (B/s)", "ICI bidi (B/s)"]
header_html = "".join(
    f"<th style='text-align:{'left' if i < 3 else 'right'};padding:0 10px 8px;"
    f"font-family:\"IBM Plex Sans Condensed\",sans-serif;font-size:10.5px;letter-spacing:.08em;"
    f"text-transform:uppercase;color:{INK_SOFT}'>{h}</th>"
    for i, h in enumerate(headers)
)

st.markdown(
    f"""
    <div style="background:{PANEL};border:1px solid {RULE};border-radius:12px;padding:14px 6px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12px;color:{INK_SOFT}">
        <thead><tr style="border-bottom:1px solid {RULE}">{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="spec-note" style="padding:8px 10px 0">
        PCIe host↔chip bandwidth: {fmt(PCIE_BW)} bytes/s (fixed).
        Highlighted row is the currently selected chip. Dense peak numbers, no sparsity.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
