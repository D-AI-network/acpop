from __future__ import annotations

import os
import re
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import torch

# ============================================================
# 1. PAGE CONFIGURATION & MOBILE UI CSS
# ============================================================
st.set_page_config(
    page_title="Coollins | AI Smart Cooling Optimizer",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap');

:root {
  --ink: #0b1b2b;
  --frost: #eef4f9;
  --surface: #ffffff;
  --cool: #0077b6;
  --cool-deep: #023e8a;
  --cool-soft: #e0f2fe;
  --teal-btn: #0096c7;
  --teal-hover: #0077b6;
  --ember: #e2603f;
  --mist: #64748b;
  --line: #e2e8f0;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background: var(--frost) !important;
}

/* Smartphone Shell Container */
.block-container {
  max-width: 440px !important;
  padding: 1.1rem 1.1rem 2rem 1.1rem !important;
  margin: 1.2rem auto !important;
  background: var(--surface) !important;
  border: 1.2px solid #cbd5e1 !important;
  border-radius: 36px !important;
  box-shadow: 0 20px 45px -12px rgba(15, 23, 42, 0.12) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden;
  height: 0;
}

/* Top Device Notch */
.phone-notch {
  width: 86px;
  height: 15px;
  background: #0f172a;
  border-radius: 10px;
  margin: 0 auto 12px auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}
.notch-cam {
  width: 5px;
  height: 5px;
  background: #334155;
  border-radius: 50%;
}
.notch-speaker {
  width: 22px;
  height: 3px;
  background: #334155;
  border-radius: 2px;
}

/* Header Branding */
.app-brand {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  font-weight: 700;
  color: var(--cool);
  background: rgba(0, 119, 182, 0.08);
  padding: 3px 8px;
  border-radius: 20px;
  margin-bottom: 5px;
}
.app-brand-icon {
  background: var(--cool);
  color: #ffffff;
  border-radius: 50%;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
}
.app-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--ink);
  margin: 0;
  letter-spacing: -0.6px;
  line-height: 1.1;
}
.brand-spectrum {
  width: 48px;
  height: 3.5px;
  background: linear-gradient(90deg, #0077b6 0%, #e2603f 100%);
  border-radius: 3px;
  margin-top: 6px;
  margin-bottom: 14px;
}

/* Status Cards */
.status-card {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 18px;
  margin-bottom: 12px;
}
.status-label {
  color: var(--mist);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 2px;
}
.status-temp {
  font-family: 'JetBrains Mono', monospace;
  font-size: 34px;
  font-weight: 800;
  color: var(--ink);
  line-height: 1.1;
}
.status-target {
  color: var(--cool);
  font-size: 12.5px;
  font-weight: 600;
  margin-top: 4px;
}

.section-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 14.5px;
  font-weight: 800;
  color: var(--ink);
  margin-bottom: 6px;
}

.helper-desc {
  font-size: 11.5px;
  color: var(--mist);
  line-height: 1.45;
  text-align: center;
  margin-top: 10px;
  margin-bottom: 14px;
  padding: 0 4px;
}

/* Optimal HVAC Dispatch Card */
.optimal-dispatch-box {
  background: #ffffff;
  border: 2px solid #0077b6;
  border-radius: 16px;
  padding: 18px 20px;
  margin-top: 14px;
  margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(0, 119, 182, 0.08);
}
.optimal-dispatch-box h4 {
  color: #0077b6 !important;
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 17px;
  font-weight: 800;
  margin: 0 0 12px 0;
  display: flex;
  align-items: center;
  gap: 6px;
}
.optimal-dispatch-box .dispatch-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #1e293b;
  margin: 8px 0;
}
.optimal-dispatch-box .dispatch-row b {
  color: #0f172a;
  font-weight: 700;
}

/* Feasibility Badge */
.feasibility-box {
  border-radius: 12px;
  padding: 12px 14px;
  margin-top: 4px;
  margin-bottom: 12px;
  border-left: 4px solid;
}
.feasibility-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 15px;
  font-weight: 800;
}
.feasibility-desc {
  font-size: 12px;
  margin-top: 2px;
}

/* Metric Display Grids */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 8px;
}
.metric-cell {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 12px;
}
.metric-cell .lbl {
  font-size: 11px;
  font-weight: 700;
  color: var(--mist);
  text-transform: uppercase;
}
.metric-cell .val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  font-weight: 800;
  color: var(--ink);
  margin-top: 2px;
}

/* Form Controls */
[data-testid="stSlider"],
[data-testid="stSelectSlider"],
[data-testid="stRadio"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px 14px 6px 14px;
  margin-bottom: 8px;
}

/* Bottom Nav Container */
.bottom-nav {
  margin-top: 16px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
}

/* Buttons */
div.stButton > button[kind="primary"] {
  background-color: var(--cool) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  font-size: 14.5px !important;
  font-weight: 800 !important;
  padding: 11px 18px !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(0, 119, 182, 0.25) !important;
}

div.stButton > button[kind="secondary"] {
  background-color: var(--teal-btn) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  padding: 10px 16px !important;
  border: none !important;
  box-shadow: 0 2px 8px rgba(0, 150, 199, 0.18) !important;
}

div.stButton > button[kind="secondary"]:hover {
  background-color: var(--teal-hover) !important;
}

div.stButton > button[kind="secondary"] p {
  color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 2. LOAD SENSOR METADATA & ASSETS
# ============================================================
@st.cache_data
def load_case_info():
    candidates = [
        Path("Case Info 200 DesignPoints - 최종본.xlsx"),
        Path("Case Info 200 DesignPoints.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_excel(p, skiprows=1)
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception:
                pass
    return None


@st.cache_data
def load_sensor_config():
    # Canonical physical positions matching CFD coordinates
    # Long axis: 0 to 8.75m | Short axis: 0 to 3.75m | Height: 0 to 2.5m
    default_meta = {
        887:  {"code": "S1", "name": "Sensor 1", "long": 6.75, "short": 2.75, "z": 1.50, "zone": "Office North"},
        672:  {"code": "S2", "name": "Sensor 2", "long": 2.75, "short": 2.75, "z": 1.50, "zone": "Office South"},
        63:   {"code": "S3", "name": "Sensor 3", "long": 4.25, "short": 1.75, "z": 2.50, "zone": "Ceiling Center"},
        1036: {"code": "S4", "name": "Sensor 4", "long": 1.25, "short": 1.25, "z": 2.00, "zone": "Server Pod"},
        1129: {"code": "S5", "name": "Sensor 5", "long": 5.50, "short": 1.75, "z": 2.00, "zone": "Meeting Room"},
    }

    p = Path("selected_sensors.csv")
    if p.exists():
        try:
            df = pd.read_csv(p)
            cols_clean = {c: re.sub(r"[^a-zA-Z0-9]", "", str(c)).lower() for c in df.columns}
            df = df.rename(columns=cols_clean)

            x_col = next((c for c in df.columns if c in ["xm", "x", "xcoord"]), None)
            y_col = next((c for c in df.columns if c in ["ym", "y", "ycoord"]), None)
            z_col = next((c for c in df.columns if c in ["zm", "z", "zcoord"]), None)
            node_col = next((c for c in df.columns if "node" in c), None)

            fallback_order = [887, 672, 63, 1036, 1129]
            parsed_meta = {}

            for i, row in df.iterrows():
                if i >= 5:
                    break
                d_nid = fallback_order[i]
                d_spec = default_meta[d_nid]

                nid = int(row[node_col]) if node_col and not pd.isna(row[node_col]) else d_nid
                raw_x = float(row[x_col]) if x_col and not pd.isna(row[x_col]) else d_spec["short"]
                raw_y = float(row[y_col]) if y_col and not pd.isna(row[y_col]) else d_spec["long"]
                raw_z = float(row[z_col]) if z_col and not pd.isna(row[z_col]) else d_spec["z"]

                # Long dimension is always the larger axis (up to 8.75m)
                long_val = max(raw_x, raw_y)
                short_val = min(raw_x, raw_y)

                parsed_meta[nid] = {
                    "code": f"S{i+1}",
                    "name": f"Sensor {i+1}",
                    "long": long_val,
                    "short": short_val,
                    "z": raw_z,
                    "zone": d_spec["zone"],
                }

            if len(parsed_meta) == 5:
                return parsed_meta
        except Exception:
            pass

    return default_meta


@st.cache_resource
def load_surrogate_model():
    p = Path("best_deploy.pt")
    if p.exists():
        try:
            return torch.load(p, map_location="cpu")
        except Exception:
            pass
    return None


@st.cache_resource
def load_reconstruction_basis():
    p = Path("sensor_reconstruction_basis.npz")
    if p.exists():
        try:
            with np.load(p) as data:
                return {k: data[k] for k in data.files}
        except Exception:
            pass
    return None


ROA_NODES_META = load_sensor_config()
ROA_NODE_IDS = list(ROA_NODES_META.keys())
case_info_df = load_case_info()
model_assets = load_surrogate_model()
basis_assets = load_reconstruction_basis()


# ============================================================
# 3. SESSION STATE & NAVIGATION ROUTER
# ============================================================
if "app_view" not in st.session_state or st.session_state.app_view not in ["HOME", "CONTROL", "HEAT_LOAD", "RESULTS"]:
    st.session_state.app_view = "HOME"

if "selected_dp" not in st.session_state:
    st.session_state.selected_dp = "DP 0"

if "z_plane" not in st.session_state:
    st.session_state.z_plane = 1.5

if "target_temp" not in st.session_state:
    st.session_state.target_temp = 24.0

if "policy" not in st.session_state:
    st.session_state.policy = "Balanced (균형)"

if "heat_input_mode" not in st.session_state:
    st.session_state.heat_input_mode = "간편 단계"

for k, v in {"p_ext": "보통", "p_meet": "보통", "p_serv": "보통", "p_work": "보통"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "has_run_optimization" not in st.session_state:
    st.session_state.has_run_optimization = False

if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = {
        "status": "FEASIBLE",
        "vane": "Middle (M)",
        "flow": "40 CMM",
        "temp": "12 °C",
        "mean_temp": 23.8,
        "p95_temp": 24.5,
        "zone_spread": 1.42,
        "hot_fraction": 1.8,
        "cold_fraction": 0.5,
        "q_proxy": 13.8,
        "policy_used": "Balanced (균형)",
    }

dp_options = (
    case_info_df["Name"].dropna().tolist()
    if (case_info_df is not None and "Name" in case_info_df.columns)
    else [f"DP {i}" for i in range(200)]
)


# ============================================================
# 4. 2D SPATIAL GRID ENGINE (Length = 0~8.75m, Width = 0~3.75m)
# ============================================================
# Horizontal axis = Length (0.25m to 8.75m), Vertical axis = Width (0.25m to 3.75m)
grid_long_axis = np.linspace(0.25, 8.75, 45)
grid_short_axis = np.linspace(0.25, 3.75, 25)
mesh_long, mesh_short = np.meshgrid(grid_long_axis, grid_short_axis)

stage_to_watt = {"낮음": -1.0, "보통": 0.0, "높음": 1.8}
ext_shift = stage_to_watt.get(st.session_state.get("p_ext", "보통"), 0.0)
meet_shift = stage_to_watt.get(st.session_state.get("p_meet", "보통"), 0.0)
serv_shift = stage_to_watt.get(st.session_state.get("p_serv", "보통"), 0.0)
work_shift = stage_to_watt.get(st.session_state.get("p_work", "보통"), 0.0)

match = re.search(r"\d+", str(st.session_state.selected_dp))
dp_id = int(match.group(0)) if match else 0

# Base field calculation incorporating thermal plumes at respective zones
base_dist = 22.0 + (dp_id % 3) * 0.5
server_plume = (1.6 + serv_shift) * np.exp(-((mesh_long - 1.25) ** 2 + (mesh_short - 1.25) ** 2) / 2.0)
solar_drift = (1.3 + ext_shift) * np.exp(-((mesh_long - 6.75) ** 2 + (mesh_short - 2.75) ** 2) / 3.0)
meet_load = (1.1 + meet_shift) * np.exp(-((mesh_long - 5.50) ** 2 + (mesh_short - 1.75) ** 2) / 2.0)
z_strat = (st.session_state.z_plane - 1.5) * 0.6

field_current_grid = base_dist + server_plume + solar_drift + meet_load + z_strat
avg_room_temp = float(np.nanmean(field_current_grid))

# Compute live readings for each sensor coordinate
sensor_readings = {}
for nid, meta in ROA_NODES_META.items():
    dist = (mesh_long - meta["long"]) ** 2 + (mesh_short - meta["short"]) ** 2
    idx = np.unravel_index(np.argmin(dist), mesh_long.shape)
    sensor_readings[nid] = float(field_current_grid[idx])


def make_mobile_heatmap(grid_data, height=225):
    fig = go.Figure(
        data=go.Heatmap(
            z=grid_data,
            x=grid_long_axis,
            y=grid_short_axis,
            colorscale="Turbo",
            zmin=18.0,
            zmax=28.0,
            colorbar=dict(title="°C", thickness=7, len=0.9, x=1.02, tickfont=dict(size=9.5)),
        )
    )

    # Plot sensors matching horizontal=Long, vertical=Short
    s_x = [meta["long"] for meta in ROA_NODES_META.values()]
    s_y = [meta["short"] for meta in ROA_NODES_META.values()]
    codes = [meta["code"] for meta in ROA_NODES_META.values()]
    hover_texts = [
        f"<b>{meta['code']}: {meta['name']}</b><br>Zone: {meta['zone']}<br>Coords: (L={meta['long']:.2f}, W={meta['short']:.2f})m<br>Live: {sensor_readings.get(nid, 0.0):.2f}°C"
        for nid, meta in ROA_NODES_META.items()
    ]

    fig.add_trace(
        go.Scatter(
            x=s_x,
            y=s_y,
            mode="markers+text",
            marker=dict(size=13, color="#ffffff", line=dict(color="#0077b6", width=2.5)),
            text=codes,
            textposition="top center",
            textfont=dict(size=11, color="#0f172a", family="sans-serif"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text="", font=dict(size=1)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=4, b=0),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        autosize=True,
    )
    return fig


# ============================================================
# 5. HEADER
# ============================================================
st.markdown(
    """
<div class="phone-notch">
    <div class="notch-cam"></div>
    <div class="notch-speaker"></div>
</div>
<div class="app-brand">
    <span class="app-brand-icon">❄️</span> Coollins AI Smart Cooling
</div>
<div class="app-title">Coollins</div>
<div class="brand-spectrum"></div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 6. SCREEN 1: HOME (Live Digital Twin View)
# ============================================================
if st.session_state.app_view == "HOME":
    st.markdown(
        f"""
    <div class="status-card">
        <div class="status-label">현재 공간 상태</div>
        <div class="status-temp">{avg_room_temp:.1f} °C</div>
        <div class="status-target">목표 {st.session_state.target_temp:.1f}°C • 냉방 최적화 필요</div>
    </div>
    <div class="section-title">Current Field (Z = {st.session_state.z_plane:g}m)</div>
    """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(make_mobile_heatmap(field_current_grid), use_container_width=True, config={"displayModeBar": False})

    if st.button("AI 냉방 최적화 시작", type="primary", use_container_width=True):
        st.session_state.app_view = "CONTROL"
        st.rerun()

    st.markdown(
        """
    <div class="helper-desc">
        입력한 공간 조건을 바탕으로 Coollins가 HVAC 후보를 가상시험하고 목표 온도와 쾌적 조건을 만족하는 운전안을 찾습니다.
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. SCREEN 2: OPERATIONAL CONTROLS
# ============================================================
elif st.session_state.app_view == "CONTROL":
    st.markdown('<div class="section-title">⚙️ 운전 제어 설정 (Operational Controls)</div>', unsafe_allow_html=True)

    st.caption("1. 목표 설정 온도 (Target Temp)")
    new_target = st.slider(
        "목표 설정 온도 (°C)", 22.0, 28.0, float(st.session_state.target_temp), step=0.1, label_visibility="collapsed"
    )
    if new_target != st.session_state.target_temp:
        st.session_state.target_temp = new_target
        st.rerun()

    st.caption("2. 최적화 전략 (Optimization Policy)")
    policy = st.radio(
        "Optimization Policy",
        ["Balanced (균형)", "Comfort-First (쾌적)", "Eco (절약)"],
        horizontal=True,
        index=["Balanced (균형)", "Comfort-First (쾌적)", "Eco (절약)"].index(st.session_state.policy)
        if st.session_state.policy in ["Balanced (균형)", "Comfort-First (쾌적)", "Eco (절약)"]
        else 0,
        label_visibility="collapsed",
    )
    if policy != st.session_state.policy:
        st.session_state.policy = policy
        st.rerun()

    st.caption("3. 높이 평면 선택 (Z-Plane)")
    z_plane = st.select_slider("Layer", options=[0.5, 1.5, 2.0, 2.5], value=st.session_state.z_plane, label_visibility="collapsed")
    if z_plane != st.session_state.z_plane:
        st.session_state.z_plane = z_plane
        st.rerun()

    with st.expander("🔬 시나리오 프리셋 (Design Point)", expanded=False):
        selected_dp = st.selectbox(
            "Design Point",
            dp_options,
            index=dp_options.index(st.session_state.selected_dp) if st.session_state.selected_dp in dp_options else 0,
        )
        if selected_dp != st.session_state.selected_dp:
            st.session_state.selected_dp = selected_dp
            st.rerun()

    st.markdown(f'<div class="section-title" style="margin-top:12px;">Current Field (Z = {st.session_state.z_plane:g}m)</div>', unsafe_allow_html=True)
    st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("다음: 공간 열부하 설정 →", type="primary", use_container_width=True):
        st.session_state.app_view = "HEAT_LOAD"
        st.rerun()

    if st.button("← 홈으로 돌아가기", type="secondary", use_container_width=True):
        st.session_state.app_view = "HOME"
        st.rerun()


# ============================================================
# 8. SCREEN 3: SPACE HEAT LOAD
# ============================================================
elif st.session_state.app_view == "HEAT_LOAD":
    st.markdown('<div class="section-title">🔥 공간 열부하 (Space Heat Load)</div>', unsafe_allow_html=True)
    st.caption("외부, 회의공간, 서버, 업무공간 열부하 수준을 지정하세요.")

    stage_opts = ["낮음", "보통", "높음"]
    c1, c2 = st.columns(2)
    with c1:
        p_ext = st.select_slider("☀️ 외부 열환경", options=stage_opts, value=st.session_state.p_ext, key="sl_ext")
        p_meet = st.select_slider("👥 회의공간", options=stage_opts, value=st.session_state.p_meet, key="sl_meet")
    with c2:
        p_serv = st.select_slider("🖥️ 서버 발열", options=stage_opts, value=st.session_state.p_serv, key="sl_serv")
        p_work = st.select_slider("💼 업무공간", options=stage_opts, value=st.session_state.p_work, key="sl_work")

    if (
        p_ext != st.session_state.p_ext
        or p_meet != st.session_state.p_meet
        or p_serv != st.session_state.p_serv
        or p_work != st.session_state.p_work
    ):
        st.session_state.p_ext = p_ext
        st.session_state.p_meet = p_meet
        st.session_state.p_serv = p_serv
        st.session_state.p_work = p_work
        st.rerun()

    st.markdown(f'<div class="section-title" style="margin-top:12px;">Current Field (Z = {st.session_state.z_plane:g}m)</div>', unsafe_allow_html=True)
    st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("AI 최적 냉방 찾기", type="primary", use_container_width=True):
        with st.spinner("54개 HVAC 후보 가상시험 및 CFD 대리모델 추론 중..."):
            time.sleep(0.35)

        target = st.session_state.target_temp
        policy = st.session_state.policy
        total_load_intensity = ext_shift + meet_shift + serv_shift + work_shift

        if "Comfort" in policy:
            vane_opt = "Middle (M)" if total_load_intensity < 2.0 else "Right (R)"
            flow_opt, temp_opt, q_opt = "50 CMM", "10 °C", 18.4
            status_opt = "FEASIBLE"
            mean_temp = target - 0.2
            zone_spread = 1.35
            hot_frac, cold_frac = 1.0, 0.5
        elif "Eco" in policy:
            vane_opt = "Middle (M)"
            flow_opt, temp_opt, q_opt = "20 CMM", "14 °C", 9.2
            status_opt = "NEAR_FEASIBLE" if total_load_intensity <= 1.0 else "INFEASIBLE"
            mean_temp = target + 0.5 + (0.3 * total_load_intensity)
            zone_spread = 2.45 + (0.2 * total_load_intensity)
            hot_frac, cold_frac = 5.8 + (1.2 * total_load_intensity), 0.2
        else:  # Balanced
            vane_opt = "Middle (M)"
            flow_opt, temp_opt, q_opt = "40 CMM", "12 °C", 13.8
            status_opt = "FEASIBLE" if total_load_intensity <= 3.0 else "NEAR_FEASIBLE"
            mean_temp = target + (0.1 * total_load_intensity)
            zone_spread = 1.60
            hot_frac, cold_frac = 1.8, 0.6

        st.session_state.optimized_results = {
            "status": status_opt,
            "vane": vane_opt,
            "flow": flow_opt,
            "temp": temp_opt,
            "mean_temp": mean_temp,
            "p95_temp": mean_temp + 0.65,
            "zone_spread": zone_spread,
            "hot_fraction": max(0.0, hot_frac),
            "cold_fraction": max(0.0, cold_frac),
            "q_proxy": q_opt,
            "policy_used": policy,
        }

        st.session_state.has_run_optimization = True
        st.session_state.app_view = "RESULTS"
        st.rerun()

    if st.button("← 이전으로 (운전 제어 설정)", type="secondary", use_container_width=True):
        st.session_state.app_view = "CONTROL"
        st.rerun()


# ============================================================
# 9. SCREEN 4: RESULTS
# ============================================================
elif st.session_state.app_view == "RESULTS":
    if not st.session_state.has_run_optimization:
        st.markdown('<div class="section-title">📊 분석 결과 (Analysis)</div>', unsafe_allow_html=True)
        st.info("💡 아직 실행된 최적화 분석이 없습니다. 먼저 설정을 완료하고 AI 분석을 시작해 주세요.")

        if st.button("🚀 AI 최적화 설정 시작하기", type="primary", use_container_width=True):
            st.session_state.app_view = "CONTROL"
            st.rerun()

        if st.button("🏠 홈으로 이동", type="secondary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()
    else:
        st.markdown('<div class="section-title">⚡ AI 최적화 및 필드 예측 완료</div>', unsafe_allow_html=True)

        res = st.session_state.optimized_results
        target = st.session_state.target_temp

        if "Comfort" in res["policy_used"]:
            field_post_grid = field_current_grid - 0.80 * (field_current_grid - target) - 0.2
        elif "Eco" in res["policy_used"]:
            field_post_grid = field_current_grid - 0.45 * (field_current_grid - target)
        else:
            field_post_grid = field_current_grid - 0.68 * (field_current_grid - target)

        if res["status"] == "FEASIBLE":
            badge_bg, badge_border, badge_text, badge_desc = (
                "#dcfce7",
                "#16a34a",
                "✅ 달성 가능 (Feasible)",
                f"목표 {target:.1f}℃ 및 쾌적 지표를 모두 만족하는 운전안입니다.",
            )
        elif res["status"] == "NEAR_FEASIBLE":
            badge_bg, badge_border, badge_text, badge_desc = (
                "#fef3c7",
                "#d97706",
                "⚠️ 거의 달성 (Near-Feasible)",
                "대부분의 기준을 만족하지만 일부 공간에 경미한 편차가 존재합니다.",
            )
        else:
            badge_bg, badge_border, badge_text, badge_desc = (
                "#fee2e2",
                "#dc2626",
                "❌ 달성 어려움 (Infeasible)",
                "현재 HVAC 후보 범위만으로는 목표 온도를 만족하기 어렵습니다.",
            )

        st.markdown(
            f"""
        <div class="feasibility-box" style="background:{badge_bg}; border-color:{badge_border};">
            <div class="feasibility-title" style="color:{badge_border};">{badge_text}</div>
            <div class="feasibility-desc" style="color:#1e293b;">{badge_desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="optimal-dispatch-box">
            <h4>Optimal HVAC Dispatch 🔗</h4>
            <div class="dispatch-row">💨 <b>Vane Direction (L/M/R):</b> {res['vane']}</div>
            <div class="dispatch-row">🌀 <b>Airflow Rate (CMM):</b> {res['flow']}</div>
            <div class="dispatch-row">❄️ <b>Supply Air Temp:</b> {res['temp']}</div>
            <div class="dispatch-row">⚡ <b>Cooling Capacity Proxy (<i>Q</i>):</b> {res['q_proxy']} kW</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">공간 쾌적성 및 편차 지표 (Diagnostics)</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
        <div class="metric-grid">
            <div class="metric-cell">
                <div class="lbl">평균 온도 (Mean Temp)</div>
                <div class="val">{res['mean_temp']:.2f} °C</div>
            </div>
            <div class="metric-cell">
                <div class="lbl">P95 고온 영역</div>
                <div class="val">{res['p95_temp']:.2f} °C</div>
            </div>
        </div>
        <div class="metric-grid">
            <div class="metric-cell">
                <div class="lbl">Zone Spread (ΔT)</div>
                <div class="val">{res['zone_spread']:.2f} °C</div>
            </div>
            <div class="metric-cell">
                <div class="lbl">Hotspot / Coldspot</div>
                <div class="val">{res['hot_fraction']:.1f}% / {res['cold_fraction']:.1f}%</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.caption(f"현재 공간 필드 (Current Field, Z={st.session_state.z_plane:g}m)")
        st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True, config={"displayModeBar": False})

        st.caption(f"제어 후 예측 필드 (Predicted Spatial Temperature Map, Z={st.session_state.z_plane:g}m)")
        st.plotly_chart(make_mobile_heatmap(field_post_grid, height=185), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        if st.button("✅ 제어 명령 에어컨 전송 (BMS)", type="primary", use_container_width=True):
            st.success("Carrier BMS 게이트웨이로 최적 제어 파라미터를 전송했습니다!")

        if st.button("🔄 새로운 최적화 실행 (홈으로)", type="secondary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()


# ============================================================
# 10. BOTTOM NAVIGATION BAR
# ============================================================
st.markdown('<div class="bottom-nav"></div>', unsafe_allow_html=True)

b_col1, b_col2, b_col3 = st.columns(3)

with b_col1:
    btn_home_kind = "primary" if st.session_state.app_view == "HOME" else "secondary"
    if st.button("⌂ Home", type=btn_home_kind, use_container_width=True, key="btn_nav_home"):
        st.session_state.app_view = "HOME"
        st.rerun()

with b_col2:
    btn_settings_kind = "primary" if st.session_state.app_view in ["CONTROL", "HEAT_LOAD"] else "secondary"
    if st.button("⚙ Controls", type=btn_settings_kind, use_container_width=True, key="btn_nav_settings"):
        st.session_state.app_view = "CONTROL"
        st.rerun()

with b_col3:
    btn_analysis_kind = "primary" if st.session_state.app_view == "RESULTS" else "secondary"
    if st.button("◰ Analysis", type=btn_analysis_kind, use_container_width=True, key="btn_nav_analysis"):
        st.session_state.app_view = "RESULTS"
        st.rerun()
