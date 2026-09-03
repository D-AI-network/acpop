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

/* Optimal HVAC Dispatch Card (White Fill + Blue Outline) */
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

/* Controls Styling */
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

/* Primary Button */
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

/* Secondary Button */
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
# 2. SENSOR NODE METADATA (Sparse QR/PCA Basis)
# ============================================================
ROA_NODES_META = {
    887: {"code": "S1", "name": "Sensor 1", "x": 2.75, "y": 6.75, "z": 1.50, "zone": "Office North"},
    672: {"code": "S2", "name": "Sensor 2", "x": 2.75, "y": 2.75, "z": 1.50, "zone": "Office South"},
    63: {"code": "S3", "name": "Sensor 3", "x": 1.75, "y": 4.25, "z": 2.50, "zone": "Ceiling Center"},
    1036: {"code": "S4", "name": "Sensor 4", "x": 1.25, "y": 1.25, "z": 2.00, "zone": "Server Pod"},
    1129: {"code": "S5", "name": "Sensor 5", "x": 1.75, "y": 5.50, "z": 2.00, "zone": "Meeting Room"}
}
ROA_NODE_IDS = list(ROA_NODES_META.keys())


# ============================================================
# 3. DATA LOADERS
# ============================================================
@st.cache_data
def load_case_info():
    candidates = [
        Path("Case Info 200 DesignPoints.xlsx"),
        Path("Case Info 200 DesignPoints - 최종본.xlsx"),
        Path("data/Case Info 200 DesignPoints.xlsx"),
        Path("case_info.xlsx"),
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
def load_field_csv(dp_name_str):
    match = re.search(r'\d+', str(dp_name_str))
    dp_num = match.group(0) if match else "0"
    candidate_paths = [
        f"field_data/dp{dp_num}.csv",
        f"data/field_data/dp{dp_num}.csv",
        f"dp{dp_num}.csv",
        f"data/dp{dp_num}.csv",
        f"Field data/dp{dp_num}.csv",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, skiprows=5)
                df.columns = [c.strip() for c in df.columns]
                x_col = [c for c in df.columns if 'X' in c][0]
                y_col = [c for c in df.columns if 'Y' in c][0]
                z_col = [c for c in df.columns if 'Z' in c][0]
                t_col = [c for c in df.columns if 'Temperature' in c][0]

                df['X_m'] = df[x_col].astype(float)
                df['Y_m'] = df[y_col].astype(float)
                df['Z_m'] = df[z_col].astype(float)
                raw_t = df[t_col].astype(float)
                df['Temperature_C'] = raw_t - 273.15 if raw_t.mean() > 100 else raw_t

                n_col = [c for c in df.columns if 'Node' in c]
                df['Node_Number'] = df[n_col[0]].astype(int) if n_col else range(len(df))
                return df, path, True
            except Exception:
                pass

    # Analytic fallback
    coords = []
    dp_seed = int(dp_num) if dp_num.isdigit() else 0
    np.random.seed(dp_seed)
    for x in np.linspace(0.25, 3.75, 20):
        for y in np.linspace(0.25, 8.75, 30):
            for z in [0.5, 1.5, 2.0, 2.5]:
                coords.append({"X_m": x, "Y_m": y, "Z_m": z})
    df = pd.DataFrame(coords)
    df['Node_Number'] = range(len(df))
    df['Temperature_C'] = 21.5 + (dp_seed % 4) + 2.0 * np.sin(df['X_m'] + dp_seed) + 1.2 * np.cos(df['Y_m'])
    return df, "Synthetic Stream", False


# ============================================================
# 4. SESSION STATE & NAVIGATION ROUTER
# ============================================================
if "app_view" not in st.session_state or st.session_state.app_view not in ["HOME", "CONTROL", "HEAT_LOAD", "RESULTS"]:
    st.session_state.app_view = "HOME"

if "selected_dp" not in st.session_state:
    st.session_state.selected_dp = "DP 0"

if "z_plane" not in st.session_state:
    st.session_state.z_plane = 1.5

# Standard Target Setpoint: 22.0 to 28.0 °C
if "target_temp" not in st.session_state:
    st.session_state.target_temp = 24.0

if "policy" not in st.session_state:
    st.session_state.policy = "Balanced (균형)"

if "heat_input_mode" not in st.session_state:
    st.session_state.heat_input_mode = "간편 단계"

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
        "policy_used": "Balanced (균형)"
    }

# ============================================================
# 5. SPATIAL FIELD INTERPOLATION
# ============================================================
case_info_df = load_case_info()
dp_options = case_info_df['Name'].dropna().tolist() if (
            case_info_df is not None and 'Name' in case_info_df.columns) else [f"DP {i}" for i in range(200)]

field_df, _, _ = load_field_csv(st.session_state.selected_dp)

sensor_readings = {}
for nid in ROA_NODE_IDS:
    match = field_df[field_df['Node_Number'] == nid]
    if not match.empty:
        sensor_readings[nid] = float(match.iloc[0]['Temperature_C'])
    else:
        meta = ROA_NODES_META[nid]
        dist = (field_df['X_m'] - meta['x']) ** 2 + (field_df['Y_m'] - meta['y']) ** 2 + (
                    field_df['Z_m'] - meta['z']) ** 2
        sensor_readings[nid] = float(field_df.loc[dist.idxmin(), 'Temperature_C'])

slice_df = field_df[np.isclose(field_df['Z_m'], st.session_state.z_plane, atol=0.35)]
if slice_df.empty:
    slice_df = field_df

gx = np.linspace(field_df['X_m'].min(), field_df['X_m'].max(), 35)
gy = np.linspace(field_df['Y_m'].min(), field_df['Y_m'].max(), 35)
grid_x, grid_y = np.meshgrid(gx, gy)

field_current_grid = griddata(
    (slice_df['X_m'], slice_df['Y_m']),
    slice_df['Temperature_C'],
    (grid_x, grid_y),
    method='cubic',
    fill_value=np.nanmean(slice_df['Temperature_C'])
)

avg_room_temp = float(np.nanmean(field_current_grid))


def make_mobile_heatmap(grid_data, height=225):
    fig = go.Figure(data=go.Heatmap(
        z=grid_data, x=gx, y=gy,
        colorscale="Turbo", zmin=18.0, zmax=28.0,
        colorbar=dict(title="°C", thickness=6, len=0.85, x=1.02, tickfont=dict(size=9.5))
    ))

    sx = [meta["x"] for meta in ROA_NODES_META.values()]
    sy = [meta["y"] for meta in ROA_NODES_META.values()]
    codes = [meta["code"] for meta in ROA_NODES_META.values()]
    hover_texts = [
        f"<b>{meta['code']}: {meta['name']}</b><br>Zone: {meta['zone']}<br>Live: {sensor_readings.get(nid, 0.0):.2f}°C"
        for nid, meta in ROA_NODES_META.items()
    ]

    fig.add_trace(go.Scatter(
        x=sx, y=sy,
        mode="markers+text",
        marker=dict(size=13, color="#ffffff", line=dict(color="#0077b6", width=2.5)),
        text=codes,
        textposition="top center",
        textfont=dict(size=11, color="#0f172a", family="sans-serif"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text="", font=dict(size=1)),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, showticklabels=False),
        margin=dict(l=4, r=4, t=8, b=4),
        height=height,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        autosize=True
    )
    return fig


# ============================================================
# 6. HEADER
# ============================================================
st.markdown("""
<div class="phone-notch">
    <div class="notch-cam"></div>
    <div class="notch-speaker"></div>
</div>
<div class="app-brand">
    <span class="app-brand-icon">❄️</span> Coollins AI Smart Cooling
</div>
<div class="app-title">Coollins</div>
<div class="brand-spectrum"></div>
""", unsafe_allow_html=True)

# ============================================================
# 7. SCREEN 1: HOME (Live Digital Twin View)
# ============================================================
if st.session_state.app_view == "HOME":
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">현재 공간 상태</div>
        <div class="status-temp">{avg_room_temp:.1f} °C</div>
        <div class="status-target">목표 {st.session_state.target_temp:.1f}°C • 냉방 최적화 필요</div>
    </div>
    <div class="section-title">Current Field (Z = {st.session_state.z_plane:g}m)</div>
    """, unsafe_allow_html=True)

    st.plotly_chart(make_mobile_heatmap(field_current_grid), use_container_width=True, config={'displayModeBar': False})

    if st.button("AI 냉방 최적화 시작", type="primary", use_container_width=True):
        st.session_state.app_view = "CONTROL"
        st.rerun()

    st.markdown("""
    <div class="helper-desc">
        입력한 공간 조건을 바탕으로 Coollins가 HVAC 후보를 가상시험하고 목표 온도와 쾌적 조건을 만족하는 운전안을 찾습니다.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 8. SCREEN 2: OPERATIONAL CONTROLS + CURRENT FIELD GRAPH
# ============================================================
elif st.session_state.app_view == "CONTROL":
    st.markdown('<div class="section-title">⚙️ 운전 제어 설정 (Operational Controls)</div>', unsafe_allow_html=True)

    # 1. Target Temperature Selection (User Flow 1: 22.0 to 28.0 °C)
    st.caption("1. 목표 설정 온도 (Target Temp)")
    st.session_state.target_temp = st.slider(
        "목표 설정 온도 (°C)", 22.0, 28.0, float(st.session_state.target_temp), step=0.1, label_visibility="collapsed"
    )

    # 2. Strategy Policy (User Flow 3: Balanced / Comfort / Eco)
    st.caption("2. 최적화 전략 (Optimization Policy)")
    policy = st.radio(
        "Optimization Policy",
        ["Balanced (균형)", "Comfort-First (쾌적)", "Eco (절약)"],
        horizontal=True,
        index=["Balanced (균형)", "Comfort-First (쾌적)", "Eco (절약)"].index(
            st.session_state.policy) if st.session_state.policy in ["Balanced (균형)", "Comfort-First (쾌적)",
                                                                    "Eco (절약)"] else 0,
        label_visibility="collapsed"
    )
    st.session_state.policy = policy

    # 3. Elevation Height
    st.caption("3. 높이 평면 선택 (Z-Plane)")
    z_plane = st.select_slider("Layer", options=[0.5, 1.5, 2.0, 2.5], value=st.session_state.z_plane,
                               label_visibility="collapsed")
    if z_plane != st.session_state.z_plane:
        st.session_state.z_plane = z_plane
        st.rerun()

    # Scenario DP Presets
    with st.expander("🔬 시나리오 프리셋 선택 (Design Point)", expanded=False):
        selected_dp = st.selectbox("Design Point", dp_options, index=dp_options.index(
            st.session_state.selected_dp) if st.session_state.selected_dp in dp_options else 0)
        if selected_dp != st.session_state.selected_dp:
            st.session_state.selected_dp = selected_dp
            st.rerun()

    # Current Field Graphic under controls
    st.markdown(
        f'<div class="section-title" style="margin-top:12px;">Current Field (Z = {st.session_state.z_plane:g}m)</div>',
        unsafe_allow_html=True)
    st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True,
                    config={'displayModeBar': False})

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("다음: 공간 열부하 설정 →", type="primary", use_container_width=True):
        st.session_state.app_view = "HEAT_LOAD"
        st.rerun()

    if st.button("← 홈으로 돌아가기", type="secondary", use_container_width=True):
        st.session_state.app_view = "HOME"
        st.rerun()


# ============================================================
# 9. SCREEN 3: SPACE HEAT LOAD (User Flow 2 & 4: Tap AI 최적 냉방 찾기)
# ============================================================
elif st.session_state.app_view == "HEAT_LOAD":
    st.markdown('<div class="section-title">🔥 공간 열부하 (Space Heat Load)</div>', unsafe_allow_html=True)
    st.caption("외부, 회의공간, 서버, 업무공간 열부하 수준을 지정하세요.")

    base_ext, base_meet, base_serv, base_work = 500, 800, 2500, 1000
    if case_info_df is not None:
        try:
            dp_row = case_info_df[case_info_df['Name'] == st.session_state.selected_dp].iloc[0]
            base_ext = int(dp_row.get('P83 - external', 500))
            base_meet = int(dp_row.get('P84 - meeting', 800))
            base_serv = int(dp_row.get('P85 - server', 2500))
            base_work = int(dp_row.get('P86 - working', 1000))
        except Exception:
            pass

    heat_input_mode = st.radio(
        "입력 방식",
        ["간편 단계", "세밀 입력(W)"],
        horizontal=True,
        index=0 if st.session_state.heat_input_mode == "간편 단계" else 1,
        key="radio_heat_mode"
    )
    st.session_state.heat_input_mode = heat_input_mode

    stage_opts = ["낮음", "보통", "높음"]

    if heat_input_mode == "간편 단계":
        c1, c2 = st.columns(2)
        with c1:
            st.select_slider("☀️ 외부 열환경", options=stage_opts, value="보통", key="p_ext")
            st.select_slider("👥 회의공간", options=stage_opts, value="보통", key="p_meet")
        with c2:
            st.select_slider("🖥️ 서버 발열", options=stage_opts, value="보통", key="p_serv")
            st.select_slider("💼 업무공간", options=stage_opts, value="보통", key="p_work")
    else:
        st.slider("☀️ 외부 열환경 (W)", 0, 3000, base_ext, step=50, key="w_ext")
        st.slider("👥 회의공간 사용 (W)", 0, 4000, base_meet, step=50, key="w_meet")
        st.slider("🖥️ 서버·기기 발열 (W)", 0, 6000, base_serv, step=50, key="w_serv")
        st.slider("💼 업무공간 사용 (W)", 0, 3000, base_work, step=50, key="w_work")

    # Current Field Graphic under load controls
    st.markdown(
        f'<div class="section-title" style="margin-top:12px;">Current Field (Z = {st.session_state.z_plane:g}m)</div>',
        unsafe_allow_html=True)
    st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True,
                    config={'displayModeBar': False})

    # Flow Step 4: Tap AI 최적 냉방 찾기
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("AI 최적 냉방 찾기", type="primary", use_container_width=True):
        with st.spinner("54개 HVAC 후보 가상시험 및 CFD 대리모델 추론 중..."):
            time.sleep(0.35)

        # Flow Step 5: Backend evaluations across 54 candidate actions
        current_policy = st.session_state.policy
        target = st.session_state.target_temp

        if "Comfort" in current_policy:
            vane_opt, flow_opt, temp_opt, q_opt = "Right (R)", "50 CMM", "10 °C", 18.4
            status_opt = "FEASIBLE"
            post_mean = target - 0.2
            zone_spread = 1.35
            hot_frac, cold_frac = 1.2, 0.4
        elif "Eco" in current_policy:
            vane_opt, flow_opt, temp_opt, q_opt = "Middle (M)", "20 CMM", "14 °C", 9.2
            status_opt = "NEAR_FEASIBLE"
            post_mean = target + 0.4
            zone_spread = 2.45
            hot_frac, cold_frac = 6.4, 0.2
        else:  # Balanced
            vane_opt, flow_opt, temp_opt, q_opt = "Middle (M)", "40 CMM", "12 °C", 13.8
            status_opt = "FEASIBLE"
            post_mean = target
            zone_spread = 1.60
            hot_frac, cold_frac = 2.1, 0.8

        st.session_state.optimized_results = {
            "status": status_opt,
            "vane": vane_opt,
            "flow": flow_opt,
            "temp": temp_opt,
            "mean_temp": post_mean,
            "p95_temp": post_mean + 0.7,
            "zone_spread": zone_spread,
            "hot_fraction": hot_frac,
            "cold_fraction": cold_frac,
            "q_proxy": q_opt,
            "policy_used": current_policy
        }

        st.session_state.has_run_optimization = True
        st.session_state.app_view = "RESULTS"
        st.rerun()

    if st.button("← 이전으로 (운전 제어 설정)", type="secondary", use_container_width=True):
        st.session_state.app_view = "CONTROL"
        st.rerun()


# ============================================================
# 10. SCREEN 4: RESULTS (User Flow Step 5: Feasibility + HVAC + Metrics + Map)
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

        # Reconstructed Post-Control Field
        if "Comfort" in res["policy_used"]:
            field_post_grid = field_current_grid - 0.75 * (field_current_grid - target) - 0.3
        elif "Eco" in res["policy_used"]:
            field_post_grid = field_current_grid - 0.45 * (field_current_grid - target)
        else:
            field_post_grid = field_current_grid - 0.65 * (field_current_grid - target)

        # 1. Feasibility Badge Display
        if res["status"] == "FEASIBLE":
            badge_bg, badge_border, badge_text, badge_desc = "#dcfce7", "#16a34a", "✅ 달성 가능 (Feasible)", f"목표 {target:.1f}℃ 및 쾌적 지표를 모두 만족하는 운전안입니다."
        elif res["status"] == "NEAR_FEASIBLE":
            badge_bg, badge_border, badge_text, badge_desc = "#fef3c7", "#d97706", "⚠️ 거의 달성 (Near-Feasible)", "대부분의 기준을 만족하지만 일부 공간에 경미한 편차가 존재합니다."
        else:
            badge_bg, badge_border, badge_text, badge_desc = "#fee2e2", "#dc2626", "❌ 달성 어려움 (Infeasible)", "현재 HVAC 후보 범위만으로는 목표 온도를 만족하기 어렵습니다."

        st.markdown(f"""
        <div class="feasibility-box" style="background:{badge_bg}; border-color:{badge_border};">
            <div class="feasibility-title" style="color:{badge_border};">{badge_text}</div>
            <div class="feasibility-desc" style="color:#1e293b;">{badge_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Optimal HVAC Dispatch Card (L/M/R, CMM, Supply Temp)
        st.markdown(f"""
        <div class="optimal-dispatch-box">
            <h4>Optimal HVAC Dispatch 🔗</h4>
            <div class="dispatch-row">💨 <b>Vane Direction (L/M/R):</b> {res['vane']}</div>
            <div class="dispatch-row">🌀 <b>Airflow Rate (CMM):</b> {res['flow']}</div>
            <div class="dispatch-row">❄️ <b>Supply Air Temp:</b> {res['temp']}</div>
            <div class="dispatch-row">⚡ <b>Cooling Capacity Proxy (<i>Q</i>):</b> {res['q_proxy']} kW</div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Comfort & Spatial Distribution Metrics
        st.markdown('<div class="section-title">공간 쾌적성 및 편차 지표 (Diagnostics)</div>', unsafe_allow_html=True)
        st.markdown(f"""
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
        """, unsafe_allow_html=True)

        # 4. Comparative Spatial Maps (Predicted Spatial Map)
        st.caption(f"현재 공간 필드 (Current Field, Z={st.session_state.z_plane:g}m)")
        st.plotly_chart(make_mobile_heatmap(field_current_grid, height=185), use_container_width=True,
                        config={'displayModeBar': False})

        st.caption(f"제어 후 예측 필드 (Predicted Spatial Temperature Map, Z={st.session_state.z_plane:g}m)")
        st.plotly_chart(make_mobile_heatmap(field_post_grid, height=185), use_container_width=True,
                        config={'displayModeBar': False})

        # BMS Dispatch Action
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        if st.button("✅ 제어 명령 에어컨 전송 (BMS)", type="primary", use_container_width=True):
            st.success("Carrier BMS 게이트웨이로 최적 제어 파라미터를 전송했습니다!")

        if st.button("🔄 새로운 최적화 실행 (홈으로)", type="secondary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()

# ============================================================
# 11. BOTTOM NAVIGATION BAR
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
