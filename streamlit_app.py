from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

import demo_v2_backend as hvac


# ============================================================
# Page / mobile UI
# ============================================================
st.set_page_config(
    page_title="PopField AI Smart Cooling",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --pf-bg: #f6f9fd;
  --pf-card: #ffffff;
  --pf-text: #131e33;
  --pf-muted: #616e85;
  --pf-primary: #144f8c;
  --pf-primary-soft: #e8f2fa;
  --pf-line: #e8edf4;
}

html, body, [class*="css"] {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background: #edf1f5;
}

.block-container {
  max-width: 430px !important;
  padding: 0 !important;
  margin: 0 auto !important;
  min-height: 100vh;
  background: var(--pf-bg);
}

#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden;
  height: 0;
}

.pf-shell {
  padding: 24px 20px 18px 20px;
}

.pf-title {
  color: var(--pf-text);
  font-size: 26px;
  font-weight: 700;
  line-height: 1.2;
  margin: 0 0 12px 0;
}

.pf-subtitle {
  color: var(--pf-muted);
  font-size: 14px;
  margin: 0 0 16px 0;
}

.pf-card {
  background: var(--pf-card);
  border-radius: 20px;
  padding: 18px;
  margin: 0 0 16px 0;
  border: 1px solid rgba(232,237,244,.55);
}

.pf-label {
  color: var(--pf-muted);
  font-size: 13px;
  margin-bottom: 6px;
}

.pf-temp {
  color: #121f33;
  font-size: 34px;
  font-weight: 700;
  line-height: 1.05;
  margin-bottom: 8px;
}

.pf-blue-text {
  color: #3073b8;
  font-size: 14px;
}

.pf-section-title {
  color: var(--pf-text);
  font-size: 17px;
  font-weight: 700;
  margin: 4px 0 10px 0;
}

.pf-twin {
  background: var(--pf-primary-soft);
  height: 250px;
  border-radius: 22px;
  margin: 0 0 16px 0;
  padding: 18px;
  position: relative;
  overflow: hidden;
}

.pf-twin-title {
  color: #1a2e47;
  font-size: 16px;
  font-weight: 700;
}

.pf-blob {
  position: absolute;
  border-radius: 999px;
  opacity: .88;
}
.pf-blob.cool {
  width: 92px;
  height: 92px;
  left: 44px;
  top: 104px;
  background: #5b95da;
}
.pf-blob.hot {
  width: 74px;
  height: 74px;
  right: 46px;
  top: 110px;
  background: #ef6b56;
}

.pf-status {
  border-radius: 20px;
  padding: 18px;
  margin: 0 0 16px 0;
}
.pf-ok { background: #eaf8f0; border: 1px solid #b8e7cb; }
.pf-near { background: #fff7e5; border: 1px solid #f3d68b; }
.pf-no { background: #fff0f0; border: 1px solid #f2bbbb; }
.pf-status-title {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 5px;
  color: var(--pf-text);
}

.pf-metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.pf-metric {
  background: white;
  border: 1px solid var(--pf-line);
  border-radius: 18px;
  padding: 14px;
}
.pf-metric-label {
  color: var(--pf-muted);
  font-size: 12px;
  margin-bottom: 7px;
}
.pf-metric-value {
  color: var(--pf-text);
  font-weight: 800;
  font-size: 20px;
}

.pf-bottom {
  background: white;
  border-top: 1px solid var(--pf-line);
  padding: 9px 12px 4px 12px;
  margin-top: 18px;
  position: sticky;
  bottom: 0;
  z-index: 10;
}

div[data-testid="stButton"] > button {
  width: 100%;
  border-radius: 16px;
  min-height: 52px;
  font-weight: 800;
  font-size: 1rem;
}

div[data-testid="stButton"] > button[kind="primary"] {
  background: var(--pf-primary);
  border-color: var(--pf-primary);
}

div[data-testid="stDownloadButton"] > button {
  width: 100%;
  border-radius: 16px;
  min-height: 48px;
}

div[data-testid="stMetric"] {
  background: white;
  border: 1px solid var(--pf-line);
  padding: 12px;
  border-radius: 18px;
}

div[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
}

[data-testid="stSlider"],
[data-testid="stSelectSlider"],
[data-testid="stRadio"] {
  background: white;
  border: 1px solid var(--pf-line);
  border-radius: 18px;
  padding: 14px 16px 10px 16px;
  margin-bottom: 10px;
}

.pf-note {
  color: var(--pf-muted);
  font-size: 12px;
  line-height: 1.5;
}

@media (max-width: 430px) {
  .stApp { background: var(--pf-bg); }
  .block-container { max-width: 100% !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers / AI backend
# ============================================================
LEVEL_KO_TO_KEY = {"낮음": "low", "보통": "medium", "높음": "high"}
POLICY_KO_TO_KEY = {
    "⚖️ 균형": "balanced",
    "🛋️ 쾌적 우선": "comfort_first",
    "🍃 절약 우선": "eco_first",
}


def _materialize_upload(uploaded, suffix: str) -> str:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / "popfield_streamlit_assets"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def _find_local_case_info() -> str | None:
    candidates = [
        Path("Case Info 200 DesignPoints - 최종본.xlsx"),
        Path("Case_Info.xlsx"),
        Path("case_info.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def _find_local_checkpoint() -> str | None:
    for p in [Path("best.pt"), Path("model/best.pt"), Path("assets/best.pt")]:
        if p.exists():
            return str(p.resolve())
    return None


@st.cache_resource(show_spinner=False)
def load_runtime(checkpoint_path: str, case_info_path: str, force_cpu: bool):
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    ckpt, model, scalers, coords = hvac.load_checkpoint(checkpoint_path, device)
    coords_norm_t = torch.from_numpy(
        scalers["coord"].transform(coords).astype(np.float32)
    ).to(device)
    case_df = hvac.load_case_info(case_info_path)
    level_mapping = {
        "external": hvac._observed_level_map(case_df, "P83 - external"),
        "meeting": hvac._observed_level_map(case_df, "P84 - meeting"),
        "server": hvac._observed_level_map(case_df, "P85 - server"),
        "working": hvac._observed_level_map(case_df, "P86 - working"),
    }
    return ckpt, model, scalers, coords, coords_norm_t, case_df, level_mapping, device


def _args_for_diag(target: float) -> SimpleNamespace:
    return SimpleNamespace(
        target_temp=float(target),
        comfort_band=2.0,
        max_zone_range=2.0,
        max_hot_fraction=0.05,
        max_cold_fraction=0.05,
        max_p95_temp=None,
        demo_near_zone_margin=0.25,
        demo_near_hot_margin_pp=1.0,
        demo_near_cold_margin_pp=1.0,
        demo_near_p95_margin=0.25,
    )


def run_ai(
    checkpoint_path: str,
    case_info_path: str,
    target_temp: float,
    levels: Dict[str, str],
    policy: str,
    force_cpu: bool = False,
) -> Dict:
    ckpt, model, scalers, coords, coords_norm_t, case_df, level_mapping, device = load_runtime(
        checkpoint_path, case_info_path, force_cpu
    )
    loads = {k: float(level_mapping[k][v]) for k, v in levels.items()}

    output_dir = Path(tempfile.gettempdir()) / "popfield_streamlit_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    opt = hvac.optimize_hvac(
        model=model,
        case_df=case_df,
        loads=loads,
        cond_scaler=scalers["cond"],
        coords=coords,
        coords_norm_t=coords_norm_t,
        field_scaler=scalers["field"],
        ra_scaler=scalers["ra"],
        device=device,
        save_dir=output_dir,
        zone_json=None,
        target_temp_c=float(target_temp),
        comfort_band_c=2.0,
        max_zone_range_c=2.0,
        max_hot_fraction=0.05,
        max_cold_fraction=0.05,
        max_p95_temp_c=None,
        energy_weight=0.35,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    recs = json.loads(
        (output_dir / "hvac_recommendations.json").read_text(encoding="utf-8")
    )
    strict_feasible = bool(recs.get("fully_feasible_action_exists", False))
    if strict_feasible:
        rec = recs[policy]
        policy_used = policy
    else:
        rec = recs["best_achievable"]
        policy_used = "best_achievable"

    diag = hvac._demo_constraint_diagnostics(rec, _args_for_diag(target_temp))
    field_path = hvac._save_demo_selected_field(
        model, rec, loads, scalers, coords, coords_norm_t, device, output_dir
    )
    field_df = pd.read_csv(field_path)

    return {
        "status": str(diag["status"]),
        "status_label": str(diag["label_ko"]),
        "diag": diag,
        "recommendation": rec,
        "loads": loads,
        "levels": levels,
        "policy": policy,
        "policy_used": policy_used,
        "strict_feasible": strict_feasible,
        "num_actions": int(len(opt)),
        "decision_ms": float(elapsed_ms),
        "field": field_df,
        "all_candidates": opt,
        "device": device,
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "additional_capacity": recs.get("additional_capacity_estimate", {}),
    }


def direction_text(rec: Dict) -> str:
    return hvac._direction_text(rec)


def status_box(status: str, target: float):
    if status == "FEASIBLE":
        cls, icon, title, desc = (
            "pf-ok",
            "✅",
            "달성 가능",
            f"목표 {target:.1f}℃를 만족하는 운전안을 찾았습니다.",
        )
    elif status == "NEAR_FEASIBLE":
        cls, icon, title, desc = (
            "pf-near",
            "⚠️",
            "거의 달성",
            "대부분의 기준은 만족하지만 일부 조건을 조금 초과합니다.",
        )
    else:
        cls, icon, title, desc = (
            "pf-no",
            "❌",
            "달성 어려움",
            "현재 HVAC 후보 범위만으로 모든 쾌적 기준을 만족하기 어렵습니다.",
        )
    st.markdown(
        f'<div class="pf-status {cls}">'
        f'<div class="pf-status-title">{icon} {title}</div>'
        f'<div>{desc}</div></div>',
        unsafe_allow_html=True,
    )


def constraint_rows(diag: Dict) -> pd.DataFrame:
    labels = {
        "zone_range": ("Zone 편차", "℃"),
        "hot_fraction": ("Hotspot", "%"),
        "cold_fraction": ("Coldspot", "%"),
        "p95_temperature": ("P95 온도", "℃"),
    }
    rows = []
    details = diag.get("details") or diag.get("constraints") or {}
    for key, (label, unit) in labels.items():
        d = details.get(key, {})
        if not d:
            continue
        val = float(d["value"])
        limit = float(d["limit"])
        exceed = float(d["exceedance"])
        met = bool(d["met"])
        if key in {"hot_fraction", "cold_fraction"}:
            val, limit, exceed = val * 100, limit * 100, exceed * 100
            extra = "" if met else f"+{exceed:.2f}%p"
            rows.append(
                ["✅" if met else "⚠️", label, f"{val:.2f}%", f"≤ {limit:.2f}%", extra]
            )
        else:
            extra = "" if met else f"+{exceed:.2f}℃"
            rows.append(
                ["✅" if met else "⚠️", label, f"{val:.2f}℃", f"≤ {limit:.2f}℃", extra]
            )
    return pd.DataFrame(rows, columns=["", "항목", "예측", "기준", "초과"])


def temperature_map(field_df: pd.DataFrame):
    z_values = np.sort(field_df["z_m"].unique())
    target_z = z_values[np.argmin(np.abs(z_values - 1.5))]
    d = field_df[np.isclose(field_df["z_m"], target_z)].copy()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#e8f2fa")

    sc = ax.scatter(
        d["x_m"],
        d["y_m"],
        c=d["pred_temperature_C"],
        s=74,
        cmap="coolwarm",
        edgecolors="none",
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Temperature (°C)", fontsize=9)

    ax.set_title(f"Digital Twin · z={target_z:g} m", fontsize=12, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================
# State
# ============================================================
defaults = {
    "page": "home",
    "target_temp": 24.0,
    "external_ko": "보통",
    "meeting_ko": "높음",
    "server_ko": "높음",
    "working_ko": "보통",
    "policy_ko": "⚖️ 균형",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


local_ckpt = _find_local_checkpoint()
local_case = _find_local_case_info()

# Uploaded overrides survive reruns in session state
if "checkpoint_path" not in st.session_state:
    st.session_state["checkpoint_path"] = local_ckpt
if "case_info_path" not in st.session_state:
    st.session_state["case_info_path"] = local_case
if "force_cpu" not in st.session_state:
    st.session_state["force_cpu"] = False


def app_header(title: str):
    st.markdown(
        f"""
        <div class="pf-shell" style="padding-bottom:0">
          <div class="pf-title">{title}</div>
          <div class="pf-subtitle">PopField AI Smart Cooling</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go(page: str):
    st.session_state["page"] = page
    st.rerun()


def bottom_nav(active: str):
    st.markdown('<div class="pf-bottom">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Home", key=f"nav_home_{active}", use_container_width=True):
            go("home")
    with c2:
        label = "Analysis"
        if st.button(label, key=f"nav_analysis_{active}", use_container_width=True):
            if "last_result" in st.session_state:
                go("result")
            else:
                go("setup")
    with c3:
        if st.button("Settings", key=f"nav_settings_{active}", use_container_width=True):
            go("setup")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================
if st.session_state["page"] == "home":
    app_header("PopField")

    st.markdown(
        """
        <div class="pf-shell" style="padding-top:0">
          <div class="pf-card">
            <div class="pf-label">현재 공간 상태</div>
            <div class="pf-temp">25.3°C</div>
            <div class="pf-blue-text">목표 24.0°C · 냉방 최적화 필요</div>
          </div>

          <div class="pf-twin">
            <div class="pf-twin-title">Digital Twin · Temperature Field</div>
            <div class="pf-blob cool"></div>
            <div class="pf-blob hot"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([0.05, 0.9, 0.05])
    with center:
        if st.button("AI 냉방 최적화 시작", type="primary", use_container_width=True):
            go("setup")

    st.markdown(
        """
        <div class="pf-shell" style="padding-top:10px;padding-bottom:0">
          <div class="pf-note">
            입력한 공간 조건을 바탕으로 PopField가 HVAC 후보를 가상시험하고
            목표 온도와 쾌적 조건을 만족하는 운전안을 찾습니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    bottom_nav("home")


# ============================================================
# AI SETUP
# ============================================================
elif st.session_state["page"] == "setup":
    app_header("AI Cooling Setup")

    st.markdown(
        """
        <div class="pf-shell" style="padding-top:0;padding-bottom:6px">
          <div class="pf-card">
            <div class="pf-section-title">원하는 실내 환경</div>
            <div class="pf-note">
              일반 사용자는 W 단위를 입력할 필요 없이 공간 상태만 선택하면 됩니다.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["target_temp"] = st.slider(
        "🌡️ 목표 온도",
        min_value=22.0,
        max_value=28.0,
        value=float(st.session_state["target_temp"]),
        step=0.5,
        format="%.1f℃",
        key="target_temp_widget",
    )

    st.session_state["external_ko"] = st.select_slider(
        "☀️ 외부 열환경",
        options=["낮음", "보통", "높음"],
        value=st.session_state["external_ko"],
        key="external_widget",
    )
    st.session_state["meeting_ko"] = st.select_slider(
        "👥 회의공간 사용",
        options=["낮음", "보통", "높음"],
        value=st.session_state["meeting_ko"],
        key="meeting_widget",
    )
    st.session_state["server_ko"] = st.select_slider(
        "🖥️ 서버·기기 발열",
        options=["낮음", "보통", "높음"],
        value=st.session_state["server_ko"],
        key="server_widget",
    )
    st.session_state["working_ko"] = st.select_slider(
        "💼 업무공간 사용",
        options=["낮음", "보통", "높음"],
        value=st.session_state["working_ko"],
        key="working_widget",
    )

    st.session_state["policy_ko"] = st.radio(
        "🎯 운전 목표",
        ["⚖️ 균형", "🛋️ 쾌적 우선", "🍃 절약 우선"],
        horizontal=False,
        index=["⚖️ 균형", "🛋️ 쾌적 우선", "🍃 절약 우선"].index(
            st.session_state["policy_ko"]
        ),
        key="policy_widget",
    )

    with st.expander(
        "⚙️ 모델 연결",
        expanded=not (
            st.session_state.get("checkpoint_path")
            and st.session_state.get("case_info_path")
        ),
    ):
        st.caption(
            "배포 시 best.pt와 Case Info.xlsx를 앱 폴더에 두면 자동 연결됩니다."
        )
        uploaded_pt = st.file_uploader("best.pt", type=["pt"], key="pt")
        uploaded_xlsx = st.file_uploader("Case Info.xlsx", type=["xlsx"], key="xlsx")
        if uploaded_pt:
            st.session_state["checkpoint_path"] = _materialize_upload(uploaded_pt, ".pt")
        if uploaded_xlsx:
            st.session_state["case_info_path"] = _materialize_upload(uploaded_xlsx, ".xlsx")
        st.session_state["force_cpu"] = st.checkbox(
            "CPU로 실행",
            value=bool(st.session_state["force_cpu"]),
            help="GPU가 없으면 자동으로 CPU를 사용합니다.",
        )
        if st.session_state.get("checkpoint_path") and st.session_state.get("case_info_path"):
            st.success("AI 모델이 연결되었습니다.")
        else:
            st.warning("best.pt와 Case Info.xlsx를 연결해 주세요.")

    levels = {
        "external": LEVEL_KO_TO_KEY[st.session_state["external_ko"]],
        "meeting": LEVEL_KO_TO_KEY[st.session_state["meeting_ko"]],
        "server": LEVEL_KO_TO_KEY[st.session_state["server_ko"]],
        "working": LEVEL_KO_TO_KEY[st.session_state["working_ko"]],
    }
    policy = POLICY_KO_TO_KEY[st.session_state["policy_ko"]]

    ready = bool(
        st.session_state.get("checkpoint_path")
        and st.session_state.get("case_info_path")
    )

    if st.button(
        "✨ AI 분석 시작",
        type="primary",
        disabled=not ready,
        use_container_width=True,
    ):
        try:
            with st.spinner("PopField가 54개 HVAC 운전을 가상시험하고 있습니다…"):
                result = run_ai(
                    checkpoint_path=st.session_state["checkpoint_path"],
                    case_info_path=st.session_state["case_info_path"],
                    target_temp=st.session_state["target_temp"],
                    levels=levels,
                    policy=policy,
                    force_cpu=st.session_state["force_cpu"],
                )
            st.session_state["last_result"] = result
            st.session_state["last_target"] = st.session_state["target_temp"]
            st.session_state["page"] = "result"
            st.rerun()
        except Exception as exc:
            st.exception(exc)

    bottom_nav("setup")


# ============================================================
# AI RESULT
# ============================================================
elif st.session_state["page"] == "result":
    app_header("AI Recommendation")

    if "last_result" not in st.session_state:
        st.warning("아직 AI 분석 결과가 없습니다.")
        if st.button("AI 설정으로 이동", type="primary", use_container_width=True):
            go("setup")
        bottom_nav("result_empty")
    else:
        result = st.session_state["last_result"]
        target_for_result = float(
            st.session_state.get("last_target", st.session_state["target_temp"])
        )
        rec = result["recommendation"]

        status_box(result["status"], target_for_result)

        st.markdown(
            f"""
            <div class="pf-shell" style="padding-top:0;padding-bottom:0">
              <div class="pf-section-title">추천 HVAC 설정</div>
              <div class="pf-metric-grid">
                <div class="pf-metric">
                  <div class="pf-metric-label">토출 방향</div>
                  <div class="pf-metric-value">{direction_text(rec)}</div>
                </div>
                <div class="pf-metric">
                  <div class="pf-metric-label">풍량</div>
                  <div class="pf-metric-value">{float(rec["CMM"]):.0f} CMM</div>
                </div>
                <div class="pf-metric">
                  <div class="pf-metric-label">토출 온도</div>
                  <div class="pf-metric-value">{float(rec["AirTemp_C"]):.0f}℃</div>
                </div>
                <div class="pf-metric">
                  <div class="pf-metric-label">예상 평균온도</div>
                  <div class="pf-metric-value">{float(rec["mean_temp_C"]):.2f}℃</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)
        st.markdown('<div class="pf-section-title">Digital Twin · 예상 온도분포</div>', unsafe_allow_html=True)
        fig = temperature_map(result["field"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="pf-shell" style="padding-top:0;padding-bottom:0">
              <div class="pf-card">
                <div class="pf-label">분석 정보</div>
                <div class="pf-blue-text">
                  {result["num_actions"]}개 후보 평가 · {result["decision_ms"]:.0f} ms · {result["device"]}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 쾌적 조건 확인")
        checks = constraint_rows(result["diag"])
        if len(checks):
            st.dataframe(checks, use_container_width=True, hide_index=True)

        if result["status"] != "FEASIBLE":
            cap = result.get("additional_capacity", {}) or {}
            gap = cap.get(
                "additional_sensible_cooling_kw_lower_bound_at_best_achievable"
            )
            unavoidable = cap.get(
                "additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity"
            )
            with st.expander("🏢 설비 한계 참고"):
                if gap is not None:
                    st.write(
                        f"열수지 기준 추가 냉방 여유 참고값: **{float(gap):.2f} kW**"
                    )
                if unavoidable is not None:
                    st.write(
                        f"최대 후보 냉방에서도 남는 열수지 차이: **{float(unavoidable):.2f} kW**"
                    )
                st.caption(
                    "현열 열수지 기반 참고치이며 실제 증설 용량, 전력소비 또는 전기요금 절감량이 아닙니다."
                )

        with st.expander("🔎 입력값이 AI 내부에서 어떻게 변환됐나"):
            mapped = pd.DataFrame(
                [
                    ["외부 열환경", st.session_state["external_ko"], result["loads"]["external"]],
                    ["회의공간", st.session_state["meeting_ko"], result["loads"]["meeting"]],
                    ["서버·기기", st.session_state["server_ko"], result["loads"]["server"]],
                    ["업무공간", st.session_state["working_ko"], result["loads"]["working"]],
                ],
                columns=["입력", "사용자 선택", "모델 입력(W)"],
            )
            st.dataframe(mapped, use_container_width=True, hide_index=True)

        csv_data = result["field"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "예측 온도장 CSV 저장",
            data=csv_data,
            file_name="popfield_selected_temperature_field.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("← 조건 다시 설정", use_container_width=True):
            go("setup")

        st.markdown(
            """
            <div class="pf-shell" style="padding-top:6px;padding-bottom:0">
              <div class="pf-note">
                Demo scope · steady-state CFD surrogate decision support.
                실제 전력/요금 절감 및 동적 폐루프 제어 검증을 의미하지 않습니다.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bottom_nav("result")
