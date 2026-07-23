from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

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
:root { --pf-bg:#f5f7fb; --pf-card:#ffffff; --pf-text:#172033; --pf-muted:#687386; }
.stApp { background: var(--pf-bg); }
.block-container { max-width: 760px; padding-top: 1.1rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(245,247,251,.92); }
.pf-hero {
  border-radius: 26px; padding: 22px 20px; margin: 0 0 14px 0;
  background: linear-gradient(135deg,#13233b 0%,#1c4770 60%,#267a9b 100%);
  color:white; box-shadow:0 10px 28px rgba(19,35,59,.18);
}
.pf-hero h1 { font-size:1.72rem; line-height:1.15; margin:0 0 8px 0; color:white; }
.pf-hero p { margin:0; color:rgba(255,255,255,.82); font-size:.96rem; }
.pf-card {
  background:white; border:1px solid #e8edf4; border-radius:22px;
  padding:18px; margin:10px 0; box-shadow:0 4px 14px rgba(25,42,70,.05);
}
.pf-status { border-radius:22px; padding:18px; margin:12px 0; }
.pf-ok { background:#eaf8f0; border:1px solid #b8e7cb; }
.pf-near { background:#fff7e5; border:1px solid #f3d68b; }
.pf-no { background:#fff0f0; border:1px solid #f2bbbb; }
.pf-status-title { font-size:1.35rem; font-weight:800; margin-bottom:5px; }
.pf-muted { color:#687386; font-size:.9rem; }
.pf-big { font-size:2rem; font-weight:800; color:#172033; line-height:1; }
.pf-chip { display:inline-block; padding:6px 10px; border-radius:999px; background:#eef3f8; margin:2px 3px; font-weight:700; font-size:.85rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e8edf4; padding:12px; border-radius:18px; }
.stButton > button { width:100%; border-radius:16px; min-height:52px; font-weight:800; font-size:1.02rem; }
.stDownloadButton > button { width:100%; border-radius:14px; }
@media (max-width: 640px) {
  .block-container { padding-left: .85rem; padding-right: .85rem; padding-top:.65rem; }
  .pf-hero { padding:20px 17px; border-radius:22px; }
  .pf-hero h1 { font-size:1.52rem; }
  .pf-card { padding:15px; border-radius:18px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================
LEVEL_KO_TO_KEY = {"낮음": "low", "보통": "medium", "높음": "high"}
POLICY_KO_TO_KEY = {"⚖️ 균형": "balanced", "🛋️ 쾌적 우선": "comfort_first", "🍃 절약 우선": "eco_first"}


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

    recs = json.loads((output_dir / "hvac_recommendations.json").read_text(encoding="utf-8"))
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
        cls, icon, title, desc = "pf-ok", "✅", "달성 가능", f"목표 {target:.1f}℃를 만족하는 운전안을 찾았습니다."
    elif status == "NEAR_FEASIBLE":
        cls, icon, title, desc = "pf-near", "⚠️", "거의 달성", "대부분의 기준은 만족하지만 일부 조건을 조금 초과합니다."
    else:
        cls, icon, title, desc = "pf-no", "❌", "달성 어려움", "현재 HVAC 후보 범위만으로 모든 쾌적 기준을 만족하기 어렵습니다."
    st.markdown(
        f'<div class="pf-status {cls}"><div class="pf-status-title">{icon} {title}</div><div>{desc}</div></div>',
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
        val, limit, exceed, met = float(d["value"]), float(d["limit"]), float(d["exceedance"]), bool(d["met"])
        if key in {"hot_fraction", "cold_fraction"}:
            val, limit, exceed = val * 100, limit * 100, exceed * 100
            extra = "" if met else f"+{exceed:.2f}%p"
            rows.append(["✅" if met else "⚠️", label, f"{val:.2f}%", f"≤ {limit:.2f}%", extra])
        else:
            extra = "" if met else f"+{exceed:.2f}℃"
            rows.append(["✅" if met else "⚠️", label, f"{val:.2f}℃", f"≤ {limit:.2f}℃", extra])
    return pd.DataFrame(rows, columns=["", "항목", "예측", "기준", "초과"])


def temperature_map(field_df: pd.DataFrame):
    z_values = np.sort(field_df["z_m"].unique())
    target_z = z_values[np.argmin(np.abs(z_values - 1.5))]
    d = field_df[np.isclose(field_df["z_m"], target_z)].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sc = ax.scatter(
        d["x_m"], d["y_m"], c=d["pred_temperature_C"],
        s=58, cmap="coolwarm", edgecolors="none"
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Temperature (°C)")
    ax.set_title(f"Predicted temperature field · z={target_z:g} m")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


# ============================================================
# Header
# ============================================================
st.markdown(
    """
<div class="pf-hero">
  <h1>❄️ PopField AI Smart Cooling</h1>
  <p>공간 상태를 입력하면 AI Digital Twin이 54개 냉방 운전을 가상시험해 가장 적합한 설정을 추천합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Model assets
# ============================================================
local_ckpt = _find_local_checkpoint()
local_case = _find_local_case_info()

with st.expander("⚙️ 모델 연결", expanded=(local_ckpt is None or local_case is None)):
    st.caption("배포 시 best.pt와 Case Info.xlsx를 앱 폴더에 두면 사용자는 이 설정을 볼 필요가 없습니다.")
    uploaded_pt = st.file_uploader("best.pt", type=["pt"], key="pt")
    uploaded_xlsx = st.file_uploader("Case Info.xlsx", type=["xlsx"], key="xlsx")
    checkpoint_path = _materialize_upload(uploaded_pt, ".pt") if uploaded_pt else local_ckpt
    case_info_path = _materialize_upload(uploaded_xlsx, ".xlsx") if uploaded_xlsx else local_case
    force_cpu = st.checkbox("CPU로 실행", value=False, help="GPU가 없는 배포 환경에서는 자동으로 CPU를 사용합니다.")
    if checkpoint_path and case_info_path:
        st.success("AI 모델이 연결되었습니다.")
    else:
        st.warning("best.pt와 Case Info.xlsx를 연결해 주세요.")

# ============================================================
# Main inputs
# ============================================================
st.markdown('<div class="pf-card"><b>1. 원하는 실내 환경</b><div class="pf-muted">일반 사용자는 W 단위를 몰라도 됩니다.</div></div>', unsafe_allow_html=True)

target_temp = st.slider("🌡️ 목표 온도", min_value=22.0, max_value=28.0, value=24.0, step=0.5, format="%.1f℃")

col1, col2 = st.columns(2)
with col1:
    external_ko = st.select_slider("☀️ 외부 열환경", options=["낮음", "보통", "높음"], value="보통")
    meeting_ko = st.select_slider("👥 회의공간 사용", options=["낮음", "보통", "높음"], value="높음")
with col2:
    server_ko = st.select_slider("🖥️ 서버·기기 발열", options=["낮음", "보통", "높음"], value="높음")
    working_ko = st.select_slider("💼 업무공간 사용", options=["낮음", "보통", "높음"], value="보통")

policy_ko = st.radio(
    "🎯 운전 목표",
    ["⚖️ 균형", "🛋️ 쾌적 우선", "🍃 절약 우선"],
    horizontal=True,
    index=0,
)

levels = {
    "external": LEVEL_KO_TO_KEY[external_ko],
    "meeting": LEVEL_KO_TO_KEY[meeting_ko],
    "server": LEVEL_KO_TO_KEY[server_ko],
    "working": LEVEL_KO_TO_KEY[working_ko],
}
policy = POLICY_KO_TO_KEY[policy_ko]

run = st.button("✨ AI 최적 냉방 찾기", type="primary", disabled=not (checkpoint_path and case_info_path))

if run:
    try:
        with st.spinner("PopField가 54개 HVAC 운전을 가상시험하고 있습니다…"):
            result = run_ai(
                checkpoint_path=checkpoint_path,
                case_info_path=case_info_path,
                target_temp=target_temp,
                levels=levels,
                policy=policy,
                force_cpu=force_cpu,
            )
        st.session_state["last_result"] = result
        st.session_state["last_target"] = target_temp
    except Exception as e:
        st.exception(e)

# ============================================================
# Results
# ============================================================
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    target_for_result = float(st.session_state.get("last_target", target_temp))
    rec = result["recommendation"]

    st.markdown("### 2. AI 분석 결과")
    status_box(result["status"], target_for_result)

    st.markdown('<div class="pf-card"><b>추천 HVAC 설정</b></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("토출 방향", direction_text(rec))
    c2.metric("풍량", f'{float(rec["CMM"]):.0f} CMM')
    c3.metric("토출 온도", f'{float(rec["AirTemp_C"]):.0f}℃')

    c1, c2 = st.columns(2)
    c1.metric("예상 평균온도", f'{float(rec["mean_temp_C"]):.2f}℃')
    c2.metric("P95 온도", f'{float(rec["p95_temp_C"]):.2f}℃')

    st.caption(f'AI가 {result["num_actions"]}개 후보를 평가 · 계산 {result["decision_ms"]:.0f} ms · 실행장치 {result["device"]}')

    st.markdown("#### 쾌적 조건 확인")
    checks = constraint_rows(result["diag"])
    if len(checks):
        st.dataframe(checks, use_container_width=True, hide_index=True)

    if result["status"] != "FEASIBLE":
        cap = result.get("additional_capacity", {}) or {}
        gap = cap.get("additional_sensible_cooling_kw_lower_bound_at_best_achievable")
        unavoidable = cap.get("additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity")
        with st.expander("🏢 설비 한계 참고"):
            if gap is not None:
                st.write(f"열수지 기준 추가 냉방 여유 참고값: **{float(gap):.2f} kW**")
            if unavoidable is not None:
                st.write(f"최대 후보 냉방에서도 남는 열수지 차이: **{float(unavoidable):.2f} kW**")
            st.caption("이 값은 현열 열수지 기반 참고치이며 실제 증설 용량, 전력소비 또는 전기요금 절감량이 아닙니다.")

    st.markdown("#### 🌡️ AI 추천 후 예상 온도분포")
    fig = temperature_map(result["field"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("현재 데이터에는 공식 Zone 경계가 없어 Zone 지표는 임시 XY 4분할 기준입니다. 온도분포 자체는 PopField가 1,270개 공간점에 대해 예측합니다.")

    with st.expander("🔎 입력값이 AI 내부에서 어떻게 변환됐나"):
        mapped = pd.DataFrame([
            ["외부 열환경", external_ko, result["loads"]["external"]],
            ["회의공간", meeting_ko, result["loads"]["meeting"]],
            ["서버·기기", server_ko, result["loads"]["server"]],
            ["업무공간", working_ko, result["loads"]["working"]],
        ], columns=["입력", "사용자 선택", "모델 입력(W)"])
        st.dataframe(mapped, use_container_width=True, hide_index=True)
        st.caption("낮음/보통/높음은 Case Info에 실제 존재하는 열부하 단계로 매핑되는 데모용 입력입니다.")

    csv_data = result["field"].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "예측 온도장 CSV 저장",
        data=csv_data,
        file_name="popfield_selected_temperature_field.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Demo scope · steady-state CFD surrogate decision support. 실제 전력/요금 절감 및 동적 폐루프 제어 검증을 의미하지 않습니다.")
