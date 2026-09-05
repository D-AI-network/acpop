from __future__ import annotations

# CFD_RETRIEVAL_FINAL_V4 = 2026-09-03
# UI_REFINEMENT_BUILD = 2026-09-03-v31
# Robust repo-root CFD ZIP auto-discovery (dp*.csv archive detection)

# CFD_RETRIEVAL_BUILD = 2026-09-03-v1_NEAREST_200_REAL_CASES
# FACTOR_UI_BUILD = 2026-09-04-v69

# COOLING_FACTORS_BUILD = 2026-09-03-v20

# SENSOR_RADAR_ROUNDED_BUILD = 2026-09-03-v12

import base64
import io
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

# ------------------------------------------------------------------
# Real PopField inference backend.
# IMPORTANT FOR FAST START:
# Do NOT import PyTorch / the PopField model module during the splash screen.
# They are imported lazily only when inference/optimization is actually needed.
# ------------------------------------------------------------------
COND_COLS = [
    "P80 - Inlet L",
    "P81 - Inlet M",
    "P82 - Inlet R",
    "P83 - external",
    "P84 - meeting",
    "P85 - server",
    "P86 - working",
    "P87 - CMM",
    "P88 - AirTemp",
]
popfield_load_case_info = None
popfield_load_checkpoint = None
popfield_optimize_hvac = None
popfield_predict_conditions = None
POPFIELD_BACKEND_IMPORT_ERROR = None


def _lazy_import_popfield_modules():
    """Import heavy PopField/PyTorch dependencies only when AI inference is requested."""
    global COND_COLS
    global popfield_load_case_info, popfield_load_checkpoint
    global popfield_optimize_hvac, popfield_predict_conditions
    global POPFIELD_BACKEND_IMPORT_ERROR

    if popfield_load_checkpoint is not None and popfield_optimize_hvac is not None:
        return True

    try:
        from demo_v3_hackathon_enhanced import (
            COND_COLS as _COND_COLS,
            load_case_info as _load_case_info,
            load_checkpoint as _load_checkpoint,
            optimize_hvac as _optimize_hvac,
            predict_conditions as _predict_conditions,
        )
        COND_COLS = _COND_COLS
        popfield_load_case_info = _load_case_info
        popfield_load_checkpoint = _load_checkpoint
        popfield_optimize_hvac = _optimize_hvac
        popfield_predict_conditions = _predict_conditions
        POPFIELD_BACKEND_IMPORT_ERROR = None
        return True
    except Exception as exc:
        POPFIELD_BACKEND_IMPORT_ERROR = repr(exc)
        return False

SENSOR_ICON_SVG_B64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI3MiIgaGVpZ2h0PSI3MiIgdmlld0JveD0iMCAwIDcyIDcyIj4KICA8cmVjdCB4PSIxOCIgeT0iOCIgd2lkdGg9IjM2IiBoZWlnaHQ9IjE0IiByeD0iNyIgZmlsbD0iIzEyM2I1ZCIvPgogIDxjaXJjbGUgY3g9IjM2IiBjeT0iMTUiIHI9IjIuNiIgZmlsbD0iIzlmZTRmZiIvPgogIDxwYXRoIGQ9Ik0yOSAzMCBRMzYgMjQgNDMgMzAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzEyM2I1ZCIgc3Ryb2tlLXdpZHRoPSIzLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogIDxwYXRoIGQ9Ik0yNCAzOCBRMzYgMjkgNDggMzgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzEyM2I1ZCIgc3Ryb2tlLXdpZHRoPSIzLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgogIDxwYXRoIGQ9Ik0xOSA0NyBRMzYgMzQgNTMgNDciIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzEyM2I1ZCIgc3Ryb2tlLXdpZHRoPSIzLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4="
RESULT_TITLE_SNOWFLAKE_B64 = "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"
WHITE_SENSOR_DROP_SVG_B64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI3MiIgaGVpZ2h0PSI3MiIgdmlld0JveD0iMCAwIDcyIDcyIj4KICA8cmVjdCB4PSIxMyIgeT0iMTAiIHdpZHRoPSI0NiIgaGVpZ2h0PSI5IiByeD0iMi44IiBmaWxsPSIjZmZmZmZmIi8+CiAgPHJlY3QgeD0iMTMiIHk9IjIyIiB3aWR0aD0iNDYiIGhlaWdodD0iMy41IiByeD0iMS43NSIgZmlsbD0iI2ZmZmZmZiIvPgogIDxwYXRoIGQ9Ik0xNiAyOQogICAgICAgICAgIEMxNiA0NiAyNCA1OCAzNiA1OAogICAgICAgICAgIEM0OCA1OCA1NiA0NiA1NiAyOQogICAgICAgICAgIFoiCiAgICAgICAgZmlsbD0iI2ZmZmZmZiIvPgogIDxjaXJjbGUgY3g9IjM2IiBjeT0iNDMiIHI9IjYuMiIgZmlsbD0iIzEyM2I1ZCIvPgo8L3N2Zz4="

# MERGED_HOME_BUILD = 2026-09-03-v8_HOME_CONTROL_COMBINED

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
  --navy-bg: #102a43;
  --navy-shell: #143552;
  --navy-surface: #183f5f;
  --navy-surface-2: #1d496b;
  --sky: #aee4ff;
  --sky-strong: #d9f3ff;
  --cool: #38bdf8;
  --cool-deep: #1689c9;
  --cool-soft: rgba(56, 189, 248, 0.14);
  --green: #59e391;
  --mist: #9dbfd4;
  --line: rgba(174, 228, 255, 0.16);
  --line-strong: rgba(174, 228, 255, 0.28);
}

html, body, [class*="css"] {
  font-family: 'Noto Sans KR', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background: #0d243a !important;
}

/* Smartphone Shell Container */
.block-container {
  max-width: 440px !important;
  padding: 1.05rem 1.05rem 2rem 1.05rem !important;
  margin: 1.1rem auto !important;
  background: linear-gradient(180deg, #173a59 0%, #102c47 100%) !important;
  border: 1.2px solid rgba(133, 202, 245, 0.20) !important;
  border-radius: 36px !important;
  box-shadow: 0 22px 48px -16px rgba(0, 8, 20, 0.48) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden;
  height: 0;
}

/* Make Streamlit text readable on the navy theme */
.block-container p,
.block-container label,
.block-container span,
.block-container [data-testid="stWidgetLabel"] p,
.block-container [data-testid="stCaptionContainer"] p {
  color: var(--sky-strong);
}

.block-container [data-testid="stCaptionContainer"] p {
  color: var(--mist) !important;
}

/* Top Device Notch */
.phone-notch {
  width: 86px;
  height: 15px;
  background: #07192b;
  border: 1px solid rgba(56, 189, 248, 0.16);
  border-radius: 10px;
  margin: 0 auto 18px auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}
.notch-cam {
  width: 5px;
  height: 5px;
  background: #37536a;
  border-radius: 50%;
}
.notch-speaker {
  width: 22px;
  height: 3px;
  background: #37536a;
  border-radius: 2px;
}

/* Old badge removed */
.app-brand,
.app-brand-icon {
  display: none !important;
}

/* Main app header shown from HOME onward */
.app-title-lockup {
  display: inline-block;
  margin-bottom: 18px;
}
.app-title {
  font-family: 'Outfit', 'Inter', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--sky) !important;
  margin: 0;
  letter-spacing: -0.8px;
  line-height: 1.04;
}
.brand-spectrum {
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, #49cfff 0%, #eefaff 100%);
  border-radius: 999px;
  margin-top: 8px;
  margin-bottom: 0;
  box-shadow: 0 0 12px rgba(110, 220, 255, 0.22);
}

/* HOME - Current Field becomes the hero */
.home-field-head {
  margin: 2px 0 12px 0;
}
.home-field-title {
  font-family: 'Outfit', 'Inter', sans-serif;
  color: #f3fbff;
  font-size: 32px;
  font-weight: 600;
  letter-spacing: -0.9px;
  line-height: 1.02;
  margin: 0;
}
.field-panel {
  margin: 0 0 18px 0;
}
.map-shell {
  background: #b8c8d4;
  border: 2px solid rgba(74, 97, 112, 0.95);
  border-radius: 22px;
  overflow: hidden;
  padding: 14px 14px 8px 14px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.22);
}
.map-shell [data-testid="stPlotlyChart"],
.map-shell .js-plotly-plot,
.map-shell .plot-container,
.map-shell .svg-container {
  border-radius: 16px !important;
  overflow: hidden !important;
}

/* Real Streamlit map card: wrapped in a rounded frame that matches the optimization button style. */
.st-key-temperature_map_card,
div[class*="st-key-temperature_map_card"] {
  background:
    linear-gradient(180deg, #0a2340 0%, #0d2d4d 100%) padding-box,
    linear-gradient(90deg, #5be0ff 0%, #2aa7ff 48%, #ff6278 100%) border-box !important;
  border: 2px solid transparent !important;
  border-radius: 28px !important;
  padding: 12px 12px 6px 12px !important;
  margin: 8px 0 22px 0 !important;
  overflow: hidden !important;
  box-shadow: 0 0 18px rgba(72, 202, 255, 0.18), 0 0 18px rgba(255, 98, 120, 0.10) !important;
}

.st-key-temperature_map_card [data-testid="stPlotlyChart"],
div[class*="st-key-temperature_map_card"] [data-testid="stPlotlyChart"] {
  border-radius: 22px !important;
  overflow: hidden !important;
  margin: 0 !important;
}

.st-key-temperature_map_card .js-plotly-plot,
.st-key-temperature_map_card .plot-container,
.st-key-temperature_map_card .svg-container,
div[class*="st-key-temperature_map_card"] .js-plotly-plot,
div[class*="st-key-temperature_map_card"] .plot-container,
div[class*="st-key-temperature_map_card"] .svg-container {
  border-radius: 20px !important;
  overflow: hidden !important;
}

/* Compact room summary shown under Current Field */
.avg-temp-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  background: rgba(11, 39, 63, 0.62);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 13px 16px;
  margin: 0 0 12px 0;
}
.avg-temp-label {
  color: var(--mist);
  font-size: 12px;
  font-weight: 700;
}
.avg-temp-value {
  color: #eefaff;
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
}

/* Simplified target-temperature control */
.target-input-wrap {
  margin: 0 0 18px 0;
}
.target-input-wrap [data-testid="stNumberInput"] {
  background: rgba(11, 39, 63, 0.62);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 9px 14px 8px 14px;
  margin-bottom: 0;
}
.target-input-wrap [data-testid="stNumberInput"] label,
.target-input-wrap [data-testid="stWidgetLabel"] p {
  color: var(--mist) !important;
  font-size: 12px !important;
  font-weight: 700 !important;
}
.target-input-wrap [data-testid="stNumberInput"] input {
  color: #eefaff !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  background: transparent !important;
}
.target-input-wrap button {
  color: #123b5d !important;
}

/* Target temperature is entered in navy on the light input surface. */
div[data-testid="stNumberInput"] input {
  color: #123b5d !important;
  -webkit-text-fill-color: #123b5d !important;
  opacity: 1 !important;
}
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInput"] button svg {
  color: #123b5d !important;
  fill: #123b5d !important;
}

/* Slightly soften the map frame corners. */
div[data-testid="stPlotlyChart"] {
  border-radius: 18px !important;
  overflow: hidden !important;
}


/* Dark target-temperature input, visually aligned with the average-temperature card */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
  background: rgba(11, 39, 63, 0.68) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}
div[data-testid="stNumberInput"] input {
  background: transparent !important;
  color: #123b5d !important;
  -webkit-text-fill-color: #123b5d !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  opacity: 1 !important;
}
div[data-testid="stNumberInput"] button {
  background: transparent !important;
  color: #123b5d !important;
  border: none !important;
}
div[data-testid="stNumberInput"] button svg {
  fill: #123b5d !important;
  color: #123b5d !important;
}
div[data-testid="stNumberInput"] button:hover {
  background: rgba(126, 215, 255, 0.08) !important;
}

/* Give the spatial map more visual weight */
div[data-testid="stPlotlyChart"] {
  margin-top: 2px !important;
  margin-bottom: 18px !important;
}

/* Kept for result cards and diagnostics */
.status-card {
  background: var(--navy-surface);
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
  color: #f2fbff;
  line-height: 1.1;
}
.status-target {
  color: #68cbff;
  font-size: 12.5px;
  font-weight: 600;
  margin-top: 4px;
}

.section-title {
  font-family: 'Outfit', 'Inter', sans-serif;
  font-size: 15px;
  font-weight: 800;
  color: #e8f7ff !important;
  margin-bottom: 7px;
}

.helper-desc {
  font-size: 11.5px;
  color: var(--mist) !important;
  line-height: 1.5;
  text-align: center;
  margin-top: 10px;
  margin-bottom: 14px;
  padding: 0 4px;
}

/* AI recommended HVAC setting — compact visual 2x2 panel */
.optimal-dispatch-box {
  background: linear-gradient(155deg, rgba(18, 59, 89, 0.96), rgba(12, 45, 72, 0.96));
  border: 1.5px solid rgba(74, 196, 244, 0.48);
  border-radius: 20px;
  padding: 16px;
  margin-top: 14px;
  margin-bottom: 14px;
  box-shadow: 0 9px 24px rgba(2, 20, 38, 0.22);
}

.optimal-dispatch-box h4 {
  color: #eaf8ff !important;
  font-family: 'Outfit', 'Inter', sans-serif;
  font-size: 17px;
  font-weight: 750;
  margin: 0 0 13px 2px;
}

.hvac-visual-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.hvac-mini-card {
  min-height: 132px;
  padding: 12px 11px 10px 11px;
  border-radius: 16px;
  background: rgba(7, 40, 67, 0.72);
  border: 1px solid rgba(121, 195, 232, 0.20);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
}

.hvac-mini-label {
  color: #9bcce5;
  font-size: 10.5px;
  font-weight: 750;
  letter-spacing: -0.01em;
  margin-bottom: 3px;
}

.hvac-mini-value {
  color: #f5fbff;
  font-size: 16px;
  line-height: 1.15;
  font-weight: 800;
  letter-spacing: -0.02em;
  white-space: nowrap;
}

/* Direction diagram */
.air-direction-wrap {
  height: 58px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  margin-top: 4px;
}
.ac-mini {
  width: 54px;
  height: 12px;
  border: 1.5px solid #91ddff;
  border-radius: 4px 4px 6px 6px;
  position: relative;
  background: rgba(112, 211, 255, 0.06);
  box-shadow: 0 0 10px rgba(64, 196, 255, 0.09);
}
.ac-mini::after {
  content: "";
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 2px;
  height: 2px;
  border-radius: 2px;
  background: #8bdcff;
  opacity: 0.75;
}
.air-rays {
  width: 126px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  margin-top: 6px;
}
.air-dir {
  width: 36px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  gap: 1px;
}
.air-dir-tag {
  font-size: 11px;
  line-height: 1.0;
  font-weight: 800;
  color: rgba(129, 215, 255, 0.32);
  letter-spacing: 0.02em;
}
.air-ray {
  width: 36px;
  height: 34px;
  text-align: center;
  color: rgba(129, 215, 255, 0.24);
  font-size: 31px;
  line-height: 32px;
  font-weight: 800;
  transition: .15s ease;
}
.air-dir.active .air-dir-tag {
  color: #dff7ff;
}
.air-dir.active .air-ray {
  color: #73dcff;
  text-shadow: 0 0 12px rgba(82, 209, 255, 0.38);
  transform: translateY(-1px) scale(1.06);
}

/* Flow strength: five vertical bars */
.flow-bars {
  height: 53px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 5px;
  padding-top: 7px;
}
.flow-bar {
  width: 10px;
  border-radius: 5px 5px 2px 2px;
  background: rgba(118, 206, 244, 0.14);
  border: 1px solid rgba(118, 206, 244, 0.12);
}
.flow-bar:nth-child(1){height:16px;}
.flow-bar:nth-child(2){height:23px;}
.flow-bar:nth-child(3){height:30px;}
.flow-bar:nth-child(4){height:37px;}
.flow-bar:nth-child(5){height:44px;}
.flow-bar.active {
  background: linear-gradient(180deg, #92e9ff 0%, #36c6f4 100%);
  border-color: rgba(164, 237, 255, 0.76);
  box-shadow: 0 0 8px rgba(69, 204, 246, 0.18);
}

/* Temperature / power track */
.hvac-track-wrap {
  margin-top: 15px;
}
.hvac-track {
  height: 8px;
  border-radius: 99px;
  position: relative;
  overflow: visible;
}
.temp-track {
  background: linear-gradient(90deg, #79ddff 0%, #45caf3 38%, #b8ecf7 72%, #ffd2a1 100%);
}
.power-track {
  background: rgba(83, 130, 158, 0.28);
}
.power-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: inherit;
  background: linear-gradient(90deg, #5ad7ff 0%, #7ee5e2 56%, #d9f5ff 100%);
}
.hvac-marker {
  position: absolute;
  top: 50%;
  width: 13px;
  height: 13px;
  margin-left: -6.5px;
  transform: translateY(-50%);
  border-radius: 50%;
  background: #ffffff;
  border: 3px solid #1b6c97;
  box-shadow: 0 0 0 2px rgba(106, 218, 255, 0.16), 0 0 9px rgba(255,255,255,.25);
}
.hvac-range {
  display: flex;
  justify-content: space-between;
  color: #6f9bb4;
  font-size: 8.5px;
  font-weight: 650;
  margin-top: 5px;
}

.hvac-card-note {
  color: #77a9c2;
  font-size: 8.5px;
  line-height: 1.25;
  margin-top: 5px;
}

/* Preserve 2 columns on phone width, but tighten typography further. */
@media (max-width: 420px) {
  .hvac-visual-grid { gap: 8px; }
  .hvac-mini-card { min-height: 124px; padding: 10px; }
  .hvac-mini-value { font-size: 14px; }
  .hvac-mini-label { font-size: 9.5px; }
}

/* Feasibility / target-achievement card */
.feasibility-box {
  border-radius: 16px;
  padding: 14px 16px;
  margin-top: 6px;
  margin-bottom: 14px;
  border: 1.5px solid;
  box-shadow: 0 6px 16px rgba(2, 20, 38, 0.12);
}
.feasibility-title {
  font-family: 'Outfit', 'Noto Sans KR', sans-serif;
  font-size: 15px;
  font-weight: 800;
}
.feasibility-desc {
  font-size: 12px;
  margin-top: 3px;
}
.results-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.results-title-icon {
  width: 18px;
  height: 18px;
  object-fit: contain;
  flex: 0 0 auto;
}
.results-title-glyph {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  color: #61cfff;
  font-size: 17px;
  line-height: 1;
  flex: 0 0 auto;
}
.field-map-title {
  font-family: 'Outfit', 'Inter', sans-serif;
  color: #e8f7ff !important;
  font-size: 15px;
  font-weight: 800;
  margin: 13px 0 8px 0;
}
.field-map-title.current-title {
  font-size: 24px;
  line-height: 1.05;
  letter-spacing: -0.6px;
  font-weight: 800;
  color: #f4fbff !important;
  margin-top: 15px;
  margin-bottom: 12px;
}

/* Current Field + Predicted Field grouped in one result card. */
.st-key-field_comparison_card,
div[class*="st-key-field_comparison_card"] {
  background: rgba(8, 31, 52, 0.48) !important;
  border: 1.5px solid rgba(89, 211, 255, 0.54) !important;
  border-radius: 22px !important;
  padding: 10px 12px 8px 12px !important;
  margin: 14px 0 18px 0 !important;
  box-shadow: 0 7px 20px rgba(2, 20, 38, 0.18) !important;
  overflow: hidden !important;
}

.st-key-field_comparison_card [data-testid="stPlotlyChart"],
div[class*="st-key-field_comparison_card"] [data-testid="stPlotlyChart"] {
  border-radius: 16px !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}
.st-key-field_comparison_card .js-plotly-plot,
div[class*="st-key-field_comparison_card"] .js-plotly-plot,
.st-key-current_field_card .js-plotly-plot,
div[class*="st-key-current_field_card"] .js-plotly-plot {
  width: 100% !important;
}

.field-map-divider {
  height: 1px;
  background: rgba(174, 228, 255, 0.14);
  margin: 8px 0 8px 0;
}


/* Metric Display Grids */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 8px;
}
.metric-cell {
  background: #173b59;
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
  color: #f1fbff;
  margin-top: 2px;
}

.metric-cell .metric-help {
  color: #88aec6;
  font-size: 9px;
  font-weight: 500;
  line-height: 1.25;
  margin-top: 2px;
  margin-bottom: 3px;
  text-transform: none;
}

/* Form Controls */
[data-testid="stSlider"],
[data-testid="stSelectSlider"],
[data-testid="stRadio"] {
  background: rgba(18, 51, 78, 0.70);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px 14px 6px 14px;
  margin-bottom: 8px;
}

/* Selectbox / expander surfaces */
[data-testid="stExpander"] {
  border-color: var(--line) !important;
  background: rgba(18, 51, 78, 0.55) !important;
  border-radius: 14px !important;
}

/* Bottom Nav Container */
.bottom-nav {
  margin-top: 16px;
  padding-top: 9px;
  border-top: 1px solid var(--line);
}

/* Buttons */
div.stButton > button[kind="primary"] {
  background:
    linear-gradient(180deg, #0a2340 0%, #0d2d4d 100%) padding-box,
    linear-gradient(90deg, #5be0ff 0%, #2aa7ff 48%, #ff6278 100%) border-box !important;
  color: #f6fcff !important;
  border-radius: 16px !important;
  font-family: 'Noto Sans KR', 'Outfit', 'Inter', sans-serif !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  padding: 13px 18px !important;
  border: 2px solid transparent !important;
  box-shadow: 0 0 18px rgba(72, 202, 255, 0.22), 0 0 20px rgba(255, 98, 120, 0.12) !important;
}

div.stButton > button[kind="primary"] p {
  color: #f6fcff !important;
}

div.stButton > button[kind="secondary"] p {
  color: #ffffff !important;
}

div.stButton > button[kind="secondary"] {
  background: #1a4666 !important;
  color: #e6f6ff !important;
  border-radius: 13px !important;
  font-family: 'Outfit', 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  padding: 10px 16px !important;
  border: 1px solid rgba(174, 228, 255, 0.15) !important;
  box-shadow: none !important;
}

div.stButton > button[kind="secondary"]:hover {
  background: #205474 !important;
  border-color: rgba(174, 228, 255, 0.27) !important;
}

/* Compact temperature summary shown above Cooling Influence Factors */
.factor-temp-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 2px 0 16px 0;
}
.factor-temp-card {
  background: rgba(11, 39, 63, 0.68);
  border: 1px solid rgba(174, 228, 255, 0.18);
  border-radius: 14px;
  padding: 11px 13px;
}
.factor-temp-label {
  color: #9fc3d9;
  font-size: 10.5px;
  font-weight: 650;
  margin-bottom: 3px;
}
.factor-temp-value {
  color: #f4fbff;
  font-family: 'JetBrains Mono', monospace;
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.4px;
}

/* Cooling-factor screen */
.cooling-factor-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 2px 0 6px 0;
  font-family: 'Outfit', 'Noto Sans KR', 'Inter', sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: #f3fbff;
}
.cooling-factor-title svg {
  width: 30px;
  height: 24px;
  flex: 0 0 auto;
  filter: drop-shadow(0 0 7px rgba(85, 210, 255, 0.16));
}
.cooling-factor-desc {
  color: #b8d5e6;
  font-size: 12px;
  font-weight: 650;
  line-height: 1.5;
  margin: 0 0 15px 0;
}
.cooling-factor-desc .step-emphasis {
  color: #f5fbff;
  font-weight: 850;
}

/* The keyed wrapper stays invisible so each factor has only ONE rounded card. */
.st-key-sl_ext,
.st-key-sl_serv,
.st-key-sl_meet,
.st-key-sl_work,
div[class*="st-key-sl_ext"],
div[class*="st-key-sl_serv"],
div[class*="st-key-sl_meet"],
div[class*="st-key-sl_work"] {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.st-key-sl_ext [data-testid="stSelectSlider"],
.st-key-sl_serv [data-testid="stSelectSlider"],
.st-key-sl_meet [data-testid="stSelectSlider"],
.st-key-sl_work [data-testid="stSelectSlider"],
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"] {
  background: rgba(18, 51, 78, 0.58) !important;
  border: 1px solid rgba(174, 228, 255, 0.16) !important;
  border-radius: 18px !important;
  padding: 11px 13px 8px 13px !important;
}

/* Factor name > selected value > endpoint labels, in that visual hierarchy. */
.st-key-sl_ext [data-testid="stWidgetLabel"] p,
.st-key-sl_serv [data-testid="stWidgetLabel"] p,
.st-key-sl_meet [data-testid="stWidgetLabel"] p,
.st-key-sl_work [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_ext"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_serv"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_meet"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_work"] [data-testid="stWidgetLabel"] p {
  font-size: 15px !important;
  font-weight: 500 !important;
}

/* Smaller selected value + endpoint labels for all four cooling-factor sliders. */
.st-key-sl_ext [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
.st-key-sl_serv [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
.st-key-sl_meet [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
.st-key-sl_work [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
.st-key-sl_ext [data-testid="stTickBarMin"],
.st-key-sl_ext [data-testid="stTickBarMax"],
.st-key-sl_serv [data-testid="stTickBarMin"],
.st-key-sl_serv [data-testid="stTickBarMax"],
.st-key-sl_meet [data-testid="stTickBarMin"],
.st-key-sl_meet [data-testid="stTickBarMax"],
.st-key-sl_work [data-testid="stTickBarMin"],
.st-key-sl_work [data-testid="stTickBarMax"] {
  font-size: 8px !important;
  line-height: 1 !important;
  font-weight: 400 !important;
  color: #9fbed1 !important;
}

.st-key-sl_ext [data-testid="stThumbValue"],
.st-key-sl_serv [data-testid="stThumbValue"],
.st-key-sl_meet [data-testid="stThumbValue"],
.st-key-sl_work [data-testid="stThumbValue"],
div[class*="st-key-sl_ext"] [data-testid="stThumbValue"],
div[class*="st-key-sl_serv"] [data-testid="stThumbValue"],
div[class*="st-key-sl_meet"] [data-testid="stThumbValue"],
div[class*="st-key-sl_work"] [data-testid="stThumbValue"],
.st-key-sl_ext [data-testid="stThumbValue"] *,
.st-key-sl_serv [data-testid="stThumbValue"] *,
.st-key-sl_meet [data-testid="stThumbValue"] *,
.st-key-sl_work [data-testid="stThumbValue"] * {
  font-size: 10px !important;
  line-height: 1 !important;
  font-weight: 500 !important;
}

/* Fallback for Streamlit versions that render slider labels without the test-id wrappers above. */
.st-key-sl_ext [data-testid="stSelectSlider"] p:not([data-testid="stWidgetLabel"] p),
.st-key-sl_serv [data-testid="stSelectSlider"] p:not([data-testid="stWidgetLabel"] p),
.st-key-sl_meet [data-testid="stSelectSlider"] p:not([data-testid="stWidgetLabel"] p),
.st-key-sl_work [data-testid="stSelectSlider"] p:not([data-testid="stWidgetLabel"] p) {
  font-size: 9px !important;
  line-height: 1.05 !important;
  font-weight: 400 !important;
}


/* v24: make selected stage + endpoint labels about the same size as the 5단계 helper text. */
.st-key-sl_ext [data-testid="stSelectSlider"] p,
.st-key-sl_ext [data-testid="stSelectSlider"] span,
.st-key-sl_serv [data-testid="stSelectSlider"] p,
.st-key-sl_serv [data-testid="stSelectSlider"] span,
.st-key-sl_meet [data-testid="stSelectSlider"] p,
.st-key-sl_meet [data-testid="stSelectSlider"] span,
.st-key-sl_work [data-testid="stSelectSlider"] p,
.st-key-sl_work [data-testid="stSelectSlider"] span,
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"] p,
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"] span,
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"] p,
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"] span,
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"] p,
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"] span,
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"] p,
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"] span {
  font-size: 12px !important;
  line-height: 1.1 !important;
  font-weight: 500 !important;
}

/* v25: Streamlit SelectSlider renders the visible value/endpoints inside BaseWeb divs.
   Target the BaseWeb slider itself so the text size is actually reduced. */
.st-key-sl_ext [data-baseweb="slider"] *,
.st-key-sl_serv [data-baseweb="slider"] *,
.st-key-sl_meet [data-baseweb="slider"] *,
.st-key-sl_work [data-baseweb="slider"] *,
div[class*="st-key-sl_ext"] [data-baseweb="slider"] *,
div[class*="st-key-sl_serv"] [data-baseweb="slider"] *,
div[class*="st-key-sl_meet"] [data-baseweb="slider"] *,
div[class*="st-key-sl_work"] [data-baseweb="slider"] * {
  font-size: 12px !important;
  line-height: 1.05 !important;
  font-weight: 500 !important;
}


/* v26 FINAL OVERRIDE:
   Force the SelectSlider's visible stage text (보통/높음/매우 낮음/매우 높음)
   to be small. The factor title is explicitly restored below. */
.st-key-sl_ext [data-testid="stSelectSlider"],
.st-key-sl_serv [data-testid="stSelectSlider"],
.st-key-sl_meet [data-testid="stSelectSlider"],
.st-key-sl_work [data-testid="stSelectSlider"],
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"],
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"],
.st-key-sl_ext [data-testid="stSelectSlider"] div,
.st-key-sl_serv [data-testid="stSelectSlider"] div,
.st-key-sl_meet [data-testid="stSelectSlider"] div,
.st-key-sl_work [data-testid="stSelectSlider"] div,
div[class*="st-key-sl_ext"] [data-testid="stSelectSlider"] div,
div[class*="st-key-sl_serv"] [data-testid="stSelectSlider"] div,
div[class*="st-key-sl_meet"] [data-testid="stSelectSlider"] div,
div[class*="st-key-sl_work"] [data-testid="stSelectSlider"] div,
.st-key-sl_ext [data-testid="stSelectSlider"] span,
.st-key-sl_serv [data-testid="stSelectSlider"] span,
.st-key-sl_meet [data-testid="stSelectSlider"] span,
.st-key-sl_work [data-testid="stSelectSlider"] span,
.st-key-sl_ext [data-testid="stSelectSlider"] p,
.st-key-sl_serv [data-testid="stSelectSlider"] p,
.st-key-sl_meet [data-testid="stSelectSlider"] p,
.st-key-sl_work [data-testid="stSelectSlider"] p {
  font-size: 9px !important;
  line-height: 1.0 !important;
  font-weight: 700 !important;
}

/* Restore only the four factor names so they remain visually dominant. */
.st-key-sl_ext [data-testid="stWidgetLabel"] p,
.st-key-sl_serv [data-testid="stWidgetLabel"] p,
.st-key-sl_meet [data-testid="stWidgetLabel"] p,
.st-key-sl_work [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_ext"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_serv"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_meet"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_work"] [data-testid="stWidgetLabel"] p {
  font-size: 15px !important;
  line-height: 1.25 !important;
  font-weight: 500 !important;
}

/* Explicit endpoint/value hooks when Streamlit exposes them. */
.st-key-sl_ext [data-testid="stThumbValue"],
.st-key-sl_serv [data-testid="stThumbValue"],
.st-key-sl_meet [data-testid="stThumbValue"],
.st-key-sl_work [data-testid="stThumbValue"],
.st-key-sl_ext [data-testid="stTickBar"],
.st-key-sl_serv [data-testid="stTickBar"],
.st-key-sl_meet [data-testid="stTickBar"],
.st-key-sl_work [data-testid="stTickBar"],
.st-key-sl_ext [data-testid="stTickBar"] *,
.st-key-sl_serv [data-testid="stTickBar"] *,
.st-key-sl_meet [data-testid="stTickBar"] *,
.st-key-sl_work [data-testid="stTickBar"] * {
  font-size: 9px !important;
  line-height: 1.0 !important;
  font-weight: 700 !important;
}


/* v27: hard override for the visible 5-step labels.
   Applies to 선택값(보통/높음/매우 높음) and endpoints(매우 낮음/매우 높음). */
[data-testid="stSelectSlider"] [data-baseweb="slider"],
[data-testid="stSelectSlider"] [data-baseweb="slider"] *,
[data-testid="stSelectSlider"] [data-testid="stThumbValue"],
[data-testid="stSelectSlider"] [data-testid="stThumbValue"] *,
[data-testid="stSelectSlider"] [data-testid="stTickBar"],
[data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
[data-testid="stSelectSlider"] [data-testid="stTickBarMin"],
[data-testid="stSelectSlider"] [data-testid="stTickBarMax"] {
  font-size: 9px !important;
  line-height: 1 !important;
  font-weight: 800 !important;
}

/* Keep factor names readable; they sit outside the BaseWeb slider itself. */
.st-key-sl_ext [data-testid="stWidgetLabel"] p,
.st-key-sl_serv [data-testid="stWidgetLabel"] p,
.st-key-sl_meet [data-testid="stWidgetLabel"] p,
.st-key-sl_work [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_ext"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_serv"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_meet"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-sl_work"] [data-testid="stWidgetLabel"] p {
  font-size: 15px !important;
  line-height: 1.25 !important;
  font-weight: 500 !important;
}

.cooling-load-card {
  background: linear-gradient(180deg, rgba(11, 38, 62, 0.94) 0%, rgba(13, 47, 75, 0.92) 100%);
  border: 1px solid rgba(174, 228, 255, 0.20);
  border-radius: 22px;
  padding: 15px 16px 14px 16px;
  margin: 17px 0 16px 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.cooling-load-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.cooling-load-label {
  font-family: 'Outfit', 'Noto Sans KR', sans-serif;
  font-size: 17.5px;
  font-weight: 800;
  color: #e9f8ff;
  letter-spacing: -0.3px;
}
.cooling-load-level {
  font-family: 'Outfit', 'Noto Sans KR', sans-serif;
  font-size: 20px;
  font-weight: 800;
  color: #f5fbff;
}
.cooling-load-segments {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 5px;
  margin-bottom: 2px;
}
.cooling-load-segment {
  height: 6px;
  border-radius: 999px;
  background: rgba(151, 190, 214, 0.18);
}
.cooling-load-segment.on-1 { background: #66d9ff; }
.cooling-load-segment.on-2 { background: #52c9ef; }
.cooling-load-segment.on-3 { background: #8edbcb; }
.cooling-load-segment.on-4 { background: #ffad66; }
.cooling-load-segment.on-5 { background: #ff6b7a; }

.major-factor-card {
  background: linear-gradient(180deg, rgba(12, 42, 67, 0.95) 0%, rgba(14, 50, 79, 0.92) 100%);
  border: 1px solid rgba(174, 228, 255, 0.20);
  border-radius: 18px;
  padding: 14px 16px;
  margin: -4px 0 16px 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.major-factor-title {
  font-family: 'Outfit', 'Noto Sans KR', sans-serif;
  font-size: 14px;
  font-weight: 800;
  color: #dff5ff;
  margin-bottom: 9px;
}
.major-factor-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.major-factor-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 6px 9px;
  border-radius: 999px;
  background: rgba(255, 173, 102, 0.11);
  border: 1px solid rgba(255, 173, 102, 0.32);
  color: #f4fbff;
  font-size: 10.5px;
  font-weight: 700;
  white-space: nowrap;
}
.major-factor-chip.very-high {
  background: rgba(255, 107, 122, 0.12);
  border-color: rgba(255, 107, 122, 0.40);
}

/* Factor-specific chip frames */
.major-factor-chip.factor-ext {
  background: rgba(255, 159, 67, 0.11) !important;
  border-color: rgba(255, 159, 67, 0.78) !important;
}
.major-factor-chip.factor-serv {
  background: rgba(66, 165, 245, 0.11) !important;
  border-color: rgba(66, 165, 245, 0.78) !important;
}
.major-factor-chip.factor-meet {
  background: rgba(156, 93, 202, 0.11) !important;
  border-color: rgba(156, 93, 202, 0.80) !important;
}
.major-factor-chip.factor-work {
  background: rgba(141, 110, 99, 0.13) !important;
  border-color: rgba(141, 110, 99, 0.86) !important;
}

.major-factor-empty {
  color: #9fbfd3;
  font-size: 11px;
  font-weight: 600;
}

/* Info/success boxes stay readable in the dark shell */
[data-testid="stAlert"] {
  border-radius: 13px !important;
}

/* ============================================================
   v28 ABSOLUTE FINAL SLIDER TYPOGRAPHY OVERRIDE
   All text INSIDE SelectSlider -> 8px bold.
   Only the factor-name label is restored to 15px.
   ============================================================ */
div[data-testid="stSelectSlider"] *,
div[data-testid="stSelectSlider"] *::before,
div[data-testid="stSelectSlider"] *::after {
  font-size: 9px !important;
  line-height: 1 !important;
  font-weight: 800 !important;
  letter-spacing: -0.15px !important;
}

/* Restore only the factor title: 외부 열환경 / 서버 발열 / 회의공간 / 업무공간 */
div[data-testid="stSelectSlider"] label[data-testid="stWidgetLabel"] *,
div[data-testid="stSelectSlider"] [data-testid="stWidgetLabel"] * {
  font-size: 15px !important;
  line-height: 1.25 !important;
  font-weight: 500 !important;
  letter-spacing: -0.25px !important;
}

/* Extra direct hooks for the visible selected value and both endpoint texts. */
div[data-testid="stSelectSlider"] [role="slider"] *,
div[data-testid="stSelectSlider"] [data-testid="stThumbValue"],
div[data-testid="stSelectSlider"] [data-testid="stThumbValue"] *,
div[data-testid="stSelectSlider"] [data-testid="stTickBar"],
div[data-testid="stSelectSlider"] [data-testid="stTickBar"] *,
div[data-testid="stSelectSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSelectSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSelectSlider"] [data-baseweb="slider"] *,
div[data-testid="stSelectSlider"] [aria-valuenow] * {
  font-size: 9px !important;
  line-height: 1 !important;
  font-weight: 800 !important;
}


/* ============================================================
   v31 cooling-factor slider typography
   Make only the stage/value texts tiny (9px); keep factor names readable.
   Current Streamlit may expose select_slider through stSlider rather than stSelectSlider.
   ============================================================ */
.st-key-sl_ext [data-testid="stSlider"] *,
.st-key-sl_serv [data-testid="stSlider"] *,
.st-key-sl_meet [data-testid="stSlider"] *,
.st-key-sl_work [data-testid="stSlider"] *,
div[class*="st-key-sl_ext"] [data-testid="stSlider"] *,
div[class*="st-key-sl_serv"] [data-testid="stSlider"] *,
div[class*="st-key-sl_meet"] [data-testid="stSlider"] *,
div[class*="st-key-sl_work"] [data-testid="stSlider"] *,
.st-key-sl_ext [data-testid="stSelectSlider"] *,
.st-key-sl_serv [data-testid="stSelectSlider"] *,
.st-key-sl_meet [data-testid="stSelectSlider"] *,
.st-key-sl_work [data-testid="stSelectSlider"] * {
  font-size: 9px !important;
  line-height: 1 !important;
  font-weight: 800 !important;
  letter-spacing: -0.15px !important;
}

/* Restore only the four factor names. */
.st-key-sl_ext [data-testid="stWidgetLabel"] *,
.st-key-sl_serv [data-testid="stWidgetLabel"] *,
.st-key-sl_meet [data-testid="stWidgetLabel"] *,
.st-key-sl_work [data-testid="stWidgetLabel"] *,
div[class*="st-key-sl_ext"] [data-testid="stWidgetLabel"] *,
div[class*="st-key-sl_serv"] [data-testid="stWidgetLabel"] *,
div[class*="st-key-sl_meet"] [data-testid="stWidgetLabel"] *,
div[class*="st-key-sl_work"] [data-testid="stWidgetLabel"] * {
  font-size: 15px !important;
  line-height: 1.25 !important;
  font-weight: 500 !important;
}

/* Explicit selected-value / endpoint hooks as a fallback. */
.st-key-sl_ext [data-testid="stThumbValue"],
.st-key-sl_serv [data-testid="stThumbValue"],
.st-key-sl_meet [data-testid="stThumbValue"],
.st-key-sl_work [data-testid="stThumbValue"],
.st-key-sl_ext [data-testid="stTickBar"] *,
.st-key-sl_serv [data-testid="stTickBar"] *,
.st-key-sl_meet [data-testid="stTickBar"] *,
.st-key-sl_work [data-testid="stTickBar"] *,
.st-key-sl_ext [data-baseweb="slider"] *,
.st-key-sl_serv [data-baseweb="slider"] *,
.st-key-sl_meet [data-baseweb="slider"] *,
.st-key-sl_work [data-baseweb="slider"] * {
  font-size: 9px !important;
  line-height: 1 !important;
  font-weight: 800 !important;
}


/* Center and simplify the optimization progress message */
div[data-testid="stSpinner"] {
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
}
div[data-testid="stSpinner"] > div {
  width: 100% !important;
  justify-content: center !important;
}
div[data-testid="stSpinner"] p {
  width: 100% !important;
  text-align: center !important;
  color: #eefaff !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  letter-spacing: -0.2px !important;
}


/* Field view selector: compact 3D / 2D toggle inside each Field section */
[data-testid="stSegmentedControl"] {
  width: fit-content !important;
  margin: 0 0 8px auto !important;
}
[data-testid="stSegmentedControl"] button {
  min-height: 30px !important;
  padding: 4px 13px !important;
  font-family: 'Outfit', 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 700 !important;
}
.st-key-home_field_view [data-testid="stRadio"],
.st-key-result_current_view [data-testid="stRadio"],
.st-key-result_predicted_view [data-testid="stRadio"],
div[class*="st-key-home_field_view"] [data-testid="stRadio"],
div[class*="st-key-result_current_view"] [data-testid="stRadio"],
div[class*="st-key-result_predicted_view"] [data-testid="stRadio"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 0 7px 0 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 2. CANONICAL SENSOR CONFIGURATION & REPO ASSETS
# ============================================================
# Canonical positions strictly aligned with the room heatmap:
# Horizontal (x_plot, Length): 0 to 9.0 m | Vertical (y_plot, Width): 0 to 4.0 m
ROA_NODES_META = {
    653:  {"code": "S1", "name": "Sensor 1 · Node 653",  "x_plot": 6.75, "y_plot": 2.75, "z": 1.50, "zone": "Core Sensor"},
    887:  {"code": "S2", "name": "Sensor 2 · Node 887",  "x_plot": 2.75, "y_plot": 2.75, "z": 1.50, "zone": "Core Sensor"},
    1036: {"code": "S3", "name": "Sensor 3 · Node 1036", "x_plot": 4.25, "y_plot": 1.75, "z": 2.50, "zone": "Core Sensor"},
    639:  {"code": "S4", "name": "Sensor 4 · Node 639",  "x_plot": 1.25, "y_plot": 1.25, "z": 2.00, "zone": "Core Sensor"},
    1229: {"code": "S5", "name": "Sensor 5 · Node 1229", "x_plot": 5.50, "y_plot": 1.75, "z": 2.00, "zone": "Core Sensor"},
}
ROA_NODE_IDS = list(ROA_NODES_META.keys())

# Validation-selected strict nested hierarchy:
# 5 ⊂ 6 ⊂ ... ⊂ 14 ⊂ 15
FINAL_NESTED_SENSOR_ORDER = (
    653, 887, 1036, 639, 1229,
    670, 323, 859, 1050, 551,
    739, 750, 4, 1255, 721,
)
MIN_ACTIVE_SENSORS = 5
MAX_ACTIVE_SENSORS = 15


APP_ROOT = Path(__file__).resolve().parent


def _first_existing_path(candidates):
    """Resolve deployment assets relative to the Streamlit app file, not process CWD."""
    for candidate in candidates:
        p = Path(candidate)
        if not p.is_absolute():
            p = APP_ROOT / p
        if p.exists() and p.is_file():
            return p
    return None


CASE_INFO_PATH = _first_existing_path([
    "Case Info 200 DesignPoints - 최종본.xlsx",
    "Case Info 200 DesignPoints - 최종본 (1).xlsx",
    "Case Info 200 DesignPoints.xlsx",
])

CHECKPOINT_PATH = _first_existing_path([
    "best_deploy.pt",
    "best.pt",
])


def _count_dp_csvs_in_zip(path: Path) -> int:
    """Return the number of distinct dpN.csv CFD cases in a valid archive."""
    if not path.exists() or not path.is_file() or not zipfile.is_zipfile(path):
        return 0
    try:
        ids = set()
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                m = re.search(r"dp\s*(\d+)\.csv$", Path(name).name, flags=re.IGNORECASE)
                if m:
                    ids.add(int(m.group(1)))
        return len(ids)
    except Exception:
        return 0


def _discover_cfd_zip():
    """Find the real CFD archive regardless of browser/GitHub filename changes.

    We first test common filenames, including the literal URL-encoded filename
    ``Field%20data.zip``.  Then every *.zip under the repository root is inspected.
    The valid archive containing the largest number of dpN.csv files wins.
    """
    preferred_names = [
        "Field data.zip",
        "field_data.zip",
        "Field%20data.zip",
        "Field data (1).zip",
        "Field data (1)(1).zip",
    ]

    candidates = []
    seen = set()

    def add_candidate(p: Path):
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    for name in preferred_names:
        p = APP_ROOT / name
        if p.exists() and p.is_file():
            add_candidate(p)

    # GitHub/Streamlit may preserve a renamed upload. Search all repo ZIPs instead
    # of requiring one exact spelling. Avoid .git internals.
    try:
        for p in APP_ROOT.rglob("*.zip"):
            if ".git" in p.parts:
                continue
            if p.is_file():
                add_candidate(p)
    except Exception:
        pass

    valid = []
    diagnostics = []
    for p in candidates:
        try:
            if not zipfile.is_zipfile(p):
                head = p.read_bytes()[:256]
                if b"git-lfs.github.com/spec/v1" in head:
                    diagnostics.append(f"{p.name}: Git LFS pointer")
                else:
                    diagnostics.append(f"{p.name}: not a valid ZIP")
                continue
            n_dp = _count_dp_csvs_in_zip(p)
            if n_dp <= 0:
                diagnostics.append(f"{p.name}: ZIP has no dpN.csv")
                continue
            valid.append((n_dp, p))
        except Exception as exc:
            diagnostics.append(f"{p.name}: {type(exc).__name__}")

    if valid:
        # Prefer the archive with the most real CFD cases; 200 should win naturally.
        valid.sort(key=lambda item: item[0], reverse=True)
        n_dp, p = valid[0]
        return p, None, int(n_dp)

    if not candidates:
        return None, f"no ZIP file found under {APP_ROOT.name}/", 0
    return None, "; ".join(diagnostics) if diagnostics else "no CFD dpN.csv archive found", 0


# Deferred until after the INTRO splash for faster cold start.
FIELD_ZIP_PATH, FIELD_ZIP_ERROR, FIELD_ZIP_DP_COUNT = None, None, 0


@st.cache_data(show_spinner=False)
def load_case_info():
    if CASE_INFO_PATH is None:
        return None

    # Use the exact parser from the training/deployment code whenever possible.
    if popfield_load_case_info is not None:
        try:
            return popfield_load_case_info(CASE_INFO_PATH)
        except Exception:
            pass

    # Robust local fallback: row 0 is the header, row 1 contains units,
    # and actual design points begin at row 2.
    try:
        raw = pd.read_excel(CASE_INFO_PATH, sheet_name=0, header=None)
        header = raw.iloc[0].astype(str).tolist()
        df = raw.iloc[2:].copy()
        df.columns = header
        df = df.dropna(how="all").reset_index(drop=True)
        if "Name" in df.columns:
            df["dp_id"] = (
                df["Name"].astype(str)
                .str.extract(r"(?i)DP\s*(\d+)", expand=False)
                .astype(float)
                .astype("Int64")
            )
        for c in COND_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_popfield_backend_cached(
    checkpoint_path_str: str,
    checkpoint_size: int,
    checkpoint_mtime_ns: int,
    checkpoint_sha256: str,
):
    """Load the heavy PopField backend for one exact checkpoint version.

    The checkpoint metadata/hash are cache-key arguments on purpose. If the PT file
    is replaced, Streamlit automatically creates a fresh resource instead of reusing
    a stale model. Exceptions are intentionally allowed to escape this cached helper,
    so a transient/partial checkpoint read is never stored as a cached failure result.
    """
    import torch

    checkpoint_path = Path(checkpoint_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, model, scalers, coords = popfield_load_checkpoint(checkpoint_path, device)
    coords = np.asarray(coords, dtype=np.float32)
    coords_norm_t = torch.from_numpy(
        scalers["coord"].transform(coords).astype(np.float32)
    ).to(device)
    return {
        "ok": True,
        "checkpoint": ckpt,
        "model": model,
        "scalers": scalers,
        "coords": coords,
        "coords_norm_t": coords_norm_t,
        "device": device,
        "checkpoint_path": str(checkpoint_path),

        # Keep the actual callables with the cached backend.
        # This avoids NoneType-callable errors after Streamlit reruns.
        "optimize_hvac_fn": popfield_optimize_hvac,
        "predict_conditions_fn": popfield_predict_conditions,
    }


def load_popfield_backend():
    # PyTorch + PopField architecture are imported only here, when needed.
    # This wrapper itself is intentionally NOT cached: only successful heavyweight
    # model loads are cached by _load_popfield_backend_cached().
    if not _lazy_import_popfield_modules():
        return {
            "ok": False,
            "error": f"demo_v3_hackathon_enhanced.py import failed: {POPFIELD_BACKEND_IMPORT_ERROR}",
        }

    if CHECKPOINT_PATH is None:
        return {
            "ok": False,
            "error": "best_deploy.pt (or best.pt) was not found in the repository root.",
        }

    try:
        import hashlib

        checkpoint_path = CHECKPOINT_PATH.resolve()
        stat = checkpoint_path.stat()
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        return _load_popfield_backend_cached(
            str(checkpoint_path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            checkpoint_sha256,
        )
    except Exception as exc:
        # Failure is returned to the UI, but it is NOT cached. A later rerun can retry
        # immediately after a deployment/file replacement without stale-error reuse.
        return {
            "ok": False,
            "error": f"Checkpoint/model load failed: {type(exc).__name__}: {exc}",
        }


@st.cache_resource(show_spinner=False)
def load_reconstruction_basis():
    p = APP_ROOT / "sensor_reconstruction_basis.npz"
    if p.exists():
        try:
            with np.load(p) as data:
                return {k: data[k] for k in data.files}
        except Exception:
            pass
    return None


def _nested_sensor_order(n_nodes: int):
    """Return the validated 5→15 sensor order from the deployment NPZ."""
    order = None

    if basis_assets is not None:
        try:
            raw = np.asarray(
                basis_assets.get("nested_sensor_order", []),
                dtype=np.int64,
            ).reshape(-1)
            if len(raw) >= MAX_ACTIVE_SENSORS:
                order = [int(x) for x in raw.tolist()]
        except Exception:
            order = None

    # Exact validated fallback in case the NPZ is temporarily unavailable.
    if order is None:
        order = list(FINAL_NESTED_SENSOR_ORDER)

    cleaned = []
    for nid in order:
        nid = int(nid)
        if 0 <= nid < int(n_nodes) and nid not in cleaned:
            cleaned.append(nid)
        if len(cleaned) >= MAX_ACTIVE_SENSORS:
            break

    return cleaned


def _active_sensor_count_from_temperature(reference_temp_c: float, target_temp_c: float) -> int:
    """
    5–15 active-sensor operating policy.

    One extra active sensor is enabled for each 0.5°C of absolute target
    deviation, with a hard minimum of 5 and a hard maximum of 15.

    Example for target 24°C:
      <24.5 → 5
       24.5 → 6
       25.0 → 7
       25.5 → 8
       26.0 → 9
       26.5 → 10
       27.0 → 11
       27.5 → 12
       28.0 → 13
       28.5 → 14
       29.0+ → 15
    """
    error_c = abs(float(reference_temp_c) - float(target_temp_c))
    extra = int(np.floor((error_c + 1e-9) / 0.5))
    return int(np.clip(
        MIN_ACTIVE_SENSORS + extra,
        MIN_ACTIVE_SENSORS,
        MAX_ACTIVE_SENSORS,
    ))


# Heavy/data assets are initialized only after the INTRO splash.
case_info_df = None
basis_assets = None


# ============================================================
# 3. SESSION STATE & NAVIGATION ROUTER
# ============================================================
VALID_VIEWS = ["INTRO", "HOME", "HEAT_LOAD", "RESULTS", "COMPARE"]

if "app_view" not in st.session_state or st.session_state.app_view not in VALID_VIEWS:
    st.session_state.app_view = "INTRO"

# INTRO 이미지의 실제 버튼 영역을 누르면 ?enter=1 로 들어옵니다.
# 이 값을 감지해 기존 HOME 화면으로 이동합니다.
if st.query_params.get("enter") == "1":
    st.session_state.app_view = "HOME"
    st.query_params.clear()

if "selected_dp" not in st.session_state:
    st.session_state.selected_dp = "DP 0"

if "z_plane" not in st.session_state:
    st.session_state.z_plane = 1.5
# HOME 화면에서는 측정 높이 선택을 사용하지 않고 1.5m로 고정합니다.
st.session_state.z_plane = 1.5

if "target_temp" not in st.session_state:
    st.session_state.target_temp = 24.0

# User-described CURRENT room temperature used to retrieve the closest real CFD case.
# Start the demo at 28.0 °C. The user can still edit it afterwards.
# This is deliberately separate from the HOME widget key so it survives navigation.
if "current_temp_query" not in st.session_state:
    st.session_state.current_temp_query = 28.0

# One-time v52 migration: when this new build is first loaded in an already-open
# Streamlit session, reset the HOME current-temperature field to the intended
# demo starting value. It will NOT reset again on ordinary reruns/navigation.
if "default_current_temp_v52_initialized" not in st.session_state:
    st.session_state.current_temp_query = 28.0
    st.session_state.home_current_temp_widget = 28.0
    st.session_state.default_current_temp_v52_initialized = True

# Optimization policy is intentionally fixed in the simplified UI.
st.session_state.policy = "Balanced (균형)"

if "heat_input_mode" not in st.session_state:
    st.session_state.heat_input_mode = "간편 단계"

for k, v in {"p_ext": "보통", "p_meet": "보통", "p_serv": "보통", "p_work": "보통"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# One-time migration: initialize the qualitative factors near the supplied DP 0 (Current)
# condition so the first HOME field starts from the official current CFD scenario.
if "cfd_retrieval_defaults_v1" not in st.session_state:
    st.session_state.p_ext = "매우 낮음"
    st.session_state.p_meet = "낮음"
    st.session_state.p_serv = "매우 낮음"
    st.session_state.p_work = "낮음"
    st.session_state.cfd_retrieval_defaults_v1 = True

if "has_run_optimization" not in st.session_state:
    st.session_state.has_run_optimization = False

# RESULTS 화면에서 AI 추천 제어안을 "적용해 본 결과"를 보여줄지 여부.
# 실제 BMS 전송이 아니라, 선택된 제어안을 PopField 예측 결과로 시뮬레이션합니다.
if "show_control_simulation" not in st.session_state:
    st.session_state.show_control_simulation = False

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

# Sensor-policy migration guard.
# Old browser sessions may still contain 20/30-sensor values from a previous build.
if "recommended_sensor_count" in st.session_state:
    try:
        st.session_state.recommended_sensor_count = int(np.clip(
            int(st.session_state.recommended_sensor_count),
            MIN_ACTIVE_SENSORS,
            MAX_ACTIVE_SENSORS,
        ))
    except Exception:
        st.session_state.recommended_sensor_count = MIN_ACTIVE_SENSORS

if isinstance(st.session_state.optimized_results, dict):
    for _sensor_key in ("initial_sensor_count", "recommended_sensor_count"):
        if _sensor_key in st.session_state.optimized_results:
            try:
                st.session_state.optimized_results[_sensor_key] = int(np.clip(
                    int(st.session_state.optimized_results[_sensor_key]),
                    MIN_ACTIVE_SENSORS,
                    MAX_ACTIVE_SENSORS,
                ))
            except Exception:
                st.session_state.optimized_results.pop(_sensor_key, None)

# ============================================================
# INTRO SCREEN ASSET (embedded in this single Python file)
# ============================================================
INTRO_IMAGE_WEBP_B64 = """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"""

if st.session_state.app_view == "INTRO":
    st.markdown(
        """
        <style>
        /* INTRO: restore a darker navy page and add a mobile-app shell. */
        .stApp {
            background: linear-gradient(180deg, #061427 0%, #071a2f 100%) !important;
        }

        .block-container {
            max-width: 472px !important;
            padding: 0.85rem 0.85rem 1.3rem 0.85rem !important;
            margin: 0.7rem auto 1.0rem auto !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        .coollins-intro-shell {
            width: 100%;
            max-width: 440px;
            margin: 0 auto;
            padding: 1.0rem 0.95rem 1.15rem 0.95rem;
            background: linear-gradient(180deg, #173a59 0%, #102c47 100%);
            border: 1.2px solid rgba(133, 202, 245, 0.22);
            border-radius: 36px;
            box-shadow: 0 24px 56px -20px rgba(0, 8, 20, 0.55);
            overflow: hidden;
        }

        .coollins-intro-shell .phone-notch {
            margin: 0 auto 14px auto;
        }

        .coollins-intro-target {
            position: relative;
            width: 100%;
            margin: 0 auto;
            line-height: 0;
            overflow: hidden;
            background: #020816;
            border-radius: 28px;
        }

        .coollins-intro-target img {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-drag: none;
            border-radius: 28px;
        }

        /* Transparent real click target placed exactly over the button in the artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 7.2%;
            top: 86.45%;
            width: 85.6%;
            height: 9.35%;
            display: block;
            border-radius: 22px;
            cursor: pointer;
            text-decoration: none !important;
            background: rgba(0,0,0,0.001);
            z-index: 10;
            outline: none;
            -webkit-tap-highlight-color: transparent;
        }

        .coollins-intro-enter:focus-visible {
            outline: 2px solid #38bdf8;
            outline-offset: -5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FAST INTRO GATE
# ============================================================
# Render the splash immediately and stop execution here.
# This prevents CFD ZIP scanning, 200-case indexing, Plotly field construction,
# and PopField/PyTorch loading before the user presses the intro button.
if st.session_state.app_view == "INTRO":
    intro_html = (
        f'<div class="coollins-intro-shell">'
        f'<div class="phone-notch"><div class="notch-cam"></div><div class="notch-speaker"></div></div>'
        f'<div class="coollins-intro-target">'
        f'<img src="data:image/webp;base64,{INTRO_IMAGE_WEBP_B64}" '
        f'alt="COOLLINS AI Smart Cooling Optimizer 소개 화면" />'
        f'<a class="coollins-intro-enter" href="?enter=1" target="_self" '
        f'aria-label="냉방 상태 확인하기" title="냉방 상태 확인하기"></a>'
        f'</div></div>'
    )
    st.markdown(intro_html, unsafe_allow_html=True)
    st.stop()


# From HOME onward, initialize data assets.
FIELD_ZIP_PATH, FIELD_ZIP_ERROR, FIELD_ZIP_DP_COUNT = _discover_cfd_zip()
case_info_df = load_case_info()
basis_assets = load_reconstruction_basis()

dp_options = (
    case_info_df["Name"].dropna().tolist()
    if (case_info_df is not None and "Name" in case_info_df.columns)
    else [f"DP {i}" for i in range(200)]
)


# ============================================================
# 4. REAL CURRENT FIELD + POPFIELD INFERENCE ENGINE
# ============================================================
STAGE_OPTS = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
LOAD_COL_MAP = {
    "external": "P83 - external",
    "meeting": "P84 - meeting",
    "server": "P85 - server",
    "working": "P86 - working",
}


def _resolve_field_column(columns, prefix):
    for c in columns:
        if str(c).strip().lower().startswith(prefix.lower()):
            return c
    raise KeyError(f"Field column starting with {prefix!r} was not found")


@st.cache_data(show_spinner=False)
def load_actual_cfd_case(zip_path_str: str, dp_id: int):
    """Load only one DP from Field data.zip instead of unpacking all 200 cases."""
    if not zip_path_str:
        return None
    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            target_name = None
            for name in zf.namelist():
                m = re.search(r"dp\s*(\d+)\.csv$", Path(name).name, flags=re.IGNORECASE)
                if m and int(m.group(1)) == int(dp_id):
                    target_name = name
                    break
            if target_name is None:
                return None

            raw = zf.read(target_name).decode("utf-8-sig", errors="replace")
            raw_lines = raw.splitlines()
            header_idx = next(
                i for i, line in enumerate(raw_lines)
                if line.strip().lower().startswith("node number")
            )
            df = pd.read_csv(
                io.StringIO("\n".join(raw_lines[header_idx:])),
                skipinitialspace=True,
            )
            df.columns = [str(c).strip() for c in df.columns]

            xcol = _resolve_field_column(df.columns, "X [")
            ycol = _resolve_field_column(df.columns, "Y [")
            zcol = _resolve_field_column(df.columns, "Z [")
            tcol = _resolve_field_column(df.columns, "Temperature")
            racol = _resolve_field_column(df.columns, "RA temp")
            ucol = _resolve_field_column(df.columns, "Velocity u")
            vcol = _resolve_field_column(df.columns, "Velocity v")
            wcol = _resolve_field_column(df.columns, "Velocity w")

            coords = df[[xcol, ycol, zcol]].to_numpy(np.float32)
            temp_c = df[tcol].to_numpy(np.float32) - 273.15
            velocity = df[[ucol, vcol, wcol]].to_numpy(np.float32)
            ra_c = float(np.nanmean(df[racol].to_numpy(np.float32)) - 273.15)

            return {
                "coords": coords,
                "temp_c": temp_c,
                "velocity": velocity,
                "ra_temp_c": ra_c,
                "mean_temp_c": float(np.nanmean(temp_c)),
                "source": f"Actual CFD · DP {int(dp_id)}",
            }
    except Exception:
        return None


def _selected_case_row(df, selected_dp_id: int):
    if df is None or len(df) == 0:
        return None
    if "dp_id" in df.columns:
        hit = df[pd.to_numeric(df["dp_id"], errors="coerce") == int(selected_dp_id)]
        if len(hit):
            return hit.iloc[0]
    if "Name" in df.columns:
        nums = df["Name"].astype(str).str.extract(r"(?i)DP\s*(\d+)", expand=False)
        hit = df[pd.to_numeric(nums, errors="coerce") == int(selected_dp_id)]
        if len(hit):
            return hit.iloc[0]
    return df.iloc[0]


def _predict_case_field_with_popfield(selected_dp_id: int):
    backend = load_popfield_backend()
    if not backend.get("ok", False):
        return None

    row = _selected_case_row(case_info_df, selected_dp_id)
    if row is None or any(c not in row.index for c in COND_COLS):
        return None

    try:
        cond = np.asarray([[float(row[c]) for c in COND_COLS]], dtype=np.float32)
        predict_fn = backend.get("predict_conditions_fn")
        if not callable(predict_fn):
            # Defensive recovery for an old/stale cached backend.
            if not _lazy_import_popfield_modules():
                return None
            predict_fn = popfield_predict_conditions

        if not callable(predict_fn):
            return None

        pred_field, pred_ra = predict_fn(
            backend["model"],
            cond,
            backend["scalers"]["cond"],
            backend["coords_norm_t"],
            backend["scalers"]["field"],
            backend["scalers"]["ra"],
            backend["device"],
        )
        return {
            "coords": backend["coords"],
            "temp_c": np.asarray(pred_field[0, :, 0], dtype=np.float32),
            "velocity": np.asarray(pred_field[0, :, 1:4], dtype=np.float32),
            "ra_temp_c": float(pred_ra[0]),
            "mean_temp_c": float(np.mean(pred_field[0, :, 0])),
            "source": f"PopField estimate · DP {int(selected_dp_id)}",
        }
    except Exception:
        return None


def _five_stage_query_map(df: pd.DataFrame, col: str):
    """Map five qualitative UI stages across the observed CFD load range.

    These values are used ONLY to retrieve the closest real CFD scenario. The
    eventual PopField optimization uses the matched scenario's ACTUAL heat loads,
    so we never pretend that an interpolated qualitative value is measured CFD.
    """
    values = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    if len(values) == 0:
        raise ValueError(f"No observed values available for {col}")
    lo, hi = float(values.min()), float(values.max())
    stage_values = np.linspace(lo, hi, len(STAGE_OPTS))
    return dict(zip(STAGE_OPTS, [float(v) for v in stage_values]))


def _requested_heat_loads_from_ui():
    if case_info_df is None:
        raise RuntimeError("Case Info Excel could not be loaded.")

    maps = {
        key: _five_stage_query_map(case_info_df, col)
        for key, col in LOAD_COL_MAP.items()
    }
    return {
        "external": maps["external"][st.session_state.get("p_ext", "보통")],
        "meeting": maps["meeting"][st.session_state.get("p_meet", "보통")],
        "server": maps["server"][st.session_state.get("p_serv", "보통")],
        "working": maps["working"][st.session_state.get("p_work", "보통")],
    }, maps


@st.cache_data(show_spinner=False)
def load_cfd_temperature_index(zip_path_str: str, file_mtime_ns: int, file_size: int):
    """
    Load per-DP temperature statistics.

    FAST PATH:
      1) cfd_temperature_index.csv next to streamlit_app.py
      2) cfd_index.csv next to streamlit_app.py
      3) a runtime /tmp sidecar generated by a previous session

    FALLBACK:
      Scan the CFD ZIP once, then write the small sidecar index so later sessions
      in the same deployment/container do not need to parse all ~200 CSVs again.
    """
    del file_mtime_ns, file_size  # only used to invalidate Streamlit cache

    required_cols = ["dp_id", "mean_temp_c", "p95_temp_c", "min_temp_c", "max_temp_c"]
    runtime_index = Path(tempfile.gettempdir()) / "coollins_cfd_temperature_index.csv"
    sidecar_candidates = [
        APP_ROOT / "cfd_temperature_index.csv",
        APP_ROOT / "cfd_index.csv",
        runtime_index,
    ]

    for idx_path in sidecar_candidates:
        if not idx_path.exists():
            continue
        try:
            cached = pd.read_csv(idx_path)
            if all(c in cached.columns for c in required_cols) and len(cached):
                cached = cached[required_cols].copy()
                cached["dp_id"] = pd.to_numeric(cached["dp_id"], errors="coerce")
                cached = cached.dropna(subset=["dp_id"])
                cached["dp_id"] = cached["dp_id"].astype(int)
                return cached.sort_values("dp_id").reset_index(drop=True)
        except Exception:
            pass

    zip_path = Path(zip_path_str)
    rows = []
    if not zip_path.exists():
        return pd.DataFrame(columns=required_cols)

    try:
        zf_ctx = zipfile.ZipFile(zip_path, "r")
    except (zipfile.BadZipFile, OSError):
        return pd.DataFrame(columns=required_cols)

    with zf_ctx as zf:
        for name in zf.namelist():
            m = re.search(r"dp\s*(\d+)\.csv$", Path(name).name, flags=re.IGNORECASE)
            if not m:
                continue
            dp_case = int(m.group(1))
            try:
                raw = zf.read(name).decode("utf-8-sig", errors="replace")
                raw_lines = raw.splitlines()
                header_idx = next(
                    i for i, line in enumerate(raw_lines)
                    if line.strip().lower().startswith("node number")
                )
                df = pd.read_csv(
                    io.StringIO("\n".join(raw_lines[header_idx:])),
                    skipinitialspace=True,
                )
                df.columns = [str(c).strip() for c in df.columns]
                tcol = _resolve_field_column(df.columns, "Temperature")
                temp_c = pd.to_numeric(df[tcol], errors="coerce").to_numpy(np.float64) - 273.15
                temp_c = temp_c[np.isfinite(temp_c)]
                if len(temp_c) == 0:
                    continue
                rows.append({
                    "dp_id": dp_case,
                    "mean_temp_c": float(np.mean(temp_c)),
                    "p95_temp_c": float(np.percentile(temp_c, 95)),
                    "min_temp_c": float(np.min(temp_c)),
                    "max_temp_c": float(np.max(temp_c)),
                })
            except Exception:
                continue

    result = pd.DataFrame(rows, columns=required_cols).sort_values("dp_id").reset_index(drop=True)

    # Runtime sidecar: helps every later rerun/session in the same container.
    if len(result):
        try:
            result.to_csv(runtime_index, index=False)
        except Exception:
            pass

        # If the deployment filesystem is writable, also create a repo-side sidecar.
        # Committing this CSV to GitHub gives the fastest possible cold HOME startup.
        try:
            repo_index = APP_ROOT / "cfd_temperature_index.csv"
            if not repo_index.exists():
                result.to_csv(repo_index, index=False)
        except Exception:
            pass

    return result


def _build_cfd_scenario_table():
    if FIELD_ZIP_PATH is None or case_info_df is None:
        return None
    stat = FIELD_ZIP_PATH.stat()
    idx = load_cfd_temperature_index(str(FIELD_ZIP_PATH), int(stat.st_mtime_ns), int(stat.st_size))
    if idx is None or len(idx) == 0:
        return None

    cases = case_info_df.copy()
    if "dp_id" not in cases.columns:
        if "Name" not in cases.columns:
            return None
        cases["dp_id"] = pd.to_numeric(
            cases["Name"].astype(str).str.extract(r"(?i)DP\s*(\d+)", expand=False),
            errors="coerce",
        )
    cases["dp_id"] = pd.to_numeric(cases["dp_id"], errors="coerce")
    for col in LOAD_COL_MAP.values():
        if col in cases.columns:
            cases[col] = pd.to_numeric(cases[col], errors="coerce")
    return cases.merge(idx, on="dp_id", how="inner")


def _find_nearest_cfd_scenario(query_temp_c: float, query_loads: dict):
    """Retrieve the real CFD case closest to temperature + four heat-load descriptors.

    Distance is normalized by the observed spread of each variable. Temperature
    receives total weight 4, roughly balancing the four heat-load dimensions.
    """
    table = _build_cfd_scenario_table()
    if table is None or len(table) == 0:
        return None

    required = ["mean_temp_c", *LOAD_COL_MAP.values()]
    valid = table.dropna(subset=required).copy()
    if len(valid) == 0:
        return None

    temp_std = max(float(valid["mean_temp_c"].std(ddof=0)), 1.0)
    score_sq = 4.0 * ((valid["mean_temp_c"].astype(float) - float(query_temp_c)) / temp_std) ** 2

    query_by_col = {
        LOAD_COL_MAP["external"]: float(query_loads["external"]),
        LOAD_COL_MAP["meeting"]: float(query_loads["meeting"]),
        LOAD_COL_MAP["server"]: float(query_loads["server"]),
        LOAD_COL_MAP["working"]: float(query_loads["working"]),
    }
    for col, q in query_by_col.items():
        scale = max(float(valid[col].astype(float).std(ddof=0)), 1.0)
        score_sq = score_sq + ((valid[col].astype(float) - q) / scale) ** 2

    valid["retrieval_score"] = np.sqrt(score_sq)
    best = valid.sort_values(["retrieval_score", "dp_id"]).iloc[0]
    matched_loads = {
        key: float(best[col]) for key, col in LOAD_COL_MAP.items()
    }
    return {
        "dp_id": int(best["dp_id"]),
        "name": str(best.get("Name", f"DP {int(best['dp_id'])}")),
        "mean_temp_c": float(best["mean_temp_c"]),
        "retrieval_score": float(best["retrieval_score"]),
        "temperature_gap_c": float(best["mean_temp_c"] - float(query_temp_c)),
        "query_loads": {k: float(v) for k, v in query_loads.items()},
        "matched_loads": matched_loads,
        "row": best,
    }


def _sync_current_temp_from_home_widget():
    st.session_state.current_temp_query = float(st.session_state.home_current_temp_widget)


def _temperature_plane_grid(coords, temp_c, z_plane, x_axis, y_axis):
    coords = np.asarray(coords, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    gx, gy = np.meshgrid(x_axis, y_axis)

    z_values = np.unique(coords[:, 2])
    z_use = float(z_values[np.argmin(np.abs(z_values - float(z_plane)))])
    mask = np.isclose(coords[:, 2], z_use, atol=1e-6)
    pts = coords[mask, :2]
    vals = temp_c[mask]

    if len(vals) < 3:
        return np.full(gx.shape, float(np.nanmean(temp_c)), dtype=float)

    grid = griddata(pts, vals, (gx, gy), method="linear")
    if np.isnan(grid).any():
        nearest = griddata(pts, vals, (gx, gy), method="nearest")
        grid = np.where(np.isnan(grid), nearest, grid)
    return np.asarray(grid, dtype=float)


def _direction_label(rec):
    active = []
    if int(round(float(rec["Inlet_L"]))) == 1:
        active.append(("Left", "L"))
    if int(round(float(rec["Inlet_M"]))) == 1:
        active.append(("Middle", "M"))
    if int(round(float(rec["Inlet_R"]))) == 1:
        active.append(("Right", "R"))
    if len(active) == 1:
        return f"{active[0][0]} ({active[0][1]})"
    if not active:
        return "None"
    return " / ".join(code for _, code in active)


def _demo_status_from_row(rec, target_temp):
    p95_limit = float(target_temp) + 2.0
    strict = (
        float(rec["zone_range_C"]) <= 2.0
        and float(rec["hot_fraction"]) <= 0.05
        and float(rec["cold_fraction"]) <= 0.05
        and float(rec["p95_temp_C"]) <= p95_limit
    )
    if strict:
        return "FEASIBLE"

    # Same near-feasible margins used by the deployment demo:
    # +0.25 C zone range, +1 percentage point hot/cold, +0.25 C P95.
    near = (
        float(rec["zone_range_C"]) <= 2.25
        and float(rec["hot_fraction"]) <= 0.06
        and float(rec["cold_fraction"]) <= 0.06
        and float(rec["p95_temp_C"]) <= p95_limit + 0.25
    )
    return "NEAR_FEASIBLE" if near else "INFEASIBLE"


# ------------------------------------------------------------
# Retrieve CURRENT FIELD from the 200 real CFD scenarios.
# Query = user current mean temperature + four qualitative heat-load settings.
# ------------------------------------------------------------
query_loads_for_current, _stage_maps_for_current = _requested_heat_loads_from_ui()
scenario_table = _build_cfd_scenario_table()

# Never crash the whole app because a deployment asset is malformed.  Surface the
# problem clearly and let the rest of the UI load so the file can be replaced.
if FIELD_ZIP_PATH is None and FIELD_ZIP_ERROR:
    st.warning(
        "CFD 데이터 ZIP을 읽을 수 없습니다. GitHub의 실제 ZIP 파일을 다시 업로드해 주세요. "
        f"({FIELD_ZIP_ERROR})"
    )

# Safety fallback only. Normal demo startup is fixed at 28.0 °C above.
if st.session_state.current_temp_query is None:
    st.session_state.current_temp_query = 28.0

matched_scenario = _find_nearest_cfd_scenario(
    float(st.session_state.current_temp_query),
    query_loads_for_current,
)

current_field = None
if matched_scenario is not None and FIELD_ZIP_PATH is not None:
    current_field = load_actual_cfd_case(str(FIELD_ZIP_PATH), int(matched_scenario["dp_id"]))
    if current_field is not None:
        current_field["source"] = f"Actual CFD · DP {int(matched_scenario['dp_id'])} (nearest scenario)"

# Deployment can still boot without the raw archive, but this is explicitly a fallback.
# The real nearest-scenario workflow requires Field data.zip in the repository.
if current_field is None:
    fallback_dp = int(matched_scenario["dp_id"]) if matched_scenario is not None else 0
    current_field = _predict_case_field_with_popfield(fallback_dp)

if current_field is None:
    fallback_x = np.linspace(0.25, 8.75, 45)
    fallback_y = np.linspace(0.25, 3.75, 25)
    fallback_mx, fallback_my = np.meshgrid(fallback_x, fallback_y)
    fallback_temp = np.full_like(fallback_mx, float(st.session_state.current_temp_query), dtype=float)
    fallback_coords = np.stack(
        [fallback_mx.ravel(), fallback_my.ravel(), np.full(fallback_mx.size, 1.5)],
        axis=-1,
    )
    current_field = {
        "coords": fallback_coords,
        "temp_c": fallback_temp.ravel(),
        "velocity": np.zeros((fallback_mx.size, 3), dtype=np.float32),
        "ra_temp_c": float(st.session_state.current_temp_query),
        "mean_temp_c": float(st.session_state.current_temp_query),
        "source": "Fallback field (required deployment assets missing)",
    }

current_coords = np.asarray(current_field["coords"], dtype=np.float32)
current_temp_nodes = np.asarray(current_field["temp_c"], dtype=np.float32)
avg_room_temp = float(current_field["mean_temp_c"])
current_field_source = str(current_field["source"])
matched_dp_id = int(matched_scenario["dp_id"]) if matched_scenario is not None else 0
matched_mean_temp_c = float(matched_scenario["mean_temp_c"]) if matched_scenario is not None else avg_room_temp
matched_actual_loads = (
    dict(matched_scenario["matched_loads"])
    if matched_scenario is not None
    else dict(query_loads_for_current)
)

# Use the actual/model coordinate envelope rather than a hand-crafted crop.
x_min, x_max = float(np.min(current_coords[:, 0])), float(np.max(current_coords[:, 0]))
y_min, y_max = float(np.min(current_coords[:, 1])), float(np.max(current_coords[:, 1]))
grid_len_axis = np.linspace(x_min, x_max, 48)
grid_wid_axis = np.linspace(y_min, y_max, 44)
mesh_len, mesh_wid = np.meshgrid(grid_len_axis, grid_wid_axis)

field_current_grid = _temperature_plane_grid(
    current_coords,
    current_temp_nodes,
    st.session_state.z_plane,
    grid_len_axis,
    grid_wid_axis,
)

# Sensor values come from the retrieved real CFD field by NODE ID.
sensor_plot_meta = {}
sensor_readings = {}
for nid, meta in ROA_NODES_META.items():
    m = dict(meta)
    if 0 <= int(nid) < len(current_coords):
        m["x_plot"] = float(current_coords[int(nid), 0])
        m["y_plot"] = float(current_coords[int(nid), 1])
        m["z"] = float(current_coords[int(nid), 2])
        sensor_readings[nid] = float(current_temp_nodes[int(nid)])
    else:
        dist = (
            (current_coords[:, 0] - float(meta["x_plot"])) ** 2
            + (current_coords[:, 1] - float(meta["y_plot"])) ** 2
            + (current_coords[:, 2] - float(meta["z"])) ** 2
        )
        idx = int(np.argmin(dist))
        sensor_readings[nid] = float(current_temp_nodes[idx])
    sensor_plot_meta[nid] = m



def field_view_selector(key: str) -> str:
    """Small 3D / 2D selector. Uses segmented control when available."""
    if key not in st.session_state:
        st.session_state[key] = "3D"

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Field view",
            options=["3D", "2D"],
            selection_mode="single",
            key=key,
            label_visibility="collapsed",
        )
        return selected or "3D"

    return st.radio(
        "Field view",
        options=["3D", "2D"],
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )



def _select_adaptive_sensor_points(coords_xyz, temp_nodes, sensor_count):
    """
    Return the exact validated nested active sensor set.

    The active set is always the first K nodes of:
    653, 887, 1036, 639, 1229, 670, 323, 859, 1050, 551,
    739, 750, 4, 1255, 721.
    """
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    temp_nodes = np.asarray(temp_nodes, dtype=float).reshape(-1)

    valid = (
        coords_xyz.ndim == 2
        and coords_xyz.shape[1] >= 3
        and len(coords_xyz) == len(temp_nodes)
    )
    if not valid:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
            [],
        )

    sensor_count = int(np.clip(
        int(sensor_count),
        MIN_ACTIVE_SENSORS,
        MAX_ACTIVE_SENSORS,
    ))

    finite = np.isfinite(coords_xyz[:, :3]).all(axis=1) & np.isfinite(temp_nodes)
    order = _nested_sensor_order(len(coords_xyz))
    selected_nodes = [
        int(nid)
        for nid in order
        if 0 <= int(nid) < len(coords_xyz) and bool(finite[int(nid)])
    ][:sensor_count]

    if not selected_nodes:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
            [],
        )

    idx = np.asarray(selected_nodes, dtype=np.int64)
    selected_xyz = coords_xyz[idx, :3]
    selected_temp = temp_nodes[idx]
    selected_names = [
        f"S{i + 1} · Node {int(nid)}"
        for i, nid in enumerate(selected_nodes)
    ]

    return selected_xyz, selected_temp, selected_names



def make_2d_heatmap(grid_data, height=315, show_sensors=True, sensor_count=5, coords_xyz=None, temp_nodes=None):
    """Classic top-down 2D temperature map used when the user selects 2D."""
    heatmap_data = np.asarray(grid_data, dtype=float)

    temp_scale = [
        [0.00, "#8ee7ff"],
        [0.18, "#50c9ff"],
        [0.36, "#17bed0"],
        [0.54, "#4edb78"],
        [0.70, "#b9e63d"],
        [0.84, "#ffa13a"],
        [1.00, "#e63a32"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=grid_len_axis,
            y=grid_wid_axis,
            colorscale=temp_scale,
            hoverongaps=False,
            zmin=18.0,
            zmax=28.0,
            colorbar=dict(
                title=dict(text="°C", font=dict(size=10, color="#d9f3ff")),
                thickness=5,
                len=0.68,
                x=0.99,
                tickvals=[18, 20, 22, 24, 26, 28],
                tickfont=dict(size=8, color="#d9f3ff"),
                outlinecolor="rgba(174,228,255,0.18)",
            ),
            hovertemplate=(
                "X: %{x:.2f} m<br>"
                "Y: %{y:.2f} m<br>"
                "온도: %{z:.2f} °C"
                "<extra></extra>"
            ),
        )
    )

    if show_sensors:
        if coords_xyz is not None and temp_nodes is not None:
            selected_xyz, selected_temp, selected_names = _select_adaptive_sensor_points(
                coords_xyz,
                temp_nodes,
                sensor_count,
            )
            sx = selected_xyz[:, 0].tolist() if len(selected_xyz) else []
            sy = selected_xyz[:, 1].tolist() if len(selected_xyz) else []
            sensor_hover = [
                (
                    f"<b>{name}</b><br>"
                    f"X={xyz[0]:.2f}m, Y={xyz[1]:.2f}m, Z={xyz[2]:.2f}m<br>"
                    f"온도={temp:.2f}°C"
                )
                for xyz, temp, name in zip(selected_xyz, selected_temp, selected_names)
            ]
        else:
            sx = [meta["x_plot"] for meta in sensor_plot_meta.values()]
            sy = [meta["y_plot"] for meta in sensor_plot_meta.values()]
            sensor_hover = []
            for nid, meta in sensor_plot_meta.items():
                ix = int(np.argmin(np.abs(grid_len_axis - float(meta["x_plot"]))))
                iy = int(np.argmin(np.abs(grid_wid_axis - float(meta["y_plot"]))))
                sampled = float(heatmap_data[iy, ix])
                sensor_hover.append(
                    f"<b>{meta['name']}</b><br>"
                    f"X={meta['x_plot']:.2f}m, Y={meta['y_plot']:.2f}m<br>"
                    f"온도={sampled:.2f}°C"
                )

        n_sensor_vis = max(1, len(sx))
        marker_size = 9 if n_sensor_vis <= 5 else 7 if n_sensor_vis <= 10 else 5 if n_sensor_vis <= 20 else 4

        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="#ffffff",
                    line=dict(color="#65ddff", width=1.1),
                    opacity=0.98,
                ),
                hovertext=sensor_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=4, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            range=[float(grid_len_axis.min()), float(grid_len_axis.max())],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[float(grid_wid_axis.min()), float(grid_wid_axis.max())],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig

def make_mobile_heatmap(grid_data, height=340, show_sensors=True, sensor_count=5):
    """Interactive 3D spatial-temperature surface used by HOME and RESULTS."""
    surface_data = np.asarray(grid_data, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            z=surface_data,
            x=grid_len_axis,
            y=grid_wid_axis,
            surfacecolor=surface_data,
            colorscale=[
                    [0.00, "#8ee7ff"],
                    [0.18, "#5cc8ff"],
                    [0.38, "#43d8b1"],
                    [0.58, "#b7ef4a"],
                    [0.78, "#ffb347"],
                    [1.00, "#e53935"],
                ],
            cmin=18.0,
            cmax=28.0,
            showscale=True,
            colorbar=dict(
                title=dict(text="°C", font=dict(size=10, color="#d9f3ff")),
                thickness=8,
                len=0.72,
                x=0.965,
                tickfont=dict(size=9, color="#d9f3ff"),
                outlinecolor="rgba(174,228,255,0.18)",
            ),
            hovertemplate=(
                "X: %{x:.2f} m<br>"
                "Y: %{y:.2f} m<br>"
                "온도: %{z:.2f} °C"
                "<extra></extra>"
            ),
            lighting=dict(
                ambient=0.78,
                diffuse=0.72,
                specular=0.10,
                roughness=0.92,
            ),
        )
    )

    if show_sensors:
        sx_plot = [meta["x_plot"] for meta in sensor_plot_meta.values()]
        sy_plot = [meta["y_plot"] for meta in sensor_plot_meta.values()]
        sz_plot = [sensor_readings.get(nid, np.nan) + 0.15 for nid in sensor_plot_meta.keys()]
        hover_texts = [
            (
                f"<b>{meta['name']}</b><br>"
                f"Zone: {meta['zone']}<br>"
                f"Coords: (L={meta['x_plot']:.2f}, W={meta['y_plot']:.2f})m<br>"
                f"Live: {sensor_readings.get(nid, 0.0):.2f}°C"
            )
            for nid, meta in sensor_plot_meta.items()
        ]

        fig.add_trace(
            go.Scatter3d(
                x=sx_plot,
                y=sy_plot,
                z=[z + 0.12 for z in sz_plot],
                mode="markers",
                marker=dict(
                    size=5.5,
                    color="#ffffff",
                    line=dict(color="#65ddff", width=1.0),
                    symbol="circle",
                    opacity=1.0,
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text="", font=dict(size=1)),
        showlegend=False,
        autosize=True,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            domain=dict(x=[0.00, 0.955], y=[0.00, 1.00]),
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="",
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                title="",
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            zaxis=dict(
                title="",
                range=[18.0, 28.0],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.15, y=1.25, z=0.34),
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=0.0, y=-2.15, z=0.78),
                center=dict(x=0.0, y=0.0, z=-0.08),
            ),
        ),
    )

    return fig


def make_true_3d_field(coords_xyz, temp_nodes, height=390, max_points=2800, show_sensors=True, sensor_count=5):
    """
    Clean 3D room-style temperature map.

    Instead of plotting the raw CFD node layout directly, interpolate the real CFD
    temperatures onto a regular XYZ lattice. This preserves the field pattern while
    producing the clean rectangular 3D map used in the UI reference.
    """
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    temp_nodes = np.asarray(temp_nodes, dtype=float).reshape(-1)

    valid = (
        coords_xyz.ndim == 2
        and coords_xyz.shape[1] >= 3
        and len(coords_xyz) == len(temp_nodes)
    )
    if not valid:
        return make_mobile_heatmap(field_current_grid, height=height, show_sensors=show_sensors, sensor_count=sensor_count)

    finite = np.isfinite(coords_xyz[:, :3]).all(axis=1) & np.isfinite(temp_nodes)
    coords = coords_xyz[finite, :3]
    temps = temp_nodes[finite]

    if len(coords) == 0:
        return make_mobile_heatmap(field_current_grid, height=height, show_sensors=show_sensors, sensor_count=sensor_count)

    xmin, ymin, zmin = np.min(coords, axis=0)
    xmax, ymax, zmax = np.max(coords, axis=0)

    # Regular 3D lattice for the clean "room volume" look.
    # Keep point count moderate for mobile responsiveness.
    nx, ny, nz = 15, 11, 8
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    gz = np.linspace(zmin, zmax, nz)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing="xy")
    query_pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Linear interpolation first; fill boundary gaps with nearest-neighbor values.
    try:
        interp_temp = griddata(coords, temps, query_pts, method="linear")
        missing = ~np.isfinite(interp_temp)
        if np.any(missing):
            interp_temp[missing] = griddata(
                coords, temps, query_pts[missing], method="nearest"
            )
    except Exception:
        interp_temp = griddata(coords, temps, query_pts, method="nearest")

    interp_temp = np.asarray(interp_temp, dtype=float)
    good = np.isfinite(interp_temp)
    query_pts = query_pts[good]
    interp_temp = interp_temp[good]

    temp_scale = [
        [0.00, "#8ee7ff"],  # 18 C - sky blue
        [0.18, "#50c9ff"],
        [0.36, "#17bed0"],
        [0.54, "#4edb78"],
        [0.70, "#b9e63d"],
        [0.84, "#ffa13a"],
        [1.00, "#e63a32"],  # 28 C - red
    ]

    fig = go.Figure()

    # Temperature nodes on the regular 3D lattice.
    fig.add_trace(
        go.Scatter3d(
            x=query_pts[:, 0],
            y=query_pts[:, 1],
            z=query_pts[:, 2],
            mode="markers",
            marker=dict(
                size=2.65,
                color=interp_temp,
                colorscale=temp_scale,
                cmin=18.0,
                cmax=28.0,
                opacity=0.82,
                colorbar=dict(
                    title=dict(text="°C", font=dict(size=11, color="#eefaff")),
                    thickness=5,
                    len=0.56,
                    x=0.992,
                    xpad=2,
                    tickvals=[18, 20, 22, 24, 26, 28],
                    tickfont=dict(size=8, color="#dff4ff"),
                    outlinecolor="rgba(174,228,255,0.20)",
                ),
            ),
            hovertemplate=(
                "X: %{x:.2f} m<br>"
                "Y: %{y:.2f} m<br>"
                "Z: %{z:.2f} m<br>"
                "온도: %{marker.color:.2f} °C"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # White room wireframe.
    corners = {
        "000": (xmin, ymin, zmin), "100": (xmax, ymin, zmin),
        "010": (xmin, ymax, zmin), "110": (xmax, ymax, zmin),
        "001": (xmin, ymin, zmax), "101": (xmax, ymin, zmax),
        "011": (xmin, ymax, zmax), "111": (xmax, ymax, zmax),
    }
    edges = [
        ("000","100"), ("000","010"), ("100","110"), ("010","110"),
        ("001","101"), ("001","011"), ("101","111"), ("011","111"),
        ("000","001"), ("100","101"), ("010","011"), ("110","111"),
    ]
    for a, b in edges:
        xa, ya, za = corners[a]
        xb, yb, zb = corners[b]
        fig.add_trace(
            go.Scatter3d(
                x=[xa, xb], y=[ya, yb], z=[za, zb],
                mode="lines",
                line=dict(color="rgba(239,249,255,0.88)", width=2.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Adaptive sensor overlay.
    # HOME can hide it entirely; comparison screens use the validated 15 -> 5 hierarchy.
    if show_sensors:
        selected_xyz, selected_temp, selected_names = _select_adaptive_sensor_points(
            coords_xyz,
            temp_nodes,
            sensor_count,
        )

        if len(selected_xyz):
            n_sensor_vis = len(selected_xyz)
            marker_size = (
                6.2 if n_sensor_vis <= 5
                else 5.0 if n_sensor_vis <= 10
                else 3.8 if n_sensor_vis <= 20
                else 3.1
            )
            hover_texts = [
                (
                    f"<b>{name}</b><br>"
                    f"X={xyz[0]:.2f}m, Y={xyz[1]:.2f}m, Z={xyz[2]:.2f}m<br>"
                    f"온도={temp:.2f}°C"
                )
                for xyz, temp, name in zip(selected_xyz, selected_temp, selected_names)
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=selected_xyz[:, 0],
                    y=selected_xyz[:, 1],
                    z=selected_xyz[:, 2] + 0.055,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color="#ffffff",
                        line=dict(color="#65ddff", width=1.0),
                        symbol="circle",
                        opacity=0.98,
                    ),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    # Perspective and proportions tuned to the reference mockup.
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(
            domain=dict(x=[0.00, 0.91], y=[0.00, 1.00]),
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="X (m)", font=dict(size=10, color="#ff604d")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8, color="#ff604d"),
                color="#ff604d",
                linecolor="#ff604d",
            ),
            yaxis=dict(
                title=dict(text="Y (m)", font=dict(size=10, color="#71d64e")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8, color="#71d64e"),
                color="#71d64e",
                linecolor="#71d64e",
            ),
            zaxis=dict(
                title=dict(text="Z (m)", font=dict(size=10, color="#7fdcff")),
                showbackground=False,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=8, color="#7fdcff"),
                color="#7fdcff",
                linecolor="#7fdcff",
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.92, y=1.08, z=0.72),
            camera=dict(
                eye=dict(x=1.30, y=-1.48, z=1.00),
                center=dict(x=0.0, y=0.0, z=-0.02),
            ),
        ),
    )

    return fig


# ============================================================
# 5. HEADER
# ============================================================
if st.session_state.app_view != "INTRO":
    st.markdown(
        """
<div class="phone-notch">
    <div class="notch-cam"></div>
    <div class="notch-speaker"></div>
</div>
<div class="app-title-lockup">
    <div class="app-title">AI Smart Cooling</div>
    <div class="brand-spectrum"></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 6. SCREEN 0: INTRO + SCREEN 1: HOME / COOLING SETUP
# ============================================================
if st.session_state.app_view == "HOME":
    # Editable CURRENT temperature is a retrieval query, not a synthetic temperature shift.
    if "home_current_temp_widget" not in st.session_state:
        st.session_state.home_current_temp_widget = float(st.session_state.current_temp_query)
    st.number_input(
        "현재 공간 평균 온도 (°C)",
        min_value=15.0,
        max_value=40.0,
        step=0.1,
        format="%.1f",
        key="home_current_temp_widget",
        on_change=_sync_current_temp_from_home_widget,
    )

    # Keep the HOME screen clean: do not show the nearest-CFD diagnostic under the current temperature.
    if matched_scenario is not None and current_field_source.startswith("Actual CFD"):
        pass
    elif current_field_source.startswith("PopField"):
        if FIELD_ZIP_PATH is None:
            st.warning(
                "실제 CFD ZIP을 앱이 찾지 못했습니다. "
                f"진단: {FIELD_ZIP_ERROR or 'unknown'}"
            )
        elif scenario_table is None or len(scenario_table) == 0:
            st.warning(
                f"CFD ZIP은 로드되었습니다 ({FIELD_ZIP_PATH.name}, {FIELD_ZIP_DP_COUNT} cases). "
                "하지만 Case Info와 CFD 시나리오 인덱스를 연결하지 못해 모델 추정값을 표시합니다."
            )
        else:
            st.warning(
                f"CFD ZIP과 시나리오 표는 로드되었지만 DP {matched_dp_id} 실제 field를 읽지 못해 "
                "모델 추정값을 표시합니다."
            )
    else:
        st.warning("Current Field용 실제 CFD 자산을 불러오지 못했습니다.")

    new_target = st.number_input(
        "목표 온도 (°C)",
        min_value=18.0,
        max_value=30.0,
        value=float(st.session_state.target_temp),
        step=0.1,
        format="%.1f",
        key="home_target_temp_input",
    )

    if float(new_target) != float(st.session_state.target_temp):
        st.session_state.target_temp = float(new_target)
        st.rerun()

    st.markdown(
        """
        <div class="home-field-head" style="margin-top:18px; margin-bottom:10px;">
            <div class="home-field-title">Current Field</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="temperature_map_card"):
        home_field_view = field_view_selector("home_field_view")
        if home_field_view == "3D":
            home_fig = make_true_3d_field(current_coords, current_temp_nodes, height=410, show_sensors=False)
        else:
            home_fig = make_2d_heatmap(field_current_grid, height=315, show_sensors=False)

        st.plotly_chart(
            home_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    if st.button("냉방 최적화", type="primary", use_container_width=True, key="btn_home_to_heat"):
        st.session_state.app_view = "HEAT_LOAD"
        st.rerun()


# ============================================================
# 7. SCREEN 2: COOLING INFLUENCE FACTORS
# ============================================================
elif st.session_state.app_view == "HEAT_LOAD":
    # Keep the two temperatures visible while users tune the cooling factors.
    st.markdown(
        f"""
        <div class="factor-temp-summary">
            <div class="factor-temp-card">
                <div class="factor-temp-label">현재 온도 입력</div>
                <div class="factor-temp-value">{float(st.session_state.current_temp_query):.1f} °C</div>
            </div>
            <div class="factor-temp-card">
                <div class="factor-temp-label">목표 온도</div>
                <div class="factor-temp-value">{st.session_state.target_temp:.1f} °C</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # The title uses a small wall-mounted AC SVG instead of a flame emoji.
    st.markdown(
        """
        <div class="cooling-factor-title">
            <svg viewBox="0 0 96 62" aria-hidden="true">
                <rect x="5" y="8" width="86" height="39" rx="10" fill="none" stroke="#aee4ff" stroke-width="4"/>
                <path d="M13 36 H83 Q82 47 73 50 H23 Q14 47 13 36 Z" fill="rgba(110,210,255,0.12)" stroke="#74d7ff" stroke-width="3"/>
                <path d="M24 40 H72" stroke="#dff7ff" stroke-width="2.5" stroke-linecap="round" opacity="0.9"/>
                <circle cx="70" cy="21" r="2.4" fill="#67d7ff"/>
                <circle cx="77" cy="21" r="2.4" fill="#67d7ff" opacity="0.75"/>
            </svg>
            <span>냉방 영향 요소</span>
        </div>
        <div class="cooling-factor-desc">공간 온도에 영향을 주는 조건을 <span class="step-emphasis">5단계</span>로 설정하세요.</div>
        """,
        unsafe_allow_html=True,
    )

    stage_opts = STAGE_OPTS

    # Old sessions are already compatible because 낮음/보통/높음 remain valid options.
    c1, c2 = st.columns(2)
    with c1:
        p_ext = st.select_slider(
            "☀️ 외부 열환경",
            options=stage_opts,
            value=st.session_state.p_ext if st.session_state.p_ext in stage_opts else "보통",
            key="sl_ext",
        )
        p_meet = st.select_slider(
            "👥 회의공간",
            options=stage_opts,
            value=st.session_state.p_meet if st.session_state.p_meet in stage_opts else "보통",
            key="sl_meet",
        )
    with c2:
        p_serv = st.select_slider(
            "🖥️ 서버 발열",
            options=stage_opts,
            value=st.session_state.p_serv if st.session_state.p_serv in stage_opts else "보통",
            key="sl_serv",
        )
        p_work = st.select_slider(
            "💼 업무공간",
            options=stage_opts,
            value=st.session_state.p_work if st.session_state.p_work in stage_opts else "보통",
            key="sl_work",
        )

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

    # Lightweight interpretation of the four user inputs.
    # The summary is intentionally simple: all four settings contribute equally.
    stage_index = {name: i + 1 for i, name in enumerate(stage_opts)}
    factor_values = {
        "외부 열환경": stage_index[p_ext],
        "서버 발열": stage_index[p_serv],
        "회의공간": stage_index[p_meet],
        "업무공간": stage_index[p_work],
    }
    burden_score = sum(factor_values.values()) / len(factor_values)
    burden_index = max(1, min(5, int(round(burden_score))))
    burden_label = stage_opts[burden_index - 1]
    burden_color_map = {
        "매우 낮음": "#66d9ff",
        "낮음": "#7bd6ef",
        "보통": "#8edbcb",
        "높음": "#ffad66",
        "매우 높음": "#ff6b7a",
    }
    burden_label_color = burden_color_map.get(burden_label, "#f5fbff")
    # 주요 영향 요인은 '높음(4)' 또는 '매우 높음(5)'으로 설정된 항목만 별도 표시합니다.
    factor_icons = {
        "외부 열환경": "☀️",
        "서버 발열": "🖥️",
        "회의공간": "👥",
        "업무공간": "💼",
    }
    factor_chip_class = {
        "외부 열환경": "factor-ext",
        "서버 발열": "factor-serv",
        "회의공간": "factor-meet",
        "업무공간": "factor-work",
    }
    major_factors = [(name, level) for name, level in factor_values.items() if level >= 4]

    segments_html = "".join(
        f'<div class="cooling-load-segment {f"on-{i}" if i <= burden_index else ""}"></div>'
        for i in range(1, 6)
    )

    if major_factors:
        major_chips_html = "".join(
            f'<span class="major-factor-chip {factor_chip_class[name]}">'             f'{factor_icons[name]} {name} · {stage_opts[level - 1]}</span>'
            for name, level in major_factors
        )
    else:
        major_chips_html = '<span class="major-factor-empty">현재 열환경 수준에 영향 요인이 없습니다.</span>'

    st.markdown(
        f"""
        <div class="cooling-load-card">
            <div class="cooling-load-top">
                <div class="cooling-load-label">종합 열환경 수준</div>
                <div class="cooling-load-level" style="color:{burden_label_color};">{burden_label}</div>
            </div>
            <div class="cooling-load-segments">{segments_html}</div>
        </div>

        <div class="major-factor-card">
            <div class="major-factor-title">주요 영향 요인</div>
            <div class="major-factor-chips">{major_chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("AI 최적 냉방 찾기", type="primary", use_container_width=True, key="btn_run_cooling_opt"):
        backend = load_popfield_backend()

        if not backend.get("ok", False):
            st.error(
                "PopField 모델을 실행할 수 없습니다. "
                + str(backend.get("error", "Unknown model loading error"))
            )
        elif case_info_df is None:
            st.error("Case Info Excel을 불러오지 못했습니다.")
        else:
            try:
                target = float(st.session_state.target_temp)
                policy = st.session_state.policy
                query_loads, stage_load_maps = _requested_heat_loads_from_ui()
                retrieval = _find_nearest_cfd_scenario(
                    float(st.session_state.current_temp_query),
                    query_loads,
                )
                if retrieval is None or FIELD_ZIP_PATH is None:
                    raise RuntimeError(
                        "실제 CFD scenario retrieval에는 GitHub 루트의 Field data.zip이 필요합니다."
                    )

                matched_current = load_actual_cfd_case(str(FIELD_ZIP_PATH), int(retrieval["dp_id"]))
                if matched_current is None:
                    raise RuntimeError(f"dp{int(retrieval['dp_id'])}.csv를 Field data.zip에서 읽지 못했습니다.")

                # Critical consistency rule: the Current Field and the optimization share
                # the matched scenario's ACTUAL four heat loads. Only HVAC actions change.
                loads = dict(retrieval["matched_loads"])

                runtime_dir = Path(tempfile.gettempdir()) / "acpop_streamlit_runtime"
                runtime_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner("AI 예측 중..."):
                    optimize_fn = backend.get("optimize_hvac_fn")
                    predict_fn = backend.get("predict_conditions_fn")

                    # Defensive recovery in case Streamlit is holding an old
                    # cached backend object from the previous build.
                    if not callable(optimize_fn) or not callable(predict_fn):
                        if not _lazy_import_popfield_modules():
                            raise RuntimeError(
                                f"PopField 모듈을 불러오지 못했습니다: {POPFIELD_BACKEND_IMPORT_ERROR}"
                            )
                        optimize_fn = popfield_optimize_hvac
                        predict_fn = popfield_predict_conditions

                    if not callable(optimize_fn):
                        raise RuntimeError("PopField optimize_hvac 함수를 불러오지 못했습니다.")
                    if not callable(predict_fn):
                        raise RuntimeError("PopField predict_conditions 함수를 불러오지 못했습니다.")

                    opt_df = optimize_fn(
                        model=backend["model"],
                        case_df=case_info_df,
                        loads=loads,
                        cond_scaler=backend["scalers"]["cond"],
                        coords=backend["coords"],
                        coords_norm_t=backend["coords_norm_t"],
                        field_scaler=backend["scalers"]["field"],
                        ra_scaler=backend["scalers"]["ra"],
                        device=backend["device"],
                        save_dir=runtime_dir,
                        zone_json=None,
                        target_temp_c=target,
                        comfort_band_c=2.0,
                        max_zone_range_c=2.0,
                        max_hot_fraction=0.05,
                        max_cold_fraction=0.05,
                        max_p95_temp_c=target + 2.0,
                        energy_weight=0.35,
                    )

                    feasible_df = opt_df[opt_df["comfort_constraint_met"].astype(bool)].copy()
                    if len(feasible_df):
                        # Balanced policy from the original deployment optimizer:
                        # minimize the model's combined comfort + cooling-load score.
                        rec = feasible_df.sort_values(
                            ["combined_score", "comfort_raw"]
                        ).iloc[0]
                    else:
                        # Constraint-first optimizer already puts the least-violating
                        # action at the top when no fully feasible action exists.
                        rec = opt_df.iloc[0]

                    cond = np.asarray([[
                        float(rec["Inlet_L"]),
                        float(rec["Inlet_M"]),
                        float(rec["Inlet_R"]),
                        float(loads["external"]),
                        float(loads["meeting"]),
                        float(loads["server"]),
                        float(loads["working"]),
                        float(rec["CMM"]),
                        float(rec["AirTemp_C"]),
                    ]], dtype=np.float32)

                    pred_field, pred_ra = predict_fn(
                        backend["model"],
                        cond,
                        backend["scalers"]["cond"],
                        backend["coords_norm_t"],
                        backend["scalers"]["field"],
                        backend["scalers"]["ra"],
                        backend["device"],
                    )

                pred_temp_nodes = np.asarray(pred_field[0, :, 0], dtype=np.float32)
                field_post_grid = _temperature_plane_grid(
                    backend["coords"],
                    pred_temp_nodes,
                    st.session_state.z_plane,
                    grid_len_axis,
                    grid_wid_axis,
                )

                status_opt = _demo_status_from_row(rec, target)
                st.session_state.optimized_results = {
                    "status": status_opt,
                    "vane": _direction_label(rec),
                    "flow": f"{float(rec['CMM']):.0f} CMM",
                    "temp": f"{float(rec['AirTemp_C']):.0f} °C",
                    "mean_temp": float(rec["mean_temp_C"]),
                    "p95_temp": float(rec["p95_temp_C"]),
                    "zone_spread": float(rec["zone_range_C"]),
                    "hot_fraction": float(rec["hot_fraction"]) * 100.0,
                    "cold_fraction": float(rec["cold_fraction"]) * 100.0,
                    "q_proxy": float(rec["estimated_sensible_cooling_kw"]),

                    # Candidate ranges used only for recommendation-card visualization.
                    # This makes bar positions relative to the ACTUAL candidate set,
                    # rather than using arbitrary hard-coded min/max values.
                    "flow_min": float(pd.to_numeric(opt_df["CMM"], errors="coerce").min()),
                    "flow_max": float(pd.to_numeric(opt_df["CMM"], errors="coerce").max()),
                    "supply_temp_min": float(pd.to_numeric(opt_df["AirTemp_C"], errors="coerce").min()),
                    "supply_temp_max": float(pd.to_numeric(opt_df["AirTemp_C"], errors="coerce").max()),
                    "q_min": float(pd.to_numeric(opt_df["estimated_sensible_cooling_kw"], errors="coerce").min()),
                    "q_max": float(pd.to_numeric(opt_df["estimated_sensible_cooling_kw"], errors="coerce").max()),

                    "policy_used": policy,
                    "field_post_grid": np.asarray(field_post_grid, dtype=np.float32),
                    "field_post_coords": np.asarray(backend["coords"], dtype=np.float32),
                    "field_post_temp_nodes": np.asarray(pred_temp_nodes, dtype=np.float32),
                    "pred_ra_temp_c": float(pred_ra[0]),
                    "num_candidates": int(len(opt_df)),
                    "mapped_loads_W": {k: float(v) for k, v in loads.items()},
                    "checkpoint_used": str(backend["checkpoint_path"]),
                    "model_inference_used": True,
                    "matched_dp_id": int(retrieval["dp_id"]),
                    "current_temp_query_c": float(st.session_state.current_temp_query),
                    "matched_mean_temp_c": float(retrieval["mean_temp_c"]),
                    "retrieval_score": float(retrieval["retrieval_score"]),
                    "query_loads_W": {k: float(v) for k, v in query_loads.items()},
                    "matched_loads_W": {k: float(v) for k, v in loads.items()},
                    "field_current_grid": np.asarray(
                        _temperature_plane_grid(
                            matched_current["coords"],
                            matched_current["temp_c"],
                            st.session_state.z_plane,
                            grid_len_axis,
                            grid_wid_axis,
                        ),
                        dtype=np.float32,
                    ),
                    "field_current_coords": np.asarray(matched_current["coords"], dtype=np.float32),
                    "field_current_temp_nodes": np.asarray(matched_current["temp_c"], dtype=np.float32),
                }

                st.session_state.has_run_optimization = True
                st.session_state.show_control_simulation = False
                st.session_state.app_view = "RESULTS"
                st.rerun()

            except Exception as exc:
                st.error(f"PopField 최적화 실행 중 오류: {type(exc).__name__}: {exc}")


# ============================================================
# 8. SCREEN 3: RESULTS
# ============================================================
elif st.session_state.app_view == "RESULTS":
    if not st.session_state.has_run_optimization:
        st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)
        st.info("아직 실행된 최적화 분석이 없습니다. 먼저 냉방 조건을 설정하고 AI 최적화를 실행해 주세요.")

        if st.button("AI 최적화 설정 시작하기", type="primary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()

        if st.button("홈으로 이동", type="secondary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()

    else:
        st.markdown(
            '<div class="section-title results-title-row"><span class="results-title-glyph">❄</span>AI 최적 냉방 결과</div>',
            unsafe_allow_html=True,
        )

        res = st.session_state.optimized_results

        vane_map = {
            "Left (L)": "좌측 (L)",
            "Middle (M)": "중앙 (M)",
            "Right (R)": "우측 (R)",
            "L / M": "좌측 / 중앙",
            "M / R": "중앙 / 우측",
            "L / R": "좌측 / 우측",
        }
        vane_display = vane_map.get(str(res["vane"]), str(res["vane"]))

        # --------------------------------------------------------
        # Visual recommendation cards
        # --------------------------------------------------------
        flow_cmm = float(str(res["flow"]).replace("CMM", "").strip())
        supply_temp_c = float(str(res["temp"]).replace("°C", "").replace("° C", "").strip())
        q_kw = float(res["q_proxy"])

        flow_min = float(res.get("flow_min", flow_cmm))
        flow_max = float(res.get("flow_max", flow_cmm))
        supply_min = float(res.get("supply_temp_min", supply_temp_c))
        supply_max = float(res.get("supply_temp_max", supply_temp_c))
        q_min = float(res.get("q_min", q_kw))
        q_max = float(res.get("q_max", q_kw))

        def _pct(value, lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return 50.0
            return float(np.clip((value - lo) / (hi - lo) * 100.0, 0.0, 100.0))

        flow_pct = _pct(flow_cmm, flow_min, flow_max)
        temp_pct = _pct(supply_temp_c, supply_min, supply_max)
        q_pct = _pct(q_kw, q_min, q_max)

        # Convert the relative flow position into 1–5 visual bars.
        flow_level = int(np.clip(np.ceil(max(flow_pct, 1.0) / 20.0), 1, 5))
        flow_bars_html = "".join(
            f'<span class="flow-bar {"active" if i <= flow_level else ""}"></span>'
            for i in range(1, 6)
        )

        vane_raw = str(res.get("vane", ""))
        left_on = ("Left" in vane_raw) or ("L" in vane_raw.split(" / ")) or ("좌측" in vane_display)
        middle_on = ("Middle" in vane_raw) or ("M" in vane_raw.split(" / ")) or ("중앙" in vane_display)
        right_on = ("Right" in vane_raw) or ("R" in vane_raw.split(" / ")) or ("우측" in vane_display)

        direction_html = (
            f'<div class="air-direction-wrap">'
            f'<div class="ac-mini"></div>'
            f'<div class="air-rays">'
            f'<div class="air-dir {"active" if left_on else ""}"><span class="air-ray">↙</span><span class="air-dir-tag">좌</span></div>'
            f'<div class="air-dir {"active" if middle_on else ""}"><span class="air-ray">↓</span><span class="air-dir-tag">중</span></div>'
            f'<div class="air-dir {"active" if right_on else ""}"><span class="air-ray">↘</span><span class="air-dir-tag">우</span></div>'
            f'</div>'
            f'</div>'
        )

        recommendation_html = (
            f'<div class="optimal-dispatch-box">'
            f'<h4>AI 추천 냉방 설정</h4>'
            f'<div class="hvac-visual-grid">'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">바람 방향</div>'
            f'<div class="hvac-mini-value">{vane_display}</div></div>'
            f'{direction_html}'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">풍량</div>'
            f'<div class="hvac-mini-value">{flow_cmm:.0f} CMM</div></div>'
            f'<div class="flow-bars">{flow_bars_html}</div>'
            f'<div class="hvac-card-note">후보 범위 내 상대 세기</div>'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">공급 공기 온도</div>'
            f'<div class="hvac-mini-value">{supply_temp_c:.0f}°C</div></div>'
            f'<div class="hvac-track-wrap">'
            f'<div class="hvac-track temp-track">'
            f'<span class="hvac-marker" style="left:{temp_pct:.1f}%;"></span>'
            f'</div>'
            f'<div class="hvac-range"><span>{supply_min:.0f}°</span>'
            f'<span>{supply_max:.0f}°</span></div>'
            f'</div>'
            f'<div class="hvac-card-note">추천 공급 공기 설정</div>'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">예상 냉방 출력</div>'
            f'<div class="hvac-mini-value">{q_kw:.2f} kW</div></div>'
            f'<div class="hvac-track-wrap">'
            f'<div class="hvac-track power-track">'
            f'<span class="power-fill" style="width:{q_pct:.1f}%;"></span>'
            f'<span class="hvac-marker" style="left:{q_pct:.1f}%;"></span>'
            f'</div>'
            f'<div class="hvac-range"><span>{q_min:.1f}</span>'
            f'<span>{q_max:.1f} kW</span></div>'
            f'</div>'
            f'<div class="hvac-card-note">후보 범위 내 상대 출력</div>'
            f'</div>'

            f'</div>'
            f'</div>'
        )

        st.markdown(recommendation_html, unsafe_allow_html=True)


        # --------------------------------------------------------
        # Adaptive Sensor Plan
        # Validated nested sensor hierarchy: 5 -> 6 -> ... -> 14 -> 15.
        # Only ACTIVE monitoring count changes; the installed pool is capped at 15.
        # --------------------------------------------------------
        current_reference_temp = float(
            res.get("current_temp_query_c", st.session_state.current_temp_query)
        )
        predicted_reference_temp = float(
            res.get("mean_temp", st.session_state.target_temp)
        )
        target_sensor_temp = float(st.session_state.target_temp)

        current_sensor_count = _active_sensor_count_from_temperature(
            current_reference_temp,
            target_sensor_temp,
        )
        recommended_sensor_count = _active_sensor_count_from_temperature(
            predicted_reference_temp,
            target_sensor_temp,
        )

        # Absolute hard cap: this build can never persist or render >15 sensors.
        current_sensor_count = int(np.clip(
            current_sensor_count,
            MIN_ACTIVE_SENSORS,
            MAX_ACTIVE_SENSORS,
        ))
        recommended_sensor_count = int(np.clip(
            recommended_sensor_count,
            MIN_ACTIVE_SENSORS,
            MAX_ACTIVE_SENSORS,
        ))

        predicted_error_c = abs(predicted_reference_temp - target_sensor_temp)

        if recommended_sensor_count <= 5:
            sensor_stage = "안정 운전"
            sensor_reason = "목표 온도에 가까워져 검증된 핵심 센서 5개만 활성화합니다."
        elif recommended_sensor_count <= 8:
            sensor_stage = "안정화 단계"
            sensor_reason = f"목표 편차 {predicted_error_c:.1f}°C에 맞춰 {recommended_sensor_count}개 센서를 활성화합니다."
        elif recommended_sensor_count <= 12:
            sensor_stage = "정밀 모니터링"
            sensor_reason = f"목표 편차 {predicted_error_c:.1f}°C가 남아 {recommended_sensor_count}개 센서를 활성화합니다."
        elif recommended_sensor_count < 15:
            sensor_stage = "고밀도 모니터링"
            sensor_reason = f"목표 편차가 커 {recommended_sensor_count}개 센서를 활성화합니다."
        else:
            sensor_stage = "최대 모니터링"
            sensor_reason = "목표 온도와의 차이가 커 최대 15개 센서를 활성화합니다."

        deactivated_sensor_count = current_sensor_count - recommended_sensor_count

        # Persist exact active counts for the BEFORE/AFTER comparison.
        res["initial_sensor_count"] = int(current_sensor_count)
        res["recommended_sensor_count"] = int(recommended_sensor_count)
        res["adaptive_sensor_stage"] = str(sensor_stage)
        st.session_state.recommended_sensor_count = int(recommended_sensor_count)
        st.session_state.optimized_results = res

        # Keep the SAME schematic card/layout, but use 15 sensor positions only.
        # The positions are derived from the validated node coordinates.
        _sensor_order = _nested_sensor_order(len(current_coords))
        _pool_nodes = _sensor_order[:MAX_ACTIVE_SENSORS]

        if len(_pool_nodes) > 0:
            _pool_xy = np.asarray(current_coords, dtype=float)[
                np.asarray(_pool_nodes, dtype=int), :2
            ]
            _xmin, _ymin = np.nanmin(_pool_xy, axis=0)
            _xmax, _ymax = np.nanmax(_pool_xy, axis=0)
            _xspan = max(float(_xmax - _xmin), 1e-9)
            _yspan = max(float(_ymax - _ymin), 1e-9)

            sensor_points = [
                (
                    12.0 + 76.0 * (float(x) - float(_xmin)) / _xspan,
                    12.0 + 76.0 * (float(y) - float(_ymin)) / _yspan,
                )
                for x, y in _pool_xy
            ]
        else:
            sensor_points = [
                (14, 20), (32, 16), (50, 20), (68, 16), (86, 20),
                (14, 50), (32, 46), (50, 50), (68, 46), (86, 50),
                (14, 80), (32, 76), (50, 80), (68, 76), (86, 80),
            ]

        active_before = set(range(min(current_sensor_count, len(sensor_points))))
        active_after = set(range(min(recommended_sensor_count, len(sensor_points))))

        before_dots = "".join(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.25" fill="#70e8ff" '
                f'stroke="#d8f8ff" stroke-width="0.55" '
                f'style="filter:drop-shadow(0 0 3px rgba(87,222,255,.82));"/>'
                if i in active_before
                else
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="#415c72" '
                f'opacity="0.55" stroke="#688197" stroke-width="0.35"/>'
            )
            for i, (x, y) in enumerate(sensor_points)
        )

        after_dots = "".join(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.35" fill="#70e8ff" '
                f'stroke="#e6fbff" stroke-width="0.65" '
                f'style="filter:drop-shadow(0 0 3px rgba(87,222,255,.82));"/>'
                if i in active_after
                else
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="#415c72" '
                f'opacity="0.55" stroke="#688197" stroke-width="0.35"/>'
            )
            for i, (x, y) in enumerate(sensor_points)
        )

        if deactivated_sensor_count > 0:
            reduction_text = f"{deactivated_sensor_count}개 비활성화"
        elif deactivated_sensor_count < 0:
            reduction_text = f"{abs(deactivated_sensor_count)}개 추가 활성화"
        else:
            reduction_text = "활성 센서 수 유지"

        adaptive_sensor_html = f"""
        <style>
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            background: transparent;
            font-family: Inter, "Noto Sans KR", Arial, sans-serif;
            color: #f5fbff;
          }}
          .asp-shell {{
            width: 100%;
            border-radius: 22px;
            padding: 18px 18px 16px;
            background: linear-gradient(160deg, #0a2c4b 0%, #08243f 100%);
            border: 1px solid rgba(81,194,242,.34);
            box-shadow: 0 10px 24px rgba(0,0,0,.16);
          }}
          .asp-head {{
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:12px;
          }}
          .asp-title {{
            font-size: 22px;
            line-height: 1.1;
            font-weight: 850;
            letter-spacing: -0.025em;
          }}
          .asp-sub {{
            margin-top: 5px;
            font-size: 11px;
            font-weight: 750;
            color: #66dcff;
          }}
          .asp-stage {{
            font-size: 10px;
            font-weight: 800;
            color: #9edbf5;
            border: 1px solid rgba(102,220,255,.30);
            border-radius: 999px;
            padding: 6px 9px;
            white-space: nowrap;
            background: rgba(9,48,79,.70);
          }}
          .asp-count {{
            margin-top: 15px;
            display: grid;
            grid-template-columns: 1fr 44px 1fr;
            align-items: center;
          }}
          .asp-count-side {{ text-align:center; }}
          .asp-num {{
            font-size: 37px;
            font-weight: 900;
            line-height: 1;
            color: #f7fdff;
          }}
          .asp-num.after {{
            color: #6fe2ff;
            text-shadow: 0 0 14px rgba(74,210,255,.18);
          }}
          .asp-caption {{
            margin-top: 5px;
            font-size: 9.5px;
            font-weight: 750;
            color: #a8cfdf;
          }}
          .asp-arrow {{
            text-align:center;
            font-size: 27px;
            color:#77dcff;
            font-weight:800;
          }}
          .asp-reason {{
            margin: 12px auto 13px;
            max-width: 94%;
            text-align:center;
            color:#b9d9e8;
            font-size: 10.5px;
            line-height:1.55;
          }}
          .asp-maps {{
            margin-top: 20px;
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap:14px;
            padding: 12px;
            border-radius:16px;
            background: rgba(4,25,45,.58);
            border: 1px solid rgba(86,168,209,.18);
          }}
          .asp-map-title {{
            display:flex;
            justify-content:space-between;
            align-items:baseline;
            gap:4px;
            padding: 0 3px 6px;
          }}
          .asp-map-title b {{ font-size:13px; color:#eafaff; }}
          .asp-map-title span {{
            font-size:9.5px;
            color:#79bfdc;
            font-weight:700;
          }}
          .room {{
            width:100%;
            height:172px;
            display:block;
            border-radius:11px;
            background:linear-gradient(145deg,#0a223a,#0c2c49);
            border:1px solid rgba(126,208,244,.15);
          }}
</style>

        <div class="asp-shell">
          <div class="asp-head">
            <div>
              <div class="asp-title">Adaptive Sensor Plan</div>
              <div class="asp-sub">활성 센서 조정</div>
            </div>
            <div class="asp-stage">{sensor_stage}</div>
          </div>

          <div class="asp-count">
            <div class="asp-count-side">
              <div class="asp-num">{current_sensor_count}</div>
              <div class="asp-caption">초기 정밀 모니터링</div>
            </div>
            <div class="asp-arrow">→</div>
            <div class="asp-count-side">
              <div class="asp-num after">{recommended_sensor_count}</div>
              <div class="asp-caption">안정화 후 핵심 유지</div>
            </div>
          </div>

          <div class="asp-maps">
            <div>
              <div class="asp-map-title">
                <b>Before</b><span>활성 센서 {current_sensor_count}개</span>
              </div>
              <svg class="room" viewBox="0 0 100 100">
                <rect x="5" y="5" width="90" height="90" rx="5"
                      fill="#092641" stroke="#315b77" stroke-width="1"/>
                <rect x="30" y="35" width="40" height="22" rx="5"
                      fill="#173c5d" opacity=".9"/>
                <rect x="38" y="66" width="24" height="14" rx="3"
                      fill="#173b59" opacity=".78"/>
                <path d="M10 25 H90 M10 75 H90 M25 10 V90 M75 10 V90"
                      stroke="#214964" stroke-width=".45" opacity=".5"/>
                {before_dots}
              </svg>
            </div>

            <div>
              <div class="asp-map-title">
                <b>After</b><span>활성 센서 {recommended_sensor_count}개</span>
              </div>
              <svg class="room" viewBox="0 0 100 100">
                <rect x="5" y="5" width="90" height="90" rx="5"
                      fill="#092641" stroke="#315b77" stroke-width="1"/>
                <rect x="30" y="35" width="40" height="22" rx="5"
                      fill="#173c5d" opacity=".9"/>
                <rect x="38" y="66" width="24" height="14" rx="3"
                      fill="#173b59" opacity=".78"/>
                <path d="M10 25 H90 M10 75 H90 M25 10 V90 M75 10 V90"
                      stroke="#214964" stroke-width=".45" opacity=".5"/>
                {after_dots}
              </svg>
            </div>
          </div>

        </div>
        """

        components.html(adaptive_sensor_html, height=410, scrolling=False)

        if st.button(
            "AI 제어안 시뮬레이션",
            type="primary",
            use_container_width=True,
            key="btn_control_simulation",
        ):
            st.session_state.show_control_simulation = True
            st.session_state.app_view = "COMPARE"
            st.rerun()

        if st.button(
            "새로운 최적화 실행",
            type="secondary",
            use_container_width=True,
            key="btn_restart_from_results",
        ):
            st.session_state.show_control_simulation = False
            st.session_state.app_view = "HOME"
            st.rerun()


# ============================================================
# 9. SCREEN 4: BEFORE → AFTER EFFECT COMPARISON
# ============================================================
elif st.session_state.app_view == "COMPARE":
    if not st.session_state.has_run_optimization:
        st.session_state.app_view = "RESULTS"
        st.rerun()

    res = st.session_state.optimized_results
    target = float(st.session_state.target_temp)

    result_current_grid = np.asarray(
        res.get("field_current_grid", field_current_grid),
        dtype=float,
    )
    result_pred_grid = np.asarray(
        res.get("field_post_grid", field_current_grid),
        dtype=float,
    )
    result_current_coords = np.asarray(
        res.get("field_current_coords", current_coords),
        dtype=float,
    )
    result_current_nodes = np.asarray(
        res.get("field_current_temp_nodes", current_temp_nodes),
        dtype=float,
    )
    result_pred_coords = np.asarray(
        res.get("field_post_coords", result_current_coords),
        dtype=float,
    )
    result_pred_nodes = np.asarray(
        res.get("field_post_temp_nodes", result_current_nodes),
        dtype=float,
    )

    # Use the same definitions for BEFORE and AFTER so the comparison is fair.
    before_mean = float(np.nanmean(result_current_nodes))
    after_mean = float(np.nanmean(result_pred_nodes))

    before_p05 = float(np.nanpercentile(result_current_nodes, 5))
    before_p95 = float(np.nanpercentile(result_current_nodes, 95))
    after_p05 = float(np.nanpercentile(result_pred_nodes, 5))
    after_p95 = float(np.nanpercentile(result_pred_nodes, 95))

    before_spread = max(0.0, before_p95 - before_p05)
    after_spread = max(0.0, after_p95 - after_p05)

    # "목표 초과 영역" is easier to understand than HVAC-specific hotspot jargon.
    # We count points more than 1°C above the target.
    before_hot = float(np.mean(result_current_nodes > (target + 1.0)) * 100.0)
    after_hot = float(np.mean(result_pred_nodes > (target + 1.0)) * 100.0)

    mean_delta = after_mean - before_mean
    spread_improve_pct = (
        max(0.0, (before_spread - after_spread) / before_spread * 100.0)
        if before_spread > 1e-8
        else 0.0
    )
    hot_improve_pp = before_hot - after_hot

    status = str(res.get("status", "INFEASIBLE"))
    if status == "FEASIBLE":
        status_text = "목표 온도 달성 가능"
        status_color = "#74e0a8"
        status_symbol = "✓"
    elif status == "NEAR_FEASIBLE":
        status_text = "목표 온도 근접 달성"
        status_color = "#ffd36b"
        status_symbol = "•"
    else:
        status_text = "목표 온도 달성 어려움"
        status_color = "#ff7d8b"
        status_symbol = "×"

    # Comparison-screen-only styling.
    st.markdown(
        """
        <style>
        .compare-eyebrow {
            font-size: 12px;
            letter-spacing: 0.16em;
            color: #62d6ff;
            font-weight: 800;
            margin: 2px 0 6px 0;
        }
        .compare-title {
            font-size: 30px;
            color: #f4fbff;
            font-weight: 800;
            margin: 0 0 16px 0;
            letter-spacing: -0.02em;
        }
        .compare-hero {
            border-radius: 22px;
            padding: 20px 18px;
            background: linear-gradient(145deg, rgba(8,43,72,.96), rgba(14,55,83,.92));
            border: 1px solid rgba(118,203,244,.24);
            text-align: center;
            margin-bottom: 16px;
        }
        .compare-temp-row {
            display:flex;
            align-items:center;
            justify-content:center;
            gap:14px;
            flex-wrap:wrap;
        }
        .compare-temp {
            font-size: 36px;
            font-weight: 800;
            color:#ffffff;
        }
        .compare-arrow {
            font-size: 28px;
            color:#6bd8ff;
            font-weight:800;
        }
        .compare-target {
            margin-top:8px;
            font-size:13px;
            color:#a9d2e8;
        }
        .compare-status {
            margin-top:10px;
            font-size:15px;
            font-weight:800;
        }
        .compare-card {
            border-radius: 18px;
            background: rgba(15,57,87,.72);
            border: 1px solid rgba(116,191,230,.20);
            padding: 15px 14px;
            margin-bottom: 10px;
        }
        .compare-card-title {
            font-size: 13px;
            color:#b4d7e8;
            font-weight:700;
            margin-bottom:8px;
        }
        .compare-values {
            display:grid;
            grid-template-columns:1fr auto 1fr;
            align-items:center;
            gap:8px;
        }
        .compare-before, .compare-after {
            font-size:24px;
            font-weight:800;
            color:#f7fbff;
        }
        .compare-after { text-align:right; }
        .compare-mini-arrow {
            color:#55d3ff;
            font-size:32px;
            font-weight:900;
            line-height:1;
            text-shadow:0 0 12px rgba(85,211,255,.22);
        }
        .compare-change {
            display:inline-flex;
            align-items:center;
            margin-top:10px;
            padding:4px 9px;
            border-radius:999px;
            background:rgba(89,218,167,.10);
            border:1px solid rgba(113,225,175,.22);
            font-size:13px;
            color:#86edbd;
            font-weight:800;
        }
        .compare-map-label {
            display:inline-flex;
            align-items:center;
            gap:7px;
            width:fit-content;
            padding:9px 13px;
            border-radius:13px;
            background:#102d4d;
            border:1px solid rgba(105,202,243,.34);
            box-shadow:0 5px 14px rgba(2,18,34,.18);
            font-size:15px;
            color:#e9f8ff;
            font-weight:800;
            margin-top:8px;
            margin-bottom:8px;
        }
        .compare-map-label .sensor-count {
            color:#8fdfff;
            font-size:14px;
            font-weight:800;
        }
        .compare-summary {
            margin:16px 0 12px 0;
            padding:17px 16px;
            border-radius:20px;
            background:rgba(66,188,147,.10);
            border:1px solid rgba(114,224,168,.50);
            color:#dffbf0;
            line-height:1.65;
            font-size:13px;
        }
        .compare-summary strong {
            color:#78e0aa;
            font-size:16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="compare-eyebrow">BEFORE → AFTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="compare-title">AI 냉방 효과 분석</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="compare-hero">
            <div class="compare-temp-row">
                <div class="compare-temp">{before_mean:.1f}°C</div>
                <div class="compare-arrow">→</div>
                <div class="compare-temp">{after_mean:.1f}°C</div>
            </div>
            <div class="compare-target">목표 온도 {target:.1f}°C</div>
            <div class="compare-status" style="color:{status_color};">
                {status_symbol} {status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="compare-card">
            <div class="compare-card-title">평균 온도</div>
            <div class="compare-values">
                <div class="compare-before">{before_mean:.2f}°C</div>
                <div class="compare-mini-arrow">→</div>
                <div class="compare-after">{after_mean:.2f}°C</div>
            </div>
            <div class="compare-change">{abs(mean_delta):.2f}°C 변화</div>
        </div>

        <div class="compare-card">
            <div class="compare-card-title">공간 온도 편차</div>
            <div class="compare-values">
                <div class="compare-before">{before_spread:.2f}°C</div>
                <div class="compare-mini-arrow">→</div>
                <div class="compare-after">{after_spread:.2f}°C</div>
            </div>
            <div class="compare-change">온도 불균형 {spread_improve_pct:.0f}% 개선</div>
        </div>

        <div class="compare-card">
            <div class="compare-card-title">목표 초과 영역</div>
            <div class="compare-values">
                <div class="compare-before">{before_hot:.1f}%</div>
                <div class="compare-mini-arrow">→</div>
                <div class="compare-after">{after_hot:.1f}%</div>
            </div>
            <div class="compare-change">{max(0.0, hot_improve_pp):.1f}%p 감소</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:18px;">공간 온도 변화</div>', unsafe_allow_html=True)

    if "compare_field_mode" not in st.session_state:
        st.session_state.compare_field_mode = "BEFORE"

    if hasattr(st, "segmented_control"):
        compare_field_mode = st.segmented_control(
            "Before / After",
            options=["BEFORE", "AFTER"],
            selection_mode="single",
            key="compare_field_mode",
            label_visibility="collapsed",
        ) or "BEFORE"
    else:
        compare_field_mode = st.radio(
            "Before / After",
            options=["BEFORE", "AFTER"],
            horizontal=True,
            key="compare_field_mode",
            label_visibility="collapsed",
        )

    compare_view = field_view_selector("compare_map_view")

    before_active_sensor_count = int(np.clip(
        int(res.get("initial_sensor_count", MAX_ACTIVE_SENSORS)),
        MIN_ACTIVE_SENSORS,
        MAX_ACTIVE_SENSORS,
    ))
    after_active_sensor_count = int(np.clip(
        int(
            res.get(
                "recommended_sensor_count",
                st.session_state.get("recommended_sensor_count", MIN_ACTIVE_SENSORS),
            )
        ),
        MIN_ACTIVE_SENSORS,
        MAX_ACTIVE_SENSORS,
    ))

    # Sanitize stale values from older 20/30-sensor sessions.
    res["initial_sensor_count"] = before_active_sensor_count
    res["recommended_sensor_count"] = after_active_sensor_count
    st.session_state.recommended_sensor_count = after_active_sensor_count
    st.session_state.optimized_results = res

    if compare_field_mode == "BEFORE":
        st.markdown(
            f'<div class="compare-map-label"><span>Current Field</span><span class="sensor-count">· 활성 센서 {before_active_sensor_count}개</span></div>',
            unsafe_allow_html=True,
        )
        if compare_view == "3D":
            compare_fig = make_true_3d_field(
                result_current_coords,
                result_current_nodes,
                height=430,
                show_sensors=True,
                sensor_count=before_active_sensor_count,
            )
        else:
            compare_fig = make_2d_heatmap(
                result_current_grid,
                height=330,
                show_sensors=True,
                sensor_count=before_active_sensor_count,
                coords_xyz=result_current_coords,
                temp_nodes=result_current_nodes,
            )
    else:
        st.markdown(
            f'<div class="compare-map-label"><span>Predicted Field</span><span class="sensor-count">· 활성 센서 {after_active_sensor_count}개</span></div>',
            unsafe_allow_html=True,
        )
        if compare_view == "3D":
            compare_fig = make_true_3d_field(
                result_pred_coords,
                result_pred_nodes,
                height=430,
                show_sensors=True,
                sensor_count=after_active_sensor_count,
            )
        else:
            compare_fig = make_2d_heatmap(
                result_pred_grid,
                height=330,
                show_sensors=True,
                sensor_count=after_active_sensor_count,
                coords_xyz=result_pred_coords,
                temp_nodes=result_pred_nodes,
            )

    st.plotly_chart(
        compare_fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"compare_plot_{compare_field_mode}_{compare_view}",
    )

    if status == "FEASIBLE":
        conclusion = (
            f"추천 제어안을 적용하면 목표 {target:.1f}°C에 도달하면서 "
            f"공간 온도 불균형이 약 {spread_improve_pct:.0f}% 감소할 것으로 예측됩니다."
        )
    elif status == "NEAR_FEASIBLE":
        conclusion = (
            f"추천 제어안 적용 후 목표 온도에 근접하며, "
            f"공간 온도 불균형은 약 {spread_improve_pct:.0f}% 개선될 것으로 예측됩니다."
        )
    else:
        conclusion = (
            f"현재 냉방 후보 범위만으로는 목표 {target:.1f}°C 달성이 어렵지만, "
            f"공간 온도 분포 변화와 개선 가능성을 사전에 확인할 수 있습니다."
        )

    st.markdown(
        f"""
        <div class="compare-summary">
            <strong>{status_symbol} 냉방 효과</strong><br>
            {conclusion}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#789eb4;
            font-size:10px;
            line-height:1.5;
            margin:4px 8px 14px 8px;
        ">
            실제 에어컨에 명령을 전송한 결과가 아니라,
            AI 추천 제어안을 적용했을 때의 공간 온도를 예측한 시뮬레이션입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "새로운 최적화 실행",
        type="secondary",
        use_container_width=True,
        key="btn_restart_from_compare",
    ):
        st.session_state.show_control_simulation = False
        st.session_state.app_view = "HOME"
        st.rerun()


# ============================================================
# 10. BOTTOM NAVIGATION BAR
# ============================================================
if st.session_state.app_view != "INTRO":
    st.markdown('<div class="bottom-nav"></div>', unsafe_allow_html=True)

    b_col1, b_col2, b_col3 = st.columns(3)

    with b_col1:
        btn_home_kind = "primary" if st.session_state.app_view == "HOME" else "secondary"
        if st.button("⌂ Home", type=btn_home_kind, use_container_width=True, key="btn_nav_home"):
            st.session_state.app_view = "HOME"
            st.rerun()

    with b_col2:
        btn_settings_kind = "primary" if st.session_state.app_view == "HEAT_LOAD" else "secondary"
        if st.button("Load", type=btn_settings_kind, use_container_width=True, key="btn_nav_settings"):
            st.session_state.app_view = "HEAT_LOAD"
            st.rerun()

    with b_col3:
        btn_analysis_kind = "primary" if st.session_state.app_view in ("RESULTS", "COMPARE") else "secondary"
        if st.button("Analysis", type=btn_analysis_kind, use_container_width=True, key="btn_nav_analysis"):
            if st.session_state.has_run_optimization and st.session_state.show_control_simulation:
                st.session_state.app_view = "COMPARE"
            else:
                st.session_state.app_view = "RESULTS"
            st.rerun()
