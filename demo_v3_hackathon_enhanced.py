from __future__ import annotations

# CFD_RETRIEVAL_FINAL_V4 = 2026-09-03
# UI_REFINEMENT_BUILD = 2026-09-03-v31
# Robust repo-root CFD ZIP auto-discovery (dp*.csv archive detection)

# CFD_RETRIEVAL_BUILD = 2026-09-03-v1_NEAREST_200_REAL_CASES
# FACTOR_UI_BUILD = 2026-09-04-v69
# COMPARE_ZONE_VIEW_BUILD = 2026-09-07-v44_NEW_INTRO_CONNECTED

# COOLING_FACTORS_BUILD = 2026-09-03-v20

# SENSOR_RADAR_ROUNDED_BUILD = 2026-09-03-v12

import base64
import io
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion

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
            max-width: 490px !important;
            padding: 0.35rem 0.55rem 0.8rem 0.55rem !important;
            margin: 0.25rem auto 0.6rem auto !important;
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
            max-width: 460px;
            margin: 0 auto;
            padding: 0;
            background: transparent;
            border: none;
            border-radius: 0;
            box-shadow: none;
            overflow: visible;
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
            background: transparent;
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

        /* Transparent real click target placed exactly over the START button in the artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 27.0%;
            top: 91.1%;
            width: 46.0%;
            height: 7.0%;
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
        f'<div class="coollins-intro-target">'
        f'<img src="data:image/webp;base64,{INTRO_IMAGE_WEBP_B64}" '
        f'alt="COOLLINS AI Smart Cooling Optimizer 소개 화면" />'
        f'<a class="coollins-intro-enter" href="?enter=1" target="_self" '
        f'aria-label="START" title="START"></a>'
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

    # Slightly relaxed near-feasible band for the user-facing demo.
    # Strict FEASIBLE thresholds above remain unchanged.
    near = (
        float(rec["zone_range_C"]) <= 2.40
        and float(rec["hot_fraction"]) <= 0.07
        and float(rec["cold_fraction"]) <= 0.07
        and float(rec["p95_temp_C"]) <= p95_limit + 0.40
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



def field_view_selector(key: str, include_zone: bool = False) -> str:
    """Small field-view selector. Results can optionally expose a 4-zone summary."""
    options = ["3D", "2D", "ZONE"] if include_zone else ["3D", "2D"]

    if key not in st.session_state or st.session_state.get(key) not in options:
        st.session_state[key] = "3D"

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Field view",
            options=options,
            selection_mode="single",
            key=key,
            label_visibility="collapsed",
        )
        return selected or "3D"

    return st.radio(
        "Field view",
        options=options,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )



def _collapse_zone_ids_to_xy(coords_xyz, zone_ids):
    """Collapse 3D node-wise zone labels into a single zone id per XY location."""
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    zone_ids = np.asarray(zone_ids, dtype=np.int64).reshape(-1)
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] < 2 or len(coords_xyz) != len(zone_ids):
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=np.int64)

    xy = np.round(coords_xyz[:, :2], 6)
    unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
    xy_zone_ids = np.zeros(len(unique_xy), dtype=np.int64)

    for idx in range(len(unique_xy)):
        vals = zone_ids[inverse == idx]
        if len(vals) == 0:
            xy_zone_ids[idx] = -1
            continue
        labels, counts = np.unique(vals, return_counts=True)
        xy_zone_ids[idx] = int(labels[np.argmax(counts)])

    valid = xy_zone_ids >= 0
    return unique_xy[valid], xy_zone_ids[valid]



def make_zone_mean_map(
    zone_coords_xyz,
    zone_ids,
    zone_labels,
    before_zone_means,
    after_zone_means,
    target,
    height=335,
):
    """Render a dark zone-outline map that emphasizes temperature reduction per zone."""
    zone_xy, xy_zone_ids = _collapse_zone_ids_to_xy(zone_coords_xyz, zone_ids)
    if len(zone_xy) == 0:
        return go.Figure()

    xx, yy = np.meshgrid(grid_len_axis, grid_wid_axis)
    zone_index_lookup = {int(zid): idx for idx, zid in enumerate(zone_labels)}
    zone_index_per_xy = np.asarray(
        [zone_index_lookup.get(int(zid), -1) for zid in xy_zone_ids],
        dtype=float,
    )
    zone_grid_idx = griddata(zone_xy, zone_index_per_xy, (xx, yy), method="nearest")
    zone_grid_idx = np.asarray(zone_grid_idx, dtype=float)

    before_zone_means = np.asarray(before_zone_means, dtype=float)
    after_zone_means = np.asarray(after_zone_means, dtype=float)
    drop_values = np.maximum(0.0, before_zone_means - after_zone_means)

    # Requested fixed color assignment:
    # Zone 1 = orange, Zone 2 = sky blue, Zone 3 = green, Zone 4 = purple
    zone_palette = {
        0: "#ffad47",  # zone 1
        1: "#45d2ff",  # zone 2
        2: "#54e39b",  # zone 3
        3: "#a86bff",  # zone 4
    }

    fig = go.Figure()

    # Navy room background so only the zone outlines carry color.
    room_bg = np.zeros_like(zone_grid_idx, dtype=float)
    fig.add_trace(
        go.Heatmap(
            z=room_bg,
            x=grid_len_axis,
            y=grid_wid_axis,
            colorscale=[[0.0, "#0a2d4b"], [1.0, "#0a2d4b"]],
            showscale=False,
            hoverinfo="skip",
            meta={"role": "room_background"},
        )
    )

    # Interactive hover layers. They are almost invisible at rest, but the custom
    # Plotly JS component raises their opacity when the mouse enters a zone.
    # This gives each irregular zone a soft fill without changing the base navy map.
    for idx, zid in enumerate(zone_labels):
        zone_bool = np.isclose(zone_grid_idx, float(idx), atol=0.49)
        zone_color = zone_palette.get(idx, "#aee4ff")
        hover_fill = np.where(zone_bool, 1.0, np.nan)
        hover_custom = np.where(zone_bool, float(idx + 1), np.nan)
        fig.add_trace(
            go.Heatmap(
                z=hover_fill,
                x=grid_len_axis,
                y=grid_wid_axis,
                customdata=hover_custom,
                colorscale=[[0.0, zone_color], [1.0, zone_color]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                opacity=0.012,
                hoverongaps=False,
                hovertemplate=(
                    f"<b>ZONE {idx + 1}</b><br>"
                    "클릭하여 상세 분석 보기"
                    "<extra></extra>"
                ),
                meta={"role": "zone_fill", "zone_number": int(idx + 1)},
                name=f"zone_fill_{idx + 1}",
            )
        )

    # Draw a CLOSED outline for every zone.  We erode each zone by one grid cell
    # before contouring it.  That places the outline slightly INSIDE its own zone,
    # so shared borders appear as two parallel colored lines instead of one zone
    # overwriting the other (the mockup-style separation the demo needs).
    for idx, zid in enumerate(zone_labels):
        zone_bool = np.isclose(zone_grid_idx, float(idx), atol=0.49)
        inset_bool = binary_erosion(zone_bool, iterations=1, border_value=0)
        if not np.any(inset_bool):
            inset_bool = zone_bool
        zone_mask = inset_bool.astype(float)
        zone_color = zone_palette.get(idx, "#aee4ff")

        # Soft neon halo.
        fig.add_trace(
            go.Contour(
                z=zone_mask,
                x=grid_len_axis,
                y=grid_wid_axis,
                showscale=False,
                hoverinfo="skip",
                contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlines=True),
                line=dict(color=zone_color, width=9.0),
                opacity=0.17,
                meta={"role": "zone_outline", "zone_number": int(idx + 1)},
                name=f"zone_glow_{idx + 1}",
            )
        )
        # Clear zone perimeter.
        fig.add_trace(
            go.Contour(
                z=zone_mask,
                x=grid_len_axis,
                y=grid_wid_axis,
                showscale=False,
                hoverinfo="skip",
                contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlines=True),
                line=dict(color=zone_color, width=3.5),
                opacity=1.0,
                meta={"role": "zone_outline", "zone_number": int(idx + 1)},
                name=f"zone_outline_{idx + 1}",
            )
        )

    # Card positions follow zone centroids with small manual nudges for readability.
    centroids = {}
    for idx, zid in enumerate(zone_labels):
        mask = xy_zone_ids == int(zid)
        if np.any(mask):
            centroids[idx] = [
                float(np.nanmean(zone_xy[mask, 0])),
                float(np.nanmean(zone_xy[mask, 1])),
            ]
        else:
            centroids[idx] = [
                float(np.nanmean(grid_len_axis)),
                float(np.nanmean(grid_wid_axis)),
            ]

    offsets = {
        0: (0.45, 0.45),   # zone 1
        1: (0.55, -0.40),  # zone 2
        2: (-0.55, 0.25),  # zone 3
        3: (-0.25, -0.35), # zone 4
    }

    x_span = float(grid_len_axis.max() - grid_len_axis.min())
    y_span = float(grid_wid_axis.max() - grid_wid_axis.min())
    card_w = 0.30 * x_span
    card_h = 0.28 * y_span

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    x_lo = float(grid_len_axis.min()) + card_w * 0.55
    x_hi = float(grid_len_axis.max()) - card_w * 0.55
    y_lo = float(grid_wid_axis.min()) + card_h * 0.55
    y_hi = float(grid_wid_axis.max()) - card_h * 0.55

    for idx, zid in enumerate(zone_labels):
        zone_color = zone_palette.get(idx, "#aee4ff")
        cx, cy = centroids[idx]
        dx, dy = offsets.get(idx, (0.0, 0.0))
        ax = _clamp(cx + dx, x_lo, x_hi)
        ay = _clamp(cy + dy, y_lo, y_hi)
        half_w = card_w / 2.0
        half_h = card_h / 2.0

        fig.add_shape(
            type="rect",
            x0=ax - half_w,
            x1=ax + half_w,
            y0=ay - half_h,
            y1=ay + half_h,
            fillcolor="rgba(8, 31, 52, 0.88)",
            line=dict(color="rgba(126, 190, 225, 0.30)", width=1.2),
            layer="above",
        )
        # Small zone-color accent bar, matching the visual mockup.
        fig.add_shape(
            type="line",
            x0=ax - half_w + 0.10,
            x1=ax - half_w + 0.10,
            y0=ay + half_h - 0.12 * card_h,
            y1=ay + half_h - 0.30 * card_h,
            line=dict(color=zone_color, width=8),
            layer="above",
        )

        b = float(before_zone_means[idx])
        a = float(after_zone_means[idx])
        drop = float(drop_values[idx])
        bdev = abs(b - float(target))
        adev = abs(a - float(target))

        # Layout is expressed as fractions of card height to prevent text collisions
        # when Streamlit changes the plot's rendered pixel size.
        name_y = ay + 0.34 * card_h
        drop_y = ay + 0.11 * card_h
        divider_y = ay - 0.08 * card_h
        temp_y = ay - 0.20 * card_h
        dev_y = ay - 0.36 * card_h

        fig.add_shape(
            type="line",
            x0=ax - half_w + 0.16,
            x1=ax + half_w - 0.16,
            y0=divider_y,
            y1=divider_y,
            line=dict(color="rgba(214,238,255,0.34)", width=1.1),
            layer="above",
        )

        fig.add_annotation(
            x=ax - half_w + 0.30,
            y=name_y,
            text=f"<b>ZONE {idx + 1}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=12, color="#eefaff"),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=drop_y,
            text=f"<b>↓ {drop:.1f}°C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=24, color=zone_color),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=temp_y,
            text=f"<b>{b:.1f} → {a:.1f}°C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=14, color="#ffffff"),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=dev_y,
            text=f"목표 편차 {bdev:.1f} → {adev:.1f}°C",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=9.5, color="#cfe5f3"),
        )

        # Transparent square target makes the visible zone card itself clickable too.
        fig.add_trace(
            go.Scatter(
                x=[ax],
                y=[ay],
                mode="markers",
                marker=dict(
                    size=88,
                    symbol="square",
                    color="rgba(255,255,255,0.001)",
                    line=dict(width=0),
                ),
                customdata=[[int(idx + 1)]],
                hovertemplate=(
                    f"<b>ZONE {idx + 1}</b><br>"
                    "클릭하여 상세 분석 보기"
                    "<extra></extra>"
                ),
                meta={"role": "zone_card_target", "zone_number": int(idx + 1)},
                showlegend=False,
                name=f"zone_card_target_{idx + 1}",
            )
        )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            range=[float(grid_len_axis.min()), float(grid_len_axis.max())],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[float(grid_wid_axis.min()), float(grid_wid_axis.max())],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True,
        ),
    )
    return fig



def _render_interactive_zone_map(zone_fig, before_zone_means, after_zone_means, target, height=365):
    """Render Plotly in an HTML component with true hover-fill and click popup behavior."""
    before_zone_means = np.asarray(before_zone_means, dtype=float)
    after_zone_means = np.asarray(after_zone_means, dtype=float)
    palette = {1: "#ffad47", 2: "#45d2ff", 3: "#54e39b", 4: "#a86bff"}

    zone_data = {}
    for zone_number in range(1, min(4, len(before_zone_means), len(after_zone_means)) + 1):
        idx = zone_number - 1
        before = float(before_zone_means[idx])
        after = float(after_zone_means[idx])
        temp_drop = max(0.0, before - after)
        before_dev = abs(before - float(target))
        after_dev = abs(after - float(target))
        dev_drop = max(0.0, before_dev - after_dev)
        zone_data[str(zone_number)] = {
            "zone": zone_number,
            "color": palette[zone_number],
            "before": round(before, 2),
            "after": round(after, 2),
            "drop": round(temp_drop, 2),
            "beforeDev": round(before_dev, 2),
            "afterDev": round(after_dev, 2),
            "devDrop": round(dev_drop, 2),
            "target": round(float(target), 2),
        }

    # Keep the map itself free of Plotly toolbars; all interaction is direct hover/click.
    plot_html = zone_fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
        },
    )
    zone_json = json.dumps(zone_data, ensure_ascii=False)

    html = f"""
    <div id="zone-interactive-wrap" style="position:relative;width:100%;height:{int(height)}px;overflow:hidden;border-radius:20px;background:#0a2d4b;">
        <style>
            #zone-interactive-wrap .plotly-graph-div {{ width:100% !important; height:{int(height)}px !important; }}
            #zone-detail-float {{
                position:absolute; z-index:50; width:210px; max-width:calc(100% - 24px);
                display:none; border-radius:16px; padding:13px 13px 12px 13px;
                background:linear-gradient(150deg,rgba(6,30,51,.98),rgba(10,47,75,.98));
                border:1px solid #45d2ff; box-shadow:0 16px 34px rgba(0,8,20,.46);
                backdrop-filter:blur(8px); color:white; font-family:'Noto Sans KR','Inter',sans-serif;
                pointer-events:auto;
            }}
            #zone-detail-float .zclose {{
                position:absolute; right:8px; top:7px; width:24px; height:24px; border:0; border-radius:50%;
                color:#cfeafb; background:rgba(255,255,255,.08); font-size:15px; line-height:24px; cursor:pointer;
            }}
            #zone-detail-float .zhead {{display:flex;align-items:center;gap:8px;margin-bottom:10px;}}
            #zone-detail-float .zbar {{width:5px;height:25px;border-radius:99px;background:#45d2ff;box-shadow:0 0 12px rgba(69,210,255,.5);}}
            #zone-detail-float .zname {{font-size:16px;font-weight:900;color:#f4fbff;}}
            #zone-detail-float .ztarget {{font-size:9px;font-weight:700;color:#91bbd1;margin-top:1px;}}
            #zone-detail-float .ztemp {{font-size:18px;font-weight:900;color:#fff;letter-spacing:-.4px;}}
            #zone-detail-float .zarrow {{color:#68d9ff;padding:0 4px;}}
            #zone-detail-float .zdrop {{font-size:24px;font-weight:950;margin-top:2px;}}
            #zone-detail-float .zgrid {{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px;}}
            #zone-detail-float .zmini {{background:rgba(255,255,255,.055);border-radius:10px;padding:8px 8px 7px 8px;}}
            #zone-detail-float .zlabel {{font-size:8.5px;color:#91bbd1;font-weight:750;margin-bottom:3px;}}
            #zone-detail-float .zvalue {{font-size:12px;color:#f5fbff;font-weight:850;white-space:nowrap;}}
        </style>
        {plot_html}
        <div id="zone-detail-float">
            <button class="zclose" id="zone-detail-close">×</button>
            <div class="zhead"><span class="zbar" id="zone-detail-bar"></span><div><div class="zname" id="zone-detail-name"></div><div class="ztarget" id="zone-detail-target"></div></div></div>
            <div class="ztemp"><span id="zone-before"></span><span class="zarrow">→</span><span id="zone-after"></span></div>
            <div class="zdrop" id="zone-drop"></div>
            <div class="zgrid">
                <div class="zmini"><div class="zlabel">목표 편차</div><div class="zvalue" id="zone-dev"></div></div>
                <div class="zmini"><div class="zlabel">편차 감소</div><div class="zvalue" id="zone-devdrop"></div></div>
            </div>
        </div>
    </div>
    <script>
    (() => {{
        const zoneData = {zone_json};
        const wrap = document.getElementById('zone-interactive-wrap');
        const plot = wrap.querySelector('.plotly-graph-div');
        const panel = document.getElementById('zone-detail-float');
        const closeBtn = document.getElementById('zone-detail-close');
        let activeZone = null;
        let hoverZone = null;

        function traceZone(trace) {{
            if (!trace || !trace.meta) return null;
            const z = Number(trace.meta.zone_number || 0);
            return z >= 1 && z <= 4 ? z : null;
        }}

        function fillIndexForZone(zone) {{
            for (let i = 0; i < plot.data.length; i++) {{
                const tr = plot.data[i];
                if (tr && tr.meta && tr.meta.role === 'zone_fill' && Number(tr.meta.zone_number) === Number(zone)) return i;
            }}
            return -1;
        }}

        function setZoneHover(zone) {{
            if (hoverZone === zone) return;
            for (let z = 1; z <= 4; z++) {{
                const idx = fillIndexForZone(z);
                if (idx >= 0) Plotly.restyle(plot, {{opacity: (z === zone ? 0.20 : 0.012)}}, [idx]);
            }}
            hoverZone = zone;
        }}

        function clearHover() {{
            for (let z = 1; z <= 4; z++) {{
                const idx = fillIndexForZone(z);
                if (idx >= 0) Plotly.restyle(plot, {{opacity: 0.012}}, [idx]);
            }}
            hoverZone = null;
        }}

        function showPanel(zone, ev) {{
            const d = zoneData[String(zone)];
            if (!d) return;
            activeZone = zone;
            panel.style.borderColor = d.color;
            document.getElementById('zone-detail-bar').style.background = d.color;
            document.getElementById('zone-detail-bar').style.boxShadow = `0 0 12px ${{d.color}}`;
            document.getElementById('zone-detail-name').textContent = `ZONE ${{zone}} 상세 분석`;
            document.getElementById('zone-detail-target').textContent = `목표 온도 ${{d.target.toFixed(1)}}°C 기준`;
            document.getElementById('zone-before').textContent = `${{d.before.toFixed(1)}}°C`;
            document.getElementById('zone-after').textContent = `${{d.after.toFixed(1)}}°C`;
            const drop = document.getElementById('zone-drop');
            drop.textContent = `↓ ${{d.drop.toFixed(1)}}°C`;
            drop.style.color = d.color;
            document.getElementById('zone-dev').textContent = `${{d.beforeDev.toFixed(1)}} → ${{d.afterDev.toFixed(1)}}°C`;
            const dd = document.getElementById('zone-devdrop');
            dd.textContent = `↓ ${{d.devDrop.toFixed(1)}}°C`;
            dd.style.color = '#79e6b4';
            panel.style.display = 'block';

            const wr = wrap.getBoundingClientRect();
            const prW = Math.min(210, wr.width - 24);

            // The panel must be measured AFTER it becomes visible.  The old code
            // assumed a fixed 150px height, so clicks in the lower zones could
            // push the actual ~200px panel below the iframe and clip its bottom.
            const panelRect = panel.getBoundingClientRect();
            const prH = Math.min(panelRect.height || 205, wr.height - 16);

            let x = wr.width - prW - 10;
            let y = 8;
            if (ev && Number.isFinite(ev.clientX) && Number.isFinite(ev.clientY)) {{
                const clickX = ev.clientX - wr.left;
                const clickY = ev.clientY - wr.top;

                // Prefer the right side of the click; flip left when needed.
                x = clickX + 14;
                if (x + prW > wr.width - 8) x = clickX - prW - 14;
                x = Math.max(8, Math.min(x, wr.width - prW - 8));

                // Prefer below the click for upper zones.  For lower zones,
                // automatically flip above so the whole mini window remains visible.
                y = clickY + 12;
                if (y + prH > wr.height - 8) y = clickY - prH - 12;
                y = Math.max(8, Math.min(y, wr.height - prH - 8));
            }}
            panel.style.left = `${{x}}px`;
            panel.style.top = `${{y}}px`;
        }}

        plot.on('plotly_hover', (data) => {{
            if (!data || !data.points || !data.points.length) return;
            const zone = traceZone(data.points[0].data);
            if (zone) setZoneHover(zone);
        }});
        plot.on('plotly_unhover', () => clearHover());
        plot.on('plotly_click', (data) => {{
            if (!data || !data.points || !data.points.length) return;
            const zone = traceZone(data.points[0].data);
            if (!zone) return;
            const ev = data.event || null;
            showPanel(zone, ev);
            setZoneHover(zone);
        }});
        closeBtn.addEventListener('click', () => {{ panel.style.display = 'none'; activeZone = null; }});
        wrap.addEventListener('mouseleave', () => {{ if (!activeZone) clearHover(); }});
    }})();
    </script>
    """
    components.html(html, height=int(height), scrolling=False)


def _extract_zone_from_plotly_selection(event):
    """Return a 1-based zone number from Streamlit Plotly selection state."""
    if event is None:
        return None

    selection = None
    try:
        selection = event.selection
    except Exception:
        pass
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")

    points = None
    if selection is not None:
        try:
            points = selection.points
        except Exception:
            pass
        if points is None and isinstance(selection, dict):
            points = selection.get("points")

    if not points:
        return None

    point = points[-1]
    customdata = None
    if isinstance(point, dict):
        customdata = point.get("customdata")
    else:
        try:
            customdata = point.customdata
        except Exception:
            pass

    if isinstance(customdata, (list, tuple, np.ndarray)) and len(customdata):
        customdata = customdata[0]

    try:
        zone_number = int(customdata)
    except Exception:
        return None

    return zone_number if 1 <= zone_number <= 4 else None


def _zone_detail_panel_html(zone_number, before_zone_means, after_zone_means, target):
    """Build the compact detail card shown inside the zone dialog."""
    idx = int(zone_number) - 1
    before_zone_means = np.asarray(before_zone_means, dtype=float)
    after_zone_means = np.asarray(after_zone_means, dtype=float)
    if idx < 0 or idx >= len(before_zone_means) or idx >= len(after_zone_means):
        return ""

    palette = {1: "#ffad47", 2: "#45d2ff", 3: "#54e39b", 4: "#a86bff"}
    accent = palette.get(int(zone_number), "#7fdcff")
    before = float(before_zone_means[idx])
    after = float(after_zone_means[idx])
    temp_drop = max(0.0, before - after)
    before_dev = abs(before - float(target))
    after_dev = abs(after - float(target))
    dev_drop = max(0.0, before_dev - after_dev)

    return textwrap.dedent(
        f"""
        <div style="background:linear-gradient(150deg,rgba(8,35,58,.98),rgba(12,48,76,.96));border:1px solid {accent};border-radius:20px;padding:18px 16px 16px 16px;box-shadow:0 10px 28px rgba(0,15,30,.24);">
            <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px;">
                <span style="display:inline-block;width:6px;height:28px;border-radius:99px;background:{accent};box-shadow:0 0 12px {accent};"></span>
                <div>
                    <div style="color:#f3fbff;font-size:20px;font-weight:850;line-height:1.05;">ZONE {zone_number}</div>
                    <div style="color:#9dc6dc;font-size:11px;font-weight:700;margin-top:4px;">목표 온도 {float(target):.1f}°C 기준</div>
                </div>
            </div>
            <div style="background:rgba(6,28,48,.55);border-radius:15px;padding:14px;margin-bottom:10px;">
                <div style="color:#9fc5d9;font-size:11px;font-weight:700;margin-bottom:5px;">평균 온도 변화</div>
                <div style="color:#ffffff;font-size:24px;font-weight:900;letter-spacing:-.5px;">{before:.1f}°C <span style="color:#6fdcff;">→</span> {after:.1f}°C</div>
                <div style="color:{accent};font-size:25px;font-weight:900;margin-top:5px;">↓ {temp_drop:.1f}°C</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;">
                <div style="background:rgba(6,28,48,.55);border-radius:14px;padding:12px;">
                    <div style="color:#9fc5d9;font-size:10px;font-weight:700;margin-bottom:5px;">목표 편차</div>
                    <div style="color:#f5fbff;font-size:16px;font-weight:850;">{before_dev:.1f} → {after_dev:.1f}°C</div>
                </div>
                <div style="background:rgba(6,28,48,.55);border-radius:14px;padding:12px;">
                    <div style="color:#9fc5d9;font-size:10px;font-weight:700;margin-bottom:5px;">편차 감소</div>
                    <div style="color:#7de8b5;font-size:16px;font-weight:850;">↓ {dev_drop:.1f}°C</div>
                </div>
            </div>
        </div>
        """
    ).strip()


def _show_zone_detail_popup(zone_number, before_zone_means, after_zone_means, target):
    """Show a modal dialog when supported; otherwise render an inline detail panel."""
    panel_html = _zone_detail_panel_html(zone_number, before_zone_means, after_zone_means, target)
    if not panel_html:
        return

    if hasattr(st, "dialog"):
        @st.dialog(f"ZONE {int(zone_number)} 상세 분석")
        def _zone_dialog():
            st.markdown(panel_html, unsafe_allow_html=True)
            if st.button("닫기", use_container_width=True, key=f"close_zone_detail_{int(zone_number)}"):
                st.session_state.selected_zone_detail = None
                st.rerun()
        _zone_dialog()
    else:
        st.markdown(f'<div class="section-title" style="margin-top:10px;">ZONE {int(zone_number)} 상세 분석</div>', unsafe_allow_html=True)
        st.markdown(panel_html, unsafe_allow_html=True)
        if st.button("상세 보기 닫기", use_container_width=True, key=f"close_zone_detail_inline_{int(zone_number)}"):
            st.session_state.selected_zone_detail = None
            st.rerun()

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



def make_2d_heatmap(grid_data, height=315, show_sensors=True, sensor_count=5, coords_xyz=None, temp_nodes=None, temp_max=32.0):
    """Classic top-down 2D temperature map used when the user selects 2D."""
    heatmap_data = np.asarray(grid_data, dtype=float)
    temp_max = float(temp_max)
    temp_ticks = [18, 21, 24, 27, 30] if temp_max <= 30.0 else [18, 22, 26, 30, 32]

    temp_scale = [
        [0.00, "#8ee7ff"],
        [0.14, "#63d6ff"],
        [0.28, "#41b8ff"],
        [0.42, "#1fc9d2"],
        [0.57, "#53dd84"],
        [0.72, "#c4e45a"],
        [0.86, "#ffae47"],
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
            zmax=temp_max,
            colorbar=dict(
                title=dict(text="°C", font=dict(size=10, color="#d9f3ff")),
                thickness=5,
                len=0.68,
                x=0.99,
                tickvals=temp_ticks,
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

def make_mobile_heatmap(grid_data, height=340, show_sensors=True, sensor_count=5, temp_max=32.0):
    """Interactive 3D spatial-temperature surface used by HOME and RESULTS."""
    surface_data = np.asarray(grid_data, dtype=float)
    temp_max = float(temp_max)
    temp_ticks = [18, 21, 24, 27, 30] if temp_max <= 30.0 else [18, 22, 26, 30, 32]

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            z=surface_data,
            x=grid_len_axis,
            y=grid_wid_axis,
            surfacecolor=surface_data,
            colorscale=[
                    [0.00, "#8ee7ff"],
                    [0.14, "#63d6ff"],
                    [0.28, "#41b8ff"],
                    [0.42, "#1fc9d2"],
                    [0.57, "#53dd84"],
                    [0.72, "#c4e45a"],
                    [0.86, "#ffae47"],
                    [1.00, "#e63a32"],
                ],
            cmin=18.0,
            cmax=temp_max,
            showscale=True,
            colorbar=dict(
                title=dict(text="°C", font=dict(size=10, color="#d9f3ff")),
                thickness=8,
                len=0.72,
                x=0.965,
                tickvals=[18, 22, 26, 30, 32],
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
                range=[18.0, temp_max],
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


def make_true_3d_field(coords_xyz, temp_nodes, height=390, max_points=2800, show_sensors=True, sensor_count=5, temp_max=32.0):
    """
    Clean 3D room-style temperature map.

    Instead of plotting the raw CFD node layout directly, interpolate the real CFD
    temperatures onto a regular XYZ lattice. This preserves the field pattern while
    producing the clean rectangular 3D map used in the UI reference.
    """
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    temp_nodes = np.asarray(temp_nodes, dtype=float).reshape(-1)
    temp_max = float(temp_max)
    temp_ticks = [18, 21, 24, 27, 30] if temp_max <= 30.0 else [18, 22, 26, 30, 32]

    valid = (
        coords_xyz.ndim == 2
        and coords_xyz.shape[1] >= 3
        and len(coords_xyz) == len(temp_nodes)
    )
    if not valid:
        return make_mobile_heatmap(field_current_grid, height=height, show_sensors=show_sensors, sensor_count=sensor_count, temp_max=temp_max)

    finite = np.isfinite(coords_xyz[:, :3]).all(axis=1) & np.isfinite(temp_nodes)
    coords = coords_xyz[finite, :3]
    temps = temp_nodes[finite]

    if len(coords) == 0:
        return make_mobile_heatmap(field_current_grid, height=height, show_sensors=show_sensors, sensor_count=sensor_count, temp_max=temp_max)

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
        [0.00, "#8ee7ff"],
        [0.14, "#63d6ff"],
        [0.28, "#41b8ff"],
        [0.42, "#1fc9d2"],
        [0.57, "#53dd84"],
        [0.72, "#c4e45a"],
        [0.86, "#ffae47"],
        [1.00, "#e63a32"],
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
                cmax=temp_max,
                opacity=0.82,
                colorbar=dict(
                    title=dict(text="°C", font=dict(size=11, color="#eefaff")),
                    thickness=5,
                    len=0.56,
                    x=0.992,
                    xpad=2,
                    tickvals=temp_ticks,
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
            home_fig = make_true_3d_field(current_coords, current_temp_nodes, height=410, show_sensors=False, temp_max=30.0)
        else:
            home_fig = make_2d_heatmap(field_current_grid, height=315, show_sensors=False, temp_max=30.0)

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

                # Current spatial imbalance = worst internal P95-P05 among the four XY thermal zones.
                _opt_zone_path = APP_ROOT / "thermal_zones_train140.npz"
                if not _opt_zone_path.exists():
                    raise FileNotFoundError(f"Thermal zone mask not found: {_opt_zone_path}")
                with np.load(_opt_zone_path) as _opt_zone_data:
                    if "zone_ids" not in _opt_zone_data:
                        raise KeyError("thermal_zones_train140.npz has no 'zone_ids'")
                    _opt_zone_ids = np.asarray(_opt_zone_data["zone_ids"], dtype=np.int64)
                    _opt_zone_coords = (
                        np.asarray(_opt_zone_data["coords"], dtype=np.float32)
                        if "coords" in _opt_zone_data
                        else None
                    )

                _current_opt_temp = np.asarray(matched_current["temp_c"], dtype=float)
                _current_opt_coords = np.asarray(matched_current["coords"], dtype=np.float32)
                if len(_opt_zone_ids) != len(_current_opt_temp):
                    raise ValueError(
                        f"Thermal zone/node count mismatch: zones={len(_opt_zone_ids)}, current={len(_current_opt_temp)}"
                    )
                if _opt_zone_coords is not None and (
                    _opt_zone_coords.shape != _current_opt_coords.shape
                    or not np.allclose(_opt_zone_coords, _current_opt_coords, atol=1e-6, rtol=0.0)
                ):
                    raise ValueError(
                        "thermal_zones_train140.npz node ordering/coordinates do not match current CFD field"
                    )
                _opt_zone_labels = np.sort(np.unique(_opt_zone_ids))
                if len(_opt_zone_labels) != 4:
                    raise ValueError(f"Expected 4 thermal zones, found {len(_opt_zone_labels)}")
                _current_zone_spreads = np.asarray([
                    np.nanpercentile(_current_opt_temp[_opt_zone_ids == _zid], 95)
                    - np.nanpercentile(_current_opt_temp[_opt_zone_ids == _zid], 5)
                    for _zid in _opt_zone_labels
                ], dtype=float)
                current_worst_zone_spread_c = float(np.nanmax(_current_zone_spreads))

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
                        baseline_zone_range_c=current_worst_zone_spread_c,
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

    # Spatial temperature spread: worst internal P95-P05 among the four data-derived XY thermal zones.
    _zone_path = APP_ROOT / "thermal_zones_train140.npz"
    if not _zone_path.exists():
        raise FileNotFoundError(f"Thermal zone mask not found: {_zone_path}")

    with np.load(_zone_path) as _zone_data:
        if "zone_ids" not in _zone_data:
            raise KeyError("thermal_zones_train140.npz has no 'zone_ids'")
        _zone_ids = np.asarray(_zone_data["zone_ids"], dtype=np.int64)
        _zone_coords = (
            np.asarray(_zone_data["coords"], dtype=np.float32)
            if "coords" in _zone_data
            else None
        )

    if len(_zone_ids) != len(result_current_nodes) or len(_zone_ids) != len(result_pred_nodes):
        raise ValueError(
            "Thermal zone/node count mismatch: "
            f"zones={len(_zone_ids)}, current={len(result_current_nodes)}, predicted={len(result_pred_nodes)}"
        )

    if _zone_coords is not None:
        if (
            _zone_coords.shape != result_current_coords.shape
            or not np.allclose(_zone_coords, result_current_coords.astype(np.float32), atol=1e-6, rtol=0.0)
            or _zone_coords.shape != result_pred_coords.shape
            or not np.allclose(_zone_coords, result_pred_coords.astype(np.float32), atol=1e-6, rtol=0.0)
        ):
            raise ValueError(
                "thermal_zones_train140.npz node ordering/coordinates do not match displayed fields"
            )

    _zone_labels = np.sort(np.unique(_zone_ids))
    if len(_zone_labels) != 4:
        raise ValueError(f"Expected 4 thermal zones, found {len(_zone_labels)}")

    _before_zone_spreads = np.asarray([
        np.nanpercentile(result_current_nodes[_zone_ids == _zid], 95)
        - np.nanpercentile(result_current_nodes[_zone_ids == _zid], 5)
        for _zid in _zone_labels
    ], dtype=float)
    _after_zone_spreads = np.asarray([
        np.nanpercentile(result_pred_nodes[_zone_ids == _zid], 95)
        - np.nanpercentile(result_pred_nodes[_zone_ids == _zid], 5)
        for _zid in _zone_labels
    ], dtype=float)

    # Zone view uses the exact same four training-zone masks as the optimizer.
    # Each card reports the zone mean before/after and its absolute distance from target.
    _before_zone_means = np.asarray([
        np.nanmean(result_current_nodes[_zone_ids == _zid])
        for _zid in _zone_labels
    ], dtype=float)
    _after_zone_means = np.asarray([
        np.nanmean(result_pred_nodes[_zone_ids == _zid])
        for _zid in _zone_labels
    ], dtype=float)
    _before_zone_target_dev = np.abs(_before_zone_means - target)
    _after_zone_target_dev = np.abs(_after_zone_means - target)

    before_spread = max(0.0, float(np.nanmax(_before_zone_spreads)))
    after_spread = max(0.0, float(np.nanmax(_after_zone_spreads)))

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

    # Keep the model's raw feasibility status for diagnostics, but do not turn it
    # into a pass/fail message on the demo screen. "완료" means the optimization
    # process finished; the numerical cards below still show the actual outcome.
    status = str(res.get("status", "INFEASIBLE"))
    status_text = "AI 냉방 최적화 완료"
    status_color = "#74e0a8"
    status_symbol = "✓"

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
        .zone-view-head {
            display:flex;
            align-items:flex-end;
            justify-content:space-between;
            gap:10px;
            margin:10px 0 10px 0;
        }
        .zone-view-title {
            color:#edf9ff;
            font-size:16px;
            font-weight:800;
        }
        .zone-view-sub {
            color:#8fb9d0;
            font-size:11px;
            font-weight:650;
            text-align:right;
            line-height:1.35;
        }
        .zone-grid {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:10px;
            margin:4px 0 14px 0;
        }
        .zone-card {
            min-width:0;
            border-radius:18px;
            padding:14px 13px 12px 13px;
            background:linear-gradient(150deg, rgba(11,43,70,.96), rgba(15,56,84,.90));
            border:1px solid rgba(126,203,240,.20);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
            position:relative;
            overflow:hidden;
        }
        .zone-card::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            bottom:0;
            width:4px;
            background:var(--zone-accent,#62d6ff);
        }
        .zone-card-top {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:6px;
            margin-bottom:8px;
        }
        .zone-name {
            color:#dff5ff;
            font-size:13px;
            font-weight:850;
        }
        .zone-mode {
            color:#8fc8e5;
            font-size:9px;
            font-weight:800;
            letter-spacing:.08em;
        }
        .zone-main-temp {
            color:#ffffff;
            font-size:25px;
            line-height:1.05;
            font-weight:850;
            margin-bottom:8px;
        }
        .zone-flow {
            display:grid;
            grid-template-columns:1fr auto 1fr;
            gap:5px;
            align-items:center;
            color:#d8edf8;
            font-size:11px;
            font-weight:750;
        }
        .zone-flow .after {
            text-align:right;
        }
        .zone-flow .arrow {
            color:#5fd6ff;
            font-size:15px;
            font-weight:900;
        }
        .zone-target-dev {
            margin-top:8px;
            padding-top:7px;
            border-top:1px solid rgba(171,222,246,.10);
            color:#9fc4d8;
            font-size:9.5px;
            line-height:1.35;
            font-weight:650;
        }
        .zone-target-dev strong {
            color:#7ce6b2;
            font-weight:850;
        }
        @media (max-width: 380px) {
            .zone-grid { gap:8px; }
            .zone-card { padding:12px 10px 10px 10px; }
            .zone-main-temp { font-size:22px; }
            .zone-flow { font-size:10px; }
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

    compare_view = field_view_selector("compare_map_view", include_zone=True)

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

    if compare_view == "ZONE":
        st.markdown(
            f'<div class="zone-view-head"><div class="zone-view-title">4-ZONE MAP</div><div class="zone-view-sub">BEFORE → AFTER<br>목표 {target:.1f}°C</div></div>',
            unsafe_allow_html=True,
        )
        zone_fig = make_zone_mean_map(
            result_current_coords,
            _zone_ids,
            _zone_labels,
            _before_zone_means,
            _after_zone_means,
            target=target,
            height=345,
        )
        _render_interactive_zone_map(
            zone_fig,
            _before_zone_means,
            _after_zone_means,
            target=target,
            height=365,
        )
    else:
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
