from __future__ import annotations

# CFD_RETRIEVAL_FINAL_V4 = 2026-09-03
# UI_REFINEMENT_BUILD = 2026-09-03-v31
# Robust repo-root CFD ZIP auto-discovery (dp*.csv archive detection)

# CFD_RETRIEVAL_BUILD = 2026-09-03-v1_NEAREST_200_REAL_CASES
# FACTOR_UI_BUILD = 2026-09-04-v69
# COMPARE_ZONE_VIEW_BUILD = 2026-09-07-v7_NEW_INTRO_IMAGE

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
    "Case_Info_MIX198T40_PSEUDO200_trainonly.xlsx",
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
INTRO_IMAGE_PNG_B64 = """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"""

if st.session_state.app_view == "INTRO":
    st.markdown(
        """
        <style>
        /* INTRO: restore a darker navy page and add a mobile-app shell. */
        .stApp {
            background: linear-gradient(180deg, #061427 0%, #071a2f 100%) !important;
        }

        .block-container {
            max-width: 430px !important;
            padding: 0.15rem 0.35rem 0.45rem 0.35rem !important;
            margin: 0.15rem auto 0.35rem auto !important;
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
            max-width: 410px;
            margin: 0 auto;
            padding: 0;
            background: transparent;
            border: none;
            border-radius: 0;
            box-shadow: none;
            overflow: visible;
        }

        .coollins-intro-shell .phone-notch {
            display: none !important;
        }

        .coollins-intro-target {
            position: relative;
            width: 100%;
            margin: 0 auto;
            line-height: 0;
            overflow: hidden;
            background: transparent;
            border-radius: 0;
        }

        .coollins-intro-target img {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-drag: none;
            border-radius: 0;
        }

        /* Transparent real click target placed exactly over the button in the artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 26.5%;
            top: 90.9%;
            width: 47.0%;
            height: 7.7%;
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
        f'<img src="data:image/png;base64,{INTRO_IMAGE_PNG_B64}" '
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

# 사용자 입력 현재온도에 Current Field 평균을 맞춤
requested_current_temp = float(st.session_state.current_temp_query)
retrieved_mean_temp = float(np.nanmean(current_temp_nodes))
temp_offset = requested_current_temp - retrieved_mean_temp

current_temp_nodes = current_temp_nodes + temp_offset
avg_room_temp = requested_current_temp

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
                            current_coords,
                            current_temp_nodes,
                            st.session_state.z_plane,
                            grid_len_axis,
                            grid_wid_axis,
                        ),
                        dtype=np.float32,
                    ),
                    "field_current_coords": np.asarray(current_coords, dtype=np.float32),
                    "field_current_temp_nodes": np.asarray(current_temp_nodes, dtype=np.float32),
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

    before_spread = float(
    np.nanmax(_before_zone_means) - np.nanmin(_before_zone_means)
    )
    
    after_spread = float(
        np.nanmax(_after_zone_means) - np.nanmin(_after_zone_means)
    )

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
