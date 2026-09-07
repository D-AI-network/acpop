from __future__ import annotations

# CFD_RETRIEVAL_FINAL_V4 = 2026-09-03
# UI_REFINEMENT_BUILD = 2026-09-03-v31
# Robust repo-root CFD ZIP auto-discovery (dp*.csv archive detection)

# CFD_RETRIEVAL_BUILD = 2026-09-03-v1_NEAREST_200_REAL_CASES
# FACTOR_UI_BUILD = 2026-09-04-v69
# COMPARE_ZONE_VIEW_BUILD = 2026-09-07-v6_REMOVE_SUMMARY_HOME_TEMP30

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
            border-radius: 34px;
        }

        .coollins-intro-target img {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-drag: none;
            border-radius: 34px;
        }

        /* Transparent real click target placed exactly over the START button in the artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 28.2%;
            top: 87.0%;
            width: 43.6%;
            height: 9.2%;
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
