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
            padding: 0;
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            overflow: visible;
        }

        /* The third reference image already contains the navy phone frame + top notch. */
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

        /* Real click target aligned to START in the third reference artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 31.5%;
            top: 83.7%;
            width: 40.0%;
            height: 6.2%;
            display: block;
            border-radius: 999px;
            cursor: pointer;
            text-decoration: none !important;
            background: rgba(0,0,0,0.001);
            z-index: 10;
            outline: none;
            -webkit-tap-highlight-color: transparent;
        }

        .coollins-intro-enter:focus-visible {
            outline: 2px solid #38bdf8;
            outline-offset: -4px;
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
    card_w = 0.265 * x_span
    card_h = 0.235 * y_span

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
        name_y = ay + 0.32 * card_h
        drop_y = ay + 0.09 * card_h
        divider_y = ay - 0.10 * card_h
        temp_y = ay - 0.22 * card_h
        dev_y = ay - 0.38 * card_h

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
            font=dict(size=10.5, color="#eefaff"),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=drop_y,
            text=f"<b>↓ {drop:.1f}°C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=20, color=zone_color),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=temp_y,
            text=f"<b>{b:.1f} → {a:.1f}°C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=12, color="#ffffff"),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=dev_y,
            text=f"목표 편차 {bdev:.1f} → {adev:.1f}°C",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=8.5, color="#cfe5f3"),
        )

        # Transparent square target makes the visible zone card itself clickable too.
        fig.add_trace(
            go.Scatter(
                x=[ax],
                y=[ay],
                mode="markers",
                marker=dict(
                    size=104,
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
                position:absolute; z-index:50; width:198px; max-width:calc(100% - 24px);
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
            const prW = Math.min(198, wr.width - 24);

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
        plot.on('plotly_unhover', () => {{ if (!activeZone) clearHover(); }});
        plot.on('plotly_click', (data) => {{
            if (!data || !data.points || !data.points.length) return;
            const zone = traceZone(data.points[0].data);
            if (!zone) return;
            const ev = data.event || null;
            showPanel(zone, ev);
            setZoneHover(zone);
        }});

        // Fallback hit-test for clicks on Plotly annotations/shapes. Plotly does not
        // always emit plotly_click when the visible text card itself receives the pointer.
        function nearestIndex(arr, value) {{
            if (!Array.isArray(arr) || !arr.length || !Number.isFinite(value)) return -1;
            let best = 0;
            let bestDist = Math.abs(Number(arr[0]) - value);
            for (let i = 1; i < arr.length; i++) {{
                const d = Math.abs(Number(arr[i]) - value);
                if (d < bestDist) {{ best = i; bestDist = d; }}
            }}
            return best;
        }}

        function zoneFromPointer(ev) {{
            if (!ev || !plot) return null;
            const rect = plot.getBoundingClientRect();
            if (!rect.width || !rect.height) return null;
            const px = ev.clientX - rect.left;
            const py = ev.clientY - rect.top;
            if (px < 0 || py < 0 || px > rect.width || py > rect.height) return null;

            // First try the actual zone heatmap data, so the irregular zone boundary is respected.
            try {{
                const xa = plot._fullLayout && plot._fullLayout.xaxis;
                const ya = plot._fullLayout && plot._fullLayout.yaxis;
                const xv = xa && typeof xa.p2d === 'function' ? xa.p2d(px) : NaN;
                const yv = ya && typeof ya.p2d === 'function' ? ya.p2d(py) : NaN;
                if (Number.isFinite(xv) && Number.isFinite(yv)) {{
                    for (let i = 0; i < plot.data.length; i++) {{
                        const tr = plot.data[i];
                        if (!tr || !tr.meta || tr.meta.role !== 'zone_fill') continue;
                        const xi = nearestIndex(tr.x, xv);
                        const yi = nearestIndex(tr.y, yv);
                        if (xi < 0 || yi < 0 || !tr.z || !tr.z[yi]) continue;
                        const zv = Number(tr.z[yi][xi]);
                        if (Number.isFinite(zv)) {{
                            const znum = Number(tr.meta.zone_number || 0);
                            if (znum >= 1 && znum <= 4) return znum;
                        }}
                    }}
                }}
            }} catch (err) {{ /* use geometric fallback below */ }}

            // Geometric fallback mirrors the stable 4-zone layout used by the demo.
            const nx = px / rect.width;
            const ny = py / rect.height;
            if (nx <= 0.36 && ny <= 0.62) return 3;
            if (nx >= 0.37 && ny <= 0.46) return 1;
            if (nx >= 0.56 && ny >= 0.45) return 2;
            if (nx <= 0.58 && ny >= 0.43) return 4;
            return null;
        }}

        // Capture pointer events before annotation layers swallow them.
        wrap.addEventListener('pointerdown', (ev) => {{
            if (panel && panel.contains(ev.target)) return;
            const zone = zoneFromPointer(ev);
            if (!zone) return;
            showPanel(zone, ev);
            setZoneHover(zone);
        }}, true);

        // Keep hover highlighting working over annotations/cards too.
        wrap.addEventListener('pointermove', (ev) => {{
            if (panel && panel.contains(ev.target)) return;
            const zone = zoneFromPointer(ev);
            if (zone) setZoneHover(zone);
            else if (!activeZone) clearHover();
        }}, true);

        closeBtn.addEventListener('click', (ev) => {{
            ev.stopPropagation();
            panel.style.display = 'none';
            activeZone = null;
            clearHover();
        }});
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
