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

        /* Real smartphone frame built in CSS. The artwork itself no longer contains the outer frame. */
        .coollins-intro-shell {
            position: relative;
            width: min(438px, calc(100vw - 14px));
            margin: 0 auto;
            padding: 0;
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            overflow: visible;
        }

        .coollins-phone-device {
            position: relative;
            width: 100%;
            padding: 5px;
            box-sizing: border-box;
            border-radius: 58px;
            background:
                linear-gradient(145deg, #75b9ea 0%, #245b86 18%, #0b2239 46%, #1d4d72 76%, #6db7e8 100%);
            border: 1px solid rgba(180, 226, 255, 0.50);
            box-shadow:
                0 24px 52px rgba(0, 7, 18, 0.52),
                0 0 0 1px rgba(40, 111, 164, 0.52),
                inset 0 0 0 1px rgba(220, 245, 255, 0.18);
        }

        .coollins-phone-bezel {
            position: relative;
            width: 100%;
            padding: 7px;
            box-sizing: border-box;
            border-radius: 53px;
            background: #02070d;
            box-shadow:
                inset 0 0 0 1px rgba(255,255,255,0.07),
                inset 0 0 10px rgba(0,0,0,0.85);
        }

        .coollins-phone-screen {
            position: relative;
            width: 100%;
            overflow: hidden;
            border-radius: 46px;
            background: #071a2f;
            line-height: 0;
        }

        .coollins-phone-screen::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 46px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
            pointer-events: none;
            z-index: 35;
        }

        /* Hardware strip floats above the artwork so the image reaches the true rounded top corners. */
        .coollins-phone-hardware {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 56px;
            width: 100%;
            background: linear-gradient(180deg, rgba(9, 23, 37, 0.96) 0%, rgba(10, 28, 45, 0.82) 58%, rgba(10, 28, 45, 0.14) 100%);
            border-bottom: 1px solid rgba(95, 160, 210, 0.12);
            z-index: 30;
            pointer-events: none;
        }

        .coollins-dynamic-island {
            position: absolute;
            left: 50%;
            top: 9px;
            transform: translateX(-50%);
            width: 112px;
            height: 28px;
            border-radius: 18px;
            background: #000307;
            border: 1px solid rgba(62, 114, 151, 0.22);
            box-shadow: inset 0 -1px 0 rgba(255,255,255,0.025);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .coollins-dynamic-island .camera {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #123f68;
            box-shadow: inset 0 0 0 2px #071724, 0 0 4px rgba(60,160,230,.26);
        }

        .coollins-dynamic-island .speaker {
            width: 31px;
            height: 4px;
            border-radius: 999px;
            background: #29465b;
        }

        .coollins-intro-target {
            position: relative;
            width: 100%;
            margin: 0;
            line-height: 0;
            overflow: hidden;
            background: #071a2f;
            border-radius: 46px;
        }

        .coollins-intro-target img {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-drag: none;
            border-radius: 46px;
        }

        /* Physical side buttons, also pure CSS. */
        .coollins-side-button {
            position: absolute;
            z-index: -1;
            background: linear-gradient(180deg, #2d638d, #0d2b45);
            border: 1px solid rgba(143,204,244,.28);
            box-shadow: 0 2px 5px rgba(0,0,0,.35);
        }
        .coollins-side-left-1 {
            left: -4px; top: 19%; width: 4px; height: 42px;
            border-radius: 4px 0 0 4px;
        }
        .coollins-side-left-2 {
            left: -4px; top: 25%; width: 4px; height: 62px;
            border-radius: 4px 0 0 4px;
        }
        .coollins-side-right {
            right: -4px; top: 24%; width: 4px; height: 92px;
            border-radius: 0 4px 4px 0;
        }

        /* START hotspot after cropping the old pre-rendered frame from the artwork. */
        .coollins-intro-enter {
            position: absolute;
            left: 28.0%;
            top: 83.4%;
            width: 42.7%;
            height: 6.7%;
            display: block;
            border-radius: 999px;
            cursor: pointer;
            text-decoration: none !important;
            background: rgba(0,0,0,0.001);
            z-index: 40;
            outline: none;
            -webkit-tap-highlight-color: transparent;
        }

        .coollins-intro-enter:focus-visible {
            outline: 2px solid #38bdf8;
            outline-offset: -4px;
        }

        @media (max-width: 480px) {
            .block-container {
                max-width: 100% !important;
                padding-left: 0.35rem !important;
                padding-right: 0.35rem !important;
            }
            .coollins-intro-shell {
                width: min(438px, calc(100vw - 8px));
            }
            .coollins-phone-device { border-radius: 52px; }
            .coollins-phone-bezel { border-radius: 48px; }
            .coollins-phone-screen { border-radius: 42px; }
            .coollins-phone-screen::before { border-radius: 42px; }
            .coollins-intro-target,
            .coollins-intro-target img { border-radius: 42px; }
            .coollins-phone-hardware { height: 52px; }
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
        f'<div class="coollins-phone-device">'
        f'<span class="coollins-side-button coollins-side-left-1"></span>'
        f'<span class="coollins-side-button coollins-side-left-2"></span>'
        f'<span class="coollins-side-button coollins-side-right"></span>'
        f'<div class="coollins-phone-bezel">'
        f'<div class="coollins-phone-screen">'
        f'<div class="coollins-phone-hardware">'
        f'<div class="coollins-dynamic-island"><span class="camera"></span><span class="speaker"></span></div>'
        f'</div>'
        f'<div class="coollins-intro-target">'
        f'<img src="data:image/webp;base64,{INTRO_IMAGE_WEBP_B64}" '
        f'alt="COOLLINS AI Smart Cooling Optimizer 소개 화면" />'
        f'<a class="coollins-intro-enter" href="?enter=1" target="_self" '
        f'aria-label="START" title="START"></a>'
        f'</div></div></div></div></div>'
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
