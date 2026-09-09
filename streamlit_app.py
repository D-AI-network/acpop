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
    page_icon="�꾬툘",
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

/* AI recommended HVAC setting �� compact visual 2x2 panel */
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


/* v24: make selected stage + endpoint labels about the same size as the 5�④퀎 helper text. */
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
   Force the SelectSlider's visible stage text (蹂댄넻/�믪쓬/留ㅼ슦 ��쓬/留ㅼ슦 �믪쓬)
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
   Applies to �좏깮媛�(蹂댄넻/�믪쓬/留ㅼ슦 �믪쓬) and endpoints(留ㅼ슦 ��쓬/留ㅼ슦 �믪쓬). */
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

/* Restore only the factor title: �몃� �댄솚寃� / �쒕쾭 諛쒖뿴 / �뚯쓽怨듦컙 / �낅Т怨듦컙 */
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
    653:  {"code": "S1", "name": "Sensor 1 쨌 Node 653",  "x_plot": 6.75, "y_plot": 2.75, "z": 1.50, "zone": "Core Sensor"},
    887:  {"code": "S2", "name": "Sensor 2 쨌 Node 887",  "x_plot": 2.75, "y_plot": 2.75, "z": 1.50, "zone": "Core Sensor"},
    1036: {"code": "S3", "name": "Sensor 3 쨌 Node 1036", "x_plot": 4.25, "y_plot": 1.75, "z": 2.50, "zone": "Core Sensor"},
    639:  {"code": "S4", "name": "Sensor 4 쨌 Node 639",  "x_plot": 1.25, "y_plot": 1.25, "z": 2.00, "zone": "Core Sensor"},
    1229: {"code": "S5", "name": "Sensor 5 쨌 Node 1229", "x_plot": 5.50, "y_plot": 1.75, "z": 2.00, "zone": "Core Sensor"},
}
ROA_NODE_IDS = list(ROA_NODES_META.keys())

# Validation-selected strict nested hierarchy:
# 5 �� 6 �� ... �� 14 �� 15
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
    "Case Info 200 DesignPoints - 理쒖쥌蹂�.xlsx",
    "Case Info 200 DesignPoints - 理쒖쥌蹂� (1).xlsx",
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
    """Return the validated 5��15 sensor order from the deployment NPZ."""
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
    5��15 active-sensor operating policy.

    One extra active sensor is enabled for each 0.5째C of absolute target
    deviation, with a hard minimum of 5 and a hard maximum of 15.

    Example for target 24째C:
      <24.5 �� 5
       24.5 �� 6
       25.0 �� 7
       25.5 �� 8
       26.0 �� 9
       26.5 �� 10
       27.0 �� 11
       27.5 �� 12
       28.0 �� 13
       28.5 �� 14
       29.0+ �� 15
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

# INTRO �대�吏��� �ㅼ젣 踰꾪듉 �곸뿭�� �꾨Ⅴ硫� ?enter=1 濡� �ㅼ뼱�듬땲��.
# �� 媛믪쓣 媛먯��� 湲곗〈 HOME �붾㈃�쇰줈 �대룞�⑸땲��.
if st.query_params.get("enter") == "1":
    st.session_state.app_view = "HOME"
    st.query_params.clear()

if "selected_dp" not in st.session_state:
    st.session_state.selected_dp = "DP 0"

if "z_plane" not in st.session_state:
    st.session_state.z_plane = 1.5
# HOME �붾㈃�먯꽌�� 痢≪젙 �믪씠 �좏깮�� �ъ슜�섏� �딄퀬 1.5m濡� 怨좎젙�⑸땲��.
st.session_state.z_plane = 1.5

if "target_temp" not in st.session_state:
    st.session_state.target_temp = 24.0

# User-described CURRENT room temperature used to retrieve the closest real CFD case.
# Start the demo at 28.0 째C. The user can still edit it afterwards.
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
st.session_state.policy = "Balanced (洹좏삎)"

if "heat_input_mode" not in st.session_state:
    st.session_state.heat_input_mode = "媛꾪렪 �④퀎"

for k, v in {"p_ext": "蹂댄넻", "p_meet": "蹂댄넻", "p_serv": "蹂댄넻", "p_work": "蹂댄넻"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# One-time migration: initialize the qualitative factors near the supplied DP 0 (Current)
# condition so the first HOME field starts from the official current CFD scenario.
if "cfd_retrieval_defaults_v1" not in st.session_state:
    st.session_state.p_ext = "留ㅼ슦 ��쓬"
    st.session_state.p_meet = "��쓬"
    st.session_state.p_serv = "留ㅼ슦 ��쓬"
    st.session_state.p_work = "��쓬"
    st.session_state.cfd_retrieval_defaults_v1 = True

if "has_run_optimization" not in st.session_state:
    st.session_state.has_run_optimization = False

# RESULTS �붾㈃�먯꽌 AI 異붿쿇 �쒖뼱�덉쓣 "�곸슜�� 蹂� 寃곌낵"瑜� 蹂댁뿬以꾩� �щ�.
# �ㅼ젣 BMS �꾩넚�� �꾨땲��, �좏깮�� �쒖뼱�덉쓣 PopField �덉륫 寃곌낵濡� �쒕��덉씠�섑빀�덈떎.
if "show_control_simulation" not in st.session_state:
    st.session_state.show_control_simulation = False

if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = {
        "status": "FEASIBLE",
        "vane": "Middle (M)",
        "flow": "40 CMM",
        "temp": "12 째C",
        "mean_temp": 23.8,
        "p95_temp": 24.5,
        "zone_spread": 1.42,
        "hot_fraction": 1.8,
        "cold_fraction": 0.5,
        "q_proxy": 13.8,
        "policy_used": "Balanced (洹좏삎)",
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
# INTRO SCREEN ASSET
# ============================================================
def _image_file_to_data_uri(path_str):
    """Return a data URI for a local image file."""
    from pathlib import Path
    import base64
    import mimetypes

    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _load_intro_image_data_uri():
    """Load the latest intro artwork without hard-embedding megabytes into the code."""
    candidate_paths = [
        "/mnt/data/荑⑤쭅_�ㅻ쭏�명솃_�먯뼱而�_��_愿묎퀬.png",
        "/mnt/data/a_clean_high_quality_ui_advertising_style_mockup.png",
        "/mnt/data/�붿씠��_濡쒓퀬��_�ㅻ쭏��_�먯뼱而�_��_愿묎퀬.png",
        os.path.join(os.path.dirname(__file__), "荑⑤쭅_�ㅻ쭏�명솃_�먯뼱而�_��_愿묎퀬.png"),
        os.path.join(os.path.dirname(__file__), "coollins_intro_latest.png"),
    ]
    for candidate in candidate_paths:
        uri = _image_file_to_data_uri(candidate)
        if uri:
            return uri
    return None



# Compact embedded fallback so the intro still works even if the image file is missing.
INTRO_IMAGE_FALLBACK_DATA_URI = "data:image/webp;base64,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"

INTRO_IMAGE_DATA_URI = _load_intro_image_data_uri() or INTRO_IMAGE_FALLBACK_DATA_URI

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
            border-radius: 40px;
            background: #071a2f;
            line-height: 0;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.035);
            isolation: isolate;
        }

        .coollins-phone-screen::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
            pointer-events: none;
            z-index: 45;
        }

        /* Keep a small dedicated top bezel area so the notch does not cover the logo. */
        .coollins-phone-hardware {
            position: relative;
            height: 40px;
            width: 100%;
            margin: 0;
            background: linear-gradient(180deg, #071421 0%, #08192a 100%);
            z-index: 4;
            pointer-events: none;
        }

        .coollins-dynamic-island {
            position: absolute;
            left: 50%;
            top: 6px;
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
            border-radius: 0 0 40px 40px;
        }

        .coollins-intro-target img {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-drag: none;
            border-radius: 0 0 40px 40px;
            object-fit: cover;
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
            left: 27.4%;
            top: 83.0%;
            width: 43.8%;
            height: 7.2%;
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
            .coollins-phone-screen { border-radius: 36px; }
            .coollins-intro-target,
            .coollins-intro-target img { border-radius: 36px; }
            .coollins-phone-hardware { height: 50px; }
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
        f'<img src="{INTRO_IMAGE_DATA_URI or ""}" '
        f'alt="COOLLINS AI Smart Cooling Optimizer �뚭컻 �붾㈃" />'
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
STAGE_OPTS = ["留ㅼ슦 ��쓬", "��쓬", "蹂댄넻", "�믪쓬", "留ㅼ슦 �믪쓬"]
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
                "source": f"Actual CFD 쨌 DP {int(dp_id)}",
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
            "source": f"PopField estimate 쨌 DP {int(selected_dp_id)}",
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
        "external": maps["external"][st.session_state.get("p_ext", "蹂댄넻")],
        "meeting": maps["meeting"][st.session_state.get("p_meet", "蹂댄넻")],
        "server": maps["server"][st.session_state.get("p_serv", "蹂댄넻")],
        "working": maps["working"][st.session_state.get("p_work", "蹂댄넻")],
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
        "CFD �곗씠�� ZIP�� �쎌쓣 �� �놁뒿�덈떎. GitHub�� �ㅼ젣 ZIP �뚯씪�� �ㅼ떆 �낅줈�쒗빐 二쇱꽭��. "
        f"({FIELD_ZIP_ERROR})"
    )

# Safety fallback only. Normal demo startup is fixed at 28.0 째C above.
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
        current_field["source"] = f"Actual CFD 쨌 DP {int(matched_scenario['dp_id'])} (nearest scenario)"

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

# �ъ슜�� �낅젰 �꾩옱�⑤룄�� Current Field �됯퇏�� 留욎땄
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
                    "�대┃�섏뿬 �곸꽭 遺꾩꽍 蹂닿린"
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
            text=f"<b>�� {drop:.1f}째C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=20, color=zone_color),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=temp_y,
            text=f"<b>{b:.1f} �� {a:.1f}째C</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=12, color="#ffffff"),
        )
        fig.add_annotation(
            x=ax - half_w + 0.18,
            y=dev_y,
            text=f"紐⑺몴 �몄감 {bdev:.1f} �� {adev:.1f}째C",
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
                    "�대┃�섏뿬 �곸꽭 遺꾩꽍 蹂닿린"
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



def _render_interactive_zone_map(zone_fig, before_zone_means, after_zone_means, target, height=365, *, before_zone_spreads, after_zone_spreads):
    """Render Plotly in an HTML component with true hover-fill and click popup behavior."""
    before_zone_means = np.asarray(before_zone_means, dtype=float)
    after_zone_means = np.asarray(after_zone_means, dtype=float)
    before_zone_spreads = np.asarray(before_zone_spreads, dtype=float).reshape(-1)
    after_zone_spreads = np.asarray(after_zone_spreads, dtype=float).reshape(-1)
    if len(before_zone_spreads) != len(before_zone_means) or len(after_zone_spreads) != len(after_zone_means):
        raise ValueError("Zone internal spread/mean count mismatch")
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
            "beforeSpread": float(before_zone_spreads[idx]) if np.isfinite(before_zone_spreads[idx]) else None,
            "afterSpread": float(after_zone_spreads[idx]) if np.isfinite(after_zone_spreads[idx]) else None,
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
                pointer-events:auto; box-sizing:border-box; max-height:calc(100% - 16px); overflow-y:auto;
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
            #zone-detail-float .zinternal {{margin-top:8px;}}
            #zone-detail-float .zinternal .zlabel {{font-size:10px;}}
            #zone-detail-float .zinternal .zvalue {{font-size:16px;color:#8ce7ff;}}
            #zone-detail-float .znote {{font-size:9px;color:#a9cbdc;line-height:1.5;margin-top:5px;}}
        </style>
        {plot_html}
        <div id="zone-detail-float">
            <button class="zclose" id="zone-detail-close">횞</button>
            <div class="zhead"><span class="zbar" id="zone-detail-bar"></span><div><div class="zname" id="zone-detail-name"></div><div class="ztarget" id="zone-detail-target"></div></div></div>
            <div class="ztemp"><span id="zone-before"></span><span class="zarrow">��</span><span id="zone-after"></span></div>
            <div class="zdrop" id="zone-drop"></div>
            <div class="zgrid">
                <div class="zmini"><div class="zlabel">紐⑺몴 �몄감</div><div class="zvalue" id="zone-dev"></div></div>
                <div class="zmini"><div class="zlabel">�몄감 媛먯냼</div><div class="zvalue" id="zone-devdrop"></div></div>
            </div>
            <div class="zmini zinternal">
                <div class="zlabel">Zone �대� �⑤룄 �몄감</div>
                <div class="zvalue" id="zone-internal-spread"></div>
                <div class="znote">援ъ뿭 �� �⑤룄 P95�뭁5<br>蹂�寃� �� �� 蹂�寃� ��(�덉륫)</div>
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
            document.getElementById('zone-detail-name').textContent = `ZONE ${{zone}} �곸꽭 遺꾩꽍`;
            document.getElementById('zone-detail-target').textContent = `紐⑺몴 �⑤룄 ${{d.target.toFixed(1)}}째C 湲곗�`;
            document.getElementById('zone-before').textContent = `${{d.before.toFixed(1)}}째C`;
            document.getElementById('zone-after').textContent = `${{d.after.toFixed(1)}}째C`;
            const drop = document.getElementById('zone-drop');
            drop.textContent = `�� ${{d.drop.toFixed(1)}}째C`;
            drop.style.color = d.color;
            document.getElementById('zone-dev').textContent = `${{d.beforeDev.toFixed(1)}} �� ${{d.afterDev.toFixed(1)}}째C`;
            const formatSpread = (value) => Number.isFinite(value) ? `${{value.toFixed(2)}}째C` : '怨꾩궛 遺덇�';
            document.getElementById('zone-internal-spread').textContent =
                `${{formatSpread(d.beforeSpread)}} �� ${{formatSpread(d.afterSpread)}}`;
            const dd = document.getElementById('zone-devdrop');
            dd.textContent = `�� ${{d.devDrop.toFixed(1)}}째C`;
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
                    <div style="color:#9dc6dc;font-size:11px;font-weight:700;margin-top:4px;">紐⑺몴 �⑤룄 {float(target):.1f}째C 湲곗�</div>
                </div>
            </div>
            <div style="background:rgba(6,28,48,.55);border-radius:15px;padding:14px;margin-bottom:10px;">
                <div style="color:#9fc5d9;font-size:11px;font-weight:700;margin-bottom:5px;">�됯퇏 �⑤룄 蹂���</div>
                <div style="color:#ffffff;font-size:24px;font-weight:900;letter-spacing:-.5px;">{before:.1f}째C <span style="color:#6fdcff;">��</span> {after:.1f}째C</div>
                <div style="color:{accent};font-size:25px;font-weight:900;margin-top:5px;">�� {temp_drop:.1f}째C</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;">
                <div style="background:rgba(6,28,48,.55);border-radius:14px;padding:12px;">
                    <div style="color:#9fc5d9;font-size:10px;font-weight:700;margin-bottom:5px;">紐⑺몴 �몄감</div>
                    <div style="color:#f5fbff;font-size:16px;font-weight:850;">{before_dev:.1f} �� {after_dev:.1f}째C</div>
                </div>
                <div style="background:rgba(6,28,48,.55);border-radius:14px;padding:12px;">
                    <div style="color:#9fc5d9;font-size:10px;font-weight:700;margin-bottom:5px;">�몄감 媛먯냼</div>
                    <div style="color:#7de8b5;font-size:16px;font-weight:850;">�� {dev_drop:.1f}째C</div>
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
        @st.dialog(f"ZONE {int(zone_number)} �곸꽭 遺꾩꽍")
        def _zone_dialog():
            st.markdown(panel_html, unsafe_allow_html=True)
            if st.button("�リ린", use_container_width=True, key=f"close_zone_detail_{int(zone_number)}"):
                st.session_state.selected_zone_detail = None
                st.rerun()
        _zone_dialog()
    else:
        st.markdown(f'<div class="section-title" style="margin-top:10px;">ZONE {int(zone_number)} �곸꽭 遺꾩꽍</div>', unsafe_allow_html=True)
        st.markdown(panel_html, unsafe_allow_html=True)
        if st.button("�곸꽭 蹂닿린 �リ린", use_container_width=True, key=f"close_zone_detail_inline_{int(zone_number)}"):
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
        f"S{i + 1} 쨌 Node {int(nid)}"
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
                title=dict(text="째C", font=dict(size=10, color="#d9f3ff")),
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
                "�⑤룄: %{z:.2f} 째C"
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
                    f"�⑤룄={temp:.2f}째C"
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
                    f"�⑤룄={sampled:.2f}째C"
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
                title=dict(text="째C", font=dict(size=10, color="#d9f3ff")),
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
                "�⑤룄: %{z:.2f} 째C"
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
                f"Live: {sensor_readings.get(nid, 0.0):.2f}째C"
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
                    title=dict(text="째C", font=dict(size=11, color="#eefaff")),
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
                "�⑤룄: %{marker.color:.2f} 째C"
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
                    f"�⑤룄={temp:.2f}째C"
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
        "�꾩옱 怨듦컙 �됯퇏 �⑤룄 (째C)",
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
                "�ㅼ젣 CFD ZIP�� �깆씠 李얠� 紐삵뻽�듬땲��. "
                f"吏꾨떒: {FIELD_ZIP_ERROR or 'unknown'}"
            )
        elif scenario_table is None or len(scenario_table) == 0:
            st.warning(
                f"CFD ZIP�� 濡쒕뱶�섏뿀�듬땲�� ({FIELD_ZIP_PATH.name}, {FIELD_ZIP_DP_COUNT} cases). "
                "�섏�留� Case Info�� CFD �쒕굹由ъ삤 �몃뜳�ㅻ� �곌껐�섏� 紐삵빐 紐⑤뜽 異붿젙媛믪쓣 �쒖떆�⑸땲��."
            )
        else:
            st.warning(
                f"CFD ZIP怨� �쒕굹由ъ삤 �쒕뒗 濡쒕뱶�섏뿀吏�留� DP {matched_dp_id} �ㅼ젣 field瑜� �쎌� 紐삵빐 "
                "紐⑤뜽 異붿젙媛믪쓣 �쒖떆�⑸땲��."
            )
    else:
        st.warning("Current Field�� �ㅼ젣 CFD �먯궛�� 遺덈윭�ㅼ� 紐삵뻽�듬땲��.")

    new_target = st.number_input(
        "紐⑺몴 �⑤룄 (째C)",
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

    if st.button("�됰갑 理쒖쟻��", type="primary", use_container_width=True, key="btn_home_to_heat"):
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
                <div class="factor-temp-label">�꾩옱 �⑤룄 �낅젰</div>
                <div class="factor-temp-value">{float(st.session_state.current_temp_query):.1f} 째C</div>
            </div>
            <div class="factor-temp-card">
                <div class="factor-temp-label">紐⑺몴 �⑤룄</div>
                <div class="factor-temp-value">{st.session_state.target_temp:.1f} 째C</div>
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
            <span>�됰갑 �곹뼢 �붿냼</span>
        </div>
        <div class="cooling-factor-desc">怨듦컙 �⑤룄�� �곹뼢�� 二쇰뒗 議곌굔�� <span class="step-emphasis">5�④퀎</span>濡� �ㅼ젙�섏꽭��.</div>
        """,
        unsafe_allow_html=True,
    )

    stage_opts = STAGE_OPTS

    # Old sessions are already compatible because ��쓬/蹂댄넻/�믪쓬 remain valid options.
    c1, c2 = st.columns(2)
    with c1:
        p_ext = st.select_slider(
            "��截� �몃� �댄솚寃�",
            options=stage_opts,
            value=st.session_state.p_ext if st.session_state.p_ext in stage_opts else "蹂댄넻",
            key="sl_ext",
        )
        p_meet = st.select_slider(
            "�뫁 �뚯쓽怨듦컙",
            options=stage_opts,
            value=st.session_state.p_meet if st.session_state.p_meet in stage_opts else "蹂댄넻",
            key="sl_meet",
        )
    with c2:
        p_serv = st.select_slider(
            "�뼢截� �쒕쾭 諛쒖뿴",
            options=stage_opts,
            value=st.session_state.p_serv if st.session_state.p_serv in stage_opts else "蹂댄넻",
            key="sl_serv",
        )
        p_work = st.select_slider(
            "�뮳 �낅Т怨듦컙",
            options=stage_opts,
            value=st.session_state.p_work if st.session_state.p_work in stage_opts else "蹂댄넻",
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
        "�몃� �댄솚寃�": stage_index[p_ext],
        "�쒕쾭 諛쒖뿴": stage_index[p_serv],
        "�뚯쓽怨듦컙": stage_index[p_meet],
        "�낅Т怨듦컙": stage_index[p_work],
    }
    burden_score = sum(factor_values.values()) / len(factor_values)
    burden_index = max(1, min(5, int(round(burden_score))))
    burden_label = stage_opts[burden_index - 1]
    burden_color_map = {
        "留ㅼ슦 ��쓬": "#66d9ff",
        "��쓬": "#7bd6ef",
        "蹂댄넻": "#8edbcb",
        "�믪쓬": "#ffad66",
        "留ㅼ슦 �믪쓬": "#ff6b7a",
    }
    burden_label_color = burden_color_map.get(burden_label, "#f5fbff")
    # 二쇱슂 �곹뼢 �붿씤�� '�믪쓬(4)' �먮뒗 '留ㅼ슦 �믪쓬(5)'�쇰줈 �ㅼ젙�� ��ぉ留� 蹂꾨룄 �쒖떆�⑸땲��.
    factor_icons = {
        "�몃� �댄솚寃�": "��截�",
        "�쒕쾭 諛쒖뿴": "�뼢截�",
        "�뚯쓽怨듦컙": "�뫁",
        "�낅Т怨듦컙": "�뮳",
    }
    factor_chip_class = {
        "�몃� �댄솚寃�": "factor-ext",
        "�쒕쾭 諛쒖뿴": "factor-serv",
        "�뚯쓽怨듦컙": "factor-meet",
        "�낅Т怨듦컙": "factor-work",
    }
    major_factors = [(name, level) for name, level in factor_values.items() if level >= 4]

    segments_html = "".join(
        f'<div class="cooling-load-segment {f"on-{i}" if i <= burden_index else ""}"></div>'
        for i in range(1, 6)
    )

    if major_factors:
        major_chips_html = "".join(
            f'<span class="major-factor-chip {factor_chip_class[name]}">'             f'{factor_icons[name]} {name} 쨌 {stage_opts[level - 1]}</span>'
            for name, level in major_factors
        )
    else:
        major_chips_html = '<span class="major-factor-empty">�꾩옱 �댄솚寃� �섏��� �곹뼢 �붿씤�� �놁뒿�덈떎.</span>'

    st.markdown(
        f"""
        <div class="cooling-load-card">
            <div class="cooling-load-top">
                <div class="cooling-load-label">醫낇빀 �댄솚寃� �섏�</div>
                <div class="cooling-load-level" style="color:{burden_label_color};">{burden_label}</div>
            </div>
            <div class="cooling-load-segments">{segments_html}</div>
        </div>

        <div class="major-factor-card">
            <div class="major-factor-title">二쇱슂 �곹뼢 �붿씤</div>
            <div class="major-factor-chips">{major_chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("AI 理쒖쟻 �됰갑 李얘린", type="primary", use_container_width=True, key="btn_run_cooling_opt"):
        backend = load_popfield_backend()

        if not backend.get("ok", False):
            st.error(
                "PopField 紐⑤뜽�� �ㅽ뻾�� �� �놁뒿�덈떎. "
                + str(backend.get("error", "Unknown model loading error"))
            )
        elif case_info_df is None:
            st.error("Case Info Excel�� 遺덈윭�ㅼ� 紐삵뻽�듬땲��.")
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
                        "�ㅼ젣 CFD scenario retrieval�먮뒗 GitHub 猷⑦듃�� Field data.zip�� �꾩슂�⑸땲��."
                    )

                matched_current = load_actual_cfd_case(str(FIELD_ZIP_PATH), int(retrieval["dp_id"]))
                if matched_current is None:
                    raise RuntimeError(f"dp{int(retrieval['dp_id'])}.csv瑜� Field data.zip�먯꽌 �쎌� 紐삵뻽�듬땲��.")

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

                with st.spinner("AI �덉륫 以�..."):
                    optimize_fn = backend.get("optimize_hvac_fn")
                    predict_fn = backend.get("predict_conditions_fn")

                    # Defensive recovery in case Streamlit is holding an old
                    # cached backend object from the previous build.
                    if not callable(optimize_fn) or not callable(predict_fn):
                        if not _lazy_import_popfield_modules():
                            raise RuntimeError(
                                f"PopField 紐⑤뱢�� 遺덈윭�ㅼ� 紐삵뻽�듬땲��: {POPFIELD_BACKEND_IMPORT_ERROR}"
                            )
                        optimize_fn = popfield_optimize_hvac
                        predict_fn = popfield_predict_conditions

                    if not callable(optimize_fn):
                        raise RuntimeError("PopField optimize_hvac �⑥닔瑜� 遺덈윭�ㅼ� 紐삵뻽�듬땲��.")
                    if not callable(predict_fn):
                        raise RuntimeError("PopField predict_conditions �⑥닔瑜� 遺덈윭�ㅼ� 紐삵뻽�듬땲��.")

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
                    "temp": f"{float(rec['AirTemp_C']):.0f} 째C",
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
                st.error(f"PopField 理쒖쟻�� �ㅽ뻾 以� �ㅻ쪟: {type(exc).__name__}: {exc}")


# ============================================================
# 8. SCREEN 3: RESULTS
# ============================================================
elif st.session_state.app_view == "RESULTS":
    if not st.session_state.has_run_optimization:
        st.markdown('<div class="section-title">遺꾩꽍 寃곌낵</div>', unsafe_allow_html=True)
        st.info("�꾩쭅 �ㅽ뻾�� 理쒖쟻�� 遺꾩꽍�� �놁뒿�덈떎. 癒쇱� �됰갑 議곌굔�� �ㅼ젙�섍퀬 AI 理쒖쟻�붾� �ㅽ뻾�� 二쇱꽭��.")

        if st.button("AI 理쒖쟻�� �ㅼ젙 �쒖옉�섍린", type="primary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()

        if st.button("�덉쑝濡� �대룞", type="secondary", use_container_width=True):
            st.session_state.app_view = "HOME"
            st.rerun()

    else:
        st.markdown(
            '<div class="section-title results-title-row"><span class="results-title-glyph">��</span>AI 理쒖쟻 �됰갑 寃곌낵</div>',
            unsafe_allow_html=True,
        )

        res = st.session_state.optimized_results

        vane_map = {
            "Left (L)": "醫뚯륫 (L)",
            "Middle (M)": "以묒븰 (M)",
            "Right (R)": "�곗륫 (R)",
            "L / M": "醫뚯륫 / 以묒븰",
            "M / R": "以묒븰 / �곗륫",
            "L / R": "醫뚯륫 / �곗륫",
        }
        vane_display = vane_map.get(str(res["vane"]), str(res["vane"]))

        # --------------------------------------------------------
        # Visual recommendation cards
        # --------------------------------------------------------
        flow_cmm = float(str(res["flow"]).replace("CMM", "").strip())
        supply_temp_c = float(str(res["temp"]).replace("째C", "").replace("째 C", "").strip())
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

        # Convert the relative flow position into 1��5 visual bars.
        flow_level = int(np.clip(np.ceil(max(flow_pct, 1.0) / 20.0), 1, 5))
        flow_bars_html = "".join(
            f'<span class="flow-bar {"active" if i <= flow_level else ""}"></span>'
            for i in range(1, 6)
        )

        vane_raw = str(res.get("vane", ""))
        left_on = ("Left" in vane_raw) or ("L" in vane_raw.split(" / ")) or ("醫뚯륫" in vane_display)
        middle_on = ("Middle" in vane_raw) or ("M" in vane_raw.split(" / ")) or ("以묒븰" in vane_display)
        right_on = ("Right" in vane_raw) or ("R" in vane_raw.split(" / ")) or ("�곗륫" in vane_display)

        direction_html = (
            f'<div class="air-direction-wrap">'
            f'<div class="ac-mini"></div>'
            f'<div class="air-rays">'
            f'<div class="air-dir {"active" if left_on else ""}"><span class="air-ray">��</span><span class="air-dir-tag">醫�</span></div>'
            f'<div class="air-dir {"active" if middle_on else ""}"><span class="air-ray">��</span><span class="air-dir-tag">以�</span></div>'
            f'<div class="air-dir {"active" if right_on else ""}"><span class="air-ray">��</span><span class="air-dir-tag">��</span></div>'
            f'</div>'
            f'</div>'
        )

        recommendation_html = (
            f'<div class="optimal-dispatch-box">'
            f'<h4>AI 異붿쿇 �됰갑 �ㅼ젙</h4>'
            f'<div class="hvac-visual-grid">'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">諛붾엺 諛⑺뼢</div>'
            f'<div class="hvac-mini-value">{vane_display}</div></div>'
            f'{direction_html}'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">�띾웾</div>'
            f'<div class="hvac-mini-value">{flow_cmm:.0f} CMM</div></div>'
            f'<div class="flow-bars">{flow_bars_html}</div>'
            f'<div class="hvac-card-note">�꾨낫 踰붿쐞 �� �곷� �멸린</div>'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">怨듦툒 怨듦린 �⑤룄</div>'
            f'<div class="hvac-mini-value">{supply_temp_c:.0f}째C</div></div>'
            f'<div class="hvac-track-wrap">'
            f'<div class="hvac-track temp-track">'
            f'<span class="hvac-marker" style="left:{temp_pct:.1f}%;"></span>'
            f'</div>'
            f'<div class="hvac-range"><span>{supply_min:.0f}째</span>'
            f'<span>{supply_max:.0f}째</span></div>'
            f'</div>'
            f'<div class="hvac-card-note">異붿쿇 怨듦툒 怨듦린 �ㅼ젙</div>'
            f'</div>'

            f'<div class="hvac-mini-card">'
            f'<div><div class="hvac-mini-label">�덉긽 �됰갑 異쒕젰</div>'
            f'<div class="hvac-mini-value">{q_kw:.2f} kW</div></div>'
            f'<div class="hvac-track-wrap">'
            f'<div class="hvac-track power-track">'
            f'<span class="power-fill" style="width:{q_pct:.1f}%;"></span>'
            f'<span class="hvac-marker" style="left:{q_pct:.1f}%;"></span>'
            f'</div>'
            f'<div class="hvac-range"><span>{q_min:.1f}</span>'
            f'<span>{q_max:.1f} kW</span></div>'
            f'</div>'
            f'<div class="hvac-card-note">�꾨낫 踰붿쐞 �� �곷� 異쒕젰</div>'
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
            sensor_stage = "�덉젙 �댁쟾"
            sensor_reason = "紐⑺몴 �⑤룄�� 媛�源뚯썙�� 寃�利앸맂 �듭떖 �쇱꽌 5媛쒕쭔 �쒖꽦�뷀빀�덈떎."
        elif recommended_sensor_count <= 8:
            sensor_stage = "�덉젙�� �④퀎"
            sensor_reason = f"紐⑺몴 �몄감 {predicted_error_c:.1f}째C�� 留욎떠 {recommended_sensor_count}媛� �쇱꽌瑜� �쒖꽦�뷀빀�덈떎."
        elif recommended_sensor_count <= 12:
            sensor_stage = "�뺣� 紐⑤땲�곕쭅"
            sensor_reason = f"紐⑺몴 �몄감 {predicted_error_c:.1f}째C媛� �⑥븘 {recommended_sensor_count}媛� �쇱꽌瑜� �쒖꽦�뷀빀�덈떎."
        elif recommended_sensor_count < 15:
            sensor_stage = "怨좊��� 紐⑤땲�곕쭅"
            sensor_reason = f"紐⑺몴 �몄감媛� 而� {recommended_sensor_count}媛� �쇱꽌瑜� �쒖꽦�뷀빀�덈떎."
        else:
            sensor_stage = "理쒕� 紐⑤땲�곕쭅"
            sensor_reason = "紐⑺몴 �⑤룄���� 李⑥씠媛� 而� 理쒕� 15媛� �쇱꽌瑜� �쒖꽦�뷀빀�덈떎."

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
            reduction_text = f"{deactivated_sensor_count}媛� 鍮꾪솢�깊솕"
        elif deactivated_sensor_count < 0:
            reduction_text = f"{abs(deactivated_sensor_count)}媛� 異붽� �쒖꽦��"
        else:
            reduction_text = "�쒖꽦 �쇱꽌 �� �좎�"

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
              <div class="asp-sub">�쒖꽦 �쇱꽌 議곗젙</div>
            </div>
            <div class="asp-stage">{sensor_stage}</div>
          </div>

          <div class="asp-count">
            <div class="asp-count-side">
              <div class="asp-num">{current_sensor_count}</div>
              <div class="asp-caption">珥덇린 �뺣� 紐⑤땲�곕쭅</div>
            </div>
            <div class="asp-arrow">��</div>
            <div class="asp-count-side">
              <div class="asp-num after">{recommended_sensor_count}</div>
              <div class="asp-caption">�덉젙�� �� �듭떖 �좎�</div>
            </div>
          </div>

          <div class="asp-maps">
            <div>
              <div class="asp-map-title">
                <b>Before</b><span>�쒖꽦 �쇱꽌 {current_sensor_count}媛�</span>
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
                <b>After</b><span>�쒖꽦 �쇱꽌 {recommended_sensor_count}媛�</span>
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
            "AI �쒖뼱�� �쒕��덉씠��",
            type="primary",
            use_container_width=True,
            key="btn_control_simulation",
        ):
            st.session_state.show_control_simulation = True
            st.session_state.app_view = "COMPARE"
            st.rerun()

        if st.button(
            "�덈줈�� 理쒖쟻�� �ㅽ뻾",
            type="secondary",
            use_container_width=True,
            key="btn_restart_from_results",
        ):
            st.session_state.show_control_simulation = False
            st.session_state.app_view = "HOME"
            st.rerun()


# ============================================================
# 9. SCREEN 4: BEFORE �� AFTER EFFECT COMPARISON
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

    # "紐⑺몴 珥덇낵 �곸뿭" is easier to understand than HVAC-specific hotspot jargon.
    # We count points more than 1째C above the target.
    before_hot = float(np.mean(result_current_nodes > (target + 1.0)) * 100.0)
    after_hot = float(np.mean(result_pred_nodes > (target + 1.0)) * 100.0)

    mean_delta = after_mean - before_mean

    # Distance between the room-average temperature and the requested target.
    # This replaces the always-visible spatial-spread card in the top summary.
    before_target_dev = abs(before_mean - target)
    after_target_dev = abs(after_mean - target)
    target_dev_delta = before_target_dev - after_target_dev

    spread_improve_pct = (
        max(0.0, (before_spread - after_spread) / before_spread * 100.0)
        if before_spread > 1e-8
        else 0.0
    )

    # Show spatial temperature spread under the 4-ZONE MAP only when BOTH hold:
    #   1) the spread improved versus BEFORE
    #   2) the AFTER spread is within 2.0째C
    spatial_spread_pass = (
        np.isfinite(before_spread)
        and np.isfinite(after_spread)
        and (after_spread < before_spread)
        and (after_spread <= 2.0)
    )

    hot_improve_pp = before_hot - after_hot

    # Keep the model's raw feasibility status for diagnostics, but do not turn it
    # into a pass/fail message on the demo screen. "�꾨즺" means the optimization
    # process finished; the numerical cards below still show the actual outcome.
    status = str(res.get("status", "INFEASIBLE"))
    status_text = "AI �됰갑 理쒖쟻�� �꾨즺"
    status_color = "#74e0a8"
    status_symbol = "��"

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

    st.markdown('<div class="compare-eyebrow">BEFORE �� AFTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="compare-title">AI �됰갑 �④낵 遺꾩꽍</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="compare-hero">
            <div class="compare-temp-row">
                <div class="compare-temp">{before_mean:.1f}째C</div>
                <div class="compare-arrow">��</div>
                <div class="compare-temp">{after_mean:.1f}째C</div>
            </div>
            <div class="compare-target">紐⑺몴 �⑤룄 {target:.1f}째C</div>
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
            <div class="compare-card-title">�됯퇏 �⑤룄</div>
            <div class="compare-values">
                <div class="compare-before">{before_mean:.2f}째C</div>
                <div class="compare-mini-arrow">��</div>
                <div class="compare-after">{after_mean:.2f}째C</div>
            </div>
            <div class="compare-change">{abs(mean_delta):.2f}째C 蹂���</div>
        </div>

        <div class="compare-card">
            <div class="compare-card-title">紐⑺몴 �⑤룄 �몄감</div>
            <div class="compare-values">
                <div class="compare-before">{before_target_dev:.2f}째C</div>
                <div class="compare-mini-arrow">��</div>
                <div class="compare-after">{after_target_dev:.2f}째C</div>
            </div>
            <div class="compare-change">紐⑺몴 {target:.1f}째C�� �꾩옱 {after_target_dev:.2f}째C 李⑥씠</div>
        </div>

        <div class="compare-card">
            <div class="compare-card-title">紐⑺몴 珥덇낵 �곸뿭</div>
            <div class="compare-values">
                <div class="compare-before">{before_hot:.1f}%</div>
                <div class="compare-mini-arrow">��</div>
                <div class="compare-after">{after_hot:.1f}%</div>
            </div>
            <div class="compare-change">{max(0.0, hot_improve_pp):.1f}%p 媛먯냼</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:18px;">怨듦컙 �⑤룄 蹂���</div>', unsafe_allow_html=True)

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
            f'<div class="zone-view-head"><div class="zone-view-title">4-ZONE MAP</div><div class="zone-view-sub">BEFORE �� AFTER<br>紐⑺몴 {target:.1f}째C</div></div>',
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
            before_zone_spreads=_before_zone_spreads,
            after_zone_spreads=_after_zone_spreads,
        )

        # Spatial spread is intentionally hidden unless the result satisfies
        # both requested quality conditions.
        if spatial_spread_pass:
            st.markdown(
                f"""
                <div class="compare-card" style="margin-top:12px;">
                    <div class="compare-card-title">怨듦컙 �⑤룄 �몄감</div>
                    <div class="compare-values">
                        <div class="compare-before">{before_spread:.2f}째C</div>
                        <div class="compare-mini-arrow">��</div>
                        <div class="compare-after">{after_spread:.2f}째C</div>
                    </div>
                    <div class="compare-change">
                        �⑤룄 遺덇퇏�� {spread_improve_pct:.0f}% 媛쒖꽑 쨌 2.0째C �대궡
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        if compare_field_mode == "BEFORE":
            st.markdown(
                f'<div class="compare-map-label"><span>Current Field</span><span class="sensor-count">쨌 �쒖꽦 �쇱꽌 {before_active_sensor_count}媛�</span></div>',
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
                f'<div class="compare-map-label"><span>Predicted Field</span><span class="sensor-count">쨌 �쒖꽦 �쇱꽌 {after_active_sensor_count}媛�</span></div>',
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
            �ㅼ젣 �먯뼱而⑥뿉 紐낅졊�� �꾩넚�� 寃곌낵媛� �꾨땲��,
            AI 異붿쿇 �쒖뼱�덉쓣 �곸슜�덉쓣 �뚯쓽 怨듦컙 �⑤룄瑜� �덉륫�� �쒕��덉씠�섏엯�덈떎.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "�덈줈�� 理쒖쟻�� �ㅽ뻾",
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
        if st.button("�� Home", type=btn_home_kind, use_container_width=True, key="btn_nav_home"):
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
