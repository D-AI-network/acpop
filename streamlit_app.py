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

import demo_v3_hackathon_enhanced as hvac


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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
  /* Core palette — "cool vs warm" mirrors the temperature spectrum
     the whole app is built around. */
  --ink: #0b1b2b;
  --frost: #eef4f9;
  --surface: #ffffff;
  --cool: #0e7c9e;
  --cool-deep: #0a5972;
  --cool-soft: #e3f1f6;
  --ember: #e2603f;
  --ember-soft: #fcebe6;
  --ember-line: #f3c6b8;
  --mist: #64768a;
  --line: #e2eaf1;
  --leaf: #1e9e6b;
  --leaf-soft: #e6f7ef;
  --leaf-line: #bee7d3;
  --amber: #c98a1f;
  --amber-soft: #fbf2e1;
  --amber-line: #f0dbaa;

  /* Legacy aliases — keep every existing --pf-* reference working
     without having to touch each call site individually. */
  --pf-bg: var(--frost);
  --pf-card: var(--surface);
  --pf-text: var(--ink);
  --pf-muted: var(--mist);
  --pf-primary: var(--cool);
  --pf-primary-dark: var(--cool-deep);
  --pf-primary-soft: var(--cool-soft);
  --pf-line: var(--line);
  --pf-success: var(--leaf);
  --pf-success-soft: var(--leaf-soft);
  --pf-success-line: var(--leaf-line);
  --pf-warning: var(--amber);
  --pf-warning-soft: var(--amber-soft);
  --pf-warning-line: var(--amber-line);
  --pf-danger: var(--ember);
  --pf-danger-soft: var(--ember-soft);
  --pf-danger-line: var(--ember-line);
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Display face for headers — technical/geometric, distinct from body text */
.pf-title, .pf-section-title, .pf-status-title, .pf-twin-title, .pf-brand-name {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  letter-spacing: -0.01em;
}

/* Instrument-panel signature: every number reads like a sensor readout */
.pf-temp, .pf-metric-value {
  font-family: 'JetBrains Mono', 'Inter', monospace;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}

.stApp {
  background: var(--frost);
}

.block-container {
  max-width: 440px !important;
  padding: 0 !important;
  margin: 0 auto !important;
  min-height: 100vh;
  background: var(--frost);
}

#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden;
  height: 0;
}

.pf-shell {
  padding: 24px 20px 18px 20px;
}

/* Brand mark + wordmark above the page title */
.pf-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.pf-brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: linear-gradient(135deg, var(--cool), var(--cool-deep));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  box-shadow: 0 4px 10px rgba(14, 124, 158, 0.28);
  flex-shrink: 0;
}
.pf-brand-name {
  color: var(--mist);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.pf-title {
  color: var(--ink);
  font-size: 25px;
  font-weight: 700;
  line-height: 1.2;
  margin: 0 0 12px 0;
}

/* Signature element: cool-to-ember spectrum bar under every header */
.pf-spectrum {
  height: 4px;
  width: 56px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--cool) 0%, var(--cool) 45%, var(--ember) 100%);
  margin: 0 0 16px 0;
}

.pf-subtitle {
  color: var(--mist);
  font-size: 14px;
  margin: 0 0 16px 0;
}

.pf-card {
  background: var(--surface);
  border-radius: 20px;
  padding: 18px;
  margin: 0 0 16px 0;
  border: 1px solid var(--line);
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.05), 0 8px 20px rgba(11, 27, 43, 0.03);
}

.pf-label {
  color: var(--mist);
  font-size: 12.5px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 7px;
}

.pf-temp {
  color: var(--ink);
  font-size: 34px;
  font-weight: 700;
  line-height: 1.05;
  margin-bottom: 8px;
}

.pf-blue-text {
  color: var(--cool);
  font-weight: 600;
  font-size: 14px;
}

.pf-section-title {
  color: var(--ink);
  font-size: 17px;
  font-weight: 700;
  margin: 4px 0 10px 0;
}

/* Digital-twin hero panel — soft grid + radial glow instead of flat blobs,
   so it reads as an instrument readout rather than a decoration. */
.pf-twin {
  height: 250px;
  border-radius: 24px;
  margin: 0 0 16px 0;
  padding: 18px;
  position: relative;
  overflow: hidden;
  background-color: var(--cool-soft);
  background-image:
    radial-gradient(120% 100% at 18% 15%, rgba(14, 124, 158, 0.16), transparent 60%),
    radial-gradient(90% 90% at 85% 80%, rgba(226, 96, 63, 0.14), transparent 55%),
    linear-gradient(rgba(11, 27, 43, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(11, 27, 43, 0.05) 1px, transparent 1px);
  background-size: auto, auto, 26px 26px, 26px 26px;
  border: 1px solid var(--line);
}

.pf-twin-title {
  color: var(--ink);
  font-size: 15px;
  font-weight: 700;
}

.pf-blob {
  position: absolute;
  border-radius: 999px;
  filter: blur(2px);
}
.pf-blob.cool {
  width: 92px;
  height: 92px;
  left: 44px;
  top: 104px;
  background: radial-gradient(circle at 35% 32%, #a9d6ec, var(--cool));
  opacity: 0.92;
}
.pf-blob.hot {
  width: 74px;
  height: 74px;
  right: 46px;
  top: 110px;
  background: radial-gradient(circle at 35% 32%, #f5b39d, var(--ember));
  opacity: 0.92;
}

/* Status card — left accent stripe + faint wash instead of a solid pastel
   fill, so it reads as a modern notification card. */
.pf-status {
  border-radius: 16px;
  padding: 16px 18px;
  margin: 0 0 16px 0;
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 4px solid var(--mist);
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.05);
}
.pf-ok {
  border-left-color: var(--leaf);
  background: linear-gradient(90deg, var(--leaf-soft), var(--surface) 55%);
}
.pf-near {
  border-left-color: var(--amber);
  background: linear-gradient(90deg, var(--amber-soft), var(--surface) 55%);
}
.pf-no {
  border-left-color: var(--ember);
  background: linear-gradient(90deg, var(--ember-soft), var(--surface) 55%);
}
.pf-status-title {
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 5px;
  color: var(--ink);
}

.pf-metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.pf-metric {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.04);
}
.pf-metric-label {
  color: var(--mist);
  font-size: 11.5px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 7px;
}
.pf-metric-value {
  color: var(--ink);
  font-weight: 700;
  font-size: 19px;
}

/* NOTE: the .pf-bottom marker div itself is intentionally unstyled —
   Streamlit renders each st.markdown() call in its own isolated
   container, so this div never actually wraps the st.columns() row
   that follows it. All real nav-bar styling below reaches across to
   the actual sibling column layout via :has(), see below. */

/* ------------------------------------------------------------
   Buttons — one consistent blue system across the whole app.
   Primary  = solid blue fill, white text (main actions).
   Secondary (default st.button, no type="primary") = white fill,
   blue outline + blue text (back/download/utility actions).
------------------------------------------------------------- */
div[data-testid="stButton"] > button {
  width: 100%;
  border-radius: 14px;
  min-height: 52px;
  font-weight: 700;
  font-size: 0.98rem;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
}

div[data-testid="stButton"] > button[kind="primary"] {
  background: var(--cool) !important;
  border: 1.5px solid var(--cool) !important;
  color: #ffffff !important;
  box-shadow: 0 6px 16px rgba(14, 124, 158, 0.24);
}
div[data-testid="stButton"] > button[kind="primary"] p {
  color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: var(--cool-deep) !important;
  border-color: var(--cool-deep) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover p {
  color: #ffffff !important;
}

div[data-testid="stButton"] > button:not([kind="primary"]) {
  background: #ffffff !important;
  border: 1.5px solid var(--cool) !important;
  color: var(--cool) !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]) p {
  color: var(--cool) !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
  background: var(--cool-soft) !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover p {
  color: var(--cool) !important;
}

div[data-testid="stButton"] > button:disabled {
  opacity: 0.42 !important;
  cursor: not-allowed;
  box-shadow: none !important;
}

div[data-testid="stDownloadButton"] > button {
  width: 100%;
  border-radius: 14px;
  min-height: 48px;
  background: #ffffff !important;
  border: 1.5px solid var(--cool) !important;
  color: var(--cool) !important;
  font-weight: 700;
  transition: background 0.15s ease;
}
div[data-testid="stDownloadButton"] > button p {
  color: var(--cool) !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background: var(--cool-soft) !important;
}

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  padding: 12px;
  border-radius: 16px;
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.04);
}
div[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', 'Inter', monospace !important;
}

div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--line);
}

[data-testid="stSlider"],
[data-testid="stSelectSlider"],
[data-testid="stRadio"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 16px 10px 16px;
  margin-bottom: 10px;
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.03);
}

[data-testid="stWidgetLabel"] {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  flex-wrap: wrap;
}
[data-testid="stWidgetLabel"] p {
  color: var(--ink) !important;
  font-size: 14.5px;
  font-weight: 600;
  white-space: normal !important;
  word-break: keep-all;
}

/* Step progress bar — gradient fill matching the spectrum signature */
div[data-testid="stProgress"] > div > div {
  background: var(--line) !important;
  height: 6px !important;
  border-radius: 999px !important;
}
div[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, var(--cool), var(--cool-deep)) !important;
  border-radius: 999px !important;
}

.pf-note {
  color: var(--mist);
  font-size: 12px;
  line-height: 1.5;
}

@media (max-width: 430px) {
  .stApp { background: var(--frost); }
  .block-container { max-width: 100% !important; }
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 18px !important;
  border-color: var(--line) !important;
  background: var(--surface) !important;
  padding: 6px 12px !important;
  color: #000000 !important;
  box-shadow: 0 1px 2px rgba(11, 27, 43, 0.04);
}

div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] span,
div[data-testid="stVerticalBlockBorderWrapper"] label,
div[data-testid="stVerticalBlockBorderWrapper"] div {
  color: #000000 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
  color: var(--mist) !important;
}

/* Bottom nav bar — pill-shaped, solid blue in every state, so clicking
   one button never leaves it looking different from the other two.
   REAL FIX: the marker div (.pf-bottom) never wraps the st.columns()
   row that follows it — Streamlit renders each st.markdown() call in
   its own isolated stElementContainer, as a plain sibling of the
   column layout, not a parent of it. Every rule reaches across with
   :has() from the marker's container to that sibling instead of
   relying on nesting.
   VERSION NOTE: which element sits immediately next to the marker
   differs by Streamlit version — stLayoutWrapper wraps the columns
   in newer releases (verified on 1.60), while older releases
   (verified on 1.41) put stHorizontalBlock there directly with no
   wrapper. Every selector below is duplicated for both shapes so
   the same stylesheet works across versions. */
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"],
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] {
  background: var(--surface);
  border-top: 1px solid var(--line);
  box-shadow: 0 -6px 18px rgba(11, 27, 43, 0.05);
  padding: 8px 8px 4px 8px;
  margin-top: 18px;
  position: sticky;
  bottom: 0;
  z-index: 10;
  display: flex !important;
  gap: 6px !important;
  justify-content: center !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stHorizontalBlock"] {
  display: flex !important;
  gap: 6px !important;
  justify-content: center !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stColumn"]:nth-of-type(1),
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-of-type(1) {
  width: 96px !important;
  flex: 0 0 96px !important;
  min-width: 96px !important;
  max-width: 96px !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stColumn"]:nth-of-type(2),
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stColumn"]:nth-of-type(3),
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-of-type(2),
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-of-type(3) {
  width: 140px !important;
  flex: 0 0 140px !important;
  min-width: 140px !important;
  max-width: 140px !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  width: 100% !important;
  background: var(--cool) !important;
  border: 1px solid var(--cool) !important;
  color: #ffffff !important;
  min-height: 42px !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
  font-size: 12.5px !important;
  letter-spacing: 0;
  box-shadow: none !important;
  outline: none !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  padding-left: 8px !important;
  padding-right: 8px !important;
  transition: background 0.15s ease, border-color 0.15s ease;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p {
  color: #ffffff !important;
  filter: grayscale(1) brightness(0) invert(1);
  white-space: nowrap !important;
  font-size: 12.5px !important;
  margin: 0 !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:hover,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:focus,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:focus-visible,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:active,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:focus,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:focus-visible,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:active {
  background: var(--cool-deep) !important;
  border-color: var(--cool-deep) !important;
  color: #ffffff !important;
  box-shadow: none !important;
  outline: none !important;
}
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:hover p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:focus p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:focus-visible p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stLayoutWrapper"] div[data-testid="stButton"] > button:active p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:focus p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:focus-visible p,
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .pf-bottom)
  + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:active p {
  color: #ffffff !important;
  filter: grayscale(1) brightness(0) invert(1);
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
    candidates = [
        Path("best_deploy.pt"),
        Path("best.pt"),
        Path("model/best_deploy.pt"),
        Path("model/best.pt"),
        Path("assets/best_deploy.pt"),
        Path("assets/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def _find_local_sensor_basis() -> str | None:
    for p in [
        Path("sensor_reconstruction_basis.npz"),
        Path("model/sensor_reconstruction_basis.npz"),
        Path("assets/sensor_reconstruction_basis.npz"),
    ]:
        if p.exists():
            return str(p.resolve())
    return None


def _find_local_selected_sensors() -> str | None:
    for p in [
        Path("selected_sensors.csv"),
        Path("model/selected_sensors.csv"),
        Path("assets/selected_sensors.csv"),
    ]:
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


@st.cache_data(show_spinner=False)
def load_input_metadata(case_info_path: str):
    case_df = hvac.load_case_info(case_info_path)
    col_map = {
        "external": "P83 - external",
        "meeting": "P84 - meeting",
        "server": "P85 - server",
        "working": "P86 - working",
    }
    bounds = {}
    observed = {}
    for key, col in col_map.items():
        values = np.sort(case_df[col].astype(float).unique())
        bounds[key] = (float(values.min()), float(values.max()))
        observed[key] = [float(v) for v in values.tolist()]
    return bounds, observed


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
    policy: str,
    levels: Dict[str, str] | None = None,
    exact_loads: Dict[str, float] | None = None,
    sensor_basis_path: str | None = None,
    sensor_values_c: list[float] | None = None,
    force_cpu: bool = False,
) -> Dict:
    ckpt, model, scalers, coords, coords_norm_t, case_df, level_mapping, device = load_runtime(
        checkpoint_path, case_info_path, force_cpu
    )

    if exact_loads is not None:
        loads = {k: float(exact_loads[k]) for k in ["external", "meeting", "server", "working"]}
        input_labels = {k: "exact_W" for k in loads}
        input_mode = "continuous"
    else:
        if levels is None:
            raise ValueError("Either levels or exact_loads must be provided.")
        loads = {k: float(level_mapping[k][v]) for k, v in levels.items()}
        input_labels = dict(levels)
        input_mode = "levels"

    output_dir = Path(tempfile.gettempdir()) / "popfield_streamlit_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_range_diagnostics = hvac._load_range_diagnostics(case_df, loads)

    current_temp_override = None
    sensor_info = None
    if sensor_basis_path and sensor_values_c is not None:
        current_temp_override, sensor_info = hvac.reconstruct_current_temperature_from_sensors(
            sensor_values_c=sensor_values_c,
            sensor_basis_path=sensor_basis_path,
            coords=coords,
        )
        idx = np.asarray(sensor_info["selected_sensor_idx"], dtype=int)
        sensor_info["sensor_locations"] = [
            {
                "sensor_order": int(j + 1),
                "node_index": int(node_idx),
                "x_m": float(coords[node_idx, 0]),
                "y_m": float(coords[node_idx, 1]),
                "z_m": float(coords[node_idx, 2]),
                "temperature_C": float(sensor_values_c[j]),
            }
            for j, node_idx in enumerate(idx.tolist())
        ]

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
        airflow_weight=0.25,
        current_temp_override=current_temp_override,
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
        model, rec, loads, scalers, coords, coords_norm_t, device, output_dir,
        case_df=case_df, current_temp_override=current_temp_override,
    )
    field_df = pd.read_csv(field_path)

    spatial_change = hvac.build_demo_spatial_change_report(
        model=model,
        rec=rec,
        loads=loads,
        case_df=case_df,
        scalers=scalers,
        coords=coords,
        coords_norm_t=coords_norm_t,
        device=device,
        save_dir=output_dir,
        target_temp_c=float(target_temp),
        comfort_band_c=2.0,
        max_zone_range_c=2.0,
        max_hot_fraction=0.05,
        max_cold_fraction=0.05,
        max_p95_temp_c=None,
        zone_json=None,
        top_k=5,
        min_distance_m=1.0,
        current_temp_override=current_temp_override,
        sensor_info=sensor_info,
    )
    before_after_df = pd.read_csv(spatial_change["all_node_comparison_csv"])
    hotspot_df = pd.read_csv(spatial_change["hotspot_summary_csv"])

    return {
        "status": str(diag["status"]),
        "status_label": str(diag["label_ko"]),
        "diag": diag,
        "recommendation": rec,
        "loads": loads,
        "levels": input_labels,
        "input_mode": input_mode,
        "input_range_diagnostics": input_range_diagnostics,
        "policy": policy,
        "policy_used": policy_used,
        "strict_feasible": strict_feasible,
        "num_actions": int(len(opt)),
        "decision_ms": float(elapsed_ms),
        "field": field_df,
        "before_after_field": before_after_df,
        "hotspots": hotspot_df,
        "spatial_change": spatial_change,
        "all_candidates": opt,
        "device": device,
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "additional_capacity": recs.get("additional_capacity_estimate", {}),
        "sensor_mode": bool(current_temp_override is not None),
        "sensor_info": sensor_info,
        "current_state_source": spatial_change.get("current_state_source", ""),
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


def temperature_map(
    field_df: pd.DataFrame,
    value_col: str = "pred_temperature_C",
    title: str = "Digital Twin",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Polished 2D CFD slice view without filling locations where no CFD node exists."""
    z_values = np.sort(field_df["z_m"].unique())
    target_z = z_values[np.argmin(np.abs(z_values - 1.5))]
    d = field_df[np.isclose(field_df["z_m"], target_z)].copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")

    sc = ax.scatter(
        d["x_m"],
        d["y_m"],
        c=d[value_col],
        s=92,
        cmap="coolwarm",
        edgecolors="white",
        linewidths=0.35,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )

    # Mark the hottest point on this displayed slice.
    if len(d):
        hot_row = d.loc[d[value_col].astype(float).idxmax()]
        ax.scatter(
            [hot_row["x_m"]],
            [hot_row["y_m"]],
            s=185,
            facecolors="none",
            edgecolors="#1f2937",
            linewidths=1.6,
            zorder=5,
        )
        ax.annotate(
            f'Max {float(hot_row[value_col]):.1f}℃',
            (float(hot_row["x_m"]), float(hot_row["y_m"])),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.92),
            zorder=6,
        )

    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.035)
    cb.set_label("Temperature (°C)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.6)

    ax.set_title(f"{title}  ·  z = {target_z:g} m", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X position (m)", fontsize=10)
    ax.set_ylabel("Y position (m)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28, zorder=0)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    ax.margins(x=0.03, y=0.05)
    fig.tight_layout()
    return fig


def input_range_rows(diag: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    labels = {
        "external": "외부 열환경",
        "meeting": "회의공간",
        "server": "서버·기기",
        "working": "업무공간",
    }
    rows = []
    for key in ["external", "meeting", "server", "working"]:
        d = diag[key]
        value = float(d["value_W"])
        if bool(d["exact_observed_level"]):
            status = "✅ 관측 CFD 단계"
            detail = "학습 데이터에 직접 존재"
        elif bool(d["inside_observed_range"]):
            b0, b1 = d["bracketing_observed_W"]
            status = "◌ 연속 보간"
            detail = f"{float(b0):.0f}~{float(b1):.0f} W 사이"
        else:
            status = "⚠ 범위 밖"
            detail = f"학습범위 {float(d['observed_min_W']):.0f}~{float(d['observed_max_W']):.0f} W"
        rows.append([labels[key], f"{value:.0f} W", status, detail])
    return pd.DataFrame(rows, columns=["입력", "값", "판정", "설명"])


# ------------------------------------------------------------
# NEW: glossary + before->after delta helpers (used on result page)
# ------------------------------------------------------------
METRIC_GLOSSARY = {
    "zone_range": "공간 내 서로 다른 위치 간 온도 차이입니다. 값이 작을수록 공간 전체가 고르게 냉방됩니다.",
    "hot_fraction": "기준보다 더운 위치가 전체 공간에서 차지하는 비율입니다.",
    "cold_fraction": "기준보다 추운 위치가 전체 공간에서 차지하는 비율입니다.",
    "p95_temperature": "공간 온도를 정렬했을 때 상위 5% 지점의 온도입니다. 국소적으로 더운 구역이 있는지 보여줍니다.",
    "combined_score": "풍향·풍량 후보를 비교하는 종합 점수입니다. 낮을수록 더 유리한 후보입니다.",
    "airflow_score": "바람이 고온 구역에 실제로 도달하는 정도를 나타내는 점수입니다. 낮을수록 더 유리합니다.",
    "priority_stagnant_fraction": "고온 우선영역 중 공기 흐름이 거의 없는(정체된) 비율입니다.",
}


def delta_badge(before: float, after: float, unit: str = "", lower_is_better: bool = True) -> str:
    """Return a small colored HTML badge summarizing a before->after change."""
    diff = after - before
    if abs(diff) < 0.005:
        return '<span style="color:var(--pf-muted);font-weight:700;font-size:12.5px">변화 없음</span>'
    improved = (diff < 0) if lower_is_better else (diff > 0)
    color = "var(--pf-primary)" if improved else "var(--pf-danger)"
    arrow = "↓" if diff < 0 else "↑"
    sign = "" if diff < 0 else "+"
    return (
        f'<span style="color:{color};font-weight:700;font-size:12.5px">'
        f"{arrow} {sign}{diff:.2f}{unit}</span>"
    )


def metric_pair_card(label: str, before: float, after: float, unit: str, lower_is_better: bool = True):
    """Renders a compact 'before -> after' metric card with a color-coded delta badge."""
    badge = delta_badge(before, after, unit, lower_is_better)
    st.markdown(
        f"""
        <div class="pf-metric">
          <div class="pf-metric-label">{label}</div>
          <div class="pf-metric-value" style="font-size:17px">
            {before:.2f}{unit} → {after:.2f}{unit}
          </div>
          <div style="margin-top:4px">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    "input_mode_ko": "간편 단계",
    "external_w": 750.0,
    "meeting_w": 1750.0,
    "server_w": 4200.0,
    "working_w": 1600.0,
    "use_sensor_current": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


local_ckpt = _find_local_checkpoint()
local_case = _find_local_case_info()
local_sensor_basis = _find_local_sensor_basis()
local_selected_sensors = _find_local_selected_sensors()

# Uploaded overrides survive reruns in session state
if "checkpoint_path" not in st.session_state:
    st.session_state["checkpoint_path"] = local_ckpt
if "case_info_path" not in st.session_state:
    st.session_state["case_info_path"] = local_case
if "sensor_basis_path" not in st.session_state:
    st.session_state["sensor_basis_path"] = local_sensor_basis
if "selected_sensors_path" not in st.session_state:
    st.session_state["selected_sensors_path"] = local_selected_sensors
if "use_sensor_current_initialized" not in st.session_state:
    if local_sensor_basis:
        st.session_state["use_sensor_current"] = True
    st.session_state["use_sensor_current_initialized"] = True
if "force_cpu" not in st.session_state:
    st.session_state["force_cpu"] = False


def go(page: str):
    if page == "setup" and st.session_state.get("page") != "setup":
        st.session_state["setup_step"] = 1
    st.session_state["page"] = page
    st.rerun()


# ------------------------------------------------------------
# Navigation — bottom nav bar
# ------------------------------------------------------------
def app_header(title: str):
    st.markdown(
        f"""
        <div class="pf-shell" style="padding-bottom:0">
          <div class="pf-brand">
            <div class="pf-brand-mark">❄️</div>
            <div class="pf-brand-name">PopField AI Smart Cooling</div>
          </div>
          <div class="pf-title">{title}</div>
          <div class="pf-spectrum"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bottom_nav(active: str):
    st.markdown('<div class="pf-bottom">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏠 Home", key=f"nav_home_{active}", use_container_width=False):
            go("home")
    with c2:
        if st.button("📈 Analysis", key=f"nav_analysis_{active}", use_container_width=False):
            if "last_result" in st.session_state:
                go("result")
            else:
                go("setup")
    with c3:
        if st.button("⚙️ Settings", key=f"nav_settings_{active}", use_container_width=False):
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
# AI SETUP — 4-step wizard (목표 → 열부하 → 연결 → 시작)
# ============================================================
elif st.session_state["page"] == "setup":
    app_header("AI Cooling Setup")

    if "setup_step" not in st.session_state:
        st.session_state["setup_step"] = 1

    STEP_LABELS = ["목표", "열부하", "연결", "시작"]
    TOTAL_STEPS = len(STEP_LABELS)
    step = int(st.session_state["setup_step"])

    # ---- Step indicator ----
    st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)
    ind_cols = st.columns(TOTAL_STEPS)
    for i, col in enumerate(ind_cols, start=1):
        with col:
            if i < step:
                dot, color = "✓", "var(--pf-primary)"
            elif i == step:
                dot, color = "●", "var(--pf-primary)"
            else:
                dot, color = "○", "var(--pf-muted)"
            weight = "800" if i == step else "500"
            st.markdown(
                f'<div style="text-align:center;font-size:12px;font-weight:{weight};color:{color}">'
                f"{dot}<br/>{STEP_LABELS[i - 1]}</div>",
                unsafe_allow_html=True,
            )
    st.progress(step / TOTAL_STEPS)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)

    # ------------------------------------------------------
    # STEP 1 — 🎯 목표 설정
    # ------------------------------------------------------
    if step == 1:
        with st.container(border=True):
            st.markdown('<div class="pf-section-title">🎯 목표 설정</div>', unsafe_allow_html=True)
            st.caption("먼저 원하는 목표 온도와 운전 방식을 선택하세요.")

            st.session_state["target_temp"] = st.slider(
                "🌡️ 목표 온도 (℃)",
                min_value=22.0,
                max_value=28.0,
                value=float(st.session_state["target_temp"]),
                step=0.1,
                format="%.1f",
                key="target_temp_widget",
            )

            st.session_state["policy_ko"] = st.radio(
                "운전 목표",
                ["⚖️ 균형", "🛋️ 쾌적 우선", "🍃 절약 우선"],
                horizontal=False,
                index=["⚖️ 균형", "🛋️ 쾌적 우선", "🍃 절약 우선"].index(
                    st.session_state["policy_ko"]
                ),
                key="policy_widget",
            )

    # ------------------------------------------------------
    # STEP 2 — 🔥 공간 열부하
    # ------------------------------------------------------
    elif step == 2:
        with st.container(border=True):
            st.markdown('<div class="pf-section-title">🔥 공간 열부하</div>', unsafe_allow_html=True)
            st.caption("일반 사용자는 W 단위를 입력할 필요 없이 공간 상태만 선택하면 됩니다.")

            st.session_state["input_mode_ko"] = st.radio(
                "입력 방식",
                ["간편 단계", "세밀 입력(W)"],
                horizontal=True,
                index=["간편 단계", "세밀 입력(W)"].index(st.session_state["input_mode_ko"]),
                key="input_mode_widget",
                help="세밀 입력은 학습된 CFD 범위 안의 연속 보간 질의로 사용할 수 있습니다.",
            )

            if st.session_state["input_mode_ko"] == "간편 단계":
                st.session_state["external_ko"] = st.select_slider(
                    "☀️ 외부 열환경", options=["낮음", "보통", "높음"],
                    value=st.session_state["external_ko"], key="external_widget",
                )
                st.session_state["meeting_ko"] = st.select_slider(
                    "👥 회의공간 사용", options=["낮음", "보통", "높음"],
                    value=st.session_state["meeting_ko"], key="meeting_widget",
                )
                st.session_state["server_ko"] = st.select_slider(
                    "🖥️ 서버·기기 발열", options=["낮음", "보통", "높음"],
                    value=st.session_state["server_ko"], key="server_widget",
                )
                st.session_state["working_ko"] = st.select_slider(
                    "💼 업무공간 사용", options=["낮음", "보통", "높음"],
                    value=st.session_state["working_ko"], key="working_widget",
                )
            else:
                st.caption("CFD 관측 범위 안의 중간값도 입력할 수 있습니다. 범위를 벗어나면 결과 화면에 경고가 표시됩니다.")
                bounds = None
                if st.session_state.get("case_info_path"):
                    try:
                        bounds, _observed = load_input_metadata(st.session_state["case_info_path"])
                    except Exception:
                        bounds = None

                def _range_text(key: str) -> str:
                    if not bounds:
                        return ""
                    lo, hi = bounds[key]
                    return f"관측 CFD 범위: {lo:.0f}~{hi:.0f} W"

                st.session_state["external_w"] = st.number_input(
                    "☀️ 외부 열부하 (W)", min_value=0.0, value=float(st.session_state["external_w"]),
                    step=50.0, help=_range_text("external"), key="external_w_widget",
                )
                st.session_state["meeting_w"] = st.number_input(
                    "👥 회의공간 열부하 (W)", min_value=0.0, value=float(st.session_state["meeting_w"]),
                    step=50.0, help=_range_text("meeting"), key="meeting_w_widget",
                )
                st.session_state["server_w"] = st.number_input(
                    "🖥️ 서버·기기 열부하 (W)", min_value=0.0, value=float(st.session_state["server_w"]),
                    step=50.0, help=_range_text("server"), key="server_w_widget",
                )
                st.session_state["working_w"] = st.number_input(
                    "💼 업무공간 열부하 (W)", min_value=0.0, value=float(st.session_state["working_w"]),
                    step=50.0, help=_range_text("working"), key="working_w_widget",
                )

    # ------------------------------------------------------
    # STEP 3 — 🌡️ 실측 센서 (선택) + ⚙️ 모델 연결
    # ------------------------------------------------------
    elif step == 3:
        with st.container(border=True):
            st.markdown(
                '<div class="pf-section-title">🌡️ 실측 센서 '
                '<span style="font-weight:400;font-size:13px;color:var(--pf-muted)">(선택)</span></div>',
                unsafe_allow_html=True,
            )

            st.session_state["use_sensor_current"] = st.toggle(
                "실제 센서 온도로 현재 공간 상태 복원",
                value=bool(st.session_state.get("use_sensor_current", False)),
                help="현재 온도장은 PCA+QR로 선정된 센서들의 실제 측정값에서 복원됩니다.",
                key="use_sensor_current_widget",
            )

            if st.session_state["use_sensor_current"]:
                basis_path = st.session_state.get("sensor_basis_path")
                ckpt_path = st.session_state.get("checkpoint_path")
                case_path = st.session_state.get("case_info_path")
                if basis_path and ckpt_path and case_path:
                    try:
                        _ckpt_s, _model_s, _scalers_s, coords_s, _coords_norm_s, _case_s, _levels_s, _device_s = load_runtime(
                            ckpt_path, case_path, st.session_state.get("force_cpu", False)
                        )
                        sensor_assets = hvac.load_sensor_reconstruction_basis(basis_path, coords=coords_s)
                        sensor_idx = np.asarray(sensor_assets["selected_sensor_idx"], dtype=int)
                        st.caption(
                            f"선정된 {len(sensor_idx)}개 센서의 실제 온도를 입력하세요. "
                            "이 값으로 현재 전체 온도장을 복원합니다."
                        )
                        sensor_values_c = []
                        for j, node_idx in enumerate(sensor_idx.tolist(), 1):
                            state_key = f"sensor_temp_{j}"
                            if state_key not in st.session_state:
                                st.session_state[state_key] = 24.0
                            value = st.number_input(
                                f"Sensor {j} · Node {node_idx} · XYZ "
                                f"({coords_s[node_idx,0]:.2f}, {coords_s[node_idx,1]:.2f}, {coords_s[node_idx,2]:.2f}) m · 온도 (℃)",
                                min_value=10.0, max_value=40.0,
                                value=float(st.session_state[state_key]), step=0.1, format="%.1f",
                                key=f"sensor_temp_widget_{j}",
                            )
                            st.session_state[state_key] = float(value)
                            sensor_values_c.append(float(value))
                        st.session_state["_sensor_values_c"] = sensor_values_c
                    except Exception as exc:
                        st.warning(f"센서 basis를 읽지 못했습니다: {exc}")
                        st.session_state["_sensor_values_c"] = None
                else:
                    st.warning("sensor_reconstruction_basis.npz를 연결해야 실제 센서 기반 현재 상태 복원을 사용할 수 있습니다.")
                    st.session_state["_sensor_values_c"] = None
            else:
                st.session_state["_sensor_values_c"] = None

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="pf-section-title">⚙️ 모델 연결</div>', unsafe_allow_html=True)
            st.caption(
                "배포 시 best_deploy.pt, Case Info.xlsx, sensor_reconstruction_basis.npz를 앱 폴더에 두면 자동 연결됩니다. selected_sensors.csv는 위치 확인용이라 선택 사항입니다."
            )
            uploaded_pt = st.file_uploader("best_deploy.pt / best.pt", type=["pt"], key="pt")
            uploaded_xlsx = st.file_uploader("Case Info.xlsx", type=["xlsx"], key="xlsx")
            uploaded_basis = st.file_uploader(
                "sensor_reconstruction_basis.npz", type=["npz"], key="sensor_basis_upload"
            )
            uploaded_sensor_csv = st.file_uploader(
                "selected_sensors.csv (선택)", type=["csv"], key="selected_sensors_upload"
            )
            if uploaded_pt:
                st.session_state["checkpoint_path"] = _materialize_upload(uploaded_pt, ".pt")
            if uploaded_xlsx:
                st.session_state["case_info_path"] = _materialize_upload(uploaded_xlsx, ".xlsx")
            if uploaded_basis:
                st.session_state["sensor_basis_path"] = _materialize_upload(uploaded_basis, ".npz")
                st.session_state["use_sensor_current"] = True
            if uploaded_sensor_csv:
                st.session_state["selected_sensors_path"] = _materialize_upload(uploaded_sensor_csv, ".csv")
            st.session_state["force_cpu"] = st.checkbox(
                "CPU로 실행",
                value=bool(st.session_state["force_cpu"]),
                help="GPU가 없으면 자동으로 CPU를 사용합니다.",
            )
            if st.session_state.get("checkpoint_path") and st.session_state.get("case_info_path"):
                st.success("AI 모델이 연결되었습니다.")
                if st.session_state.get("sensor_basis_path"):
                    st.success("실제 센서 기반 현재 상태 복원도 연결되었습니다.")
                else:
                    st.info("센서 basis가 없으면 현재 상태는 PopField 추정값을 사용합니다.")
            else:
                st.warning("best_deploy.pt(또는 best.pt)와 Case Info.xlsx를 연결해 주세요.")

    # ------------------------------------------------------
    # STEP 4 — ✅ 최종 확인 및 AI 분석 시작
    # ------------------------------------------------------
    elif step == 4:
        with st.container(border=True):
            st.markdown('<div class="pf-section-title">✅ 최종 확인</div>', unsafe_allow_html=True)
            st.caption("아래 설정으로 AI 분석을 실행합니다. 변경이 필요하면 이전 단계로 돌아가세요.")

            if st.session_state["input_mode_ko"] == "간편 단계":
                load_summary = (
                    f"외부 {st.session_state['external_ko']} · 회의 {st.session_state['meeting_ko']} · "
                    f"서버 {st.session_state['server_ko']} · 업무 {st.session_state['working_ko']}"
                )
            else:
                load_summary = (
                    f"외부 {st.session_state['external_w']:.0f}W · 회의 {st.session_state['meeting_w']:.0f}W · "
                    f"서버 {st.session_state['server_w']:.0f}W · 업무 {st.session_state['working_w']:.0f}W"
                )

            s1, s2 = st.columns(2)
            with s1:
                st.markdown(
                    f'<div class="pf-metric"><div class="pf-metric-label">목표 온도 · 운전 목표</div>'
                    f'<div class="pf-metric-value" style="font-size:16px">'
                    f'{st.session_state["target_temp"]:.1f}℃ · {st.session_state["policy_ko"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    f'<div class="pf-metric"><div class="pf-metric-label">공간 열부하</div>'
                    f'<div class="pf-metric-value" style="font-size:14px">{load_summary}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            model_ok = bool(
                st.session_state.get("checkpoint_path") and st.session_state.get("case_info_path")
            )
            if model_ok:
                st.success("AI 모델 연결 완료")
            else:
                st.warning("모델이 아직 연결되지 않았습니다 — '이전' 버튼으로 돌아가 연결해 주세요.")
            if st.session_state.get("use_sensor_current"):
                st.caption("🌡️ 실측 센서 기반 현재 상태 복원 사용 중")

        levels = None
        exact_loads = None
        if st.session_state["input_mode_ko"] == "간편 단계":
            levels = {
                "external": LEVEL_KO_TO_KEY[st.session_state["external_ko"]],
                "meeting": LEVEL_KO_TO_KEY[st.session_state["meeting_ko"]],
                "server": LEVEL_KO_TO_KEY[st.session_state["server_ko"]],
                "working": LEVEL_KO_TO_KEY[st.session_state["working_ko"]],
            }
        else:
            exact_loads = {
                "external": float(st.session_state["external_w"]),
                "meeting": float(st.session_state["meeting_w"]),
                "server": float(st.session_state["server_w"]),
                "working": float(st.session_state["working_w"]),
            }
        policy = POLICY_KO_TO_KEY[st.session_state["policy_ko"]]
        sensor_values_c = st.session_state.get("_sensor_values_c")

        ready = bool(
            model_ok
            and (
                not st.session_state.get("use_sensor_current", False)
                or (st.session_state.get("sensor_basis_path") and sensor_values_c is not None)
            )
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

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
                        policy=policy,
                        levels=levels,
                        exact_loads=exact_loads,
                        sensor_basis_path=(
                            st.session_state.get("sensor_basis_path")
                            if st.session_state.get("use_sensor_current", False) else None
                        ),
                        sensor_values_c=(sensor_values_c if st.session_state.get("use_sensor_current", False) else None),
                        force_cpu=st.session_state["force_cpu"],
                    )
                st.session_state["last_result"] = result
                st.session_state["last_target"] = st.session_state["target_temp"]
                st.session_state["page"] = "result"
                st.rerun()
            except Exception as exc:
                st.error(
                    "분석 중 문제가 발생했습니다. 입력값과 모델 연결 상태를 "
                    "확인한 뒤 다시 시도해 주세요."
                )
                with st.expander("기술적 세부 정보 (개발자용)"):
                    st.exception(exc)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Previous / Next navigation
    # ------------------------------------------------------
    st.markdown('<div class="pf-shell" style="padding-top:8px;padding-bottom:0">', unsafe_allow_html=True)
    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("← 이전", use_container_width=True, disabled=(step == 1), key="setup_prev"):
            st.session_state["setup_step"] = max(1, step - 1)
            st.rerun()
    with nav_next:
        if step < TOTAL_STEPS:
            if st.button("다음 →", type="primary", use_container_width=True, key="setup_next"):
                st.session_state["setup_step"] = min(TOTAL_STEPS, step + 1)
                st.rerun()
        else:
            st.markdown(
                '<div class="pf-note" style="text-align:center;padding-top:10px">마지막 단계입니다</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
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
        spatial = result["spatial_change"]
        hottest = spatial["hottest_current_location"]
        cur_m = spatial["current_metrics"]
        new_m = spatial["recommended_metrics"]

        # ------------------------------------------------------
        # TIER 1 — "What do I do" (always visible)
        # ------------------------------------------------------
        status_box(result["status"], target_for_result)

        if result.get("policy_used") != result.get("policy"):
            st.markdown(
                """
                <div class="pf-shell" style="padding-top:0;padding-bottom:0">
                  <div class="pf-note" style="background:var(--pf-warning-soft);border:1px solid var(--pf-warning-line);
                       border-radius:14px;padding:10px 14px;">
                    ⚠️ 선택하신 운전 목표로는 모든 조건을 만족하는 운전안을 찾지 못해,
                    가장 가까운 대안(best_achievable)을 대신 보여드립니다.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        action_line = (
            f"→ 풍향 {direction_text(rec)} · 풍량 {float(rec['CMM']):.0f} CMM · "
            f"토출온도 {float(rec['AirTemp_C']):.0f}℃로 설정하세요"
        )
        st.markdown(
            f"""
            <div class="pf-shell" style="padding-top:0;padding-bottom:0">
              <div class="pf-card" style="background:var(--pf-primary-soft);border:none;">
                <div style="font-size:16px;font-weight:800;color:var(--pf-primary)">{action_line}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Before/After picture FIRST — a glance tells the story faster than numbers.
        comp = result["before_after_field"]
        both_vals = np.concatenate([
            comp["current_estimated_temp_C"].to_numpy(float),
            comp["recommended_pred_temp_C"].to_numpy(float),
        ])
        vmin, vmax = float(np.nanmin(both_vals)), float(np.nanmax(both_vals))

        st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)
        st.markdown('<div class="pf-section-title">Digital Twin · Before / After</div>', unsafe_allow_html=True)
        before_label = "현재 센서 복원" if result.get("sensor_mode") else "현재 추정"
        tab_before, tab_after = st.tabs([before_label, "추천 적용 후"])
        with tab_before:
            fig = temperature_map(comp, value_col="current_estimated_temp_C", title="Current estimated field", vmin=vmin, vmax=vmax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with tab_after:
            fig = temperature_map(comp, value_col="recommended_pred_temp_C", title="Recommended field", vmin=vmin, vmax=vmax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # Core recommendation numbers, kept short.
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

        # Hottest-point headline card (kept — it's the single most persuasive number).
        current_high = float(hottest["current_estimated_temp_C"])
        recommended_high = float(hottest["recommended_pred_temp_C"])
        temp_change = recommended_high - current_high
        is_current_hotspot = bool(hottest.get("current_hotspot_above_band", False))
        location_label = "🔥 현재 Hotspot" if is_current_hotspot else "🌡️ 현재 최고온도 위치"

        if temp_change < -0.01:
            change_label, change_color = f"🔵 예상 냉각 {abs(temp_change):.2f}℃", "var(--pf-primary)"
        elif temp_change > 0.01:
            change_label, change_color = f"🔴 예상 온도 상승 +{temp_change:.2f}℃", "var(--pf-danger)"
        else:
            change_label, change_color = "⚪ 온도 변화 거의 없음", "var(--pf-muted)"

        safety_ok = bool(rec.get("hotspot_safety_constraint_met", True))
        safety_text = "🛡️ Hotspot Safety 통과" if safety_ok else "⚠️ Hotspot Safety 확인 필요"

        st.markdown(
            f"""
            <div class="pf-shell" style="padding-top:0;padding-bottom:0">
              <div class="pf-card">
                <div class="pf-label">{location_label}</div>
                <div class="pf-temp" style="font-size:28px">{current_high:.2f}℃ → {recommended_high:.2f}℃</div>
                <div style="font-size:14px;font-weight:800;color:{change_color};margin-bottom:7px">{change_label}</div>
                <div class="pf-blue-text">
                  {hottest["zone"]} · Node {int(hottest["node_index"])} ·
                  XYZ ({float(hottest["xyz_m"][0]):.2f}, {float(hottest["xyz_m"][1]):.2f}, {float(hottest["xyz_m"][2]):.2f}) m
                </div>
                <div class="pf-note" style="margin-top:10px">{safety_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------
        # TIER 2 — "Why" (one combined expander, plain-language depth)
        # ------------------------------------------------------
        st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)
        with st.expander("📋 더 자세히 보기 — 공간 전체 변화와 근거"):

            st.markdown("**공간 전체 지표 변화**")
            g1, g2 = st.columns(2)
            with g1:
                metric_pair_card("Zone 편차", float(cur_m["zone_range_C"]), float(new_m["zone_range_C"]), "℃")
                metric_pair_card("공간 최대온도", float(cur_m["max_temp_C"]), float(new_m["max_temp_C"]), "℃")
            with g2:
                metric_pair_card("Hot 영역", 100 * float(cur_m["hot_fraction"]), 100 * float(new_m["hot_fraction"]), "%")
                metric_pair_card("P95 온도", float(cur_m["p95_temp_C"]), float(new_m["p95_temp_C"]), "℃")

            st.caption(
                (
                    "현재 값은 실제 센서 입력을 PCA/QR basis로 복원한 현재 온도장입니다. "
                    "추천 후 온도장은 이 현재 상태에 PopField가 예측한 HVAC 변경 효과(ΔT)를 더해 계산합니다."
                )
                if result.get("sensor_mode")
                else (
                    "현재 값은 센서 실측이 아니라 현재 HVAC 설정 + 입력 열부하에 대한 PopField 정상상태 추정입니다."
                )
            )
            st.caption("추천 후보는 새로운 Hotspot 생성, 기존 Hotspot 악화, 최대온도 악화를 막는 Safety Guardrail을 거쳐 선택됩니다.")

            st.divider()

            st.markdown("**💨 왜 이 풍향인가요?**")
            priority_air_speed = float(rec.get("priority_air_speed_mps", float("nan")))
            priority_cooling = float(rec.get("priority_temp_improvement_C", float("nan")))
            priority_stagnant = 100.0 * float(rec.get("priority_stagnant_fraction", float("nan")))
            st.write(
                "온도만 비교하지 않고, 현재 고온 우선영역에 실제로 바람이 도달하는지와 "
                "그 영역의 예상 냉각 효과를 함께 반영해 방향을 선택합니다."
            )
            a1, a2, a3 = st.columns(3)
            a1.metric("고온영역 풍속", f"{priority_air_speed:.3f} m/s")
            a2.metric("고온영역 예상 냉각", f"{priority_cooling:+.2f}℃")
            a3.metric("우선영역 정체 비율", f"{priority_stagnant:.1f}%", help=METRIC_GLOSSARY["priority_stagnant_fraction"])

            st.divider()

            st.markdown("**🌡️ 대표 고온 위치 변화**")
            if result.get("sensor_mode") and result.get("sensor_info"):
                sensor_rows = result["sensor_info"].get("sensor_locations", [])
                if sensor_rows:
                    st.caption("현재 실측 센서 입력")
                    st.dataframe(pd.DataFrame(sensor_rows), use_container_width=True, hide_index=True)

            hotspot_show = result["hotspots"].copy()
            if len(hotspot_show):
                def _spot_status(row):
                    was_hot = bool(row["current_hotspot_above_band"])
                    remains_hot = bool(row["remaining_hotspot_above_band"])
                    if was_hot and not remains_hot:
                        return "✅ Hotspot 해소"
                    if was_hot and remains_hot:
                        return "⚠ Hotspot 잔존"
                    if (not was_hot) and remains_hot:
                        return "🔴 신규 Hotspot"
                    return "✅ 정상 범위"

                hotspot_show["상태"] = hotspot_show.apply(_spot_status, axis=1)
                hotspot_table = hotspot_show[[
                    "rank", "zone", "node_index", "current_estimated_temp_C",
                    "recommended_pred_temp_C", "temperature_change_C", "상태",
                ]].rename(columns={
                    "rank": "#", "zone": "Zone", "node_index": "Node",
                    "current_estimated_temp_C": "현재(℃)",
                    "recommended_pred_temp_C": "추천 후(℃)",
                    "temperature_change_C": "온도 변화(℃)",
                })
                hotspot_table["온도 변화(℃)"] = hotspot_table["온도 변화(℃)"].map(lambda x: round(float(x), 2))
                st.dataframe(hotspot_table, use_container_width=True, hide_index=True)

            st.divider()

            st.markdown("**✅ 쾌적 조건 확인**")
            checks = constraint_rows(result["diag"])
            if len(checks):
                st.dataframe(checks, use_container_width=True, hide_index=True)
            with st.popover("용어가 궁금하신가요? ℹ️"):
                for key in ["zone_range", "hot_fraction", "cold_fraction", "p95_temperature"]:
                    st.caption(f"**{key}** — {METRIC_GLOSSARY[key]}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------------
        # TIER 3 — "Technical details" (engineer-only, collapsed)
        # ------------------------------------------------------
        st.markdown('<div class="pf-shell" style="padding-top:0;padding-bottom:0">', unsafe_allow_html=True)
        with st.expander("🔧 기술 세부사항 (엔지니어용)"):

            st.markdown("**분석 정보**")
            st.caption(
                f"{result['num_actions']}개 후보 평가 · {result['decision_ms']:.0f} ms · {result['device']}"
            )

            st.markdown("**풍향·풍량 후보 비교**")
            cand = result["all_candidates"].copy()
            show_cols = [
                "rank", "Inlet_L", "Inlet_M", "Inlet_R", "CMM", "AirTemp_C",
                "mean_temp_C", "priority_air_speed_mps", "priority_temp_improvement_C",
                "priority_stagnant_fraction", "airflow_score", "combined_score",
                "recommendation_constraint_met",
            ]
            show_cols = [c for c in show_cols if c in cand.columns]
            cand_show = cand[show_cols].head(15).copy()
            if "priority_stagnant_fraction" in cand_show:
                cand_show["priority_stagnant_fraction"] *= 100.0
            st.dataframe(cand_show, use_container_width=True, hide_index=True)
            st.caption(
                "airflow_score와 combined_score는 낮을수록 유리합니다. "
                f"({METRIC_GLOSSARY['combined_score']} {METRIC_GLOSSARY['airflow_score']})"
            )

            if result["status"] != "FEASIBLE":
                st.markdown("**🏢 설비 한계 참고**")
                cap = result.get("additional_capacity", {}) or {}
                gap = cap.get("additional_sensible_cooling_kw_lower_bound_at_best_achievable")
                unavoidable = cap.get("additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity")
                if gap is not None:
                    st.write(f"열수지 기준 추가 냉방 여유 참고값: **{float(gap):.2f} kW**")
                if unavoidable is not None:
                    st.write(f"최대 후보 냉방에서도 남는 열수지 차이: **{float(unavoidable):.2f} kW**")
                st.caption("현열 열수지 기반 참고치이며 실제 증설 용량, 전력소비 또는 전기요금 절감량이 아닙니다.")

            st.markdown("**입력값 / 학습범위 확인**")
            st.dataframe(
                input_range_rows(result["input_range_diagnostics"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("◌ 연속 보간은 학습된 CFD 범위 안의 질의입니다. 범위 밖 값은 예측 불확실성이 커질 수 있습니다.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------------
        # Downloads + nav (bottom, de-emphasized)
        # ------------------------------------------------------
        before_after_csv = result["before_after_field"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "현재 ↔ 추천 후 전체 위치 CSV 저장",
            data=before_after_csv,
            file_name="DEMO_CURRENT_VS_RECOMMENDED_FIELD.csv",
            mime="text/csv",
            use_container_width=True,
        )
        hotspot_csv = result["hotspots"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Hotspot 변화 CSV 저장",
            data=hotspot_csv,
            file_name="DEMO_HOTSPOT_CHANGE.csv",
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
