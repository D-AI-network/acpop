"""
PopField HVAC Digital Twin v5 — 3-stage user demo
============================
Adapted from the user's graph-free DR-SWM / PopField idea for the K-DATA HVAC hackathon.

What this single script does (including complete console reporting)
----------------------------
1) Loads Case Info 200 DesignPoints.xlsx + Field data.zip (200 CFD scenarios).
2) Trains a graph-free PopField surrogate:
      HVAC/heat condition encoder
      + XYZ coordinate encoder
      + node embedding
      + 2-layer FFN + shared population-field mixer
      -> full 3D Temperature/u/v/w field + RA temperature.
3) Scenario-level train/val/test split (NO node-level leakage).
4) Learns a minimal temperature-sensor layout from TRAIN CFD fields using PCA + QR pivoting.
   Sensor count is selected on VALIDATION only; TEST is used once for final reporting.
5) Keeps the two operational roles separate:
      - sparse sensors = reconstruct the CURRENT measured temperature field
      - PopField = predict COUNTERFACTUAL unexecuted HVAC actions
   (No unstable residual pseudo-inverse fusion is used.)
6) Searches all observed HVAC actions (direction x CMM x supply temperature) for fixed heat loads.
7) Applies strict comfort constraints: zone spread, hot/cold fractions, and 95th-percentile temperature,
   then ranks feasible actions by comfort + cooling-load proxy and writes Pareto/recommendation tables.
8) Compares DP 0 (Current) against AI recommendations and reports temperature-uniformity / hotspot / estimated sensible-cooling-capacity improvements.
9) Provides RECOMMEND/DEMO modes: enter current conditions -> evaluate all HVAC candidates -> return Balanced / Comfort / Eco actions with 3-stage feasibility (feasible / near-feasible / infeasible).
10) Optionally exports thermal influence maps for external/meeting/server/working heat loads.

Important
---------
- The CFD files are steady-state scenarios, not time-series transitions. This code therefore performs
  low-latency surrogate-based counterfactual decision support, NOT reinforcement learning.
- Estimated sensible cooling capacity is computed from RA temperature, supply temperature and CMM using
  standard air-property assumptions (rho=1.20 kg/m^3, cp=1.006 kJ/kg-K). It is thermal cooling capacity,
  NOT measured electrical power consumption or electricity-cost savings.
- The provided files do not contain official Zone masks. By default this script uses four coordinate
  quadrants only as a runnable placeholder. Supply an official zone JSON when available.
- Sensors and PopField are intentionally NOT fused into one estimator: sensors reconstruct the current
  measured state; PopField evaluates unexecuted HVAC counterfactuals. This avoids the unstable sparse
  residual inversion observed in the first prototype.
- Field CSV temperatures are stored numerically in Kelvin; this loader converts them to Celsius.

Colab examples
---------------
# Train + sensor design + default HVAC optimization
!python -u PopField_HVAC_DigitalTwin.py \
  --mode train \
  --case_info "/content/Case Info 200 DesignPoints - 최종본.xlsx" \
  --field_zip "/content/Field data.zip" \
  --save_dir "/content/popfield_hvac_runs" \
  --epochs 200 --batch_size 8 --d_model 64 --num_layers 2 \
  --optimize_after_train

# Optimize a new heat-load scenario using a trained checkpoint
!python -u PopField_HVAC_DigitalTwin.py \
  --mode optimize \
  --case_info "/content/Case Info 200 DesignPoints - 최종본.xlsx" \
  --field_zip "/content/Field data.zip" \
  --checkpoint "/content/popfield_hvac_runs/best.pt" \
  --save_dir "/content/popfield_hvac_runs" \
  --external 500 --meeting 3000 --server 5000 --working 2000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. Reproducibility / utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hr(char: str = "=", width: int = 100) -> None:
    print(char * width)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# ============================================================
# 1. Column definitions / configuration
# ============================================================

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

LOAD_COLS = [
    "P83 - external",
    "P84 - meeting",
    "P85 - server",
    "P86 - working",
]

ACTION_COLS = [
    "P80 - Inlet L",
    "P81 - Inlet M",
    "P82 - Inlet R",
    "P87 - CMM",
    "P88 - AirTemp",
]

FIELD_NAMES = ["temperature_c", "velocity_u", "velocity_v", "velocity_w"]


@dataclass
class TrainConfig:
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 8
    d_model: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    epochs: int = 200
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad: float = 5.0
    velocity_weight: float = 0.25
    ra_weight: float = 0.30
    use_node_embedding: bool = True
    stable_init: bool = False
    num_workers: int = 0


# ============================================================
# 2. Data loading
# ============================================================


def _dp_number(value: str) -> int:
    m = re.search(r"DP\s*(\d+)", str(value), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse DP number from: {value!r}")
    return int(m.group(1))


def load_case_info(path: str | Path) -> pd.DataFrame:
    """Load the 200-design-point Excel file robustly, skipping the Units row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name=0, header=None)
    header = raw.iloc[0].astype(str).tolist()
    df = raw.iloc[2:].copy()
    df.columns = header
    df = df.dropna(how="all").reset_index(drop=True)

    missing = [c for c in ["Name"] + COND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Case Info missing columns: {missing}")

    df["dp_id"] = df["Name"].map(_dp_number).astype(int)
    for c in COND_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.float32)

    if df["dp_id"].duplicated().any():
        raise ValueError("Duplicate DP ids in Case Info")

    return df.sort_values("dp_id").reset_index(drop=True)


def _read_field_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read one Field CSV; the first lines contain [Name]/[Data] metadata."""
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("node number"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate 'Node Number' header in Field CSV")
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve_col(columns: Sequence[str], startswith: str) -> str:
    for c in columns:
        if str(c).strip().lower().startswith(startswith.lower()):
            return c
    raise KeyError(f"Column starting with {startswith!r} not found. columns={list(columns)}")


def load_field_zip(
    zip_path: str | Path,
    expected_dp_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    coords: [N,3] meters
    fields: [S,N,4] = temp C, u, v, w
    ra_temp_c: [S]
    dp_ids: [S]
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        name_map: Dict[int, str] = {}
        for name in zf.namelist():
            m = re.search(r"dp\s*(\d+)\.csv$", Path(name).name, flags=re.IGNORECASE)
            if m:
                name_map[int(m.group(1))] = name

        missing = [int(i) for i in expected_dp_ids if int(i) not in name_map]
        if missing:
            raise FileNotFoundError(f"Missing DP CSVs in zip: {missing[:20]}")

        coords_ref = None
        fields = []
        ra_list = []
        kept_ids = []

        for dp in expected_dp_ids:
            raw = zf.read(name_map[int(dp)])
            df = _read_field_csv_bytes(raw)

            xcol = _resolve_col(df.columns, "X [")
            ycol = _resolve_col(df.columns, "Y [")
            zcol = _resolve_col(df.columns, "Z [")
            racol = _resolve_col(df.columns, "RA temp")
            tcol = _resolve_col(df.columns, "Temperature")
            ucol = _resolve_col(df.columns, "Velocity u")
            vcol = _resolve_col(df.columns, "Velocity v")
            wcol = _resolve_col(df.columns, "Velocity w")

            coords = df[[xcol, ycol, zcol]].to_numpy(np.float32)
            ra_k = df[racol].to_numpy(np.float32)
            temp_k = df[tcol].to_numpy(np.float32)
            u = df[ucol].to_numpy(np.float32)
            v = df[vcol].to_numpy(np.float32)
            w = df[wcol].to_numpy(np.float32)

            if coords_ref is None:
                coords_ref = coords
            else:
                if coords.shape != coords_ref.shape or not np.allclose(coords, coords_ref, atol=1e-6):
                    raise ValueError(f"Node coordinates differ at DP {dp}")

            # Numeric values are Kelvin in the supplied CSVs.
            temp_c = temp_k - 273.15
            ra_c = float(np.nanmean(ra_k) - 273.15)
            field = np.stack([temp_c, u, v, w], axis=-1).astype(np.float32)

            if not np.isfinite(field).all() or not np.isfinite(ra_c):
                raise ValueError(f"NaN/Inf found in DP {dp}")

            fields.append(field)
            ra_list.append(ra_c)
            kept_ids.append(int(dp))

    return (
        np.asarray(coords_ref, dtype=np.float32),
        np.stack(fields).astype(np.float32),
        np.asarray(ra_list, dtype=np.float32),
        np.asarray(kept_ids, dtype=np.int64),
    )


def build_or_load_cache(
    case_info_path: str | Path,
    field_zip_path: str | Path,
    cache_path: str | Path,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=False)
        print(f"[Cache] loaded: {cache_path}")
        return {k: data[k] for k in data.files}

    case_df = load_case_info(case_info_path)
    coords, fields, ra_c, dp_ids = load_field_zip(field_zip_path, case_df["dp_id"].tolist())

    case_by_dp = case_df.set_index("dp_id")
    cond = case_by_dp.loc[dp_ids, COND_COLS].to_numpy(np.float32)

    out = {
        "dp_ids": dp_ids,
        "conditions": cond,
        "coords": coords,
        "fields": fields,
        "ra_temp_c": ra_c,
    }
    ensure_dir(cache_path.parent)
    np.savez_compressed(cache_path, **out)
    print(f"[Cache] built: {cache_path}")
    return out


# ============================================================
# 3. Split / normalization
# ============================================================


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, axis=None, eps: float = 1e-6) -> "Standardizer":
        mean = np.mean(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.std(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state(self) -> Dict[str, list]:
        return {"mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_state(cls, state: Dict[str, list]) -> "Standardizer":
        return cls(np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32))


def scenario_split(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return tr, va, te


class CFDDataset(Dataset):
    def __init__(
        self,
        conditions: np.ndarray,
        fields: np.ndarray,
        ra_temp_c: np.ndarray,
        indices: np.ndarray,
        cond_scaler: Standardizer,
        field_scaler: Standardizer,
        ra_scaler: Standardizer,
    ) -> None:
        self.cond = cond_scaler.transform(conditions[indices]).astype(np.float32)
        self.field = field_scaler.transform(fields[indices]).astype(np.float32)
        self.ra = ra_scaler.transform(ra_temp_c[indices]).astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.cond[i]),
            torch.from_numpy(self.field[i]),
            torch.tensor(self.ra[i], dtype=torch.float32),
            torch.tensor(self.indices[i], dtype=torch.long),
        )


# ============================================================
# 4. PopField HVAC model
# ============================================================


class ConditionEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class CoordinateEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)


class FFNFusionLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))


class HVACMeanFieldMixer(nn.Module):
    """
    Graph-free population-field interaction.

    Key idea kept from PopField:
      1) all nodes contribute to one shared latent population field z_global,
      2) a sample-specific gate scales that shared field,
      3) each node responds differently through a node-conditioned output gate.

    Difference from the traffic model:
      - no time-window statistics;
      - the field gate is driven by HVAC + heat-load conditions.
    """

    def __init__(self, cond_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.field_gate = nn.Sequential(
            nn.Linear(cond_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(d_model)
        self.context_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.node_update = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        cond_norm: torch.Tensor,
        cond_latent: torch.Tensor,
        return_diag: bool = False,
    ):
        b, n, d = h.shape
        z_global = h.mean(dim=1)
        field_gate = self.field_gate(cond_norm)                       # [B,1]
        context_raw = self.context_mlp(self.context_norm(z_global))   # [B,D]
        context = context_raw * field_gate

        ctx = context[:, None, :].expand(-1, n, -1)
        cnd = cond_latent[:, None, :].expand(-1, n, -1)
        node_context = torch.cat([h, ctx, cnd], dim=-1)
        update = self.node_update(node_context)
        alpha = self.out_gate(node_context)
        h_sp = alpha * update
        out = self.norm(h + h_sp)

        if not return_diag:
            return out, None

        diag = {
            "field_gate_mean": float(field_gate.mean().detach().cpu()),
            "field_gate_std": float(field_gate.std(unbiased=False).detach().cpu()),
            "out_gate_mean": float(alpha.mean().detach().cpu()),
            "out_gate_std": float(alpha.std(unbiased=False).detach().cpu()),
            "z_global_norm": float(z_global.norm(dim=-1).mean().detach().cpu()),
            "context_norm": float(context.norm(dim=-1).mean().detach().cpu()),
            "spatial_update_norm": float(h_sp.norm(dim=-1).mean().detach().cpu()),
        }
        return out, diag


class PopFieldHVACTwin(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        cond_dim: int = 9,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_node_embedding: bool = True,
        stable_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cond_dim = int(cond_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.use_node_embedding = bool(use_node_embedding)

        self.condition_encoder = ConditionEncoder(cond_dim, d_model, dropout)
        self.coordinate_encoder = CoordinateEncoder(d_model, dropout)
        self.condition_to_nodes = nn.Linear(d_model, d_model)

        if self.use_node_embedding:
            self.node_emb = nn.Embedding(num_nodes, d_model)
        else:
            self.node_emb = None

        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([FFNFusionLayer(d_model, dropout) for _ in range(num_layers)])

        # Shared across layer positions, matching the user's original PopField implementation.
        self.spatial_mixer = HVACMeanFieldMixer(cond_dim, d_model, dropout)

        self.field_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.ra_decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights(stable_init=stable_init)

    def _init_weights(self, stable_init: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
        if stable_init:
            nn.init.zeros_(self.field_decoder[-1].weight)
            nn.init.zeros_(self.field_decoder[-1].bias)
            nn.init.zeros_(self.ra_decoder[-1].weight)
            nn.init.zeros_(self.ra_decoder[-1].bias)

    def forward(
        self,
        cond_norm: torch.Tensor,
        coords_norm: torch.Tensor,
        return_diag: bool = False,
    ):
        """
        cond_norm : [B,9]
        coords_norm: [N,3] or [B,N,3]
        returns field_norm [B,N,4], ra_norm [B]
        """
        b = cond_norm.shape[0]
        if coords_norm.ndim == 2:
            coords_norm = coords_norm.unsqueeze(0).expand(b, -1, -1)
        if coords_norm.shape[1] != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got {coords_norm.shape[1]}")

        cond_latent = self.condition_encoder(cond_norm)                    # [B,D]
        coord_latent = self.coordinate_encoder(coords_norm)                # [B,N,D]
        h = coord_latent + self.condition_to_nodes(cond_latent)[:, None, :]

        if self.node_emb is not None:
            ids = torch.arange(self.num_nodes, device=cond_norm.device)
            h = h + self.node_emb(ids)[None, :, :]
        h = self.input_norm(h)

        diag: Dict[str, float] = {}
        for li, layer in enumerate(self.layers):
            h = layer(h)
            h, d = self.spatial_mixer(h, cond_norm, cond_latent, return_diag=return_diag)
            if d is not None:
                for k, v in d.items():
                    diag[f"layer{li}_{k}"] = v

        field_norm = self.field_decoder(h)
        z_final = h.mean(dim=1)
        ra_norm = self.ra_decoder(torch.cat([z_final, cond_latent], dim=-1)).squeeze(-1)
        if return_diag:
            return field_norm, ra_norm, diag
        return field_norm, ra_norm


# ============================================================
# 5. Training / evaluation
# ============================================================


def loss_fn(
    pred_field: torch.Tensor,
    true_field: torch.Tensor,
    pred_ra: torch.Tensor,
    true_ra: torch.Tensor,
    velocity_weight: float,
    ra_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    temp_loss = F.l1_loss(pred_field[..., 0], true_field[..., 0])
    vel_loss = F.l1_loss(pred_field[..., 1:], true_field[..., 1:])
    ra_loss = F.l1_loss(pred_ra, true_ra)
    total = temp_loss + velocity_weight * vel_loss + ra_weight * ra_loss
    return total, {
        "temp": float(temp_loss.detach().cpu()),
        "vel": float(vel_loss.detach().cpu()),
        "ra": float(ra_loss.detach().cpu()),
    }


def field_inverse_torch(x: torch.Tensor, scaler: Standardizer) -> torch.Tensor:
    mean = torch.as_tensor(scaler.mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(scaler.std, dtype=x.dtype, device=x.device)
    return x * std + mean


def ra_inverse_torch(x: torch.Tensor, scaler: Standardizer) -> torch.Tensor:
    mean = torch.as_tensor(scaler.mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(scaler.std, dtype=x.dtype, device=x.device)
    return x * std + mean


def metrics_from_arrays(pred: np.ndarray, true: np.ndarray, ra_pred=None, ra_true=None) -> Dict[str, float]:
    temp_err = pred[..., 0] - true[..., 0]
    vel_err = pred[..., 1:] - true[..., 1:]
    out = {
        "temp_mae_c": float(np.mean(np.abs(temp_err))),
        "temp_rmse_c": float(np.sqrt(np.mean(temp_err ** 2))),
        "temp_max_abs_c": float(np.max(np.abs(temp_err))),
        "velocity_component_mae": float(np.mean(np.abs(vel_err))),
        "velocity_vector_rmse": float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=-1)))),
    }
    if ra_pred is not None and ra_true is not None:
        out["ra_mae_c"] = float(np.mean(np.abs(np.asarray(ra_pred) - np.asarray(ra_true))))
    return out


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues, rap, rat, idxs = [], [], [], [], []
    for cond, field, ra, idx in loader:
        cond = cond.to(device)
        field = field.to(device)
        ra = ra.to(device)
        pf, pr = model(cond, coords_norm_t)
        preds.append(to_numpy(field_inverse_torch(pf, field_scaler)))
        trues.append(to_numpy(field_inverse_torch(field, field_scaler)))
        rap.append(to_numpy(ra_inverse_torch(pr, ra_scaler)))
        rat.append(to_numpy(ra_inverse_torch(ra, ra_scaler)))
        idxs.append(idx.numpy())
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(rap, axis=0),
        np.concatenate(rat, axis=0),
        np.concatenate(idxs, axis=0),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    device: str,
) -> Dict[str, float]:
    pred, true, rap, rat, _ = predict_loader(model, loader, coords_norm_t, field_scaler, ra_scaler, device)
    return metrics_from_arrays(pred, true, rap, rat)


def train_model(
    model: PopFieldHVACTwin,
    dl_train: DataLoader,
    dl_val: DataLoader,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    cfg: TrainConfig,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        t0 = time.perf_counter()

        for cond, field, ra, _ in dl_train:
            cond = cond.to(device)
            field = field.to(device)
            ra = ra.to(device)
            optimizer.zero_grad(set_to_none=True)
            pf, pr = model(cond, coords_norm_t)
            loss, _parts = loss_fn(pf, field, pr, ra, cfg.velocity_weight, cfg.ra_weight)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val = evaluate(model, dl_val, coords_norm_t, field_scaler, ra_scaler, device)
        val_mae = val["temp_mae_c"]
        scheduler.step(val_mae)
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        row = {
            "epoch": epoch,
            "train_loss_norm": train_loss,
            "val_temp_mae_c": val_mae,
            "val_temp_rmse_c": val["temp_rmse_c"],
            "val_velocity_component_mae": val["velocity_component_mae"],
            "val_ra_mae_c": val.get("ra_mae_c", float("nan")),
            "lr": lr_now,
            "seconds": elapsed,
        }
        history.append(row)

        improved = val_mae < best_val - 1e-6
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            tag = " *BEST*"
        else:
            no_improve += 1
            tag = ""

        print(
            f"Epoch {epoch:03d} | train={train_loss:.5f} | "
            f"val T-MAE={val_mae:.4f}C RMSE={val['temp_rmse_c']:.4f}C | "
            f"velMAE={val['velocity_component_mae']:.4f} | RA={val.get('ra_mae_c', float('nan')):.4f}C | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s{tag}"
        )

        if no_improve >= cfg.patience:
            print(f"[Early stop] best val T-MAE={best_val:.4f}C")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced")
    return best_state, history


# ============================================================
# 6. Sparse sensor design for CURRENT-state reconstruction
# ============================================================


def fit_pca_modes(fields_c: np.ndarray, rank: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fields_c: [S,N]. Returns mean[N], basis[N,K], singular_values[K]."""
    mean = fields_c.mean(axis=0)
    x = fields_c - mean[None, :]
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(rank, vt.shape[0])
    basis = vt[:k].T.astype(np.float32)
    return mean.astype(np.float32), basis, s[:k].astype(np.float32)


def select_sensors(basis: np.ndarray, k: int) -> np.ndarray:
    """Select spatial rows with pivoted QR; deterministic greedy fallback if scipy is unavailable."""
    k = min(k, basis.shape[0])
    use_rank = min(max(k, 1), basis.shape[1])
    b = basis[:, :use_rank]
    try:
        from scipy.linalg import qr
        _q, _r, piv = qr(b.T, pivoting=True, mode="economic")
        return np.asarray(piv[:k], dtype=np.int64)
    except Exception:
        selected: List[int] = []
        residual = b.copy()
        for _ in range(k):
            score = np.sum(residual ** 2, axis=1)
            if selected:
                score[np.asarray(selected, dtype=int)] = -np.inf
            j = int(np.argmax(score))
            selected.append(j)
            v = residual[j:j + 1]
            denom = float(v @ v.T)
            if denom > 1e-12:
                proj = (residual @ v.T) / denom
                residual = residual - proj @ v
        return np.asarray(selected, dtype=np.int64)


def reconstruct_from_sparse(
    sensor_values: np.ndarray,
    sensor_idx: np.ndarray,
    mean_field: np.ndarray,
    basis: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    """
    Reconstruct the CURRENT full temperature field from sparse measured sensors.

    sensor_values: [B,K]
    returns: [B,N]

    Ridge regularization is used to prevent unstable inversion when the selected
    sensing matrix is poorly conditioned.
    """
    k = len(sensor_idx)
    r = min(k, basis.shape[1])
    phi = basis[:, :r]
    a = phi[sensor_idx]  # [K,r]
    ata = a.T @ a + float(ridge) * np.eye(r, dtype=np.float32)
    rhs = (sensor_values - mean_field[sensor_idx][None, :]) @ a
    coef = np.linalg.solve(ata, rhs.T).T
    return mean_field[None, :] + coef @ phi.T


def sensing_condition_number(sensor_idx: np.ndarray, basis: np.ndarray) -> float:
    """Condition number of the square/rectangular sensing matrix used for reconstruction."""
    r = min(len(sensor_idx), basis.shape[1])
    a = basis[sensor_idx, :r]
    try:
        return float(np.linalg.cond(a))
    except Exception:
        return float("inf")


def run_sensor_study(
    train_temp: np.ndarray,
    val_temp: np.ndarray,
    test_temp: np.ndarray,
    test_pred_temp: np.ndarray,
    coords: np.ndarray,
    save_dir: Path,
    sensor_counts: Sequence[int] = (3, 5, 8, 10, 15, 20),
    pca_rank: int = 20,
    target_reconstruction_mae: float = 1.0,
    ridge: float = 1e-3,
) -> Dict:
    """
    Sensor workflow with no test leakage:
      TRAIN      -> fit PCA modes + choose sensor locations for each K
      VALIDATION -> choose the smallest K that reaches the target MAE
      TEST       -> final one-time reporting only

    PopField is reported only as a separate counterfactual-surrogate reference.
    It is NOT fused with the sensor reconstruction.
    """
    mean_temp, temp_basis, svals = fit_pca_modes(train_temp, rank=pca_rank)

    rows: List[Dict] = []
    plans: Dict[int, np.ndarray] = {}
    for k in sensor_counts:
        k = int(k)
        idx = select_sensors(temp_basis, k)
        val_recon = reconstruct_from_sparse(
            val_temp[:, idx], idx, mean_temp, temp_basis, ridge=ridge
        )
        test_recon = reconstruct_from_sparse(
            test_temp[:, idx], idx, mean_temp, temp_basis, ridge=ridge
        )
        rows.append({
            "num_sensors": k,
            "val_sensor_reconstruction_mae_c": float(np.mean(np.abs(val_recon - val_temp))),
            "test_sensor_reconstruction_mae_c": float(np.mean(np.abs(test_recon - test_temp))),
            "test_popfield_counterfactual_mae_c": float(np.mean(np.abs(test_pred_temp - test_temp))),
            "sensing_condition_number": sensing_condition_number(idx, temp_basis),
        })
        plans[k] = idx

    table = pd.DataFrame(rows).sort_values("num_sensors").reset_index(drop=True)
    table.to_csv(save_dir / "sensor_study.csv", index=False)

    # IMPORTANT: choose K using VALIDATION only, never TEST.
    eligible = table[
        table["val_sensor_reconstruction_mae_c"] <= float(target_reconstruction_mae)
    ]
    if len(eligible):
        chosen_k = int(eligible.sort_values("num_sensors").iloc[0]["num_sensors"])
        reason = (
            f"smallest sensor count with VALIDATION reconstruction MAE <= "
            f"{target_reconstruction_mae:.3f}C"
        )
    else:
        chosen_k = int(
            table.sort_values(
                ["val_sensor_reconstruction_mae_c", "num_sensors"]
            ).iloc[0]["num_sensors"]
        )
        reason = "target not reached on validation; chose best validation reconstruction candidate"

    chosen_idx = plans[chosen_k]
    chosen_row = table[table["num_sensors"] == chosen_k].iloc[0]
    sensor_df = pd.DataFrame({
        "sensor_order": np.arange(1, len(chosen_idx) + 1),
        "node_index": chosen_idx,
        "x_m": coords[chosen_idx, 0],
        "y_m": coords[chosen_idx, 1],
        "z_m": coords[chosen_idx, 2],
    })
    sensor_df.to_csv(save_dir / "selected_sensors.csv", index=False)

    centered = train_temp - train_temp.mean(axis=0, keepdims=True)
    total_ss = max(float(np.sum(centered ** 2)), 1e-12)
    explained_cum = np.cumsum(svals ** 2) / total_ss

    result = {
        "chosen_num_sensors": chosen_k,
        "selection_reason": reason,
        "chosen_node_indices": chosen_idx.tolist(),
        "chosen_validation_mae_c": float(chosen_row["val_sensor_reconstruction_mae_c"]),
        "chosen_test_mae_c": float(chosen_row["test_sensor_reconstruction_mae_c"]),
        "study": table.to_dict(orient="records"),
        "pca_rank": int(min(pca_rank, temp_basis.shape[1])),
        "ridge": float(ridge),
        "first_modes_explained_variance_approx": explained_cum.tolist(),
        "operational_roles": {
            "sensors": "reconstruct CURRENT measured temperature field",
            "popfield": "predict COUNTERFACTUAL unexecuted HVAC actions",
            "fusion": "disabled by design; unstable sparse residual inversion removed",
        },
    }
    with open(save_dir / "sensor_plan.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        save_dir / "sensor_reconstruction_basis.npz",
        temperature_mean=mean_temp,
        temperature_basis=temp_basis,
        selected_sensor_idx=chosen_idx,
        ridge=np.asarray([ridge], dtype=np.float32),
    )
    return result


# ============================================================
# 7. Zone definitions / HVAC counterfactual optimization
# ============================================================


def build_zone_masks(coords: np.ndarray, zone_json: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Official zone geometry is not included in the supplied files.
    If zone_json is omitted, four XY quadrants are used as a runnable placeholder.

    JSON format example:
    {
      "Meeting": {"x": [0, 5], "y": [4, 9], "z": [0, 3]},
      "Server":  {"x": [5,10], "y": [4, 9], "z": [0, 3]}
    }
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    if zone_json:
        spec = json.loads(Path(zone_json).read_text(encoding="utf-8"))
        masks = {}
        for name, b in spec.items():
            xr = b.get("x", [float(x.min()), float(x.max())])
            yr = b.get("y", [float(y.min()), float(y.max())])
            zr = b.get("z", [float(z.min()), float(z.max())])
            mask = (
                (x >= xr[0]) & (x <= xr[1]) &
                (y >= yr[0]) & (y <= yr[1]) &
                (z >= zr[0]) & (z <= zr[1])
            )
            if not mask.any():
                raise ValueError(f"Zone {name!r} has no nodes")
            masks[str(name)] = mask
        return masks

    xm, ym = np.median(x), np.median(y)
    return {
        "Zone_Q1": (x <= xm) & (y <= ym),
        "Zone_Q2": (x > xm) & (y <= ym),
        "Zone_Q3": (x <= xm) & (y > ym),
        "Zone_Q4": (x > xm) & (y > ym),
    }


def pareto_flags(comfort: np.ndarray, energy: np.ndarray) -> np.ndarray:
    n = len(comfort)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = (
            (comfort <= comfort[i]) &
            (energy <= energy[i]) &
            ((comfort < comfort[i]) | (energy < energy[i]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


@torch.no_grad()
def predict_conditions(
    model: PopFieldHVACTwin,
    conditions_raw: np.ndarray,
    cond_scaler: Standardizer,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    device: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_field, all_ra = [], []
    cn = cond_scaler.transform(conditions_raw).astype(np.float32)
    for s in range(0, len(cn), batch_size):
        c = torch.from_numpy(cn[s:s + batch_size]).to(device)
        pf, pr = model(c, coords_norm_t)
        all_field.append(to_numpy(field_inverse_torch(pf, field_scaler)))
        all_ra.append(to_numpy(ra_inverse_torch(pr, ra_scaler)))
    return np.concatenate(all_field), np.concatenate(all_ra)


def enumerate_observed_action_space(case_df: pd.DataFrame) -> List[Tuple[int, int, int, float, float]]:
    dirs = (
        case_df[["P80 - Inlet L", "P81 - Inlet M", "P82 - Inlet R"]]
        .drop_duplicates()
        .sort_values(["P80 - Inlet L", "P81 - Inlet M", "P82 - Inlet R"])
        .to_numpy()
    )
    cmms = sorted(case_df["P87 - CMM"].astype(float).unique().tolist())
    temps = sorted(case_df["P88 - AirTemp"].astype(float).unique().tolist())
    actions = []
    for d in dirs:
        for cmm in cmms:
            for at in temps:
                actions.append((int(d[0]), int(d[1]), int(d[2]), float(cmm), float(at)))
    return actions



def find_current_case(case_df: pd.DataFrame) -> pd.Series:
    """Return the row explicitly marked Current; fall back to DP 0, then first row."""
    name_mask = case_df["Name"].astype(str).str.contains("current", case=False, na=False)
    if bool(name_mask.any()):
        return case_df.loc[name_mask].iloc[0]
    dp0 = case_df[case_df["dp_id"] == 0]
    if len(dp0):
        return dp0.iloc[0]
    return case_df.iloc[0]


def sensible_cooling_capacity_kw(
    ra_temp_c: float,
    supply_temp_c: float,
    cmm: float,
    air_density_kg_m3: float = 1.20,
    air_cp_kj_kgk: float = 1.006,
) -> float:
    """
    Estimated sensible cooling capacity [kW].

    Q = rho * Vdot * cp * max(T_RA - T_supply, 0)
    where CMM is converted to m^3/s by /60.

    This is THERMAL cooling capacity, not measured electrical input power.
    """
    volume_flow_m3_s = max(float(cmm), 0.0) / 60.0
    delta_t = max(float(ra_temp_c) - float(supply_temp_c), 0.0)
    return float(air_density_kg_m3 * volume_flow_m3_s * air_cp_kj_kgk * delta_t)


def field_comfort_metrics(
    temp: np.ndarray,
    velocity_xyz: np.ndarray,
    ra_temp_c: float,
    action: Tuple[int, int, int, float, float],
    zones: Dict[str, np.ndarray],
    target_temp_c: float,
    comfort_band_c: float,
    max_zone_range_c: float,
    max_hot_fraction: float,
    max_cold_fraction: float,
    max_p95_temp_c: float,
) -> Dict[str, float | bool]:
    """Compute the same comfort metrics used by the optimizer for one full field."""
    temp = np.asarray(temp, dtype=np.float32)
    velocity_xyz = np.asarray(velocity_xyz, dtype=np.float32)
    vel = np.linalg.norm(velocity_xyz, axis=-1)
    zone_means = {name: float(np.mean(temp[mask])) for name, mask in zones.items()}
    zvals = np.asarray(list(zone_means.values()), dtype=float)
    zone_range = float(zvals.max() - zvals.min())
    spatial_std = float(np.std(temp))
    upper = float(target_temp_c + comfort_band_c)
    lower = float(target_temp_c - comfort_band_c)
    band_violation = float(np.mean(np.maximum(np.abs(temp - target_temp_c) - comfort_band_c, 0.0)))
    hot_fraction = float(np.mean(temp > upper))
    cold_fraction = float(np.mean(temp < lower))
    p95_temp = float(np.percentile(temp, 95))
    p05_temp = float(np.percentile(temp, 5))
    l, m, r, cmm, supply = action
    cooling_proxy = float(max(float(ra_temp_c) - float(supply), 0.0) * float(cmm))
    cooling_kw = sensible_cooling_capacity_kw(float(ra_temp_c), float(supply), float(cmm))
    zone_ok = zone_range <= float(max_zone_range_c)
    hot_ok = hot_fraction <= float(max_hot_fraction)
    cold_ok = cold_fraction <= float(max_cold_fraction)
    p95_ok = p95_temp <= float(max_p95_temp_c)
    out: Dict[str, float | bool] = {
        "Inlet_L": int(l), "Inlet_M": int(m), "Inlet_R": int(r),
        "CMM": float(cmm), "AirTemp_C": float(supply),
        "pred_RA_temp_C": float(ra_temp_c),
        "zone_range_C": zone_range,
        "spatial_std_C": spatial_std,
        "mean_temp_C": float(np.mean(temp)),
        "min_temp_C": float(np.min(temp)),
        "p05_temp_C": p05_temp,
        "p95_temp_C": p95_temp,
        "max_temp_C": float(np.max(temp)),
        "comfort_band_violation_C": band_violation,
        "hot_fraction": hot_fraction,
        "cold_fraction": cold_fraction,
        "mean_air_speed_mps": float(np.mean(vel)),
        "cooling_load_proxy": cooling_proxy,
        "estimated_sensible_cooling_kw": cooling_kw,
        "zone_constraint_met": bool(zone_ok),
        "hot_fraction_constraint_met": bool(hot_ok),
        "cold_fraction_constraint_met": bool(cold_ok),
        "p95_constraint_met": bool(p95_ok),
        "comfort_constraint_met": bool(zone_ok and hot_ok and cold_ok and p95_ok),
    }
    out.update({f"mean_{k}_C": v for k, v in zone_means.items()})
    return out


def _pct_reduction(current: float, new: float) -> float:
    current = float(current)
    new = float(new)
    if abs(current) < 1e-12:
        return float("nan")
    return float((current - new) / abs(current) * 100.0)


def estimate_capacity_gap_lower_bound(
    loads: Dict[str, float],
    candidate: Dict[str, object],
    all_candidates: pd.DataFrame,
) -> Dict[str, object]:
    """
    Estimate a conservative LOWER BOUND on additional sensible cooling capacity.

    The supplied data contain specified sensible heat loads and CFD-derived RA/supply/CMM,
    but no compressor power, COP/EER, transient room thermal mass, or equipment performance curve.
    Therefore this is NOT an exact equipment-sizing calculation.

    We report:
      - total specified sensible heat load = external + meeting + server + working
      - candidate estimated sensible cooling capacity from RA/supply/CMM
      - lower-bound load-balance gap = max(total specified heat - candidate capacity, 0)

    If the lower-bound gap is 0 but comfort constraints are still violated, the likely bottleneck is
    spatial air distribution / local hotspot removal rather than simple total cooling capacity alone.
    """
    total_heat_kw = float(sum(max(float(loads[k]), 0.0) for k in ["external", "meeting", "server", "working"]) / 1000.0)
    candidate_q_kw = float(candidate.get("estimated_sensible_cooling_kw", 0.0))
    max_available_q_kw = float(all_candidates["estimated_sensible_cooling_kw"].max())
    gap_candidate_kw = max(total_heat_kw - candidate_q_kw, 0.0)
    gap_max_available_kw = max(total_heat_kw - max_available_q_kw, 0.0)
    distribution_limited = bool(gap_max_available_kw <= 1e-9)
    return {
        "total_specified_sensible_heat_load_kw": total_heat_kw,
        "best_achievable_estimated_sensible_cooling_kw": candidate_q_kw,
        "max_available_candidate_sensible_cooling_kw": max_available_q_kw,
        "additional_sensible_cooling_kw_lower_bound_at_best_achievable": float(gap_candidate_kw),
        "additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity": float(gap_max_available_kw),
        "air_distribution_may_be_limiting": distribution_limited,
        "interpretation": (
            "Lower-bound estimate only. It is computed from specified sensible heat loads minus estimated sensible cooling capacity. "
            "It is not measured electrical power and not an exact HVAC sizing requirement. If the gap is zero but no action is feasible, "
            "airflow distribution/local hotspot removal is likely the dominant limitation."
        ),
    }


def compare_current_vs_ai(
    opt_df: pd.DataFrame,
    case_df: pd.DataFrame,
    data: Dict[str, np.ndarray],
    coords: np.ndarray,
    loads: Dict[str, float],
    save_dir: Path,
    zone_json: Optional[str],
    target_temp_c: float,
    comfort_band_c: float,
    max_zone_range_c: float,
    max_hot_fraction: float,
    max_cold_fraction: float,
    max_p95_temp_c: Optional[float],
) -> Dict[str, object]:
    """
    Compare DP 0 (Current) with Balanced / Comfort / Eco AI recommendations.

    If requested loads equal the Current row loads, the baseline uses the ACTUAL CFD field.
    Otherwise the baseline uses the optimizer's PopField prediction for the same Current HVAC action
    under the requested loads, ensuring an apples-to-apples comparison.
    """
    if max_p95_temp_c is None:
        max_p95_temp_c = float(target_temp_c + comfort_band_c)

    current = find_current_case(case_df)
    current_action = (
        int(current["P80 - Inlet L"]), int(current["P81 - Inlet M"]), int(current["P82 - Inlet R"]),
        float(current["P87 - CMM"]), float(current["P88 - AirTemp"]),
    )
    current_loads = {
        "external": float(current["P83 - external"]),
        "meeting": float(current["P84 - meeting"]),
        "server": float(current["P85 - server"]),
        "working": float(current["P86 - working"]),
    }
    same_loads = all(abs(float(loads[k]) - float(current_loads[k])) < 1e-6 for k in current_loads)
    zones = build_zone_masks(coords, zone_json)

    baseline_source = "PopField prediction of Current HVAC under requested loads"
    baseline: Dict[str, object]
    if same_loads:
        dp_ids = np.asarray(data["dp_ids"], dtype=int)
        hits = np.where(dp_ids == int(current["dp_id"]))[0]
        if len(hits):
            i = int(hits[0])
            baseline = field_comfort_metrics(
                data["fields"][i, :, 0], data["fields"][i, :, 1:4], float(data["ra_temp_c"][i]),
                current_action, zones, target_temp_c, comfort_band_c, max_zone_range_c,
                max_hot_fraction, max_cold_fraction, float(max_p95_temp_c),
            )
            baseline_source = f"actual CFD for {current['Name']}"
        else:
            same_loads = False

    if not same_loads:
        mask = (
            (opt_df["Inlet_L"] == current_action[0]) &
            (opt_df["Inlet_M"] == current_action[1]) &
            (opt_df["Inlet_R"] == current_action[2]) &
            np.isclose(opt_df["CMM"], current_action[3]) &
            np.isclose(opt_df["AirTemp_C"], current_action[4])
        )
        if not bool(mask.any()):
            raise RuntimeError("Current HVAC action is not present in the enumerated candidate action space")
        baseline = opt_df.loc[mask].iloc[0].to_dict()
        baseline["estimated_sensible_cooling_kw"] = sensible_cooling_capacity_kw(
            baseline["pred_RA_temp_C"], baseline["AirTemp_C"], baseline["CMM"]
        )

    rec_path = save_dir / "hvac_recommendations.json"
    if not rec_path.exists():
        raise FileNotFoundError(rec_path)
    recommendations = json.loads(rec_path.read_text(encoding="utf-8"))

    comparison_rows = []
    if bool(recommendations.get("fully_feasible_action_exists", False)):
        policy_items = [
            ("balanced", recommendations.get("balanced")),
            ("comfort_first", recommendations.get("comfort_first")),
            ("eco_first", recommendations.get("eco_first")),
        ]
    else:
        policy_items = [("best_achievable", recommendations.get("best_achievable"))]

    for policy_key, rec in policy_items:
        if rec is None:
            continue
        ai_cooling_kw = sensible_cooling_capacity_kw(
            rec["pred_RA_temp_C"], rec["AirTemp_C"], rec["CMM"]
        )
        row = {
            "policy": policy_key,
            "baseline_source": baseline_source,
            "current_Inlet_L": baseline["Inlet_L"],
            "current_Inlet_M": baseline["Inlet_M"],
            "current_Inlet_R": baseline["Inlet_R"],
            "current_CMM": baseline["CMM"],
            "current_AirTemp_C": baseline["AirTemp_C"],
            "ai_Inlet_L": rec["Inlet_L"],
            "ai_Inlet_M": rec["Inlet_M"],
            "ai_Inlet_R": rec["Inlet_R"],
            "ai_CMM": rec["CMM"],
            "ai_AirTemp_C": rec["AirTemp_C"],
            "current_zone_range_C": baseline["zone_range_C"],
            "ai_zone_range_C": rec["zone_range_C"],
            "zone_range_reduction_pct": _pct_reduction(baseline["zone_range_C"], rec["zone_range_C"]),
            "current_spatial_std_C": baseline["spatial_std_C"],
            "ai_spatial_std_C": rec["spatial_std_C"],
            "spatial_std_reduction_pct": _pct_reduction(baseline["spatial_std_C"], rec["spatial_std_C"]),
            "current_hot_fraction": baseline["hot_fraction"],
            "ai_hot_fraction": rec["hot_fraction"],
            "hot_fraction_reduction_percentage_points": float((baseline["hot_fraction"] - rec["hot_fraction"]) * 100.0),
            "current_cold_fraction": baseline["cold_fraction"],
            "ai_cold_fraction": rec["cold_fraction"],
            "cold_fraction_reduction_percentage_points": float((baseline["cold_fraction"] - rec["cold_fraction"]) * 100.0),
            "current_p95_temp_C": baseline["p95_temp_C"],
            "ai_p95_temp_C": rec["p95_temp_C"],
            "current_cooling_load_proxy": baseline["cooling_load_proxy"],
            "ai_cooling_load_proxy": rec["cooling_load_proxy"],
            "cooling_load_proxy_saving_pct": _pct_reduction(baseline["cooling_load_proxy"], rec["cooling_load_proxy"]),
            "current_estimated_sensible_cooling_kw": baseline["estimated_sensible_cooling_kw"],
            "ai_estimated_sensible_cooling_kw": ai_cooling_kw,
            "estimated_sensible_cooling_capacity_saving_pct": _pct_reduction(
                baseline["estimated_sensible_cooling_kw"], ai_cooling_kw
            ),
            "current_comfort_constraint_met": bool(baseline["comfort_constraint_met"]),
            "ai_comfort_constraint_met": bool(rec["comfort_constraint_met"]),
        }
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(save_dir / "current_vs_ai_comparison.csv", index=False)
    summary = {
        "current_case": str(current["Name"]),
        "current_dp_id": int(current["dp_id"]),
        "requested_loads_W": {k: float(v) for k, v in loads.items()},
        "current_case_loads_W": current_loads,
        "baseline_source": baseline_source,
        "air_property_assumptions": {
            "air_density_kg_m3": 1.20,
            "air_cp_kj_kgK": 1.006,
        },
        "important_note": (
            "estimated_sensible_cooling_kw is thermal cooling capacity derived from RA/supply temperature and CMM; "
            "it is NOT measured electrical power, COP-based input power, electricity use, or electricity-cost savings."
        ),
        "comparisons": comparison_rows,
    }
    (save_dir / "current_vs_ai_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def optimize_hvac(
    model: PopFieldHVACTwin,
    case_df: pd.DataFrame,
    loads: Dict[str, float],
    cond_scaler: Standardizer,
    coords: np.ndarray,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    device: str,
    save_dir: Path,
    zone_json: Optional[str] = None,
    target_temp_c: float = 24.0,
    comfort_band_c: float = 2.0,
    max_zone_range_c: float = 2.0,
    max_hot_fraction: float = 0.05,
    max_cold_fraction: float = 0.05,
    max_p95_temp_c: Optional[float] = None,
    energy_weight: float = 0.35,
) -> pd.DataFrame:
    """
    Counterfactual HVAC search with STRICT comfort constraints.

    Feasibility requires all of:
      1) zone mean spread <= max_zone_range_c
      2) hot fraction <= max_hot_fraction
      3) cold fraction <= max_cold_fraction
      4) 95th-percentile temperature <= max_p95_temp_c

    If no action satisfies every constraint, candidates are ranked by normalized
    constraint violation first, then comfort/energy score. This prevents a candidate
    with a good zone-average but a large hotspot from being incorrectly called optimal.
    """
    if max_p95_temp_c is None:
        max_p95_temp_c = float(target_temp_c + comfort_band_c)

    actions = enumerate_observed_action_space(case_df)
    conds = []
    for l, m, r, cmm, supply in actions:
        conds.append([
            l, m, r,
            loads["external"], loads["meeting"], loads["server"], loads["working"],
            cmm, supply,
        ])
    conds = np.asarray(conds, dtype=np.float32)
    field, ra = predict_conditions(
        model, conds, cond_scaler, coords_norm_t, field_scaler, ra_scaler, device
    )

    zones = build_zone_masks(coords, zone_json)
    upper = float(target_temp_c + comfort_band_c)
    lower = float(target_temp_c - comfort_band_c)
    rows = []

    for i, action in enumerate(actions):
        temp = field[i, :, 0]
        vel = np.linalg.norm(field[i, :, 1:4], axis=-1)
        zone_means = {name: float(np.mean(temp[mask])) for name, mask in zones.items()}
        zvals = np.asarray(list(zone_means.values()), dtype=float)

        zone_range = float(zvals.max() - zvals.min())
        spatial_std = float(np.std(temp))
        band_violation = float(np.mean(np.maximum(np.abs(temp - target_temp_c) - comfort_band_c, 0.0)))
        hot_fraction = float(np.mean(temp > upper))
        cold_fraction = float(np.mean(temp < lower))
        p95_temp = float(np.percentile(temp, 95))
        p05_temp = float(np.percentile(temp, 5))
        hot_excess_mean = float(np.mean(np.maximum(temp - upper, 0.0)))
        cold_excess_mean = float(np.mean(np.maximum(lower - temp, 0.0)))

        l, m, r, cmm, supply = action
        # Cooling-load proxy only; no COP/electrical-power data are provided.
        cooling_proxy = float(max(float(ra[i]) - supply, 0.0) * cmm)
        estimated_sensible_cooling_kw = sensible_cooling_capacity_kw(float(ra[i]), supply, cmm)

        # Comfort score includes both global uniformity and local hot/cold risk.
        p95_excess = max(p95_temp - float(max_p95_temp_c), 0.0)
        comfort_raw = (
            zone_range
            + 0.50 * spatial_std
            + 2.0 * band_violation
            + 2.0 * hot_fraction
            + 2.0 * cold_fraction
            + 1.5 * p95_excess
        )

        zone_ok = zone_range <= max_zone_range_c
        hot_ok = hot_fraction <= max_hot_fraction
        cold_ok = cold_fraction <= max_cold_fraction
        p95_ok = p95_temp <= float(max_p95_temp_c)
        strict_ok = bool(zone_ok and hot_ok and cold_ok and p95_ok)

        # Dimensionless violation score for graceful fallback if no action is fully feasible.
        violation = (
            max(zone_range - max_zone_range_c, 0.0) / max(max_zone_range_c, 1e-6)
            + max(hot_fraction - max_hot_fraction, 0.0) / max(max_hot_fraction, 1e-6)
            + max(cold_fraction - max_cold_fraction, 0.0) / max(max_cold_fraction, 1e-6)
            + max(p95_temp - float(max_p95_temp_c), 0.0) / max(comfort_band_c, 1e-6)
        )

        row = {
            "Inlet_L": l,
            "Inlet_M": m,
            "Inlet_R": r,
            "CMM": cmm,
            "AirTemp_C": supply,
            "pred_RA_temp_C": float(ra[i]),
            "zone_range_C": zone_range,
            "spatial_std_C": spatial_std,
            "mean_temp_C": float(np.mean(temp)),
            "min_temp_C": float(np.min(temp)),
            "p05_temp_C": p05_temp,
            "p95_temp_C": p95_temp,
            "max_temp_C": float(np.max(temp)),
            "comfort_band_violation_C": band_violation,
            "hot_excess_mean_C": hot_excess_mean,
            "cold_excess_mean_C": cold_excess_mean,
            "hot_fraction": hot_fraction,
            "cold_fraction": cold_fraction,
            "mean_air_speed_mps": float(np.mean(vel)),
            "cooling_load_proxy": cooling_proxy,
            "estimated_sensible_cooling_kw": estimated_sensible_cooling_kw,
            "comfort_raw": comfort_raw,
            "zone_constraint_met": bool(zone_ok),
            "hot_fraction_constraint_met": bool(hot_ok),
            "cold_fraction_constraint_met": bool(cold_ok),
            "p95_constraint_met": bool(p95_ok),
            "comfort_constraint_met": strict_ok,
            "constraint_violation_score": float(violation),
        }
        row.update({f"mean_{k}_C": v for k, v in zone_means.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    c = df["comfort_raw"].to_numpy(float)
    e = df["cooling_load_proxy"].to_numpy(float)
    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-12)
    e_norm = (e - e.min()) / (e.max() - e.min() + 1e-12)
    df["combined_score"] = (1.0 - energy_weight) * c_norm + energy_weight * e_norm
    df["pareto_optimal"] = pareto_flags(c, e)

    # STRICT constraint-first ranking. If no fully feasible action exists, the least-violating
    # candidate is returned first instead of pretending that it fully satisfies comfort.
    df = df.sort_values(
        ["comfort_constraint_met", "constraint_violation_score", "combined_score", "comfort_raw", "cooling_load_proxy"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(save_dir / "hvac_optimization.csv", index=False)

    feasible = df[df["comfort_constraint_met"]].copy()
    has_feasible = bool(len(feasible) > 0)

    if has_feasible:
        balanced = feasible.sort_values(["combined_score", "comfort_raw"]).iloc[0].to_dict()
        comfort_best = feasible.sort_values(["comfort_raw", "cooling_load_proxy"]).iloc[0].to_dict()
        eco_best = feasible.sort_values(["cooling_load_proxy", "comfort_raw"]).iloc[0].to_dict()
        recommendations = {
            "status": "FEASIBLE_RECOMMENDATIONS_AVAILABLE",
            "fully_feasible_action_exists": True,
            "facility_limit_warning": False,
            "balanced": balanced,
            "comfort_first": comfort_best,
            "eco_first": eco_best,
            "best_achievable": None,
            "additional_capacity_estimate": {
                "additional_sensible_cooling_kw_lower_bound_at_best_achievable": 0.0,
                "interpretation": "At least one candidate satisfies all configured comfort constraints; no fallback capacity-gap estimate is required."
            },
            "note": "All three recommendations satisfy every configured comfort constraint."
        }
    else:
        # IMPORTANT: do NOT label infeasible low-energy actions as Eco/Comfort recommendations.
        # Return one transparent best-achievable action, ranked by minimum normalized constraint violation.
        best_achievable = df.sort_values(
            ["constraint_violation_score", "comfort_raw", "combined_score", "cooling_load_proxy"]
        ).iloc[0].to_dict()
        failed_constraints = []
        if not bool(best_achievable["zone_constraint_met"]):
            failed_constraints.append("zone_range")
        if not bool(best_achievable["hot_fraction_constraint_met"]):
            failed_constraints.append("hot_fraction")
        if not bool(best_achievable["cold_fraction_constraint_met"]):
            failed_constraints.append("cold_fraction")
        if not bool(best_achievable["p95_constraint_met"]):
            failed_constraints.append("p95_temperature")

        capacity_gap = estimate_capacity_gap_lower_bound(loads, best_achievable, df)
        recommendations = {
            "status": "NO_FEASIBLE_ACTION",
            "fully_feasible_action_exists": False,
            "facility_limit_warning": True,
            "balanced": None,
            "comfort_first": None,
            "eco_first": None,
            "best_achievable": best_achievable,
            "failed_constraints_for_best_achievable": failed_constraints,
            "additional_capacity_estimate": capacity_gap,
            "feasibility_diagnostics": {
                "min_zone_range_C_across_candidates": float(df["zone_range_C"].min()),
                "min_hot_fraction_across_candidates": float(df["hot_fraction"].min()),
                "min_cold_fraction_across_candidates": float(df["cold_fraction"].min()),
                "min_p95_temp_C_across_candidates": float(df["p95_temp_C"].min()),
                "num_candidates": int(len(df)),
                "num_fully_feasible_candidates": 0,
            },
            "note": (
                "No evaluated HVAC action satisfies all configured comfort constraints. "
                "Balanced/Comfort/Eco labels are intentionally suppressed. 'best_achievable' is the least-violating action, not a fully comfortable solution."
            ),
        }
    with open(save_dir / "hvac_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    best = df.iloc[0].to_dict()
    summary = {
        "loads_W": loads,
        "num_actions_evaluated": len(actions),
        "target_temp_c": target_temp_c,
        "comfort_band_c": comfort_band_c,
        "max_zone_range_c": max_zone_range_c,
        "max_hot_fraction": max_hot_fraction,
        "max_cold_fraction": max_cold_fraction,
        "max_p95_temp_c": float(max_p95_temp_c),
        "energy_weight": energy_weight,
        "zone_mode": "official_json" if zone_json else "placeholder_xy_quadrants",
        "fully_feasible_action_exists": has_feasible,
        "best_ranked_action": best,
        "recommendation_file": str(save_dir / "hvac_recommendations.json"),
    }
    with open(save_dir / "hvac_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return df


# ============================================================
# 8. Thermal influence maps
# ============================================================


def nearest_valid_step(values: Sequence[float], current: float) -> float:
    vals = np.sort(np.unique(np.asarray(values, dtype=float)))
    if len(vals) <= 1:
        return 1.0
    diffs = np.diff(vals)
    return float(np.min(diffs[diffs > 0]))


def export_influence_maps(
    model: PopFieldHVACTwin,
    base_condition: np.ndarray,
    case_df: pd.DataFrame,
    cond_scaler: Standardizer,
    coords_norm_t: torch.Tensor,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    device: str,
    save_dir: Path,
) -> None:
    """Finite-difference dTemperature / d(load_kW) around a base condition."""
    load_indices = {
        "external": 3,
        "meeting": 4,
        "server": 5,
        "working": 6,
    }
    col_map = {
        "external": "P83 - external",
        "meeting": "P84 - meeting",
        "server": "P85 - server",
        "working": "P86 - working",
    }
    maps = {}
    rows = []
    for name, ci in load_indices.items():
        step_w = nearest_valid_step(case_df[col_map[name]].unique(), float(base_condition[ci]))
        c_minus = base_condition.copy()
        c_plus = base_condition.copy()
        c_minus[ci] = max(0.0, c_minus[ci] - step_w)
        c_plus[ci] = c_plus[ci] + step_w
        conds = np.stack([c_minus, c_plus]).astype(np.float32)
        pred, _ = predict_conditions(
            model, conds, cond_scaler, coords_norm_t, field_scaler, ra_scaler, device
        )
        denom_kw = (c_plus[ci] - c_minus[ci]) / 1000.0
        sens = (pred[1, :, 0] - pred[0, :, 0]) / max(denom_kw, 1e-8)
        maps[name] = sens.astype(np.float32)
        top = np.argsort(np.abs(sens))[::-1][:20]
        for rank, node in enumerate(top, 1):
            rows.append({
                "source": name,
                "rank": rank,
                "node_index": int(node),
                "sensitivity_C_per_kW": float(sens[node]),
            })
    np.savez_compressed(save_dir / "thermal_influence_maps.npz", **maps)
    pd.DataFrame(rows).to_csv(save_dir / "thermal_influence_top_nodes.csv", index=False)


# ============================================================
# 9. Checkpoint helpers
# ============================================================


def save_checkpoint(
    path: Path,
    model: PopFieldHVACTwin,
    cfg: TrainConfig,
    cond_scaler: Standardizer,
    coord_scaler: Standardizer,
    field_scaler: Standardizer,
    ra_scaler: Standardizer,
    coords: np.ndarray,
    split: Dict[str, np.ndarray],
    metrics: Dict,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "model_args": {
            "num_nodes": model.num_nodes,
            "cond_dim": model.cond_dim,
            "d_model": model.d_model,
            "num_layers": model.num_layers,
            "dropout": cfg.dropout,
            "use_node_embedding": cfg.use_node_embedding,
            "stable_init": False,
        },
        "train_config": asdict(cfg),
        "cond_scaler": cond_scaler.state(),
        "coord_scaler": coord_scaler.state(),
        "field_scaler": field_scaler.state(),
        "ra_scaler": ra_scaler.state(),
        "coords": coords,
        "split": {k: np.asarray(v, dtype=np.int64) for k, v in split.items()},
        "metrics": metrics,
        "condition_columns": COND_COLS,
        "field_names": FIELD_NAMES,
    }, path)


def load_checkpoint(path: str | Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PopFieldHVACTwin(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scalers = {
        "cond": Standardizer.from_state(ckpt["cond_scaler"]),
        "coord": Standardizer.from_state(ckpt["coord_scaler"]),
        "field": Standardizer.from_state(ckpt["field_scaler"]),
        "ra": Standardizer.from_state(ckpt["ra_scaler"]),
    }
    coords = np.asarray(ckpt["coords"], dtype=np.float32)
    return ckpt, model, scalers, coords


# ============================================================
# 10. Main train pipeline
# ============================================================


def run_train(args) -> None:
    save_dir = ensure_dir(args.save_dir)
    cache_path = save_dir / "cfd_cache.npz"
    data = build_or_load_cache(args.case_info, args.field_zip, cache_path, args.force_rebuild_cache)

    dp_ids = data["dp_ids"]
    conditions = data["conditions"]
    coords = data["coords"]
    fields = data["fields"]
    ra = data["ra_temp_c"]

    n_cases, n_nodes = fields.shape[:2]
    tr, va, te = scenario_split(n_cases, args.train_ratio, args.val_ratio, args.seed)

    cond_scaler = Standardizer.fit(conditions[tr], axis=0)
    coord_scaler = Standardizer.fit(coords, axis=0)
    field_scaler = Standardizer.fit(fields[tr], axis=(0, 1))
    ra_scaler = Standardizer.fit(ra[tr], axis=0)

    cfg = TrainConfig(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        velocity_weight=args.velocity_weight,
        ra_weight=args.ra_weight,
        use_node_embedding=not args.no_node_embedding,
        stable_init=args.stable_init,
        num_workers=args.num_workers,
    )

    ds_tr = CFDDataset(conditions, fields, ra, tr, cond_scaler, field_scaler, ra_scaler)
    ds_va = CFDDataset(conditions, fields, ra, va, cond_scaler, field_scaler, ra_scaler)
    ds_te = CFDDataset(conditions, fields, ra, te, cond_scaler, field_scaler, ra_scaler)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    coords_norm = coord_scaler.transform(coords).astype(np.float32)
    coords_norm_t = torch.from_numpy(coords_norm).to(device)

    model = PopFieldHVACTwin(
        num_nodes=n_nodes,
        cond_dim=len(COND_COLS),
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_node_embedding=cfg.use_node_embedding,
        stable_init=cfg.stable_init,
    ).to(device)

    hr()
    print("PopField HVAC Digital Twin")
    print(f"Device={device} | cases={n_cases} | nodes={n_nodes} | params={count_params(model):,}")
    print(f"Split train/val/test={len(tr)}/{len(va)}/{len(te)} (scenario-level)")
    print("Model: ConditionEncoder + XYZ + NodeEmbedding + [FFN + shared PopField] x L + Field/RA decoders")
    hr()

    best_state, history = train_model(
        model, dl_tr, dl_va, coords_norm_t, field_scaler, ra_scaler, cfg, device
    )
    model.load_state_dict(best_state)

    train_pred, train_true, train_ra_pred, train_ra_true, train_idx = predict_loader(
        model, DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=False), coords_norm_t,
        field_scaler, ra_scaler, device
    )
    val_metrics = evaluate(model, dl_va, coords_norm_t, field_scaler, ra_scaler, device)
    test_pred, test_true, test_ra_pred, test_ra_true, test_idx = predict_loader(
        model, dl_te, coords_norm_t, field_scaler, ra_scaler, device
    )
    test_metrics = metrics_from_arrays(test_pred, test_true, test_ra_pred, test_ra_true)

    print("\n[Validation]", json.dumps(val_metrics, indent=2))
    print("[Test]", json.dumps(test_metrics, indent=2))

    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    np.savez_compressed(
        save_dir / "test_predictions.npz",
        dp_ids=dp_ids[test_idx],
        pred=test_pred,
        true=test_true,
        ra_pred=test_ra_pred,
        ra_true=test_ra_true,
    )

    # Sensor study with strict no-test-leakage protocol:
    # TRAIN fits PCA/sensor locations, VALIDATION chooses sensor count, TEST reports final performance.
    sensor_result = run_sensor_study(
        train_temp=fields[tr, :, 0],
        val_temp=fields[va, :, 0],
        test_temp=fields[te, :, 0],
        test_pred_temp=test_pred[..., 0],
        coords=coords,
        save_dir=save_dir,
        sensor_counts=[int(x) for x in args.sensor_counts.split(",") if x.strip()],
        pca_rank=args.sensor_pca_rank,
        target_reconstruction_mae=args.sensor_target_mae,
        ridge=args.sensor_ridge,
    )
    print("\n[Sensor plan]", json.dumps(sensor_result, indent=2))

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "sensor_plan": sensor_result,
        "params": count_params(model),
    }
    ckpt_path = save_dir / "best.pt"
    save_checkpoint(
        ckpt_path, model, cfg, cond_scaler, coord_scaler, field_scaler, ra_scaler,
        coords, {"train": tr, "val": va, "test": te}, all_metrics
    )
    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {ckpt_path}")

    if args.optimize_after_train:
        case_df = load_case_info(args.case_info)
        current_row = find_current_case(case_df)
        loads = {
            "external": float(args.external if args.external is not None else current_row["P83 - external"]),
            "meeting": float(args.meeting if args.meeting is not None else current_row["P84 - meeting"]),
            "server": float(args.server if args.server is not None else current_row["P85 - server"]),
            "working": float(args.working if args.working is not None else current_row["P86 - working"]),
        }
        opt = optimize_hvac(
            model, case_df, loads, cond_scaler, coords, coords_norm_t, field_scaler, ra_scaler,
            device, save_dir, zone_json=args.zone_json, target_temp_c=args.target_temp,
            comfort_band_c=args.comfort_band, max_zone_range_c=args.max_zone_range,
            max_hot_fraction=args.max_hot_fraction, max_cold_fraction=args.max_cold_fraction,
            max_p95_temp_c=args.max_p95_temp, energy_weight=args.energy_weight,
        )
        print("\n[Top HVAC candidates]")
        print(opt.head(10).to_string(index=False))

        comparison = compare_current_vs_ai(
            opt, case_df, data, coords, loads, save_dir, args.zone_json,
            args.target_temp, args.comfort_band, args.max_zone_range,
            args.max_hot_fraction, args.max_cold_fraction, args.max_p95_temp,
        )
        print("\n[Current vs AI recommendations]")
        for item in comparison["comparisons"]:
            print(
                f"  {item['policy']:13s} | zone-range reduction={item['zone_range_reduction_pct']:.2f}% "
                f"| hot-area change={item['hot_fraction_reduction_percentage_points']:+.2f}pp "
                f"| sensible cooling saving={item['estimated_sensible_cooling_capacity_saving_pct']:.2f}%"
            )
        print(f"[Saved] {save_dir / 'current_vs_ai_comparison.csv'}")

        if args.export_influence:
            best = opt.iloc[0]
            base = np.asarray([
                best["Inlet_L"], best["Inlet_M"], best["Inlet_R"],
                loads["external"], loads["meeting"], loads["server"], loads["working"],
                best["CMM"], best["AirTemp_C"],
            ], dtype=np.float32)
            export_influence_maps(
                model, base, case_df, cond_scaler, coords_norm_t, field_scaler, ra_scaler,
                device, save_dir
            )
            print("[Saved] thermal influence maps")


# ============================================================
# 11. Optimization-only pipeline
# ============================================================


def _resolve_requested_loads(case_df: pd.DataFrame, args) -> Dict[str, float]:
    """Use explicit CLI loads; otherwise default to the dataset's DP 0 (Current) loads."""
    current = find_current_case(case_df)
    return {
        "external": float(args.external if args.external is not None else current["P83 - external"]),
        "meeting": float(args.meeting if args.meeting is not None else current["P84 - meeting"]),
        "server": float(args.server if args.server is not None else current["P85 - server"]),
        "working": float(args.working if args.working is not None else current["P86 - working"]),
    }


def _load_comparison_data(args, checkpoint_path: str | Path) -> Dict[str, np.ndarray]:
    """Load/reuse CFD cache so DP 0 actual CFD can be used when loads match Current."""
    cache_path = Path(checkpoint_path).resolve().parent / "cfd_cache.npz"
    return build_or_load_cache(
        args.case_info, args.field_zip, cache_path, force_rebuild=args.force_rebuild_cache
    )


def run_optimize(args) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in optimize mode")
    save_dir = ensure_dir(args.save_dir)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt, model, scalers, coords = load_checkpoint(args.checkpoint, device)
    coords_norm_t = torch.from_numpy(scalers["coord"].transform(coords).astype(np.float32)).to(device)
    case_df = load_case_info(args.case_info)
    data = _load_comparison_data(args, args.checkpoint)
    loads = _resolve_requested_loads(case_df, args)

    opt = optimize_hvac(
        model, case_df, loads, scalers["cond"], coords, coords_norm_t, scalers["field"], scalers["ra"],
        device, save_dir, zone_json=args.zone_json, target_temp_c=args.target_temp,
        comfort_band_c=args.comfort_band, max_zone_range_c=args.max_zone_range,
        max_hot_fraction=args.max_hot_fraction, max_cold_fraction=args.max_cold_fraction,
        max_p95_temp_c=args.max_p95_temp, energy_weight=args.energy_weight,
    )
    print("\n[Top HVAC candidates]")
    print(opt.head(10).to_string(index=False))

    comparison = compare_current_vs_ai(
        opt, case_df, data, coords, loads, save_dir, args.zone_json,
        args.target_temp, args.comfort_band, args.max_zone_range,
        args.max_hot_fraction, args.max_cold_fraction, args.max_p95_temp,
    )
    print("\n[Current vs AI recommendations]")
    print(f"  Baseline: {comparison['baseline_source']}")
    for item in comparison["comparisons"]:
        print(
            f"  {item['policy']:13s} | zone-range reduction={item['zone_range_reduction_pct']:.2f}% "
            f"| hot-area change={item['hot_fraction_reduction_percentage_points']:+.2f}pp "
            f"| sensible cooling saving={item['estimated_sensible_cooling_capacity_saving_pct']:.2f}%"
        )

    if args.export_influence:
        best = opt.iloc[0]
        base = np.asarray([
            best["Inlet_L"], best["Inlet_M"], best["Inlet_R"],
            loads["external"], loads["meeting"], loads["server"], loads["working"],
            best["CMM"], best["AirTemp_C"],
        ], dtype=np.float32)
        export_influence_maps(
            model, base, case_df, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"],
            device, save_dir
        )


def run_recommend(args) -> None:
    """
    Low-latency decision-support mode (NOT RL / dynamic closed-loop control).

    Input: current heat loads.
    Output: Balanced / Comfort-first / Eco-first HVAC actions after evaluating all candidate actions.
    Also reports batched model inference latency for all candidates and end-to-end decision latency.
    """
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in recommend mode")
    save_dir = ensure_dir(args.save_dir)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    _, model, scalers, coords = load_checkpoint(args.checkpoint, device)
    coords_norm_t = torch.from_numpy(scalers["coord"].transform(coords).astype(np.float32)).to(device)
    case_df = load_case_info(args.case_info)
    data = _load_comparison_data(args, args.checkpoint)
    loads = _resolve_requested_loads(case_df, args)

    actions = enumerate_observed_action_space(case_df)
    conds = np.asarray([
        [l, m, r, loads["external"], loads["meeting"], loads["server"], loads["working"], cmm, supply]
        for l, m, r, cmm, supply in actions
    ], dtype=np.float32)

    # Warm-up then time the batched surrogate inference for all HVAC candidates.
    for _ in range(max(int(args.latency_warmup), 0)):
        _ = predict_conditions(
            model, conds, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"], device
        )
    times_ms = []
    for _ in range(max(int(args.latency_repeat), 1)):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_conditions(
            model, conds, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"], device
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    t_all = time.perf_counter()
    opt = optimize_hvac(
        model, case_df, loads, scalers["cond"], coords, coords_norm_t, scalers["field"], scalers["ra"],
        device, save_dir, zone_json=args.zone_json, target_temp_c=args.target_temp,
        comfort_band_c=args.comfort_band, max_zone_range_c=args.max_zone_range,
        max_hot_fraction=args.max_hot_fraction, max_cold_fraction=args.max_cold_fraction,
        max_p95_temp_c=args.max_p95_temp, energy_weight=args.energy_weight,
    )
    comparison = compare_current_vs_ai(
        opt, case_df, data, coords, loads, save_dir, args.zone_json,
        args.target_temp, args.comfort_band, args.max_zone_range,
        args.max_hot_fraction, args.max_cold_fraction, args.max_p95_temp,
    )
    end_to_end_ms = (time.perf_counter() - t_all) * 1000.0

    recommendations = json.loads((save_dir / "hvac_recommendations.json").read_text(encoding="utf-8"))
    latency = {
        "num_hvac_candidates": int(len(actions)),
        "model_batch_inference_ms_mean": float(np.mean(times_ms)),
        "model_batch_inference_ms_median": float(np.median(times_ms)),
        "model_batch_inference_ms_min": float(np.min(times_ms)),
        "end_to_end_decision_ms_including_ranking_and_file_output": float(end_to_end_ms),
        "device": device,
        "note": "Decision-support latency only; the supplied CFD dataset has no temporal transitions, so this is not RL or dynamic closed-loop control.",
    }
    payload = {
        "loads_W": loads,
        "latency": latency,
        "recommendations": recommendations,
        "current_vs_ai": comparison,
    }
    (save_dir / "realtime_recommendation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("LOW-LATENCY HVAC DECISION SUPPORT")
    print(f"Loads [W]: {loads}")
    print(f"Candidates evaluated: {len(actions)}")
    print(f"Batched PopField inference: median={latency['model_batch_inference_ms_median']:.3f} ms")
    print(f"End-to-end decision: {end_to_end_ms:.3f} ms")

    if bool(recommendations.get("fully_feasible_action_exists", False)):
        print("Status: FEASIBLE — at least one HVAC action satisfies all configured comfort constraints.")
        for key in ["balanced", "comfort_first", "eco_first"]:
            r = recommendations[key]
            print(
                f"{key:13s}: L/M/R={int(r['Inlet_L'])}/{int(r['Inlet_M'])}/{int(r['Inlet_R'])}, "
                f"CMM={r['CMM']:.0f}, Supply={r['AirTemp_C']:.0f}C, "
                f"ZoneRange={r['zone_range_C']:.3f}C, Hot={100*r['hot_fraction']:.2f}%, "
                f"Q_sensible={r['estimated_sensible_cooling_kw']:.3f}kW"
            )
    else:
        r = recommendations["best_achievable"]
        gap = recommendations["additional_capacity_estimate"]
        failed = recommendations.get("failed_constraints_for_best_achievable", [])
        print("Status: NO FEASIBLE ACTION")
        print("WARNING: No evaluated HVAC action satisfies all configured comfort constraints.")
        print("Balanced / Comfort-first / Eco-first recommendations are suppressed to avoid unsafe labeling.")
        print(
            f"best_achievable: L/M/R={int(r['Inlet_L'])}/{int(r['Inlet_M'])}/{int(r['Inlet_R'])}, "
            f"CMM={r['CMM']:.0f}, Supply={r['AirTemp_C']:.0f}C, "
            f"ZoneRange={r['zone_range_C']:.3f}C, Hot={100*r['hot_fraction']:.2f}%, "
            f"Cold={100*r['cold_fraction']:.2f}%, P95={r['p95_temp_C']:.3f}C, "
            f"Q_sensible={r['estimated_sensible_cooling_kw']:.3f}kW"
        )
        print(f"Failed constraints: {failed}")
        print(
            "Additional sensible cooling lower-bound estimate: "
            f"{gap['additional_sensible_cooling_kw_lower_bound_at_best_achievable']:.3f} kW "
            "(load-balance lower bound; NOT exact equipment sizing)."
        )
        if bool(gap.get("air_distribution_may_be_limiting", False)):
            print("Diagnostic: Total candidate cooling capacity can cover the specified heat load, so airflow distribution/local hotspot removal may be the limiting factor.")
        else:
            print(
                "Diagnostic: Even the evaluated cooling capacity is below the specified sensible heat load by at least "
                f"{gap['additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity']:.3f} kW at maximum candidate capacity."
            )
    print(f"[Saved] {save_dir / 'realtime_recommendation.json'}")
    print("=" * 100)



# ============================================================
# 12. Comprehensive experiment pipeline
# ============================================================


def _split_label_array(n_cases: int, split: Dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full(n_cases, "unknown", dtype=object)
    for name in ["train", "val", "test"]:
        if name in split:
            labels[np.asarray(split[name], dtype=int)] = name
    return labels


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def evaluate_all_observed_cfd_cases(
    model: PopFieldHVACTwin,
    ckpt: Dict,
    scalers: Dict[str, Standardizer],
    coords: np.ndarray,
    data: Dict[str, np.ndarray],
    case_df: pd.DataFrame,
    device: str,
    save_dir: Path,
    zone_json: Optional[str],
    target_temp_c: float,
    comfort_band_c: float,
    max_zone_range_c: float,
    max_hot_fraction: float,
    max_cold_fraction: float,
    max_p95_temp_c: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Evaluate the trained surrogate on every one of the 200 observed CFD design points."""
    if max_p95_temp_c is None:
        max_p95_temp_c = float(target_temp_c + comfort_band_c)

    conditions = np.asarray(data["conditions"], dtype=np.float32)
    true_field = np.asarray(data["fields"], dtype=np.float32)
    true_ra = np.asarray(data["ra_temp_c"], dtype=np.float32)
    dp_ids = np.asarray(data["dp_ids"], dtype=int)
    coords_norm_t = torch.from_numpy(scalers["coord"].transform(coords).astype(np.float32)).to(device)

    t0 = time.perf_counter()
    pred_field, pred_ra = predict_conditions(
        model, conditions, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"], device,
        batch_size=64,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    split_labels = _split_label_array(len(dp_ids), ckpt.get("split", {}))
    zones = build_zone_masks(coords, zone_json)
    case_lookup = {int(r.dp_id): r for r in case_df.itertuples(index=False)}

    rows = []
    for i, dp_id in enumerate(dp_ids):
        t_true = true_field[i, :, 0]
        t_pred = pred_field[i, :, 0]
        v_true = true_field[i, :, 1:4]
        v_pred = pred_field[i, :, 1:4]
        abs_t = np.abs(t_pred - t_true)
        vel_err = v_pred - v_true

        cond = conditions[i]
        action = (int(cond[0]), int(cond[1]), int(cond[2]), float(cond[7]), float(cond[8]))
        actual_cm = field_comfort_metrics(
            t_true, v_true, float(true_ra[i]), action, zones,
            target_temp_c, comfort_band_c, max_zone_range_c,
            max_hot_fraction, max_cold_fraction, float(max_p95_temp_c),
        )
        pred_cm = field_comfort_metrics(
            t_pred, v_pred, float(pred_ra[i]), action, zones,
            target_temp_c, comfort_band_c, max_zone_range_c,
            max_hot_fraction, max_cold_fraction, float(max_p95_temp_c),
        )

        row = {
            "case_index": int(i),
            "dp_id": int(dp_id),
            "split": str(split_labels[i]),
            "temp_mae_c": float(np.mean(abs_t)),
            "temp_rmse_c": float(np.sqrt(np.mean((t_pred - t_true) ** 2))),
            "temp_p95_abs_error_c": float(np.percentile(abs_t, 95)),
            "temp_max_abs_error_c": float(np.max(abs_t)),
            "velocity_component_mae": float(np.mean(np.abs(vel_err))),
            "velocity_vector_rmse": float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=-1)))),
            "ra_abs_error_c": float(abs(float(pred_ra[i]) - float(true_ra[i]))),
            "actual_zone_range_c": float(actual_cm["zone_range_C"]),
            "pred_zone_range_c": float(pred_cm["zone_range_C"]),
            "zone_range_abs_error_c": float(abs(float(pred_cm["zone_range_C"]) - float(actual_cm["zone_range_C"]))),
            "actual_hot_fraction": float(actual_cm["hot_fraction"]),
            "pred_hot_fraction": float(pred_cm["hot_fraction"]),
            "hot_fraction_abs_error": float(abs(float(pred_cm["hot_fraction"]) - float(actual_cm["hot_fraction"]))),
            "actual_cold_fraction": float(actual_cm["cold_fraction"]),
            "pred_cold_fraction": float(pred_cm["cold_fraction"]),
            "actual_p95_temp_c": float(actual_cm["p95_temp_C"]),
            "pred_p95_temp_c": float(pred_cm["p95_temp_C"]),
            "actual_comfort_feasible": bool(actual_cm["comfort_constraint_met"]),
            "pred_comfort_feasible": bool(pred_cm["comfort_constraint_met"]),
            "comfort_feasibility_match": bool(actual_cm["comfort_constraint_met"] == pred_cm["comfort_constraint_met"]),
            "Inlet_L": int(action[0]), "Inlet_M": int(action[1]), "Inlet_R": int(action[2]),
            "external_W": float(cond[3]), "meeting_W": float(cond[4]),
            "server_W": float(cond[5]), "working_W": float(cond[6]),
            "CMM": float(action[3]), "AirTemp_C": float(action[4]),
        }
        if int(dp_id) in case_lookup:
            row["case_name"] = str(getattr(case_lookup[int(dp_id)], "Name", f"DP {dp_id}"))
        rows.append(row)

    per_case = pd.DataFrame(rows)
    per_case.to_csv(save_dir / "all_200_case_prediction_metrics.csv", index=False)

    agg_rows = []
    for split_name in ["train", "val", "test", "overall"]:
        sub = per_case if split_name == "overall" else per_case[per_case["split"] == split_name]
        if len(sub) == 0:
            continue
        agg_rows.append({
            "split": split_name,
            "num_cases": int(len(sub)),
            "temp_mae_c_mean": float(sub["temp_mae_c"].mean()),
            "temp_rmse_c_global_approx": float(np.sqrt(np.mean(sub["temp_rmse_c"].to_numpy(float) ** 2))),
            "temp_p95_abs_error_c_mean": float(sub["temp_p95_abs_error_c"].mean()),
            "temp_max_abs_error_c_max": float(sub["temp_max_abs_error_c"].max()),
            "velocity_component_mae_mean": float(sub["velocity_component_mae"].mean()),
            "ra_abs_error_c_mean": float(sub["ra_abs_error_c"].mean()),
            "zone_range_abs_error_c_mean": float(sub["zone_range_abs_error_c"].mean()),
            "hot_fraction_abs_error_mean": float(sub["hot_fraction_abs_error"].mean()),
            "comfort_feasibility_accuracy": float(sub["comfort_feasibility_match"].mean()),
        })
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(save_dir / "prediction_metrics_by_split.csv", index=False)

    timing = {
        "num_cases": int(len(dp_ids)),
        "num_nodes_per_case": int(coords.shape[0]),
        "batched_prediction_all_200_cases_ms": float(elapsed_ms),
        "ms_per_case_amortized": float(elapsed_ms / max(len(dp_ids), 1)),
        "device": device,
    }
    (save_dir / "all_case_inference_timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return per_case, aggregate, pred_field, pred_ra


def run_all_load_hvac_experiment(
    model: PopFieldHVACTwin,
    scalers: Dict[str, Standardizer],
    coords: np.ndarray,
    data: Dict[str, np.ndarray],
    case_df: pd.DataFrame,
    device: str,
    save_dir: Path,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all 54 observed HVAC actions for every UNIQUE observed heat-load combination.
    Every one of the 200 CFD cases is represented through its load-combination membership, while
    duplicate load combinations are computed only once.
    """
    load_cols = ["P83 - external", "P84 - meeting", "P85 - server", "P86 - working"]
    observed_counts = (
        case_df.groupby(load_cols, dropna=False)
        .size().reset_index(name="observed_case_count")
    )
    if getattr(args, "full_load_space", "combinatorial") == "observed":
        load_table = observed_counts.copy()
    else:
        # Full Cartesian product of every heat-load level supplied by the dataset.
        # This covers both the 200 observed CFD scenarios and valid but unobserved combinations.
        import itertools
        levels = [sorted(case_df[c].astype(float).unique().tolist()) for c in load_cols]
        combos = list(itertools.product(*levels))
        load_table = pd.DataFrame(combos, columns=load_cols)
        load_table = load_table.merge(observed_counts, on=load_cols, how="left")
        load_table["observed_case_count"] = load_table["observed_case_count"].fillna(0).astype(int)
    load_table["observed_in_cfd"] = load_table["observed_case_count"].astype(int) > 0
    load_table = load_table.sort_values(load_cols).reset_index(drop=True)
    max_groups = int(getattr(args, "full_max_load_groups", 0) or 0)
    if max_groups > 0:
        load_table = load_table.head(max_groups).reset_index(drop=True)
    coords_norm_t = torch.from_numpy(scalers["coord"].transform(coords).astype(np.float32)).to(device)
    temp_dir = ensure_dir(save_dir / "_tmp_one_load")

    candidate_tables = []
    rec_rows = []
    comparison_rows = []
    status_rows = []
    t_start = time.perf_counter()

    for load_id, lr in load_table.iterrows():
        loads = {
            "external": float(lr["P83 - external"]),
            "meeting": float(lr["P84 - meeting"]),
            "server": float(lr["P85 - server"]),
            "working": float(lr["P86 - working"]),
        }
        opt = optimize_hvac(
            model, case_df, loads, scalers["cond"], coords, coords_norm_t, scalers["field"], scalers["ra"],
            device, temp_dir, zone_json=args.zone_json, target_temp_c=args.target_temp,
            comfort_band_c=args.comfort_band, max_zone_range_c=args.max_zone_range,
            max_hot_fraction=args.max_hot_fraction, max_cold_fraction=args.max_cold_fraction,
            max_p95_temp_c=args.max_p95_temp, energy_weight=args.energy_weight,
        )
        opt = opt.copy()
        opt.insert(0, "load_id", int(load_id))
        opt.insert(1, "observed_case_count", int(lr["observed_case_count"]))
        opt.insert(2, "observed_in_cfd", bool(lr["observed_in_cfd"]))
        opt.insert(2, "external_W", loads["external"])
        opt.insert(3, "meeting_W", loads["meeting"])
        opt.insert(4, "server_W", loads["server"])
        opt.insert(5, "working_W", loads["working"])
        candidate_tables.append(opt)

        rec = json.loads((temp_dir / "hvac_recommendations.json").read_text(encoding="utf-8"))
        has_feasible = bool(rec.get("fully_feasible_action_exists", False))
        if has_feasible:
            policies = [("balanced", rec["balanced"]), ("comfort_first", rec["comfort_first"]), ("eco_first", rec["eco_first"])]
            failed = []
            add_gap = 0.0
            add_gap_max = 0.0
        else:
            policies = [("best_achievable", rec["best_achievable"])]
            failed = list(rec.get("failed_constraints_for_best_achievable", []))
            gap = rec.get("additional_capacity_estimate", {})
            add_gap = float(gap.get("additional_sensible_cooling_kw_lower_bound_at_best_achievable", float("nan")))
            add_gap_max = float(gap.get("additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity", float("nan")))

        for policy, r in policies:
            rec_rows.append({
                "load_id": int(load_id),
                "observed_case_count": int(lr["observed_case_count"]),
                "observed_in_cfd": bool(lr["observed_in_cfd"]),
                "external_W": loads["external"], "meeting_W": loads["meeting"],
                "server_W": loads["server"], "working_W": loads["working"],
                "total_specified_heat_load_kw": float(sum(loads.values()) / 1000.0),
                "status": "FEASIBLE" if has_feasible else "NO_FEASIBLE_ACTION",
                "policy": policy,
                "Inlet_L": int(r["Inlet_L"]), "Inlet_M": int(r["Inlet_M"]), "Inlet_R": int(r["Inlet_R"]),
                "CMM": float(r["CMM"]), "AirTemp_C": float(r["AirTemp_C"]),
                "zone_range_C": float(r["zone_range_C"]),
                "hot_fraction": float(r["hot_fraction"]), "cold_fraction": float(r["cold_fraction"]),
                "p95_temp_C": float(r["p95_temp_C"]), "max_temp_C": float(r["max_temp_C"]),
                "estimated_sensible_cooling_kw": float(r["estimated_sensible_cooling_kw"]),
                "comfort_constraint_met": bool(r["comfort_constraint_met"]),
                "constraint_violation_score": float(r["constraint_violation_score"]),
                "failed_constraints": "|".join(failed),
                "additional_cooling_kw_lower_bound_best": add_gap,
                "additional_cooling_kw_lower_bound_even_at_max_capacity": add_gap_max,
            })

        comp = compare_current_vs_ai(
            opt, case_df, data, coords, loads, temp_dir, args.zone_json,
            args.target_temp, args.comfort_band, args.max_zone_range,
            args.max_hot_fraction, args.max_cold_fraction, args.max_p95_temp,
        )
        for cr in comp.get("comparisons", []):
            rr = dict(cr)
            rr.update({
                "load_id": int(load_id),
                "observed_case_count": int(lr["observed_case_count"]),
                "observed_in_cfd": bool(lr["observed_in_cfd"]),
                "external_W": loads["external"], "meeting_W": loads["meeting"],
                "server_W": loads["server"], "working_W": loads["working"],
                "status": "FEASIBLE" if has_feasible else "NO_FEASIBLE_ACTION",
            })
            comparison_rows.append(rr)

        status_rows.append({
            "load_id": int(load_id),
            "observed_case_count": int(lr["observed_case_count"]),
            "observed_in_cfd": bool(lr["observed_in_cfd"]),
            "external_W": loads["external"], "meeting_W": loads["meeting"],
            "server_W": loads["server"], "working_W": loads["working"],
            "total_specified_heat_load_kw": float(sum(loads.values()) / 1000.0),
            "status": "FEASIBLE" if has_feasible else "NO_FEASIBLE_ACTION",
            "num_feasible_actions": int(opt["comfort_constraint_met"].sum()),
            "best_min_violation_score": float(opt["constraint_violation_score"].min()),
            "failed_constraints_best_achievable": "|".join(failed),
            "additional_cooling_kw_lower_bound_best": add_gap,
            "additional_cooling_kw_lower_bound_even_at_max_capacity": add_gap_max,
        })

        if (load_id + 1) % 10 == 0 or load_id + 1 == len(load_table):
            print(f"[Full experiment] HVAC load groups {load_id + 1}/{len(load_table)} completed")

    elapsed = time.perf_counter() - t_start
    all_candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    recommendations = pd.DataFrame(rec_rows)
    comparisons = pd.DataFrame(comparison_rows)
    statuses = pd.DataFrame(status_rows)

    all_candidates.to_csv(save_dir / "all_load_hvac_candidates.csv", index=False)
    recommendations.to_csv(save_dir / "all_load_recommendations.csv", index=False)
    comparisons.to_csv(save_dir / "all_load_current_vs_ai.csv", index=False)
    statuses.to_csv(save_dir / "all_load_feasibility.csv", index=False)

    timing = {
        "load_space_mode": str(getattr(args, "full_load_space", "combinatorial")),
        "heat_load_combinations_evaluated": int(len(load_table)),
        "combinations_observed_in_cfd": int(load_table["observed_in_cfd"].sum()),
        "combinations_unobserved_in_cfd": int((~load_table["observed_in_cfd"]).sum()),
        "observed_cfd_cases_covered": int(load_table["observed_case_count"].sum()),
        "hvac_actions_per_load": int(len(enumerate_observed_action_space(case_df))),
        "total_counterfactual_actions_evaluated": int(len(all_candidates)),
        "total_seconds": float(elapsed),
        "seconds_per_unique_load_average": float(elapsed / max(len(load_table), 1)),
        "device": device,
    }
    (save_dir / "all_load_experiment_timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Clean temporary per-load files; all information is already aggregated above.
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    return recommendations, comparisons, statuses


def create_full_experiment_plots(
    prediction_by_split: pd.DataFrame,
    sensor_table: pd.DataFrame,
    feasibility: pd.DataFrame,
    recommendations: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Save simple publication/presentation-ready diagnostic plots using matplotlib defaults."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plots] skipped: {e}")
        return

    fig_dir = ensure_dir(save_dir / "figures")

    if len(prediction_by_split):
        p = prediction_by_split[prediction_by_split["split"].isin(["train", "val", "test"])]
        if len(p):
            plt.figure(figsize=(7, 4.5))
            plt.bar(p["split"], p["temp_mae_c_mean"])
            plt.ylabel("Temperature MAE (°C)")
            plt.xlabel("Split")
            plt.title("PopField Temperature Prediction")
            plt.tight_layout()
            plt.savefig(fig_dir / "prediction_mae_by_split.png", dpi=220)
            plt.close()

    if len(sensor_table):
        plt.figure(figsize=(7, 4.5))
        plt.plot(sensor_table["num_sensors"], sensor_table["val_sensor_reconstruction_mae_c"], marker="o", label="Validation")
        plt.plot(sensor_table["num_sensors"], sensor_table["test_sensor_reconstruction_mae_c"], marker="o", label="Test")
        plt.xlabel("Number of sensors")
        plt.ylabel("Reconstruction MAE (°C)")
        plt.title("Sparse Sensor Reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "sensor_count_vs_mae.png", dpi=220)
        plt.close()

    if len(feasibility):
        counts = feasibility["status"].value_counts()
        plt.figure(figsize=(6, 4.5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.ylabel("Unique heat-load combinations")
        plt.title("HVAC Feasibility Across Observed Load Conditions")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(fig_dir / "feasibility_summary.png", dpi=220)
        plt.close()

    if len(recommendations):
        sub = recommendations[recommendations["policy"].isin(["balanced", "best_achievable"])]
        if len(sub):
            plt.figure(figsize=(7, 4.5))
            plt.scatter(sub["total_specified_heat_load_kw"], sub["hot_fraction"] * 100.0, s=28)
            plt.xlabel("Specified sensible heat load (kW)")
            plt.ylabel("Hot area (%)")
            plt.title("Recommended/Best-Achievable Hotspot Risk")
            plt.tight_layout()
            plt.savefig(fig_dir / "heat_load_vs_hot_area.png", dpi=220)
            plt.close()



def _fmt_num(x, digits: int = 4) -> str:
    """Human-readable numeric formatter for the final console report."""
    try:
        v = float(x)
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.{digits}f}"
    except Exception:
        return str(x)


def build_full_console_report(
    save_dir: Path,
    summary: Dict,
    per_case: pd.DataFrame,
    pred_split: pd.DataFrame,
    sensor_table: pd.DataFrame,
    sensor_result: Dict,
    feasibility: pd.DataFrame,
    recommendations: pd.DataFrame,
    comparisons: pd.DataFrame,
    comparison_agg: pd.DataFrame,
    action_freq: pd.DataFrame,
    failure_counter: Dict[str, int],
    latency_summary: Dict,
    args,
) -> str:
    """
    Build one comprehensive text report that is BOTH printed to the Colab output window
    and saved as FULL_CONSOLE_REPORT.txt.

    Default console_detail='full' prints:
      - training summary if training_history.csv exists
      - all split prediction metrics
      - all 200 per-case prediction rows (selected key columns)
      - complete sensor-count table + selected sensor coordinates
      - all heat-load feasibility rows
      - all per-load recommendations (Balanced/Comfort/Eco or Best-achievable)
      - Current-vs-AI aggregate AND all per-load comparison rows
      - failed-constraint counts
      - action-selection frequency
      - cooling-capacity-gap statistics
      - latency and limitations

    Use --print_every_candidate to additionally print every HVAC candidate row
    (up to 7,776 rows for 144 load combinations x 54 actions).
    """
    lines: List[str] = []
    W = 118

    def hr(ch: str = "="):
        lines.append(ch * W)

    def title(text: str):
        lines.append("")
        hr("=")
        lines.append(text)
        hr("=")

    def section(text: str):
        lines.append("")
        hr("-")
        lines.append(text)
        hr("-")

    def table(df: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: Optional[int] = None):
        if df is None or len(df) == 0:
            lines.append("(no rows)")
            return
        x = df.copy()
        if columns is not None:
            cols = [c for c in columns if c in x.columns]
            x = x[cols]
        if max_rows is not None and len(x) > max_rows:
            lines.append(x.head(max_rows).to_string(index=False))
            lines.append(f"... ({len(x)-max_rows} additional rows omitted from console)")
        else:
            lines.append(x.to_string(index=False))

    title("POPFIELD HVAC — FINAL ALL-RESULTS CONSOLE REPORT")
    lines.append(f"Results directory : {save_dir}")
    lines.append(f"Checkpoint        : {summary.get('checkpoint')}")
    lines.append(f"Device            : {latency_summary.get('device')}")
    lines.append(f"Model parameters  : {summary['model']['params']:,}")
    lines.append(f"Observed CFD cases: {summary['model']['num_observed_cfd_cases']}")
    lines.append(f"Spatial nodes/case: {summary['model']['num_nodes']}")
    lines.append(f"Load-space mode   : {summary['hvac_counterfactual'].get('load_space_mode')}")

    # 0) Training summary, when this run used train_full or shares the training directory.
    training_csv = save_dir / "training_history.csv"
    if training_csv.exists():
        section("[0] TRAINING SUMMARY")
        try:
            h = pd.read_csv(training_csv)
            lines.append(f"Epochs actually run : {len(h)}")
            if "val_temp_mae_c" in h.columns:
                bi = int(h["val_temp_mae_c"].astype(float).idxmin())
                lines.append(f"Best epoch          : {int(h.loc[bi, 'epoch']) if 'epoch' in h.columns else bi+1}")
                lines.append(f"Best val temp MAE   : {_fmt_num(h.loc[bi, 'val_temp_mae_c'])} C")
            elif "val_temp_mae" in h.columns:
                bi = int(h["val_temp_mae"].astype(float).idxmin())
                lines.append(f"Best epoch          : {int(h.loc[bi, 'epoch']) if 'epoch' in h.columns else bi+1}")
                lines.append(f"Best val temp MAE   : {_fmt_num(h.loc[bi, 'val_temp_mae'])} C")
            if "lr" in h.columns:
                lines.append(f"Final learning rate : {h.iloc[-1]['lr']}")
            lines.append("\nTraining history (all epochs):")
            table(h)
        except Exception as e:
            lines.append(f"Could not read training history: {e}")
    else:
        section("[0] TRAINING SUMMARY")
        lines.append("training_history.csv not found in this results directory.")
        lines.append("This usually means an existing best.pt was reused with --mode full_experiment.")

    # 1) Prediction
    section("[1] POPFIELD PREDICTION — SPLIT SUMMARY")
    table(pred_split)
    lines.append("")
    lines.append("Headline test metrics")
    lines.append(f"  Test temperature MAE                 : {_fmt_num(summary['prediction']['test_temp_mae_c_mean'])} C")
    lines.append(f"  Test mean P95 absolute temp error    : {_fmt_num(summary['prediction']['test_temp_p95_abs_error_c_mean'])} C")
    lines.append(f"  Test RA absolute error               : {_fmt_num(summary['prediction']['test_ra_abs_error_c_mean'])} C")
    lines.append(f"  Comfort-feasibility classification   : {_fmt_num(100*summary['prediction']['test_comfort_feasibility_accuracy'], 2)} %")

    if str(getattr(args, "console_detail", "full")) in {"full", "everything"}:
        section("[1-A] ALL OBSERVED CFD CASES — PER-CASE RESULTS")
        per_cols = [
            "case_index","dp_id","case_name","split","temp_mae_c","temp_rmse_c",
            "temp_p95_abs_error_c","temp_max_abs_error_c","ra_abs_error_c",
            "zone_range_abs_error_c","hot_fraction_abs_error","comfort_feasibility_match",
            "external_W","meeting_W","server_W","working_W","Inlet_L","Inlet_M","Inlet_R","CMM","AirTemp_C"
        ]
        table(per_case.sort_values(["split","case_index"]), per_cols)

        section("[1-B] TEST CASES — BEST/WORST TEMPERATURE MAE")
        test_cases = per_case[per_case["split"] == "test"].copy()
        if len(test_cases):
            lines.append("Best 10 test cases:")
            table(test_cases.nsmallest(min(10,len(test_cases)), "temp_mae_c"), ["dp_id","case_name","temp_mae_c","temp_rmse_c","temp_p95_abs_error_c","ra_abs_error_c"])
            lines.append("\nWorst 10 test cases:")
            table(test_cases.nlargest(min(10,len(test_cases)), "temp_mae_c"), ["dp_id","case_name","temp_mae_c","temp_rmse_c","temp_p95_abs_error_c","ra_abs_error_c"])

    # 2) Sensors
    section("[2] SPARSE SENSOR DESIGN — ALL SENSOR COUNTS")
    table(sensor_table)
    lines.append("")
    lines.append(f"Chosen number of sensors : {sensor_result['chosen_num_sensors']}")
    lines.append(f"Selection reason          : {sensor_result.get('selection_reason','')}")
    lines.append(f"Validation reconstruction : {_fmt_num(sensor_result['chosen_validation_mae_c'])} C")
    lines.append(f"Test reconstruction       : {_fmt_num(sensor_result['chosen_test_mae_c'])} C")
    selected_csv = save_dir / "selected_sensors.csv"
    if selected_csv.exists():
        lines.append("\nSelected sensor locations:")
        try:
            table(pd.read_csv(selected_csv))
        except Exception as e:
            lines.append(f"Could not read selected sensors: {e}")

    # 3) HVAC global KPIs
    section("[3] HVAC COUNTERFACTUAL SEARCH — GLOBAL SUMMARY")
    hv = summary["hvac_counterfactual"]
    lines.append(f"Heat-load combinations evaluated : {hv['heat_load_combinations_evaluated']}")
    lines.append(f"  Observed in supplied CFD        : {hv.get('combinations_observed_in_cfd', 0)}")
    lines.append(f"  Unobserved combinations         : {hv.get('combinations_unobserved_in_cfd', 0)}")
    lines.append(f"HVAC actions per load             : {hv['actions_per_load']}")
    lines.append(f"TOTAL counterfactual actions      : {hv['total_counterfactual_actions_evaluated']}")
    lines.append(f"FEASIBLE load combinations        : {hv['feasible_load_combinations']}")
    lines.append(f"NO-FEASIBLE load combinations     : {hv['no_feasible_load_combinations']}")
    lines.append(f"Feasible rate                     : {_fmt_num(hv['feasible_load_combination_rate_pct'],2)} %")
    lines.append(f"Observed-case weighted rate       : {_fmt_num(hv['observed_case_weighted_feasible_rate_pct'],2)} %")

    # 4) All load feasibility rows
    section("[4] ALL HEAT-LOAD CONDITIONS — FEASIBILITY / FACILITY-LIMIT RESULTS")
    feas_cols = [
        "load_id","observed_in_cfd","observed_case_count","external_W","meeting_W","server_W","working_W",
        "total_specified_heat_load_kw","status","num_feasible_actions","best_min_violation_score",
        "failed_constraints_best_achievable","additional_cooling_kw_lower_bound_best",
        "additional_cooling_kw_lower_bound_even_at_max_capacity"
    ]
    table(feasibility.sort_values("load_id"), feas_cols)

    infeas = feasibility[feasibility["status"] == "NO_FEASIBLE_ACTION"].copy()
    if len(infeas):
        lines.append("")
        lines.append("Facility-limit / additional-cooling statistics for NO-FEASIBLE conditions")
        best_gap = infeas["additional_cooling_kw_lower_bound_best"].astype(float)
        maxcap_gap = infeas["additional_cooling_kw_lower_bound_even_at_max_capacity"].astype(float)
        lines.append(f"  Count                                : {len(infeas)}")
        lines.append(f"  Mean lower bound at best-achievable  : {_fmt_num(best_gap.mean())} kW")
        lines.append(f"  Median lower bound                   : {_fmt_num(best_gap.median())} kW")
        lines.append(f"  Max lower bound                      : {_fmt_num(best_gap.max())} kW")
        lines.append(f"  Mean unavoidable gap at max capacity : {_fmt_num(maxcap_gap.mean())} kW")
        lines.append(f"  Max unavoidable gap at max capacity  : {_fmt_num(maxcap_gap.max())} kW")

    # 5) All recommendations
    section("[5] ALL LOAD CONDITIONS — AI RECOMMENDATIONS")
    rec_cols = [
        "load_id","observed_in_cfd","external_W","meeting_W","server_W","working_W","total_specified_heat_load_kw",
        "status","policy","Inlet_L","Inlet_M","Inlet_R","CMM","AirTemp_C","zone_range_C","hot_fraction",
        "cold_fraction","p95_temp_C","max_temp_C","estimated_sensible_cooling_kw","comfort_constraint_met",
        "constraint_violation_score","failed_constraints","additional_cooling_kw_lower_bound_best"
    ]
    rec_print = recommendations.copy()
    if "hot_fraction" in rec_print:
        rec_print["hot_fraction_pct"] = rec_print["hot_fraction"].astype(float)*100
        rec_cols = [c if c != "hot_fraction" else "hot_fraction_pct" for c in rec_cols]
    if "cold_fraction" in rec_print:
        rec_print["cold_fraction_pct"] = rec_print["cold_fraction"].astype(float)*100
        rec_cols = [c if c != "cold_fraction" else "cold_fraction_pct" for c in rec_cols]
    table(rec_print.sort_values(["load_id","policy"]), rec_cols)

    # 6) Current vs AI
    section("[6] CURRENT HVAC vs AI — AGGREGATE IMPROVEMENT")
    if comparison_agg is not None and len(comparison_agg):
        table(comparison_agg)
    else:
        lines.append("No comparison rows were available.")

    if str(getattr(args, "console_detail", "full")) in {"full", "everything"}:
        section("[6-A] CURRENT HVAC vs AI — ALL LOAD-BY-LOAD COMPARISONS")
        cmp_cols = [
            "load_id","observed_in_cfd","external_W","meeting_W","server_W","working_W","status","policy",
            "zone_range_reduction_pct","spatial_std_reduction_pct","hot_fraction_reduction_percentage_points",
            "cold_fraction_reduction_percentage_points","estimated_sensible_cooling_capacity_saving_pct",
            "current_comfort_constraint_met","ai_comfort_constraint_met"
        ]
        table(comparisons.sort_values(["load_id","policy"]), cmp_cols)

    # 7) Failures
    section("[7] WHY HVAC FAILS — CONSTRAINT FAILURE COUNTS")
    if failure_counter:
        fail_df = pd.DataFrame([
            {"constraint": k, "num_load_conditions": v, "rate_pct_of_all_loads": 100.0*v/max(len(feasibility),1)}
            for k,v in sorted(failure_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ])
        table(fail_df)
    else:
        lines.append("No failed constraints: every evaluated load condition had at least one feasible action.")

    # 8) action frequencies
    section("[8] MOST FREQUENTLY SELECTED HVAC ACTIONS")
    if action_freq is not None and len(action_freq):
        topn = int(getattr(args, "console_top_actions", 20))
        for policy in action_freq["policy"].dropna().astype(str).unique().tolist():
            lines.append(f"\nPolicy = {policy}")
            sub = action_freq[action_freq["policy"].astype(str)==policy].copy()
            table(sub.head(topn))
    else:
        lines.append("No recommendation-frequency table available.")

    # 9) Timing
    section("[9] SPEED / LATENCY")
    lines.append(f"54-action batched inference median : {_fmt_num(latency_summary['median_ms'],3)} ms")
    lines.append(f"54-action batched inference mean   : {_fmt_num(latency_summary['mean_ms'],3)} ms")
    lines.append(f"54-action batched inference minimum: {_fmt_num(latency_summary['min_ms'],3)} ms")
    all_case_time = save_dir / "all_case_inference_timing.json"
    if all_case_time.exists():
        try:
            tj = json.loads(all_case_time.read_text(encoding="utf-8"))
            lines.append(f"All observed CFD cases batch time  : {_fmt_num(tj.get('batched_prediction_all_200_cases_ms'),3)} ms")
            lines.append(f"Amortized time per CFD case        : {_fmt_num(tj.get('ms_per_case_amortized'),3)} ms")
        except Exception:
            pass
    load_time = save_dir / "all_load_experiment_timing.json"
    if load_time.exists():
        try:
            tj = json.loads(load_time.read_text(encoding="utf-8"))
            lines.append(f"Full exhaustive HVAC experiment    : {_fmt_num(tj.get('total_seconds'),3)} s")
            lines.append(f"Average seconds / load condition   : {_fmt_num(tj.get('seconds_per_unique_load_average'),4)} s")
        except Exception:
            pass

    # 10) Optional every-candidate dump.
    if bool(getattr(args, "print_every_candidate", False)):
        section("[10] EVERY HVAC CANDIDATE ROW (VERY LARGE OUTPUT)")
        cand_csv = save_dir / "all_load_hvac_candidates.csv"
        if cand_csv.exists():
            try:
                cand = pd.read_csv(cand_csv)
                table(cand)
            except Exception as e:
                lines.append(f"Could not print candidate CSV: {e}")
        else:
            lines.append("all_load_hvac_candidates.csv not found")

    # 11) Headline scoreboard
    section("[11] FINAL HEADLINE SCOREBOARD")
    lines.append(f"Prediction | Test temperature MAE                 = {_fmt_num(summary['prediction']['test_temp_mae_c_mean'])} C")
    lines.append(f"Prediction | Test P95 absolute temperature error  = {_fmt_num(summary['prediction']['test_temp_p95_abs_error_c_mean'])} C")
    lines.append(f"Prediction | Test RA temperature error            = {_fmt_num(summary['prediction']['test_ra_abs_error_c_mean'])} C")
    lines.append(f"Sensors    | Selected sensor count                = {summary['sensor_design']['chosen_num_sensors']}")
    lines.append(f"Sensors    | Selected-sensor test recon MAE       = {_fmt_num(summary['sensor_design']['chosen_test_mae_c'])} C")
    lines.append(f"HVAC       | Counterfactual actions evaluated     = {hv['total_counterfactual_actions_evaluated']}")
    lines.append(f"HVAC       | Feasible load-combination rate       = {_fmt_num(hv['feasible_load_combination_rate_pct'],2)} %")
    lines.append(f"HVAC       | NO-FEASIBLE load combinations        = {hv['no_feasible_load_combinations']}")
    lines.append(f"Facility   | Mean extra cooling lower bound       = {_fmt_num(hv['mean_additional_cooling_lower_bound_kw_when_infeasible'])} kW")
    lines.append(f"Facility   | Max extra cooling lower bound        = {_fmt_num(hv['max_additional_cooling_lower_bound_kw_when_infeasible'])} kW")
    lines.append(f"Latency    | 54-action inference median            = {_fmt_num(latency_summary['median_ms'],3)} ms")

    section("[12] IMPORTANT INTERPRETATION LIMITS")
    for i, item in enumerate(summary.get("important_limitations", []), 1):
        lines.append(f"{i}. {item}")
    lines.append("5. Unobserved heat-load combinations are counterfactual extrapolation/interpolation within supplied factor levels; they are not direct CFD ground truth.")

    hr("=")
    lines.append("END OF FULL REPORT")
    hr("=")
    return "\n".join(lines)

def run_full_experiment(args) -> None:
    """
    One-command comprehensive evaluation using an EXISTING best.pt checkpoint.

    Outputs:
      A) all 200 observed CFD prediction metrics and split summaries
      B) sensor-count experiment rebuilt from the checkpoint's original train/val/test split
      C) all heat-load combinations (full Cartesian product by default) x all 54 HVAC actions
      D) feasible/no-feasible statistics, fallback diagnostics, capacity-gap lower bounds
      E) Current-HVAC vs AI comparison across all load conditions
      F) latency/efficiency summaries and figures
    """
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in full_experiment mode; reuse the existing best.pt")

    save_dir = ensure_dir(args.save_dir)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt, model, scalers, coords = load_checkpoint(args.checkpoint, device)
    case_df = load_case_info(args.case_info)
    data = _load_comparison_data(args, args.checkpoint)

    print("\n" + "=" * 100)
    print("POPFIELD HVAC — FULL EXPERIMENT SUITE")
    print(f"Device={device} | checkpoint={args.checkpoint}")
    print(f"Observed CFD cases={len(data['dp_ids'])} | nodes={coords.shape[0]} | params={count_params(model):,}")
    print("Existing best.pt is reused; NO retraining is performed.")
    print("=" * 100)

    # A. Evaluate all 200 observed CFD scenarios.
    per_case, pred_split, all_pred, all_ra_pred = evaluate_all_observed_cfd_cases(
        model, ckpt, scalers, coords, data, case_df, device, save_dir,
        args.zone_json, args.target_temp, args.comfort_band, args.max_zone_range,
        args.max_hot_fraction, args.max_cold_fraction, args.max_p95_temp,
    )
    print("[1/5] All observed CFD prediction metrics saved")

    # B. Reproduce sensor study with original checkpoint split.
    split = ckpt.get("split", {})
    tr = np.asarray(split.get("train", []), dtype=int)
    va = np.asarray(split.get("val", []), dtype=int)
    te = np.asarray(split.get("test", []), dtype=int)
    if not (len(tr) and len(va) and len(te)):
        raise RuntimeError("Checkpoint does not contain train/val/test split indices")
    sensor_result = run_sensor_study(
        train_temp=data["fields"][tr, :, 0],
        val_temp=data["fields"][va, :, 0],
        test_temp=data["fields"][te, :, 0],
        test_pred_temp=all_pred[te, :, 0],
        coords=coords,
        save_dir=save_dir,
        sensor_counts=[int(x) for x in args.sensor_counts.split(",") if x.strip()],
        pca_rank=args.sensor_pca_rank,
        target_reconstruction_mae=args.sensor_target_mae,
        ridge=args.sensor_ridge,
    )
    print(f"[2/5] Sensor study saved | chosen sensors={sensor_result['chosen_num_sensors']}")

    # C. Exhaustive counterfactual HVAC search for every unique observed heat-load condition.
    recommendations, comparisons, feasibility = run_all_load_hvac_experiment(
        model, scalers, coords, data, case_df, device, save_dir, args
    )
    print("[3/5] All observed heat-load x HVAC counterfactual experiments saved")

    # D. Latency benchmark for all 54 actions under Current loads.
    current = find_current_case(case_df)
    current_loads = {
        "external": float(current["P83 - external"]),
        "meeting": float(current["P84 - meeting"]),
        "server": float(current["P85 - server"]),
        "working": float(current["P86 - working"]),
    }
    actions = enumerate_observed_action_space(case_df)
    conds = np.asarray([
        [l, m, r, current_loads["external"], current_loads["meeting"], current_loads["server"], current_loads["working"], cmm, supply]
        for l, m, r, cmm, supply in actions
    ], dtype=np.float32)
    coords_norm_t = torch.from_numpy(scalers["coord"].transform(coords).astype(np.float32)).to(device)
    for _ in range(max(int(args.latency_warmup), 0)):
        _ = predict_conditions(model, conds, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"], device)
    latency_samples = []
    for _ in range(max(int(args.latency_repeat), 1)):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_conditions(model, conds, scalers["cond"], coords_norm_t, scalers["field"], scalers["ra"], device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_samples.append((time.perf_counter() - t0) * 1000.0)
    latency_summary = {
        "num_actions_batched": int(len(actions)),
        "median_ms": float(np.median(latency_samples)),
        "mean_ms": float(np.mean(latency_samples)),
        "min_ms": float(np.min(latency_samples)),
        "device": device,
    }
    (save_dir / "54_action_inference_latency.json").write_text(
        json.dumps(latency_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[4/5] Latency benchmark saved")

    # E. Aggregate KPIs and plots.
    sensor_table = pd.read_csv(save_dir / "sensor_study.csv")
    n_loads = int(len(feasibility))
    n_feasible = int((feasibility["status"] == "FEASIBLE").sum()) if len(feasibility) else 0
    n_infeasible = n_loads - n_feasible
    weighted_total = int(feasibility["observed_case_count"].sum()) if len(feasibility) else 0
    weighted_feasible = int(feasibility.loc[feasibility["status"] == "FEASIBLE", "observed_case_count"].sum()) if len(feasibility) else 0

    # Recommendation action frequencies.
    action_freq = pd.DataFrame()
    if len(recommendations):
        action_freq = (
            recommendations.groupby(["policy", "Inlet_L", "Inlet_M", "Inlet_R", "CMM", "AirTemp_C"])
            .agg(num_load_conditions=("load_id", "count"), represented_observed_cases=("observed_case_count", "sum"))
            .reset_index()
            .sort_values(["policy", "represented_observed_cases"], ascending=[True, False])
        )
        action_freq.to_csv(save_dir / "recommended_action_frequency.csv", index=False)

    failure_counter: Dict[str, int] = {}
    for s in feasibility.get("failed_constraints_best_achievable", pd.Series(dtype=str)).fillna("").astype(str):
        for part in [x for x in s.split("|") if x]:
            failure_counter[part] = failure_counter.get(part, 0) + 1
    pd.DataFrame([{"constraint": k, "num_load_conditions": v} for k, v in sorted(failure_counter.items())]).to_csv(
        save_dir / "failed_constraint_counts.csv", index=False
    )

    comparison_agg = pd.DataFrame()
    if len(comparisons):
        comparison_agg = (
            comparisons.groupby(["status", "policy"], dropna=False)
            .agg(
                num_load_conditions=("load_id", "count"),
                mean_zone_range_reduction_pct=("zone_range_reduction_pct", "mean"),
                median_zone_range_reduction_pct=("zone_range_reduction_pct", "median"),
                mean_hot_fraction_reduction_pp=("hot_fraction_reduction_percentage_points", "mean"),
                mean_sensible_cooling_capacity_saving_pct=("estimated_sensible_cooling_capacity_saving_pct", "mean"),
            ).reset_index()
        )
        comparison_agg.to_csv(save_dir / "current_vs_ai_aggregate.csv", index=False)

    test_row = pred_split[pred_split["split"] == "test"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "retraining_performed": False,
        "model": {
            "params": int(count_params(model)),
            "num_nodes": int(coords.shape[0]),
            "num_observed_cfd_cases": int(len(data["dp_ids"])),
        },
        "prediction": {
            "test_temp_mae_c_mean": float(test_row.iloc[0]["temp_mae_c_mean"]) if len(test_row) else float("nan"),
            "test_temp_p95_abs_error_c_mean": float(test_row.iloc[0]["temp_p95_abs_error_c_mean"]) if len(test_row) else float("nan"),
            "test_ra_abs_error_c_mean": float(test_row.iloc[0]["ra_abs_error_c_mean"]) if len(test_row) else float("nan"),
            "test_comfort_feasibility_accuracy": float(test_row.iloc[0]["comfort_feasibility_accuracy"]) if len(test_row) else float("nan"),
        },
        "sensor_design": {
            "chosen_num_sensors": int(sensor_result["chosen_num_sensors"]),
            "chosen_validation_mae_c": float(sensor_result["chosen_validation_mae_c"]),
            "chosen_test_mae_c": float(sensor_result["chosen_test_mae_c"]),
        },
        "hvac_counterfactual": {
            "load_space_mode": str(getattr(args, "full_load_space", "combinatorial")),
            "heat_load_combinations_evaluated": n_loads,
            "combinations_observed_in_cfd": int(feasibility["observed_in_cfd"].sum()) if "observed_in_cfd" in feasibility else n_loads,
            "combinations_unobserved_in_cfd": int((~feasibility["observed_in_cfd"].astype(bool)).sum()) if "observed_in_cfd" in feasibility else 0,
            "observed_cases_covered": weighted_total,
            "actions_per_load": int(len(actions)),
            "total_counterfactual_actions_evaluated": int(n_loads * len(actions)),
            "feasible_load_combinations": n_feasible,
            "no_feasible_load_combinations": n_infeasible,
            "feasible_load_combination_rate_pct": float(100.0 * n_feasible / max(n_loads, 1)),
            "observed_case_weighted_feasible_rate_pct": float(100.0 * weighted_feasible / max(weighted_total, 1)),
            "common_failed_constraints": failure_counter,
            "mean_additional_cooling_lower_bound_kw_when_infeasible": _safe_mean(
                feasibility.loc[feasibility["status"] == "NO_FEASIBLE_ACTION", "additional_cooling_kw_lower_bound_best"].to_numpy(float)
            ) if len(feasibility) else float("nan"),
            "max_additional_cooling_lower_bound_kw_when_infeasible": float(
                feasibility.loc[feasibility["status"] == "NO_FEASIBLE_ACTION", "additional_cooling_kw_lower_bound_best"].max()
            ) if bool((feasibility["status"] == "NO_FEASIBLE_ACTION").any()) else 0.0,
        },
        "latency": latency_summary,
        "important_limitations": [
            "Estimated sensible cooling capacity is thermal capacity, not measured electrical power or electricity-cost savings.",
            "The CFD scenarios are steady-state; the decision-support timing is not RL or dynamic closed-loop validation.",
            "If no official zone JSON is supplied, four XY quadrants are placeholder zones rather than official room zones.",
            "Additional cooling capacity is a load-balance lower-bound estimate, not exact equipment sizing.",
        ],
    }
    (save_dir / "FULL_EXPERIMENT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    kpi_rows = [
        {"category": "Prediction", "metric": "Test temperature MAE (C)", "value": summary["prediction"]["test_temp_mae_c_mean"]},
        {"category": "Prediction", "metric": "Test mean P95 absolute temp error (C)", "value": summary["prediction"]["test_temp_p95_abs_error_c_mean"]},
        {"category": "Sensors", "metric": "Chosen number of sensors", "value": summary["sensor_design"]["chosen_num_sensors"]},
        {"category": "Sensors", "metric": "Chosen sensor test reconstruction MAE (C)", "value": summary["sensor_design"]["chosen_test_mae_c"]},
        {"category": "HVAC", "metric": "Heat-load combinations evaluated", "value": n_loads},
        {"category": "HVAC", "metric": "Total counterfactual HVAC actions", "value": n_loads * len(actions)},
        {"category": "HVAC", "metric": "Feasible load-combination rate (%)", "value": summary["hvac_counterfactual"]["feasible_load_combination_rate_pct"]},
        {"category": "HVAC", "metric": "No-feasible load combinations", "value": n_infeasible},
        {"category": "Latency", "metric": "54-action batched inference median (ms)", "value": latency_summary["median_ms"]},
    ]
    pd.DataFrame(kpi_rows).to_csv(save_dir / "SUMMARY_KPIS.csv", index=False)

    create_full_experiment_plots(pred_split, sensor_table, feasibility, recommendations, save_dir)

    manifest = """PopField HVAC Full Experiment Outputs
====================================

Core prediction evaluation
- all_200_case_prediction_metrics.csv       : per-case prediction errors for all observed CFD cases
- prediction_metrics_by_split.csv           : train/validation/test/overall summaries
- all_case_inference_timing.json            : batched inference timing over all observed CFD cases

Sparse sensor experiment
- sensor_study.csv                           : 3/5/8/10/15/20 sensor reconstruction results
- selected_sensors.csv                       : selected sensor coordinates
- sensor_plan.json                           : validation-only sensor-count selection summary
- sensor_reconstruction_basis.npz            : PCA basis and selected sensor indices

HVAC exhaustive counterfactual experiment
- all_load_hvac_candidates.csv               : every unique observed heat-load condition x every HVAC action
- all_load_recommendations.csv               : feasible policies or best-achievable fallback per load condition
- all_load_feasibility.csv                   : FEASIBLE / NO_FEASIBLE_ACTION diagnosis per load condition
- all_load_current_vs_ai.csv                 : Current-HVAC baseline vs AI decisions across load conditions
- current_vs_ai_aggregate.csv                : aggregate improvement statistics by policy/status
- failed_constraint_counts.csv               : frequency of each failed comfort constraint
- recommended_action_frequency.csv           : how often each action is selected
- all_load_experiment_timing.json            : exhaustive experiment timing

Efficiency and summary
- 54_action_inference_latency.json           : low-latency 54-action surrogate benchmark
- SUMMARY_KPIS.csv                           : compact headline metrics
- FULL_EXPERIMENT_SUMMARY.json                : complete machine-readable summary
- figures/*.png                              : summary plots

Interpretation limits
- sensible cooling capacity != measured electrical power
- steady-state CFD != RL / temporal closed-loop validation
- default XY quadrants != official zones unless --zone_json is provided
"""
    (save_dir / "README_RESULTS.txt").write_text(manifest, encoding="utf-8")

    print("[5/5] Aggregate summary and figures saved")

    # Build one complete report for the Colab output window AND save the exact same text.
    report_text = build_full_console_report(
        save_dir=save_dir,
        summary=summary,
        per_case=per_case,
        pred_split=pred_split,
        sensor_table=sensor_table,
        sensor_result=sensor_result,
        feasibility=feasibility,
        recommendations=recommendations,
        comparisons=comparisons,
        comparison_agg=comparison_agg,
        action_freq=action_freq,
        failure_counter=failure_counter,
        latency_summary=latency_summary,
        args=args,
    )
    (save_dir / "FULL_CONSOLE_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Saved complete console report] {save_dir / 'FULL_CONSOLE_REPORT.txt'}")



# ============================================================
# 12. Consumer-facing PT demo
# ============================================================

def _observed_level_map(case_df: pd.DataFrame, col: str) -> Dict[str, float]:
    """
    Map easy demo labels (off/low/medium/high) to values that actually exist
    in the supplied Case Info. This avoids inventing arbitrary heat-load levels.
    """
    values = np.sort(case_df[col].astype(float).unique())
    if len(values) == 0:
        raise ValueError(f"No observed values found for {col}")

    zero_like = values[np.isclose(values, 0.0)]
    off_value = float(zero_like[0]) if len(zero_like) else float(values[0])

    positive = values[values > 1e-9]
    if len(positive) == 0:
        low = medium = high = off_value
    else:
        low = float(positive[0])
        high = float(positive[-1])
        # Choose an actually observed central positive level.
        medium = float(positive[(len(positive) - 1) // 2])

    return {
        "off": off_value,
        "low": low,
        "medium": medium,
        "high": high,
    }


def _normalize_level_name(value: str) -> str:
    aliases = {
        "0": "off", "off": "off", "none": "off", "없음": "off", "사용안함": "off",
        "1": "low", "low": "low", "낮음": "low", "적음": "low",
        "2": "medium", "medium": "medium", "mid": "medium", "보통": "medium", "중간": "medium",
        "3": "high", "high": "high", "높음": "high", "많음": "high",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown level {value!r}. Use off/low/medium/high.")
    return aliases[key]


def _prompt_text(prompt: str, default: str) -> str:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw if raw else default


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = _prompt_text(prompt, f"{default:g}")
        try:
            return float(raw)
        except ValueError:
            print("숫자로 입력해주세요.")


def _resolve_demo_inputs(case_df: pd.DataFrame, args) -> Tuple[Dict[str, float], Dict[str, str], str]:
    """
    Resolve consumer-friendly demo inputs.

    Priority:
      1) Exact W values (--external/--meeting/--server/--working)
      2) Easy levels (--*_level: off/low/medium/high)
      3) Interactive prompt when --demo_interactive is used
      4) Safe demo defaults (medium)

    Level labels are mapped ONLY to values observed in Case Info.
    """
    col_map = {
        "external": "P83 - external",
        "meeting": "P84 - meeting",
        "server": "P85 - server",
        "working": "P86 - working",
    }
    level_maps = {k: _observed_level_map(case_df, c) for k, c in col_map.items()}

    requested_levels: Dict[str, str] = {}
    loads: Dict[str, float] = {}

    if bool(getattr(args, "demo_interactive", False)):
        print("\n" + "=" * 88)
        print("POPFIELD HVAC — CONSUMER DEMO INPUT")
        print("W 단위를 몰라도 됩니다. 사용 정도를 off/low/medium/high 로 입력하세요.")
        print("=" * 88)

        args.target_temp = _prompt_float("원하는 실내 목표온도(°C)", float(args.target_temp))
        print("\n외부 열환경: off / low / medium / high")
        args.external_level = _normalize_level_name(_prompt_text("외부 열환경", args.external_level or "medium"))
        print("회의공간 사용: off / low / medium / high")
        args.meeting_level = _normalize_level_name(_prompt_text("회의공간 사용", args.meeting_level or "medium"))
        print("서버/전자기기 사용: off / low / medium / high")
        args.server_level = _normalize_level_name(_prompt_text("서버/전자기기 사용", args.server_level or "medium"))
        print("업무공간 사용: off / low / medium / high")
        args.working_level = _normalize_level_name(_prompt_text("업무공간 사용", args.working_level or "medium"))
        print("운전 목표: balanced / comfort_first / eco_first")
        args.demo_policy = _prompt_text("운전 목표", args.demo_policy).strip().lower()
        if args.demo_policy in {"comfort", "쾌적"}:
            args.demo_policy = "comfort_first"
        elif args.demo_policy in {"eco", "절약"}:
            args.demo_policy = "eco_first"
        elif args.demo_policy in {"balance", "균형"}:
            args.demo_policy = "balanced"
        if args.demo_policy not in {"balanced", "comfort_first", "eco_first"}:
            raise ValueError("운전 목표는 balanced / comfort_first / eco_first 중 하나여야 합니다.")

    for key in ["external", "meeting", "server", "working"]:
        exact = getattr(args, key, None)
        if exact is not None:
            loads[key] = float(exact)
            requested_levels[key] = "exact_W"
            continue

        level_attr = f"{key}_level"
        level = getattr(args, level_attr, None) or "medium"
        level = _normalize_level_name(level)
        requested_levels[key] = level
        loads[key] = float(level_maps[key][level])

    return loads, requested_levels, str(args.demo_policy)


def _direction_text(rec: Dict[str, object]) -> str:
    active = []
    if int(round(float(rec["Inlet_L"]))) == 1:
        active.append("L")
    if int(round(float(rec["Inlet_M"]))) == 1:
        active.append("M")
    if int(round(float(rec["Inlet_R"]))) == 1:
        active.append("R")
    return "+".join(active) if active else "OFF"


def _demo_constraint_diagnostics(rec: Dict[str, object], args) -> Dict[str, object]:
    """Build transparent per-constraint pass/fail and exceedance information for the demo UI."""
    target = float(args.target_temp)
    comfort_band = float(args.comfort_band)
    p95_limit = float(args.max_p95_temp) if args.max_p95_temp is not None else target + comfort_band

    values = {
        "zone_range": float(rec["zone_range_C"]),
        "hot_fraction": float(rec["hot_fraction"]),
        "cold_fraction": float(rec["cold_fraction"]),
        "p95_temperature": float(rec["p95_temp_C"]),
    }
    limits = {
        "zone_range": float(args.max_zone_range),
        "hot_fraction": float(args.max_hot_fraction),
        "cold_fraction": float(args.max_cold_fraction),
        "p95_temperature": p95_limit,
    }
    margins = {
        "zone_range": float(args.demo_near_zone_margin),
        "hot_fraction": float(args.demo_near_hot_margin_pp) / 100.0,
        "cold_fraction": float(args.demo_near_cold_margin_pp) / 100.0,
        "p95_temperature": float(args.demo_near_p95_margin),
    }

    detail: Dict[str, Dict[str, object]] = {}
    for key in values:
        exceed = max(values[key] - limits[key], 0.0)
        detail[key] = {
            "value": values[key],
            "limit": limits[key],
            "exceedance": exceed,
            "met": bool(exceed <= 1e-12),
            "within_near_margin": bool(exceed <= margins[key] + 1e-12),
            "near_margin": margins[key],
        }

    all_met = all(bool(v["met"]) for v in detail.values())
    all_within_near = all(bool(v["within_near_margin"]) for v in detail.values())

    if all_met:
        status = "FEASIBLE"
        label_ko = "달성 가능"
        icon = "✅"
    elif all_within_near:
        status = "NEAR_FEASIBLE"
        label_ko = "거의 달성"
        icon = "⚠"
    else:
        status = "INFEASIBLE"
        label_ko = "달성 어려움"
        icon = "❌"

    return {
        "status": status,
        "label_ko": label_ko,
        "icon": icon,
        "target_temp_C": target,
        "comfort_band_C": comfort_band,
        "acceptable_temperature_band_C": [target - comfort_band, target + comfort_band],
        "constraints": detail,
    }


def _print_demo_constraint_table(diag: Dict[str, object]) -> None:
    """Print constraint values with explicit limits and exceedances for easy interpretation."""
    d = diag["constraints"]
    rows = [
        ("Zone range", "zone_range", "C"),
        ("Hot fraction", "hot_fraction", "%"),
        ("Cold fraction", "cold_fraction", "%"),
        ("P95 temperature", "p95_temperature", "C"),
    ]
    print("\n[Comfort constraint check]")
    for label, key, unit in rows:
        item = d[key]
        met = bool(item["met"])
        mark = "✅" if met else "⚠"
        value = float(item["value"])
        limit = float(item["limit"])
        exceed = float(item["exceedance"])
        if unit == "%":
            value *= 100.0
            limit *= 100.0
            exceed *= 100.0
            print(f"  {mark} {label:15s}: {value:6.2f}% | 기준 <= {limit:6.2f}%", end="")
            if not met:
                print(f" | {exceed:.2f}%p 초과")
            else:
                print()
        else:
            print(f"  {mark} {label:15s}: {value:6.2f} C | 기준 <= {limit:6.2f} C", end="")
            if not met:
                print(f" | {exceed:.2f} C 초과")
            else:
                print()


def _save_demo_selected_field(
    model: PopFieldHVACTwin,
    rec: Dict[str, object],
    loads: Dict[str, float],
    scalers: Dict[str, Standardizer],
    coords: np.ndarray,
    coords_norm_t: torch.Tensor,
    device: str,
    save_dir: Path,
) -> Path:
    """Re-run only the selected HVAC action and save its predicted full field for a heatmap/demo."""
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

    field, ra = predict_conditions(
        model,
        cond,
        scalers["cond"],
        coords_norm_t,
        scalers["field"],
        scalers["ra"],
        device,
        batch_size=1,
    )
    pred = field[0]
    vel_mag = np.linalg.norm(pred[:, 1:4], axis=-1)
    out = pd.DataFrame({
        "node_index": np.arange(len(coords), dtype=int),
        "x_m": coords[:, 0],
        "y_m": coords[:, 1],
        "z_m": coords[:, 2],
        "pred_temperature_C": pred[:, 0],
        "pred_velocity_u": pred[:, 1],
        "pred_velocity_v": pred[:, 2],
        "pred_velocity_w": pred[:, 3],
        "pred_air_speed_mps": vel_mag,
        "pred_RA_temperature_C": float(ra[0]),
    })
    path = save_dir / "DEMO_SELECTED_TEMPERATURE_FIELD.csv"
    out.to_csv(path, index=False)
    return path


def run_demo(args) -> None:
    """
    User-facing demo using a trained best.pt.

    Required:
      - --checkpoint best.pt
      - --case_info Case Info Excel

    The original 200 Field CSVs are NOT required for this demo mode because:
      - model weights, XYZ coordinates and normalization statistics are stored in best.pt;
      - Case Info is used only to recover the allowed HVAC action set and observed load levels.

    Two input styles:
      A) Consumer-friendly levels:
         --external_level medium --meeting_level high --server_level high --working_level medium
      B) Exact heat loads in W:
         --external 500 --meeting 3000 --server 5000 --working 2000

    Add --demo_interactive to type the inputs interactively.
    """
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in demo mode")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if not Path(args.case_info).exists():
        raise FileNotFoundError(args.case_info)

    save_dir = ensure_dir(args.save_dir)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ckpt, model, scalers, coords = load_checkpoint(args.checkpoint, device)
    coords_norm_t = torch.from_numpy(
        scalers["coord"].transform(coords).astype(np.float32)
    ).to(device)
    case_df = load_case_info(args.case_info)

    loads, requested_levels, policy = _resolve_demo_inputs(case_df, args)

    # Show exactly how easy labels were mapped to the supplied CFD factor levels.
    level_mapping = {
        "external": _observed_level_map(case_df, "P83 - external"),
        "meeting": _observed_level_map(case_df, "P84 - meeting"),
        "server": _observed_level_map(case_df, "P85 - server"),
        "working": _observed_level_map(case_df, "P86 - working"),
    }

    t0 = time.perf_counter()
    opt = optimize_hvac(
        model=model,
        case_df=case_df,
        loads=loads,
        cond_scaler=scalers["cond"],
        coords=coords,
        coords_norm_t=coords_norm_t,
        field_scaler=scalers["field"],
        ra_scaler=scalers["ra"],
        device=device,
        save_dir=save_dir,
        zone_json=args.zone_json,
        target_temp_c=float(args.target_temp),
        comfort_band_c=float(args.comfort_band),
        max_zone_range_c=float(args.max_zone_range),
        max_hot_fraction=float(args.max_hot_fraction),
        max_cold_fraction=float(args.max_cold_fraction),
        max_p95_temp_c=args.max_p95_temp,
        energy_weight=float(args.energy_weight),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    recs = json.loads(
        (save_dir / "hvac_recommendations.json").read_text(encoding="utf-8")
    )

    feasible = bool(recs.get("fully_feasible_action_exists", False))
    if feasible:
        rec = recs[policy]
        policy_used = policy
    else:
        rec = recs["best_achievable"]
        policy_used = "best_achievable"

    # Demo status is deliberately more informative than the optimizer's strict binary flag.
    # A marginal miss (e.g., Hot 5.20% vs 5.00%, P95 28.04C vs 28.00C) is shown as
    # NEAR_FEASIBLE rather than the same hard failure label used for clearly impossible targets.
    demo_diag = _demo_constraint_diagnostics(rec, args)
    status = str(demo_diag["status"])

    selected_field_path = _save_demo_selected_field(
        model, rec, loads, scalers, coords, coords_norm_t, device, save_dir
    )

    num_actions = int(len(opt))
    result = {
        "demo_input": {
            "target_temp_C": float(args.target_temp),
            "requested_levels": requested_levels,
            "mapped_heat_loads_W": {k: float(v) for k, v in loads.items()},
            "requested_policy": policy,
        },
        "dataset_level_mapping_W": level_mapping,
        "status": status,
        "status_label_ko": demo_diag["label_ko"],
        "strict_optimizer_feasible": feasible,
        "comfort_constraint_diagnostics": demo_diag,
        "policy_used": policy_used,
        "num_HVAC_candidates_evaluated": num_actions,
        "decision_time_ms": float(elapsed_ms),
        "recommendation": rec,
        "failed_constraints": recs.get("failed_constraints_for_best_achievable", []),
        "additional_capacity_estimate": recs.get("additional_capacity_estimate", {}),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "selected_field_csv": str(selected_field_path),
        "notes": [
            "This is steady-state surrogate-based decision support, not RL/dynamic closed-loop control.",
            "Estimated sensible cooling capacity is thermal capacity, not measured electrical power.",
            "If no official zone JSON is supplied, the four XY quadrants are placeholder zones.",
            "Consumer-friendly low/medium/high inputs are mapped to heat-load levels observed in Case Info; they are demo abstractions, not direct sensor measurements.",
            "Demo status uses three levels: FEASIBLE, NEAR_FEASIBLE (small threshold exceedance), and INFEASIBLE. Strict optimizer feasibility is also stored separately.",
            "Consumer UI recommended target-temperature range defaults to 22-28C; out-of-range targets are still evaluated but explicitly warned.",
        ],
    }
    result_path = save_dir / "DEMO_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    direction = _direction_text(rec)
    print("\n" + "=" * 96)
    print("POPFIELD HVAC — USER DEMO RESULT")
    print("=" * 96)
    print(f"Device                 : {device}")
    print(f"Target room temperature: {float(args.target_temp):.1f} °C")
    print(f"Requested mode         : {policy}")
    print("\n[Easy input -> model heat-load mapping]")
    for k in ["external", "meeting", "server", "working"]:
        print(f"  {k:8s}: {requested_levels[k]:>8s} -> {loads[k]:.0f} W")

    print("\n[AI virtual test]")
    print(f"  HVAC candidates evaluated: {num_actions}")
    print(f"  Decision computation      : {elapsed_ms:.2f} ms")

    demo_status = str(demo_diag["status"])
    if demo_status == "FEASIBLE":
        print("\n✅ 달성 가능: 현재 후보 HVAC 범위 안에서 설정한 쾌적 조건을 만족하는 운전안이 있습니다.")
    elif demo_status == "NEAR_FEASIBLE":
        print("\n⚠ 거의 달성: 모든 기준을 완전히 만족하지는 않지만, 최선 설정이 허용기준을 소폭만 초과합니다.")
        print("  17C 같은 명확한 실패와 구분하여 '거의 달성'으로 표시합니다.")
    else:
        print("\n❌ 달성 어려움: 현재 후보 HVAC 범위에서는 설정한 쾌적 조건을 충분히 만족하기 어렵습니다.")
        print("  아래 결과는 54개 후보 중 위반이 가장 적은 best-achievable 설정입니다.")

    # Consumer demo guardrail: the model can score any target, but very extreme targets are not
    # representative of the intended room-comfort demo range.
    if float(args.target_temp) < float(args.demo_target_min) or float(args.target_temp) > float(args.demo_target_max):
        print(
            f"\n⚠ 입력한 목표온도 {float(args.target_temp):.1f}C는 권장 데모 범위 "
            f"{float(args.demo_target_min):.0f}~{float(args.demo_target_max):.0f}C 밖입니다."
        )
        print("  결과는 계산되지만, 소비자용 UI에서는 이 범위를 슬라이더 한계로 두는 것을 권장합니다.")

    print("\n[Recommended HVAC]")
    print(f"  Direction        : {direction}")
    print(f"  Fan / CMM        : {float(rec['CMM']):.0f}")
    print(f"  Supply temp      : {float(rec['AirTemp_C']):.1f} °C")
    print(f"  Pred. mean temp  : {float(rec['mean_temp_C']):.2f} °C")
    print(f"  Pred. P95 temp   : {float(rec['p95_temp_C']):.2f} °C")
    print(f"  Zone range       : {float(rec['zone_range_C']):.2f} °C")
    print(f"  Hot fraction     : {100.0 * float(rec['hot_fraction']):.2f} %")
    print(f"  Cold fraction    : {100.0 * float(rec['cold_fraction']):.2f} %")
    print(f"  Sensible cooling : {float(rec['estimated_sensible_cooling_kw']):.2f} kW (thermal estimate)")

    _print_demo_constraint_table(demo_diag)

    if demo_status != "FEASIBLE":
        failed = [
            k for k, v in demo_diag["constraints"].items()
            if not bool(v["met"])
        ]
        print(f"\n[Why not fully feasible] {failed}")
        if demo_status == "NEAR_FEASIBLE":
            print("  해석: 목표에 매우 가깝지만 일부 지표가 기준을 소폭 초과했습니다.")
            print("  앱에서는 '불가능' 대신 '거의 달성'으로 보여주는 것이 더 적절합니다.")
        else:
            print("  해석: 현재 54개 HVAC 후보만으로는 목표 쾌적조건과 차이가 큽니다.")

        gap = recs.get("additional_capacity_estimate", {})
        if gap:
            print("\n[Facility reference — thermal load-balance estimate only]")
            print(
                "  Best-achievable 열수지 기준 추가 냉방 여유 참고값: "
                f"{float(gap.get('additional_sensible_cooling_kw_lower_bound_at_best_achievable', 0.0)):.2f} kW"
            )
            print(
                "  최대 후보 냉방능력에서도 남는 열수지 gap 참고값    : "
                f"{float(gap.get('additional_sensible_cooling_kw_lower_bound_even_at_max_candidate_capacity', 0.0)):.2f} kW"
            )
            print("  ※ 실제 에어컨 증설용량, 전력소비, 전기요금 절감량이 아닙니다.")
            print("  ※ 특히 '거의 달성'은 국소 Hotspot/공기분배 문제일 수 있어 kW만으로 해결을 단정하면 안 됩니다.")

    print("\n[Saved]")
    print(f"  Result JSON      : {result_path}")
    print(f"  Selected field   : {selected_field_path}")
    print(f"  All 54 candidates: {save_dir / 'hvac_optimization.csv'}")
    print("\nNOTE: low/medium/high는 Case Info에 실제 존재하는 열부하 단계로 매핑한 데모 입력입니다.")
    print("      실제 제품에서는 센서/점유/기기전력/외기정보를 자동 입력으로 연결하는 것이 좋습니다.")
    print("=" * 96)


# ============================================================
# 12. CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train", "optimize", "recommend", "demo", "full_experiment", "train_full"], default="train",
                   help="train_full = train from scratch, save best.pt, then automatically run the complete experiment and print all results")
    p.add_argument("--case_info", type=str, default="/content/Case Info 200 DesignPoints - 최종본.xlsx")
    p.add_argument("--field_zip", type=str, default="/content/Field data.zip")
    p.add_argument("--save_dir", type=str, default="/content/popfield_hvac_runs")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--force_rebuild_cache", action="store_true")

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--velocity_weight", type=float, default=0.25)
    p.add_argument("--ra_weight", type=float, default=0.30)
    p.add_argument("--no_node_embedding", action="store_true")
    p.add_argument("--stable_init", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    # Sensor study
    p.add_argument("--sensor_counts", type=str, default="3,5,8,10,15,20")
    p.add_argument("--sensor_pca_rank", type=int, default=20)
    p.add_argument("--sensor_target_mae", type=float, default=1.0,
                   help="Validation MAE target used to choose the minimum sensor count")
    p.add_argument("--sensor_ridge", type=float, default=1e-3,
                   help="Ridge regularization for stable sparse-sensor reconstruction")

    # HVAC optimization
    p.add_argument("--optimize_after_train", action="store_true")
    p.add_argument("--external", type=float, default=None)
    p.add_argument("--meeting", type=float, default=None)
    p.add_argument("--server", type=float, default=None)
    p.add_argument("--working", type=float, default=None)

    # Consumer demo inputs. Exact W values above take priority over these easy levels.
    p.add_argument("--external_level", choices=["off", "low", "medium", "high"], default=None)
    p.add_argument("--meeting_level", choices=["off", "low", "medium", "high"], default=None)
    p.add_argument("--server_level", choices=["off", "low", "medium", "high"], default=None)
    p.add_argument("--working_level", choices=["off", "low", "medium", "high"], default=None)
    p.add_argument("--demo_policy", choices=["balanced", "comfort_first", "eco_first"], default="balanced")
    p.add_argument("--demo_interactive", action="store_true",
                   help="Prompt for target temperature, easy heat-load levels and policy in demo mode")
    p.add_argument("--demo_target_min", type=float, default=22.0,
                   help="Recommended minimum target temperature shown in the consumer demo UI")
    p.add_argument("--demo_target_max", type=float, default=28.0,
                   help="Recommended maximum target temperature shown in the consumer demo UI")
    p.add_argument("--demo_near_zone_margin", type=float, default=0.25,
                   help="NEAR_FEASIBLE margin above max zone range [C]")
    p.add_argument("--demo_near_hot_margin_pp", type=float, default=1.0,
                   help="NEAR_FEASIBLE margin above hot-fraction limit [percentage points]")
    p.add_argument("--demo_near_cold_margin_pp", type=float, default=1.0,
                   help="NEAR_FEASIBLE margin above cold-fraction limit [percentage points]")
    p.add_argument("--demo_near_p95_margin", type=float, default=0.25,
                   help="NEAR_FEASIBLE margin above P95 temperature limit [C]")

    p.add_argument("--target_temp", type=float, default=24.0)
    p.add_argument("--comfort_band", type=float, default=2.0)
    p.add_argument("--max_zone_range", type=float, default=2.0)
    p.add_argument("--max_hot_fraction", type=float, default=0.05,
                   help="Maximum allowed fraction above target_temp + comfort_band")
    p.add_argument("--max_cold_fraction", type=float, default=0.05,
                   help="Maximum allowed fraction below target_temp - comfort_band")
    p.add_argument("--max_p95_temp", type=float, default=None,
                   help="Maximum allowed 95th-percentile temperature; default=target_temp+comfort_band")
    p.add_argument("--energy_weight", type=float, default=0.35)
    p.add_argument("--zone_json", type=str, default=None)
    p.add_argument("--export_influence", action="store_true")

    # Low-latency recommendation timing
    p.add_argument("--latency_warmup", type=int, default=3)
    p.add_argument("--latency_repeat", type=int, default=20)

    # Comprehensive experiment controls
    p.add_argument("--full_load_space", choices=["combinatorial", "observed"], default="combinatorial",
                   help="combinatorial evaluates the full Cartesian product of supplied heat-load levels; observed evaluates only load combinations present in the 200 CFD cases")
    p.add_argument("--full_max_load_groups", type=int, default=0,
                   help="0 evaluates all load groups; positive values are intended only for quick smoke tests")
    p.add_argument("--console_detail", choices=["summary", "full", "everything"], default="full",
                   help="full prints all 200 observed cases, all load feasibility/recommendation/comparison rows, plus summaries")
    p.add_argument("--console_top_actions", type=int, default=20,
                   help="How many most-frequent HVAC actions to print per policy")
    p.add_argument("--print_every_candidate", action="store_true",
                   help="Also print every HVAC candidate row (up to 7,776 rows); very large Colab output")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[argparse] ignored unknown args: {unknown}")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "train_full":
        # One-command path with NO pre-existing .pt required:
        # train -> save best.pt -> exhaustive full experiment -> print complete final report.
        run_train(args)
        args.checkpoint = str(Path(args.save_dir) / "best.pt")
        print("\n" + "#" * 118)
        print("TRAINING COMPLETE -> STARTING FULL EXPERIMENT WITH THE NEW best.pt")
        print(f"Checkpoint: {args.checkpoint}")
        print("#" * 118 + "\n")
        run_full_experiment(args)
    elif args.mode == "optimize":
        run_optimize(args)
    elif args.mode == "recommend":
        run_recommend(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        run_full_experiment(args)


if __name__ == "__main__":
    main()
