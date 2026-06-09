"""
core/model_registry.py

Production-grade model registry with lazy loading.
All three models (RF, LightGBM, FNN) are loaded once and cached.
Import this module anywhere; no model re-loading occurs.
"""

import joblib
import torch
import torch.nn as nn

from core.config import PATHS, FNN_CONFIG


# ─── FNN Definition (single source of truth) ──────────────────────────────────

class FNN(nn.Module):
    """Binary classifier: 78 → 256 → 128 → 64 → 1"""

    def __init__(self, input_dim: int):
        super().__init__()
        dims = [input_dim] + FNN_CONFIG["hidden_dims"]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if i < len(dims) - 2:
                layers.append(nn.Dropout(FNN_CONFIG["dropout"]))
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─── Lazy Singleton Cache ─────────────────────────────────────────────────────

_cache: dict = {}


def get_rf():
    if "rf" not in _cache:
        _cache["rf"] = joblib.load(PATHS["rf_model"])
    return _cache["rf"]


def get_lgbm():
    if "lgbm" not in _cache:
        _cache["lgbm"] = joblib.load(PATHS["lgbm_model"])
    return _cache["lgbm"]


def get_fnn_scaler():
    if "fnn_scaler" not in _cache:
        _cache["fnn_scaler"] = joblib.load(PATHS["fnn_scaler"])
    return _cache["fnn_scaler"]


def get_fnn() -> FNN:
    if "fnn" not in _cache:
        scaler = get_fnn_scaler()
        input_dim = len(scaler.mean_)
        model = FNN(input_dim)
        model.load_state_dict(
            torch.load(PATHS["fnn_model"], weights_only=True)
        )
        model.eval()
        _cache["fnn"] = model
    return _cache["fnn"]


def get_all():
    """Load all models. Call once at app startup to pre-warm."""
    return get_rf(), get_lgbm(), get_fnn(), get_fnn_scaler()
