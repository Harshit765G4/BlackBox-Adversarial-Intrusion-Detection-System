"""
core/preprocessor.py

Centralized preprocessing pipeline.
Guarantees that any DataFrame passed to a model matches the
exact schema used at training time.

FIX: Resolves the `ValueError: Feature names should match those
     that were passed during fit` crash in the Traffic Detection page.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from core.model_registry import get_fnn_scaler


# ─── Training Schema ──────────────────────────────────────────────────────────
# These are the 78 feature names as they appear after df.columns.str.strip()
# at training time.  Loaded lazily from the scaler's feature_names_in_.

_TRAINING_FEATURES: Optional[List[str]] = None


def get_training_features() -> List[str]:
    global _TRAINING_FEATURES
    if _TRAINING_FEATURES is None:
        scaler = get_fnn_scaler()
        if hasattr(scaler, "feature_names_in_"):
            _TRAINING_FEATURES = list(scaler.feature_names_in_)
        else:
            raise RuntimeError(
                "Scaler was fitted without feature names. "
                "Re-train with a DataFrame (not a NumPy array)."
            )
    return _TRAINING_FEATURES


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare an arbitrary CSV upload for ensemble prediction.

    Steps
    -----
    1. Strip whitespace from column names.
    2. Drop 'Label' column if present.
    3. Reorder/select columns to match training schema exactly.
    4. Fill any missing columns with 0 (graceful degradation).
    5. Clip obvious physical impossibilities (negatives in count cols).

    Returns
    -------
    pd.DataFrame with shape (n, 78) ready for model.predict().
    """
    df = df.copy()

    # 1. Strip column names
    df.columns = df.columns.str.strip()

    # 2. Drop label column if present
    for label_col in ["Label", "label"]:
        if label_col in df.columns:
            df = df.drop(columns=[label_col])

    # 3. Align to training schema
    expected = get_training_features()
    missing = [c for c in expected if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected]

    if missing:
        for col in missing:
            df[col] = 0.0

    if extra:
        df = df.drop(columns=extra)

    df = df[expected]  # enforce column order

    # 4. Replace inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    # 5. Clip non-negative features (packet counts, byte counts, durations)
    NON_NEGATIVE_PATTERNS = [
        "Packet", "Bytes", "Duration", "Length",
        "IAT", "Header", "Seg", "Fwd", "Bwd"
    ]
    for col in df.columns:
        if any(p.lower() in col.lower() for p in NON_NEGATIVE_PATTERNS):
            df[col] = df[col].clip(lower=0)

    return df.astype(np.float32)


def scale_for_fnn(df: pd.DataFrame) -> np.ndarray:
    """Apply the saved StandardScaler and return a numpy array."""
    scaler = get_fnn_scaler()
    return scaler.transform(df).astype(np.float32)
