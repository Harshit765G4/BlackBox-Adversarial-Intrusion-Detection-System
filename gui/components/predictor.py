"""
gui/components/predictor.py

Ensemble and per-model prediction with preprocessing.
All calls go through core.preprocessor to guarantee schema alignment.
"""

import numpy as np
import pandas as pd
import torch

from core.preprocessor import preprocess, scale_for_fnn
from core.model_registry import get_rf, get_lgbm, get_fnn
from core.config import ENSEMBLE_THRESHOLD


def predict_ensemble(df: pd.DataFrame) -> dict:
    """
    Run all three models and return ensemble + per-model results.

    Parameters
    ----------
    df : raw uploaded DataFrame (any column order, with or without Label)

    Returns
    -------
    dict with keys:
      predictions : np.ndarray (N,) of 0/1
      probabilities : np.ndarray (N,) ensemble probability
      rf_probs, lgbm_probs, fnn_probs : np.ndarray (N,) per-model
    """
    X = preprocess(df)

    rf    = get_rf()
    lgbm  = get_lgbm()
    fnn   = get_fnn()

    rf_probs   = rf.predict_proba(X)[:, 1]
    lgbm_probs = lgbm.predict_proba(X)[:, 1]

    X_scaled = scale_for_fnn(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        fnn_probs = torch.sigmoid(fnn(X_tensor)).squeeze().numpy()

    ensemble_probs = (rf_probs + lgbm_probs + fnn_probs) / 3.0
    predictions    = (ensemble_probs >= ENSEMBLE_THRESHOLD).astype(int)

    return {
        "predictions":  predictions,
        "probabilities": ensemble_probs,
        "rf_probs":      rf_probs,
        "lgbm_probs":    lgbm_probs,
        "fnn_probs":     fnn_probs,
    }


def predict_rf(df: pd.DataFrame) -> np.ndarray:
    X = preprocess(df)
    return get_rf().predict(X)
