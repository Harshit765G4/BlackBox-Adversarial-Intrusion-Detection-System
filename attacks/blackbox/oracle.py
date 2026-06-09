"""
attacks/blackbox/oracle.py

Ensemble oracle for black-box adversarial evaluation.

Design decisions
----------------
* No top-level side effects: models are NOT loaded at import time.
  The GUI can import this module without triggering disk I/O.
* Query budget is encapsulated in an Oracle object so multiple
  concurrent attack runs don't share global state.
* Returns rich diagnostics for each query.
"""

import numpy as np
import pandas as pd
import torch

from core.model_registry import get_rf, get_lgbm, get_fnn, get_fnn_scaler
from core.config import ENSEMBLE_THRESHOLD


class EnsembleOracle:
    """
    Stateful oracle: tracks query count and enforces a budget.

    Parameters
    ----------
    query_limit : int
        Hard cap on number of oracle calls.
    """

    def __init__(self, query_limit: int = 10_000):
        self.query_limit = query_limit
        self._query_count = 0
        # Lazy model handles
        self._rf = None
        self._lgbm = None
        self._fnn = None
        self._scaler = None

    def _load(self):
        if self._rf is None:
            self._rf     = get_rf()
            self._lgbm   = get_lgbm()
            self._fnn    = get_fnn()
            self._scaler = get_fnn_scaler()

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def query_count(self) -> int:
        return self._query_count

    def reset(self):
        self._query_count = 0

    def budget_exceeded(self) -> bool:
        return self._query_count >= self.query_limit

    def query(self, sample: list | np.ndarray, feature_names: list) -> dict:
        """
        Query the ensemble oracle with a single traffic sample.

        Parameters
        ----------
        sample       : 1-D array-like of length 78
        feature_names: ordered list of feature names

        Returns
        -------
        dict with keys: prediction, probability, rf_prob, lgbm_prob, fnn_prob
        """
        if self.budget_exceeded():
            raise RuntimeError(
                f"Query budget of {self.query_limit} exceeded."
            )

        self._load()
        self._query_count += 1

        x = pd.DataFrame([sample], columns=feature_names)

        # RF
        rf_prob = float(self._rf.predict_proba(x)[0][1])

        # LightGBM
        lgbm_prob = float(self._lgbm.predict_proba(x)[0][1])

        # FNN
        x_scaled = self._scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            fnn_prob = float(
                torch.sigmoid(self._fnn(x_tensor)).item()
            )

        ensemble_prob = (rf_prob + lgbm_prob + fnn_prob) / 3.0
        prediction = int(ensemble_prob >= ENSEMBLE_THRESHOLD)

        return {
            "prediction":    prediction,
            "probability":   ensemble_prob,
            "rf_prob":       rf_prob,
            "lgbm_prob":     lgbm_prob,
            "fnn_prob":      fnn_prob,
        }


# ─── Module-level default oracle (optional convenience) ──────────────────────
_default_oracle: EnsembleOracle | None = None


def get_default_oracle(query_limit: int = 10_000) -> EnsembleOracle:
    global _default_oracle
    if _default_oracle is None:
        _default_oracle = EnsembleOracle(query_limit)
    return _default_oracle
