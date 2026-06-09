"""
attacks/blackbox/genome.py

Feature schema and seed sample loader.

ORIGINAL BUG: loaded the full 500 K-row CSV at import time on every
module that imported genome.py.  Now everything is lazy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd

from core.config import PATHS


@lru_cache(maxsize=1)
def get_feature_names() -> List[str]:
    df = pd.read_csv(PATHS["dataset"], nrows=1)
    df.columns = df.columns.str.strip()
    return list(df.drop("Label", axis=1).columns)


@lru_cache(maxsize=1)
def get_attack_seeds() -> np.ndarray:
    """Return all attack-labelled rows as a numpy array. Cached."""
    df = pd.read_csv(PATHS["dataset"])
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    attack = df[df["Label"] == 1].drop("Label", axis=1)
    return attack.values.astype(np.float64)


# Convenience aliases kept for compatibility
FEATURES = property(lambda _: get_feature_names())
NUM_FEATURES = property(lambda _: len(get_feature_names()))
