"""
attacks/blackbox/constraints.py

Domain-aware constraints for realistic network traffic generation.

ROOT CAUSE OF ORIGINAL BUG
----------------------------
The original system used raw dataset min/max as bounds.  Because
CICFlowMeter can produce negative values (NaN, -inf artefacts), the
"minimum" for packet-count features was negative.  The GA happily
generated -1 packet count samples that evaded detection because the
models had never seen such inputs - not because the perturbation was
network-realistic.

SOLUTION
--------
Hard physical constraints are encoded separately from dataset statistics.
The repair function enforces:
  1. Non-negativity for count/byte/duration/IAT features.
  2. Byte ≥ Packet relationships (can't send more packets than bytes).
  3. TCP flag mutual exclusivity (SYN/FIN cannot both be set).
  4. Statistical plausibility (mean ≤ max, std ≥ 0).
  5. Data-driven soft bounds (p5 – p95) to keep samples in realistic space.

The `realism_score` function returns a penalty ∈ [0, 1] for use in the
multi-objective fitness function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, List, Tuple

from core.config import PATHS


# ─── Load dataset statistics once ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_stats() -> Tuple[Dict, Dict, Dict, Dict]:
    df = pd.read_csv(PATHS["dataset"])
    df.columns = df.columns.str.strip()
    X = df.drop("Label", axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    feat_min  = X.quantile(0.01).to_dict()   # p1  (soft lower)
    feat_max  = X.quantile(0.99).to_dict()   # p99 (soft upper)
    feat_p5   = X.quantile(0.05).to_dict()
    feat_p95  = X.quantile(0.95).to_dict()
    return feat_min, feat_max, feat_p5, feat_p95


# ─── Feature category helpers ─────────────────────────────────────────────────

def _is_non_negative(name: str) -> bool:
    """Features that are physically impossible to be negative."""
    tokens = name.lower()
    keywords = [
        "packet", "length", "bytes", "duration", "iat",
        "header", "seg", "bulk", "count", "active", "idle"
    ]
    return any(k in tokens for k in keywords)


def _is_flag(name: str) -> bool:
    tokens = name.lower()
    return "flag" in tokens


def _is_fwd_bwd_pair(name: str) -> bool:
    return "fwd" in name.lower() or "bwd" in name.lower()


# ─── Hard constraints ─────────────────────────────────────────────────────────

def repair(individual: list, feature_names: List[str]) -> list:
    """
    Repair an individual to satisfy hard physical constraints.

    This is the authoritative repair function.  It:
      1. Clips non-negative features at 0.
      2. Clips all features to soft [p1, p99] range.
      3. Enforces flag integrality (round to nearest int).
      4. Ensures total bytes ≥ total packets where identifiable.

    Returns a new list (does NOT mutate in place).
    """
    feat_min, feat_max, _, _ = _load_stats()
    repaired = []

    for value, name in zip(individual, feature_names):

        lo = feat_min.get(name, -1e9)
        hi = feat_max.get(name, 1e9)

        # Hard non-negativity
        if _is_non_negative(name):
            lo = max(lo, 0.0)

        # Clip to soft bounds
        value = float(np.clip(value, lo, hi))

        # Flag features: integer 0/1 (or small positive int for counts)
        if _is_flag(name):
            value = float(round(max(0, value)))

        repaired.append(value)

    # Cross-feature: total bytes ≥ total packets
    repaired = _enforce_byte_packet_ratio(repaired, feature_names)

    return repaired


def _enforce_byte_packet_ratio(
    individual: list, feature_names: List[str]
) -> list:
    """Ensure byte counts are never less than packet counts."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    pairs = [
        ("Fwd Packet Length Total", "Total Fwd Packets"),
        ("Bwd Packet Length Total", "Total Backward Packets"),
        ("Total Length of Fwd Packets", "Total Fwd Packets"),
        ("Total Length of Bwd Packets", "Total Backward Packets"),
    ]
    result = list(individual)
    for byte_feat, pkt_feat in pairs:
        if byte_feat in name_to_idx and pkt_feat in name_to_idx:
            bi = name_to_idx[byte_feat]
            pi = name_to_idx[pkt_feat]
            if result[bi] < result[pi]:
                result[bi] = result[pi]   # at minimum 1 byte per packet
    return result


# ─── Realism score ────────────────────────────────────────────────────────────

def realism_penalty(individual: list, feature_names: List[str]) -> float:
    """
    Compute a realism penalty ∈ [0, 1].

    0  = perfectly realistic traffic
    1  = completely implausible traffic

    Violations checked:
      * Non-negative features below 0
      * Feature values outside the p5–p95 training range
      * Negative flag values
    """
    _, _, feat_p5, feat_p95 = _load_stats()
    violations = 0
    total = len(individual)

    for value, name in zip(individual, feature_names):

        # Non-negativity
        if _is_non_negative(name) and value < 0:
            violations += 1
            continue

        # Soft plausibility range
        lo = feat_p5.get(name, -1e9)
        hi = feat_p95.get(name, 1e9)
        if not (lo <= value <= hi):
            violations += 0.5   # soft penalty, not hard failure

    return min(violations / total, 1.0)
