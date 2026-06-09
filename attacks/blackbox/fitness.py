"""
attacks/blackbox/fitness.py

Multi-objective fitness for adversarial evasion.

ORIGINAL BUG
------------
The original fitness was:
    fitness = (1.0 - probability) - (distance * 0.000001)

Problems:
  1. distance was measured from the zero vector, not from the original seed.
     This means the GA could freely move to any region of feature space,
     not just near the original attack traffic.
  2. The perturbation penalty weight (1e-6) was effectively zero, so there
     was no incentive to stay close to the seed.
  3. No realism penalty → GA exploited impossible negative counts.

FIXED OBJECTIVE
---------------
    fitness = evasion_score
              - λ * perturbation_norm       # stay near original
              - μ * realism_penalty         # stay network-plausible

Where:
    evasion_score  = 1 - ensemble_probability   (maximize evasion)
    perturbation   = L2(adversarial - original) / L2(original)  (normalized)
    realism_penalty = fraction of features violating domain constraints

This is a weighted sum approximation of a Pareto front.
λ and μ are tuned via config.py.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from attacks.blackbox.constraints import realism_penalty
from attacks.blackbox.oracle import EnsembleOracle
from core.config import GA_CONFIG


def evaluate(
    individual: list,
    original: list,
    feature_names: List[str],
    oracle: EnsembleOracle,
) -> Tuple[float, int, float, float]:
    """
    Evaluate a single individual.

    Parameters
    ----------
    individual    : current adversarial candidate (repaired)
    original      : the attack seed sample before perturbation
    feature_names : ordered feature names
    oracle        : EnsembleOracle instance

    Returns
    -------
    (fitness, prediction, probability, perturbation_norm)
    """
    result = oracle.query(individual, feature_names)
    prob   = result["probability"]
    pred   = result["prediction"]

    # ── Evasion term ──────────────────────────────────────────────────────────
    evasion = 1.0 - prob      # higher = better evasion

    # ── Perturbation term (normalized L2) ─────────────────────────────────────
    x     = np.array(individual, dtype=np.float64)
    x0    = np.array(original,   dtype=np.float64)
    denom = np.linalg.norm(x0) + 1e-9
    perturbation = np.linalg.norm(x - x0) / denom

    # ── Realism term ──────────────────────────────────────────────────────────
    unrealism = realism_penalty(individual, feature_names)

    # ── Combined fitness ──────────────────────────────────────────────────────
    lam = GA_CONFIG["perturbation_weight"]
    mu  = GA_CONFIG["realism_weight"]

    fitness = evasion - lam * perturbation - mu * unrealism

    return fitness, pred, prob, perturbation
