"""
core/config.py
Centralized configuration for EnCryptoGuard.
All paths, hyperparameters, and constants live here.
"""

import os

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
PATHS = {
    "rf_model":      os.path.join(ROOT, "models", "rf_model.pkl"),
    "lgbm_model":    os.path.join(ROOT, "models", "lgbm_model.pkl"),
    "fnn_model":     os.path.join(ROOT, "models", "fnn_v2_model.pth"),
    "fnn_scaler":    os.path.join(ROOT, "models", "fnn_scaler.pkl"),
    "X_test":        os.path.join(ROOT, "artifacts", "X_test.pkl"),
    "y_test":        os.path.join(ROOT, "artifacts", "y_test.pkl"),
    "dataset":       os.path.join(ROOT, "dataset", "processed", "fnn_dataset.csv"),
    "results":       os.path.join(ROOT, "results"),
}

# ─── FNN Architecture ─────────────────────────────────────────────────────────
FNN_CONFIG = {
    "hidden_dims": [256, 128, 64],
    "dropout":     0.3,
}

# ─── Ensemble ─────────────────────────────────────────────────────────────────
ENSEMBLE_THRESHOLD = 0.5          # probability ≥ threshold → Attack

# ─── Black-Box GA ─────────────────────────────────────────────────────────────
GA_CONFIG = {
    "population_size":    100,
    "generations":        80,
    "elite_size":         5,
    "crossover_prob":     0.7,
    "mutation_prob":      0.30,
    "gene_mutation_prob": 0.10,
    "query_limit":        10_000,
    "evasion_threshold":  0.45,   # ensemble prob < this → success
    "perturbation_weight": 0.05,  # λ for Lp-norm penalty in fitness
    "realism_weight":     0.10,   # μ for domain-violation penalty
}

# ─── White-Box FGSM ───────────────────────────────────────────────────────────
FGSM_EPSILONS = [0.01, 0.03, 0.05, 0.07, 0.10]

# ─── White-Box PGD ────────────────────────────────────────────────────────────
PGD_CONFIG = {
    "epsilon":    0.05,
    "alpha":      0.01,
    "iterations": 20,
}
