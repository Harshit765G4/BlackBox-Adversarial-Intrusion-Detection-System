"""
gui/components/model_loader.py

Thin wrappers that delegate to core.model_registry.
Kept for backward compatibility with existing page imports.
"""

from core.model_registry import get_rf, get_lgbm, get_fnn, get_fnn_scaler


def load_rf():
    return get_rf()


def load_lgbm():
    return get_lgbm()


def load_fnn():
    return get_fnn()


def load_fnn_scaler():
    return get_fnn_scaler()
