"""Scoring metrics for probabilistic predictions.

For xG we care about *probability quality*, not classification accuracy — a good
model says "0.1" for a chance that goes in 10% of the time. The two standard
measures:

- log loss: punishes confident wrong predictions hard (the primary xG metric).
- Brier score: mean squared error of the probability (0 = perfect, lower better).

Plain numpy so the numbers are inspectable and match the StatsBomb benchmark we
computed in Phase 1 (log loss 0.279, Brier 0.081).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-15


def log_loss(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def summary(y_true, p_pred) -> dict[str, float]:
    """All the headline numbers for one set of predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "pred_goals": float(p.sum()),   # summed xG ...
        "actual_goals": float(y.sum()), # ... vs reality (aggregate calibration)
    }
