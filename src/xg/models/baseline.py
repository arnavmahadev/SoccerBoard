"""Baseline xG models: logistic regression and XGBoost.

Logistic regression is the interpretable floor (a linear model on scaled
features). XGBoost is the gradient-boosted tree challenger. We train both, score
them against StatsBomb's own xG on the *same* test shots, and persist the winner
(by log loss) to models/baseline.joblib.

`predict(state, shot_type=...)` is the serving entry point used by the API, the
frontend, and the sanity tests. It returns open-play model xG, or — for the
out-of-band `shot_type="penalty"` hint — the canonical penalty value.

Run:  python -m xg.models.baseline    (trains, evaluates, saves)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from xg.data.schema import GameState
from xg.eval.metrics import summary
from xg.features.build import (
    FEATURE_NAMES,
    build_dataset,
    state_to_features,
    test_mask_by_match,
)

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "baseline.joblib"

# Canonical penalty conversion rate — penalties are special-cased at serve time
# rather than learned by the open-play model.
PENALTY_XG = 0.76

_model = None  # lazily loaded artifact cache


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_models() -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        # Modest depth/estimators: ~2k samples, we want calibrated probabilities,
        # not an overfit tree.
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def train(save: bool = True) -> dict:
    X, y, groups, sb_xg = build_dataset()
    is_test = test_mask_by_match(groups)
    Xtr, Xte = X[~is_test], X[is_test]
    ytr, yte = y[~is_test], y[is_test]

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in _make_models().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        results[name] = summary(yte, p)
        fitted[name] = model

    # The bar to clear: StatsBomb's own xG on the identical test shots.
    results["statsbomb (benchmark)"] = summary(yte, sb_xg[is_test])

    best = min(("logreg", "xgboost"), key=lambda n: results[n]["log_loss"])

    print(f"{'model':22s} {'log_loss':>9s} {'brier':>8s} {'pred_g':>8s} {'actual_g':>9s}")
    for name, m in results.items():
        flag = "  <- best" if name == best else ""
        print(f"{name:22s} {m['log_loss']:9.4f} {m['brier']:8.4f} "
              f"{m['pred_goals']:8.1f} {m['actual_goals']:9.0f}{flag}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": fitted[best], "name": best, "features": FEATURE_NAMES,
             "metrics": results[best]},
            MODEL_PATH,
        )
        print(f"\nSaved best model ({best}) -> {MODEL_PATH}")
    return results


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
def load_model() -> dict:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run: python -m xg.models.baseline"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(state: GameState, shot_type: str = "open_play") -> float:
    """xG for a game state. `shot_type` is an out-of-band serving hint, not part
    of the GameState contract; "penalty" returns the canonical constant."""
    if shot_type == "penalty":
        return PENALTY_XG
    model = load_model()["model"]
    x = state_to_features(state).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


if __name__ == "__main__":
    train()
