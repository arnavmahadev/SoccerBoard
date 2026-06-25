"""Neural-net xG: a small PyTorch MLP on the same 9 features as the baseline.

This is the apples-to-apples PyTorch comparison (same features, same test split
as XGBoost). On ~2k rows of tabular data we expect trees to be hard to beat; the
value here is the training loop, honest comparison, and being able to explain the
result.

Run:  python -m xg.models.nn    (trains, evaluates vs baseline, saves)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from xg.data.schema import GameState
from xg.eval.metrics import summary
from xg.features.build import (
    FEATURE_NAMES,
    build_dataset,
    state_to_features,
    test_mask_by_match,
)

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "nn.pt"
_bundle = None  # lazily loaded (model + scaler) cache


class MLP(nn.Module):
    """9 -> 32 -> 16 -> 1 logit. Dropout regularizes the small dataset."""

    def __init__(self, n_features: int, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits


def _standardize(X, mean, std):
    return (X - mean) / std


def train(epochs: int = 200, save: bool = True) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    X, y, groups, sb_xg = build_dataset()
    X = X.to_numpy(dtype=np.float32)
    y = y.astype(np.float32)

    is_test = test_mask_by_match(groups)                       # same split as baseline
    Xtr_all, ytr_all = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]

    # Carve a validation set out of the training matches (different seed).
    is_val = test_mask_by_match(groups[~is_test], test_frac=0.15, seed=7)
    Xtr, ytr = Xtr_all[~is_val], ytr_all[~is_val]
    Xval, yval = Xtr_all[is_val], ytr_all[is_val]

    # Standardize on train statistics only.
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr_t = torch.tensor(_standardize(Xtr, mean, std))
    Xval_t = torch.tensor(_standardize(Xval, mean, std))
    Xte_t = torch.tensor(_standardize(Xte, mean, std))
    ytr_t, yval_t = torch.tensor(ytr), torch.tensor(yval)

    model = MLP(len(FEATURE_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n, batch = len(Xtr_t), 64
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(Xval_t)).numpy()
        val_ll = summary(yval, val_p)["log_loss"]
        if val_ll < best_val:                       # keep the best epoch
            best_val, best_state = val_ll, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_p = torch.sigmoid(model(Xte_t)).numpy()
    metrics = summary(yte, test_p)

    print(f"{'model':22s} {'log_loss':>9s} {'brier':>8s} {'pred_g':>8s} {'actual_g':>9s}")
    print(f"{'mlp (this)':22s} {metrics['log_loss']:9.4f} {metrics['brier']:8.4f} "
          f"{metrics['pred_goals']:8.1f} {metrics['actual_goals']:9.0f}")
    print(f"{'statsbomb (benchmark)':22s} "
          f"{summary(yte, sb_xg[is_test])['log_loss']:9.4f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "mean": mean, "std": std,
                    "features": FEATURE_NAMES, "metrics": metrics}, MODEL_PATH)
        print(f"\nSaved MLP -> {MODEL_PATH}")
    return metrics


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No MLP at {MODEL_PATH}. Run: python -m xg.models.nn")
        b = torch.load(MODEL_PATH, weights_only=False)
        model = MLP(len(b["features"]))
        model.load_state_dict(b["state_dict"])
        model.eval()
        _bundle = (model, b["mean"], b["std"])
    return _bundle


def predict(state: GameState, shot_type: str = "open_play") -> float:
    if shot_type == "penalty":
        from xg.models.baseline import PENALTY_XG
        return PENALTY_XG
    model, mean, std = load_model()
    x = _standardize(state_to_features(state).astype(np.float32), mean, std)
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.tensor(x).unsqueeze(0)))[0])


if __name__ == "__main__":
    train()
