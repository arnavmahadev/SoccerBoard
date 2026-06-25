"""DeepSets xG: a permutation-invariant network over the raw player set.

Instead of the 9 hand-engineered features, this model consumes the variable-
length set of players directly. Each player is embedded by a shared MLP (phi),
the embeddings are mean-pooled (order-independent), concatenated with a little
shot-level geometry, and passed through a second MLP (rho) to a logit.

Why it matters:
- It literally eats raw positions, so a future tracking pipeline feeds it with
  no feature engineering — the architecture principle taken to its conclusion.
- It is robust where the plain MLP was brittle: a wide-open chance is simply a
  smaller set (no nearby defenders), which pools cleanly — there is no "no
  defender" sentinel to push the network out of distribution.

This is a post-ship stretch; the served model remains XGBoost.

Run:  python -m xg.models.deepsets    (trains, evaluates vs benchmark, saves)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from xg.data.schema import GameState, GOAL_CENTER
from xg.eval.metrics import summary
from xg.features.build import build_dataset, shot_angle, shot_distance, test_mask_by_match

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "deepsets.pt"
_bundle = None

PLAYER_DIM = 9   # per-player feature width (see player_features)
GLOBAL_DIM = 3   # shot-level feature width (see global_features)


def player_features(state: GameState) -> np.ndarray:
    """Per-player rows: geometry relative to goal and shooter, plus role one-hots.
    Shape (n_players, PLAYER_DIM); empty set -> shape (0, PLAYER_DIM)."""
    sx, sy = state.shot_xy
    rows = []
    for p in state.players:
        px, py = p.xy
        rows.append([
            GOAL_CENTER[0] - px, GOAL_CENTER[1] - py,          # vector to goal
            math.hypot(GOAL_CENTER[0] - px, GOAL_CENTER[1] - py),
            sx - px, sy - py,                                   # vector to shooter
            math.hypot(sx - px, sy - py),
            1.0 if p.team == "att" else 0.0,
            1.0 if (p.team == "def" and not p.is_gk) else 0.0,
            1.0 if p.is_gk else 0.0,
        ])
    return np.array(rows, dtype=np.float32).reshape(-1, PLAYER_DIM)


def global_features(state: GameState) -> np.ndarray:
    return np.array(
        [shot_distance(state), shot_angle(state), abs(state.shot_xy[1] - GOAL_CENTER[1])],
        dtype=np.float32,
    )


class DeepSets(nn.Module):
    def __init__(self, player_dim=PLAYER_DIM, global_dim=GLOBAL_DIM, hidden=32, p_drop=0.2):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(player_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, players, mask, glob):
        # players: (B, P, player_dim), mask: (B, P), glob: (B, global_dim)
        emb = self.phi(players) * mask.unsqueeze(-1)          # zero out padded players
        pooled = emb.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # mean pool
        return self.rho(torch.cat([pooled, glob], dim=1)).squeeze(-1)


def _build_padded():
    """Build padded player tensors + masks + globals from the open-play dataset.
    Returns dict of numpy arrays and the match groups + StatsBomb benchmark."""
    X, y, groups, sb_xg = build_dataset()  # X unused; we rebuild raw from states
    # Re-read the states the same way build_dataset filtered them (open play).
    import pandas as pd
    df = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "processed" / "shots.parquet")
    df = df[df["shot_type"] == "Open Play"].reset_index(drop=True)
    states = [GameState.model_validate_json(g) for g in df["game_state"]]

    player_arrs = [player_features(s) for s in states]
    globals_ = np.stack([global_features(s) for s in states])
    max_p = max(a.shape[0] for a in player_arrs)

    n = len(states)
    players = np.zeros((n, max_p, PLAYER_DIM), dtype=np.float32)
    mask = np.zeros((n, max_p), dtype=np.float32)
    for i, a in enumerate(player_arrs):
        players[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = 1.0
    return dict(players=players, mask=mask, globals=globals_,
                y=y.astype(np.float32), groups=groups, sb_xg=sb_xg)


def _standardize_fit(players, mask, globals_):
    """Per-column mean/std for player features (over valid players only) and globals."""
    valid = players[mask.astype(bool)]                 # (sum_valid, PLAYER_DIM)
    p_mean, p_std = valid.mean(0), valid.std(0) + 1e-6
    g_mean, g_std = globals_.mean(0), globals_.std(0) + 1e-6
    return p_mean, p_std, g_mean, g_std


def train(epochs: int = 250, save: bool = True) -> dict:
    torch.manual_seed(42); np.random.seed(42)
    d = _build_padded()
    is_test = test_mask_by_match(d["groups"])

    def split(arr):
        return arr[~is_test], arr[is_test]
    pl_tr, pl_te = split(d["players"]); mk_tr, mk_te = split(d["mask"])
    gl_tr, gl_te = split(d["globals"]); y_tr, y_te = split(d["y"])

    # Standardize on train statistics; carve a match-based val split from train.
    p_mean, p_std, g_mean, g_std = _standardize_fit(pl_tr, mk_tr, gl_tr)
    pl_tr = (pl_tr - p_mean) / p_std; pl_te = (pl_te - p_mean) / p_std
    gl_tr = (gl_tr - g_mean) / g_std; gl_te = (gl_te - g_mean) / g_std

    is_val = test_mask_by_match(d["groups"][~is_test], test_frac=0.15, seed=7)
    t = lambda a: torch.tensor(a)
    tr = {k: t(v[~is_val]) for k, v in dict(p=pl_tr, m=mk_tr, g=gl_tr, y=y_tr).items()}
    va = {k: t(v[is_val]) for k, v in dict(p=pl_tr, m=mk_tr, g=gl_tr, y=y_tr).items()}

    model = DeepSets()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n, batch = len(tr["y"]), 64
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(tr["p"][idx], tr["m"][idx], tr["g"][idx]), tr["y"][idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(va["p"], va["m"], va["g"])).numpy()
        vll = summary(va["y"].numpy(), vp)["log_loss"]
        if vll < best_val:
            best_val, best_state = vll, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(t(pl_te), t(mk_te), t(gl_te))).numpy()
    metrics = summary(y_te, tp)

    print(f"{'model':22s} {'log_loss':>9s} {'brier':>8s}")
    print(f"{'deepsets (this)':22s} {metrics['log_loss']:9.4f} {metrics['brier']:8.4f}")
    print(f"{'statsbomb (benchmark)':22s} {summary(y_te, d['sb_xg'][is_test])['log_loss']:9.4f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "p_mean": p_mean, "p_std": p_std,
                    "g_mean": g_mean, "g_std": g_std, "metrics": metrics}, MODEL_PATH)
        print(f"\nSaved DeepSets -> {MODEL_PATH}")
    return metrics


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No DeepSets model at {MODEL_PATH}. Run: python -m xg.models.deepsets")
        b = torch.load(MODEL_PATH, weights_only=False)
        model = DeepSets(); model.load_state_dict(b["state_dict"]); model.eval()
        _bundle = (model, b)
    return _bundle


def predict(state: GameState, shot_type: str = "open_play") -> float:
    if shot_type == "penalty":
        from xg.models.baseline import PENALTY_XG
        return PENALTY_XG
    model, b = load_model()
    pl = (player_features(state) - b["p_mean"]) / b["p_std"]      # (n, PLAYER_DIM)
    gl = (global_features(state) - b["g_mean"]) / b["g_std"]
    players = torch.tensor(pl).unsqueeze(0)                        # (1, n, D)
    mask = torch.ones(1, pl.shape[0]) if pl.shape[0] else torch.zeros(1, 1)
    if pl.shape[0] == 0:                                           # empty set -> one padded row
        players = torch.zeros(1, 1, PLAYER_DIM)
    with torch.no_grad():
        return float(torch.sigmoid(model(players, mask, torch.tensor(gl).unsqueeze(0)))[0])


if __name__ == "__main__":
    train()
