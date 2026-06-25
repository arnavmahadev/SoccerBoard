"""Turn a GameState into a fixed feature vector.

`state_to_features` is THE single source of truth used by both training and
serving — guaranteeing the model sees identical features in both. Every feature
is computed from positions alone (no body part, no shot type), so anything the
frontend can draw, the model can score.

Run:  python -m xg.features.build   (prints feature shapes + a scenario check)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from xg.data.schema import (
    GameState,
    GOAL_CENTER,
    GOAL_POST_LEFT,
    GOAL_POST_RIGHT,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

# Ordered feature list — the column contract for X. Keep in sync with _features().
FEATURE_NAMES: list[str] = [
    "distance",            # to goal centre
    "angle",               # goal mouth subtended at the shooter (radians)
    "abs_y_offset",        # |y - 40|: how far off-centre
    "n_defenders",         # defenders visible in the freeze frame
    "defenders_in_cone",   # defenders inside the shot triangle (shooter -> posts)
    "nearest_def_dist",    # distance to closest outfield defender
    "gk_visible",          # 1 if a defending keeper is in view
    "gk_dist_to_goal",     # how far the keeper is off his line
    "gk_dist_to_shot",     # keeper distance from the shooter
]

_FAR = float(math.hypot(PITCH_LENGTH, PITCH_WIDTH))  # ~144, used as "no one nearby"
_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shot_distance(state: GameState) -> float:
    return _dist(state.shot_xy, GOAL_CENTER)


def shot_angle(state: GameState) -> float:
    """Angle (radians) subtended by the two goalposts at the shooter — the
    'how much net can I see' feature. Wide open => large; tight angle => ~0."""
    s = state.shot_xy
    vl = (GOAL_POST_LEFT[0] - s[0], GOAL_POST_LEFT[1] - s[1])
    vr = (GOAL_POST_RIGHT[0] - s[0], GOAL_POST_RIGHT[1] - s[1])
    dot = vl[0] * vr[0] + vl[1] * vr[1]
    mag = math.hypot(*vl) * math.hypot(*vr)
    if mag == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / mag)))


def _point_in_triangle(p, a, b, c) -> bool:
    """Sign test: is point p inside triangle a-b-c?"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def defenders_in_cone(state: GameState) -> int:
    """Defenders standing inside the triangle from the shooter to the posts —
    bodies the shot has to beat."""
    tri = (state.shot_xy, GOAL_POST_LEFT, GOAL_POST_RIGHT)
    return sum(_point_in_triangle(d.xy, *tri) for d in state.defenders if not d.is_gk)


def nearest_defender_distance(state: GameState) -> float:
    outfield = [d for d in state.defenders if not d.is_gk]
    if not outfield:
        return _FAR
    return min(_dist(state.shot_xy, d.xy) for d in outfield)


def _features(state: GameState) -> dict[str, float]:
    gk = state.goalkeeper
    return {
        "distance": shot_distance(state),
        "angle": shot_angle(state),
        "abs_y_offset": abs(state.shot_xy[1] - GOAL_CENTER[1]),
        "n_defenders": float(len([d for d in state.defenders if not d.is_gk])),
        "defenders_in_cone": float(defenders_in_cone(state)),
        "nearest_def_dist": nearest_defender_distance(state),
        "gk_visible": 1.0 if gk is not None else 0.0,
        # If the keeper isn't in view, assume a default on-line keeper.
        "gk_dist_to_goal": _dist(gk.xy, GOAL_CENTER) if gk else 2.0,
        "gk_dist_to_shot": _dist(gk.xy, state.shot_xy) if gk else shot_distance(state),
    }


def state_to_features(state: GameState) -> np.ndarray:
    """The serving entry point: GameState -> 1-D feature array in FEATURE_NAMES order."""
    f = _features(state)
    return np.array([f[name] for name in FEATURE_NAMES], dtype=float)


def build_dataset(parquet_path: Path | None = None):
    """Read the saved shots, keep OPEN PLAY only, and build X (DataFrame), y,
    groups, and StatsBomb's own xG (carried along so we can benchmark on the
    identical test split).

    Returns (X, y, match_ids, statsbomb_xg). Penalties/free kicks/corners are
    excluded per the open-play-only modelling decision."""
    import pandas as pd  # only needed for data loading, not for serving

    path = parquet_path or (_PROCESSED / "shots.parquet")
    df = pd.read_parquet(path)
    df = df[df["shot_type"] == "Open Play"].reset_index(drop=True)

    rows = [_features(GameState.model_validate_json(g)) for g in df["game_state"]]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["is_goal"].to_numpy()
    match_ids = df["match_id"].to_numpy()
    statsbomb_xg = df["statsbomb_xg"].to_numpy()
    return X, y, match_ids, statsbomb_xg


def test_mask_by_match(match_ids, test_frac: float = 0.2, seed: int = 42) -> np.ndarray:
    """Boolean mask marking which rows are test, assigned whole-match-at-a-time
    so no match-level context leaks across the split."""
    rng = np.random.default_rng(seed)
    matches = np.unique(match_ids)
    rng.shuffle(matches)
    test_matches = set(matches[: int(len(matches) * test_frac)].tolist())
    return np.array([m in test_matches for m in match_ids])


def split_by_match(X, y, match_ids, test_frac: float = 0.2, seed: int = 42):
    """Convenience wrapper: split X and y by match into (Xtr, Xte, ytr, yte)."""
    is_test = test_mask_by_match(match_ids, test_frac, seed)
    return (X[~is_test], X[is_test], y[~is_test], y[is_test])


def main() -> None:
    from xg.scenarios import ALL

    X, y, groups, _ = build_dataset()
    Xtr, Xte, ytr, yte = split_by_match(X, y, groups)
    print(f"Open-play shots: {len(X)} | features: {len(FEATURE_NAMES)}")
    print(f"Train {len(Xtr)} ({ytr.mean():.1%} goals) | "
          f"Test {len(Xte)} ({yte.mean():.1%} goals) | "
          f"{len(np.unique(groups))} matches split by game")
    print("\nScenario feature check (distance / angle):")
    for sc in ALL:
        print(f"  {sc.name:20s} dist={shot_distance(sc.state):5.1f}  "
              f"angle={math.degrees(shot_angle(sc.state)):5.1f} deg")


if __name__ == "__main__":
    main()
