"""Backtest + calibration for the Dixon-Coles scoreline model.

This is the honest-evaluation core of the forecaster. We do a strict
chronological split — train on every international match before a cutoff, then
predict matches *after* it — and score the predicted match-outcome
probabilities (home / draw / away) the model never saw:

  - **log loss** and **multiclass Brier score** (lower is better);
  - a **reliability/calibration curve** (predicted-probability bucket vs. observed
    frequency) and its **ECE** (expected calibration error);
  - a **naive base-rate baseline** (predict the train-set home/draw/away
    frequencies for every match) so the headline numbers have context.

Calibration is the part that signals the model is trustworthy: a predicted 60%
should win ~60% of the time. We pool all three class probabilities across all
test matches into the reliability diagram (each match contributes 3 points).

Run:  python -m forecaster.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecaster.data import WC2026_KICKOFF, load_matches
from forecaster import dixon_coles as dc

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
EPS = 1e-15


def _outcome(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _metrics(probs: np.ndarray, actual: np.ndarray) -> dict:
    """probs: (N, 3) predicted; actual: (N,) class indices."""
    n = len(actual)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual] = 1.0
    p = np.clip(probs, EPS, 1.0)
    log_loss = float(-np.mean(np.log(p[np.arange(n), actual])))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == actual))
    return {"log_loss": log_loss, "brier": brier, "accuracy": accuracy, "n": n}


def _calibration(probs: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict:
    """Pooled multiclass reliability curve + ECE over all predicted probabilities."""
    n = len(actual)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual] = 1.0
    p = probs.reshape(-1)
    y = onehot.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    total = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_pred = float(p[mask].mean())
        mean_obs = float(y[mask].mean())
        bins.append({"p_pred": mean_pred, "p_obs": mean_obs, "n": cnt})
        ece += (cnt / total) * abs(mean_pred - mean_obs)
    return {"bins": bins, "ece": float(ece), "n_pairs": total}


def backtest(
    train_since: str = "2008-01-01",
    cutoff: str = "2024-01-01",
    test_end: str = WC2026_KICKOFF,
    xi: float = 0.35,
    reg: float = 0.02,
) -> dict:
    """Train < cutoff, test in [cutoff, test_end). Returns a metrics dict."""
    train = load_matches(as_of=cutoff, since=train_since)
    train = [m for m in train if m.date < cutoff]
    test = [m for m in load_matches(as_of=test_end, since=cutoff) if m.date >= cutoff]
    if not test:
        raise ValueError("empty test set")

    params = dc.fit(train, xi=xi, reg=reg, ref_date=cutoff)

    probs = np.zeros((len(test), 3))
    actual = np.zeros(len(test), dtype=int)
    for i, m in enumerate(test):
        mat = dc.predict(params, m.home, m.away, neutral=m.neutral)
        probs[i] = dc.outcome_probs(mat)
        actual[i] = _outcome(m.home_goals, m.away_goals)

    # Baseline: predict the train-set base rates for every match.
    rates = np.bincount(
        [_outcome(m.home_goals, m.away_goals) for m in train], minlength=3
    ).astype(float)
    rates /= rates.sum()
    base_probs = np.tile(rates, (len(test), 1))

    model_m = _metrics(probs, actual)
    base_m = _metrics(base_probs, actual)
    calib = _calibration(probs, actual)

    return {
        "competition": "world_cup_2026",
        "model_name": "Dixon-Coles (time-decay + ridge)",
        "config": {
            "train_since": train_since,
            "cutoff": cutoff,
            "test_end": test_end,
            "xi": xi,
            "reg": reg,
            "n_train": len(train),
            "n_test": len(test),
            "n_teams": len(params.teams),
        },
        "model": {**model_m, "ece": calib["ece"]},
        "baseline": {"name": "base rates (home/draw/away)", **base_m},
        "base_rates": {"home": rates[0], "draw": rates[1], "away": rates[2]},
        "calibration": calib,
    }


def _print_summary(r: dict) -> None:
    c = r["config"]
    print(f"\nBacktest: train<{c['cutoff']} ({c['n_train']} matches, {c['n_teams']} teams)"
          f" -> test {c['cutoff']}..{c['test_end']} ({c['n_test']} matches)")
    print(f"{'':22s}{'log loss':>10s}{'Brier':>9s}{'ECE':>8s}{'acc':>8s}")
    m, b = r["model"], r["baseline"]
    print(f"{'Dixon-Coles':22s}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
          f"{m['ece']:>8.4f}{m['accuracy']:>8.1%}")
    print(f"{b['name']:22s}{b['log_loss']:>10.4f}{b['brier']:>9.4f}{'—':>8s}"
          f"{b['accuracy']:>8.1%}")
    rr = r["base_rates"]
    print(f"\nTest base rates: home {rr['home']:.0%} / draw {rr['draw']:.0%} / "
          f"away {rr['away']:.0%}")


def main() -> None:
    r = backtest()
    _print_summary(r)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "metrics.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
