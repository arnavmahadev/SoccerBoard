"""Serving layer for the forecaster — the only forecaster code the API imports.

Loads the committed artifacts once (fitted Dixon-Coles params, the competition
config, the backtest metrics, the pre-tournament group forecast) and exposes the
queries the FastAPI routes need. Imports numpy + scipy only at fit time; serving
predictions and simulations are pure numpy, mirroring how the xG side keeps the
serving path lightweight.

The **live** behaviour lives here: `simulation()` re-reads the latest results
(clamped to an as-of date), re-derives which knockout games are settled and
re-simulates the rest, with a short TTL cache so repeated requests are cheap.
Everything else (the frozen strengths, the backtest, the pre-tournament group
forecast) is static and committed.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np

from forecaster import dixon_coles as dc
from forecaster.data import (
    get_competition,
    list_competitions as _list_competitions,
    load_matches,
    tournament_matches,
)
from forecaster.dixon_coles import Params
from forecaster.formats.base import MatchSampler, monte_carlo
from forecaster.formats.world_cup import WorldCupFormat

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SIM_TTL_SECONDS = 120.0  # re-simulate at most this often per as-of date
LIVE_TTL = 120.0         # re-fetch the results feed at most this often

_lock = threading.Lock()
_params: Params | None = None
_sampler: MatchSampler | None = None
_metrics: dict | None = None
_group_forecast: dict | None = None
_sim_cache: dict[tuple, tuple[float, dict]] = {}


def load():
    """Warm the cache / fail fast if artifacts are missing."""
    global _params, _sampler, _metrics, _group_forecast
    if _params is not None:
        return
    with _lock:
        if _params is not None:
            return
        pp = ARTIFACTS / "params.json"
        if not pp.exists():
            raise FileNotFoundError(
                f"No fitted params at {pp}. Run: python -m forecaster.build_artifacts"
            )
        _params = Params.from_json(pp)
        _sampler = MatchSampler(_params)
        mp = ARTIFACTS / "metrics.json"
        _metrics = json.loads(mp.read_text()) if mp.exists() else {}
        gp = ARTIFACTS / "group_forecast.json"
        _group_forecast = json.loads(gp.read_text()) if gp.exists() else {}


def _today() -> str:
    return date.today().isoformat()


# --- Catalogue ---------------------------------------------------------------
def list_competitions() -> list[dict]:
    return _list_competitions()


def teams(competition: str) -> list[str]:
    cfg = get_competition(competition)
    return sorted({t for members in cfg["groups"].values() for t in members})


# --- Match predictor ---------------------------------------------------------
def predict_match(home: str, away: str, neutral: bool = True, display: int = 6) -> dict:
    load()
    mat = dc.predict(_params, home, away, neutral)
    ph, pd, pa = dc.outcome_probs(mat)
    lam, mu = dc.goal_expectations(_params, home, away, neutral)
    i, j = dc.most_likely_score(mat)
    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "prob_home": ph,
        "prob_draw": pd,
        "prob_away": pa,
        "exp_home": lam,
        "exp_away": mu,
        "most_likely": [int(i), int(j)],
        # small goal-matrix slice for the SVG heatmap (rows = home goals 0..N)
        "matrix": mat[:display, :display].tolist(),
        "matrix_max": int(display - 1),
    }


# --- Live competition simulation ---------------------------------------------
def simulation(competition: str, as_of: str | None = None, n: int = 10000) -> dict:
    """Live per-team stage probabilities. Re-derives settled knockout games from
    the latest results (<= as_of) and re-simulates the remainder."""
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()
    key = (competition, as_of, n)

    cached = _sim_cache.get(key)
    if cached and (time.time() - cached[0]) < SIM_TTL_SECONDS:
        return cached[1]

    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    fixed = WorldCupFormat.fixed_knockout(cfg, matches)
    fmt = WorldCupFormat(cfg, fixed_results=fixed)
    raw = monte_carlo(fmt, _sampler, n=n, seed=12345)

    rows = []
    group_of = {t: g for g, members in cfg["groups"].items() for t in members}
    for team, probs in raw["teams"].items():
        rows.append({"team": team, "group": group_of.get(team), **probs})
    rows.sort(key=lambda r: (-r.get("champion", 0), -r.get("final", 0)))

    settled = [
        {"teams": sorted(pair), "winner": w} for pair, w in fixed.items()
    ]
    result = {
        "competition": competition,
        "name": cfg["name"],
        "as_of": as_of,
        "n": n,
        "stages": raw["stages"],
        "teams": rows,
        "settled_knockout": settled,
        "kickoff": cfg.get("kickoff"),
    }
    _sim_cache[key] = (time.time(), result)
    return result


_ROUND_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarterfinal": "Quarter-finals",
    "semifinal": "Semi-finals",
    "final": "Final",
}


def _advance_prob(home: str, away: str) -> float:
    """P(home advances) in a neutral knockout tie (normal time or shootout)."""
    mat = dc.predict(_params, home, away, neutral=True)
    ph, pdraw, pa = dc.outcome_probs(mat)
    denom = ph + pa
    share = ph / denom if denom > 1e-12 else 0.5
    return ph + pdraw * share


_BRACKET_ROUNDS = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]


def _bracket_view(cfg: dict, played: dict, mode: str) -> dict:
    """Build one bracket. mode='prediction' advances the model's favourite for
    every game (ignoring results); mode='actual' advances the real winners and
    leaves undecided ties as TBD, flagging whether each result matched the pick."""
    ko = cfg["knockout"]
    r32 = {f["match"]: tuple(f["teams"]) for f in ko["round_of_32"]}
    tree = {int(k): v for k, v in ko["tree"].items()}
    match_round = {m: r for r, ms in ko["rounds"].items() for m in ms}
    order = sorted(set(r32) | set(tree))

    winners: dict[int, str | None] = {}
    built: dict[int, dict] = {}
    correct = decided = 0
    for m in order:
        if m in r32:
            a, b = r32[m]
        else:
            a, b = winners.get(tree[m][0]), winners.get(tree[m][1])
        entry = {"match": m, "a": a, "b": b}
        if mode == "prediction":
            pa = _advance_prob(a, b)
            winner = a if pa >= 0.5 else b
            entry.update(prob_a=pa, prob_b=1.0 - pa, winner=winner, settled=False)
        else:
            rec = played.get(frozenset((a, b))) if (a and b) else None
            if rec:
                winner = rec["winner"]
                score = (
                    [rec["home_goals"], rec["away_goals"]]
                    if rec["home"] == a else [rec["away_goals"], rec["home_goals"]]
                )
                pa = _advance_prob(a, b)
                pred = a if pa >= 0.5 else b
                hit = winner == pred
                decided += 1
                correct += int(hit)
                entry.update(winner=winner, score=score, settled=True,
                             predicted_winner=pred, correct=hit)
            else:
                winner = None
                entry.update(winner=None, settled=False)
        winners[m] = winner
        built[m] = entry

    rounds = [
        {"round": r, "label": _ROUND_LABELS[r], "matches": [built[m] for m in ko["rounds"][r]]}
        for r in _BRACKET_ROUNDS
    ]
    out = {"champion": winners.get(ko["final_match"]), "rounds": rounds}
    if mode == "actual":
        out["correct"] = correct
        out["decided"] = decided
    return out


def bracket(competition: str, as_of: str | None = None) -> dict:
    """Two brackets for side-by-side comparison: the model's predictions for every
    knockout game, and the actual results as they come in (with hit/miss flags)."""
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    played = WorldCupFormat.knockout_played(cfg, matches)
    return {
        "competition": competition,
        "name": cfg["name"],
        "as_of": as_of,
        "settled_count": len(played),
        "prediction": _bracket_view(cfg, {}, "prediction"),
        "actual": _bracket_view(cfg, played, "actual"),
    }


# --- Group stage: forecast + actual (both views) -----------------------------
def _actual_group_standings(cfg: dict, matches: list) -> dict[str, list[dict]]:
    """Real group tables from played group fixtures (each team's first 3 games)."""
    from collections import defaultdict

    group_pairs = {
        frozenset((fx["home"], fx["away"])) for fx in cfg["group_fixtures"]
    }
    pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int); pl = defaultdict(int)
    for m in matches:
        if not m.played:
            continue
        if frozenset((m.home, m.away)) not in group_pairs:
            continue
        pl[m.home] += 1; pl[m.away] += 1
        gf[m.home] += m.home_goals; ga[m.home] += m.away_goals
        gf[m.away] += m.away_goals; ga[m.away] += m.home_goals
        if m.home_goals > m.away_goals:
            pts[m.home] += 3
        elif m.home_goals < m.away_goals:
            pts[m.away] += 3
        else:
            pts[m.home] += 1; pts[m.away] += 1

    qualifiers = {t for f in cfg["knockout"]["round_of_32"] for t in f["teams"]}
    out = {}
    for letter, members in cfg["groups"].items():
        table = sorted(
            members, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True
        )
        out[letter] = [
            {
                "team": t, "played": pl[t], "points": pts[t],
                "gd": gf[t] - ga[t], "gf": gf[t],
                "position": i + 1, "advanced": t in qualifiers,
            }
            for i, t in enumerate(table)
        ]
    return out


def group_view(competition: str, as_of: str | None = None) -> dict:
    """Per-group: pre-tournament advancement forecast next to the actual table."""
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    actual = _actual_group_standings(cfg, matches)
    fc = _group_forecast.get(competition, {})

    groups = {}
    for letter, rows in actual.items():
        merged = []
        for r in rows:
            f = fc.get(r["team"], {})
            merged.append({
                **r,
                "forecast_win_group": f.get("win_group"),
                "forecast_advance": f.get("advance"),
            })
        groups[letter] = merged
    return {"competition": competition, "name": cfg["name"], "groups": groups}


def group_matches(competition: str, as_of: str | None = None) -> dict:
    """Per-match group-stage predictions (W/D/L + expected score) vs the actual
    result each game got."""
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    group_pairs = {frozenset((f["home"], f["away"])): f for f in cfg["group_fixtures"]}
    actual_by_pair = {}
    for m in matches:
        if m.played and frozenset((m.home, m.away)) in group_pairs:
            actual_by_pair[frozenset((m.home, m.away))] = m

    out = []
    for fx in cfg["group_fixtures"]:
        pred = predict_match(fx["home"], fx["away"], neutral=fx["neutral"])
        am = actual_by_pair.get(frozenset((fx["home"], fx["away"])))
        actual = None
        if am is not None:
            # orient actual to the fixture's home/away
            if am.home == fx["home"]:
                ah, aa = am.home_goals, am.away_goals
            else:
                ah, aa = am.away_goals, am.home_goals
            actual = {"home_goals": ah, "away_goals": aa}
        out.append({
            "home": fx["home"], "away": fx["away"],
            "prob_home": pred["prob_home"], "prob_draw": pred["prob_draw"],
            "prob_away": pred["prob_away"], "most_likely": pred["most_likely"],
            "exp_home": pred["exp_home"], "exp_away": pred["exp_away"],
            "actual": actual,
        })
    return {"competition": competition, "name": cfg["name"], "matches": out}


# --- Metrics -----------------------------------------------------------------
def metrics(competition: str | None = None) -> dict:
    load()
    return _metrics or {}


def model_info() -> dict:
    load()
    return {
        "model": "dixon-coles",
        "teams": len(_params.teams),
        "n_matches": _params.n_matches,
        "home_adv": _params.home_adv,
        "rho": _params.rho,
        "xi": _params.xi,
        "ref_date": _params.ref_date,
    }
