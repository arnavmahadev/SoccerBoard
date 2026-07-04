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
import math
import threading
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from forecaster import dixon_coles as dc
from forecaster.data import (
    Match,
    get_competition,
    list_competitions as _list_competitions,
    load_matches,
    normalize_team,
    tournament_matches,
)
from forecaster.dixon_coles import Params
from forecaster.formats.base import MatchSampler, monte_carlo
from forecaster.formats.world_cup import WorldCupFormat
from forecaster.player_model import PlayerDeltas

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
NEWS_ITEMS_PATH = Path(__file__).resolve().parent / "news_items.json"
DEFAULT_COMPETITION = "world_cup_2026"
SIM_TTL_SECONDS = 120.0  # re-simulate at most this often per as-of date
LIVE_TTL = 120.0         # re-fetch the results feed at most this often

# In-tournament performance overlay: nudge a team's frozen strength by how far
# its actual tournament scorelines have diverged from what the model expected, so
# a side that has been labouring (or flying) shifts the live forecast a little.
PERF_ETA = 0.20      # fraction of the pooled log-surprise folded into the rating
PERF_SMOOTH = 0.7    # per-goal smoothing in the log ratio (avoids log 0, damps blowups)
PERF_CAP = 0.5       # cap on |adjustment| per team per side

# Knockout single-game variance: how far each knockout tie's advance probability
# is regressed toward a coin flip. A knockout is one match, far noisier than a
# season-long rating, so without this the simulator over-concentrates the title
# on the top seed and understates how often favourites actually go out.
KNOCKOUT_UPSET = 0.20

_lock = threading.Lock()
_params: Params | None = None
_sampler: MatchSampler | None = None
_metrics: dict | None = None
_group_forecast: dict | None = None
_news: dict | None = None             # raw news_items.json content
_player_deltas: PlayerDeltas | None = None
_adj_params: dict[str, Params] = {}   # competition -> player-adjusted params
_sim_cache: dict[tuple, tuple[float, dict]] = {}

_reality_lock = threading.Lock()
_reality_params: Params | None = None
_reality_sampler: MatchSampler | None = None
_reality_cache_time: float = 0.0
_REALITY_TTL = 3600.0        # re-fit reality DC at most once per hour
_REALITY_FIT_SINCE = "2010-01-01"


def load():
    """Warm the cache / fail fast if artifacts are missing."""
    global _params, _sampler, _metrics, _group_forecast, _news, _player_deltas
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
        _sampler = MatchSampler(_params, upset=KNOCKOUT_UPSET)
        mp = ARTIFACTS / "metrics.json"
        _metrics = json.loads(mp.read_text()) if mp.exists() else {}
        gp = ARTIFACTS / "group_forecast.json"
        _group_forecast = json.loads(gp.read_text()) if gp.exists() else {}
        _news = (
            json.loads(NEWS_ITEMS_PATH.read_text()) if NEWS_ITEMS_PATH.exists() else {}
        )
        dp = ARTIFACTS / "player_deltas.json"
        _player_deltas = PlayerDeltas.from_json(dp) if dp.exists() else None


# --- Player-delta news overlay -----------------------------------------------
# Absent players (injuries / suspensions) listed in news_items.json are looked
# up in the fitted PlayerDeltas artifact. Their learned attack and defense
# contributions are subtracted from their team's effective params. No rubric
# tiers — magnitudes come from what the model learnt across international
# tournaments (StatsBomb data). Players not covered by that data get delta≈0,
# meaning no adjustment, which is the correct regularisation default.

def _absent_players(competition: str) -> dict[str, list[str]]:
    """team -> list of absent player names from news_items.json."""
    comp = (_news or {}).get(competition) or {}
    teams = comp.get("teams") or {}
    return {team: [it["player"] for it in items] for team, items in teams.items() if items}


def _params_for(competition: str) -> Params:
    """Base params with absent players' learned deltas subtracted."""
    load()
    absent = _absent_players(competition)
    if not absent or _player_deltas is None:
        return _params
    if competition not in _adj_params:
        attack = list(_params.attack)
        defense = list(_params.defense)
        tidx = _params.index()
        for team, players in absent.items():
            if team not in tidx:
                continue
            ti = tidx[team]
            for player in players:
                pi = _player_deltas.find_player(player)
                if pi is None:
                    continue  # player not in training data → no adjustment
                # Only apply when the player's delta is positive (they genuinely
                # help the team). A negative delta likely reflects sparse data or
                # lineup correlation artifacts, not that the player hurts the team;
                # we cap at 0 so an injury can never make a team look stronger.
                attack[ti]  -= max(_player_deltas.att_delta[pi], 0.0)
                defense[ti] -= max(_player_deltas.def_delta[pi], 0.0)
        _adj_params[competition] = replace(_params, attack=attack, defense=defense)
    return _adj_params[competition]



# --- In-tournament performance overlay ---------------------------------------
# The news overlay above adjusts for who is *available*; this adjusts for how a
# team has actually *played* so far. We pool each team's tournament goals scored
# and conceded against what the frozen model expected across those same fixtures,
# and shift attack/defense by a small, capped fraction of that surprise (on a log
# scale). Working from pooled totals — not a per-match sum — keeps the nudge a
# monotonic function of "scored X vs Y expected", so the plain-English panel the
# UI shows can never contradict the direction the rating actually moved.
# Performing exactly to expectation — a favourite easing past a minnow it was
# meant to beat — moves nothing; only over/under-performance relative to strength
# registers. The small learning rate keeps the pre-tournament fit dominant: this
# bends the live odds, it doesn't break them, which is the point of the
# frozen-strength design.

def _played_tournament_matches(cfg: dict, feed_matches: list[Match]) -> list[Match]:
    """Every played tournament game: the live feed plus knockout result-overrides
    (recent games not yet in the feed), deduped by the pair of teams."""
    played = [m for m in feed_matches if m.played]
    seen = {frozenset((m.home, m.away)) for m in played}
    for r in cfg.get("knockout", {}).get("result_overrides", []):
        home, away = normalize_team(r["home"]), normalize_team(r["away"])
        if frozenset((home, away)) in seen:
            continue
        played.append(Match(
            date=cfg.get("group_stage_end", ""), home=home, away=away,
            home_goals=r["home_goals"], away_goals=r["away_goals"],
            neutral=True, tournament=cfg["tournament_label"],
        ))
    return played


def _performance_overlay(base: Params, played: list[Match]) -> tuple[Params, list[dict]]:
    """Base params nudged by tournament over/under-performance, plus a per-team
    breakdown of goals scored/conceded vs expected and the applied attack/defense
    deltas (largest movement first)."""
    tidx = base.index()
    # team -> [matches, goals_for, exp_for, goals_against, exp_against]
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0, 0.0])
    for m in played:
        if m.home not in tidx or m.away not in tidx:
            continue
        lam, mu = dc.goal_expectations(base, m.home, m.away, m.neutral)
        h = agg[m.home]; h[0] += 1; h[1] += m.home_goals; h[2] += lam; h[3] += m.away_goals; h[4] += mu
        a = agg[m.away]; a[0] += 1; a[1] += m.away_goals; a[2] += mu; a[3] += m.home_goals; a[4] += lam

    def clamp(v: float) -> float:
        return max(-PERF_CAP, min(PERF_CAP, PERF_ETA * v))

    attack = list(base.attack)
    defense = list(base.defense)
    rows = []
    for t, (n, gf, xgf, ga, xga) in agg.items():
        c = PERF_SMOOTH * n  # per-goal smoothing scaled to the number of games
        # Scored more than expected -> attack up; conceded fewer -> defense up.
        da = clamp(math.log((gf + c) / (xgf + c)))
        dd = clamp(math.log((xga + c) / (ga + c)))
        attack[tidx[t]] += da
        defense[tidx[t]] += dd
        rows.append({
            "team": t, "att_delta": round(da, 4), "def_delta": round(dd, 4),
            "matches": int(n), "gf": int(gf), "xgf": round(xgf, 1),
            "ga": int(ga), "xga": round(xga, 1),
        })
    rows.sort(key=lambda r: abs(r["att_delta"]) + abs(r["def_delta"]), reverse=True)
    return replace(base, attack=attack, defense=defense), rows


def adjustments(competition: str = DEFAULT_COMPETITION) -> dict:
    """Active injury/suspension news for a competition, with learned delta magnitudes."""
    load()
    comp = (_news or {}).get(competition) or {}
    teams_news = comp.get("teams") or {}
    tidx = _params.index()

    rows = []
    for team, items in teams_news.items():
        if team not in tidx or not items:
            continue
        enriched = []
        for it in items:
            pname = it["player"]
            pi = _player_deltas.find_player(pname) if _player_deltas else None
            att_d = float(max(_player_deltas.att_delta[pi], 0.0)) if pi is not None and _player_deltas else 0.0
            def_d = float(max(_player_deltas.def_delta[pi], 0.0)) if pi is not None and _player_deltas else 0.0
            n_matches = int(_player_deltas.n_matches[pi]) if pi is not None and _player_deltas else 0
            enriched.append({
                "player": pname,
                "issue": it.get("issue", ""),
                "att_delta": round(att_d, 4),
                "def_delta": round(def_d, 4),
                "covered": pi is not None,
                "n_matches": n_matches,
            })
        seen_sources: dict[str, str] = {}
        for it in items:
            if it.get("source") and it["source"] not in seen_sources:
                seen_sources[it["source"]] = it.get("url", "")
        rows.append({
            "team": team,
            "items": enriched,
            "sources": [{"label": s, "url": u} for s, u in seen_sources.items()],
        })
    rows.sort(key=lambda r: r["team"])
    return {"competition": competition, "updated": comp.get("updated"), "teams": rows}


def _get_reality_params(as_of: str) -> tuple[Params, MatchSampler]:
    """Re-fit DC on all available matches up to as_of (including tournament games)."""
    global _reality_params, _reality_sampler, _reality_cache_time
    with _reality_lock:
        if _reality_params is not None and (time.time() - _reality_cache_time) < _REALITY_TTL:
            return _reality_params, _reality_sampler
        all_matches = load_matches(
            as_of=as_of, since=_REALITY_FIT_SINCE,
            prefer_live=True, ttl_seconds=LIVE_TTL,
        )
        p = dc.fit(all_matches, xi=0.25, reg=0.02, ref_date=as_of)
        _reality_params = p
        _reality_sampler = MatchSampler(p, upset=KNOCKOUT_UPSET)
        _reality_cache_time = time.time()
    return _reality_params, _reality_sampler


def _today() -> str:
    return date.today().isoformat()


# --- Catalogue ---------------------------------------------------------------
def list_competitions() -> list[dict]:
    return _list_competitions()


def teams(competition: str) -> list[str]:
    cfg = get_competition(competition)
    return sorted({t for members in cfg["groups"].values() for t in members})


# --- Match predictor ---------------------------------------------------------
def predict_match(
    home: str, away: str, neutral: bool = True, display: int = 6,
    competition: str = DEFAULT_COMPETITION,
) -> dict:
    load()
    params = _params_for(competition)  # base ratings + any news overlay
    mat = dc.predict(params, home, away, neutral)
    ph, pd, pa = dc.outcome_probs(mat)
    lam, mu = dc.goal_expectations(params, home, away, neutral)
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
def simulation(
    competition: str, as_of: str | None = None, n: int = 10000, mode: str = "live",
) -> dict:
    """Per-team stage probabilities.

    mode='live'         pre-tournament DC params + news overlay + real bracket.
    mode='pretournament' pre-tournament DC params, no overlay, no settled games
                        (simulates all R32 ties from scratch).
    mode='reality'      DC re-fitted with all tournament results to date + real
                        bracket (expensive; cached 1 h).
    """
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()

    cache_as_of = "__pretournament__" if mode == "pretournament" else as_of
    key = (competition, cache_as_of, n, mode)
    sim_ttl = _REALITY_TTL if mode in ("pretournament", "reality") else SIM_TTL_SECONDS

    cached = _sim_cache.get(key)
    if cached and (time.time() - cached[0]) < sim_ttl:
        return cached[1]

    perf_rows: list[dict] = []
    if mode == "pretournament":
        fmt = WorldCupFormat(cfg)
        raw = monte_carlo(fmt, _sampler, n=n, seed=12345)
        fixed = {}
    else:
        matches = tournament_matches(
            cfg["tournament_label"], cfg["season_year"], as_of=as_of,
            prefer_live=True, ttl_seconds=LIVE_TTL,
        )
        fixed = WorldCupFormat.fixed_knockout(cfg, matches)
        fmt = WorldCupFormat(cfg, fixed_results=fixed)
        if mode == "reality":
            _, r_sampler = _get_reality_params(as_of)
            raw = monte_carlo(fmt, r_sampler, n=n, seed=12345)
        else:
            # Live: frozen strengths + news overlay, then bent by how each side has
            # actually played in the tournament so far.
            base = _params_for(competition)
            played = _played_tournament_matches(cfg, matches)
            perf_params, perf_rows = _performance_overlay(base, played)
            raw = monte_carlo(fmt, MatchSampler(perf_params, upset=KNOCKOUT_UPSET), n=n, seed=12345)

    rows = []
    group_of = {t: g for g, members in cfg["groups"].items() for t in members}
    for team, probs in raw["teams"].items():
        rows.append({"team": team, "group": group_of.get(team), **probs})
    rows.sort(key=lambda r: (-r.get("champion", 0), -r.get("final", 0)))

    settled = [
        {"teams": sorted(pair), "winner": w} for pair, w in fixed.items()
    ]
    # Only surface performance nudges for teams in the bracket, still alive (not
    # knocked out by a settled result), that actually moved. Drops group-stage
    # non-qualifiers, who played but can never appear in the odds list.
    in_bracket = {r["team"] for r in rows}
    eliminated = {t for pair, w in fixed.items() for t in pair if t != w}
    perf = [
        r for r in perf_rows
        if r["team"] in in_bracket and r["team"] not in eliminated
        and (abs(r["att_delta"]) + abs(r["def_delta"]) >= 0.01)
    ]
    result = {
        "competition": competition,
        "name": cfg["name"],
        "as_of": as_of,
        "mode": mode,
        "n": n,
        "stages": raw["stages"],
        "teams": rows,
        "settled_knockout": settled,
        "performance": perf,
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


def _advance_prob(
    home: str, away: str, competition: str = DEFAULT_COMPETITION,
    params: "Params | None" = None,
) -> float:
    """P(home advances) in a neutral knockout tie (normal time or shootout)."""
    p = params if params is not None else _params_for(competition)
    mat = dc.predict(p, home, away, neutral=True)
    ph, pdraw, pa = dc.outcome_probs(mat)
    denom = ph + pa
    share = ph / denom if denom > 1e-12 else 0.5
    return ph + pdraw * share


_BRACKET_ROUNDS = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]


def _bracket_view(
    cfg: dict, played: dict, mode: str, competition: str = DEFAULT_COMPETITION,
    params: "Params | None" = None,
) -> dict:
    """Build one bracket. mode='prediction' advances each tie's most likely winner
    (head-to-head); mode='actual' advances the real winners and leaves undecided
    ties as TBD, flagging whether each result matched the pick. Both use the same
    head-to-head favourite, so the two views never disagree about a pick.
    Pass params to override the default injury-adjusted ratings (e.g. base params
    for a pre-tournament view)."""
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
            # The model's single most likely winner of this tie (head-to-head, in
            # a neutral knockout), advanced round by round. This is the same rule
            # the "actual" view grades results against, so the Prediction and
            # Results views can never disagree about who was picked.
            pa = _advance_prob(a, b, competition, params=params)
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
                pa = _advance_prob(a, b, competition, params=params)
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

    # Third place match: losers of the two semifinals.
    sf_matches = ko["rounds"]["semifinal"]
    sf_losers = []
    for sm in sf_matches:
        e = built[sm]
        w = winners.get(sm)
        if w and e["a"] and e["b"]:
            sf_losers.append(e["b"] if w == e["a"] else e["a"])
        else:
            sf_losers.append(None)
    a3, b3 = (sf_losers + [None, None])[:2]
    if mode == "prediction" and a3 and b3:
        pa3 = _advance_prob(a3, b3, competition, params=params)
        w3 = a3 if pa3 >= 0.5 else b3
        out["third_place"] = {"a": a3, "b": b3, "prob_a": pa3, "prob_b": 1.0 - pa3,
                              "winner": w3, "settled": False}
    elif mode == "actual":
        rec3 = played.get(frozenset((a3, b3))) if (a3 and b3) else None
        if rec3:
            w3 = rec3["winner"]
            score3 = ([rec3["home_goals"], rec3["away_goals"]]
                      if rec3["home"] == a3 else [rec3["away_goals"], rec3["home_goals"]])
            pa3 = _advance_prob(a3, b3, competition, params=params)
            pred3 = a3 if pa3 >= 0.5 else b3
            out["third_place"] = {"a": a3, "b": b3, "winner": w3, "score": score3,
                                  "settled": True, "predicted_winner": pred3,
                                  "correct": w3 == pred3}
        else:
            out["third_place"] = {"a": a3, "b": b3, "winner": None, "settled": False}

    return out


def bracket(competition: str, as_of: str | None = None) -> dict:
    """Three bracket views: pre-tournament predictions (base ratings, no injury
    overlay), live predictions (injury-adjusted ratings), and actual results
    with hit/miss flags. All three use the same head-to-head rule, so the
    win-chance numbers in each prediction view are consistent with the matching
    title-odds simulation mode."""
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
        "pretournament_prediction": _bracket_view(
            cfg, {}, "prediction", competition=competition, params=_params
        ),
        "prediction": _bracket_view(cfg, {}, "prediction", competition=competition),
        "actual": _bracket_view(cfg, played, "actual", competition=competition),
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
        pred = predict_match(fx["home"], fx["away"], neutral=fx["neutral"], competition=competition)
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
