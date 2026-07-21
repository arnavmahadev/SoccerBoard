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
import logging
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


def reload_news() -> dict:
    """Re-read news_items.json into memory (e.g. after scripts/refresh_news.py
    rewrote it) so a running server picks up the new injuries without a restart.
    Also drops the player-adjusted params and the simulation cache, since both
    are derived from the news overlay and would otherwise stay stale."""
    global _news
    load()
    with _lock:
        _news = (
            json.loads(NEWS_ITEMS_PATH.read_text()) if NEWS_ITEMS_PATH.exists() else {}
        )
        _adj_params.clear()
        _sim_cache.clear()
    return _news


_news_log = logging.getLogger("forecaster.news")
_news_autorefresh_started = False
_NEWS_REFRESH_N = 10000   # Monte-Carlo sims for the top-N ranking in a refresh


def refresh_news_overlay(
    competition: str = DEFAULT_COMPETITION,
    *,
    top: int = 10,
    n: int = _NEWS_REFRESH_N,
    max_age_days: int | None = None,
) -> dict:
    """Scrape the live injury tracker and apply it to the in-memory overlay.

    Runs the same pipeline as scripts/refresh_news.py (rank the top-N by live
    title odds, fetch + recency-filter their injuries), persists to
    news_items.json when the filesystem is writable (best effort), and always
    updates the in-memory overlay + drops the derived caches so a running server
    reflects it immediately. Raises news_fetch.ApiError on a fetch/parse failure
    (callers that want fail-open behaviour should catch it)."""
    from forecaster import news_fetch as nf

    global _news
    load()
    kwargs = {} if max_age_days is None else {"max_age_days": max_age_days}
    teams = nf.top_title_odds_teams(competition, top, n)
    records = nf.fetch_injuries()
    teams_news = nf.build_teams_news(records, teams, **kwargs)

    # Persist to the committed file when we can; on a read-only filesystem (e.g.
    # Hugging Face Spaces) fall back to the in-memory update below.
    try:
        nf.write_news(competition, teams_news)
    except OSError as e:
        _news_log.info("news_items.json not writable (%s); updating in memory only", e)

    with _lock:
        doc = dict(_news or {})
        doc.setdefault("_about", nf.DEFAULT_ABOUT)
        doc[competition] = {"updated": date.today().isoformat(), "teams": teams_news}
        _news = doc
        _adj_params.pop(competition, None)  # force rebuild of injury-adjusted params
        _sim_cache.clear()
    return {
        "competition": competition,
        "top_teams": teams,
        "records_fetched": len(records),
        "teams_with_news_count": len(teams_news),
        "injuries_kept": sum(len(v) for v in teams_news.values()),
    }


def start_news_autorefresh(
    interval_seconds: float = 24 * 3600.0,
    *,
    competition: str = DEFAULT_COMPETITION,
    delay_first: float = 8.0,
) -> None:
    """Start a daemon thread that refreshes the injury overlay on an interval
    (default daily) so the deployed app serves fresh injuries to everyone without
    a manual run or redeploy. Idempotent (a second call is a no-op). Fails open:
    a scrape/parse error is logged and the last good overlay is kept."""
    global _news_autorefresh_started
    with _lock:
        if _news_autorefresh_started:
            return
        _news_autorefresh_started = True

    def _loop() -> None:
        time.sleep(delay_first)  # let startup settle before the first scrape
        while True:
            try:
                summary = refresh_news_overlay(competition)
                _news_log.info(
                    "injury overlay refreshed: %d injuries across %d team(s) — %s",
                    summary["injuries_kept"], summary["teams_with_news_count"],
                    ", ".join(summary["top_teams"]),
                )
            except Exception as e:  # never let the refresher kill the thread
                _news_log.warning("injury refresh failed (keeping last overlay): %s", e)
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, name="news-autorefresh", daemon=True).start()
    _news_log.info("news auto-refresh started (every %.0f h)", interval_seconds / 3600.0)


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
    competition: str = DEFAULT_COMPETITION, params: Params | None = None,
) -> dict:
    load()
    if params is None:
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
    # Surface performance nudges for every bracket team still alive (not knocked
    # out by a settled result). No magnitude gate: a team's form tag stays put as
    # long as it's alive, rather than flickering on and off as a live recompute
    # nudges a borderline team across a threshold. A team that has performed to
    # expectation reads as "barely moves" in the panel, so nothing misleads.
    # Drops group-stage non-qualifiers, who played but can't appear in the odds list.
    in_bracket = {r["team"] for r in rows}
    eliminated = {t for pair, w in fixed.items() for t in pair if t != w}
    perf = [
        r for r in perf_rows
        if r["team"] in in_bracket and r["team"] not in eliminated
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


# --- Title-odds timeline (retrospective, tournament complete) -----------------
# How the champion odds moved as the tournament actually unfolded. At each
# checkpoint we take the frozen pre-tournament ratings, bend them by how each side
# had *played up to that point* (the same in-tournament form overlay the live sim
# used), lock in every knockout tie decided so far, and simulate the rest. So the
# early checkpoints are near the pre-tournament odds and each knockout round
# collapses the field further, until the final leaves the real champion at 100%.
# Injury news is intentionally NOT applied: that overlay is a live "who's fit now"
# adjustment, and replaying it against past checkpoints would be anachronistic —
# the timeline is a pure results-driven replay. Committed as an artifact
# (title_odds_timeline.json) because it's static now the tournament is over.

TIMELINE_PATH = ARTIFACTS / "title_odds_timeline.json"
_KO_ROUND_ORDER = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]

# (key, label, group-form cutoff [None=none, "md"=matchday date, "all"=whole group
#  stage], completed knockout rounds fed into the fixed bracket + form overlay)
_TIMELINE_SPEC = [
    ("pre",   "Pre-tournament",       None, []),
    ("md1",   "After Matchday 1",     0,    []),
    ("md2",   "After Matchday 2",     1,    []),
    ("group", "After Group Stage",    "all", []),
    ("r32",   "After Round of 32",    "all", ["round_of_32"]),
    ("r16",   "After Round of 16",    "all", ["round_of_32", "round_of_16"]),
    ("qf",    "After Quarter-finals", "all", ["round_of_32", "round_of_16", "quarterfinal"]),
    ("sf",    "After Semi-finals",    "all", ["round_of_32", "round_of_16", "quarterfinal", "semifinal"]),
    ("final", "Champion",             "all", _KO_ROUND_ORDER),
]

_timeline_cache: dict[tuple, tuple[float, dict]] = {}
_TIMELINE_TTL = _REALITY_TTL  # static once built; long TTL for the compute fallback

# Which checkpoint's ratings a knockout round's ✓/✗ is graded against: the stage
# right before that round was played, i.e. the exact ratings the run-through used
# to *predict* that round. Grading a result against the prediction that was
# standing when it kicked off means the ✓/✗ can never contradict the winner the
# bracket had projected (e.g. the final is judged on the semi-final-stage ratings
# that named Spain, so Spain lifting the trophy reads as a correct call, not a miss).
_GRADE_STAGE = {
    "round_of_32": "group",
    "round_of_16": "r32",
    "quarterfinal": "r16",
    "semifinal": "qf",
    "final": "sf",
}


def _resolve_knockout_rounds(cfg: dict, played_ko: dict) -> dict[str, list[tuple]]:
    """Walk the bracket so each played knockout tie is tagged with the round it
    belongs to: {round_name: [(frozenset(pair), record), ...]} in bracket order."""
    ko = cfg["knockout"]
    r32 = {f["match"]: tuple(f["teams"]) for f in ko["round_of_32"]}
    tree = {int(k): v for k, v in ko["tree"].items()}
    match_round = {m: r for r, ms in ko["rounds"].items() for m in ms}
    order = sorted(set(r32) | set(tree))
    winners: dict[int, str] = {}
    by_round: dict[str, list[tuple]] = {r: [] for r in _KO_ROUND_ORDER}
    for m in order:
        a, b = r32[m] if m in r32 else (winners.get(tree[m][0]), winners.get(tree[m][1]))
        if not a or not b:
            continue
        rec = played_ko.get(frozenset((a, b)))
        if not rec:
            continue
        winners[m] = rec["winner"]
        by_round[match_round[m]].append((frozenset((a, b)), rec))
    return by_round


def _matchday_cutoffs(cfg: dict, feed_matches: list[Match]) -> list[str]:
    """ISO dates by which group matchday 1 and matchday 2 are complete, derived from
    the fixture calendar (each of the 3 matchdays is a third of the group games)."""
    group_pairs = {frozenset((f["home"], f["away"])) for f in cfg["group_fixtures"]}
    dates = sorted(
        m.date for m in feed_matches
        if m.played and frozenset((m.home, m.away)) in group_pairs
    )
    end = cfg.get("group_stage_end", dates[-1] if dates else "")
    if len(dates) < 3:
        return [end, end]
    per = len(dates) // 3  # 24 of 72
    return [dates[per - 1], dates[2 * per - 1]]


def compute_timeline(cfg: dict, base: Params, matches: list[Match], n: int = 10000) -> dict:
    """Champion-odds snapshots at each stage of the tournament. Pure/offline: given
    the config, the frozen ratings and the tournament's matches, it replays the
    title odds checkpoint by checkpoint. Used both to build the committed artifact
    and as the live fallback if that artifact is missing."""
    group_pairs = {frozenset((f["home"], f["away"])) for f in cfg["group_fixtures"]}
    played_group = [
        m for m in matches
        if m.played and frozenset((m.home, m.away)) in group_pairs
    ]
    played_ko = WorldCupFormat.knockout_played(cfg, matches)
    by_round = _resolve_knockout_rounds(cfg, played_ko)
    # The third-place playoff isn't a node in the knockout tree (its winner feeds
    # nothing), so it's the one played tie not tagged to a round. Pull it out so it
    # can be locked into the bracket at the final stage, when it's actually played.
    _bracket_pairs = {pair for r in _KO_ROUND_ORDER for pair, _ in by_round[r]}
    third_rec = next((rec for pair, rec in played_ko.items() if pair not in _bracket_pairs), None)
    md_cuts = _matchday_cutoffs(cfg, matches)
    tlabel = cfg["tournament_label"]
    qualifiers = {t for f in cfg["knockout"]["round_of_32"] for t in f["teams"]}
    group_of = {t: g for g, members in cfg["groups"].items() for t in members}

    def ko_as_matches(round_name: str) -> list[Match]:
        return [
            Match(date=cfg.get("group_stage_end", ""), home=rec["home"], away=rec["away"],
                  home_goals=rec["home_goals"], away_goals=rec["away_goals"],
                  neutral=True, tournament=tlabel)
            for _pair, rec in by_round[round_name]
        ]

    checkpoints = []
    params_by_key: dict[str, Params] = {}
    for key, label, form_cut, ko_rounds in _TIMELINE_SPEC:
        # Games feeding the in-tournament form overlay for this checkpoint.
        played: list[Match] = []
        if form_cut == "all":
            played += played_group
        elif form_cut is not None:  # matchday index into md_cuts
            cutoff = md_cuts[form_cut]
            played += [m for m in played_group if m.date <= cutoff]
        for r in ko_rounds:
            played += ko_as_matches(r)

        params = base if not played else _performance_overlay(base, played)[0]
        params_by_key[key] = params

        fixed = {pair: rec["winner"] for r in ko_rounds for pair, rec in by_round[r]}
        fmt = WorldCupFormat(cfg, fixed_results=fixed)
        raw = monte_carlo(fmt, MatchSampler(params, upset=KNOCKOUT_UPSET), n=n, seed=12345)

        rows = [{"team": t, "group": group_of.get(t), **probs}
                for t, probs in raw["teams"].items()]
        rows.sort(key=lambda r: (-r.get("champion", 0), -r.get("final", 0)))

        eliminated = {t for pair, w in fixed.items() for t in pair if t != w}
        alive = sorted(qualifiers - eliminated)

        # Games decided in the most recent knockout round (for the "what happened"
        # strip). Group matchdays decide 24 games at once, too many to list, so the
        # strip is knockout-only; group checkpoints lean on the caption instead.
        newest = ko_rounds[-1] if ko_rounds else None
        decided = [
            {"home": rec["home"], "away": rec["away"],
             "home_goals": rec["home_goals"], "away_goals": rec["away_goals"],
             "winner": rec["winner"], "note": _result_note(rec)}
            for _pair, rec in (by_round[newest] if newest else [])
        ]

        # Per-stage bracket: the same run-through as the odds. Every knockout tie
        # decided by this stage is locked to its real result; the rest is predicted
        # on the stage's ratings. So it starts as the pure pre-tournament bracket and
        # ends as the finished result, and prediction and result always line up on the
        # games actually played. Locked ties are graded (✓/✗) against the frozen
        # pre-tournament ratings, so the grade reflects the original prediction.
        played_upto = {pair: rec for r in ko_rounds for pair, rec in by_round[r]}
        # The third-place game is decided alongside the final, so lock it in only
        # once the final round is in play; earlier stages still predict it.
        if "final" in ko_rounds and third_rec is not None:
            played_upto[frozenset((third_rec["home"], third_rec["away"]))] = third_rec
        # Grade each locked round against the ratings that predicted it (the stage
        # before it was played), so a result never contradicts the projection the
        # bracket showed for that tie the stage before. Sources are all earlier
        # checkpoints, so they're already in params_by_key by the time we need them.
        grade_by_round = {
            rnd: params_by_key[src] for rnd, src in _GRADE_STAGE.items() if src in params_by_key
        }
        bracket = _bracket_view(
            cfg, played_upto, "prediction", cfg["id"],
            params=params, grade_params_by_round=grade_by_round,
        )

        checkpoints.append({
            "key": key, "label": label,
            "caption": _timeline_caption(key, label, len(alive), rows),
            "alive": len(alive), "teams": rows, "decided": decided, "bracket": bracket,
        })

    champion = checkpoints[-1]["teams"][0]["team"] if checkpoints[-1]["teams"] else None
    return {
        "competition": cfg["id"], "name": cfg["name"], "n": n,
        "champion": champion, "stages": _KO_ROUND_ORDER, "checkpoints": checkpoints,
    }


def _timeline_caption(key: str, label: str, alive: int, rows: list[dict]) -> str:
    """One-line plain-English read of a checkpoint for the stage caption."""
    lead = rows[0]["team"] if rows else "the field"
    if key == "pre":
        return f"Before a ball is kicked. The pre-tournament model makes {lead} the favourite."
    if key in ("md1", "md2"):
        n = 1 if key == "md1" else 2
        return (f"Group matchday {n} complete. The ratings pick up each side's early form, "
                f"but the field is still wide open.")
    if key == "group":
        return "Group stage done. The 32 qualifiers head into the Round of 32."
    if key == "final":
        return f"{lead} are champions of the world."
    return f"{label.replace('After ', '')} complete. {alive} teams still standing."


def timeline(competition: str = DEFAULT_COMPETITION, n: int = 10000) -> dict:
    """Champion-odds timeline for the frontend slider. Serves the committed
    artifact when present (fast, offline, deterministic); otherwise computes it
    live from the latest feed and caches it."""
    load()
    if TIMELINE_PATH.exists():
        try:
            return json.loads(TIMELINE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            pass  # fall through to a live compute
    key = (competition, n)
    cached = _timeline_cache.get(key)
    if cached and (time.time() - cached[0]) < _TIMELINE_TTL:
        return cached[1]
    cfg = get_competition(competition)
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    result = compute_timeline(cfg, _params, matches, n=n)
    _timeline_cache[key] = (time.time(), result)
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


def _result_note(rec: dict) -> str:
    """Short tag for how a settled knockout tie was decided ('pens'/'AET'), so the
    bracket can flag a shootout or extra-time result whose scoreline alone (e.g. a
    1-1 won on penalties) wouldn't otherwise show a winner."""
    if rec.get("penalty"):
        return "pens"
    if rec.get("aet"):
        return "AET"
    return ""


def _bracket_view(
    cfg: dict, played: dict, mode: str, competition: str = DEFAULT_COMPETITION,
    params: "Params | None" = None, grade_params: "Params | None" = None,
    grade_params_by_round: "dict[str, Params] | None" = None,
) -> dict:
    """Build one bracket. mode='prediction' locks in the games already played
    (from `played`) and advances the most likely winner (head-to-head) only for
    ties still to come, so a team the results show was knocked out never advances;
    pass played={} for a from-scratch prediction (the pre-tournament view). This is
    also the per-stage run-through: as more rounds land in `played`, more of the
    bracket switches from predicted to real, until at the end it is the full result.
    mode='actual' advances the real winners and leaves undecided ties as TBD,
    flagging whether each result matched the pick. Both grade against the same
    head-to-head favourite, so the two views never disagree about a pick.
    Pass params to override the default injury-adjusted ratings (e.g. base params
    for a pre-tournament view). A settled tie's ✓/✗ is judged against grade_params
    (defaults to params), or against grade_params_by_round[round] when given — pass
    the per-round pre-game ratings so each result is graded against the very
    prediction that was standing before it was played, and the ✓/✗ can never
    contradict the winner the bracket had projected for that tie."""
    ko = cfg["knockout"]
    r32 = {f["match"]: tuple(f["teams"]) for f in ko["round_of_32"]}
    tree = {int(k): v for k, v in ko["tree"].items()}
    match_round = {m: r for r, ms in ko["rounds"].items() for m in ms}
    order = sorted(set(r32) | set(tree))
    grade_params = grade_params if grade_params is not None else params

    def _grade_p(round_name: str) -> "Params | None":
        if grade_params_by_round and round_name in grade_params_by_round:
            return grade_params_by_round[round_name]
        return grade_params

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
            # Live prediction: lock in ties that have actually been played and only
            # predict the ones still to come, so a team the results show was
            # knocked out is never advanced here (its real conqueror carries on
            # instead). For an unplayed tie, advance the model's single most likely
            # winner (head-to-head, neutral knockout) — the same rule the "actual"
            # view grades against, so the two views never disagree about a pick.
            rec = played.get(frozenset((a, b))) if (a and b) else None
            if rec:
                winner = rec["winner"]
                score = (
                    [rec["home_goals"], rec["away_goals"]]
                    if rec["home"] == a else [rec["away_goals"], rec["home_goals"]]
                )
                pg = _advance_prob(a, b, competition, params=_grade_p(match_round[m]))
                pred = a if pg >= 0.5 else b
                decided += 1
                correct += int(winner == pred)
                entry.update(winner=winner, score=score, settled=True, note=_result_note(rec),
                             predicted_winner=pred, correct=winner == pred)
            else:
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
                entry.update(winner=winner, score=score, settled=True, note=_result_note(rec),
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
    out = {"champion": winners.get(ko["final_match"]), "rounds": rounds,
           "correct": correct, "decided": decided}

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
        rec3 = played.get(frozenset((a3, b3)))
        if rec3:
            w3 = rec3["winner"]
            score3 = ([rec3["home_goals"], rec3["away_goals"]]
                      if rec3["home"] == a3 else [rec3["away_goals"], rec3["home_goals"]])
            pg3 = _advance_prob(a3, b3, competition, params=_grade_p("final"))
            pred3 = a3 if pg3 >= 0.5 else b3
            out["third_place"] = {"a": a3, "b": b3, "winner": w3, "score": score3,
                                  "settled": True, "note": _result_note(rec3),
                                  "predicted_winner": pred3, "correct": w3 == pred3}
        else:
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
                                  "settled": True, "note": _result_note(rec3),
                                  "predicted_winner": pred3, "correct": w3 == pred3}
        else:
            out["third_place"] = {"a": a3, "b": b3, "winner": None, "settled": False}

    return out


def _live_ratings(competition: str, as_of: str) -> Params:
    """Base ratings + news overlay, then bent by in-tournament over/under-
    performance. This is the single set of live ratings every live forecaster
    surface shares (title odds, predicted bracket, head-to-head), so none of them
    can disagree about how strong a team is right now."""
    cfg = get_competition(competition)
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    base = _params_for(competition)
    params, _ = _performance_overlay(base, _played_tournament_matches(cfg, matches))
    return params


def live_match(
    home: str, away: str, neutral: bool = True,
    competition: str = DEFAULT_COMPETITION, as_of: str | None = None,
) -> dict:
    """Head-to-head forecast on the live ratings (base + news + in-tournament
    form), so it agrees with the title odds and the predicted bracket."""
    load()
    as_of = as_of or _today()
    return predict_match(home, away, neutral=neutral, competition=competition,
                         params=_live_ratings(competition, as_of))


def bracket(competition: str, as_of: str | None = None) -> dict:
    """Three bracket views: pre-tournament predictions (base ratings, no overlay,
    every tie predicted from scratch), live predictions (injury- and in-tournament-
    form-adjusted ratings, with games already played locked to their real result
    and only the remaining ties predicted), and actual results with hit/miss flags.
    The live views use the same ratings as the live title-odds simulation, so their
    win-chance numbers and picks are consistent with the odds; the pre-tournament
    view matches the pre-tournament odds mode."""
    load()
    cfg = get_competition(competition)
    as_of = as_of or _today()
    matches = tournament_matches(
        cfg["tournament_label"], cfg["season_year"], as_of=as_of,
        prefer_live=True, ttl_seconds=LIVE_TTL,
    )
    played = WorldCupFormat.knockout_played(cfg, matches)
    live = _live_ratings(competition, as_of)
    return {
        "competition": competition,
        "name": cfg["name"],
        "as_of": as_of,
        "settled_count": len(played),
        "pretournament_prediction": _bracket_view(
            cfg, {}, "prediction", competition=competition, params=_params
        ),
        "prediction": _bracket_view(cfg, played, "prediction", competition=competition, params=live),
        "actual": _bracket_view(cfg, played, "actual", competition=competition, params=live),
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
