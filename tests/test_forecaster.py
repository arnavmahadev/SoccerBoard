"""Forecaster sanity tests — offline (use the committed snapshot, no network).

Covers the data layer, the Dixon-Coles model, the competition config, the
Monte Carlo simulator and the FastAPI routes.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecaster import dixon_coles as dc
from forecaster.data import (
    SNAPSHOT_PATH,
    get_competition,
    load_matches,
    normalize_team,
)
from forecaster.formats.base import MatchSampler, monte_carlo
from forecaster.formats.world_cup import WorldCupFormat


# Use the committed snapshot directly so tests never hit the network.
def _snapshot_matches(**kw):
    return load_matches(path=SNAPSHOT_PATH, **kw)


@pytest.fixture(scope="module")
def params():
    matches = _snapshot_matches(as_of="2026-06-10", since="2016-01-01")
    return dc.fit(matches, xi=0.4, reg=0.05)


@pytest.fixture(scope="module")
def config():
    return get_competition("world_cup_2026")


# --- data layer --------------------------------------------------------------
def test_team_normalization():
    assert normalize_team("West Germany") == "Germany"
    assert normalize_team("Türkiye") == "Turkey"
    assert normalize_team("England") == "England"


def test_as_of_clamps_future_matches():
    before = _snapshot_matches(as_of="2026-06-10", since="2024-01-01")
    after = _snapshot_matches(as_of="2026-07-01", since="2024-01-01")
    assert all(m.date <= "2026-06-10" for m in before)
    assert len(after) > len(before)  # the World Cup falls between the two cutoffs


# --- Dixon-Coles -------------------------------------------------------------
def test_home_advantage_positive(params):
    assert params.home_adv > 0


def test_goal_matrix_is_a_distribution(params):
    mat = dc.predict(params, "Brazil", "Argentina", neutral=True)
    assert mat.shape[0] == mat.shape[1]
    assert mat.min() >= 0
    assert abs(mat.sum() - 1.0) < 1e-9
    h, d, a = dc.outcome_probs(mat)
    assert abs((h + d + a) - 1.0) < 1e-9


def test_strong_beats_weak(params):
    strong = dc.outcome_probs(dc.predict(params, "Brazil", "Haiti"))[0]
    even = dc.outcome_probs(dc.predict(params, "Brazil", "Argentina"))[0]
    assert strong > even  # Brazil beats Haiti more often than it beats Argentina


def test_home_advantage_shifts_probabilities(params):
    home = dc.outcome_probs(dc.predict(params, "Mexico", "Japan", neutral=False))[0]
    neut = dc.outcome_probs(dc.predict(params, "Mexico", "Japan", neutral=True))[0]
    assert home > neut


# --- competition config ------------------------------------------------------
def test_config_shape(config):
    assert len(config["groups"]) == 12
    assert all(len(v) == 4 for v in config["groups"].values())
    r32 = config["knockout"]["round_of_32"]
    teams = {t for f in r32 for t in f["teams"]}
    assert len(r32) == 16 and len(teams) == 32


def test_bracket_tree_is_consistent(config):
    tree = config["knockout"]["tree"]
    feeders = [m for pair in tree.values() for m in pair]
    # every R32 match (73-88) and every internal node feeds exactly one parent
    assert sorted(feeders) == sorted(set(feeders))
    assert set(range(73, 89)).issubset(set(feeders))


def test_bracket_rounds_in_display_order(config):
    """Each round's match list is the order the bracket is *drawn* in, so the
    rendered columns and connector lines line up with the tree: consecutive
    pairs in a column feed the matches of the next column in order. Numeric
    order would fail this (the R32->R16 pairing is irregular)."""
    ko = config["knockout"]
    tree = {int(k): v for k, v in ko["tree"].items()}
    order = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
    for cur_r, nxt_r in zip(order, order[1:]):
        cur, nxt = ko["rounds"][cur_r], ko["rounds"][nxt_r]
        assert len(cur) == 2 * len(nxt)
        for i, parent in enumerate(nxt):
            assert set(cur[2 * i : 2 * i + 2]) == set(tree[parent])


# --- simulator ---------------------------------------------------------------
def test_group_forecast_advances_32(params, config):
    sampler = MatchSampler(params)
    fc = WorldCupFormat(config).group_forecast(sampler, n=400, seed=1)
    total = sum(v["advance"] for v in fc.values())
    assert 31.0 <= total <= 33.0  # 12*2 + 8 best thirds, within MC noise
    assert all(0 <= v["advance"] <= 1 for v in fc.values())


def test_title_probabilities_sum_to_one(params, config):
    sampler = MatchSampler(params)
    res = monte_carlo(WorldCupFormat(config), sampler, n=600, seed=2)
    champ = sum(t["champion"] for t in res["teams"].values())
    assert abs(champ - 1.0) < 1e-9
    assert all(t["advance"] == 1.0 for t in res["teams"].values())  # all 32 qualified


def test_fixed_knockout_result_is_respected(params, config):
    # Force a fixed R32 winner and check it always reaches the next round.
    a, b = config["knockout"]["round_of_32"][0]["teams"]
    fmt = WorldCupFormat(config, fixed_results={frozenset((a, b)): a})
    res = monte_carlo(fmt, MatchSampler(params), n=300, seed=3)
    assert res["teams"][a]["round_of_16"] == 1.0
    assert res["teams"][b]["round_of_16"] == 0.0


# --- API ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from xg.serve.app import app

    with TestClient(app) as c:
        yield c


def test_api_competitions(client):
    comps = client.get("/forecaster/competitions").json()
    assert any(c["id"] == "world_cup_2026" for c in comps)


def test_api_match(client):
    r = client.post("/forecaster/match", json={"home": "Spain", "away": "Japan"}).json()
    assert abs(r["prob_home"] + r["prob_draw"] + r["prob_away"] - 1.0) < 1e-6
    assert len(r["matrix"]) == r["matrix_max"] + 1


def test_api_simulation_and_metrics(client):
    sim = client.get("/forecaster/simulation?n=1000").json()
    assert sim["competition"] == "world_cup_2026"
    assert len(sim["teams"]) == 32
    mt = client.get("/forecaster/metrics").json()
    assert mt["model"]["log_loss"] < mt["baseline"]["log_loss"]  # beats the baseline


def test_player_model_normalize_and_roundtrip(tmp_path):
    """normalize_player_name strips accents for fuzzy lookup, and PlayerDeltas
    round-trips through JSON without data loss."""
    from forecaster.player_model import PlayerDeltas, normalize_player_name

    assert normalize_player_name("Kylian Mbappé") == "kylian mbappe"
    assert normalize_player_name("Nico Williams") == "nico williams"
    assert normalize_player_name("Pedri") == "pedri"

    deltas = PlayerDeltas(
        players=["Kylian Mbappé", "Pedri"],
        att_delta=[0.15, 0.08],
        def_delta=[0.02, 0.04],
        n_matches=[22, 18],
        reg=1.0,
        n_lineup_matches=120,
    )
    path = tmp_path / "player_deltas.json"
    deltas.to_json(path)
    loaded = PlayerDeltas.from_json(path)
    assert loaded.players == deltas.players
    assert loaded.att_delta == pytest.approx(deltas.att_delta)
    assert loaded.def_delta == pytest.approx(deltas.def_delta)
    # Accent-stripped lookup finds the accented name
    assert normalize_player_name("Kylian Mbappé") in loaded.normalized_index()


def test_api_adjustments_overlay(client):
    """The adjustments endpoint surfaces per-player news items for every team
    listed in news_items.json, with att_delta / def_delta / covered fields."""
    adj = client.get("/forecaster/adjustments").json()
    by_team = {t["team"]: t for t in adj["teams"]}
    assert len(by_team) >= 2, "overlay should cover multiple teams' news"
    for t in adj["teams"]:
        assert t["items"] and all(
            it["player"] and it["issue"] and "att_delta" in it and "covered" in it
            for it in t["items"]
        )


def test_player_delta_lowers_scoring():
    """When a player with a positive att_delta is absent, their team's expected
    goals drop by the learned amount. Tests the serving-path mechanism directly
    by injecting a synthetic PlayerDeltas object."""
    from forecaster import predictor as fc
    from forecaster.player_model import PlayerDeltas

    fc.load()
    synthetic = PlayerDeltas(
        players=["Nico Williams"],
        att_delta=[0.20],
        def_delta=[0.00],
        n_matches=[20],
        reg=1.0,
        n_lineup_matches=300,
    )
    old_deltas = fc._player_deltas
    old_adj = dict(fc._adj_params)
    fc._player_deltas = synthetic
    fc._adj_params.clear()

    try:
        base = fc.predict_match("Spain", "Morocco", competition="__no_overlay__")
        adj = fc.predict_match("Spain", "Morocco", competition="world_cup_2026")
        assert adj["exp_home"] < base["exp_home"]   # Nico out -> Spain scores less
        assert adj["exp_away"] == pytest.approx(base["exp_away"])  # opponent unchanged
    finally:
        fc._player_deltas = old_deltas
        fc._adj_params.clear()
        fc._adj_params.update(old_adj)


def test_performance_overlay_direction():
    """The in-tournament overlay nudges a team the right way: a side that concedes
    far more than expected loses defensive rating; one that scores far more than
    expected gains attack. Teams that play to expectation barely move."""
    from forecaster import predictor as fc
    from forecaster.data import Match

    fc.load()
    base = fc._params_for("__no_overlay__")  # frozen params, no news
    idx = base.index()
    strong, weak = "Argentina", "Cape Verde"
    assert strong in idx and weak in idx

    lam, mu = dc.goal_expectations(base, strong, weak, neutral=True)
    # A shock scoreline: the favourite is held and leaks goals it never should.
    shock = Match(date="2026-07-01", home=strong, away=weak,
                  home_goals=1, away_goals=3, neutral=True, tournament="FIFA World Cup")
    adj, rows = fc._performance_overlay(base, [shock])
    by_team = {r["team"]: r for r in rows}

    # Favourite conceded 3 vs ~0.4 expected -> defense rating falls.
    assert by_team[strong]["def_delta"] < 0
    assert adj.defense[idx[strong]] < base.defense[idx[strong]]
    # Underdog scored 3 vs a fraction expected -> attack rating rises.
    assert by_team[weak]["att_delta"] > 0
    assert adj.attack[idx[weak]] > base.attack[idx[weak]]

    # A result exactly at expectation moves nothing.
    par = Match(date="2026-07-01", home=strong, away=weak,
                home_goals=round(lam), away_goals=round(mu), neutral=True,
                tournament="FIFA World Cup")
    _, rows2 = fc._performance_overlay(base, [par])
    for r in rows2:
        if r["team"] in (strong, weak):
            assert abs(r["att_delta"]) < 0.1 and abs(r["def_delta"]) < 0.1


def test_knockout_upset_regresses_toward_coinflip(params):
    """The upset knob pulls a favourite's advance probability toward 50/50 without
    flipping the favourite, and upset=0 leaves the strength model untouched."""
    strong, weak = "Spain", "Austria"
    plain = MatchSampler(params, upset=0.0)
    tempered = MatchSampler(params, upset=0.5)
    rng = np.random.default_rng(0)
    plain.knockout_winner(strong, weak, True, rng)
    tempered.knockout_winner(strong, weak, True, rng)
    p0 = plain._advance[(strong, weak, True)]
    p1 = tempered._advance[(strong, weak, True)]
    assert p0 > 0.5                       # favourite really is favoured
    assert 0.5 < p1 < p0                  # regressed toward the coin flip, still favoured
    assert p1 == pytest.approx(0.5 + (p0 - 0.5) * 0.5)


def test_api_simulation_exposes_performance(client):
    """Live simulation surfaces a `performance` breakdown; pretournament doesn't."""
    live = client.get("/forecaster/simulation?mode=live&n=1000").json()
    assert "performance" in live and isinstance(live["performance"], list)
    for r in live["performance"]:
        assert "att_delta" in r and "def_delta" in r and r["team"]
    pre = client.get("/forecaster/simulation?mode=pretournament&n=1000").json()
    assert pre["performance"] == []


def test_bracket_views_never_contradict(client):
    """The live Prediction view locks in games already played to their real result,
    so an eliminated team never advances there and it agrees with the Results view
    about who went through (a settled tie can't show one winner in one view and the
    other winner in the other — the Mexico bug). Ties still to come are shown with
    the advancing side at the higher probability."""
    bk = client.get("/forecaster/bracket").json()
    actual_winner = {m["match"]: m["winner"]
                     for rnd in bk["actual"]["rounds"]
                     for m in rnd["matches"] if m.get("settled")}
    for rnd in bk["prediction"]["rounds"]:
        for m in rnd["matches"]:
            if m.get("settled"):
                # A played tie in the live bracket shows the REAL winner and score,
                # never the pre-game favourite, so an eliminated side cannot advance.
                assert m["winner"] == actual_winner[m["match"]]
                assert "score" in m and "prob_a" not in m
            else:
                # An unplayed tie advances the higher-probability side.
                wp = m["prob_a"] if m["winner"] == m["a"] else m["prob_b"]
                assert wp >= 0.5 - 1e-9
