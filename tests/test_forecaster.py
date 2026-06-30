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
