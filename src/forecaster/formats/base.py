"""The `CompetitionFormat` interface + the generic Monte Carlo driver.

This is the core architectural seam. A `CompetitionFormat` knows how to simulate
ONE full instance of a competition and report which stage each team reached. The
`monte_carlo` driver knows nothing about groups, brackets or tables — it just
calls `simulate_once` N times and counts outcomes. That separation is what lets
World Cup, league and Champions League share one simulator, one API shape and one
frontend rendering path.

`MatchSampler` is the bridge to the scoreline model: it turns fitted Dixon-Coles
params into sampled scorelines / knockout winners, caching per-fixture
distributions so 10k simulations run fast.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from forecaster import dixon_coles as dc
from forecaster.dixon_coles import Params


class MatchSampler:
    """Samples match outcomes from the fitted scoreline model.

    Caches each fixture's flattened goal-matrix CDF (and knockout advance
    probability) keyed by (home, away, neutral), because the same fixtures recur
    across thousands of simulations.
    """

    def __init__(self, params: Params, max_goals: int = 10, upset: float = 0.0):
        self.params = params
        self.max_goals = max_goals
        # Knockout single-game variance: each tie's advance probability is regressed
        # this fraction toward a coin flip (0 = pure strength model, 1 = 50/50). A
        # knockout is one match — far noisier than a team's season-long rating — so
        # a little regression stops the simulator over-concentrating the title on
        # the top seed and better matches how often favourites actually go out.
        self.upset = upset
        self._cdf: dict[tuple, np.ndarray] = {}
        self._ncols: dict[tuple, int] = {}
        self._advance: dict[tuple, float] = {}

    def _matrix_cdf(self, key):
        if key not in self._cdf:
            home, away, neutral = key
            mat = dc.predict(self.params, home, away, neutral, self.max_goals)
            self._cdf[key] = np.cumsum(mat.reshape(-1))
            self._ncols[key] = mat.shape[1]
        return self._cdf[key], self._ncols[key]

    def sample_score(self, home, away, neutral, rng) -> tuple[int, int]:
        cdf, ncols = self._matrix_cdf((home, away, neutral))
        idx = int(np.searchsorted(cdf, rng.random() * cdf[-1]))
        return divmod(idx, ncols)

    def knockout_winner(self, home, away, neutral, rng) -> str:
        """Winner of a must-have-a-winner match. A draw in normal time is resolved
        by a win-probability-weighted coin flip — modelling extra time + penalties
        as favouring the stronger side but much closer to 50/50 than a full match.
        Concretely P(home advances) = P(home win) + P(draw)*P(home win)/(P(home win)+P(away win))."""
        key = (home, away, neutral)
        if key not in self._advance:
            mat = dc.predict(self.params, home, away, neutral, self.max_goals)
            ph, pd, pa = dc.outcome_probs(mat)
            denom = ph + pa
            tie_share = ph / denom if denom > 1e-12 else 0.5
            p = ph + pd * tie_share
            self._advance[key] = 0.5 + (p - 0.5) * (1.0 - self.upset)
        return home if rng.random() < self._advance[key] else away


class CompetitionFormat(Protocol):
    """A competition format the Monte Carlo driver can simulate.

    Implementations: `WorldCupFormat` (full); `LeagueFormat`,
    `ChampionsLeagueFormat` (stubs).
    """

    def stages(self) -> list[str]:
        """Ordered stage labels, earliest first (e.g. advance -> ... -> champion)."""
        ...

    def teams(self) -> list[str]:
        """Teams eligible to appear in the reported standings."""
        ...

    def simulate_once(self, sampler: MatchSampler, rng: np.random.Generator) -> dict[str, str]:
        """Simulate one full instance; return {team: furthest stage reached}."""
        ...


def monte_carlo(
    fmt: CompetitionFormat,
    sampler: MatchSampler,
    n: int = 10000,
    seed: int = 0,
) -> dict:
    """Run `fmt.simulate_once` N times and aggregate per-team stage probabilities.

    Reaching a later stage implies reaching every earlier one, so P(reach stage s)
    is the fraction of simulations whose furthest stage is >= s.
    """
    stages = fmt.stages()
    stage_idx = {s: i for i, s in enumerate(stages)}
    teams = fmt.teams()
    counts = {t: np.zeros(len(stages)) for t in teams}

    rng = np.random.default_rng(seed)
    for _ in range(n):
        reached = fmt.simulate_once(sampler, rng)
        for team, furthest in reached.items():
            # increment every stage up to and including the furthest reached
            for k in range(stage_idx[furthest] + 1):
                counts[team][k] += 1.0

    out = {}
    for t in teams:
        out[t] = {s: float(counts[t][i] / n) for i, s in enumerate(stages)}
    return {"n": n, "stages": stages, "teams": out}
