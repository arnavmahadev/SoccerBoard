"""LeagueFormat — DOCUMENTED STUB (intended second implementation).

A domestic league is the simplest format and a genuinely different output, which
is why it's the planned next step after the World Cup. It is intentionally not
built in this pass.

Format
------
Double round-robin: every team plays every other twice (home and away). No
knockout. From a partial season (results so far + remaining fixtures), simulate
the remaining fixtures on top of the same fitted Dixon-Coles strengths and report
**final-table** probabilities rather than bracket progression.

What it needs (TODO)
--------------------
- A fixtures source: matches already played + the remaining schedule for the
  season being forecast (the `data` loader already returns the right schema; this
  is data wiring, not new model code).
- `simulate_once`: start from current points, simulate every remaining fixture
  with `MatchSampler.sample_score`, accumulate the table (points -> goal
  difference -> goals scored, matching the league's real tiebreakers), and return
  {team: final_position_label}.
- `stages()` returns position-based labels, e.g. ["champion", "top_4",
  "europa", "mid_table", "relegation"], so the SAME Monte Carlo driver and the
  SAME frontend rendering (which is driven by stage labels) work unchanged — only
  the labels differ from the World Cup's.

Crucially this reuses the scoreline model, the Monte Carlo driver, the API shape
and the frontend as-is. Only this file is new.
"""

from __future__ import annotations

from forecaster.formats.base import MatchSampler


class LeagueFormat:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LeagueFormat is a documented stub (the intended second implementation). "
            "See the module docstring for the design and TODO."
        )

    def stages(self) -> list[str]:
        raise NotImplementedError

    def teams(self) -> list[str]:
        raise NotImplementedError

    def simulate_once(self, sampler: MatchSampler, rng) -> dict[str, str]:
        raise NotImplementedError
