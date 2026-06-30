"""ChampionsLeagueFormat — DOCUMENTED STUB (intended third implementation).

The hardest format, hence last. It reuses the same scoreline model and Monte
Carlo driver; the new logic is all in how ties are resolved.

Format (2024+ revamp)
---------------------
A single 36-team **league phase**: each team plays 8 different opponents once;
one combined table. Top 8 advance straight to the Round of 16; teams 9-24 contest
a two-legged knockout playoff for the remaining 8 spots; 25-36 are eliminated.
Then a two-legged knockout bracket (R16 -> QF -> SF) and a single-leg final.

What makes it hard (TODO)
-------------------------
- **Two-legged aggregate ties**: simulate both legs, aggregate the scores, and —
  on a level aggregate — apply extra time in the second leg and then a penalty
  shootout. (Away-goals was abolished in 2021, so no away-goals rule.) This needs
  a two-leg-aware tie resolver rather than the single-match `knockout_winner`.
- **Extra time + penalties** modelled explicitly for the final and level second
  legs (`MatchSampler` already has the penalty-style coin flip to build on).
- **League-phase scheduling**: 8 opponents per team drawn by pot/seeding; can be
  taken from the published fixtures rather than simulated.
- `stages()`: ["league_phase", "knockout_playoff", "round_of_16",
  "quarterfinal", "semifinal", "final", "champion"] — again just labels, so the
  driver and frontend are unchanged.

Only this file (plus a two-leg tie helper) is new; nothing upstream changes.
"""

from __future__ import annotations

from forecaster.formats.base import MatchSampler


class ChampionsLeagueFormat:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ChampionsLeagueFormat is a documented stub (the intended third "
            "implementation). See the module docstring for the design and TODO."
        )

    def stages(self) -> list[str]:
        raise NotImplementedError

    def teams(self) -> list[str]:
        raise NotImplementedError

    def simulate_once(self, sampler: MatchSampler, rng) -> dict[str, str]:
        raise NotImplementedError
