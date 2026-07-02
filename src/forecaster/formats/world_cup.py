"""WorldCupFormat — the fully implemented competition format (2026, 48 teams).

Format
------
12 groups of 4 -> top 2 of each group (24) + the 8 best third-placed teams
advance to a Round of 32 -> R16 -> QF -> SF -> Final.

Two questions, one format
-------------------------
- **Group-stage forecast** (`group_forecast`): simulate the 12 groups from the
  fitted strengths and report P(win group) / P(advance to R32) per team. This is
  the pre-tournament view of the group phase.
- **Knockout forecast** (`simulate_once`, driven by the generic Monte Carlo
  loop): seed the Round of 32 with the teams that *actually* qualified, fix the
  knockout games already played (`fixed_results`), and simulate the rest to give
  P(reach each round) / P(title). Per the design, the knockout forecast uses the
  pre-tournament strengths and the real qualifiers — group results decide *who*
  advanced, never a team's strength. As more knockout games are played, more of
  the bracket is fixed and the numbers update (the live behaviour).

Modelling choices (documented for interviewers)
-----------------------------------------------
- Group tiebreakers: points (3/1/0) -> goal difference -> goals scored. Remaining
  exact ties (head-to-head / fair-play / drawing of lots in the real rules) are
  broken by a seeded random key — rare and immaterial to the probabilities.
- Knockout draws: resolved by `MatchSampler.knockout_winner` (win-probability-
  weighted coin flip standing in for extra time + penalties).
- Knockout venue: treated as neutral. Venues are fixed by bracket slot, not team,
  and aren't known for unplayed rounds; we don't give hosts a knockout home edge.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from forecaster.data import Match, normalize_team
from forecaster.formats.base import MatchSampler


class WorldCupFormat:
    def __init__(self, config: dict, fixed_results: dict[frozenset, str] | None = None):
        self.config = config
        self.fixed = fixed_results or {}

        self.groups: dict[str, list[str]] = config["groups"]
        self.best_thirds: int = config["best_thirds"]

        ko = config["knockout"]
        self.r32: dict[int, tuple[str, str]] = {
            f["match"]: tuple(f["teams"]) for f in ko["round_of_32"]
        }
        self.tree: dict[int, list[int]] = {int(k): v for k, v in ko["tree"].items()}
        self._stages: list[str] = config["stages"]

        # match number -> stage a team reaches by winning it
        nxt = {
            "round_of_32": "round_of_16",
            "round_of_16": "quarterfinal",
            "quarterfinal": "semifinal",
            "semifinal": "final",
            "final": "champion",
        }
        self.winner_stage: dict[int, str] = {}
        for round_name, matches in ko["rounds"].items():
            for m in matches:
                self.winner_stage[m] = nxt[round_name]
        self.match_order = sorted(set(self.r32) | set(self.tree))

        self.qualifiers = [t for pair in self.r32.values() for t in pair]

        # group fixtures bucketed by group (each fixture's teams share a group)
        team_group = {t: g for g, members in self.groups.items() for t in members}
        self.group_fixtures: dict[str, list[dict]] = defaultdict(list)
        for fx in config["group_fixtures"]:
            self.group_fixtures[team_group[fx["home"]]].append(fx)

    # --- CompetitionFormat interface -----------------------------------------
    def stages(self) -> list[str]:
        return self._stages

    def teams(self) -> list[str]:
        return list(self.qualifiers)

    def simulate_once(self, sampler: MatchSampler, rng) -> dict[str, str]:
        """One knockout playthrough from the real R32; returns furthest stage/team."""
        furthest = {t: "advance" for t in self.qualifiers}
        winners: dict[int, str] = {}
        for m in self.match_order:
            if m in self.r32:
                a, b = self.r32[m]
            else:
                f1, f2 = self.tree[m]
                a, b = winners[f1], winners[f2]
            pair = frozenset((a, b))
            if pair in self.fixed:
                w = self.fixed[pair]
            else:
                w = sampler.knockout_winner(a, b, neutral=True, rng=rng)
            winners[m] = w
            furthest[w] = self.winner_stage[m]
        return furthest

    # --- Group-stage forecast (format-specific) ------------------------------
    def _simulate_group(self, letter: str, sampler: MatchSampler, rng):
        """Play a group's 6 fixtures; return teams ordered 1st..4th with records."""
        pts = defaultdict(int)
        gf = defaultdict(int)
        ga = defaultdict(int)
        for fx in self.group_fixtures[letter]:
            hg, ag = sampler.sample_score(fx["home"], fx["away"], fx["neutral"], rng)
            h, a = fx["home"], fx["away"]
            gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        teams = self.groups[letter]
        # sort by points, goal difference, goals scored, then a random tiebreak
        ranked = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng.random()),
            reverse=True,
        )
        records = {t: (pts[t], gf[t] - ga[t], gf[t]) for t in teams}
        return ranked, records

    def group_forecast(
        self, sampler: MatchSampler, n: int = 10000, seed: int = 0
    ) -> dict:
        """P(win group) / P(top 2) / P(advance to R32) per team, over N sims."""
        rng = np.random.default_rng(seed)
        win = defaultdict(float)
        top2 = defaultdict(float)
        advance = defaultdict(float)
        for _ in range(n):
            thirds = []  # (team, record) across all groups, to rank best thirds
            for letter in self.groups:
                ranked, records = self._simulate_group(letter, sampler, rng)
                win[ranked[0]] += 1
                top2[ranked[0]] += 1
                top2[ranked[1]] += 1
                advance[ranked[0]] += 1
                advance[ranked[1]] += 1
                thirds.append((ranked[2], records[ranked[2]]))
            best = sorted(thirds, key=lambda tr: tr[1], reverse=True)[: self.best_thirds]
            for team, _rec in best:
                advance[team] += 1
        out = {}
        for letter, members in self.groups.items():
            for t in members:
                out[t] = {
                    "group": letter,
                    "win_group": win[t] / n,
                    "top2": top2[t] / n,
                    "advance": advance[t] / n,
                }
        return out

    # --- Live state ----------------------------------------------------------
    @staticmethod
    def knockout_played(config: dict, matches: list[Match]) -> dict[frozenset, dict]:
        """Already-played knockout results keyed by {frozenset(pair): {...}}.

        Group games are each team's first three; anything beyond that is knockout.
        Drawn knockout games (penalty shootouts, whose winner the results feed
        doesn't record) are left out and simulated."""
        played = [m for m in matches if m.played]
        played.sort(key=lambda m: (m.date, m.home))
        count: dict[str, int] = defaultdict(int)
        out: dict[frozenset, dict] = {}
        for m in played:
            if count[m.home] < 3 and count[m.away] < 3:
                count[m.home] += 1
                count[m.away] += 1
                continue
            if m.home_goals == m.away_goals:
                continue  # shootout winner unknown from score alone
            winner = m.home if m.home_goals > m.away_goals else m.away
            out[frozenset((m.home, m.away))] = {
                "winner": winner,
                "home": m.home, "away": m.away,
                "home_goals": m.home_goals, "away_goals": m.away_goals,
            }
        for r in config.get("knockout", {}).get("result_overrides", []):
            home, away = normalize_team(r["home"]), normalize_team(r["away"])
            pair = frozenset((home, away))
            if r.get("penalty"):
                winner = r["winner"]
            else:
                winner = home if r["home_goals"] > r["away_goals"] else away
            out[pair] = {
                "winner": winner,
                "home": home, "away": away,
                "home_goals": r["home_goals"], "away_goals": r["away_goals"],
                **({"penalty": True} if r.get("penalty") else {}),
            }
        return out

    @staticmethod
    def fixed_knockout(config: dict, matches: list[Match]) -> dict[frozenset, str]:
        """{frozenset(pair): winner} for the played knockout games."""
        return {
            pair: r["winner"]
            for pair, r in WorldCupFormat.knockout_played(config, matches).items()
        }
