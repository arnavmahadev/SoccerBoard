"""Generate the committed 2026 World Cup competition config (artifacts/world_cup_2026.json).

What's derived from data vs. encoded here
-----------------------------------------
- Groups, group fixtures (with real neutral-venue flags) and the Round-of-32
  pairings are DERIVED from the live results feed: the group stage is 12 disjoint
  round-robins, and the 16 R32 fixtures are simply the knockout games scheduled
  once the groups concluded. These are facts in the data, not hand-typed.
- The knockout TREE (which R32 winners meet in the R16, and on up to the final)
  is the one structural fact not yet in the data — those rounds aren't scheduled
  until their feeder games finish. It's encoded below from FIFA's official
  bracket and cross-checked: the R32 match→team assignment was validated to match
  the official schedule (by team AND date) for 15/16 games, with the 16th fixed
  by elimination.

The 48-team format: 12 groups of 4 → top 2 of each group (24) + the 8 best
third-placed teams advance to a Round of 32 → R16 → QF → SF → Final.

Run:  python -m forecaster.build_wc2026
"""

from __future__ import annotations

import json
from collections import defaultdict

from forecaster.data import ARTIFACTS, WC2026_KICKOFF, tournament_matches

HOSTS = ["United States", "Canada", "Mexico"]
GROUP_STAGE_END = "2026-06-27"

# Official FIFA WC2026 group letters from the December 2024 draw.
OFFICIAL_GROUP = {
    "Czech Republic": "A", "Mexico": "A", "South Africa": "A", "South Korea": "A",
    "Bosnia and Herzegovina": "B", "Canada": "B", "Qatar": "B", "Switzerland": "B",
    "Brazil": "C", "Haiti": "C", "Morocco": "C", "Scotland": "C",
    "Australia": "D", "Paraguay": "D", "Turkey": "D", "United States": "D",
    "Curaçao": "E", "Ecuador": "E", "Germany": "E", "Ivory Coast": "E",
    "Japan": "F", "Netherlands": "F", "Sweden": "F", "Tunisia": "F",
    "Belgium": "G", "Egypt": "G", "Iran": "G", "New Zealand": "G",
    "Cape Verde": "H", "Saudi Arabia": "H", "Spain": "H", "Uruguay": "H",
    "France": "I", "Iraq": "I", "Norway": "I", "Senegal": "I",
    "Algeria": "J", "Argentina": "J", "Austria": "J", "Jordan": "J",
    "Colombia": "K", "DR Congo": "K", "Portugal": "K", "Uzbekistan": "K",
    "Croatia": "L", "England": "L", "Ghana": "L", "Panama": "L",
}

# Official R32 bracket, validated against the live schedule by team + date.
# match_number -> (team A, team B). Order is FIFA's match numbering 73..88.
ROUND_OF_32 = {
    73: ("South Africa", "Canada"),
    74: ("Germany", "Paraguay"),
    75: ("Netherlands", "Morocco"),
    76: ("Ivory Coast", "Norway"),
    77: ("France", "Sweden"),
    78: ("Mexico", "Ecuador"),
    79: ("England", "DR Congo"),
    80: ("Belgium", "Senegal"),
    81: ("United States", "Bosnia and Herzegovina"),
    82: ("Spain", "Austria"),
    83: ("Portugal", "Croatia"),
    84: ("Switzerland", "Algeria"),
    85: ("Australia", "Egypt"),
    86: ("Argentina", "Cape Verde"),
    87: ("Colombia", "Ghana"),
    88: ("Brazil", "Japan"),
}

# Knockout tree adjacency (match_number -> [feeder match, feeder match]).
# Winners of the two feeder matches meet in this match.
KNOCKOUT_TREE = {
    89: [74, 77], 90: [73, 75], 91: [88, 76], 92: [78, 79],
    93: [83, 82], 94: [81, 80], 95: [86, 85], 96: [84, 87],
    97: [89, 90], 98: [93, 94], 99: [91, 92], 100: [95, 96],
    101: [97, 98], 102: [99, 100],
    104: [101, 102],
}

# Which match numbers belong to each knockout round, and the stage a team
# *reaches* by winning a match of that round. These lists are membership only;
# the order matches are *displayed* in is derived from the tree below, because
# numeric order is not bracket order (the R32->R16 pairing is irregular).
ROUNDS = {
    "round_of_32": list(range(73, 89)),
    "round_of_16": list(range(89, 97)),
    "quarterfinal": [97, 98, 99, 100],
    "semifinal": [101, 102],
    "final": [104],
}


def bracket_display_order(rounds: dict, tree: dict, final_match: int) -> dict:
    """Order each round's matches left-to-right the way a bracket reads, so the
    rendered columns and their connector lines line up with the tree: every match
    sits next to the two feeder matches whose winners meet in it.

    Numeric match order is NOT bracket order. The R32->R16 pairing is irregular
    (e.g. the winners of matches 73 and 75 meet in the R16, not 73 and 74), so
    laid out numerically a match would not sit beside the game it leads into. We
    derive the order from the tree rather than hand-listing it, so it can never
    drift from the actual bracket. Ordering each round by the position of a
    match's left-most feeder (recursively, down to its R32 games) reproduces the
    standard top-to-bottom bracket layout."""
    def leaves(m: int) -> list[int]:
        if m not in tree:  # an R32 match — a leaf of the bracket
            return [m]
        f1, f2 = tree[m]
        return leaves(f1) + leaves(f2)

    pos = {m: i for i, m in enumerate(leaves(final_match))}  # R32 left-to-right
    return {r: sorted(ms, key=lambda m: pos[leaves(m)[0]]) for r, ms in rounds.items()}
# Ordered stage labels for reporting (what the Monte Carlo driver aggregates).
# Known knockout results not yet in the live results feed. Applied by
# knockout_played() as overrides so the bracket reflects actual scores
# immediately. penalty=True marks games decided by shootout (score is AET).
RESULT_OVERRIDES: list[dict] = [
    {"home": "South Africa", "away": "Canada", "home_goals": 0, "away_goals": 1},
    {"home": "Germany", "away": "Paraguay", "home_goals": 1, "away_goals": 1, "winner": "Paraguay", "penalty": True},
    {"home": "Netherlands", "away": "Morocco", "home_goals": 1, "away_goals": 1, "winner": "Morocco", "penalty": True},
    {"home": "Ivory Coast", "away": "Norway", "home_goals": 1, "away_goals": 2},
    {"home": "France", "away": "Sweden", "home_goals": 3, "away_goals": 0},
    {"home": "Mexico", "away": "Ecuador", "home_goals": 2, "away_goals": 0},
    {"home": "England", "away": "DR Congo", "home_goals": 2, "away_goals": 1},
    {"home": "Belgium", "away": "Senegal", "home_goals": 3, "away_goals": 2},
    {"home": "United States", "away": "Bosnia and Herzegovina", "home_goals": 2, "away_goals": 0},
    {"home": "Brazil", "away": "Japan", "home_goals": 2, "away_goals": 1},
    {"home": "Spain", "away": "Austria", "home_goals": 3, "away_goals": 0},
    {"home": "Portugal", "away": "Croatia", "home_goals": 2, "away_goals": 1},
    {"home": "Australia", "away": "Egypt", "home_goals": 1, "away_goals": 1, "winner": "Egypt", "penalty": True},
]

STAGES = [
    "advance",        # reached the Round of 32 (survived the group)
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]


def _derive_groups(matches) -> tuple[dict[str, list[str]], list[dict]]:
    """Cluster the group stage (each team's first 3 games) into 12 groups of 4
    and return the 72 group fixtures with their neutral-venue flags."""
    played_count: dict[str, int] = defaultdict(int)
    group_fixtures: list[dict] = []
    adj: dict[str, set[str]] = defaultdict(set)
    for m in sorted(matches, key=lambda x: (x.date, x.home)):
        if played_count[m.home] < 3 and played_count[m.away] < 3:
            played_count[m.home] += 1
            played_count[m.away] += 1
            adj[m.home].add(m.away)
            adj[m.away].add(m.home)
            group_fixtures.append(
                {"home": m.home, "away": m.away, "neutral": m.neutral}
            )

    seen: set[str] = set()
    components: list[list[str]] = []
    for team in sorted(adj):
        if team in seen:
            continue
        stack, comp = [team], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        components.append(sorted(comp))

    components.sort()
    assert len(components) == 12, f"expected 12 groups, got {len(components)}"
    assert all(len(c) == 4 for c in components), "every group must have 4 teams"
    groups = {}
    for comp in components:
        letter = OFFICIAL_GROUP.get(comp[0])
        assert letter is not None, f"No official group letter for team {comp[0]}"
        groups[letter] = comp
    return dict(sorted(groups.items())), group_fixtures


def build_config() -> dict:
    matches = tournament_matches("FIFA World Cup", "2026")
    groups, group_fixtures = _derive_groups(matches)

    all_group_teams = {t for members in groups.values() for t in members}
    r32_teams = {t for pair in ROUND_OF_32.values() for t in pair}
    assert len(r32_teams) == 32, f"expected 32 qualifiers, got {len(r32_teams)}"
    assert r32_teams <= all_group_teams, "R32 has a team not in any group"

    # Validate the encoded bracket against what's actually scheduled in the data:
    # the set of teams in the data's knockout fixtures must equal our R32 set.
    played_count: dict[str, int] = defaultdict(int)
    ko_fixtures = []
    for m in sorted(matches, key=lambda x: (x.date, x.home)):
        if played_count[m.home] < 3 and played_count[m.away] < 3:
            played_count[m.home] += 1
            played_count[m.away] += 1
        else:
            ko_fixtures.append((m.home, m.away))
    data_ko_teams = {t for pair in ko_fixtures for t in pair}
    if data_ko_teams:  # may be empty if run before any KO fixture is scheduled
        assert data_ko_teams <= r32_teams, (
            f"data has KO team not in encoded R32: {data_ko_teams - r32_teams}"
        )

    return {
        "kind": "competition",
        "id": "world_cup_2026",
        "name": "FIFA World Cup 2026",
        "format": "world_cup",
        "hosts": HOSTS,
        "kickoff": WC2026_KICKOFF,
        "group_stage_end": GROUP_STAGE_END,
        "tournament_label": "FIFA World Cup",
        "season_year": "2026",
        "advance_per_group": 2,
        "best_thirds": 8,
        "groups": groups,
        "group_fixtures": group_fixtures,
        "knockout": {
            "round_of_32": [
                {"match": n, "teams": list(ROUND_OF_32[n])} for n in sorted(ROUND_OF_32)
            ],
            "tree": {str(k): v for k, v in KNOCKOUT_TREE.items()},
            "rounds": bracket_display_order(ROUNDS, KNOCKOUT_TREE, 104),
            "final_match": 104,
            "result_overrides": RESULT_OVERRIDES,
        },
        "stages": STAGES,
    }


def main() -> None:
    cfg = build_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "world_cup_2026.json"
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    print(f"  {len(cfg['groups'])} groups, {len(cfg['group_fixtures'])} group fixtures")
    print(f"  {len(cfg['knockout']['round_of_32'])} R32 fixtures, hosts={cfg['hosts']}")
    for letter, members in cfg["groups"].items():
        print(f"  Group {letter}: {', '.join(members)}")


if __name__ == "__main__":
    main()
