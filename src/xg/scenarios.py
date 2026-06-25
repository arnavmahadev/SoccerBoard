"""Known-scenario sanity checks.

A tiny set of shot situations whose xG we already know from football common
sense. Re-run these after ANY change to features or models to confirm nothing
broke. They are the project's smoke test.

Each scenario carries an expected xG *range* (not a point value) — wide enough
to tolerate model variation, tight enough to catch a genuinely broken pipeline
(e.g. a penalty scoring 0.02, or a hopeless angle scoring 0.5).

Until a trained model exists (Phase 3), the test suite only checks that these
construct and validate. Once `xg.models` can predict, the range assertions
turn on automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from xg.data.schema import GameState, Player


@dataclass(frozen=True)
class Scenario:
    name: str
    state: GameState
    xg_low: float
    xg_high: float
    note: str
    shot_type: str = "open_play"  # serving hint; "penalty" triggers the constant path


# A penalty: shot from the spot (12 yards out = x=108), dead center, only the
# keeper on his line. Historic conversion rate is ~0.76.
PENALTY = Scenario(
    name="penalty",
    state=GameState(
        shot_xy=(108.0, 40.0),
        players=[Player(xy=(119.5, 40.0), team="def", is_gk=True)],
    ),
    xg_low=0.70,
    xg_high=0.82,
    note="Penalty kick — central, 12 yards, keeper on the line. ~0.76 historically.",
    shot_type="penalty",
)

# A gilt-edged open-play chance: 8 yards out, dead center, only the keeper to
# beat and no defenders in the way. The model should rate this highly.
CLEAR_CHANCE = Scenario(
    name="clear_chance",
    state=GameState(
        shot_xy=(112.0, 40.0),
        players=[Player(xy=(119.0, 40.0), team="def", is_gk=True)],
    ),
    xg_low=0.30,
    xg_high=0.95,
    note="Open-play tap-in range, central, unguarded — a big chance.",
)

# A near-impossible chance: almost on the goal line but pushed wide toward the
# corner, so the visible goal angle is tiny. Should be ~0.
TIGHT_ANGLE_BYLINE = Scenario(
    name="tight_angle_byline",
    state=GameState(
        shot_xy=(119.0, 12.0),
        players=[Player(xy=(119.8, 40.0), team="def", is_gk=True)],
    ),
    xg_low=0.0,
    xg_high=0.05,
    note="Shot from the byline at an extreme angle — almost no goal to aim at.",
)

ALL: list[Scenario] = [PENALTY, TIGHT_ANGLE_BYLINE, CLEAR_CHANCE]
