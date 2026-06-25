"""The locked input interface for the entire project.

Every component speaks this language: the feature builder, the model, the
FastAPI endpoint, the frontend, and the sanity tests. There is exactly ONE
definition of "what a shot situation looks like" and it lives here.

Coordinate system (StatsBomb convention)
----------------------------------------
- Pitch is 120 (length) x 80 (width).
- Origin (0, 0) is one corner; x runs along the length, y across the width.
- The ATTACKING side shoots toward the goal at x = 120.
- The goal mouth spans y = 36 .. 44 (8 units wide), centered at y = 40.

This is the format StatsBomb 360 freeze frames provide, which means a future
video -> 2D-tracking pipeline can emit the same structure with zero rework.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Literal

# --- Pitch constants. Import these everywhere; never hard-code 120/80/etc. ---
PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
GOAL_CENTER: tuple[float, float] = (120.0, 40.0)
GOAL_POST_LEFT: tuple[float, float] = (120.0, 36.0)
GOAL_POST_RIGHT: tuple[float, float] = (120.0, 44.0)
GOAL_WIDTH: float = 8.0

# "att" = the shooter's team. "def" = the opponents (defenders + goalkeeper).
Team = Literal["att", "def"]


def _check_on_pitch(xy: tuple[float, float]) -> tuple[float, float]:
    """Reject coordinates that fall outside the pitch. Catches unit-mix-ups
    (e.g. metres vs StatsBomb units) and frontend bugs before they reach the model."""
    x, y = xy
    if not (0.0 <= x <= PITCH_LENGTH):
        raise ValueError(f"x={x} outside pitch [0, {PITCH_LENGTH}]")
    if not (0.0 <= y <= PITCH_WIDTH):
        raise ValueError(f"y={y} outside pitch [0, {PITCH_WIDTH}]")
    return (float(x), float(y))


class Player(BaseModel):
    """One outfield player or goalkeeper visible at the moment of the shot.

    These come from the freeze frame and exclude the shooter (whose position
    is `GameState.shot_xy`)."""

    xy: tuple[float, float]
    team: Team
    is_gk: bool = False

    @field_validator("xy")
    @classmethod
    def _xy_on_pitch(cls, v: tuple[float, float]) -> tuple[float, float]:
        return _check_on_pitch(v)


class GameState(BaseModel):
    """A complete shot situation: where the shot is taken from, plus the
    positions of every other visible player. This is the model's input unit
    and the API request body.

    `shot_xy` doubles as both the ball location and the shooter's location
    at the moment of the shot (they coincide for a shot event)."""

    shot_xy: tuple[float, float]
    players: list[Player] = Field(default_factory=list)

    @field_validator("shot_xy")
    @classmethod
    def _shot_on_pitch(cls, v: tuple[float, float]) -> tuple[float, float]:
        return _check_on_pitch(v)

    # --- Convenience accessors used by features / frontend / sanity tests ---
    @property
    def goalkeeper(self) -> Player | None:
        """The defending goalkeeper, if one is visible in the freeze frame."""
        for p in self.players:
            if p.team == "def" and p.is_gk:
                return p
        return None

    @property
    def defenders(self) -> list[Player]:
        return [p for p in self.players if p.team == "def"]
