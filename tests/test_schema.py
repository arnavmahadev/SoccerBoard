"""The schema contract is enforced at runtime — prove it."""

import pytest
from pydantic import ValidationError

from xg.data.schema import GameState, Player, PITCH_LENGTH, PITCH_WIDTH


def test_valid_state_constructs():
    s = GameState(
        shot_xy=(108.0, 40.0),
        players=[Player(xy=(119.5, 40.0), team="def", is_gk=True)],
    )
    assert s.shot_xy == (108.0, 40.0)
    assert len(s.players) == 1


def test_empty_freeze_frame_is_allowed():
    # A shot with no other visible players is valid (sparse 360 frames happen).
    s = GameState(shot_xy=(100.0, 40.0))
    assert s.players == []


def test_goalkeeper_accessor():
    s = GameState(
        shot_xy=(100.0, 40.0),
        players=[
            Player(xy=(119.0, 40.0), team="def", is_gk=True),
            Player(xy=(110.0, 35.0), team="def"),
            Player(xy=(105.0, 45.0), team="att"),
        ],
    )
    assert s.goalkeeper is not None
    assert s.goalkeeper.is_gk
    assert len(s.defenders) == 2


@pytest.mark.parametrize("bad", [(-1.0, 40.0), (PITCH_LENGTH + 1, 40.0), (60.0, PITCH_WIDTH + 5)])
def test_off_pitch_coordinates_rejected(bad):
    with pytest.raises(ValidationError):
        GameState(shot_xy=bad)


def test_invalid_team_rejected():
    with pytest.raises(ValidationError):
        Player(xy=(100.0, 40.0), team="midfield")  # type: ignore[arg-type]
