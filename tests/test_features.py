"""Feature builder must produce sane geometry for known shots."""

import math

from xg.data.schema import GameState, Player
from xg.features.build import (
    FEATURE_NAMES,
    defenders_in_cone,
    nearest_defender_distance,
    shot_angle,
    shot_distance,
    state_to_features,
)
from xg.scenarios import PENALTY, TIGHT_ANGLE_BYLINE, CLEAR_CHANCE


def test_feature_vector_shape_and_order():
    vec = state_to_features(CLEAR_CHANCE.state)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert not any(map(math.isnan, vec))


def test_penalty_distance_is_twelve():
    assert shot_distance(PENALTY.state) == 12.0


def test_central_shot_has_wider_angle_than_byline():
    # The whole point of the angle feature: central penalty sees more net than
    # an extreme byline shot.
    assert shot_angle(PENALTY.state) > shot_angle(TIGHT_ANGLE_BYLINE.state)
    assert shot_angle(TIGHT_ANGLE_BYLINE.state) < math.radians(5)  # near zero


def test_closer_shot_has_larger_angle():
    near = GameState(shot_xy=(114.0, 40.0))
    far = GameState(shot_xy=(80.0, 40.0))
    assert shot_angle(near) > shot_angle(far)


def test_defender_on_goal_line_is_in_cone():
    s = GameState(
        shot_xy=(100.0, 40.0),
        players=[Player(xy=(115.0, 40.0), team="def")],  # directly between shot and goal
    )
    assert defenders_in_cone(s) == 1


def test_defender_behind_shooter_not_in_cone():
    s = GameState(
        shot_xy=(100.0, 40.0),
        players=[Player(xy=(90.0, 40.0), team="def")],  # behind the shot
    )
    assert defenders_in_cone(s) == 0


def test_nearest_defender_default_when_none_visible():
    s = GameState(shot_xy=(100.0, 40.0))
    assert nearest_defender_distance(s) > 100  # the "no one nearby" sentinel
