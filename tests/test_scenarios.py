"""Known-scenario smoke tests.

Two layers:
  1. Always-on: the scenarios construct, validate, and have sane geometry.
  2. Activates in Phase 3: once a model can predict, assert the xG lands in the
     expected range. Until then those checks skip cleanly.
"""

import pytest

from xg.scenarios import ALL, PENALTY, TIGHT_ANGLE_BYLINE


def _try_import_predict():
    """Return a predict(GameState) -> float callable, or None if no model yet."""
    try:
        from xg.models.baseline import predict  # noqa: F811
    except Exception:
        return None
    return predict


@pytest.mark.parametrize("scenario", ALL, ids=[s.name for s in ALL])
def test_scenario_validates(scenario):
    # Constructing the GameState already ran pydantic validation; assert basics.
    assert 0.0 <= scenario.state.shot_xy[0] <= 120.0
    assert 0.0 <= scenario.xg_low <= scenario.xg_high <= 1.0


def test_penalty_is_central_and_close():
    x, y = PENALTY.state.shot_xy
    assert x == 108.0  # 12 yards from the goal line
    assert y == 40.0   # dead center


def test_byline_shot_is_extreme_angle():
    x, y = TIGHT_ANGLE_BYLINE.state.shot_xy
    assert x > 118.0          # almost on the goal line
    assert abs(y - 40.0) > 20 # pushed well wide of center


@pytest.mark.parametrize("scenario", ALL, ids=[s.name for s in ALL])
def test_scenario_xg_in_range(scenario):
    predict = _try_import_predict()
    if predict is None:
        pytest.skip("no trained model yet (activates in Phase 3)")
    xg = predict(scenario.state)
    assert scenario.xg_low <= xg <= scenario.xg_high, (
        f"{scenario.name}: xG={xg:.3f} outside "
        f"[{scenario.xg_low}, {scenario.xg_high}] — {scenario.note}"
    )
