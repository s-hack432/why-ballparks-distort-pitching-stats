"""Unit tests for wOBA bookkeeping and batted-ball geometry (src/woba.py)."""
import numpy as np
import pandas as pd
import pytest

from src.woba import (aggregate_woba, field_feet, in_woba, outcome_type, pull_angle,
                      spray_angle, woba_values)

WEIGHTS = pd.DataFrame(
    {"walk": [0.70], "hit_by_pitch": [0.72], "single": [0.89], "double": [1.26],
     "triple": [1.60], "home_run": [2.05]}, index=pd.Index([2026], name="season"))


def test_outcome_types():
    ev = pd.Series(["strikeout", "strikeout_double_play", "walk", "intent_walk", "hit_by_pitch",
                    "catcher_interf", "single", "field_error", "sac_bunt", "sac_fly",
                    "truncated_pa"])
    assert outcome_type(ev).tolist() == ["K", "K", "BB", "IBB", "HBP", "CI", "BIP", "BIP", "BIP",
                                         "BIP", "NOT_PA"]


def test_unknown_event_raises():
    with pytest.raises(ValueError):
        outcome_type(pd.Series(["single", "new_event_type"]))


def test_in_woba_denominator():
    # Denominator = AB + BB - IBB + SF + HBP: IBB, sac bunts and catcher's interference are out.
    ev = pd.Series(["walk", "intent_walk", "hit_by_pitch", "sac_bunt", "sac_fly", "catcher_interf",
                    "field_error", "strikeout"])
    assert in_woba(ev).tolist() == [True, False, True, False, True, False, True, True]


def test_woba_values_errors_and_fc_are_zero():
    ev = pd.Series(["single", "home_run", "walk", "field_error", "fielders_choice", "strikeout",
                    "double"])
    vals = woba_values(ev, pd.Series([2026] * len(ev)), WEIGHTS)
    assert vals.tolist() == [0.89, 2.05, 0.70, 0.0, 0.0, 0.0, 1.26]


def test_aggregate_woba():
    df = pd.DataFrame({
        "pitcher": [1, 1, 1, 2, 2],
        "in_woba": [True, True, False, True, True],
        "woba_val": [0.89, 0.0, 0.0, 2.05, 0.70],
        "xwoba": [0.50, 0.0, 0.0, 1.50, 0.70],
    })
    g = aggregate_woba(df, "pitcher")
    assert g.loc[1, "n"] == 2 and g.loc[1, "woba"] == pytest.approx(0.445)
    assert g.loc[2, "woba"] == pytest.approx(1.375) and g.loc[2, "xwoba"] == pytest.approx(1.10)


def test_spray_angle_directions():
    # Straight up the middle, down the left-field line, down the right-field line.
    hx = np.array([125.42, 125.42 - 100, 125.42 + 100])
    hy = np.array([98.27, 98.27, 98.27])
    np.testing.assert_allclose(spray_angle(hx, hy), [0.0, -45.0, 45.0], atol=1e-9)


def test_pull_angle_sign_by_handedness():
    # A ball to left field is pulled by a right-handed batter, pushed oppo by a lefty.
    assert pull_angle(np.array([-30.0]), np.array(["R"]))[0] == 30.0
    assert pull_angle(np.array([-30.0]), np.array(["L"]))[0] == -30.0


def test_field_feet():
    x, y = field_feet(np.array([125.42 + 10]), np.array([198.27 - 100]), 2.5)
    assert x[0] == pytest.approx(25.0) and y[0] == pytest.approx(250.0)
