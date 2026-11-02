"""Unit tests for the Shapley decomposition of park effects (src/decompose.py)."""
import numpy as np
import pytest

from src.decompose import decompose, shapley, woba_from_state

BASE = {"W": 0.093, "c": 0.682, "x": 0.367, "v": -0.002, "w_w": 0.70}


def brute_force(base, delta):
    """Shapley values from the generic 6-ordering average."""
    def f(on):
        W = base["W"] + (delta["W"] if 0 in on else 0)
        c = base["c"] + (delta["c"] if 0 in on else 0)
        x = base["x"] + (delta["x"] if 1 in on else 0)
        v = base["v"] + (delta["v"] if 2 in on else 0)
        return woba_from_state(W, c, x, v, base["w_w"])
    return shapley(f, 3)


@pytest.mark.parametrize("delta", [
    {"K": -0.025, "W": 0.004, "c": 0.021, "x": 0.010, "v": 0.035},   # Coors-like
    {"K": 0.034, "W": -0.001, "c": -0.033, "x": -0.004, "v": -0.010},  # T-Mobile-like
    {"K": 0.0, "W": 0.0, "c": 0.0, "x": 0.0, "v": 0.02},             # conversion only
])
def test_closed_form_equals_permutation_average(delta):
    d = decompose(BASE, delta)
    phi = brute_force(BASE, delta)
    np.testing.assert_allclose([d["kbb"], d["contact"], d["conv"]], phi, atol=1e-12)


def test_shapley_values_sum_to_total():
    delta = {"K": -0.02, "W": 0.005, "c": 0.015, "x": 0.012, "v": 0.03}
    d = decompose(BASE, delta)
    assert d["kbb"] + d["contact"] + d["conv"] == pytest.approx(d["total"], abs=1e-12)


def test_interaction_term_formula():
    delta = {"K": -0.02, "W": 0.005, "c": 0.015, "x": 0.012, "v": 0.03}
    d = decompose(BASE, delta)
    assert d["interaction"] == pytest.approx(delta["c"] * (delta["x"] + delta["v"]), abs=1e-12)


def test_single_channel_gets_everything():
    d = decompose(BASE, {"K": 0, "W": 0, "c": 0, "x": 0, "v": 0.02})
    assert d["conv"] == pytest.approx(0.02 * BASE["c"])
    assert d["kbb"] == 0 and d["contact"] == 0
    assert d["interaction"] == pytest.approx(0.0, abs=1e-15)


def test_strikeout_walk_split_is_exact():
    delta = {"K": -0.03, "W": 0.006, "c": 0.024, "x": 0.0, "v": 0.0}
    d = decompose(BASE, delta)
    assert d["k_part"] + d["bbhbp_part"] == pytest.approx(d["kbb"], abs=1e-12)
    # Fewer strikeouts turn into balls in play worth about xwOBAcon each.
    assert d["k_part"] == pytest.approx(0.03 * (BASE["x"] + BASE["v"]), rel=1e-9)


def test_vectorized_over_bootstrap_draws():
    rng = np.random.default_rng(0)
    delta = {k: rng.normal(0, 0.01, (100, 5)) for k in ("K", "W", "x", "v")}
    delta["c"] = -delta["K"] - delta["W"]
    d = decompose(BASE, delta)
    assert d["total"].shape == (100, 5)
    np.testing.assert_allclose(d["kbb"] + d["contact"] + d["conv"], d["total"], atol=1e-12)
