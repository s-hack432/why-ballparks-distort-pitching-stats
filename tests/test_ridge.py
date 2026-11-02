"""Unit tests for the sparse ridge machinery (src/ridge.py)."""
import numpy as np
import pandas as pd
import pytest

from src import ridge


def simulate(n=60000, n_parks=6, n_p=150, n_b=150, seed=0):
    rng = np.random.default_rng(seed)
    park_eff = np.linspace(-0.04, 0.05, n_parks)
    p_eff = rng.normal(0, 0.03, n_p)
    b_eff = rng.normal(0, 0.03, n_b)
    df = pd.DataFrame({
        "park": rng.integers(0, n_parks, n).astype(str),
        "season": rng.choice([2024, 2025], n),
        "same_hand": rng.integers(0, 2, n),
        "pitcher_is_home": rng.integers(0, 2, n),
        "pitcher": rng.integers(0, n_p, n),
        "batter": rng.integers(0, n_b, n),
        "game_pk": rng.integers(0, 2000, n),
    })
    mu = (0.30 + park_eff[df["park"].astype(int)] + 0.01 * (df["season"] == 2025)
          - 0.02 * df["same_hand"] - 0.01 * df["pitcher_is_home"]
          + p_eff[df["pitcher"]] + b_eff[df["batter"]])
    df["y"] = mu + rng.normal(0, 0.4, n)
    return df, park_eff


def test_solve_matches_dense_normal_equations():
    df, _ = simulate(n=3000, n_p=40, n_b=40)
    d = ridge.build_design(df, ridge.standard_blocks(True))
    y = df["y"].to_numpy()
    lams = {"pitcher": 50.0, "batter": 80.0}
    beta = ridge.fit(d, y, lams)
    X = d.X.toarray()
    ref = np.linalg.solve(X.T @ X + np.diag(d.penalty_vector(lams)), X.T @ y)
    np.testing.assert_allclose(beta, ref, atol=1e-8)


def test_weights_equal_row_duplication():
    df, _ = simulate(n=2000, n_p=30, n_b=30)
    d = ridge.build_design(df, ridge.standard_blocks(True))
    w = np.random.default_rng(1).integers(0, 3, len(df)).astype(float)
    lams = {"pitcher": 20.0, "batter": 20.0}
    beta_w = ridge.fit(d, df["y"].to_numpy(), lams, w=w)
    dup = df.loc[df.index.repeat(w.astype(int))].reset_index(drop=True)
    d2 = ridge.build_design(dup, ridge.standard_blocks(True))
    beta_d = ridge.fit(d2, dup["y"].to_numpy(), lams)
    # Same levels survive in both designs here, so coefficients are directly comparable.
    np.testing.assert_allclose(d.coef(beta_w, "park").values, d2.coef(beta_d, "park").values, atol=1e-8)


def test_recovers_park_effects():
    df, park_eff = simulate()
    d = ridge.build_design(df, ridge.standard_blocks(True))
    beta = ridge.fit(d, df["y"].to_numpy(), {"pitcher": 150.0, "batter": 150.0})
    est = ridge.center(d.coef(beta, "park").values, ridge.block_counts(d, "park"))
    truth = park_eff[d.levels["park"].astype(int)]
    truth = truth - np.average(truth, weights=ridge.block_counts(d, "park"))
    np.testing.assert_allclose(est, truth, atol=0.012)
    assert d.coef(beta, "same_hand").iloc[0] == pytest.approx(-0.02, abs=0.01)


def test_shared_penalty_keeps_outcome_shares_additive():
    # Three exhaustive indicators fitted with the same penalty: park coefficients sum to 1.
    df, _ = simulate(n=5000, n_p=50, n_b=50)
    rng = np.random.default_rng(3)
    cat = rng.choice(3, len(df), p=[0.23, 0.09, 0.68])
    Y = np.eye(3)[cat]
    d = ridge.build_design(df, ridge.standard_blocks(True))
    beta = ridge.fit(d, Y, {"pitcher": 100.0, "batter": 100.0})
    np.testing.assert_allclose(d.coef(beta, "park").sum(axis=1).values, 1.0, atol=1e-8)


def test_partial_onehot_block():
    # Team-specific home terms: rows with no level (road plate appearances) get no column.
    df = pd.DataFrame({"term": [1.0, np.nan, 2.0, np.nan, 1.0]})
    d = ridge.build_design(df, [ridge.Block("t", "term", allow_missing=True)])
    np.testing.assert_array_equal(d.X.toarray(), [[1, 0], [0, 0], [0, 1], [0, 0], [1, 0]])
    with pytest.raises(ValueError):
        ridge.build_design(df, [ridge.Block("t", "term")])


def test_center_weighted_mean_zero():
    e = np.array([0.1, -0.2, 0.05])
    w = np.array([10, 30, 60])
    assert np.average(ridge.center(e, w), weights=w) == pytest.approx(0.0, abs=1e-12)


def test_game_folds_keep_games_together():
    games = np.repeat(np.arange(50), 7)
    f = ridge.game_folds(games, 5, seed=0)
    assert pd.DataFrame({"g": games, "f": f}).groupby("g")["f"].nunique().max() == 1
    assert set(f) == set(range(5))


def test_stratified_bootstrap_preserves_games_per_stratum():
    games = np.arange(100)
    strata = np.repeat(["a", "b", "c", "d"], 25)
    counts = ridge.stratified_game_weights(games, strata, np.random.default_rng(0))
    by_stratum = pd.Series(counts).groupby(pd.Series(strata, index=games).reindex(list(counts))).sum()
    assert (by_stratum == 25).all()
    w = ridge.row_weights(np.array([0, 0, 1, 99]), counts)
    assert w[0] == w[1] == counts.get(0, 0)
