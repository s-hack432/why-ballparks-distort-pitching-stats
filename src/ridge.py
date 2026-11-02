"""Sparse ridge regression with block-specific penalties (the RAPM approach).

Each Step 4 model has the form

    y = park + season + same_hand + pitcher_is_home + pitcher + batter + error

with one dummy per park and no global intercept: the park dummies act as intercepts, so every
park can be left unpenalized. Pitcher and batter effects get ridge penalties (lambda_p,
lambda_b), chosen by cross-validation. Park, season, platoon and home terms get no penalty.

The fit solves the normal equations (X'WX + Lambda) beta = X'Wy with a dense Cholesky
factorization. X has hundreds of thousands of rows but only a few thousand columns, so the
system is small. W holds row weights; the bootstrap uses these as game multiplicities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


@dataclass
class Block:
    """One group of design columns."""
    name: str
    col: str
    kind: str = "onehot"          # "onehot" (one column per level) or "numeric" (as is)
    drop_first: bool = False      # drop the first level (reference category)
    penalty: str | None = None    # key into the lambda dict; None = unpenalized
    allow_missing: bool = False   # onehot only: rows with a missing level get no column


@dataclass
class Design:
    X: sparse.csr_matrix
    blocks: list[Block]
    slices: dict[str, slice] = field(default_factory=dict)
    levels: dict[str, pd.Index] = field(default_factory=dict)

    @property
    def n_cols(self) -> int:
        return self.X.shape[1]

    def penalty_vector(self, lams: dict[str, float]) -> np.ndarray:
        pen = np.zeros(self.n_cols)
        for b in self.blocks:
            if b.penalty is not None:
                pen[self.slices[b.name]] = lams[b.penalty]
        return pen

    def coef(self, beta: np.ndarray, block: str) -> pd.Series | pd.DataFrame:
        """Coefficients of one block, indexed by level (beta may have several columns)."""
        vals = beta[self.slices[block]]
        idx = self.levels[block]
        return pd.DataFrame(vals, index=idx) if vals.ndim == 2 else pd.Series(vals, index=idx)


def standard_blocks(multi_season: bool, park_col: str = "park",
                    home_block: Block | None = None) -> list[Block]:
    """Blocks of the Step 4 models: park + season + platoon + home + pitcher + batter."""
    blocks = [Block("park", park_col)]
    if multi_season:
        blocks.append(Block("season", "season", drop_first=True))
    blocks.append(Block("same_hand", "same_hand", kind="numeric"))
    blocks.append(home_block or Block("home", "pitcher_is_home", kind="numeric"))
    blocks.append(Block("pitcher", "pitcher", penalty="pitcher"))
    blocks.append(Block("batter", "batter", penalty="batter"))
    return blocks


def build_design(df: pd.DataFrame, blocks: list[Block]) -> Design:
    n = len(df)
    rows, cols, vals = [], [], []
    slices, levels = {}, {}
    start = 0
    for b in blocks:
        if b.kind == "numeric":
            v = df[b.col].to_numpy(dtype=float)
            nz = np.flatnonzero(v)
            rows.append(nz)
            cols.append(np.full(len(nz), start))
            vals.append(v[nz])
            slices[b.name] = slice(start, start + 1)
            levels[b.name] = pd.Index([b.col])
            start += 1
            continue
        codes, uniques = pd.factorize(df[b.col], sort=True)
        if (codes < 0).any() and not b.allow_missing:
            raise ValueError(f"Missing values in design column {b.col}")
        lev = pd.Index(uniques)
        if b.drop_first:
            keep = codes > 0
            r = np.flatnonzero(keep)
            c = codes[keep] - 1
            lev = lev[1:]
        else:
            r = np.flatnonzero(codes >= 0)
            c = codes[r]
        rows.append(r)
        cols.append(c + start)
        vals.append(np.ones(len(r)))
        slices[b.name] = slice(start, start + len(lev))
        levels[b.name] = lev
        start += len(lev)
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, start))
    return Design(X=X, blocks=blocks, slices=slices, levels=levels)


def gram(X: sparse.csr_matrix, w: np.ndarray | None = None) -> np.ndarray:
    """Dense X' diag(w) X."""
    Xw = X if w is None else sparse.diags(w) @ X
    return (X.T @ Xw).toarray()


def xty(X: sparse.csr_matrix, Y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    Yw = Y if w is None else Y * (w[:, None] if Y.ndim == 2 else w)
    return np.asarray(X.T @ Yw)


def solve(G: np.ndarray, B: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """Solve (G + diag(pen)) beta = B by Cholesky. A tiny jitter guards against columns with
    no rows (e.g. a pitcher absent from a bootstrap resample, which is penalized anyway)."""
    K = G.copy()
    d = np.diag_indices_from(K)
    K[d] += pen + 1e-9 * max(np.mean(np.diag(G)), 1.0)
    c = cho_factor(K, lower=False, overwrite_a=True, check_finite=False)
    return cho_solve(c, B, check_finite=False)


def fit(design: Design, Y: np.ndarray, lams: dict[str, float], w: np.ndarray | None = None) -> np.ndarray:
    """Ridge coefficients for one or more outcome columns sharing the same penalties."""
    return solve(gram(design.X, w), xty(design.X, Y, w), design.penalty_vector(lams))


def center(effects: pd.Series | np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Center effects at zero with weights `counts` (league-average park = 0)."""
    e = np.asarray(effects, float)
    wts = np.asarray(counts, float)
    return e - (wts @ e) / wts.sum() if e.ndim == 1 else e - (wts @ e) / wts.sum()


def block_counts(design: Design, block: str, w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) number of rows in each level of a one-hot block."""
    Xb = design.X[:, design.slices[block]]
    return np.asarray(Xb.sum(axis=0) if w is None else Xb.T @ w).ravel()


# --------------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------------
def game_folds(game_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole games to folds at random."""
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    fold_of_game = rng.permutation(len(games)) % n_folds
    return fold_of_game[np.searchsorted(games, game_ids)]


def cv_grid(design: Design, Y: np.ndarray, folds: np.ndarray, grid_p, grid_b,
            outcome_names: list[str]) -> pd.DataFrame:
    """Held-out mean squared error for every (lambda_p, lambda_b) pair and outcome.

    Folds are whole games, so held-out plate appearances come from games the model never saw.
    """
    Y = Y if Y.ndim == 2 else Y[:, None]
    out = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        Xtr, Xte = design.X[tr], design.X[te]
        G, B = gram(Xtr), xty(Xtr, Y[tr])
        for lp in grid_p:
            for lb in grid_b:
                beta = solve(G, B, design.penalty_vector({"pitcher": lp, "batter": lb}))
                err = ((Y[te] - Xte @ beta) ** 2).sum(axis=0)
                for name, e in zip(outcome_names, err):
                    out.append({"fold": k, "lambda_pitcher": lp, "lambda_batter": lb,
                                "outcome": name, "sse": e, "n": te.sum()})
    res = pd.DataFrame(out)
    agg = (res.groupby(["outcome", "lambda_pitcher", "lambda_batter"])
           .agg(sse=("sse", "sum"), n=("n", "sum")).reset_index())
    agg["mse"] = agg["sse"] / agg["n"]
    return agg.drop(columns="sse")


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------
def stratified_game_weights(game_ids: np.ndarray, strata: np.ndarray, rng) -> dict:
    """One bootstrap resample of games, drawn with replacement within each stratum.

    Returns {game_id: multiplicity} for games drawn at least once.
    """
    games = pd.DataFrame({"g": game_ids, "s": strata}).drop_duplicates("g")
    counts = {}
    for _, grp in games.groupby("s"):
        draw = rng.choice(grp["g"].to_numpy(), size=len(grp), replace=True)
        u, c = np.unique(draw, return_counts=True)
        counts.update(zip(u, c))
    return counts


def row_weights(row_games: np.ndarray, game_counts: dict) -> np.ndarray:
    keys = np.fromiter(game_counts.keys(), dtype=np.int64)
    vals = np.fromiter(game_counts.values(), dtype=float)
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    pos = np.searchsorted(keys, row_games)
    pos = np.clip(pos, 0, len(keys) - 1)
    return np.where(keys[pos] == row_games, vals[pos], 0.0)
