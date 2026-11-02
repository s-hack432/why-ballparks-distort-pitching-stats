"""Step 4: estimate park effects channel by channel.

Every model has the same structure, at the plate-appearance level:

    outcome ~ park + season + same_hand + pitcher_is_home + pitcher + batter

* PA sample (every plate appearance in the wOBA denominator):
    1. strikeout, 2. walk + HBP, 3. ball in play (the contact share c),
    plus total wOBA as a consistency check on Step 5.
* Ball-in-play sample:
    4. xwOBA on contact (xwOBAcon), 5. conversion residual (wOBA value - xwOBA),
    plus home runs per ball in play (used in Step 8) and wOBAcon (a check).

Linear probability / linear models with ridge penalties on pitcher and batter effects only
(src/ridge.py). The penalties come from 5-fold cross-validation over whole games on a 9 x 9 grid
of (lambda_pitcher, lambda_batter). Models 1-3 share one pair, the one minimizing their summed
held-out squared error. With a shared penalty, the predicted strikeout, walk/HBP and contact
shares sum to exactly 1 in every park, which the Step 5 decomposition needs. Each other
outcome gets its own pair.

Park effects are centered at zero, weighted by plate appearances (balls in play for the
contact models): 0 = a league-average park.

Fits
* "pooled": 2024-2026. A venue contributes only the seasons it was in use; renamed parks share
  an ID, relocated teams get new venues.
* "2026": 2026 only (robustness check).
* "pooled_literal": the pooled fit using the raw Statcast columns (woba_value, woba_denom ==
  1, estimated_woba_using_speedangle), as a sensitivity check on the approved wOBA convention.
  Point estimates only.
* Penalty sensitivity: the pooled fit with every lambda divided or multiplied by 4. Point
  estimates only.

Uncertainty: 500 bootstrap resamples of games, drawn with replacement within each season x
venue, refitting every model with the cross-validated penalties.

Outputs
    outputs/tables/park_channel_effects.csv   venue, PAs, each channel's estimate + 95% interval
    outputs/tables/model_penalties.csv        chosen penalties and held-out error
    outputs/tables/model_cv.csv               full cross-validation grid
    outputs/tables/model_fixed_effects.csv    season, platoon and home coefficients (+ intervals)
    outputs/tables/park_effects_sensitivity.csv
    data/processed/park_effects_<fit>.npz     point estimates and bootstrap draws (Step 5)
    data/processed/player_effects_pooled.parquet
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import ridge
from .common import get_logger, log_df, write_parquet_atomic
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OTHER_PARK, OUT_TABLES, RANDOM_SEED

log = get_logger("park_model", "step4_park_model.log")

LAMBDA_GRID = [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
N_BOOT = 500
N_FOLDS = 5

# outcome key -> (sample, column in the PA table, penalty group)
OUTCOMES = {
    "k":        ("pa", "is_k", "kbb"),
    "bbhbp":    ("pa", "is_bbhbp", "kbb"),
    "bip":      ("pa", "is_bip", "kbb"),
    "woba":     ("pa", "woba_val", "woba"),
    "xwobacon": ("bip", "xwoba", "xwobacon"),
    "conv":     ("bip", "conv", "conv"),
    "hr":       ("bip", "is_hr", "hr"),
    "wobacon":  ("bip", "woba_val", "wobacon"),
}
LABELS = {"k": "Strikeout rate", "bbhbp": "Walk + HBP rate", "bip": "Ball-in-play rate (c)",
          "woba": "wOBA (direct model)", "xwobacon": "xwOBA on contact",
          "conv": "Conversion (wOBAcon - xwOBAcon)", "hr": "HR per ball in play",
          "wobacon": "wOBA on contact (direct model)"}


@dataclass
class FitSpec:
    name: str
    seasons: tuple
    park_col: str
    in_woba_col: str = "in_woba"
    columns: dict | None = None   # outcome key -> column, overriding OUTCOMES


FITS = {
    "pooled": FitSpec("pooled", (2024, 2025, 2026), "park"),
    "2026": FitSpec("2026", (ANALYSIS_SEASON,), "park_2026"),
    "pooled_literal": FitSpec("pooled_literal", (2024, 2025, 2026), "park", "in_woba_literal",
                              {"woba": "woba_value_sc", "xwobacon": "xwoba_sc",
                               "conv": "conv_literal", "wobacon": "woba_value_sc"}),
}


def add_literal_columns(pa: pd.DataFrame) -> pd.DataFrame:
    """Columns for the raw-Statcast-columns sensitivity fit."""
    pa["in_woba_literal"] = pa["woba_denom_sc"].fillna(0).eq(1)
    pa["conv_literal"] = np.where(pa["in_woba_literal"] & pa["is_bip"],
                                  pa["woba_value_sc"] - pa["xwoba_sc"], np.nan)
    return pa


def samples(pa: pd.DataFrame, spec: FitSpec) -> dict[str, pd.DataFrame]:
    d = pa[pa["season"].isin(spec.seasons) & pa[spec.in_woba_col]].copy()
    d["park_key"] = d[spec.park_col].astype(str)
    out = {"pa": d.reset_index(drop=True),
           "bip": d[d["is_bip"]].reset_index(drop=True)}
    if spec.columns:  # literal fit: balls in play need a Savant xwOBA
        out["bip"] = out["bip"][out["bip"]["xwoba_sc"].notna()].reset_index(drop=True)
    return out


def outcome_matrix(df: pd.DataFrame, keys: list[str], spec: FitSpec) -> np.ndarray:
    cols = [(spec.columns or {}).get(k, OUTCOMES[k][1]) for k in keys]
    return df[cols].to_numpy(dtype=float)


def groups_for(sample: str) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for k, (s, _, grp) in OUTCOMES.items():
        if s == sample:
            g.setdefault(grp, []).append(k)
    return g


# --------------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------------
def choose_penalties(designs, data, spec: FitSpec) -> tuple[dict, pd.DataFrame]:
    """Cross-validate each penalty group; return {group: (lambda_p, lambda_b)} and the grid."""
    chosen, grids = {}, []
    for sample in ("pa", "bip"):
        keys = [k for k, (s, _, _) in OUTCOMES.items() if s == sample]
        Y = outcome_matrix(data[sample], keys, spec)
        folds = ridge.game_folds(data[sample]["game_pk"].to_numpy(), N_FOLDS, RANDOM_SEED)
        t0 = time.time()
        cv = ridge.cv_grid(designs[sample], Y, folds, LAMBDA_GRID, LAMBDA_GRID, keys)
        log.info("  CV on %s sample (%d rows, %d folds x %d penalty pairs): %.0fs", sample,
                 len(data[sample]), N_FOLDS, len(LAMBDA_GRID) ** 2, time.time() - t0)
        cv["fit"] = spec.name
        grids.append(cv)
        for grp, members in groups_for(sample).items():
            tot = (cv[cv["outcome"].isin(members)]
                   .groupby(["lambda_pitcher", "lambda_batter"])["mse"].sum())
            lp, lb = tot.idxmin()
            chosen[grp] = (float(lp), float(lb))
            edge = lp in (LAMBDA_GRID[0], LAMBDA_GRID[-1]) or lb in (LAMBDA_GRID[0], LAMBDA_GRID[-1])
            if edge:
                log.warning("  penalty for %s is on the edge of the grid: %s", grp, (lp, lb))
    return chosen, pd.concat(grids, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------
FIXED_BLOCKS = ("season", "same_hand", "home")


def fit_all(designs, Ys, keys, chosen, weights=None, scale=1.0):
    """Fit every outcome; returns {key: beta} (full coefficient vectors)."""
    betas = {}
    for sample in ("pa", "bip"):
        d = designs[sample]
        w = None if weights is None else weights[sample]
        G = ridge.gram(d.X, w)
        B = ridge.xty(d.X, Ys[sample], w)
        for grp, members in groups_for(sample).items():
            lp, lb = chosen[grp]
            idx = [keys[sample].index(m) for m in members]
            beta = ridge.solve(G, B[:, idx], d.penalty_vector({"pitcher": lp * scale,
                                                              "batter": lb * scale}))
            for j, m in enumerate(members):
                betas[m] = beta[:, j]
    return betas


def summarize_betas(designs, betas, weights=None):
    """Centered park effects and fixed coefficients from fitted coefficient vectors."""
    parks, fixed = {}, {}
    for key, beta in betas.items():
        d = designs[OUTCOMES[key][0]]
        w = None if weights is None else weights[OUTCOMES[key][0]]
        counts = ridge.block_counts(d, "park", w)
        parks[key] = ridge.center(d.coef(beta, "park").to_numpy(), counts)
        fixed[key] = np.concatenate([beta[d.slices[b]] for b in FIXED_BLOCKS if b in d.slices])
    return parks, fixed


def fixed_names(design) -> list[str]:
    names = []
    for b in FIXED_BLOCKS:
        if b in design.slices:
            names += [f"{b}:{lev}" for lev in design.levels[b]]
    return names


def _boot_chunk(seeds, designs, Ys, keys, chosen, games, strata, row_games):
    from threadpoolctl import threadpool_limits

    out = []
    with threadpool_limits(limits=2):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            counts = ridge.stratified_game_weights(games, strata, rng)
            weights = {s: ridge.row_weights(row_games[s], counts) for s in ("pa", "bip")}
            betas = fit_all(designs, Ys, keys, chosen, weights)
            out.append(summarize_betas(designs, betas, weights))
    return out


def bootstrap(designs, Ys, keys, chosen, data, n_boot, n_jobs=8):
    g = data["pa"].drop_duplicates("game_pk")
    games = g["game_pk"].to_numpy()
    strata = (g["season"].astype(str) + "_" + g["venue_id"].astype(str)).to_numpy()
    row_games = {s: data[s]["game_pk"].to_numpy() for s in ("pa", "bip")}
    seeds = [RANDOM_SEED + 1 + b for b in range(n_boot)]
    chunks = [seeds[i::n_jobs * 4] for i in range(n_jobs * 4)]
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(c, designs, Ys, keys, chosen, games,
                                                       strata, row_games) for c in chunks)
    flat = [r for chunk in res for r in chunk]
    parks = {k: np.stack([r[0][k] for r in flat]) for k in OUTCOMES}
    fixed = {k: np.stack([r[1][k] for r in flat]) for k in OUTCOMES}
    return parks, fixed


def run_fit(pa: pd.DataFrame, spec: FitSpec, n_boot: int, chosen=None, scales=()):
    log.info("-" * 70)
    log.info("Fit '%s': seasons %s, park key '%s'", spec.name, spec.seasons, spec.park_col)
    data = samples(pa, spec)
    multi = len(spec.seasons) > 1
    designs = {s: ridge.build_design(data[s], ridge.standard_blocks(multi, "park_key"))
               for s in ("pa", "bip")}
    keys = {s: [k for k, (smp, _, _) in OUTCOMES.items() if smp == s] for s in ("pa", "bip")}
    Ys = {s: outcome_matrix(data[s], keys[s], spec) for s in ("pa", "bip")}
    for s in ("pa", "bip"):
        lv = designs[s].levels
        log.info("  %s sample: %d rows, %d games; %d parks, %d pitchers, %d batters", s,
                 len(data[s]), data[s]["game_pk"].nunique(), len(lv["park"]),
                 len(lv["pitcher"]), len(lv["batter"]))

    cv = None
    if chosen is None:
        chosen, cv = choose_penalties(designs, data, spec)
    for grp, (lp, lb) in chosen.items():
        log.info("  penalty %-9s lambda_pitcher = %6g, lambda_batter = %6g", grp, lp, lb)

    betas = fit_all(designs, Ys, keys, chosen)
    point_parks, point_fixed = summarize_betas(designs, betas)
    extra = {}
    for sc in scales:
        extra[sc] = summarize_betas(designs, fit_all(designs, Ys, keys, chosen, scale=sc))[0]

    boot_parks = boot_fixed = None
    if n_boot:
        t0 = time.time()
        boot_parks, boot_fixed = bootstrap(designs, Ys, keys, chosen, data, n_boot)
        log.info("  bootstrap: %d resamples in %.0fs", n_boot, time.time() - t0)

    park_index = designs["pa"].levels["park"]
    assert list(park_index) == list(designs["bip"].levels["park"]), "park levels differ"
    counts = {s: ridge.block_counts(designs[s], "park") for s in ("pa", "bip")}
    return dict(spec=spec, data=data, designs=designs, betas=betas, chosen=chosen, cv=cv,
                parks=park_index, counts=counts, point_parks=point_parks,
                point_fixed=point_fixed, boot_parks=boot_parks, boot_fixed=boot_fixed,
                fixed_names=fixed_names(designs["pa"]), scaled=extra)


# --------------------------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------------------------
def venue_info(pa: pd.DataFrame) -> pd.DataFrame:
    v = pd.read_csv(OUT_TABLES / "venues.csv")
    v["park_key"] = v["venue_id"].astype(str)
    return v.set_index("park_key")[["venue_short", "home_team", "home_seasons"]]


def effects_table(res, vinfo) -> pd.DataFrame:
    parks = res["parks"]
    t = pd.DataFrame({"fit": res["spec"].name, "park_key": parks,
                      "pa": res["counts"]["pa"].astype(int), "bip": res["counts"]["bip"].astype(int)})
    t = t.join(vinfo, on="park_key")
    t.loc[t["park_key"] == OTHER_PARK, "venue_short"] = "Other (neutral sites)"
    seasons_used = (res["data"]["pa"].groupby("park_key")["season"]
                    .agg(lambda s: ",".join(map(str, sorted(s.unique())))))
    t["seasons_used"] = t["park_key"].map(seasons_used)
    for k in OUTCOMES:
        t[f"{k}"] = res["point_parks"][k]
        if res["boot_parks"] is not None:
            t[f"{k}_lo"] = np.percentile(res["boot_parks"][k], 2.5, axis=0)
            t[f"{k}_hi"] = np.percentile(res["boot_parks"][k], 97.5, axis=0)
            t[f"{k}_se"] = res["boot_parks"][k].std(axis=0, ddof=1)
    return t


def fixed_table(res) -> pd.DataFrame:
    rows = []
    for k in OUTCOMES:
        for j, name in enumerate(res["fixed_names"]):
            r = {"fit": res["spec"].name, "outcome": k, "term": name,
                 "estimate": res["point_fixed"][k][j]}
            if res["boot_fixed"] is not None:
                draws = res["boot_fixed"][k][:, j]
                r.update(lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5),
                         se=draws.std(ddof=1))
            rows.append(r)
    return pd.DataFrame(rows)


def save_npz(res) -> None:
    path = DATA_PROCESSED / f"park_effects_{res['spec'].name}.npz"
    arrays = {"parks": np.asarray(res["parks"], dtype=str), "outcomes": np.asarray(list(OUTCOMES)),
              "fixed_names": np.asarray(res["fixed_names"], dtype=str),
              "counts_pa": res["counts"]["pa"], "counts_bip": res["counts"]["bip"],
              "point": np.stack([res["point_parks"][k] for k in OUTCOMES], axis=1),
              "point_fixed": np.stack([res["point_fixed"][k] for k in OUTCOMES], axis=1)}
    if res["boot_parks"] is not None:
        arrays["boot"] = np.stack([res["boot_parks"][k] for k in OUTCOMES], axis=2)
        arrays["boot_fixed"] = np.stack([res["boot_fixed"][k] for k in OUTCOMES], axis=2)
    for sc, parks in res["scaled"].items():
        arrays[f"point_scale_{sc:g}"] = np.stack([parks[k] for k in OUTCOMES], axis=1)
    np.savez_compressed(path, **arrays)


def player_effects(res) -> pd.DataFrame:
    rows = []
    for k, beta in res["betas"].items():
        d = res["designs"][OUTCOMES[k][0]]
        for role in ("pitcher", "batter"):
            s = d.coef(beta, role)
            rows.append(pd.DataFrame({"outcome": k, "role": role, "player": s.index.astype(int),
                                      "effect": s.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    pa = add_literal_columns(pd.read_parquet(DATA_PROCESSED / "pa.parquet"))
    vinfo = venue_info(pa)

    results = {}
    results["pooled"] = run_fit(pa, FITS["pooled"], args.n_boot, scales=(0.25, 4.0))
    results["2026"] = run_fit(pa, FITS["2026"], args.n_boot)
    # Literal-columns sensitivity: same penalties as the pooled fit, point estimates only.
    results["pooled_literal"] = run_fit(pa, FITS["pooled_literal"], 0,
                                        chosen=results["pooled"]["chosen"])

    # ---- Save ------------------------------------------------------------------------------
    eff = pd.concat([effects_table(results[f], vinfo) for f in ("pooled", "2026")], ignore_index=True)
    eff.to_csv(OUT_TABLES / "park_channel_effects.csv", index=False)
    pd.concat([fixed_table(results[f]) for f in ("pooled", "2026")]).to_csv(
        OUT_TABLES / "model_fixed_effects.csv", index=False)
    pd.concat([results[f]["cv"] for f in ("pooled", "2026")]).to_csv(OUT_TABLES / "model_cv.csv", index=False)
    pen = []
    for f in ("pooled", "2026"):
        cv = results[f]["cv"]
        for grp, (lp, lb) in results[f]["chosen"].items():
            members = [k for k, v in OUTCOMES.items() if v[2] == grp]
            for m in members:
                row = cv[(cv["outcome"] == m) & (cv["lambda_pitcher"] == lp) & (cv["lambda_batter"] == lb)]
                own = cv[cv["outcome"] == m].sort_values("mse").iloc[0]
                pen.append({"fit": f, "outcome": m, "penalty_group": grp, "lambda_pitcher": lp,
                            "lambda_batter": lb, "cv_mse": row["mse"].iloc[0],
                            "own_best_lambda_pitcher": own["lambda_pitcher"],
                            "own_best_lambda_batter": own["lambda_batter"], "own_best_mse": own["mse"]})
    pen = pd.DataFrame(pen)
    pen.to_csv(OUT_TABLES / "model_penalties.csv", index=False)
    for f in ("pooled", "2026", "pooled_literal"):
        save_npz(results[f])
    write_parquet_atomic(player_effects(results["pooled"]), DATA_PROCESSED / "player_effects_pooled.parquet")

    # Sensitivity: penalties x0.25 / x4 and literal Statcast columns, vs the pooled fit.
    base = results["pooled"]
    sens = []
    for label, parks_dict in (("lambda_x0.25", base["scaled"][0.25]), ("lambda_x4", base["scaled"][4.0]),
                              ("literal_columns", results["pooled_literal"]["point_parks"])):
        for k in OUTCOMES:
            a, b = base["point_parks"][k], parks_dict[k]
            mod = np.asarray(base["parks"]) != OTHER_PARK
            sens.append({"variant": label, "outcome": k, "corr": np.corrcoef(a[mod], b[mod])[0, 1],
                         "max_abs_change": np.abs(a[mod] - b[mod]).max(),
                         "sd_base": a[mod].std(), "sd_variant": b[mod].std()})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT_TABLES / "park_effects_sensitivity.csv", index=False)

    # ---- Summary ---------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 4 summary")
    log_df(log, pen.round(5), "Penalties (lambda in plate appearances / balls in play) and held-out MSE:",
           index=False)
    for f in ("pooled", "2026"):
        t = eff[(eff["fit"] == f) & (eff["park_key"] != OTHER_PARK)]
        log.info("Fit '%s': %d parks + other", f, len(t))
        for k in ("k", "bbhbp", "bip", "xwobacon", "conv", "woba"):
            s = t.set_index("venue_short")[k].sort_values()
            log.info("  %-34s SD across parks %.4f | lowest: %s | highest: %s", LABELS[k], s.std(),
                     ", ".join(f"{i} {v:+.4f}" for i, v in s.head(2).items()),
                     ", ".join(f"{i} {v:+.4f}" for i, v in s.tail(3)[::-1].items()))
    fx = fixed_table(results["pooled"])
    log_df(log, fx[fx["term"].str.startswith(("home", "same_hand"))].round(4),
           "Home (pitcher's team at home) and platoon (same-handed) coefficients, pooled fit:", index=False)
    log_df(log, sens.round(4), "Sensitivity of pooled park effects (parks other than 'other'):", index=False)
    log.info("Saved park_channel_effects.csv, model_*.csv, park_effects_sensitivity.csv and "
             "data/processed/park_effects_*.npz")


if __name__ == "__main__":
    main()
