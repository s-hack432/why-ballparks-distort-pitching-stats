"""Step 3: build the plate-appearance table and validate wOBA/xwOBA against Baseball Savant.

One row per plate appearance: the pitch whose `events` is not null. Truncated PAs (an inning
ended by a baserunning out during the PA; the batter bats again later) are not plate
appearances and are dropped, with the count logged.

wOBA follows Savant's leaderboard convention (see src/woba.py for the rules and why).

* Season linear weights. BB and HBP come from the values Savant writes into
  `estimated_woba_using_speedangle` for walks and HBP. 1B/2B/3B/HR are fitted by least squares
  to Savant's *batter* leaderboard. Pitchers are kept out of the fit and used to validate.
* Balls in play whose tracking failed have no Savant xwOBA:
    - EV and LA present (pitch untracked): an emulator of Savant's xwOBA from EV, LA and
      season, a gradient-boosted regression fit on tracked balls;
    - EV or LA missing: the mean xwOBA of tracked balls with the same season, batted-ball type
      and result (out / reached on error or FC / 1B / 2B / 3B / HR).
  Every ball is flagged in `xwoba_source`.
* Spray angle = atan2(hc_x - 125.42, 198.27 - hc_y) in degrees (0 = center field, negative =
  left field). Feet from home = scale * pixel offset. The scale is calibrated on fly-ball outs
  caught by outfielders (projected distance / pixel radius).

Outputs:
    data/processed/pa.parquet
    outputs/tables/woba_weights.csv
    outputs/tables/missing_tracking_by_park.csv
    outputs/tables/validation_savant_2026.csv (and _2024, _2025)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from . import savant
from .common import get_logger, log_df, write_parquet_atomic
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OTHER_PARK, OUT_TABLES, RANDOM_SEED, SEASONS
from .fetch import load_statcast
from .woba import (HIT_EVENTS, SAC_BUNT_EVENTS, WEIGHTED_EVENTS, field_feet, in_woba,
                   outcome_type, pull_angle, spray_angle, woba_values)

log = get_logger("prepare", "step3_prepare.log")

PA_COLS = [
    "game_pk", "game_date", "game_year", "inning", "inning_topbot", "at_bat_number",
    "pitch_number", "outs_when_up", "pitcher", "batter", "p_throws", "stand", "events", "des",
    "type", "bb_type", "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc", "hc_x", "hc_y",
    "hit_location", "release_speed", "n_thruorder_pitcher",
]


# --------------------------------------------------------------------------------------------
# 1. Raw plate appearances
# --------------------------------------------------------------------------------------------
def raw_pas(season: int) -> pd.DataFrame:
    sc = load_statcast(season, columns=PA_COLS, logger=log)
    pa = sc[sc["events"].notna()].copy()
    pa["outcome"] = outcome_type(pa["events"])
    n_trunc = int((pa["outcome"] == "NOT_PA").sum())
    pa = pa[pa["outcome"] != "NOT_PA"]
    log.info("%s: %d pitches -> %d plate appearances (%d truncated PAs dropped: an inning "
             "ended on a baserunning out mid-PA, so they are not PAs)", season, len(sc),
             len(pa), n_trunc)
    pa["season"] = pa["game_year"].astype(int)
    pa["in_woba"] = in_woba(pa["events"])
    pa["is_bip"] = pa["outcome"] == "BIP"
    return pa.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 2. Season linear weights
# --------------------------------------------------------------------------------------------
def event_counts(pa: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Per-player counts of each weighted event and of in-wOBA PAs, for one season."""
    d = pa[pa["in_woba"]]
    c = pd.crosstab(d[id_col], d["events"])
    out = pd.DataFrame(index=c.index)
    for e in WEIGHTED_EVENTS:
        out[e] = c[e] if e in c else 0
    out["denom"] = d.groupby(id_col).size()
    return out


def fit_season_weights(pas: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for season, pa in pas.items():
        # BB and HBP: the constant Savant stores as the xwOBA of every tracked walk / HBP.
        xw = pa.loc[pa["in_woba"], ["events", "estimated_woba_using_speedangle"]].dropna()
        bb = xw.loc[xw["events"] == "walk", "estimated_woba_using_speedangle"].round(4)
        hbp = xw.loc[xw["events"] == "hit_by_pitch", "estimated_woba_using_speedangle"].round(4)
        w_bb, w_hbp = bb.mode().iloc[0], hbp.mode().iloc[0]
        share_bb, share_hbp = (bb == w_bb).mean(), (hbp == w_hbp).mean()

        # Hits: least squares on the batter leaderboard, with BB/HBP fixed.
        lb = savant.expected_stats("batter", season).set_index("player_id")
        cnt = event_counts(pa, "batter").join(lb[["woba", "pa"]], how="inner")
        cnt = cnt[cnt["denom"] >= 50]
        sh = cnt[list(WEIGHTED_EVENTS)].div(cnt["denom"], axis=0)
        y = cnt["woba"] - sh["walk"] * w_bb - sh["hit_by_pitch"] * w_hbp
        hits = ["single", "double", "triple", "home_run"]
        coef, *_ = np.linalg.lstsq(sh[hits].to_numpy(), y.to_numpy(), rcond=None)
        resid = y - sh[hits].to_numpy() @ coef
        # Check: the same fit with BB/HBP free should land on Savant's BB/HBP constants.
        free, *_ = np.linalg.lstsq(sh[list(WEIGHTED_EVENTS)].to_numpy(), cnt["woba"].to_numpy(), rcond=None)
        rows.append({"season": season, "walk": w_bb, "hit_by_pitch": w_hbp,
                     **dict(zip(hits, np.round(coef, 4))),
                     "fit_batters": len(cnt), "fit_resid_sd": resid.std(),
                     "fit_max_abs_resid": resid.abs().max(),
                     "share_walks_at_constant": share_bb, "share_hbp_at_constant": share_hbp,
                     "free_fit_walk": free[0], "free_fit_hbp": free[1]})
    w = pd.DataFrame(rows).set_index("season")
    return w


# --------------------------------------------------------------------------------------------
# 3. xwOBA for balls in play without a Savant estimate
# --------------------------------------------------------------------------------------------
def result_group(events: pd.Series) -> pd.Series:
    g = pd.Series("out", index=events.index)
    for e in HIT_EVENTS:
        g[events == e] = e
    g[events.isin(["field_error", "fielders_choice"])] = "reached"
    return g


def fill_xwoba(pa: pd.DataFrame) -> pd.DataFrame:
    """Fill `xwoba` for every in-wOBA PA and record where each value came from."""
    pa["xwoba_sc"] = pa["estimated_woba_using_speedangle"]
    pa["xwoba"] = np.nan
    pa["xwoba_source"] = pd.NA
    nonbip = pa["in_woba"] & ~pa["is_bip"]
    # Non-contact PAs: their own linear-weight value (0 for K), exactly as Savant does.
    pa.loc[nonbip, "xwoba"] = pa.loc[nonbip, "woba_val"]
    pa.loc[nonbip, "xwoba_source"] = "event_value"

    bip = pa["in_woba"] & pa["is_bip"]
    has_sc = bip & pa["xwoba_sc"].notna()
    pa.loc[has_sc, "xwoba"] = pa.loc[has_sc, "xwoba_sc"]
    pa.loc[has_sc, "xwoba_source"] = "statcast"

    # (a) Emulator for balls with EV and LA but no Savant estimate.
    feats = ["launch_speed", "launch_angle", "season"]
    train = pa.loc[has_sc, feats + ["xwoba_sc"]].dropna()
    tr, te = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED)
    emu = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                        random_state=RANDOM_SEED)
    emu.fit(tr[feats], tr["xwoba_sc"])
    r2 = emu.score(te[feats], te["xwoba_sc"])
    rmse = float(np.sqrt(np.mean((emu.predict(te[feats]) - te["xwoba_sc"]) ** 2)))
    need_emu = bip & pa["xwoba_sc"].isna() & pa["launch_speed"].notna() & pa["launch_angle"].notna()
    if need_emu.any():
        pa.loc[need_emu, "xwoba"] = np.clip(emu.predict(pa.loc[need_emu, feats]), 0, None)
        pa.loc[need_emu, "xwoba_source"] = "emulated_from_ev_la"
    log.info("xwOBA emulator (EV, LA, season -> Savant xwOBA), 20%% holdout: R^2 = %.3f, "
             "RMSE = %.3f; used for %d balls whose pitch was untracked but EV/LA exist",
             r2, rmse, int(need_emu.sum()))

    # (b) Cell means for balls with no EV or LA.
    pa["result_group"] = np.where(pa["is_bip"], result_group(pa["events"]), None)
    cells = (pa[has_sc].groupby(["season", "bb_type", "result_group"])["xwoba_sc"].mean()
             .rename("cell_mean"))
    cells2 = pa[has_sc].groupby(["season", "result_group"])["xwoba_sc"].mean().rename("cell_mean2")
    need_cell = bip & pa["xwoba"].isna()
    sub = pa.loc[need_cell, ["season", "bb_type", "result_group"]]
    filled = sub.join(cells, on=["season", "bb_type", "result_group"]).join(
        cells2, on=["season", "result_group"])
    pa.loc[need_cell, "xwoba"] = filled["cell_mean"].fillna(filled["cell_mean2"]).to_numpy()
    pa.loc[need_cell, "xwoba_source"] = "cell_mean"
    log.info("Cell-mean xwOBA (season x batted-ball type x result) for %d balls with no EV/LA",
             int(need_cell.sum()))
    assert pa.loc[pa["in_woba"], "xwoba"].notna().all()
    return pa


# --------------------------------------------------------------------------------------------
# 4. Checks and reports
# --------------------------------------------------------------------------------------------
def calibrate_scale(pa: pd.DataFrame) -> tuple[float, int]:
    """Feet per Statcast pixel, from fly-ball outs caught by outfielders."""
    fb = pa[(pa["events"] == "field_out") & (pa["bb_type"] == "fly_ball")
            & pa["hit_location"].isin([7, 8, 9]) & pa["hit_distance_sc"].notna()
            & pa["hc_x"].notna() & pa["hc_y"].notna()]
    r = np.hypot(fb["hc_x"] - 125.42, 198.27 - fb["hc_y"])
    ratio = fb["hit_distance_sc"] / r
    return float(ratio.median()), len(fb)


def missing_tracking_report(pa: pd.DataFrame) -> pd.DataFrame:
    bip = pa[pa["in_woba"] & pa["is_bip"]].copy()
    bip["no_ev_la"] = bip["launch_speed"].isna() | bip["launch_angle"].isna()
    bip["no_xwoba"] = bip["xwoba_sc"].isna()
    t = (bip.groupby(["season", "park", "venue_short"])
         .agg(bip=("no_ev_la", "size"), no_ev_la=("no_ev_la", "sum"), no_xwoba=("no_xwoba", "sum"))
         .reset_index())
    t["pct_no_ev_la"] = 100 * t["no_ev_la"] / t["bip"]
    t["pct_no_xwoba"] = 100 * t["no_xwoba"] / t["bip"]
    return t.sort_values(["season", "pct_no_xwoba"], ascending=[True, False])


def validate(pa: pd.DataFrame, season: int) -> pd.DataFrame:
    """Pitcher wOBA/xwOBA vs Savant's pitcher leaderboard (PA >= 100 there)."""
    lb = savant.expected_stats("pitcher", season)
    d = pa[pa["season"] == season]
    w = d[d["in_woba"]]
    agg = w.groupby("pitcher").agg(n_woba=("woba_val", "size"), woba_sum=("woba_val", "sum"),
                                   xwoba_sum=("xwoba", "sum"))
    # Savant-style xwOBA: balls in play without a Savant estimate are left out entirely.
    sv = w[(w["xwoba_source"] != "emulated_from_ev_la") & (w["xwoba_source"] != "cell_mean")]
    agg2 = sv.groupby("pitcher").agg(n2=("xwoba", "size"), xs2=("xwoba", "sum"))
    agg = agg.join(agg2)
    agg["pa"] = d.groupby("pitcher").size()
    agg["bip"] = d[d["is_bip"]].groupby("pitcher").size()
    agg["imputed_bip"] = w[w["xwoba_source"].isin(["emulated_from_ev_la", "cell_mean"])].groupby("pitcher").size()
    agg["untracked_pa"] = d[d["release_speed"].isna()].groupby("pitcher").size()
    agg = agg.fillna({"bip": 0, "imputed_bip": 0, "untracked_pa": 0})
    agg["woba"] = agg["woba_sum"] / agg["n_woba"]
    agg["xwoba"] = agg["xwoba_sum"] / agg["n_woba"]
    agg["xwoba_savant_style"] = agg["xs2"] / agg["n2"]
    # Leaderboard columns pa/bip/woba get the suffix _savant; ours keep the plain names.
    m = lb.merge(agg, left_on="player_id", right_index=True, how="left",
                 suffixes=("_savant", ""))
    m = m.rename(columns={"est_woba": "xwoba_savant"})
    m["d_woba"] = m["woba"] - m["woba_savant"]
    m["d_xwoba"] = m["xwoba"] - m["xwoba_savant"]
    m["d_xwoba_savant_style"] = m["xwoba_savant_style"] - m["xwoba_savant"]
    m["pa_match"] = m["pa"] == m["pa_savant"]
    m["bip_match"] = m["bip"] == m["bip_savant"]
    keep = ["name", "player_id", "pa_savant", "pa", "pa_match", "bip_savant", "bip", "bip_match",
            "untracked_pa", "imputed_bip", "woba_savant", "woba", "d_woba", "xwoba_savant",
            "xwoba", "d_xwoba", "xwoba_savant_style", "d_xwoba_savant_style"]
    return m[keep].sort_values("pa_savant", ascending=False).reset_index(drop=True)


def summarize_validation(v: pd.DataFrame, season: int) -> None:
    q = v[v["pa_savant"] >= 100]
    log.info("Validation %s vs Savant pitcher leaderboard: %d pitchers with 100+ PA", season, len(q))
    log.info("  PA counts identical: %.1f%%; BIP counts identical: %.1f%%",
             100 * q["pa_match"].mean(), 100 * q["bip_match"].mean())
    for col, label in (("d_woba", "wOBA"), ("d_xwoba", "xwOBA (imputed balls included)"),
                       ("d_xwoba_savant_style", "xwOBA (Savant-style, imputed balls left out)")):
        a = q[col].abs()
        log.info("  %-46s same after rounding (|diff| <= 0.0005): %5.1f%%   within 0.001: %5.1f%%"
                 "   within 0.002: %5.1f%%   max |diff| %.4f", label, 100 * (a <= 0.0005 + 1e-9).mean(),
                 100 * (a <= 0.001 + 1e-9).mean(), 100 * (a <= 0.002 + 1e-9).mean(), a.max())
    cols = ["name", "pa_savant", "untracked_pa", "imputed_bip", "woba_savant", "woba", "d_woba",
            "xwoba_savant", "xwoba", "d_xwoba", "d_xwoba_savant_style"]
    log_df(log, q.reindex(q["d_woba"].abs().sort_values(ascending=False).index)[cols].head(6).round(4),
           "  Largest wOBA discrepancies:", index=False)
    log_df(log, q.reindex(q["d_xwoba"].abs().sort_values(ascending=False).index)[cols].head(6).round(4),
           "  Largest xwOBA discrepancies:", index=False)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main():
    games = pd.read_parquet(DATA_PROCESSED / "games.parquet")
    pas = {s: raw_pas(s) for s in SEASONS}

    # ---- Season weights ----------------------------------------------------------------------
    weights = fit_season_weights(pas)
    weights.to_csv(OUT_TABLES / "woba_weights.csv")
    log_df(log, weights.round(4).T, "Season wOBA weights (BB/HBP: Savant's xwOBA constants; "
           "hits: least squares on Savant's batter leaderboard, batters with 50+ wOBA PAs):")

    pa = pd.concat(pas.values(), ignore_index=True)
    pa["woba_val"] = woba_values(pa["events"], pa["season"], weights).where(pa["in_woba"], 0.0)

    # ---- Context: venue, park key, home/away, platoon ---------------------------------------
    g = games[["game_pk", "venue_id", "venue_short", "park", "park_2026", "is_neutral",
               "home_id", "away_id"]]
    pa = pa.merge(g, on="game_pk", how="left", validate="many_to_one")
    assert pa["venue_id"].notna().all()
    pa["pitcher_is_home"] = (pa["inning_topbot"] == "Top").astype(np.int8)
    pa["pitcher_team_id"] = np.where(pa["pitcher_is_home"] == 1, pa["home_id"], pa["away_id"])
    pa["batter_team_id"] = np.where(pa["pitcher_is_home"] == 1, pa["away_id"], pa["home_id"])
    pa["same_hand"] = (pa["p_throws"] == pa["stand"]).astype(np.int8)

    # ---- Outcome indicators ------------------------------------------------------------------
    pa["is_k"] = (pa["outcome"] == "K").astype(np.int8)
    pa["is_bbhbp"] = pa["outcome"].isin(["BB", "HBP"]).astype(np.int8)
    pa["is_hr"] = (pa["events"] == "home_run").astype(np.int8)
    pa["is_bip"] = pa["is_bip"].astype(bool)
    pa["pitch_untracked"] = pa["release_speed"].isna()
    pa["no_ev_la"] = pa["is_bip"] & (pa["launch_speed"].isna() | pa["launch_angle"].isna())

    pa = fill_xwoba(pa)
    pa["conv"] = np.where(pa["in_woba"] & pa["is_bip"], pa["woba_val"] - pa["xwoba"], np.nan)

    # ---- Geometry ----------------------------------------------------------------------------
    scale, n_scale = calibrate_scale(pa)
    pa["spray_angle"] = spray_angle(pa["hc_x"], pa["hc_y"])
    pa["pull_angle"] = pull_angle(pa["spray_angle"], pa["stand"])
    pa["x_ft"], pa["y_ft"] = field_feet(pa["hc_x"], pa["hc_y"], scale)
    log.info("Hit-coordinate scale: %.3f ft per pixel (median projected distance / pixel radius "
             "of %d fly-ball outs caught by outfielders)", scale, n_scale)

    keep = ["game_pk", "game_date", "season", "venue_id", "venue_short", "park", "park_2026",
            "is_neutral", "inning", "inning_topbot", "at_bat_number", "outs_when_up",
            "n_thruorder_pitcher", "pitcher", "batter", "p_throws", "stand", "same_hand",
            "pitcher_is_home", "pitcher_team_id", "batter_team_id", "events", "outcome",
            "in_woba", "is_k", "is_bbhbp", "is_bip", "is_hr", "woba_val", "xwoba",
            "xwoba_source", "conv", "result_group", "woba_value", "woba_denom", "xwoba_sc",
            "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc", "hc_x",
            "hc_y", "hit_location", "bb_type", "spray_angle", "pull_angle", "x_ft", "y_ft",
            "pitch_untracked", "no_ev_la", "release_speed", "des"]
    pa = pa[keep].rename(columns={"woba_value": "woba_value_sc", "woba_denom": "woba_denom_sc"})
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    write_parquet_atomic(pa, DATA_PROCESSED / "pa.parquet")

    # ---- Missing tracking --------------------------------------------------------------------
    miss = missing_tracking_report(pa)
    miss.to_csv(OUT_TABLES / "missing_tracking_by_park.csv", index=False)
    log.info("Balls in play (in wOBA) without EV/LA, by season:")
    bipw = pa[pa["in_woba"] & pa["is_bip"]]
    for s in SEASONS:
        b = bipw[bipw["season"] == s]
        t = miss[miss["season"] == s]
        chi = stats.chi2_contingency(np.c_[t["no_ev_la"], t["bip"] - t["no_ev_la"]])
        top = t.sort_values("pct_no_ev_la", ascending=False).head(3)
        log.info("  %s: %d of %d (%.2f%%) lack EV/LA, %d (%.2f%%) lack a Savant xwOBA. "
                 "Homogeneity across parks: chi2 = %.0f, p = %.1g. Highest: %s", s,
                 int(b["no_ev_la"].sum()), len(b), 100 * b["no_ev_la"].mean(),
                 int(b["xwoba_sc"].isna().sum()), 100 * b["xwoba_sc"].isna().mean(),
                 chi.statistic, chi.pvalue,
                 "; ".join(f"{r.venue_short} {r.pct_no_ev_la:.1f}%" for r in top.itertuples()))
    mb = bipw[bipw["no_ev_la"]]
    log.info("  Balls without EV/LA by batted-ball type: %s (all balls in play: %s)",
             mb["bb_type"].value_counts().to_dict(),
             bipw["bb_type"].value_counts(normalize=True).round(3).to_dict())
    log.info("  ... by result: %s", mb["result_group"].value_counts().to_dict())
    log.info("  Handling: kept in every model; xwOBA imputed (see xwoba_source). Dropping them "
             "would remove mostly ground-ball outs and bias the affected park's contact and wOBA "
             "upward (Great American Ball Park in 2026).")

    # ---- Validation against Savant -------------------------------------------------------------
    for s in SEASONS:
        v = validate(pa, s)
        v.to_csv(OUT_TABLES / f"validation_savant_{s}.csv", index=False)
        summarize_validation(v, s)

    # ---- Summary -----------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 3 summary")
    for s in SEASONS:
        d = pa[pa["season"] == s]
        w = d[d["in_woba"]]
        log.info("  %s: %d PAs; %d in wOBA (excluded: IBB %d, sac bunts %d, catcher's "
                 "interference %d). League wOBA %.4f, xwOBA %.4f; K %.3f, BB+HBP %.3f, BIP %.3f; "
                 "untracked pitches %d", s, len(d), len(w), (d["outcome"] == "IBB").sum(),
                 d["events"].isin(SAC_BUNT_EVENTS).sum(), (d["outcome"] == "CI").sum(),
                 w["woba_val"].mean(), w["xwoba"].mean(), w["is_k"].mean(), w["is_bbhbp"].mean(),
                 w["is_bip"].mean(), int(d["pitch_untracked"].sum()))
    src = pa.loc[pa["in_woba"] & pa["is_bip"], "xwoba_source"].value_counts().to_dict()
    log.info("  xwOBA source for balls in play: %s", src)
    log.info("  'other' park category: %d PAs (pooled)", int((pa["park"] == OTHER_PARK).sum()))
    log.info("Saved data/processed/pa.parquet")


if __name__ == "__main__":
    main()
