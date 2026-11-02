"""Step 6: mechanism checks -- why each channel moves.

Every check compares the same players across parks rather than raw park averages.

1. Pitch movement (strikeout/walk and contact channels). For each pitcher x season x pitch
   type, the pitch's movement at a venue (pfx_x, pfx_z, in inches), velocity and spin are
   compared with the same pitcher's same pitch thrown everywhere else that season
   (leave-one-venue-out). The differences are averaged per venue, weighted by pitches at the
   venue. Main metric: movement change along the pitch's usual movement direction; negative
   = the pitch moved less than usual. A swing-level model, built like Step 4, gives each
   park's contact rate per swing; parks that flatten pitches should show more contact and
   fewer strikeouts.
2. Ball carry (conversion channel). A gradient-boosted model predicts projected distance
   (hit_distance_sc) of every fly ball and line drive from exit velocity, launch angle, spray
   angle, batter side and season (5-fold cross-fitting by game). A park's carry is its average
   residual: how much farther or shorter the ball travels there for the same contact. A
   player-adjusted version fits the residual with the Step 4 structure.
3. Where extra hits land (conversion channel). Balls in play that beat their xwOBA by 0.5+
   (e.g. a single on a ball with xwOBA < 0.39, a home run on one with xwOBA < 1.55) are mapped
   for selected parks, and their rate by field zone is compared with the league's.
4. Home-field effect. The Step 4 home coefficients are converted into wOBA points and
   compared with park effects. "Coors hangover" test: the pooled wOBA model with the single
   home term replaced by team-specific home terms, for pitchers and separately for batters.
   The Rockies' term is compared with the other 29 teams'.

Outputs (outputs/tables/): mechanism_pitch_movement.csv, mechanism_contact_rate.csv,
mechanism_carry.csv, mechanism_overperformer_zones.csv, mechanism_home_field.csv,
mechanism_team_home_terms.csv, mechanism_correlations.csv; and
data/processed/spray_selected.parquet for the spray charts.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import ridge
from .common import get_logger, log_df, write_parquet_atomic
from .config import DATA_PROCESSED, OTHER_PARK, OUT_TABLES, RANDOM_SEED, SEASONS
from .decompose import baseline, decompose
from .fetch import load_statcast

log = get_logger("mechanisms", "step6_mechanisms.log")

PITCH_GROUP = {"FF": "fastball", "SI": "fastball", "FC": "fastball",
               "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking", "KC": "breaking",
               "CH": "offspeed", "FS": "offspeed", "FO": "offspeed"}
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}
MIN_ELSEWHERE = 50          # pitches of the same pitch type elsewhere that season
OVERPERFORM = 0.5           # wOBA value - xwOBA threshold for "much better than expected"
N_BOOT_SMALL = 200


def venue_table() -> pd.DataFrame:
    v = pd.read_csv(OUT_TABLES / "venues.csv")
    v["park_key"] = v["venue_id"].astype(str)
    return v


def weighted_mean_se(d: pd.DataFrame, col: str, w: str = "n") -> tuple[float, float]:
    x, wt = d[col].to_numpy(), d[w].to_numpy(float)
    if wt.sum() == 0:
        return np.nan, np.nan
    m = np.sum(wt * x) / wt.sum()
    se = np.sqrt(np.sum(wt ** 2 * (x - m) ** 2)) / wt.sum()
    return m, se


# --------------------------------------------------------------------------------------------
# 1. Pitch movement
# --------------------------------------------------------------------------------------------
def load_pitches(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_year", "pitcher", "batter", "pitch_type", "pfx_x", "pfx_z", "release_speed",
            "release_spin_rate", "description", "p_throws", "stand", "inning_topbot"]
    frames = [load_statcast(s, columns=cols, logger=log) for s in SEASONS]
    p = pd.concat(frames, ignore_index=True)
    p = p.merge(games[["game_pk", "venue_id", "venue_short", "park", "is_neutral"]], on="game_pk")
    p["season"] = p["game_year"].astype(int)
    return p


def pitch_movement(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = p[p["pitch_type"].isin(PITCH_GROUP.keys())].dropna(subset=["pfx_x", "pfx_z", "release_speed"]).copy()
    dropped = len(p) - len(m)
    log.info("Pitch movement: %d pitches with movement data (%d excluded: missing tracking or "
             "rare pitch types such as eephus, knuckleball, position-player lobs)", len(m), dropped)
    m["x_in"], m["z_in"] = 12 * m["pfx_x"], 12 * m["pfx_z"]
    m["spin_ok"] = m["release_spin_rate"].notna()
    m["spin"] = m["release_spin_rate"].fillna(0.0)
    key = ["pitcher", "season", "pitch_type"]
    g = (m.groupby(key + ["venue_id"])
         .agg(n=("x_in", "size"), sx=("x_in", "sum"), sz=("z_in", "sum"), sv=("release_speed", "sum"),
              ss=("spin", "sum"), ns=("spin_ok", "sum")).reset_index())
    tot = g.groupby(key)[["n", "sx", "sz", "sv", "ss", "ns"]].transform("sum")
    ne = tot["n"] - g["n"]
    g = g[ne >= MIN_ELSEWHERE].copy()
    tot, ne = tot.loc[g.index], ne[ne >= MIN_ELSEWHERE]
    ex, ez = (tot["sx"] - g["sx"]) / ne, (tot["sz"] - g["sz"]) / ne
    g["dx"] = g["sx"] / g["n"] - ex
    g["dz"] = g["sz"] / g["n"] - ez
    norm = np.hypot(ex, ez)
    g["d_toward"] = (g["dx"] * ex + g["dz"] * ez) / norm       # along the usual direction
    g["d_vert"] = np.sign(ez) * g["dz"]                        # vertical, away from zero
    g["d_horiz"] = np.sign(ex) * g["dx"]                       # horizontal, away from zero
    g["d_velo"] = g["sv"] / g["n"] - (tot["sv"] - g["sv"]) / ne
    nse = tot["ns"] - g["ns"]
    g["d_spin"] = np.where((g["ns"] > 0) & (nse > 0), g["ss"] / g["ns"].clip(lower=1)
                           - (tot["ss"] - g["ss"]) / nse.clip(lower=1), np.nan)
    g["group"] = g["pitch_type"].map(PITCH_GROUP)
    g["ivb_ff"] = np.where(g["pitch_type"] == "FF", g["dz"], np.nan)

    rows = []
    for vid, d in g.groupby("venue_id"):
        r = {"venue_id": vid, "pitches": int(d["n"].sum()), "pitcher_pitch_groups": len(d)}
        for col in ("d_toward", "d_vert", "d_horiz", "d_velo"):
            r[col], r[col + "_se"] = weighted_mean_se(d, col)
        ds = d.dropna(subset=["d_spin"])
        r["d_spin"], r["d_spin_se"] = weighted_mean_se(ds, "d_spin")
        for grp in ("fastball", "breaking", "offspeed"):
            dg = d[d["group"] == grp]
            r[f"d_toward_{grp}"] = weighted_mean_se(dg, "d_toward")[0] if len(dg) else np.nan
        ff = d.dropna(subset=["ivb_ff"])
        r["d_ivb_fourseam"] = weighted_mean_se(ff, "ivb_ff")[0] if len(ff) else np.nan
        rows.append(r)
    return pd.DataFrame(rows), g


def contact_model(p: pd.DataFrame) -> pd.DataFrame:
    """Park effects on contact per swing (foul, foul tip or in play vs. swing and miss)."""
    sw = p[p["description"].isin(SWING)].copy()
    sw["contact"] = (~sw["description"].isin(WHIFF)).astype(float)
    sw["same_hand"] = (sw["p_throws"] == sw["stand"]).astype(np.int8)
    sw["pitcher_is_home"] = (sw["inning_topbot"] == "Top").astype(np.int8)
    sw = sw.reset_index(drop=True)
    d = ridge.build_design(sw, ridge.standard_blocks(True, "park"))
    y = sw["contact"].to_numpy()
    grid = [30, 100, 300, 1000, 3000]
    folds = ridge.game_folds(sw["game_pk"].to_numpy(), 5, RANDOM_SEED)
    cv = ridge.cv_grid(d, y, folds, grid, grid, ["contact"])
    best = cv.sort_values("mse").iloc[0]
    lams = {"pitcher": best["lambda_pitcher"], "batter": best["lambda_batter"]}
    beta = ridge.fit(d, y, lams)
    eff = ridge.center(d.coef(beta, "park").to_numpy(), ridge.block_counts(d, "park"))
    log.info("Contact-per-swing model: %d swings, league contact rate %.3f, CV penalties %s",
             len(sw), y.mean(), lams)
    return pd.DataFrame({"park_key": d.levels["park"], "contact_per_swing": eff,
                         "swings": ridge.block_counts(d, "park").astype(int)})


# --------------------------------------------------------------------------------------------
# 2. Ball carry
# --------------------------------------------------------------------------------------------
def carry(pa: pd.DataFrame, lams_woba) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    fb = pa[pa["in_woba"] & pa["bb_type"].isin(["fly_ball", "line_drive"])].dropna(
        subset=["launch_speed", "launch_angle", "spray_angle", "hit_distance_sc"]).copy()
    n_all = int((pa["in_woba"] & pa["bb_type"].isin(["fly_ball", "line_drive"])).sum())
    log.info("Carry model: %d fly balls and line drives with EV, LA, spray angle and distance "
             "(%d excluded for missing tracking)", len(fb), n_all - len(fb))
    fb["stand_r"] = (fb["stand"] == "R").astype(int)
    feats = ["launch_speed", "launch_angle", "spray_angle", "stand_r", "season"]
    y = fb["hit_distance_sc"].to_numpy(float)
    oof = np.zeros(len(fb))
    oof_base = np.zeros(len(fb))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(fb, groups=fb["game_pk"]):
        mdl = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                                            min_samples_leaf=40, random_state=RANDOM_SEED)
        mdl.fit(fb.iloc[tr][feats], y[tr])
        oof[te] = mdl.predict(fb.iloc[te][feats])
        base = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_SEED)
        base.fit(fb.iloc[tr][["launch_speed", "launch_angle"]], y[tr])
        oof_base[te] = base.predict(fb.iloc[te][["launch_speed", "launch_angle"]])
    fb["carry_resid"] = y - oof
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    r2b = 1 - np.sum((y - oof_base) ** 2) / np.sum((y - y.mean()) ** 2)
    fitq = {"n": len(fb), "r2": r2, "rmse_ft": float(np.sqrt(np.mean((y - oof) ** 2))),
            "mae_ft": float(np.mean(np.abs(y - oof))), "r2_ev_la_only": r2b}
    log.info("  out-of-fold fit: R^2 = %.3f, RMSE = %.1f ft, MAE = %.1f ft (EV + LA only: R^2 = %.3f)",
             r2, fitq["rmse_ft"], fitq["mae_ft"], r2b)

    # Raw park average residual.
    raw = fb.groupby("park").agg(n=("carry_resid", "size"), carry_raw=("carry_resid", "mean"),
                                 sd=("carry_resid", "std"))
    raw["carry_raw_se"] = raw["sd"] / np.sqrt(raw["n"])
    # Player-adjusted: residual ~ park + season + platoon + home + pitcher + batter.
    fb = fb.reset_index(drop=True)
    d = ridge.build_design(fb, ridge.standard_blocks(True, "park"))
    folds = ridge.game_folds(fb["game_pk"].to_numpy(), 5, RANDOM_SEED)
    grid = [100, 300, 1000, 3000, 10000]
    cv = ridge.cv_grid(d, fb["carry_resid"].to_numpy(), folds, grid, grid, ["carry"])
    best = cv.sort_values("mse").iloc[0]
    lams = {"pitcher": best["lambda_pitcher"], "batter": best["lambda_batter"]}
    yres = fb["carry_resid"].to_numpy()
    beta = ridge.fit(d, yres, lams)
    counts = ridge.block_counts(d, "park")
    adj = ridge.center(d.coef(beta, "park").to_numpy(), counts)
    # Game bootstrap for the adjusted estimate.
    games = fb.drop_duplicates("game_pk")
    strata = (games["season"].astype(str) + "_" + games["venue_id"].astype(str)).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    G0 = None
    draws = []
    for _ in range(N_BOOT_SMALL):
        cnt = ridge.stratified_game_weights(games["game_pk"].to_numpy(), strata, rng)
        w = ridge.row_weights(fb["game_pk"].to_numpy(), cnt)
        b = ridge.fit(d, yres, lams, w=w)
        draws.append(ridge.center(d.coef(b, "park").to_numpy(), ridge.block_counts(d, "park", w)))
    draws = np.array(draws)
    out = pd.DataFrame({"park_key": d.levels["park"], "carry_adj": adj,
                        "carry_adj_lo": np.percentile(draws, 2.5, axis=0),
                        "carry_adj_hi": np.percentile(draws, 97.5, axis=0),
                        "carry_adj_se": draws.std(axis=0, ddof=1)})
    out = out.join(raw[["n", "carry_raw", "carry_raw_se"]], on="park_key")
    log.info("  player-adjusted carry: CV penalties %s, %d bootstrap resamples", lams, N_BOOT_SMALL)
    return out, fitq, fb


# --------------------------------------------------------------------------------------------
# 3. Where extra hits land
# --------------------------------------------------------------------------------------------
def zone_of(spray, dist):
    direction = pd.cut(spray, [-90, -27, -9, 9, 27, 90],
                       labels=["LF", "LCF", "CF", "RCF", "RF"])
    depth = pd.cut(dist, [-1, 150, 250, 330, 1000], labels=["infield (<150 ft)", "shallow OF (150-250)",
                                                            "deep OF (250-330)", "wall/HR (330+)"])
    return direction, depth


def overperformers(pa: pd.DataFrame, selected: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pa[pa["in_woba"] & pa["is_bip"] & pa["hc_x"].notna()].copy()
    b["over"] = b["conv"] >= OVERPERFORM
    b["dist_ft"] = np.hypot(b["x_ft"], b["y_ft"])
    b["direction"], b["depth"] = zone_of(b["spray_angle"], b["dist_ft"])
    league = b.groupby(["direction", "depth"], observed=True)["over"].agg(["mean", "size"])
    rows = []
    for pk in selected:
        s = b[b["park"] == pk]
        z = s.groupby(["direction", "depth"], observed=True)["over"].agg(["mean", "size", "sum"])
        z = z.join(league, rsuffix="_league")
        z["park_key"] = pk
        rows.append(z.reset_index())
    zones = pd.concat(rows, ignore_index=True).rename(columns={
        "mean": "over_rate", "size": "bip", "sum": "overperformers",
        "mean_league": "over_rate_league", "size_league": "bip_league"})
    zones["lift"] = zones["over_rate"] / zones["over_rate_league"]
    zones["excess_overperformers"] = zones["overperformers"] - zones["bip"] * zones["over_rate_league"]
    sel = b[b["park"].isin(selected)][["park", "venue_short", "season", "events", "bb_type", "xwoba",
                                        "woba_val", "conv", "over", "x_ft", "y_ft", "spray_angle",
                                        "dist_ft", "stand", "launch_speed", "launch_angle"]]
    rates = b.groupby("park")["over"].mean()
    log.info("Over-performing balls in play (wOBA value - xwOBA >= %.1f): league %.2f%% of balls in "
             "play; selected parks: %s", OVERPERFORM, 100 * b["over"].mean(),
             ", ".join(f"{pk} {100 * rates[pk]:.2f}%" for pk in selected))
    return zones, sel


# --------------------------------------------------------------------------------------------
# 4. Home field and the Coors hangover
# --------------------------------------------------------------------------------------------
def team_home_terms(pa: pd.DataFrame, side: str, lams: dict) -> pd.DataFrame:
    """wOBA model with team-specific home terms for pitchers (side='pitcher') or batters."""
    d = pa[pa["in_woba"]].copy().reset_index(drop=True)
    if side == "pitcher":
        d["home_term"] = d["pitcher_team_id"].where(d["pitcher_is_home"] == 1)
    else:
        d["home_term"] = d["batter_team_id"].where(d["pitcher_is_home"] == 0)
    blocks = [ridge.Block("park", "park"), ridge.Block("season", "season", drop_first=True),
              ridge.Block("same_hand", "same_hand", kind="numeric"),
              ridge.Block("home_team", "home_term", allow_missing=True),
              ridge.Block("pitcher", "pitcher", penalty="pitcher"),
              ridge.Block("batter", "batter", penalty="batter")]
    des = ridge.build_design(d, blocks)
    y = d["woba_val"].to_numpy()
    beta = ridge.fit(des, y, lams)
    point = des.coef(beta, "home_team").to_numpy()
    games = d.drop_duplicates("game_pk")
    strata = (games["season"].astype(str) + "_" + games["venue_id"].astype(str)).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED + (0 if side == "pitcher" else 1))
    G = []
    for _ in range(N_BOOT_SMALL):
        cnt = ridge.stratified_game_weights(games["game_pk"].to_numpy(), strata, rng)
        w = ridge.row_weights(d["game_pk"].to_numpy(), cnt)
        G.append(des.coef(ridge.fit(des, y, lams, w=w), "home_team").to_numpy())
    G = np.array(G)
    teams = des.levels["home_team"].astype(int)
    out = pd.DataFrame({"side": side, "team_id": teams, "home_term": point,
                        "lo": np.percentile(G, 2.5, axis=0), "hi": np.percentile(G, 97.5, axis=0)})
    # Each team vs the average of the other 29, with bootstrap intervals.
    for j in range(len(teams)):
        others = np.delete(np.arange(len(teams)), j)
        diff = G[:, j] - G[:, others].mean(axis=1)
        out.loc[j, "vs_others"] = point[j] - point[others].mean()
        out.loc[j, "vs_others_lo"], out.loc[j, "vs_others_hi"] = np.percentile(diff, [2.5, 97.5])
    return out


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def corr_row(label, a, b, names=None, drop=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    r = {"comparison": label, "n_parks": int(ok.sum()),
         "pearson": stats.pearsonr(a[ok], b[ok])[0], "pearson_p": stats.pearsonr(a[ok], b[ok])[1],
         "spearman": stats.spearmanr(a[ok], b[ok])[0]}
    if drop is not None:
        keep = ok & (np.asarray(names) != drop)
        r[f"pearson_without_{drop.split()[0].lower()}"] = stats.pearsonr(a[keep], b[keep])[0]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-hangover", action="store_true",
                    help="reuse outputs/tables/mechanism_team_home_terms.csv instead of refitting the "
                         "team-specific home-term models (2 x 200 bootstrap refits, the slowest part)")
    args = ap.parse_args(argv)
    t0 = time.time()
    games = pd.read_parquet(DATA_PROCESSED / "games.parquet")
    pa = pd.read_parquet(DATA_PROCESSED / "pa.parquet")
    vt = venue_table()
    names = vt.set_index("venue_id")["venue_short"]
    eff = pd.read_csv(OUT_TABLES / "park_channel_effects.csv")
    eff = eff[eff["fit"] == "pooled"].set_index("park_key")
    dec = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    dec = dec[dec["fit"] == "pooled"].set_index("park_key")
    pens = pd.read_csv(OUT_TABLES / "model_penalties.csv")
    pw = pens[(pens["fit"] == "pooled") & (pens["outcome"] == "woba")].iloc[0]
    lams_woba = {"pitcher": pw["lambda_pitcher"], "batter": pw["lambda_batter"]}

    # ---- 1. Pitch movement --------------------------------------------------------------------
    p = load_pitches(games)
    mv, mv_groups = pitch_movement(p)
    mv["venue_short"] = mv["venue_id"].map(names)
    mv["park_key"] = mv["venue_id"].astype(str)
    mv["modeled_park"] = mv["park_key"].isin(eff.index)
    contact = contact_model(p)
    mv = mv.merge(contact, on="park_key", how="left")
    mv = mv.join(eff[["k", "bip", "bbhbp"]], on="park_key")
    mv = mv.join(dec[["kbb_pts", "contact_pts", "conv_pts", "total_pts"]], on="park_key")
    mv.sort_values("d_toward").to_csv(OUT_TABLES / "mechanism_pitch_movement.csv", index=False)
    contact.to_csv(OUT_TABLES / "mechanism_contact_rate.csv", index=False)
    del p

    # ---- 2. Carry -------------------------------------------------------------------------------
    car, fitq, fb = carry(pa, lams_woba)
    car["venue_short"] = car["park_key"].map(eff["venue_short"])
    car = car.join(eff[["conv", "hr", "xwobacon"]], on="park_key").join(dec[["conv_pts", "total_pts"]], on="park_key")
    car.sort_values("carry_adj", ascending=False).to_csv(OUT_TABLES / "mechanism_carry.csv", index=False)
    pd.DataFrame([fitq]).to_csv(OUT_TABLES / "mechanism_carry_fit.csv", index=False)

    # ---- 3. Where extra hits land ----------------------------------------------------------------
    key_of = {v: k for k, v in eff["venue_short"].items()}
    mod = dec[dec.index != OTHER_PARK]
    must = ["Coors Field", "Fenway Park", "Yankee Stadium"]
    pos = [n for n in mod.sort_values("conv_pts", ascending=False)["venue_short"] if n not in must][:1]
    neg = mod.sort_values("conv_pts")["venue_short"].iloc[:1].tolist()
    chosen = must + pos + neg
    zones, sel = overperformers(pa, [key_of[n] for n in chosen])
    zones["venue_short"] = zones["park_key"].map(eff["venue_short"])
    zones.to_csv(OUT_TABLES / "mechanism_overperformer_zones.csv", index=False)
    write_parquet_atomic(sel, DATA_PROCESSED / "spray_selected.parquet")
    pd.Series(chosen, name="venue_short").to_csv(OUT_TABLES / "spray_chart_parks.csv", index=False)

    # ---- 4. Home field ---------------------------------------------------------------------------
    fx = pd.read_csv(OUT_TABLES / "model_fixed_effects.csv")
    fx = fx[(fx["fit"] == "pooled") & (fx["term"] == "home:pitcher_is_home")].set_index("outcome")
    base = baseline(pa, "pooled")
    h = base["home_share"]
    dhome = {"K": fx.loc["k", "estimate"], "W": fx.loc["bbhbp", "estimate"], "c": fx.loc["bip", "estimate"],
             "x": fx.loc["xwobacon", "estimate"], "v": fx.loc["conv", "estimate"]}
    road = dict(base, **{k: base[k] - dhome[k] * h for k in ("W", "c", "x", "v")})
    hd = decompose(road, dhome)   # home pitcher vs road pitcher, same park
    home_tab = pd.DataFrame([{**{f"{k}_pts": 1000 * v for k, v in hd.items()},
                              "direct_woba_model_pts": 1000 * fx.loc["woba", "estimate"],
                              "direct_lo": 1000 * fx.loc["woba", "lo"], "direct_hi": 1000 * fx.loc["woba", "hi"],
                              "park_total_sd_pts": mod["total_pts"].std(),
                              "park_total_range_pts": mod["total_pts"].max() - mod["total_pts"].min()}])
    home_tab.to_csv(OUT_TABLES / "mechanism_home_field.csv", index=False)

    hang_path = OUT_TABLES / "mechanism_team_home_terms.csv"
    if args.skip_hangover:
        if not hang_path.exists():
            raise SystemExit(f"--skip-hangover needs {hang_path.name} from an earlier full run of this step.")
        hang = pd.read_csv(hang_path)
        log.info("Coors hangover: reusing %s from an earlier full run (--skip-hangover)", hang_path.name)
    else:
        abbr = games.drop_duplicates("home_id", keep="last").set_index("home_id")["home_abbr"]
        hang = pd.concat([team_home_terms(pa, "pitcher", lams_woba), team_home_terms(pa, "batter", lams_woba)])
        hang["team"] = hang["team_id"].map(abbr)
        hang.to_csv(hang_path, index=False)

    # ---- Correlations ----------------------------------------------------------------------------
    m = mv[mv["modeled_park"] & (mv["park_key"] != OTHER_PARK)]
    c = car[car["park_key"] != OTHER_PARK]
    corr = pd.DataFrame([
        corr_row("movement change vs K-rate effect", m["d_toward"], m["k"], m["venue_short"], "Coors Field"),
        corr_row("movement change vs contact-per-swing effect", m["d_toward"], m["contact_per_swing"], m["venue_short"], "Coors Field"),
        corr_row("movement change vs ball-in-play-rate effect", m["d_toward"], m["bip"], m["venue_short"], "Coors Field"),
        corr_row("movement change vs strikeout/walk channel (pts)", m["d_toward"], m["kbb_pts"], m["venue_short"], "Coors Field"),
        corr_row("contact-per-swing effect vs K-rate effect", m["contact_per_swing"], m["k"], m["venue_short"], "Coors Field"),
        corr_row("carry (adjusted) vs conversion channel (pts)", c["carry_adj"], c["conv_pts"], c["venue_short"], "Coors Field"),
        corr_row("carry (adjusted) vs HR per BIP effect", c["carry_adj"], c["hr"], c["venue_short"], "Coors Field"),
        corr_row("carry (adjusted) vs xwOBAcon effect", c["carry_adj"], c["xwobacon"], c["venue_short"], "Coors Field"),
        corr_row("carry raw vs carry adjusted", c["carry_raw"], c["carry_adj"]),
    ])
    # ---- Physical attributes: altitude and fences (MLB Stats API, current values) -----------------
    attrs = vt.set_index("park_key")[["venue_short", "elevation_ft", "roof_type", "fence_mean_ft",
                                      "fence_leftLine", "fence_center", "fence_rightLine"]]
    pk = (attrs.join(mv.set_index("park_key")[["d_toward", "d_ivb_fourseam", "contact_per_swing"]])
          .join(car.set_index("park_key")[["carry_adj"]])
          .join(eff[["k", "bip", "hr", "xwobacon", "conv"]])
          .join(dec[["kbb_pts", "contact_pts", "conv_pts", "total_pts"]]))
    pk = pk[pk.index.isin(eff.index) & (pk.index != OTHER_PARK)]
    pk.to_csv(OUT_TABLES / "mechanism_park_attributes.csv")
    nm = pk["venue_short"]
    corr = pd.concat([corr, pd.DataFrame([
        corr_row("elevation vs movement change", pk["elevation_ft"], pk["d_toward"], nm, "Coors Field"),
        corr_row("elevation vs carry (adjusted)", pk["elevation_ft"], pk["carry_adj"], nm, "Coors Field"),
        corr_row("mean fence distance vs HR per BIP effect", pk["fence_mean_ft"], pk["hr"], nm, "Coors Field"),
        corr_row("mean fence distance vs conversion channel (pts)", pk["fence_mean_ft"], pk["conv_pts"], nm, "Coors Field"),
    ])], ignore_index=True)
    corr.to_csv(OUT_TABLES / "mechanism_correlations.csv", index=False)
    regs = []
    for label, formula in (("movement (in) ~ elevation (1,000 ft)", "d_toward ~ I(elevation_ft / 1000)"),
                           ("carry (ft) ~ elevation (1,000 ft)", "carry_adj ~ I(elevation_ft / 1000)"),
                           ("HR per BIP ~ carry (ft) + mean fence (ft)", "hr ~ carry_adj + fence_mean_ft"),
                           ("conversion (pts) ~ carry (ft) + mean fence (ft)", "conv_pts ~ carry_adj + fence_mean_ft"),
                           ("conversion (pts) ~ carry (ft)", "conv_pts ~ carry_adj")):
        fit = smf.ols(formula, data=pk).fit()
        for term in fit.params.index:
            regs.append({"model": label, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                         "p": fit.pvalues[term], "r2": fit.rsquared, "n_parks": int(fit.nobs)})
    regs = pd.DataFrame(regs)
    regs.to_csv(OUT_TABLES / "mechanism_regressions.csv", index=False)
    # How much of conversion does carry explain? Regression across parks.
    slope, intercept, r, pval, se = stats.linregress(c["carry_adj"], c["conv_pts"])
    c = c.assign(conv_pred_from_carry=intercept + slope * c["carry_adj"])
    c = c.assign(conv_not_carry=c["conv_pts"] - c["conv_pred_from_carry"])

    # ---- Summary ---------------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 6 summary")
    cols = ["venue_short", "pitches", "d_toward", "d_toward_se", "d_vert", "d_horiz", "d_ivb_fourseam",
            "d_velo", "d_spin", "contact_per_swing", "k"]
    log_df(log, mv.sort_values("d_toward")[cols].round(3).head(8),
           "1. Movement change vs the same pitcher's same pitch elsewhere (inches; negative = less "
           "movement). Most reduced:", index=False)
    log_df(log, mv.sort_values("d_toward")[cols].round(3).tail(4), "   ... most increased:", index=False)
    log_df(log, corr.round(3), "Correlations across the modeled parks:", index=False)
    log_df(log, regs.round(4), "Park-level regressions on physical attributes (Stats API elevation and "
           "fence distances; Steinbrenner Field lacks power-alley distances):", index=False)
    log.info("2. Carry: out-of-fold R^2 %.3f, RMSE %.1f ft", fitq["r2"], fitq["rmse_ft"])
    log_df(log, car.sort_values("carry_adj", ascending=False)[
        ["venue_short", "n", "carry_raw", "carry_adj", "carry_adj_lo", "carry_adj_hi", "conv_pts"]].round(2).head(6),
        "   Most carry (ft):", index=False)
    log_df(log, car.sort_values("carry_adj")[
        ["venue_short", "n", "carry_raw", "carry_adj", "carry_adj_lo", "carry_adj_hi", "conv_pts"]].round(2).head(4),
        "   Least carry (ft):", index=False)
    log.info("   Conversion (pts) regressed on carry across parks: slope %.2f pts per ft, R^2 %.2f", slope, r ** 2)
    log_df(log, c.sort_values("conv_not_carry", ascending=False)[
        ["venue_short", "conv_pts", "carry_adj", "conv_pred_from_carry", "conv_not_carry"]].round(1).head(5),
        "   Conversion carry does not explain (largest positive):", index=False)
    log_df(log, c.sort_values("conv_not_carry")[
        ["venue_short", "conv_pts", "carry_adj", "conv_pred_from_carry", "conv_not_carry"]].round(1).head(3),
        "   ... largest negative:", index=False)
    top_z = (zones[zones["bip"] >= 30].sort_values("excess_overperformers", ascending=False)
             .groupby("venue_short").head(2))
    log_df(log, top_z[["venue_short", "direction", "depth", "bip", "over_rate", "over_rate_league", "lift",
                       "excess_overperformers"]].round(3),
           "3. Zones with the most excess over-performing balls in play, per selected park:", index=False)
    log_df(log, home_tab.round(2), "4. Home field (home pitcher vs road pitcher in the same park), wOBA points:",
           index=False)
    for side in ("pitcher", "batter"):
        hs = hang[hang["side"] == side]
        col = hs[hs["team"] == "COL"].iloc[0]
        # A home edge is a lower wOBA allowed (pitchers) or a higher wOBA (batters).
        edge = -hs["vs_others"] if side == "pitcher" else hs["vs_others"]
        col_edge = -col["vs_others"] if side == "pitcher" else col["vs_others"]
        rank = int((edge > col_edge).sum()) + 1
        log.info("   Coors hangover, %s side: Rockies' home term %+.4f vs the other 29 teams' mean "
                 "%+.4f: difference %+.4f wOBA [95%% CI %+.4f, %+.4f]; Rockies rank %d of %d in "
                 "home edge", side, col["home_term"], col["home_term"] - col["vs_others"],
                 col["vs_others"], col["vs_others_lo"], col["vs_others_hi"], rank, len(hs))
    log.info("Step 6 done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
