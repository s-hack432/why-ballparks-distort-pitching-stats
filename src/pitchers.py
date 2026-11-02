"""Step 7: pitcher-level analysis (2026).

Qualifying pitchers: 400+ batters faced for starters, 150+ for relievers. A pitcher is a starter
if at least half of their appearances were starts (they threw their team's first pitch).

For each pitcher:
* actual wOBA and xwOBA allowed (Savant leaderboard convention, Step 3);
* park exposure: share of the pitcher's wOBA plate appearances in each park;
* expected park distortion by channel: sum over parks of exposure share x the park's channel
  effect in wOBA points (Step 5 Shapley values, pooled 2024-2026 park effects).
  Home-team adjustment: a home game's park effect is evaluated from the baseline of a pitcher
  working at home, a road game's from the baseline of a visiting pitcher (league means shifted
  by the Step 4 home coefficient). Home-field advantage itself is not park distortion; it is
  reported separately (`home_field_pts`) and is about the same for everyone (~half of each
  pitcher's PAs are at home);
* park-neutral wOBA = actual wOBA - total expected distortion;
* rank change between actual and park-neutral wOBA, within role (positive = moves up);
* 95% intervals for the distortion from the Step 4 bootstrap draws.

Also: park-neutral xwOBA (xwOBA is not park-neutral either: it absorbs the strikeout/walk and
contact channels), three case studies whose distortions run through different channels, and a
brief check for Simpson's-paradox pairs across three park groups.

Outputs: outputs/tables/pitcher_distortion.csv, pitcher_case_studies.csv,
pitcher_exposure_2026.csv, simpson_pairs.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import get_logger, log_df
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OTHER_PARK, OUT_TABLES
from .decompose import baseline, decompose, deltas_from, home_road_baselines, load_effects
from .names import player_names

log = get_logger("pitchers", "step7_pitchers.log")

MIN_BF = {"SP": 400, "RP": 150}
CHANNELS = ("kbb", "contact", "conv")
PARTS = ("total", "kbb", "contact", "conv", "k_part", "bbhbp_part")


def roles_and_totals(pa26: pd.DataFrame, abbr: dict) -> pd.DataFrame:
    first = (pa26.sort_values(["game_pk", "inning_topbot", "at_bat_number"])
             .groupby(["game_pk", "inning_topbot"])["pitcher"].first())
    starts = first.value_counts().rename("starts")
    apps = pa26.groupby("pitcher")["game_pk"].nunique().rename("appearances")
    bf = pa26.groupby("pitcher").size().rename("bf")
    w = pa26[pa26["in_woba"]]
    agg = w.groupby("pitcher").agg(pa_woba=("woba_val", "size"), woba=("woba_val", "mean"),
                                   xwoba=("xwoba", "mean"), k_rate=("is_k", "mean"),
                                   bbhbp_rate=("is_bbhbp", "mean"), bip_rate=("is_bip", "mean"),
                                   home_share=("pitcher_is_home", "mean"))
    bip = w[w["is_bip"]].groupby("pitcher").agg(xwobacon=("xwoba", "mean"), wobacon=("woba_val", "mean"))
    teams = (pa26.groupby(["pitcher", "pitcher_team_id"]).size().rename("n").reset_index()
             .sort_values(["pitcher", "n"], ascending=[True, False]))
    teams["abbr"] = teams["pitcher_team_id"].map(abbr)
    team = teams.groupby("pitcher")["abbr"].agg(lambda s: "/".join(s)).rename("team")
    t = pd.concat([apps, starts, bf, agg, bip, team], axis=1)
    t["starts"] = t["starts"].fillna(0).astype(int)
    t["start_share"] = t["starts"] / t["appearances"]
    t["role"] = np.where(t["start_share"] >= 0.5, "SP", "RP")
    t["qualifies"] = t["bf"] >= t["role"].map(MIN_BF)
    return t


def channel_effects(pa: pd.DataFrame):
    """Per park: channel effects (wOBA units) for home and road pitchers, point and bootstrap."""
    eff = load_effects("pooled")
    outcomes = list(eff["outcomes"])
    base = baseline(pa, "pooled")
    home, road, coef = home_road_baselines(base, eff["fixed_names"], eff["point_fixed"], outcomes)
    out = {}
    for h, b in ((1, home), (0, road)):
        point = decompose(b, deltas_from(eff["point"], outcomes))
        boot = decompose(b, deltas_from(eff["boot"], outcomes))
        bx = dict(b, v=0.0)  # xwOBA: no conversion channel
        dx = deltas_from(eff["point"], outcomes)
        dx["v"] = np.zeros_like(dx["v"])
        out[h] = {"point": point, "boot": boot, "xwoba_point": decompose(bx, dx)["total"]}
    # Home-field effect itself (home vs road pitcher in the same neutral park), wOBA units.
    hf_delta = {"K": -(coef["W"] + coef["c"]), "W": coef["W"], "c": coef["c"], "x": coef["x"], "v": coef["v"]}
    hf = decompose(road, hf_delta)
    return eff["parks"].astype(str), out, base, hf["total"]


def main():
    pa = pd.read_parquet(DATA_PROCESSED / "pa.parquet")
    games = pd.read_parquet(DATA_PROCESSED / "games.parquet")
    g26 = games[games["season"] == ANALYSIS_SEASON]
    abbr = dict(zip(g26["home_id"], g26["home_abbr"])) | dict(zip(g26["away_id"], g26["away_abbr"]))
    pa26 = pa[pa["season"] == ANALYSIS_SEASON].copy()
    pa26["park"] = pa26["park"].astype(str)

    t = roles_and_totals(pa26, abbr)
    parks, ce, base, home_field = channel_effects(pa)
    names = player_names(t.index, logger=log)
    t.insert(0, "name", t.index.map(names))

    # ---- Exposure ------------------------------------------------------------------------------
    w = pa26[pa26["in_woba"]]
    cnt = w.groupby(["pitcher", "pitcher_is_home", "park"]).size()
    tot = w.groupby("pitcher").size()
    exp_h = {}
    for h in (0, 1):
        m = cnt.xs(h, level="pitcher_is_home").unstack("park", fill_value=0)
        m = m.reindex(index=t.index, columns=parks, fill_value=0)
        exp_h[h] = m.div(tot.reindex(t.index), axis=0).fillna(0.0)
    exposure = exp_h[0] + exp_h[1]
    assert np.allclose(exposure.sum(axis=1)[tot.reindex(t.index) > 0], 1.0)

    # ---- Distortion ------------------------------------------------------------------------------
    for part in PARTS:
        t[f"{part}_pts"] = 1000 * sum(exp_h[h].to_numpy() @ np.asarray(ce[h]["point"][part]) for h in (0, 1))
    t["xwoba_distortion_pts"] = 1000 * sum(exp_h[h].to_numpy() @ np.asarray(ce[h]["xwoba_point"]) for h in (0, 1))
    draws = 1000 * sum(exp_h[h].to_numpy() @ np.asarray(ce[h]["boot"]["total"]).T for h in (0, 1))
    t["total_lo"], t["total_hi"] = np.percentile(draws, [2.5, 97.5], axis=1)
    t["total_se"] = draws.std(axis=1, ddof=1)
    t["home_field_pts"] = 1000 * home_field * (t["home_share"] - base["home_share"])
    t["park_neutral_woba"] = t["woba"] - t["total_pts"] / 1000
    t["park_neutral_xwoba"] = t["xwoba"] - t["xwoba_distortion_pts"] / 1000

    # Dominant channel and how much the channels offset each other.
    absch = t[[f"{c}_pts" for c in CHANNELS]].abs()
    t["dominant_channel"] = absch.idxmax(axis=1).str.replace("_pts", "")
    t["dominant_share"] = absch.max(axis=1) / absch.sum(axis=1)
    t["offset_pts"] = absch.sum(axis=1) - t["total_pts"].abs()

    # Exposure summaries.
    vt = pd.read_csv(OUT_TABLES / "venues.csv")
    pname = dict(zip(vt["venue_id"].astype(str), vt["venue_short"])) | {OTHER_PARK: "Other"}
    dec = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    dec = dec[dec["fit"] == "pooled"].set_index("park_key")
    t["home_park_share"] = [exposure.loc[i, str(k)] if str(k) in exposure.columns else np.nan
                            for i, k in zip(t.index, t.index.map(_home_park(pa26)))]
    t["top_parks"] = [", ".join(f"{pname[c]} {100 * v:.0f}%" for c, v in row.nlargest(3).items() if v > 0)
                      for _, row in exposure.iterrows()]
    t["coors_share"] = exposure.get("19", 0.0)
    friendly = dec["total_pts"].reindex(parks).to_numpy()
    t["park_mix_pts"] = exposure.to_numpy() @ friendly   # PA-weighted average park total (no home adj.)

    # ---- Ranks within role among qualifiers ----------------------------------------------------
    t["rank_actual"] = np.nan
    t["rank_neutral"] = np.nan
    for role in ("SP", "RP"):
        q = t["qualifies"] & (t["role"] == role)
        t.loc[q, "rank_actual"] = t.loc[q, "woba"].rank(method="min")
        t.loc[q, "rank_neutral"] = t.loc[q, "park_neutral_woba"].rank(method="min")
    t["rank_gain"] = t["rank_actual"] - t["rank_neutral"]

    t = t.sort_values(["qualifies", "role", "total_pts"], ascending=[False, False, False])
    t.index.name = "pitcher"
    t.round(5).to_csv(OUT_TABLES / "pitcher_distortion.csv")
    ex = exposure.loc[t["qualifies"]].copy()
    ex.columns = [pname[c] for c in ex.columns]
    ex.insert(0, "name", t.loc[ex.index, "name"])
    ex.round(4).to_csv(OUT_TABLES / "pitcher_exposure_2026.csv")

    # ---- Case studies ----------------------------------------------------------------------------
    # Rule: one team all season (a single home park); for the single-channel cases, that channel
    # carries >= 60% of the pitcher's total channel magnitude and the 95% interval for the
    # total excludes zero; the largest effect in that channel wins. Each case comes from a
    # different team.
    q = t[t["qualifies"]].copy()
    q["ci_excludes_0"] = (q["total_lo"] > 0) | (q["total_hi"] < 0)
    single = ~q["team"].str.contains("/")
    picks = {}
    for label, ch in (("Strikeout/walk channel", "kbb"), ("Conversion channel", "conv")):
        used = q.loc[list(picks.values()), "team"].tolist()
        cand = q[(q["dominant_channel"] == ch) & (q["dominant_share"] >= 0.6) & q["ci_excludes_0"]
                 & single & ~q["team"].isin(used)]
        picks[label] = cand[f"{ch}_pts"].abs().idxmax()
    used = q.loc[list(picks.values()), "team"].tolist()
    c = q[single & ~q["team"].isin(used)]
    # Offsetting: large channel effects that mostly cancel (|total| well below the canceled part).
    strict = c[c["total_pts"].abs() < 0.25 * c["offset_pts"]]
    picks["Channels offset"] = (strict if len(strict) else c)["offset_pts"].idxmax()
    cs = q.loc[list(picks.values())].copy()
    cs.insert(0, "case", list(picks.keys()))
    cs_cols = ["case", "name", "team", "role", "bf", "woba", "xwoba", "park_neutral_woba", "total_pts",
               "total_lo", "total_hi", "kbb_pts", "k_part_pts", "bbhbp_part_pts", "contact_pts", "conv_pts",
               "home_share", "top_parks", "rank_actual", "rank_neutral", "rank_gain"]
    cs[cs_cols].to_csv(OUT_TABLES / "pitcher_case_studies.csv")

    # ---- Simpson's paradox note (3 park groups by total effect, 2026 PA tertiles) ----------------
    park_tot = dec["total_pts"].reindex(parks)
    park_pa = w.groupby("park").size().reindex(parks, fill_value=0)
    order = park_tot.sort_values().index
    cum = park_pa[order].cumsum() / park_pa.sum()
    group = pd.Series(np.select([cum <= 1 / 3 + 1e-9, cum <= 2 / 3 + 1e-9], ["pitcher-friendly", "neutral"],
                                "hitter-friendly"), index=order)
    wg = w.assign(group=w["park"].map(group))
    gw = wg.groupby(["pitcher", "group"]).agg(n=("woba_val", "size"), woba=("woba_val", "mean")).unstack("group")
    pairs = []
    for role in ("SP", "RP"):
        ids = q.index[q["role"] == role]
        G = gw.reindex(ids)
        ok = (G["n"] >= 40).all(axis=1)
        G = G[ok]
        wv, ov = G["woba"].to_numpy(), q.loc[G.index, "woba"].to_numpy()
        for i in range(len(G)):
            better = (wv[i] < wv).all(axis=1) & (ov[i] > ov)
            for j in np.flatnonzero(better):
                pairs.append({"role": role, "A": G.index[i], "B": G.index[j],
                              "A_name": q.loc[G.index[i], "name"], "B_name": q.loc[G.index[j], "name"],
                              **{f"A_{gname}": wv[i][k] for k, gname in enumerate(G["woba"].columns)},
                              **{f"B_{gname}": wv[j][k] for k, gname in enumerate(G["woba"].columns)},
                              "A_overall": ov[i], "B_overall": ov[j],
                              "min_gap_in_groups": (wv[j] - wv[i]).min(),
                              "A_park_mix_pts": q.loc[G.index[i], "park_mix_pts"],
                              "B_park_mix_pts": q.loc[G.index[j], "park_mix_pts"]})
    sp = pd.DataFrame(pairs)
    if len(sp):
        sp = sp.sort_values("min_gap_in_groups", ascending=False)
    sp.to_csv(OUT_TABLES / "simpson_pairs.csv", index=False)
    pd.DataFrame({"park_key": group.index, "venue_short": [pname[k] for k in group.index],
                  "total_pts": park_tot[group.index].to_numpy(), "group": group.to_numpy(),
                  "pa_2026": park_pa[group.index].to_numpy()}).to_csv(OUT_TABLES / "park_groups_2026.csv", index=False)

    # ---- Summary -----------------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 7 summary")
    log.info("2026 pitchers: %d (%d starters, %d relievers by >= 50%% starts). Qualifying: %d starters "
             "(400+ BF) and %d relievers (150+ BF): thresholds unchanged", len(t), (t["role"] == "SP").sum(),
             (t["role"] == "RP").sum(), (t["qualifies"] & (t["role"] == "SP")).sum(),
             (t["qualifies"] & (t["role"] == "RP")).sum())
    log.info("Expected distortion among qualifiers (wOBA points): SD %.1f, range %.1f to %+.1f; "
             "|distortion| > 95%% interval excludes 0 for %d pitchers", q["total_pts"].std(),
             q["total_pts"].min(), q["total_pts"].max(), ((q["total_lo"] > 0) | (q["total_hi"] < 0)).sum())
    log.info("Home-field term (not removed): SD across qualifiers %.2f points", q["home_field_pts"].std())
    cols = ["name", "team", "role", "bf", "woba", "park_neutral_woba", "total_pts", "total_lo", "total_hi",
            "kbb_pts", "contact_pts", "conv_pts", "rank_actual", "rank_neutral", "rank_gain"]
    log_df(log, q.sort_values("total_pts", ascending=False)[cols].head(10).round(3),
           "Most hurt by their parks (largest positive distortion = underrated):", index=False)
    log_df(log, q.sort_values("total_pts")[cols].head(10).round(3),
           "Most helped by their parks (overrated):", index=False)
    for role in ("SP", "RP"):
        r = q[q["role"] == role]
        log.info("%s: rank changes: mean |change| %.1f places, max up %d, max down %d (of %d)", role,
                 r["rank_gain"].abs().mean(), r["rank_gain"].max(), -r["rank_gain"].min(), len(r))
    log_df(log, cs[cs_cols].round(3).T, "Case studies:")
    log.info("Simpson's paradox note: %d pairs (A lower wOBA in all three park groups, higher overall); "
             "%d starter, %d reliever. In %d of them A had the more hitter-friendly park mix.",
             len(sp), (sp["role"] == "SP").sum() if len(sp) else 0, (sp["role"] == "RP").sum() if len(sp) else 0,
             (sp["A_park_mix_pts"] > sp["B_park_mix_pts"]).sum() if len(sp) else 0)
    if len(sp):
        log_df(log, sp.head(5).round(3), "Top pairs by smallest within-group gap:", index=False)
    log.info("Saved pitcher_distortion.csv, pitcher_case_studies.csv, pitcher_exposure_2026.csv, simpson_pairs.csv")


def _home_park(pa26: pd.DataFrame) -> dict:
    """Most frequent park among each pitcher's home plate appearances."""
    h = pa26[pa26["pitcher_is_home"] == 1].groupby(["pitcher", "park"]).size().reset_index(name="n")
    h = h.sort_values(["pitcher", "n"], ascending=[True, False]).drop_duplicates("pitcher")
    return dict(zip(h["pitcher"], h["park"]))


if __name__ == "__main__":
    main()
