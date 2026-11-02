"""Step 8: validate park estimates against Baseball Savant's Statcast park factors.

Savant's factors are used only here, as an outside check; nothing in Steps 4-7 uses them.

Savant publishes rolling park factors on an index scale (100 = average). The main comparison
uses the three-year window ending in 2026 (2024-2026, matching the pooled fit). The one-year
2026 factors are compared with the 2026-only fit.

Our estimates are converted to the same scale: predicted rate at the park / league rate x 100.
* wOBA on contact: Savant's wOBAcon factor vs our contact-quality + conversion channels
  combined, (xwOBAcon + conversion at the park) / league wOBAcon. Both measure actual results
  on balls in play.
* Home runs: Savant's HR factor vs our home runs per ball in play (Step 4 model 6, built like
  the others).
* Also shown: xwOBAcon, strikeouts, walks (ours: walks + HBP) and overall wOBA.

Outputs: outputs/tables/savant_validation.csv (one row per park and window) and
outputs/tables/savant_validation_summary.csv (correlations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import savant
from .common import get_logger, log_df
from .config import DATA_PROCESSED, OTHER_PARK, OUT_TABLES
from .decompose import baseline, load_effects
from .park_model import add_literal_columns

log = get_logger("validate_savant", "step8_validate_savant.log")

# (our quantity, Savant column, label)
COMPARISONS = [
    ("wobacon_idx", "index_wobacon", "wOBA on contact (contact + conversion)"),
    ("hr_idx", "index_hr", "Home runs (ours: per ball in play)"),
    ("xwobacon_idx", "index_xwobacon", "xwOBA on contact"),
    ("k_idx", "index_so", "Strikeouts"),
    ("bb_idx", "index_bb", "Walks (ours: walks + HBP)"),
    ("woba_idx", "index_woba", "wOBA (all channels)"),
]


def our_indexes(pa: pd.DataFrame, fit: str) -> pd.DataFrame:
    eff = load_effects(fit)
    base = baseline(pa, fit)
    o = list(eff["outcomes"])
    P = eff["point"]
    d = pd.read_csv(OUT_TABLES / "park_wOBA_decomposition.csv")
    d = d[d["fit"] == fit].set_index("park_key")
    bip = pa[pa["in_woba"] & pa["is_bip"] & pa["season"].isin(
        (2024, 2025, 2026) if fit == "pooled" else (2026,))]
    hr0 = bip["is_hr"].mean()
    wobacon0 = base["x"] + base["v"]
    t = pd.DataFrame({"park_key": eff["parks"].astype(str), "pa": eff["counts_pa"].astype(int)})
    t["wobacon_idx"] = 100 * (wobacon0 + P[:, o.index("xwobacon")] + P[:, o.index("conv")]) / wobacon0
    t["hr_idx"] = 100 * (hr0 + P[:, o.index("hr")]) / hr0
    t["xwobacon_idx"] = 100 * (base["x"] + P[:, o.index("xwobacon")]) / base["x"]
    t["k_idx"] = 100 * (base["K"] + P[:, o.index("k")]) / base["K"]
    t["bb_idx"] = 100 * (base["W"] + P[:, o.index("bbhbp")]) / base["W"]
    t["woba_idx"] = 100 * (base["league_woba"] + d["total_pts"].reindex(t["park_key"]).to_numpy() / 1000) / base["league_woba"]
    # Bootstrap standard errors on the index scale for the two main comparisons.
    if "boot" in eff:
        B = eff["boot"]
        t["wobacon_idx_se"] = (100 * (B[:, :, o.index("xwobacon")] + B[:, :, o.index("conv")]) / wobacon0).std(axis=0, ddof=1)
        t["hr_idx_se"] = (100 * B[:, :, o.index("hr")] / hr0).std(axis=0, ddof=1)
    return t


def main():
    pa = add_literal_columns(pd.read_parquet(DATA_PROCESSED / "pa.parquet"))
    vt = pd.read_csv(OUT_TABLES / "venues.csv")
    vt["park_key"] = vt["venue_id"].astype(str)
    names = vt.set_index("park_key")["venue_short"]
    seasons = vt.set_index("park_key")["home_seasons"]

    rows, summary = [], []
    for fit, rolling in (("pooled", 3), ("2026", 1)):
        ours = our_indexes(pa, fit)
        sv = savant.park_factors(2026, rolling)
        sv["park_key"] = sv["venue_id"].astype(str)
        log.info("Savant %d-year factors ending 2026: %d venues, window %s", rolling, len(sv),
                 ", ".join(sorted(sv["year_range"].astype(str).unique())))
        m = ours.merge(sv[["park_key", "venue_name", "name_display_club", "n_pa", "year_range"]
                          + [c for _, c, _ in COMPARISONS]], on="park_key", how="outer", indicator=True)
        only_ours = m.loc[m["_merge"] == "left_only", "park_key"].tolist()
        only_sv = m.loc[m["_merge"] == "right_only", "venue_name"].tolist()
        log.info("  parks in our fit but not in Savant's list: %s",
                 [names.get(k, k) for k in only_ours])
        log.info("  parks in Savant's list but not our fit: %s", only_sv)
        m = m[m["_merge"] == "both"].drop(columns="_merge")
        m["window"] = f"{rolling}yr" if rolling > 1 else "2026"
        m["fit"] = fit
        m["venue_short"] = m["park_key"].map(names)
        m["our_seasons"] = m["park_key"].map(seasons)
        for ours_col, sv_col, label in COMPARISONS:
            a, b = m[ours_col].to_numpy(float), m[sv_col].to_numpy(float)
            r, p = stats.pearsonr(a, b)
            rho = stats.spearmanr(a, b)[0]
            nc = m["venue_short"] != "Coors Field"
            summary.append({"fit": fit, "savant_window": m["window"].iloc[0], "comparison": label,
                            "n_parks": len(m), "pearson": r, "pearson_p": p, "spearman": rho,
                            "pearson_without_coors": stats.pearsonr(a[nc], b[nc])[0],
                            "sd_ours": a.std(ddof=1), "sd_savant": b.std(ddof=1),
                            "mean_abs_diff": np.mean(np.abs(a - b))})
            m[f"diff_{ours_col}"] = a - b
        rows.append(m)
    val = pd.concat(rows, ignore_index=True)
    val.to_csv(OUT_TABLES / "savant_validation.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_TABLES / "savant_validation_summary.csv", index=False)

    log.info("=" * 70)
    log.info("Step 8 summary")
    log_df(log, summ.round(3), "Correlations between our park estimates and Savant's (index scale):", index=False)
    p = val[val["fit"] == "pooled"]
    for col, sv_col in (("wobacon_idx", "index_wobacon"), ("hr_idx", "index_hr")):
        dd = p.reindex(p[f"diff_{col}"].abs().sort_values(ascending=False).index).head(6)
        log_df(log, dd[["venue_short", "our_seasons", "year_range", col, sv_col, f"diff_{col}", "pa", "n_pa"]].round(1),
               f"Largest disagreements, {col} vs Savant {sv_col} (pooled vs 3-year):", index=False)
    log.info("Saved savant_validation.csv and savant_validation_summary.csv")


if __name__ == "__main__":
    main()
