"""Step 5: convert each park's channel effects into wOBA points.

Within the wOBA denominator, every plate appearance is a strikeout (worth 0), a walk/HBP
(worth w_W) or a ball in play, so

    wOBA = W * w_W + c * (x + v)

    W = walk + HBP rate          c = ball-in-play rate (K = 1 - W - c)
    x = xwOBA on contact         v = conversion = wOBA on contact - x

Channels:
    1. strikeout/walk   (K, W and therefore c)
    2. contact quality  (x)
    3. conversion       (v)

Neutral park: league means of W, c, x, v over the fit sample, with w_W the mean linear weight
of a walk or HBP. A park shifts each quantity by its centered Step 4 effect (strikeout and
walk/HBP from models 1-2, c from model 3, which share a penalty so the shares stay exhaustive).

Attribution: switching one channel at a time from neutral to the park's value depends on the
order, because c multiplies (x + v). Each channel's contribution is therefore its Shapley
value: the average of its marginal effect over all 3! = 6 orderings. Shapley values add up
exactly to the total; the reported interaction is what one-at-a-time attributions would leave
unexplained, total - sum_i [f({i}) - f({})] = dc * (dx + dv), and Shapley splits it evenly
between the channels involved. Channel 1 is further split exactly into a strikeout part and a
walk/HBP part.

All results are in wOBA points (1 point = 0.001 wOBA).

Outputs: outputs/tables/park_wOBA_decomposition.csv (+ data/processed/decomposition_<fit>.npz
with bootstrap draws).
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .common import get_logger, log_df
from .config import DATA_PROCESSED, OTHER_PARK, OUT_TABLES
from .park_model import FITS, add_literal_columns, samples

log = get_logger("decompose", "step5_decompose.log")

CHANNELS = ("kbb", "contact", "conv")
CHANNEL_LABELS = {"kbb": "Strikeout/walk", "contact": "Contact quality", "conv": "Conversion"}


# --------------------------------------------------------------------------------------------
# Core functions (tested in tests/test_decompose.py)
# --------------------------------------------------------------------------------------------
def woba_from_state(W, c, x, v, w_w):
    """wOBA per PA from the walk/HBP rate, contact rate, xwOBAcon and conversion."""
    return W * w_w + c * (x + v)


def shapley(f, n: int) -> np.ndarray:
    """Shapley values of n channels for a set function f(frozenset of switched channels)."""
    phi = np.zeros(n)
    perms = list(permutations(range(n)))
    for order in perms:
        on = frozenset()
        for i in order:
            phi[i] += f(on | {i}) - f(on)
            on = on | {i}
    return phi / len(perms)


def decompose(base: dict, delta: dict) -> dict:
    """Shapley decomposition of one park, in wOBA units.

    base: neutral values {W, c, x, v, w_w}; delta: park shifts {K, W, c, x, v}. Works on numpy
    arrays (e.g. bootstrap draws) as well as scalars.
    """
    W0, c0, x0, v0, ww = base["W"], base["c"], base["x"], base["v"], base["w_w"]
    dW, dc, dx, dv, dK = delta["W"], delta["c"], delta["x"], delta["v"], delta["K"]
    total = woba_from_state(W0 + dW, c0 + dc, x0 + dx, v0 + dv, ww) - woba_from_state(W0, c0, x0, v0, ww)
    # Closed form of the 6-ordering average (verified against `shapley` in the tests).
    mid_contact = x0 + v0 + 0.5 * (dx + dv)   # Shapley-average contact value seen by channel 1
    mid_c = c0 + 0.5 * dc                      # Shapley-average contact share seen by channels 2-3
    phi_kbb = dW * ww + dc * mid_contact
    phi_contact = dx * mid_c
    phi_conv = dv * mid_c
    oat = {"kbb": dW * ww + dc * (x0 + v0), "contact": dx * c0, "conv": dv * c0}
    # Channel 1 split: dc = -dK - dW - d(other); any non-K/W/BIP remainder goes with walks.
    phi_k = -dK * mid_contact
    phi_bbhbp = phi_kbb - phi_k
    return {"total": total, "kbb": phi_kbb, "contact": phi_contact, "conv": phi_conv,
            "k_part": phi_k, "bbhbp_part": phi_bbhbp,
            "oat_kbb": oat["kbb"], "oat_contact": oat["contact"], "oat_conv": oat["conv"],
            "interaction": total - (oat["kbb"] + oat["contact"] + oat["conv"])}


def baseline(pa: pd.DataFrame, fit: str) -> dict:
    """League-average (neutral-park) values over a fit's sample."""
    spec = FITS[fit]
    data = samples(pa, spec)
    d, b = data["pa"], data["bip"]
    cols = spec.columns or {}
    wcol, xcol, vcol = cols.get("woba", "woba_val"), cols.get("xwobacon", "xwoba"), cols.get("conv", "conv")
    W = d["is_bbhbp"].mean()
    out = {"K": d["is_k"].mean(), "W": W, "c": d["is_bip"].mean(),
           "x": b[xcol].mean(), "v": b[vcol].mean(),
           "w_w": d.loc[d["is_bbhbp"] == 1, wcol].mean(),
           "home_share": d["pitcher_is_home"].mean(),
           "league_woba": d[wcol].mean()}
    # PAs that are none of K / walk-HBP / ball in play (catcher's interference in the literal
    # version) are held at their league rate and value.
    other = 1 - out["K"] - out["W"] - out["c"]
    out["other_share"] = other
    out["other_value"] = d.loc[(d["is_k"] == 0) & (d["is_bbhbp"] == 0) & (~d["is_bip"]), wcol].sum() / len(d)
    return out


def load_effects(fit: str) -> dict:
    z = np.load(DATA_PROCESSED / f"park_effects_{fit}.npz")
    return {k: z[k] for k in z.files}


def deltas_from(arr: np.ndarray, outcomes: list[str]) -> dict:
    """Map an (..., outcome) array of centered park effects to decomposition inputs."""
    idx = {o: i for i, o in enumerate(outcomes)}
    return {"K": arr[..., idx["k"]], "W": arr[..., idx["bbhbp"]], "c": arr[..., idx["bip"]],
            "x": arr[..., idx["xwobacon"]], "v": arr[..., idx["conv"]]}


def home_road_baselines(base: dict, fixed_names, point_fixed, outcomes) -> tuple[dict, dict]:
    """Neutral values for a home pitcher and a road pitcher (league means +/- home term)."""
    j = list(fixed_names).index("home:pitcher_is_home")
    idx = {o: i for i, o in enumerate(outcomes)}
    h = base["home_share"]
    coef = {"W": point_fixed[j, idx["bbhbp"]], "c": point_fixed[j, idx["bip"]],
            "x": point_fixed[j, idx["xwobacon"]], "v": point_fixed[j, idx["conv"]]}
    home = dict(base, **{k: base[k] + coef[k] * (1 - h) for k in coef})
    road = dict(base, **{k: base[k] - coef[k] * h for k in coef})
    return home, road, coef


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def decomposition_table(pa, fit: str, eff: dict, venues: pd.DataFrame, key: str = "point") -> pd.DataFrame:
    base = baseline(pa, fit)
    outcomes = list(eff["outcomes"])
    dec = decompose(base, deltas_from(eff[key], outcomes))
    t = pd.DataFrame({"fit": fit, "park_key": eff["parks"], "pa": eff["counts_pa"].astype(int)})
    t = t.join(venues, on="park_key")
    t.loc[t["park_key"] == OTHER_PARK, "venue_short"] = "Other (neutral sites)"
    for k, v in dec.items():
        t[f"{k}_pts"] = 1000 * np.asarray(v)
    t["direct_woba_model_pts"] = 1000 * eff[key][:, outcomes.index("woba")]
    if key == "point" and "boot" in eff:
        bdec = decompose(base, deltas_from(eff["boot"], outcomes))
        for k in ("total", "kbb", "contact", "conv", "k_part", "bbhbp_part", "interaction"):
            draws = 1000 * np.asarray(bdec[k])
            t[f"{k}_lo"] = np.percentile(draws, 2.5, axis=0)
            t[f"{k}_hi"] = np.percentile(draws, 97.5, axis=0)
            t[f"{k}_se"] = draws.std(axis=0, ddof=1)
        np.savez_compressed(DATA_PROCESSED / f"decomposition_{fit}.npz", parks=eff["parks"],
                            **{k: 1000 * np.asarray(bdec[k]) for k in bdec})
        # Probability the park is the single most hitter-friendly one in a resample.
        mod = t["park_key"].to_numpy() != OTHER_PARK
        tot = 1000 * np.asarray(bdec["total"])[:, mod]
        top = np.bincount(tot.argmax(axis=1), minlength=mod.sum()) / len(tot)
        t.loc[mod, "p_most_hitter_friendly"] = top
    return t, base


def main():
    pa = add_literal_columns(pd.read_parquet(DATA_PROCESSED / "pa.parquet"))
    v = pd.read_csv(OUT_TABLES / "venues.csv")
    v["park_key"] = v["venue_id"].astype(str)
    venues = v.set_index("park_key")[["venue_short", "home_team"]]

    tables, bases = [], {}
    for fit in ("pooled", "2026"):
        eff = load_effects(fit)
        t, base = decomposition_table(pa, fit, eff, venues)
        tables.append(t)
        bases[fit] = base
        recon = base["W"] * base["w_w"] + base["c"] * (base["x"] + base["v"]) + base["other_value"]
        log.info("Fit '%s' neutral park: K %.4f, walk+HBP %.4f (weight %.4f), contact c %.4f, "
                 "xwOBAcon %.4f, conversion %+.4f -> wOBA %.4f (league wOBA %.4f)", fit, base["K"],
                 base["W"], base["w_w"], base["c"], base["x"], base["v"], recon, base["league_woba"])
    dec = pd.concat(tables, ignore_index=True)
    order = ["fit", "park_key", "venue_short", "home_team", "pa", "total_pts", "total_lo", "total_hi",
             "kbb_pts", "kbb_lo", "kbb_hi", "k_part_pts", "bbhbp_part_pts", "contact_pts",
             "contact_lo", "contact_hi", "conv_pts", "conv_lo", "conv_hi", "interaction_pts",
             "oat_kbb_pts", "oat_contact_pts", "oat_conv_pts", "direct_woba_model_pts",
             "p_most_hitter_friendly"]
    dec = dec[order + [c for c in dec.columns if c not in order]]
    dec = dec.sort_values(["fit", "total_pts"], ascending=[True, False])
    dec.to_csv(OUT_TABLES / "park_wOBA_decomposition.csv", index=False)

    # ---- Sensitivity decompositions (point estimates) ----------------------------------------
    eff_p = load_effects("pooled")
    sens = {"pooled": dec[dec["fit"] == "pooled"].set_index("park_key")["total_pts"]}
    for key in [k for k in eff_p if k.startswith("point_scale_")]:
        t, _ = decomposition_table(pa, "pooled", eff_p, venues, key=key)
        sens[key.replace("point_scale_", "lambda_x")] = t.set_index("park_key")["total_pts"]
    t, _ = decomposition_table(pa, "pooled_literal", load_effects("pooled_literal"), venues)
    sens["literal_columns"] = t.set_index("park_key")["total_pts"]
    sens = pd.DataFrame(sens).drop(index=OTHER_PARK, errors="ignore")
    sens.join(venues["venue_short"]).to_csv(OUT_TABLES / "decomposition_sensitivity.csv")

    # ---- Summary and sanity checks ---------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 5 summary (wOBA points; 1 point = 0.001)")
    for fit in ("pooled", "2026"):
        t = dec[(dec["fit"] == fit)].copy()
        cols = ["venue_short", "total_pts", "total_lo", "total_hi", "kbb_pts", "contact_pts",
                "conv_pts", "interaction_pts", "direct_woba_model_pts"]
        log_df(log, t[cols].round(1), f"Fit '{fit}': parks by total effect", index=False)
        mod = t[t["park_key"] != OTHER_PARK]
        log.info("  interaction term: max |%.2f| points; Shapley total vs direct wOBA model: r = %.3f",
                 mod["interaction_pts"].abs().max(),
                 np.corrcoef(mod["total_pts"], mod["direct_woba_model_pts"])[0, 1])

    p = dec[(dec["fit"] == "pooled") & (dec["park_key"] != OTHER_PARK)].set_index("venue_short")
    top = p["total_pts"].idxmax()
    gap = p["total_pts"].nlargest(2)
    log.info("SANITY CHECK 1 (Coors Field has the largest total effect): top park = %s "
             "(%.1f points; next %s %.1f). P(Coors is #1 across bootstrap resamples) = %.2f -> %s",
             top, gap.iloc[0], gap.index[1], gap.iloc[1], p.loc["Coors Field", "p_most_hitter_friendly"],
             "PASS" if top == "Coors Field" else "FAIL")
    f = p.loc["Fenway Park"]
    parts = f[["kbb_pts", "contact_pts", "conv_pts"]]
    share = f["conv_pts"] / f["total_pts"] if f["total_pts"] != 0 else np.nan
    log.info("SANITY CHECK 2 (Fenway's effect comes mostly from conversion): total %.1f = "
             "K/BB %.1f + contact %.1f + conversion %.1f (+ interaction %.2f); conversion share %.0f%% "
             "-> %s", f["total_pts"], *parts, f["interaction_pts"], 100 * share,
             "PASS" if parts.abs().idxmax() == "conv_pts" and share > 0.5 else "CHECK")
    log_df(log, sens.join(venues["venue_short"]).set_index("venue_short").round(1)
           .sort_values("pooled", ascending=False).head(8),
           "Sensitivity of total effects (pooled): penalties x0.25 / x4, raw Statcast columns:")
    log.info("Saved park_wOBA_decomposition.csv")


if __name__ == "__main__":
    main()
