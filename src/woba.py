"""wOBA bookkeeping shared by every step (Baseball Savant leaderboard convention).

Savant's leaderboard wOBA, which this project reproduces, works as follows (approved by the
project owner in place of summing the raw `woba_value` column; see the README):

* Season-specific linear weights for BB, HBP, 1B, 2B, 3B and HR. The BB and HBP weights are
  the values Savant writes into `estimated_woba_using_speedangle` for walks and HBP. The hit
  weights are fitted to Savant's batter leaderboard (Step 3).
* Every other plate-appearance outcome is worth 0, including reached-on-error, fielder's
  choice, and a strikeout where the batter reached on a wild pitch or passed ball. The raw
  `woba_value` column scores the first two as 0.9.
* Denominator = PA - IBB - sacrifice bunts - catcher's interference (= AB + BB - IBB + SF +
  HBP). Truncated PAs (an inning ended by a baserunning out mid-PA) are not plate appearances.
* PAs on untracked pitches stay in. Savant leaves `woba_denom` blank on them, but its
  leaderboard counts them.

xwOBA uses the same denominator: Savant's `estimated_woba_using_speedangle` on batted balls
(imputed where tracking failed, see prepare.py), 0 for strikeouts, and the BB/HBP weight for
walks and HBP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk"}
IBB_EVENTS = {"intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
CI_EVENTS = {"catcher_interf"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
# Batted balls other than hits. All are in wOBA with value 0 except sacrifice bunts.
BIP_OTHER_EVENTS = {"field_out", "force_out", "grounded_into_double_play", "double_play",
                    "triple_play", "fielders_choice", "fielders_choice_out", "field_error",
                    "sac_fly", "sac_fly_double_play"}
SAC_BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
NOT_PA_EVENTS = {"truncated_pa"}

WEIGHTED_EVENTS = ("walk", "hit_by_pitch", "single", "double", "triple", "home_run")


def outcome_type(events: pd.Series) -> pd.Series:
    """Map Statcast `events` to K / BB / IBB / HBP / CI / BIP / NOT_PA.

    Raises on an event the mapping does not know, so a schema change stops the pipeline
    instead of silently misclassifying plate appearances.
    """
    mapping = {}
    for label, evs in (("K", K_EVENTS), ("BB", BB_EVENTS), ("IBB", IBB_EVENTS),
                       ("HBP", HBP_EVENTS), ("CI", CI_EVENTS), ("BIP", HIT_EVENTS),
                       ("BIP", BIP_OTHER_EVENTS), ("BIP", SAC_BUNT_EVENTS),
                       ("NOT_PA", NOT_PA_EVENTS)):
        mapping.update(dict.fromkeys(evs, label))
    out = events.map(mapping)
    unknown = sorted(events[out.isna() & events.notna()].unique())
    if unknown:
        raise ValueError(f"Unmapped Statcast events: {unknown}. Extend src/woba.py before continuing.")
    return out


def in_woba(events: pd.Series) -> pd.Series:
    """True for plate appearances in the wOBA denominator (AB + BB - IBB + SF + HBP)."""
    ot = outcome_type(events)
    return ot.isin(["K", "BB", "HBP", "BIP"]) & ~events.isin(SAC_BUNT_EVENTS)


def woba_values(events: pd.Series, season: pd.Series, weights: pd.DataFrame) -> pd.Series:
    """Linear-weight value of each PA under the season's weights (0 for unweighted events).

    `weights` is indexed by season with one column per event in WEIGHTED_EVENTS.
    """
    long = weights[list(WEIGHTED_EVENTS)].stack().rename("w")
    long.index.names = ["season", "event"]
    key = pd.MultiIndex.from_arrays([season.to_numpy(), events.to_numpy()],
                                    names=["season", "event"])
    vals = long.reindex(key).to_numpy()
    return pd.Series(np.nan_to_num(vals, nan=0.0), index=events.index)


def aggregate_woba(df: pd.DataFrame, by, value="woba_val", xvalue="xwoba",
                   flag="in_woba") -> pd.DataFrame:
    """wOBA and xwOBA by group: sums over in-wOBA PAs divided by their count."""
    d = df[df[flag]]
    g = d.groupby(by, observed=True).agg(n=(value, "size"), woba_sum=(value, "sum"),
                                         xwoba_sum=(xvalue, "sum"))
    g["woba"] = g["woba_sum"] / g["n"]
    g["xwoba"] = g["xwoba_sum"] / g["n"]
    return g


# --------------------------------------------------------------------------------------------
# Batted-ball geometry
# --------------------------------------------------------------------------------------------
# Statcast hit coordinates are pixels on a 250 x 250 field image with home plate at about
# (125.42, 198.27); y grows toward home plate.
HOME_X, HOME_Y = 125.42, 198.27


def spray_angle(hc_x, hc_y):
    """Horizontal direction of a batted ball, in degrees from the catcher's view.

    0 = straight-away center field, negative = toward left field (third-base side), positive =
    toward right field. spray = atan2(hc_x - 125.42, 198.27 - hc_y); the foul lines are at
    about -45 and +45 degrees.
    """
    return np.degrees(np.arctan2(np.asarray(hc_x, float) - HOME_X, HOME_Y - np.asarray(hc_y, float)))


def pull_angle(spray, stand):
    """Spray angle signed so positive = the batter's pull side (left field for a right-handed
    batter, right field for a left-handed batter)."""
    sign = np.where(np.asarray(stand) == "R", -1.0, 1.0)
    return np.asarray(spray, float) * sign


def field_feet(hc_x, hc_y, scale):
    """Hit coordinates in feet from home plate (x: + toward right field, y: toward center)."""
    return scale * (np.asarray(hc_x, float) - HOME_X), scale * (HOME_Y - np.asarray(hc_y, float))
