"""Step 2: assign a venue to every game.

Every game_pk is mapped to the venue named in the MLB Stats API schedule, never to
`home_team`. The venue ID is the unit, so renamed parks keep one identity (Minute Maid Park ->
Daikin Park, Guaranteed Rate Field -> Rate Field, Dodger Stadium -> UNIQLO Field at Dodger
Stadium), while a team that moved (Athletics, Rays) gets a separate venue for each park.

A team's *home park* in a season is the venue where it played most of its designated home
games. A game anywhere else is a neutral-site game. For park modeling:

* a venue gets its own park effect if it was some team's home park in a season of the fit
  window (any of 2024-2026 for the pooled fit, 2026 for the 2026-only fit); every game played
  there counts toward it, including a neutral-site game (e.g. the Red Sox "home" game at
  Tropicana Field in 2026);
* every other venue (international series, Field of Dreams, Williamsport, Las Vegas, ...) is
  pooled into one "other" category. Those games stay in pitcher totals.

Outputs: data/processed/games.parquet and outputs/tables/venues.csv.
"""
from __future__ import annotations

import pandas as pd

from . import mlb_api
from .common import get_logger, log_df, write_parquet_atomic
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OTHER_PARK, OUT_TABLES, SEASONS
from .fetch import load_statcast

log = get_logger("venues", "step2_venues.log")

# Shorter display names for long official names (labels only; the venue ID is the key).
SHORT_NAMES = {
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
    "Oriole Park at Camden Yards": "Camden Yards",
    "George M. Steinbrenner Field": "Steinbrenner Field",
    "Estadio Alfredo Harp Helu": "Estadio Alfredo Harp Helú",
}


def season_games(season: int) -> pd.DataFrame:
    """Completed games of one season with venue, teams and a neutral-site flag."""
    entries = mlb_api.schedule_entries(season)
    done = mlb_api.completed_games(season)

    # A suspended game appears on two dates; make sure it was not resumed at another park.
    multi = (entries[entries["game_pk"].isin(done["game_pk"])]
             .groupby("game_pk")["venue_id"].nunique())
    moved = multi[multi > 1]
    if len(moved):
        log.warning("%s: %d games have schedule entries at more than one venue: %s. The "
                    "venue of the completed (latest) entry is used.", season, len(moved),
                    moved.index.tolist())

    abbr = mlb_api.team_abbrevs(season)
    g = done.copy()
    g["home_abbr"] = g["home_id"].map(abbr)
    g["away_abbr"] = g["away_id"].map(abbr)

    # Home park = venue hosting most of the team's designated home games.
    home_park = (g.groupby(["home_id", "venue_id"]).size().rename("n").reset_index()
                 .sort_values(["home_id", "n"], ascending=[True, False])
                 .drop_duplicates("home_id").set_index("home_id")["venue_id"])
    g["home_park_of_home_team"] = g["home_id"].map(home_park)
    g["is_neutral"] = g["venue_id"] != g["home_park_of_home_team"]
    return g, home_park


def main():
    frames, home_parks = [], []
    for season in SEASONS:
        g, hp = season_games(season)
        frames.append(g)
        home_parks.append(pd.DataFrame({"season": season, "team_id": hp.index, "venue_id": hp.values}))
    games = pd.concat(frames, ignore_index=True)
    home_parks = pd.concat(home_parks, ignore_index=True)

    # Latest official name per venue ID, plus a short display name.
    latest = games.sort_values("game_datetime_utc").drop_duplicates("venue_id", keep="last")
    names = latest.set_index("venue_id")["venue_name"]
    games["venue_name"] = games["venue_id"].map(names)
    games["venue_short"] = games["venue_name"].replace(SHORT_NAMES)

    # Modeled venues: home parks in any season of the window (pooled) or in 2026 (2026-only).
    home_any = set(home_parks["venue_id"])
    home_2026 = set(home_parks.loc[home_parks["season"] == ANALYSIS_SEASON, "venue_id"])
    games["park"] = games["venue_id"].where(games["venue_id"].isin(home_any), other=-1)
    games["park_2026"] = games["venue_id"].where(
        games["venue_id"].isin(home_2026) & (games["season"] == ANALYSIS_SEASON), other=-1)
    games.loc[games["season"] != ANALYSIS_SEASON, "park_2026"] = pd.NA
    # Encode the pooled "other" category as a string key alongside venue IDs.
    games["park"] = games["park"].astype("Int64").astype(str).replace({"-1": OTHER_PARK})
    games["park_2026"] = games["park_2026"].astype("Int64").astype(str).replace(
        {"-1": OTHER_PARK, "<NA>": pd.NA})

    # Check: every Statcast game has a venue, and Statcast's home_team agrees with the schedule.
    log.info("Matching Statcast games to the schedule")
    pa_counts = []
    for season in SEASONS:
        sc = load_statcast(season, columns=["game_pk", "game_type", "home_team", "events",
                                            "at_bat_number", "pitch_number"], logger=log)
        sc_games = sc.drop_duplicates("game_pk")[["game_pk", "home_team"]]
        m = sc_games.merge(games[games["season"] == season][["game_pk", "home_abbr"]],
                           on="game_pk", how="left")
        unmapped = m["home_abbr"].isna().sum()
        # Statcast already labels the 2024 Athletics "ATH" where the Stats API says "OAK".
        alias = m["home_team"].replace({"ATH": "OAK"}) == m["home_abbr"]
        mismatch = m[m["home_abbr"].notna() & (m["home_team"] != m["home_abbr"]) & ~alias]
        log.info("  %s: %d Statcast games, %d unmapped, %d with Statcast home_team != schedule "
                 "home team%s", season, len(m), unmapped, len(mismatch),
                 f" {mismatch[['home_team', 'home_abbr']].drop_duplicates().values.tolist()}"
                 if len(mismatch) else "")
        if unmapped:
            raise SystemExit(f"STOP: {unmapped} Statcast games in {season} have no schedule venue.")
        # Plate appearances per game (rows with an event, excluding truncated PAs).
        pa = sc[sc["events"].notna() & (sc["events"] != "truncated_pa")]
        pa_counts.append(pa.groupby("game_pk").size().rename("pa"))
    games = games.merge(pd.concat(pa_counts), left_on="game_pk", right_index=True, how="left")

    keep = ["game_pk", "season", "official_date", "venue_id", "venue_name", "venue_short",
            "home_id", "away_id", "home_abbr", "away_abbr", "home_name", "away_name",
            "home_park_of_home_team", "is_neutral", "park", "park_2026", "series_description",
            "status", "pa"]
    games = games[keep].sort_values(["season", "official_date", "game_pk"]).reset_index(drop=True)
    write_parquet_atomic(games, DATA_PROCESSED / "games.parquet")

    # ---- Venue table -----------------------------------------------------------------------
    # Season-specific abbreviation (the Athletics are OAK in 2024 and ATH from 2025).
    hp_named = home_parks.merge(
        games[["season", "home_id", "home_abbr"]].drop_duplicates(["season", "home_id"]),
        left_on=["season", "team_id"], right_on=["season", "home_id"])
    teams_by_venue = (hp_named.sort_values("season").groupby("venue_id")
                      .agg(home_team=("home_abbr", lambda s: "/".join(dict.fromkeys(s))),
                           home_seasons=("season", lambda s: ",".join(map(str, sorted(set(s)))))))
    vt = (games.groupby(["venue_id", "season"])
          .agg(games=("game_pk", "size"), neutral_games=("is_neutral", "sum"), pa=("pa", "sum"))
          .unstack("season", fill_value=0))
    vt.columns = [f"{a}_{b}" for a, b in vt.columns]
    vt = vt.join(games.drop_duplicates("venue_id").set_index("venue_id")[["venue_name", "venue_short"]])
    vt = vt.join(teams_by_venue)
    vt["modeled_pooled"] = vt.index.isin(home_any)
    vt["modeled_2026"] = vt.index.isin(home_2026)
    # Physical attributes (Stats API, current values) for the Step 6 mechanism checks.
    attrs = pd.DataFrame([mlb_api.venue_details(int(v)) for v in vt.index]).set_index("venue_id")
    # Mean fence distance over the five points the API reports for (almost) every park: both
    # foul lines, both power alleys and center field. Missing if any of the five is missing.
    five = ["fence_leftLine", "fence_leftCenter", "fence_center", "fence_rightCenter", "fence_rightLine"]
    attrs["fence_mean_ft"] = attrs[five].astype(float).mean(axis=1, skipna=False)
    vt = vt.join(attrs)
    vt = vt.reset_index().sort_values(["modeled_pooled", "venue_name"], ascending=[False, True])
    vt.to_csv(OUT_TABLES / "venues.csv", index=False)

    # ---- Summary -----------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Step 2 summary: %d completed regular-season games at %d venues", len(games),
             games["venue_id"].nunique())
    for season in SEASONS:
        gs = games[games["season"] == season]
        neu = gs[gs["is_neutral"]]
        log.info("%s: %d games at %d venues; %d neutral-site games", season, len(gs),
                 gs["venue_id"].nunique(), len(neu))
        for (vid, vname), grp in neu.groupby(["venue_id", "venue_short"]):
            matchups = "; ".join(sorted(set(grp["away_abbr"] + " @ " + grp["home_abbr"])))
            log.info("    %-28s %d game(s): %s -> park key '%s'", vname, len(grp), matchups,
                     grp["park"].iloc[0])
    moved = (home_parks.merge(vt[["venue_id", "venue_short"]], on="venue_id")
             .pivot(index="team_id", columns="season", values="venue_short"))
    moved = moved[moved.nunique(axis=1) > 1]
    moved.index = moved.index.map(dict(zip(games["home_id"], games["home_abbr"])))
    log_df(log, moved, "Teams whose home park changed during 2024-2026:")

    for label, col in (("pooled 2024-2026", "park"), ("2026-only", "park_2026")):
        sub = games[games[col].notna()]
        other = sub[sub[col] == OTHER_PARK]
        log.info("%s fit: %d modeled venues + '%s'. '%s' holds %d games and %d of %d PAs "
                 "(%.2f%%): %s", label, sub.loc[sub[col] != OTHER_PARK, col].nunique(), OTHER_PARK,
                 OTHER_PARK, len(other), other["pa"].sum(), sub["pa"].sum(),
                 100 * other["pa"].sum() / sub["pa"].sum(),
                 ", ".join(sorted(other["venue_short"].unique())))
    log.info("Saved data/processed/games.parquet and outputs/tables/venues.csv")


if __name__ == "__main__":
    main()
