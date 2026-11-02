"""Thin, cached wrappers around the public MLB Stats API (statsapi.mlb.com)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_API

BASE = "https://statsapi.mlb.com/api/v1"


def _get_json(path: str, params: dict, cache_file: Path, refresh: bool = False) -> dict:
    """GET BASE/path with retries; cache the JSON response on disk."""
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError) as exc:  # network error or bad JSON
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"MLB Stats API request failed: {path} {params}") from last_exc
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(cache_file)
    return data


def regular_season_window(season: int) -> tuple[str, str]:
    """Official regular-season start and end dates (ISO strings)."""
    data = _get_json(f"seasons/{season}", {"sportId": 1}, DATA_API / f"season_{season}.json")
    info = data["seasons"][0]
    return info["regularSeasonStartDate"], info["regularSeasonEndDate"]


def schedule_entries(season: int) -> pd.DataFrame:
    """One row per regular-season schedule *entry*.

    Postponed and suspended games can appear more than once (once per scheduled date).
    """
    params = {"sportId": 1, "season": season, "gameType": "R"}
    data = _get_json("schedule", params, DATA_API / f"schedule_{season}.json")
    rows = []
    for day in data["dates"]:
        for g in day["games"]:
            rows.append({
                "game_pk": g["gamePk"],
                "season": season,
                "official_date": g.get("officialDate"),
                "game_datetime_utc": g.get("gameDate"),
                "status": g["status"]["detailedState"],
                "coded_state": g["status"].get("codedGameState"),
                "venue_id": g["venue"]["id"],
                "venue_name": g["venue"]["name"],
                "home_id": g["teams"]["home"]["team"]["id"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "series_description": g.get("seriesDescription"),
                "resumed_from": g.get("resumedFrom"),
            })
    return pd.DataFrame(rows)


def completed_games(season: int) -> pd.DataFrame:
    """One row per completed regular-season game (coded state 'F': Final or Completed Early).

    A suspended game is listed on both its original and its resumption date; the latest
    completed entry is kept. Step 2 checks that both entries name the same venue.
    """
    s = schedule_entries(season)
    done = s[s["coded_state"] == "F"].sort_values(["game_pk", "game_datetime_utc"])
    return done.drop_duplicates("game_pk", keep="last").reset_index(drop=True)


FIELD_KEYS = ("leftLine", "left", "leftCenter", "center", "rightCenter", "right", "rightLine")


def venue_details(venue_id: int) -> dict:
    """Elevation, coordinates, roof/turf type and fence distances (feet) for one venue.

    These are the Stats API's current (2026) values; missing fields come back as None.
    """
    data = _get_json(f"venues/{venue_id}", {"hydrate": "location,fieldInfo"},
                     DATA_API / f"venue_{venue_id}.json")
    v = (data.get("venues") or [{}])[0]
    loc, field = v.get("location") or {}, v.get("fieldInfo") or {}
    coords = loc.get("defaultCoordinates") or {}
    out = {"venue_id": venue_id, "elevation_ft": loc.get("elevation"),
           "latitude": coords.get("latitude"), "longitude": coords.get("longitude"),
           "azimuth_deg": loc.get("azimuthAngle"), "roof_type": field.get("roofType"),
           "turf_type": field.get("turfType"), "capacity": field.get("capacity")}
    out.update({f"fence_{k}": field.get(k) for k in FIELD_KEYS})
    return out


def team_abbrevs(season: int) -> dict[int, str]:
    data = _get_json("teams", {"sportId": 1, "season": season}, DATA_API / f"teams_{season}.json")
    return {t["id"]: t.get("abbreviation") for t in data["teams"]}


def people_names(ids) -> dict[int, str]:
    """Full names for MLBAM person ids (batched, cached per batch)."""
    names: dict[int, str] = {}
    ids = sorted({int(i) for i in ids})
    for k in range(0, len(ids), 150):
        batch = ids[k:k + 150]
        cache = DATA_API / f"people_{batch[0]}_{batch[-1]}_{len(batch)}.json"
        data = _get_json("people", {"personIds": ",".join(map(str, batch))}, cache)
        for p in data.get("people", []):
            names[p["id"]] = p.get("fullName")
    return names
