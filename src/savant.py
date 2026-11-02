"""Cached downloads from Baseball Savant leaderboards (used only for validation)."""
from __future__ import annotations

import io
import json
import re
import time

import pandas as pd
import requests

from .config import DATA_EXTERNAL

UA = {"User-Agent": "Mozilla/5.0 (research script; park-distortion project)"}


def _get(url: str) -> bytes:
    last = None
    for attempt in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Savant request failed: {url}") from last


def expected_stats(player_type: str, year: int, refresh: bool = False) -> pd.DataFrame:
    """Savant 'Expected Statistics' leaderboard (wOBA, xwOBA, PA, BIP) for every player with
    at least one PA. `player_type` is 'pitcher' or 'batter'."""
    path = DATA_EXTERNAL / f"savant_expected_stats_{player_type}_{year}.csv"
    if refresh or not path.exists():
        url = ("https://baseballsavant.mlb.com/leaderboard/expected_statistics?"
               f"type={player_type}&year={year}&position=&team=&filterType=pa&min=1&csv=true")
        path.write_bytes(_get(url))
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.rename(columns={"last_name, first_name": "name"})


def park_factors(year: int, rolling: int = 3, bat_side: str = "", refresh: bool = False) -> pd.DataFrame:
    """Savant Statcast park factors (index, 100 = average) for the window ending in `year`.

    The leaderboard page embeds its table as a JSON array; it is parsed and cached as CSV.
    """
    tag = f"{year}_{rolling}yr" + (f"_{bat_side}" if bat_side else "")
    path = DATA_EXTERNAL / f"savant_park_factors_{tag}.csv"
    if refresh or not path.exists():
        url = ("https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&"
               f"year={year}&batSide={bat_side}&stat=index_wOBA&condition=All&rolling={rolling}")
        html = _get(url).decode("utf-8")
        m = re.search(r"var data\s*=\s*(\[.*?\]);", html, flags=re.S)
        if not m:
            raise RuntimeError("Could not find the park-factor data on the Savant page. Download "
                               f"the CSV manually from {url} into {path}.")
        pd.DataFrame(json.loads(m.group(1))).to_csv(path, index=False)
    return pd.read_csv(path)
