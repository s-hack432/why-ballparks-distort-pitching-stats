"""Player-name lookup: pybaseball.playerid_reverse_lookup, with the MLB Stats API as fallback.

The Chadwick register behind playerid_reverse_lookup title-cases names ("Mcclanahan") and
does not yet know every 2026 debut, so the Stats API's exact spelling is preferred when it
has the player; Chadwick fills any gaps.
"""
from __future__ import annotations

import pandas as pd

from . import mlb_api
from .config import DATA_PROCESSED

NAMES_PATH = DATA_PROCESSED / "player_names.csv"


def player_names(ids, logger=None) -> dict[int, str]:
    ids = sorted({int(i) for i in ids})
    cached = (pd.read_csv(NAMES_PATH) if NAMES_PATH.exists()
              else pd.DataFrame(columns=["mlbam", "name", "source"]))
    have = set(cached["mlbam"].astype(int))
    need = [i for i in ids if i not in have]
    if need:
        found = {}
        try:
            from pybaseball import playerid_reverse_lookup

            lk = playerid_reverse_lookup(need, key_type="mlbam")
            for r in lk.itertuples():
                first = str(r.name_first).strip().title() if pd.notna(r.name_first) else ""
                last = str(r.name_last).strip().title() if pd.notna(r.name_last) else ""
                if first or last:
                    found[int(r.key_mlbam)] = f"{first} {last}".strip()
        except Exception as exc:  # noqa: BLE001 - lookup is best effort; the API is the fallback
            if logger:
                logger.warning("playerid_reverse_lookup failed (%s); using MLB Stats API names", exc)
        api = mlb_api.people_names(need)
        rows = []
        for i in need:
            if api.get(i):
                rows.append({"mlbam": i, "name": api[i],
                             "source": "mlb_api (chadwick match)" if i in found else "mlb_api"})
            elif i in found:
                rows.append({"mlbam": i, "name": found[i], "source": "chadwick"})
            else:
                rows.append({"mlbam": i, "name": f"MLBAM {i}", "source": "missing"})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(NAMES_PATH, index=False)
        if logger:
            src = pd.DataFrame(rows)["source"].value_counts().to_dict()
            logger.info("Looked up %d player names: %s", len(rows), src)
    m = dict(zip(cached["mlbam"].astype(int), cached["name"]))
    return {i: m.get(i, f"MLBAM {i}") for i in ids}
