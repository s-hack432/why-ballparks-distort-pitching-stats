"""Paths, seasons and constants shared by every step."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_API = DATA_RAW / "mlb_api"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_LOGS = ROOT / "outputs" / "logs"

for _p in (DATA_RAW, DATA_API, DATA_EXTERNAL, DATA_PROCESSED, OUT_TABLES, OUT_FIGURES, OUT_LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# Park profiles pool 2024-2026; pitcher-level analysis is 2026 only.
SEASONS = (2024, 2025, 2026)
ANALYSIS_SEASON = 2026

# Regular-season window stated in the project spec. Step 1 checks it against the MLB Stats API.
EXPECTED_WINDOW = {2026: ("2026-03-25", "2026-09-27")}

# A venue gets its own park effect if it was some team's home park in a season of the fit
# window. Every other venue (neutral sites, one-off events) is pooled into this category.
OTHER_PARK = "other"

# Columns read from the raw Statcast chunks by later steps.
STATCAST_COLS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number", "outs_when_up", "balls", "strikes",
    "pitcher", "batter", "p_throws", "stand",
    "events", "description", "type", "bb_type", "des",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc", "hc_x", "hc_y",
    "hit_location", "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "release_extension", "n_thruorder_pitcher",
]

RANDOM_SEED = 20260927
