"""Required check (Step 3): computed 2026 pitcher wOBA and xwOBA vs Baseball Savant.

Compares every pitcher with 100+ PA on Savant's 2026 expected-statistics leaderboard (cached
in data/external/ by src.prepare) with the values computed from the plate-appearance table.
Savant rounds to three decimals, so |diff| <= 0.0005 means "identical after rounding".

Thresholds:
* PA and BIP counts identical for at least 99% of pitchers;
* wOBA within 0.001 for at least 95%;
* xwOBA within 0.002 for at least 95%, using Savant's handling of balls in play that lack a
  Savant estimate (they are left out). The analysis instead keeps them with an imputed xwOBA;
  that version must still be within 0.002 for 90%.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PA_PATH = ROOT / "data" / "processed" / "pa.parquet"
LB_PATH = ROOT / "data" / "external" / "savant_expected_stats_pitcher_2026.csv"

pytestmark = pytest.mark.skipif(not (PA_PATH.exists() and LB_PATH.exists()),
                                reason="run `python -m src.prepare` first")


@pytest.fixture(scope="module")
def comparison():
    from src.prepare import validate

    pa = pd.read_parquet(PA_PATH)
    v = validate(pa, 2026)
    return v[v["pa_savant"] >= 100]


def test_enough_pitchers(comparison):
    assert len(comparison) >= 400


def test_pa_and_bip_counts(comparison):
    assert comparison["pa_match"].mean() >= 0.99
    assert comparison["bip_match"].mean() >= 0.99


def test_woba_matches_savant(comparison):
    assert (comparison["d_woba"].abs() <= 0.001 + 1e-9).mean() >= 0.95


def test_xwoba_matches_savant(comparison):
    assert (comparison["d_xwoba_savant_style"].abs() <= 0.002 + 1e-9).mean() >= 0.95
    assert (comparison["d_xwoba"].abs() <= 0.002 + 1e-9).mean() >= 0.90
