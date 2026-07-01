"""Golden validation: OI-crossover roll dates land shortly before each
contract's expiry / first-notice -- the intrinsic correctness property of an
OI-based roll. This gate catches gross breakage (the old .c.0 calendar-roll
bug rolled erratically at ~43 bars/day).

Anchored on INDEPENDENT known 2024 CME cycle facts -- equity-index 3rd-Friday
quarterly expiries and metals even-month first-notice dates -- NOT on
volume-roll "published roll date" heuristics, because an OI-crossover roll
lands a few days closer to expiry than a volume roll does.
"""
from datetime import date

import pytest

from src.data.futures.paths import statistics_dir
from scripts.data.build_roll_calendar import build_root


def _data_present() -> bool:
    return (statistics_dir() / "year=2024" / "month=1" / "data.parquet").exists()


pytestmark = pytest.mark.skipif(
    not _data_present(), reason="consolidated futures store not present"
)


def _roll_dates(root):
    df = build_root(root, date(2024, 1, 1), date(2024, 12, 31))
    return [r["date"] for r in df.iter_rows(named=True) if r["roll_trigger"] != "hold"]


# 2024 E-mini S&P quarterly 3rd-Friday expiries (independent CME calendar facts).
ES_EXPIRIES = [date(2024, 3, 15), date(2024, 6, 21), date(2024, 9, 20), date(2024, 12, 20)]

# 2024 COMEX gold even-month first-notice dates (~last business day of the
# month before delivery; independent CME calendar facts). Oct is skipped in
# GC's liquid cycle.
GC_FIRST_NOTICE = [
    date(2024, 1, 31), date(2024, 3, 28), date(2024, 5, 31),
    date(2024, 7, 31), date(2024, 11, 29),
]


def test_es_rolls_shortly_before_quarterly_expiry():
    rolls = _roll_dates("ES")
    assert 3 <= len(rolls) <= 5, f"ES should roll ~4x in 2024, got {len(rolls)}: {rolls}"
    for r in rolls:
        assert any(0 <= (exp - r).days <= 10 for exp in ES_EXPIRIES), (
            f"ES roll {r} not within 10 days before any 2024 quarterly expiry "
            f"{ES_EXPIRIES} -- OI roll should fire just before expiry"
        )


def test_gc_rolls_before_even_month_first_notice():
    rolls = _roll_dates("GC")
    assert len(rolls) >= 5, f"GC should roll >=5x in 2024, got {len(rolls)}: {rolls}"
    for r in rolls:
        assert any(3 <= (fnd - r).days <= 20 for fnd in GC_FIRST_NOTICE), (
            f"GC roll {r} not within 3-20 days before any 2024 even-month "
            f"first-notice {GC_FIRST_NOTICE} -- OI roll should fire ahead of FND"
        )
