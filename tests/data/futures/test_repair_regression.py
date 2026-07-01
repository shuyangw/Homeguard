"""Real-data regression: these MUST fail before the path repair and pass after.
They would have caught the 2026 consolidation break that the fixture-based
unit tests missed."""
from datetime import date

import pytest

from src.data.futures.paths import per_contract_1min_dir
from src.data.carry_calculator import CarryCalculator
from src.data.continuous_contract_loader import ContinuousContractDataLoader


def _data_present() -> bool:
    # 2024-01 partition is known to exist in the consolidated store
    return (per_contract_1min_dir() / "year=2024" / "month=1" / "data.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="consolidated futures store not present")


def test_carry_returns_value_for_known_gc_date():
    # GC had dense data on 2024-01-15 (GCG4 ~65k volume) -- carry must compute
    val = CarryCalculator().compute("GC", "commodity", date(2024, 1, 15))
    assert isinstance(val, float)


def test_carry_history_nonempty_for_gc_january():
    hist = CarryCalculator().compute_history("GC", "commodity", date(2024, 1, 8), date(2024, 1, 20))
    assert hist.height > 0, "carry history empty -> readers still broken"


def test_roll_dates_detected_for_gc_2024():
    rolls = ContinuousContractDataLoader().detect_roll_dates("GC", date(2024, 1, 1), date(2024, 12, 31))
    assert len(rolls) >= 4, f"GC should roll several times in 2024, got {len(rolls)}"
