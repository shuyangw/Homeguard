"""Real-data regression: these MUST fail before the path repair and pass after.
They would have caught the 2026 consolidation break that the fixture-based
unit tests missed."""
from datetime import date

import pytest

from src.data.futures.paths import per_contract_1min_dir
from src.data.carry_calculator import CarryCalculator
from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.data.derivations.futures.open_interest import aggregate_open_interest


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


def test_definition_lookup_for_known_contract():
    # GCG4 (Feb 2024 gold) is active in the 2024-01 definitions partition
    d = FuturesDefinitionsLoader().get_definition("GCG4", "GC", date(2024, 1, 15))
    assert d.expiration.year == 2024
    assert d.tick_size > 0


def test_aggregate_oi_positive_for_gc():
    # NOTE: brief specified 2024-01-15, but that is MLK Day -- CME does not
    # publish an end-of-session OI stat (stat_type 9) for the holiday itself
    # (verified: no row has timestamp date OR ts_ref date == 2024-01-15 in
    # the statistics partition), so the assertion would fail on real data
    # regardless of path correctness. 2024-01-16 is the next session with a
    # published OI stat and confirms the statistics path repoint works.
    oi = aggregate_open_interest("GC", date(2024, 1, 16))
    assert oi > 0, "aggregate OI zero -> statistics path still broken"


def test_compute_history_raises_when_dataset_dir_missing(monkeypatch, tmp_path):
    # Point the per-contract dir at an empty tmp dir -> whole dataset missing.
    monkeypatch.setattr(
        "src.data.carry_calculator.per_contract_1min_dir",
        lambda: tmp_path / "does_not_exist",
    )
    with pytest.raises(FileNotFoundError):
        CarryCalculator().compute_history("GC", "commodity", date(2024, 1, 8), date(2024, 1, 20))
