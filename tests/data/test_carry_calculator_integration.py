"""Integration tests for CarryCalculator against real ES/GC data."""
from datetime import date

import pytest

from src.data.carry_calculator import CarryCalculator


def _local_data_available() -> bool:
    from src.data.futures.paths import per_contract_1min_dir
    return (per_contract_1min_dir() / "year=2024" / "month=1" / "data.parquet").exists()


@pytest.mark.skipif(not _local_data_available(), reason="local futures store not present")
def test_es_carry_sane_magnitude_2024():
    """ES equity_index carry across 2024 should have a sane median (|<0.10|)."""
    cc = CarryCalculator()
    hist = cc.compute_history("ES", "equity_index",
                              date(2024, 1, 1), date(2024, 12, 31))
    assert hist.shape[0] > 200, f"expected >200 days, got {hist.shape[0]}"
    median_carry = hist["carry"].median()
    assert abs(median_carry) < 0.10, f"ES carry median {median_carry:.4f} unreasonable"


@pytest.mark.skipif(not _local_data_available(), reason="local futures store not present")
def test_gc_carry_sane_magnitude_2024():
    """GC commodity carry across 2024 should have a sane median (|<0.10|)."""
    cc = CarryCalculator()
    hist = cc.compute_history("GC", "commodity",
                              date(2024, 1, 1), date(2024, 12, 31))
    assert hist.shape[0] > 200
    median_carry = hist["carry"].median()
    assert abs(median_carry) < 0.10
