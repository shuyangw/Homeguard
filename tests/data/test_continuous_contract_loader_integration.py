"""Integration tests for ContinuousContractDataLoader against real local data."""
from datetime import date

import polars as pl
import pytest

from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.settings import get_local_storage_dir


def _local_data_available() -> bool:
    root = get_local_storage_dir()
    return (
        (root / "futures_1min" / "symbol=ES").exists()
        and (root / "futures_per_contract_1min").exists()
    )


@pytest.mark.skipif(not _local_data_available(), reason="local futures store not present")
def test_es_roll_dates_match_quarterly_cycle():
    """ES rolls quarterly on H/M/U/Z months. In 2024 we expect rolls in
    March, June, September, December."""
    loader = ContinuousContractDataLoader()
    rolls = loader.detect_roll_dates("ES", date(2024, 1, 1), date(2024, 12, 31))
    # At minimum we expect 3-4 rolls spread across the year
    assert len(rolls) >= 3, f"expected >=3 ES rolls in 2024, got {len(rolls)}: {rolls}"
    # Each roll should be in March, June, September, or December (CME H/M/U/Z cycle)
    quarterly = {3, 6, 9, 12}
    for r in rolls:
        assert r.month in quarterly, (
            f"unexpected ES roll in month {r.month} on {r}; expected H/M/U/Z (3, 6, 9, 12)"
        )


@pytest.mark.skipif(not _local_data_available(), reason="local futures store not present")
def test_es_daily_aggregate_finite():
    """Aggregated daily series should produce > 200 trading days in 2024
    and have no null OHLC values."""
    loader = ContinuousContractDataLoader()
    daily = loader.aggregate_to_daily(
        "ES", method="raw",
        start=date(2024, 1, 1), end=date(2024, 12, 31),
    )
    assert daily.shape[0] >= 200, (
        f"expected >=200 trading days in 2024, got {daily.shape[0]}"
    )
    for col in ("open", "high", "low", "close"):
        nulls = daily[col].null_count()
        assert nulls == 0, f"{col} has {nulls} null values"
