"""Tests for the daily-raw OHLCV cache builder (mirrors test_build_carry_cache pattern)."""
from __future__ import annotations

import pytest

from src.data.continuous_contract_loader import ContinuousContractDataLoader, continuous_1min_dir
from src.data.futures.paths import daily_raw_dir
from scripts.data.build_daily_raw_cache import build_daily_raw_cache


@pytest.mark.skipif(not (continuous_1min_dir() / "symbol=ES").exists(), reason="futures store not present")
def test_builder_writes_raw_daily():
    build_daily_raw_cache(["ES"])
    fp = daily_raw_dir() / "ES.parquet"
    assert fp.exists()

    import polars as pl

    cached = pl.read_parquet(fp).sort("timestamp")
    expected = ContinuousContractDataLoader().aggregate_to_daily("ES", method="raw").sort("timestamp")
    assert cached.height == expected.height
    assert cached["close"].to_list() == expected["close"].to_list()
