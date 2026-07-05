from datetime import date

import pytest

from src.data.continuous_contract_loader import (
    ContinuousContractDataLoader,
    _YEAR_DAILY_VOLUME_CACHE,
    continuous_1min_dir,
)
from src.data.futures.paths import roll_volume_dir


@pytest.mark.skipif(not (continuous_1min_dir() / "symbol=ES").exists(), reason="futures store not present")
def test_disk_cache_roundtrip_and_equivalence(monkeypatch, tmp_path):
    root = "ES"
    start, end = date(2010, 6, 7), date(2026, 2, 20)

    ld = ContinuousContractDataLoader()
    baseline_rolls = ld.detect_roll_dates(root, start, end)
    assert baseline_rolls

    cache_dir = roll_volume_dir() / root
    assert cache_dir.exists()
    assert sorted(cache_dir.glob("*.parquet"))

    _YEAR_DAILY_VOLUME_CACHE.clear()
    monkeypatch.setattr(
        "src.data.continuous_contract_loader.per_contract_1min_dir",
        lambda: tmp_path / "nonexistent_per_contract_1min",
    )

    ld2 = ContinuousContractDataLoader()
    from_disk_rolls = ld2.detect_roll_dates(root, start, end)

    assert from_disk_rolls == baseline_rolls


@pytest.mark.skipif(not (continuous_1min_dir() / "symbol=ES").exists(), reason="futures store not present")
def test_falls_back_when_no_disk_cache(monkeypatch, tmp_path):
    root = "ES"
    year = 2015

    _YEAR_DAILY_VOLUME_CACHE.clear()
    monkeypatch.setattr(
        "src.data.continuous_contract_loader.roll_volume_dir",
        lambda: tmp_path / "empty_roll_volume",
    )

    ld = ContinuousContractDataLoader()
    daily = ld._year_daily_symbol_volume(root, year)

    assert not daily.is_empty()
