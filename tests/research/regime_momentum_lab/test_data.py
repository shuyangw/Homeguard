"""Tests for load_universe_panel."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from src.research.regime_momentum_lab.data import (
    load_universe_panel,
    _aggregate_symbol_daily,
)


@pytest.fixture
def fake_universe_csv(tmp_path):
    csv = tmp_path / 'tiny_universe.csv'
    csv.write_text('symbol\nAAPL\nMSFT\nGOOG\n')
    return csv


@pytest.fixture
def fake_panel():
    idx = pd.date_range('2024-01-01', periods=10, freq='B')
    return pd.DataFrame({
        'AAPL': [100.0 + i for i in range(10)],
        'MSFT': [200.0 + i for i in range(10)],
        'GOOG': [150.0 + i for i in range(10)],
        'SPY': [400.0 + i for i in range(10)],
        'VIX': [15.0 + i * 0.1 for i in range(10)],
    }, index=idx)


def test_load_universe_panel_returns_wide_dataframe(fake_universe_csv, fake_panel):
    with patch('src.research.regime_momentum_lab.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = fake_panel
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert isinstance(result, pd.DataFrame)
    assert 'AAPL' in result.columns


def test_load_universe_panel_includes_spy_and_vix(fake_universe_csv, fake_panel):
    with patch('src.research.regime_momentum_lab.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = fake_panel
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert 'SPY' in result.columns
    assert 'VIX' in result.columns


def test_aggregate_symbol_daily_groups_by_rth_close(tmp_path):
    sym_dir = tmp_path / 'symbol=FAKE'
    month_dir = sym_dir / 'year=2024' / 'month=1'
    month_dir.mkdir(parents=True)
    # Two RTH days, mix of pre-/post-market bars; RTH last close should win.
    rows = []
    # 2024-01-02 ET: pre-market 08:00, RTH 10:00 + 15:59, post-market 17:00
    rows.append({'timestamp': pd.Timestamp('2024-01-02 13:00', tz='UTC'),
                 'close': 100.0})  # 08:00 ET pre-market
    rows.append({'timestamp': pd.Timestamp('2024-01-02 15:00', tz='UTC'),
                 'close': 101.0})  # 10:00 ET RTH
    rows.append({'timestamp': pd.Timestamp('2024-01-02 20:59', tz='UTC'),
                 'close': 102.0})  # 15:59 ET RTH last
    rows.append({'timestamp': pd.Timestamp('2024-01-02 22:00', tz='UTC'),
                 'close': 999.0})  # 17:00 ET post-market (must be ignored)
    # 2024-01-03 ET
    rows.append({'timestamp': pd.Timestamp('2024-01-03 15:30', tz='UTC'),
                 'close': 110.0})  # 10:30 ET RTH
    rows.append({'timestamp': pd.Timestamp('2024-01-03 20:55', tz='UTC'),
                 'close': 115.0})  # 15:55 ET RTH last
    pd.DataFrame(rows).to_parquet(month_dir / 'data.parquet', index=False)

    daily = _aggregate_symbol_daily(sym_dir, datetime(2024, 1, 1), datetime(2024, 1, 31))
    assert daily is not None
    assert len(daily) == 2
    assert daily.loc['2024-01-02'] == 102.0
    assert daily.loc['2024-01-03'] == 115.0


def test_load_universe_panel_propagates_nans_for_delisted_symbol(fake_universe_csv):
    idx = pd.date_range('2024-01-01', periods=10, freq='B')
    panel_with_delisting = pd.DataFrame({
        'AAPL': [100.0 + i for i in range(10)],
        'MSFT': [200.0 + i for i in range(10)],
        'GOOG': [150.0 + i if i < 5 else float('nan') for i in range(10)],
        'SPY': [400.0 + i for i in range(10)],
        'VIX': [15.0 + i * 0.1 for i in range(10)],
    }, index=idx)
    with patch('src.research.regime_momentum_lab.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = panel_with_delisting
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert result['GOOG'].iloc[-1] != result['GOOG'].iloc[-1]  # NaN check via !=
