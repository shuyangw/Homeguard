"""Tests for load_universe_panel."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from src.research.ramp_phase4.data import load_universe_panel


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
    with patch('src.research.ramp_phase4.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = fake_panel
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert isinstance(result, pd.DataFrame)
    assert 'AAPL' in result.columns


def test_load_universe_panel_includes_spy_and_vix(fake_universe_csv, fake_panel):
    with patch('src.research.ramp_phase4.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = fake_panel
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert 'SPY' in result.columns
    assert 'VIX' in result.columns


def test_load_universe_panel_propagates_nans_for_delisted_symbol(fake_universe_csv):
    idx = pd.date_range('2024-01-01', periods=10, freq='B')
    panel_with_delisting = pd.DataFrame({
        'AAPL': [100.0 + i for i in range(10)],
        'MSFT': [200.0 + i for i in range(10)],
        'GOOG': [150.0 + i if i < 5 else float('nan') for i in range(10)],
        'SPY': [400.0 + i for i in range(10)],
        'VIX': [15.0 + i * 0.1 for i in range(10)],
    }, index=idx)
    with patch('src.research.ramp_phase4.data._read_closes_from_parquet') as mock_read:
        mock_read.return_value = panel_with_delisting
        result = load_universe_panel(
            fake_universe_csv,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )
    assert result['GOOG'].iloc[-1] != result['GOOG'].iloc[-1]  # NaN check via !=
