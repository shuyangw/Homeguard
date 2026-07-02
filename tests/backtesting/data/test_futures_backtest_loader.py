from datetime import date
import pytest
from src.data.futures.paths import continuous_1min_dir
from src.backtesting.data.futures_backtest_loader import load_daily_panel


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="consolidated futures store not present")


def test_panel_has_roots_and_returns():
    df = load_daily_panel(["ES", "GC"], date(2024, 1, 1), date(2024, 3, 31))
    assert ("ES", "close") in df.columns
    assert ("ES", "ret") in df.columns
    assert ("GC", "close") in df.columns
    assert len(df) > 40  # ~60 trading days in the quarter
    # returns are finite where present
    assert df[("ES", "ret")].dropna().abs().max() < 0.5
