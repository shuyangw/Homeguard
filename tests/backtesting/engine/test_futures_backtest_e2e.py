from datetime import date

import pytest

from src.data.futures.paths import continuous_1min_dir
from src.backtesting.engine.futures_backtest import run_futures_backtest


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures store not present")


def test_carver_backtest_produces_equity_curve():
    cfg = {
        "strategy": {"universe": ["MES", "MGC", "6E"]},
        "dates": {"start": "2022-01-01", "end": "2023-12-31"},
        "backtest": {"initial_capital": 25000, "vol_target_per_instrument": 0.20,
                     "rebalance": "weekly"},
    }
    result = run_futures_backtest(cfg)
    assert result["n_days"] > 200
    assert "sharpe_ratio" in result["metrics"]
    assert result["equity_curve"][-1] > 0  # account didn't go to zero/negative absurdly
