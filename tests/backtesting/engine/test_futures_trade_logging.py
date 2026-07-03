import shutil
from pathlib import Path

import pytest

from src.data.futures.paths import continuous_1min_dir, carry_dir
from src.backtesting.engine.futures_backtest import run_futures_backtest


def _data_present():
    return (continuous_1min_dir() / "symbol=GC").exists() and (carry_dir() / "GC.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures/carry store not present")

_CFG = {
    "strategy": {"name": "FuturesCarry", "universe": ["GC", "CL"]},
    "dates": {"start": "2022-01-03", "end": "2022-03-31"},
    "backtest": {"initial_capital": 1_000_000, "vol_target_per_instrument": 0.20,
                 "rebalance": "weekly", "cost_mult": 1.0},
}


def test_log_trades_false_writes_nothing():
    res = run_futures_backtest(_CFG, register=False, log_trades=False)
    assert res["trade_log_dir"] is None


def test_log_trades_persists_fills_equity_margin():
    res = run_futures_backtest(_CFG, register=False, log_trades=True)
    d = Path(res["trade_log_dir"])
    try:
        assert (d / "trades.csv").exists()
        assert (d / "equity.csv").exists()
        assert (d / "margin_utilization.csv").exists()
        # the trade log holds actual fills, not an empty file
        assert sum(1 for _ in (d / "trades.csv").open()) > 1  # header + >=1 fill
    finally:
        shutil.rmtree(d, ignore_errors=True)
