import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.engine import fx_backtest


def _fake_panel(pairs, n=400):
    idx = pd.Index([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    rng = np.random.default_rng(1)
    frames = {}
    for p in pairs:
        close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
        frames[(p, "close")] = pd.Series(close, index=idx)
        frames[(p, "ret")] = pd.Series(close, index=idx).pct_change()
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_run_fx_backtest_end_to_end(monkeypatch, tmp_path):
    pairs = ["EURUSD", "USDJPY", "EURGBP", "GBPUSD"]
    panel = _fake_panel(pairs)
    monkeypatch.setattr(fx_backtest, "load_fx_daily_panel", lambda p, s, e: panel)

    def _fake_rates(currencies, index):
        return pd.DataFrame({c: [0.02] * len(index) for c in currencies}, index=index)

    monkeypatch.setattr(fx_backtest, "load_fx_rate_panel", _fake_rates)
    monkeypatch.chdir(tmp_path)

    config = {
        "asset_class": "fx",
        "strategy": {"name": "FxTrend", "universe": pairs, "params": {}},
        "dates": {"start": "2020-01-01", "end": "2021-02-01"},
        "backtest": {"initial_capital": 100_000.0, "vol_target_per_instrument": 0.2,
                     "rebalance": "weekly", "leverage_cap": 10.0, "tier": "major"},
    }
    result = fx_backtest.run_fx_backtest(config, register=False, log_trades=True)
    assert result["n_days"] == len(panel)
    assert "sharpe_ratio" in result["metrics"]
    assert result["trade_log_dir"] is not None
    import os
    assert os.path.exists(os.path.join(result["trade_log_dir"], "trades.csv"))
    assert os.path.exists(os.path.join(result["trade_log_dir"], "equity.csv"))
    assert os.path.exists(os.path.join(result["trade_log_dir"], "leverage_utilization.csv"))
