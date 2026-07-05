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


def test_run_fx_backtest_tolerates_missing_pair(monkeypatch, tmp_path):
    # Universe has 4 pairs but the loaded panel is only missing one (EURGBP).
    # FxTrend (CarverMomentumStrategy) must be built with the present-only
    # pair list so forecast_panel's `[self.universe]` column selection does
    # not KeyError on the absent pair. Pre-fix this raised KeyError.
    universe = ["EURUSD", "USDJPY", "EURGBP", "GBPUSD"]
    present_pairs = ["EURUSD", "USDJPY", "GBPUSD"]
    panel = _fake_panel(present_pairs)
    monkeypatch.setattr(fx_backtest, "load_fx_daily_panel", lambda p, s, e: panel)

    def _fake_rates(currencies, index):
        return pd.DataFrame({c: [0.02] * len(index) for c in currencies}, index=index)

    monkeypatch.setattr(fx_backtest, "load_fx_rate_panel", _fake_rates)
    monkeypatch.chdir(tmp_path)

    config = {
        "asset_class": "fx",
        "strategy": {"name": "FxTrend", "universe": universe, "params": {}},
        "dates": {"start": "2020-01-01", "end": "2021-02-01"},
        "backtest": {"initial_capital": 100_000.0, "vol_target_per_instrument": 0.2,
                     "rebalance": "weekly", "leverage_cap": 10.0},
    }
    result = fx_backtest.run_fx_backtest(config, register=False, log_trades=False)
    assert result["n_days"] > 0
    assert result["n_days"] == len(panel)


def test_tier_for_pair_minor_costs_more_than_major():
    # Same |units_traded|, price, quote_to_usd: EURJPY (cross, no USD leg) is
    # minor-tier and must cost more than EURUSD (major-tier).
    cost_fn = fx_backtest._cost_fn_factory()
    eurusd_cost = cost_fn("EURUSD", 100_000.0, 1.10, 1.0)
    eurjpy_cost = cost_fn("EURJPY", 100_000.0, 1.10, 1.0)
    assert fx_backtest._tier_for_pair("EURUSD") == "major"
    assert fx_backtest._tier_for_pair("EURJPY") == "minor"
    assert eurjpy_cost > eurusd_cost
