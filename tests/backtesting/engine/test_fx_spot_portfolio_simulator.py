import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine.fx_spot_portfolio_simulator import FxSpotPortfolioSimulator


def _dates(n):
    return pd.Index([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)])


def _flat(pair, price, n):
    idx = _dates(n)
    return pd.DataFrame({pair: [price] * n}, index=idx)


def _zero_cost(pair, units_traded, price, quote_to_usd):
    return 0.0


def test_golden_carry_flat_price():
    # Hold long EURUSD one "year" (365 daily steps) at flat price with a +2%
    # rate differential and no costs. Carry should compound the equity by ~2%
    # of the held USD notional. We fix units directly via a huge forecast + the
    # sizer, so instead we drive units deterministically: use forecast that
    # yields a known notional by setting daily_vol so units are round.
    n = 366
    idx = _dates(n)
    price = 1.00
    close = pd.DataFrame({"EURUSD": [price] * n}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0] * n}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.02] * n}, index=idx)
    # daily_vol chosen so units come out to exactly 100_000 for forecast 10:
    # units = 1.0 * capital * vol_target / (base_to_usd * daily_vol*sqrt(252))
    capital, vol_target = 1_000_000.0, 0.2
    base_to_usd = price * 1.0
    target_units = 100_000.0
    daily_vol = (1.0 * capital * vol_target) / (base_to_usd * target_units * np.sqrt(252))
    forecast = pd.DataFrame({"EURUSD": [10.0] * n}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [daily_vol] * n}, index=idx)

    sim = FxSpotPortfolioSimulator(capital, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, vol_target, quote_usd, rate_diff)
    # ~365 days of carry at 2%/yr on 100k notional ~= 100k * 0.02 = ~2000 USD
    gain = res.equity_curve.iloc[-1] - capital
    assert gain == pytest.approx(100_000.0 * 0.02, rel=0.05)


def test_usd_base_pnl_conversion():
    # Long USDJPY: price rises 150 -> 151, quote is JPY. PnL in JPY converted
    # to USD via 1/price. With units of USD held, PnL_usd = units*(dPx)*quote_to_usd.
    idx = _dates(2)
    close = pd.DataFrame({"USDJPY": [150.0, 151.0]}, index=idx)
    quote_usd = pd.DataFrame({"USDJPY": [1 / 150.0, 1 / 151.0]}, index=idx)
    rate_diff = pd.DataFrame({"USDJPY": [0.0, 0.0]}, index=idx)
    # Force a fixed long position via forecast/vol on day 1.
    forecast = pd.DataFrame({"USDJPY": [10.0, 10.0]}, index=idx)
    capital, vol_target = 1_000_000.0, 0.2
    target_units = 10_000.0
    base_to_usd = 150.0 * (1 / 150.0)  # = 1.0
    daily_vol = (capital * vol_target) / (base_to_usd * target_units * np.sqrt(252))
    dvol = pd.DataFrame({"USDJPY": [daily_vol, daily_vol]}, index=idx)
    sim = FxSpotPortfolioSimulator(capital, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, vol_target, quote_usd, rate_diff)
    # Day 2 PnL = units * (151-150) * quote_to_usd(day2) = 10_000 * 1 * (1/151)
    day2_pnl = res.equity_curve.iloc[1] - res.equity_curve.iloc[0]
    assert day2_pnl == pytest.approx(10_000.0 * 1.0 * (1 / 151.0), rel=1e-6)


def test_bankruptcy_floor():
    # Massive adverse move with high leverage floors equity at 0, never negative.
    idx = _dates(2)
    close = pd.DataFrame({"EURUSD": [1.00, 0.01]}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0, 1.0]}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.0, 0.0]}, index=idx)
    forecast = pd.DataFrame({"EURUSD": [10.0, 10.0]}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [1e-6, 1e-6]}, index=idx)  # tiny vol -> huge units
    sim = FxSpotPortfolioSimulator(100_000.0, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, 0.2, quote_usd, rate_diff)
    assert (res.equity_curve >= 0).all()
    assert res.equity_curve.iloc[-1] == 0.0
