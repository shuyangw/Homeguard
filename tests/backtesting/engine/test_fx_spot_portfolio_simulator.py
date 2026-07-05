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
    # Hold long EURUSD one "year" (366 daily steps) at flat price with a +2%
    # rate differential and no costs. Flat price means MTM is always zero, so
    # the entire equity gain is carry. We drive the sizer's output to a known,
    # round notional (100k units) by solving for the daily_vol that makes
    # size_from_forecast_fx return exactly that, so the expected carry gain
    # can be computed analytically below.
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


def test_carry_accrues_over_weekend_gap():
    # Friday -> Monday is a 3-calendar-day gap in a daily bar sequence (no
    # weekend bars). Carry must accrue for all 3 calendar days on the Monday
    # bar, not just 1 (which is what a naive per-bar accrual would give).
    friday = dt.date(2024, 1, 5)
    monday = dt.date(2024, 1, 8)
    idx = pd.Index([friday, monday])
    price = 1.00
    close = pd.DataFrame({"EURUSD": [price, price]}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0, 1.0]}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.02, 0.02]}, index=idx)
    # Fix a known held position: forecast/vol chosen so units == 100_000 on
    # the Friday rebalance, then held (no rebalance) into Monday via a large
    # enough leverage cap and daily rebalance so day-2 has no new trade delta
    # affecting the carry measurement (carry is computed BEFORE rebalance).
    capital, vol_target = 1_000_000.0, 0.2
    base_to_usd = price * 1.0
    target_units = 100_000.0
    daily_vol = (1.0 * capital * vol_target) / (base_to_usd * target_units * np.sqrt(252))
    forecast = pd.DataFrame({"EURUSD": [10.0, 10.0]}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [daily_vol, daily_vol]}, index=idx)

    sim = FxSpotPortfolioSimulator(capital, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, vol_target, quote_usd, rate_diff)

    # Monday's equity change is pure carry (flat price -> zero MTM) for the
    # 100k units held since Friday, over (Mon - Fri).days == 3 calendar days.
    monday_gain = res.equity_curve.iloc[1] - res.equity_curve.iloc[0]
    expected_one_day_carry = target_units * price * 1.0 * 0.02 / 365.0
    expected_carry = expected_one_day_carry * 3
    assert (monday - friday).days == 3
    assert monday_gain == pytest.approx(expected_carry, rel=1e-6)


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


def test_nan_price_holds_position_no_nan_cost():
    # Day 2 is a missing-data gap (NaN close + NaN quote_usd), as happens for
    # real EURUSD data around 2019-09 / 2020-10-11. The pair must be
    # forward-held through the gap: no flatten-to-zero trade, no NaN cost.
    idx = _dates(3)
    close = pd.DataFrame({"EURUSD": [1.10, np.nan, 1.11]}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0, np.nan, 1.0]}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.0, 0.0, 0.0]}, index=idx)
    forecast = pd.DataFrame({"EURUSD": [10.0, 10.0, 10.0]}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [0.005, 0.005, 0.005]}, index=idx)

    def nonzero_cost(pair, units_traded, price, quote_to_usd):
        return abs(units_traded) * 1.0

    sim = FxSpotPortfolioSimulator(100_000.0, nonzero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, 0.2, quote_usd, rate_diff)

    assert res.equity_curve.notna().all()
    nan_day = idx[1]
    assert not (res.trades["date"] == nan_day).any()
    day1_units = res.trades.loc[res.trades["date"] == idx[0], "units"].iloc[0]
    assert day1_units != 0.0


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
