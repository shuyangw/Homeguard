"""#20 London Breakout charges the MEASURED hour-of-week spread, not a flat tier.

The strategy previously booked one constant round-trip spread per run
(tier midpoint x session multiplier), blind to when the fills actually
happened. It trades the London window, which is precisely where the flat model
over-charges: measured GBPUSD round-trip there is ~0.95 bps against the flat
model's ~1.88.
"""
import datetime as dt

import pandas as pd
import pytest

from src.backtesting.costs.fx import (_pip_size, fx_round_trip_bps_at,
                                      fx_round_trip_pips)
from src.backtesting.engine.intraday_order_engine import Bar, OrderEngine
from src.strategies.advanced.fx_london_breakout import LondonBreakoutStrategy

_DAY = dt.date(2024, 1, 10)          # Wednesday
_PIP = _pip_size("GBPUSD")
_OFFSET_PIPS = 3.0
_WIDTH_PIPS = 30
_LO = 1.2500
_HI = _LO + _WIDTH_PIPS * _PIP
_ENTRY_PRICE = _HI + _OFFSET_PIPS * _PIP
_INITIAL_RISK = (_WIDTH_PIPS + _OFFSET_PIPS) * _PIP
# Entry 08:15 UTC and exit 08:20 UTC both fall in the same hour-of-week bucket.
_HOW = _DAY.weekday() * 24 + 8


def _stop_loss_day_r(**kwargs) -> float:
    idx = pd.date_range(f"{_DAY} 00:00", f"{_DAY} 16:00", freq="1min", tz="UTC")
    mid = (_HI + _LO) / 2.0
    df = pd.DataFrame({"open": mid, "high": mid, "low": mid, "close": mid}, index=idx)
    asian = df.index.hour < 7
    df.loc[asian, "high"] = _HI
    df.loc[asian, "low"] = _LO
    brk = (df.index.hour == 8) & (df.index.minute == 15)
    df.loc[brk, ["high", "close"]] = _HI + 5 * _PIP
    stp = (df.index.hour == 8) & (df.index.minute == 20)
    df.loc[stp, ["open", "high", "low", "close"]] = _LO - 1 * _PIP

    strat = LondonBreakoutStrategy("GBPUSD", atr_d1={_DAY: 2 * _WIDTH_PIPS * _PIP},
                                   **kwargs)
    OrderEngine().run([Bar(ts.to_pydatetime(), r.open, r.high, r.low, r.close)
                       for ts, r in df.iterrows()], strat)
    return strat.day_r[_DAY]


def _charged_cost_r(**kwargs) -> float:
    """A full stop loss books -1.0 - cost_R, so the cost is recoverable."""
    return -_stop_loss_day_r(**kwargs) - 1.0


def test_default_charges_the_measured_hour_of_week_cost():
    expected_bps = fx_round_trip_bps_at("GBPUSD", _HOW)
    expected_r = (expected_bps / 1e4 * _ENTRY_PRICE) / _INITIAL_RISK
    assert _charged_cost_r() == pytest.approx(expected_r, rel=1e-9)


def test_measured_london_cost_is_cheaper_than_the_flat_tier_model():
    flat_r = (fx_round_trip_pips("major", session="london") * _PIP) / _INITIAL_RISK
    assert _charged_cost_r() < flat_r


def test_cost_mult_scales_only_the_cost_leg():
    base = _charged_cost_r()
    assert _charged_cost_r(cost_mult=1.5) == pytest.approx(1.5 * base, rel=1e-9)


def test_override_pips_still_produces_the_legacy_flat_charge():
    """The escape hatch stays exact so a flat-cost leg remains reproducible."""
    legacy = (fx_round_trip_pips("major", session="london", override_pips=0.5)
              * _PIP) / _INITIAL_RISK
    assert _charged_cost_r(override_pips=0.5) == pytest.approx(legacy, rel=1e-9)
