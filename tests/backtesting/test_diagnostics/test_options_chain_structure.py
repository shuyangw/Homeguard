"""Unit tests for the Wave-0 Group B2 option-chain structure primitives.

These test DATA-PROPERTY helpers only. No strategy, no P&L, no positions.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.backtesting.diagnostics import options_chain_structure as ocs


def _row(session, expiry, strike, right, delta, mid, volume=100, oi=1000,
         spread_rel=0.05, quote_valid=True, iv=0.2, und=400.0):
    return {
        "root": "SPY",
        "session_date": session,
        "expiry": expiry,
        "strike": float(strike),
        "right": right,
        "delta": float(delta),
        "mid": mid,
        "bid": None if mid is None else mid - 0.05,
        "ask": None if mid is None else mid + 0.05,
        "spread_rel": spread_rel,
        "quote_valid": quote_valid,
        "dte": (expiry - session).days,
        "volume": volume,
        "oi_eod_lag1": oi,
        "implied_vol": iv,
        "underlying_px": und,
    }


def _frame(rows):
    return pl.DataFrame(rows, schema_overrides={"oi_eod_lag1": pl.Int64})


class TestSelectStrikeByDelta:
    def test_picks_nearest_realized_delta(self):
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([
            _row(s, e, 380, "P", -0.31, 3.0),
            _row(s, e, 370, "P", -0.22, 2.0),
            _row(s, e, 390, "P", -0.45, 4.5),
        ])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 1
        assert out["strike"][0] == 380.0
        assert out["realized_delta"][0] == pytest.approx(-0.31)
        assert out["delta_dev"][0] == pytest.approx(0.01)
        assert bool(out["within_tolerance"][0]) is True

    def test_nan_delta_rows_are_excluded_not_treated_as_zero(self):
        """The NaN-vs-NULL trap: NaN greeks must never be selectable."""
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([
            _row(s, e, 400, "P", float("nan"), 5.0),
            _row(s, e, 380, "P", -0.34, 3.0),
        ])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 1
        assert out["strike"][0] == 380.0

    def test_all_nan_delta_yields_no_selection(self):
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([_row(s, e, 400, "P", float("nan"), 5.0)])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 0

    def test_invalid_quotes_excluded(self):
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([
            _row(s, e, 380, "P", -0.30, None, quote_valid=False),
            _row(s, e, 370, "P", -0.24, 2.0),
        ])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 1
        assert out["strike"][0] == 370.0

    def test_tie_breaks_on_volume_then_oi_then_spread(self):
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([
            _row(s, e, 380, "P", -0.32, 3.0, volume=10, oi=5000),
            _row(s, e, 375, "P", -0.28, 2.5, volume=99, oi=10),
        ])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out["strike"][0] == 375.0

    def test_deviation_beyond_tolerance_flagged_not_dropped_silently(self):
        s, e = date(2020, 1, 6), date(2020, 2, 5)
        df = _frame([_row(s, e, 300, "P", -0.03, 0.2)])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 1
        assert bool(out["within_tolerance"][0]) is False
        assert out["delta_dev"][0] == pytest.approx(0.27)

    def test_single_expiry_chosen_nearest_window_midpoint(self):
        s = date(2020, 1, 6)
        near, far = date(2020, 1, 28), date(2020, 2, 7)  # 22 and 32 dte, mid=28
        df = _frame([
            _row(s, near, 380, "P", -0.30, 3.0),
            _row(s, far, 379, "P", -0.30, 3.4),
        ])
        out = ocs.select_strike_by_delta(df, "P", 0.30, (21, 35))
        assert out.height == 1
        # window midpoint is 28 dte -> the 32-dte expiry is nearer than 22-dte
        assert out["expiry"][0] == far
        assert out["dte"][0] == 32

    def test_dte_window_is_inclusive_and_binding(self):
        s = date(2020, 1, 6)
        df = _frame([_row(s, date(2020, 3, 6), 380, "P", -0.30, 3.0)])  # 60 dte
        assert ocs.select_strike_by_delta(df, "P", 0.30, (21, 35)).height == 0

    def test_multiple_sessions_one_row_each(self):
        rows = []
        for d in (date(2020, 1, 6), date(2020, 1, 7)):
            e = date(2020, 2, 5)
            rows.append(_row(d, e, 380, "P", -0.31, 3.0))
            rows.append(_row(d, e, 370, "P", -0.21, 2.0))
        out = ocs.select_strike_by_delta(_frame(rows), "P", 0.30, (21, 35))
        assert out.height == 2
        assert sorted(out["session_date"].to_list()) == [date(2020, 1, 6), date(2020, 1, 7)]


class TestCalendarHelpers:
    def test_third_friday(self):
        assert ocs.third_friday(2020, 1) == date(2020, 1, 17)
        assert ocs.third_friday(2021, 4) == date(2021, 4, 16)
        assert ocs.third_friday(2026, 2) == date(2026, 2, 20)

    def test_opex_week_membership(self):
        opex = date(2020, 1, 17)
        assert ocs.is_in_opex_week(date(2020, 1, 15), {opex})
        assert ocs.is_in_opex_week(date(2020, 1, 17), {opex})
        assert not ocs.is_in_opex_week(date(2020, 1, 10), {opex})
        assert not ocs.is_in_opex_week(date(2020, 1, 22), {opex})


class TestPullbackFlags:
    def test_single_session_drop_trigger(self):
        closes = np.array([100.0, 99.0, 96.5, 97.0])
        flags = ocs.pullback_flags_1session(closes, threshold=0.02)
        assert list(flags) == [False, False, True, False]

    def test_trailing_max_drawdown_trigger(self):
        closes = np.array([100.0, 99.5, 99.0, 97.5, 97.0])
        flags = ocs.pullback_flags_trailing_max(closes, window=3, threshold=0.02)
        assert flags[0] is np.False_ or not flags[0]
        assert flags[3]


class TestSigmaScaling:
    def test_dte_scaling_formula(self):
        assert ocs.sigma_for_dte(0.20, 252) == pytest.approx(0.20)
        assert ocs.sigma_for_dte(0.20, 63) == pytest.approx(0.10)


class TestGreekCensus:
    def test_spy_2016_excluded_2017_included(self):
        ok = ocs.greek_ok_months("SPY")
        assert (2017, 1) in ok
        assert (2016, 5) not in ok

    def test_qqq_pre_2017_included(self):
        ok = ocs.greek_ok_months("QQQ")
        assert (2013, 5) in ok
