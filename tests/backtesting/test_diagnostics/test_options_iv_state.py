"""Unit tests for the Wave 0 Group B1 options IV-state measurement layer."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtesting.diagnostics import options_iv_state as ivs


# ---------------------------------------------------------------------------
# Anti-lookahead: trailing percentile / rank must be strictly backward looking
# ---------------------------------------------------------------------------

def test_trailing_percentile_is_strictly_backward_looking():
    base = pd.Series(np.arange(10, dtype=float))
    a = ivs.trailing_percentile(base, window=4)

    mutated = base.copy()
    mutated.iloc[7:] = 999.0
    b = ivs.trailing_percentile(mutated, window=4)

    # values at index <= 6 must be identical: they depend only on t-1 .. t-4 and t
    pd.testing.assert_series_equal(a.iloc[:7], b.iloc[:7])


def test_trailing_percentile_excludes_current_from_window():
    # window of 4 strictly-prior observations; today's value is compared against them
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0])
    out = ivs.trailing_percentile(s, window=4)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[3])
    # at t=4 the prior window is [1,2,3,4]; today's 0.0 is below none of them
    assert out.iloc[4] == pytest.approx(0.0)


def test_trailing_rank_min_max():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 25.0])
    out = ivs.trailing_rank(s, window=4)
    # prior window [10,20,30,40]; (25-10)/(40-10) = 0.5
    assert out.iloc[4] == pytest.approx(0.5)
    assert np.isnan(out.iloc[3])


def test_trailing_rank_is_backward_looking():
    base = pd.Series(np.linspace(1.0, 10.0, 12))
    a = ivs.trailing_rank(base, window=5)
    mut = base.copy()
    mut.iloc[9:] = -50.0
    b = ivs.trailing_rank(mut, window=5)
    pd.testing.assert_series_equal(a.iloc[:9], b.iloc[:9])


# ---------------------------------------------------------------------------
# NaN-greek filtering -- the null_count trap
# ---------------------------------------------------------------------------

def test_greek_usable_mask_rejects_all_nan_column():
    df = pd.DataFrame({
        "implied_vol": [np.nan, np.nan, np.nan],
        "delta": [np.nan, np.nan, np.nan],
        "quote_valid": [True, True, True],
    })
    assert df["implied_vol"].isnull().sum() == 3  # pandas would flag; arrow would not
    mask = ivs.greek_usable_mask(df)
    assert mask.sum() == 0


def test_greek_usable_mask_rejects_degenerate_zero_iv():
    df = pd.DataFrame({
        "implied_vol": [0.0, 0.18, 9.0, np.nan, 0.22],
        "delta": [1.0, 0.5, 0.5, 0.5, np.nan],
        "quote_valid": [True, True, True, True, True],
    })
    mask = ivs.greek_usable_mask(df)
    assert list(mask) == [False, True, False, False, False]


def test_greek_usable_mask_requires_quote_valid():
    df = pd.DataFrame({
        "implied_vol": [0.2, 0.2],
        "delta": [0.5, 0.5],
        "quote_valid": [True, False],
    })
    assert list(ivs.greek_usable_mask(df)) == [True, False]


# ---------------------------------------------------------------------------
# Delta selection
# ---------------------------------------------------------------------------

def test_select_nearest_abs_delta_picks_closest():
    df = pd.DataFrame({
        "strike": [400.0, 410.0, 420.0],
        "delta": [-0.10, -0.24, -0.40],
        "implied_vol": [0.30, 0.22, 0.18],
    })
    row = ivs.select_nearest_abs_delta(df, target=0.25, tolerance=0.05)
    assert row is not None
    assert row["strike"] == 410.0
    assert abs(row["delta"]) == pytest.approx(0.24)


def test_select_nearest_abs_delta_respects_tolerance():
    df = pd.DataFrame({
        "strike": [400.0, 420.0],
        "delta": [-0.05, -0.60],
        "implied_vol": [0.30, 0.18],
    })
    assert ivs.select_nearest_abs_delta(df, target=0.25, tolerance=0.05) is None


def test_select_nearest_abs_delta_empty():
    df = pd.DataFrame({"strike": [], "delta": [], "implied_vol": []})
    assert ivs.select_nearest_abs_delta(df, target=0.25, tolerance=0.05) is None


# ---------------------------------------------------------------------------
# Monthly (third-Friday) expiry identification
# ---------------------------------------------------------------------------

def test_third_friday_basic():
    assert ivs.third_friday(2023, 6) == date(2023, 6, 16)
    assert ivs.third_friday(2024, 2) == date(2024, 2, 16)
    assert ivs.third_friday(2021, 1) == date(2021, 1, 15)


def test_monthly_expiry_shifts_off_good_friday():
    # 2015-04-03 was Good Friday (the third Friday of April 2015 is 2015-04-17,
    # so use a month where the third Friday IS the holiday): 2003-04-18 Good Friday.
    trading_days = {date(2003, 4, 17)}  # Thursday open, Friday closed
    assert ivs.third_friday(2003, 4) == date(2003, 4, 18)
    got = ivs.monthly_expiry(2003, 4, trading_days=trading_days)
    assert got == date(2003, 4, 17)


def test_monthly_expiry_keeps_friday_when_open():
    trading_days = {date(2023, 6, 16)}
    assert ivs.monthly_expiry(2023, 6, trading_days=trading_days) == date(2023, 6, 16)


def test_is_monthly_expiry_set():
    tds = {date(2023, 6, 16), date(2023, 7, 21)}
    s = ivs.monthly_expiry_set(date(2023, 6, 1), date(2023, 7, 31), trading_days=tds)
    assert date(2023, 6, 16) in s
    assert date(2023, 7, 21) in s
    assert date(2023, 6, 23) not in s


# ---------------------------------------------------------------------------
# Episode counting
# ---------------------------------------------------------------------------

def test_episodes_basic_no_merge():
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=10, freq="D")).date
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    eps = ivs.find_episodes(dates, flag, merge_gap_sessions=0)
    assert len(eps) == 3
    assert eps[0]["start"] == dates[1] and eps[0]["end"] == dates[2]
    assert eps[0]["duration"] == 2


def test_episodes_merge_gap():
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=10, freq="D")).date
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    # gap between episode 1 (ends idx2) and 2 (starts idx5) is 2 sessions -> merged
    eps = ivs.find_episodes(dates, flag, merge_gap_sessions=3)
    assert len(eps) == 2
    assert eps[0]["start"] == dates[1] and eps[0]["end"] == dates[5]


def test_episodes_merge_all():
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=10, freq="D")).date
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    eps = ivs.find_episodes(dates, flag, merge_gap_sessions=30)
    assert len(eps) == 1


def test_episodes_ignores_nan_flag_rows():
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=5, freq="D")).date
    flag = np.array([True, True, True, False, False])
    valid = np.array([True, False, True, True, True])
    eps = ivs.find_episodes(dates, flag, merge_gap_sessions=0, valid=valid)
    # index 1 is not measurable -> excluded, leaving two singleton runs
    assert len(eps) == 2


# ---------------------------------------------------------------------------
# ATM IV term interpolation
# ---------------------------------------------------------------------------

def test_interp_atm_iv_bracketed():
    per_expiry = pd.DataFrame({"dte": [7, 45], "atm_iv": [0.20, 0.30]})
    val, src = ivs.interp_atm_iv(per_expiry, target_dte=26)
    assert val == pytest.approx(0.20 + (0.30 - 0.20) * (26 - 7) / (45 - 7))
    assert src == "interp:7-45"


def test_interp_atm_iv_exact_match():
    per_expiry = pd.DataFrame({"dte": [30, 60], "atm_iv": [0.25, 0.28]})
    val, src = ivs.interp_atm_iv(per_expiry, target_dte=30)
    assert val == pytest.approx(0.25)
    assert src == "exact:30"


def test_interp_atm_iv_no_bracket_returns_nan():
    per_expiry = pd.DataFrame({"dte": [50, 60], "atm_iv": [0.25, 0.28]})
    val, src = ivs.interp_atm_iv(per_expiry, target_dte=30)
    assert np.isnan(val)
    assert src == "no_bracket"


def test_monthly_expiry_set_accepts_pre2015_saturday_listing():
    # Before the Feb-2015 OCC change the standard monthly was DATED the
    # Saturday after the third Friday (e.g. QQQ 2012-07-21, a Saturday).
    s = ivs.monthly_expiry_set(date(2012, 7, 1), date(2012, 7, 31),
                               trading_days={date(2012, 7, 20)})
    assert date(2012, 7, 21) in s   # Saturday listing
    assert date(2012, 7, 20) in s   # Friday listing


# ---------------------------------------------------------------------------
# Greek-census partition gating (the 2017 ETF greeks boundary)
# ---------------------------------------------------------------------------

def test_greek_census_gate_excludes_pre2017_spy(tmp_path):
    csv = tmp_path / "census.csv"
    csv.write_text("root,year,month,greek_status\n"
                   "SPY,2016,1,ALL_NAN\nSPY,2017,1,OK\nQQQ,2016,1,OK\n",
                   encoding="utf-8")
    ok = ivs.greek_ok_partitions("SPY", census_csv=csv)
    assert (2016, 1) not in ok
    assert (2017, 1) in ok


# ---------------------------------------------------------------------------
# Forward drawdown -- a price-series measurement, not a P&L
# ---------------------------------------------------------------------------

def test_forward_max_drawdown():
    c = pd.Series([100.0, 110.0, 88.0, 95.0, 99.0])
    out = ivs.forward_max_drawdown(c, horizon=4)
    assert out.iloc[0] == pytest.approx(88.0 / 110.0 - 1.0)
    assert np.isnan(out.iloc[4])


def test_forward_max_drawdown_monotone_up_is_zero():
    c = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ivs.forward_max_drawdown(c, horizon=3)
    assert out.iloc[0] == pytest.approx(0.0)


def test_trailing_percentile_tolerates_sparse_window_but_not_below_coverage():
    # one NaN inside a 20-wide window must NOT poison the next 20 outputs
    s = pd.Series(np.arange(60, dtype=float))
    s.iloc[5] = np.nan
    out = ivs.trailing_percentile(s, window=20)
    assert np.isfinite(out.iloc[25])
    # below the coverage floor -> NaN, never imputed
    s2 = pd.Series(np.arange(60, dtype=float))
    s2.iloc[5:12] = np.nan
    out2 = ivs.trailing_percentile(s2, window=20)
    assert np.isnan(out2.iloc[25])


def test_trailing_percentile_sparse_window_still_backward_looking():
    base = pd.Series(np.linspace(1.0, 30.0, 40))
    base.iloc[3] = np.nan
    a = ivs.trailing_percentile(base, window=10)
    mut = base.copy()
    mut.iloc[25:] = -99.0
    b = ivs.trailing_percentile(mut, window=10)
    pd.testing.assert_series_equal(a.iloc[:25], b.iloc[:25])
