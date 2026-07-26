import numpy as np
import pandas as pd
import pytest

from src.backtesting.validation.degenerate_signal import (DegenerateSignalError,
                                                          assert_not_degenerate,
                                                          constant_columns)

_IDX = pd.date_range("2020-01-01", periods=50, freq="D")


def test_all_zero_series_raises():
    """The carry-unwind failure: a filter that evaluates to 0.0 on every date."""
    with pytest.raises(DegenerateSignalError, match="crash_filter"):
        assert_not_degenerate(pd.Series(0.0, index=_IDX), "crash_filter")


def test_constant_nonzero_series_raises():
    with pytest.raises(DegenerateSignalError):
        assert_not_degenerate(pd.Series(3.7, index=_IDX), "sig")


def test_all_nan_series_raises():
    with pytest.raises(DegenerateSignalError):
        assert_not_degenerate(pd.Series(np.nan, index=_IDX), "sig")


def test_empty_raises():
    with pytest.raises(DegenerateSignalError):
        assert_not_degenerate(pd.Series(dtype=float), "sig")


def test_varying_series_passes():
    assert_not_degenerate(pd.Series(np.arange(50.0), index=_IDX), "sig")


def test_series_that_is_zero_except_one_day_passes():
    """A sparse but real signal is not degenerate; only a CONSTANT one is."""
    s = pd.Series(0.0, index=_IDX)
    s.iloc[10] = 1.0
    assert_not_degenerate(s, "sig")


def test_frame_all_constant_raises():
    df = pd.DataFrame(0.0, index=_IDX, columns=["EURUSD", "USDJPY"])
    with pytest.raises(DegenerateSignalError):
        assert_not_degenerate(df, "forecasts")


def test_frame_with_one_live_column_passes():
    """A strategy legitimately flat on one pair must not halt the run."""
    df = pd.DataFrame(0.0, index=_IDX, columns=["EURUSD", "USDJPY"])
    df["EURUSD"] = np.arange(50.0)
    assert_not_degenerate(df, "forecasts")


def test_constant_columns_reports_the_dead_ones():
    df = pd.DataFrame(0.0, index=_IDX, columns=["EURUSD", "USDJPY", "GBPUSD"])
    df["EURUSD"] = np.arange(50.0)
    assert constant_columns(df) == ["GBPUSD", "USDJPY"]


def test_tolerance_ignores_float_noise():
    """Float dust around a constant is still a constant."""
    s = pd.Series(1.0, index=_IDX) + np.random.default_rng(0).normal(0, 1e-15, 50)
    with pytest.raises(DegenerateSignalError):
        assert_not_degenerate(s, "sig")
