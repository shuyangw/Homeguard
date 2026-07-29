"""Unit tests for Wave-0 Group A session-mark extraction.

These are DATA-PROPERTY measurement helpers, not a backtest. No P&L here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.diagnostics.session_bars import (
    build_session_marks,
    infer_bar_label_convention,
)


def _synthetic_session(day: str, close_hhmm: str = "16:00", start_hhmm: str = "04:00",
                       end_hhmm: str = "19:59", base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(f"{day} {start_hhmm}", f"{day} {end_hhmm}", freq="1min",
                        tz="America/New_York")
    n = len(idx)
    close = base + np.arange(n) * 0.01
    return pd.DataFrame({
        "ny": idx,
        "open": close - 0.005,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        "volume": np.full(n, 100.0),
    })


def test_infer_bar_label_convention_start():
    bars = _synthetic_session("2024-03-05")
    conv, evidence = infer_bar_label_convention(bars["ny"])
    assert conv == "bar_start"
    assert "04:00" in evidence


def test_build_session_marks_extracts_registered_minutes():
    bars = _synthetic_session("2024-03-05")
    schedule = {pd.Timestamp("2024-03-05").date(): pd.Timestamp("2024-03-05 16:00",
                                                                tz="America/New_York")}
    marks = build_session_marks(bars, schedule)
    assert len(marks) == 1
    row = marks.iloc[0]
    def close_at(hhmm):
        return float(bars.loc[bars["ny"].dt.strftime("%H:%M") == hhmm, "close"].iloc[0])
    assert row["open_0930"] == pytest.approx(
        float(bars.loc[bars["ny"].dt.strftime("%H:%M") == "09:30", "open"].iloc[0]))
    assert row["c_1000"] == pytest.approx(close_at("10:00"))
    assert row["c_1001"] == pytest.approx(close_at("10:01"))
    assert row["c_1030"] == pytest.approx(close_at("10:30"))
    assert row["c_1031"] == pytest.approx(close_at("10:31"))
    assert row["c_snap"] == pytest.approx(close_at("15:45"))
    assert row["c_realclose"] == pytest.approx(close_at("15:59"))
    assert row["snapshot_minute"] == "15:45"
    assert row["n_regular_bars"] == 390


def test_build_session_marks_opening_range_is_0930_to_0959():
    bars = _synthetic_session("2024-03-05")
    reg = bars[(bars["ny"].dt.strftime("%H:%M") >= "09:30") &
               (bars["ny"].dt.strftime("%H:%M") <= "09:59")]
    schedule = {pd.Timestamp("2024-03-05").date(): pd.Timestamp("2024-03-05 16:00",
                                                                tz="America/New_York")}
    marks = build_session_marks(bars, schedule)
    assert marks.iloc[0]["or_high"] == pytest.approx(reg["high"].max())
    assert marks.iloc[0]["or_low"] == pytest.approx(reg["low"].min())


def test_build_session_marks_early_close_uses_last_bar_before_real_close():
    bars = _synthetic_session("2024-11-29")
    schedule = {pd.Timestamp("2024-11-29").date(): pd.Timestamp("2024-11-29 13:00",
                                                               tz="America/New_York")}
    marks = build_session_marks(bars, schedule)
    row = marks.iloc[0]
    expected = float(bars.loc[bars["ny"].dt.strftime("%H:%M") == "12:59", "close"].iloc[0])
    assert row["c_snap"] == pytest.approx(expected)
    assert row["c_realclose"] == pytest.approx(expected)
    assert row["snapshot_minute"] == "12:59"
    assert bool(row["is_early_close"])


def test_build_session_marks_quarantines_incomplete_session():
    bars = _synthetic_session("2024-03-05")
    bars = bars[bars["ny"].dt.strftime("%H:%M") < "10:20"]
    schedule = {pd.Timestamp("2024-03-05").date(): pd.Timestamp("2024-03-05 16:00",
                                                               tz="America/New_York")}
    marks = build_session_marks(bars, schedule)
    row = marks.iloc[0]
    assert np.isnan(row["c_snap"])
    assert bool(row["quarantined"])


def test_intraday_rv_excludes_overnight_gap():
    bars = _synthetic_session("2024-03-05")
    schedule = {pd.Timestamp("2024-03-05").date(): pd.Timestamp("2024-03-05 16:00",
                                                               tz="America/New_York")}
    marks = build_session_marks(bars, schedule)
    reg = bars[(bars["ny"].dt.strftime("%H:%M") >= "09:30") &
               (bars["ny"].dt.strftime("%H:%M") <= "15:59")]
    r = np.diff(np.log(reg["close"].to_numpy()))
    assert marks.iloc[0]["intraday_rv"] == pytest.approx(float((r ** 2).sum()))
    assert marks.iloc[0]["n_rv_returns"] == len(r)
