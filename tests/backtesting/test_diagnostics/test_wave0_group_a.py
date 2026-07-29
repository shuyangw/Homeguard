"""Unit tests for the four Wave-0 Group A data-property measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.diagnostics.wave0_group_a import (
    REGIME_RANK,
    drawdown_shape_census,
    first_hour_trend_events,
    gap_continuation_events,
    regime_downgrade_triggers,
    regime_transitions,
    rv_event_study,
    summarize_signed_returns,
    weekend_variance_stats,
)


def _marks(rows):
    df = pd.DataFrame(rows)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    for col in ("quarantined", "is_early_close"):
        if col not in df:
            df[col] = False
    return df.sort_values("session_date").reset_index(drop=True)


# --- D-014 gap continuation -------------------------------------------------

def test_gap_continuation_qualifies_and_signs_correctly():
    marks = _marks([
        dict(session_date="2024-01-02", open_0930=100.0, or_high=101.0, or_low=99.0,
             c_1000=100.5, c_1001=100.6, c_1030=100.0, c_1031=100.0,
             c_snap=100.0, c_realclose=100.0),
        # gap up +1% vs prior snap 100 -> open 101; confirmation close_1000 > or_high
        dict(session_date="2024-01-03", open_0930=101.0, or_high=101.5, or_low=100.5,
             c_1000=102.0, c_1001=102.0, c_1030=102.0, c_1031=102.0,
             c_snap=103.02, c_realclose=103.02),
    ])
    ev = gap_continuation_events(marks, gap_threshold=0.005)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["direction"] == 1
    assert e["entry"] == pytest.approx(102.0)
    assert e["exit"] == pytest.approx(103.02)
    assert e["signed_bps"] == pytest.approx(1e4 * np.log(103.02 / 102.0))


def test_gap_continuation_requires_confirmation():
    marks = _marks([
        dict(session_date="2024-01-02", open_0930=100.0, or_high=101.0, or_low=99.0,
             c_1000=100.5, c_1001=100.6, c_1030=100.0, c_1031=100.0,
             c_snap=100.0, c_realclose=100.0),
        dict(session_date="2024-01-03", open_0930=101.0, or_high=101.5, or_low=100.5,
             c_1000=101.2, c_1001=101.2, c_1030=101.2, c_1031=101.2,
             c_snap=103.0, c_realclose=103.0),
    ])
    assert len(gap_continuation_events(marks, gap_threshold=0.005)) == 0


def test_gap_continuation_below_threshold_excluded():
    marks = _marks([
        dict(session_date="2024-01-02", open_0930=100.0, or_high=101.0, or_low=99.0,
             c_1000=100.5, c_1001=100.6, c_1030=100.0, c_1031=100.0,
             c_snap=100.0, c_realclose=100.0),
        dict(session_date="2024-01-03", open_0930=100.2, or_high=100.3, or_low=100.0,
             c_1000=100.5, c_1001=100.5, c_1030=100.5, c_1031=100.5,
             c_snap=101.0, c_realclose=101.0),
    ])
    assert len(gap_continuation_events(marks, gap_threshold=0.005)) == 0


def test_gap_down_direction_is_negative_and_short_return_signed():
    marks = _marks([
        dict(session_date="2024-01-02", open_0930=100.0, or_high=101.0, or_low=99.0,
             c_1000=100.5, c_1001=100.6, c_1030=100.0, c_1031=100.0,
             c_snap=100.0, c_realclose=100.0),
        dict(session_date="2024-01-03", open_0930=99.0, or_high=99.2, or_low=98.5,
             c_1000=98.0, c_1001=98.0, c_1030=98.0, c_1031=98.0,
             c_snap=97.0, c_realclose=97.0),
    ])
    ev = gap_continuation_events(marks, gap_threshold=0.005)
    e = ev.iloc[0]
    assert e["direction"] == -1
    assert e["signed_bps"] == pytest.approx(-1e4 * np.log(97.0 / 98.0))
    assert e["signed_bps"] > 0


# --- D-042 first-hour trend -------------------------------------------------

def test_first_hour_trend_event_uses_1031_entry_and_snap_exit():
    marks = _marks([
        dict(session_date="2024-01-03", open_0930=100.0, or_high=100.5, or_low=99.5,
             c_1000=100.2, c_1001=100.2, c_1030=100.5, c_1031=100.6,
             c_snap=101.0, c_realclose=101.0),
    ])
    ev = first_hour_trend_events(marks, threshold=0.0035)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["direction"] == 1
    assert e["entry"] == pytest.approx(100.6)
    assert e["signed_bps"] == pytest.approx(1e4 * np.log(101.0 / 100.6))


def test_first_hour_trend_below_threshold_excluded():
    marks = _marks([
        dict(session_date="2024-01-03", open_0930=100.0, or_high=100.5, or_low=99.5,
             c_1000=100.2, c_1001=100.2, c_1030=100.2, c_1031=100.2,
             c_snap=101.0, c_realclose=101.0),
    ])
    assert len(first_hour_trend_events(marks, threshold=0.0035)) == 0


# --- summary stats ----------------------------------------------------------

def test_summarize_signed_returns_applies_haircut_and_is_deterministic():
    x = np.array([10.0, 20.0, -5.0, 30.0, 0.0])
    a = summarize_signed_returns(x, haircut_bps=2.0, n_boot=200, seed=42)
    b = summarize_signed_returns(x, haircut_bps=2.0, n_boot=200, seed=42)
    assert a["mean_bps_gross"] == pytest.approx(x.mean())
    assert a["mean_bps_net"] == pytest.approx(x.mean() - 2.0)
    assert a["hit_rate"] == pytest.approx(3.0 / 5.0)
    assert a["n"] == 5
    assert a["ci_low_gross"] == b["ci_low_gross"]
    assert a["ci_low_gross"] <= a["mean_bps_gross"] <= a["ci_high_gross"]


def test_summarize_signed_returns_empty():
    out = summarize_signed_returns(np.array([]), haircut_bps=2.0, n_boot=10, seed=42)
    assert out["n"] == 0
    assert np.isnan(out["mean_bps_gross"])


# --- D-040a weekend variance ------------------------------------------------

def test_weekend_variance_stats_splits_weekend_and_control_gaps():
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",  # Tue..Fri
        "2024-01-08", "2024-01-09",                                # Mon, Tue
    ])
    marks = _marks([
        dict(session_date=d, open_0930=100.0 + i, or_high=0.0, or_low=0.0,
             c_1000=0.0, c_1001=0.0, c_1030=0.0, c_1031=0.0,
             c_snap=100.0 + i + 0.5, c_realclose=100.0 + i + 0.5,
             intraday_rv=1e-5)
        for i, d in enumerate(dates)
    ])
    out = weekend_variance_stats(marks)
    gaps = out["gaps"]
    weekend = gaps[gaps["is_weekend"]]
    assert len(weekend) == 1
    assert weekend.iloc[0]["from_date"] == pd.Timestamp("2024-01-05").date()
    assert weekend.iloc[0]["calendar_days"] == 3
    control = gaps[~gaps["is_weekend"]]
    assert len(control) == 4
    assert set(control["calendar_days"]) == {1}


def test_weekend_variance_stats_reports_both_session_variance_definitions():
    dates = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    marks = _marks([
        dict(session_date=d, open_0930=px[i], or_high=0.0, or_low=0.0,
             c_1000=0.0, c_1001=0.0, c_1030=0.0, c_1031=0.0,
             c_snap=px[i] * 1.001, c_realclose=px[i] * 1.001, intraday_rv=1e-5)
        for i, d in enumerate(dates)
    ])
    out = weekend_variance_stats(marks)
    assert out["session_var_close_to_close"] > 0
    assert out["session_var_intraday_rv"] == pytest.approx(1e-5)
    assert out["n_weekend"] >= 3


# --- D-013/030 regime downgrades -------------------------------------------

def test_regime_rank_order():
    assert (REGIME_RANK["STRONG_BULL"] > REGIME_RANK["WEAK_BULL"] >
            REGIME_RANK["SIDEWAYS"] > REGIME_RANK["UNPREDICTABLE"] >
            REGIME_RANK["BEAR"])


def test_regime_downgrade_triggers_detects_strict_rank_decrease():
    reg = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                "2024-01-05", "2024-01-08"]),
        "regime": ["STRONG_BULL", "WEAK_BULL", "WEAK_BULL", "STRONG_BULL", "BEAR"],
    })
    trig = regime_downgrade_triggers(reg)
    assert list(trig["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-08"]
    assert list(trig["transition"]) == ["STRONG_BULL->WEAK_BULL", "STRONG_BULL->BEAR"]


def test_regime_transitions_detects_any_change():
    reg = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "regime": ["SIDEWAYS", "STRONG_BULL", "STRONG_BULL"],
    })
    tr = regime_transitions(reg)
    assert len(tr) == 1
    assert tr.iloc[0]["transition"] == "SIDEWAYS->STRONG_BULL"


def test_drawdown_shape_census_decomposes_gap_vs_grind():
    # 5 sessions: flat, then a pure overnight crash, then flat.
    snaps = [100.0, 100.0, 90.0, 90.0, 90.0]
    opens = [100.0, 100.0, 90.0, 90.0, 90.0]
    dates = pd.bdate_range("2024-01-02", periods=5)
    marks = _marks([
        dict(session_date=d, open_0930=opens[i], or_high=0.0, or_low=0.0,
             c_1000=0.0, c_1001=0.0, c_1030=0.0, c_1031=0.0,
             c_snap=snaps[i], c_realclose=snaps[i], intraday_rv=0.0)
        for i, d in enumerate(dates)
    ])
    trig = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")],
                         "transition": ["STRONG_BULL->BEAR"]})
    ep = drawdown_shape_census(marks, trig, window_sessions=4, meaningful_dd=0.03)
    assert len(ep) == 1
    e = ep.iloc[0]
    assert e["max_dd"] == pytest.approx(0.10, abs=1e-3)
    assert e["gap_share"] == pytest.approx(1.0)
    assert e["shape"] == "GAP"
    assert bool(e["meaningful"])


def test_drawdown_shape_census_grind_case():
    snaps = [100.0, 99.0, 98.0, 97.0, 96.0]
    dates = pd.bdate_range("2024-01-02", periods=5)
    marks = _marks([
        dict(session_date=d, open_0930=snaps[i - 1] if i else snaps[0],
             or_high=0.0, or_low=0.0, c_1000=0.0, c_1001=0.0, c_1030=0.0,
             c_1031=0.0, c_snap=snaps[i], c_realclose=snaps[i], intraday_rv=0.0)
        for i, d in enumerate(dates)
    ])
    trig = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")],
                         "transition": ["WEAK_BULL->SIDEWAYS"]})
    ep = drawdown_shape_census(marks, trig, window_sessions=4, meaningful_dd=0.03)
    e = ep.iloc[0]
    assert e["gap_share"] == pytest.approx(0.0)
    assert e["shape"] == "GRIND"


# --- D-050a RV event study --------------------------------------------------

def test_rv_event_study_peak_offset_detects_pre_transition_peak():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rv = pd.Series(np.full(len(dates), 1e-4), index=dates)
    t_idx = 100
    rv.iloc[t_idx - 2] = 5e-4  # peak two sessions BEFORE the transition
    ev = rv_event_study(rv, pd.DatetimeIndex([dates[t_idx]]), offsets=range(-10, 11),
                        trailing_window=60)
    prof = ev["profile"]
    assert prof.loc[prof["mean_norm"].idxmax(), "offset"] == -2
    assert ev["peak_offset_mean"] == -2


def test_rv_event_study_pre_post_ratio_and_tests():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rv = pd.Series(np.full(len(dates), 1e-4), index=dates)
    tdates = []
    for t_idx in range(60, 190, 20):
        rv.iloc[t_idx + 1: t_idx + 6] = 4e-4  # RV rises AFTER
        tdates.append(dates[t_idx])
    ev = rv_event_study(rv, pd.DatetimeIndex(tdates), offsets=range(-10, 11),
                        trailing_window=60)
    assert ev["n_events"] == len(tdates)
    assert ev["mean_ratio_post_pre"] > 1.5
    assert ev["peak_offset_mean"] > 0
