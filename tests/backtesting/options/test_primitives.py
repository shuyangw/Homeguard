"""Tests for the ten shared primitives P1-P10 (spec v2 Section 3, amended A3).

Negative controls included where a test could otherwise pass vacuously:
  * P8 -- a FULL-SAMPLE percentile must FAIL the backward-looking property test
  * P9/P10 -- appending post-snapshot bars must not move any output
"""
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from src.backtesting.options.primitives import (
    DEFAULT_DTE_EXIT,
    DEFAULT_PROFIT_TAKE,
    HAR_SPEC_FROZEN,
    P1_LOW_DELTA_THRESHOLD,
    DeltaSource,
    HarForecast,
    HedgeMode,
    LeakageError,
    RollTrigger,
    StrikeSelection,
    UnregisteredParameterError,
    assert_backward_looking,
    har_rv_forecast,
    hedge_ledger,
    iv_percentile,
    iv_rank,
    regime_gate,
    roll,
    select_expiry,
    select_strike_by_delta,
    shared_vega_budget,
    should_roll,
    size_by_debit,
    size_by_notional,
    size_by_vega_budget,
    standard_exit,
    trailing_percentile,
    trailing_rank,
    yang_zhang_rv_from_bars,
)
from src.backtesting.options.snapshot import EASTERN_TZ

SESSION = date(2024, 6, 3)


# ==========================================================================
# P1 -- select_strike_by_delta
# ==========================================================================

def _chain(**over) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "root": ["SPY"] * 6,
            "session_date": [SESSION] * 6,
            "expiry": [date(2024, 7, 19)] * 4 + [date(2024, 6, 21)] * 2,
            "dte": [46] * 4 + [18] * 2,
            "strike": [480.0, 500.0, 505.0, 520.0, 500.0, 520.0],
            "right": ["P"] * 6,
            "delta": [-0.05, -0.16, -0.16, -0.40, -0.16, -0.40],
            "delta_smooth": [-0.052, -0.161, -0.159, -0.402, -0.161, -0.402],
            "extrapolated": [False] * 6,
            "implied_vol": [0.22, 0.18, 0.18, 0.16, 0.18, 0.16],
            "iv_smooth": [0.21, 0.18, 0.18, 0.16, 0.18, 0.16],
            "bid": [0.50, 2.00, 2.10, 6.00, 1.00, 4.00],
            "ask": [0.55, 2.06, 2.16, 6.10, 1.05, 4.10],
            "quote_valid": [True] * 6,
            "oi_eod_lag1": [100, 5000, 200, 800, 300, 400],
            "spread_abs": [0.05, 0.06, 0.06, 0.10, 0.05, 0.10],
        }
    )
    for k, v in over.items():
        df[k] = v
    return df


def test_p1_picks_the_nearest_abs_delta_within_the_dte_window():
    sel = select_strike_by_delta(_chain(), right="P", target_delta=-0.40,
                                 dte_window=(30, 60))
    assert isinstance(sel, StrikeSelection)
    assert sel.strike == 520.0
    assert sel.expiry == date(2024, 7, 19)
    assert sel.realized_delta == pytest.approx(-0.40)


def test_p1_respects_the_dte_window():
    sel = select_strike_by_delta(_chain(), right="P", target_delta=-0.40,
                                 dte_window=(10, 25))
    assert sel.expiry == date(2024, 6, 21)
    assert sel.dte == 18


def test_p1_ties_break_to_the_more_liquid_strike():
    """Strikes 500 and 505 both sit at -0.16. Lagged OI picks 500."""
    sel = select_strike_by_delta(_chain(), right="P", target_delta=-0.16,
                                 dte_window=(30, 60))
    assert sel.strike == 500.0
    assert sel.tie_break == "oi_eod_lag1"


def test_p1_never_uses_unlagged_open_interest():
    df = _chain().drop(columns=["oi_eod_lag1"])
    df["oi_eod"] = [1, 2, 999999, 4, 5, 6]  # would flip the tie if read
    sel = select_strike_by_delta(df, right="P", target_delta=-0.16,
                                 dte_window=(30, 60))
    assert sel.tie_break != "oi_eod"
    assert sel.strike in (500.0, 505.0)


def test_p1_rejects_an_explicit_unlagged_tie_break_column():
    with pytest.raises(LeakageError):
        select_strike_by_delta(_chain(), right="P", target_delta=-0.16,
                               dte_window=(30, 60), tie_break_col="oi_eod")


def test_p1_low_delta_rule_reads_the_smoothed_surface():
    assert P1_LOW_DELTA_THRESHOLD == 0.10
    sel = select_strike_by_delta(_chain(), right="P", target_delta=-0.05,
                                 dte_window=(30, 60))
    assert sel.delta_source is DeltaSource.SMOOTH
    assert sel.realized_delta == pytest.approx(-0.052)


def test_p1_at_or_above_the_threshold_uses_shipped_greeks():
    """A3 Section 3.3: 0.10 is NOT below the threshold, so the rule does not bind."""
    sel = select_strike_by_delta(_chain(), right="P", target_delta=-0.10,
                                 dte_window=(30, 60))
    assert sel.delta_source is DeltaSource.SHIPPED


def test_p1_a3_policy_falls_back_to_shipped_where_the_surface_extrapolates():
    df = _chain()
    df["extrapolated"] = [True, False, False, False, False, False]
    sel = select_strike_by_delta(
        df, right="P", target_delta=-0.05, dte_window=(30, 60),
        delta_source=DeltaSource.A3_SMOOTH_NONEXTRAP,
    )
    assert sel.delta_source is DeltaSource.SHIPPED
    assert sel.extrapolated_fallback is True
    assert sel.realized_delta == pytest.approx(-0.05)


def test_p1_a3_policy_uses_the_surface_where_it_is_not_extrapolated():
    sel = select_strike_by_delta(
        _chain(), right="P", target_delta=-0.05, dte_window=(30, 60),
        delta_source=DeltaSource.A3_SMOOTH_NONEXTRAP,
    )
    assert sel.delta_source is DeltaSource.SMOOTH
    assert sel.extrapolated_fallback is False


def test_p1_every_selection_reports_its_source():
    for target in (-0.05, -0.16, -0.40):
        sel = select_strike_by_delta(_chain(), right="P", target_delta=target,
                                     dte_window=(30, 60))
        assert sel.delta_source in (DeltaSource.SMOOTH, DeltaSource.SHIPPED)
        assert sel.realized_delta is not None


def test_p1_excludes_rows_with_no_usable_delta_and_counts_them():
    df = _chain()
    df.loc[1, "delta"] = np.nan
    sel = select_strike_by_delta(df, right="P", target_delta=-0.16,
                                 dte_window=(30, 60))
    assert sel.strike == 505.0
    assert sel.n_excluded_unusable >= 1


def test_p1_returns_none_when_nothing_qualifies():
    assert select_strike_by_delta(_chain(), right="C", target_delta=0.16,
                                  dte_window=(30, 60)) is None


def test_p1_marks_never_come_from_trade_prints():
    df = _chain()
    df["close"] = 99.0
    df["vwap"] = 99.0
    sel = select_strike_by_delta(df, right="P", target_delta=-0.16,
                                 dte_window=(30, 60))
    assert sel.mark == pytest.approx(2.03)


# ==========================================================================
# P2 -- select_expiry
# ==========================================================================

def test_p2_picks_the_expiry_nearest_the_target_dte():
    sel = select_expiry(_chain(), dte_min=10, dte_max=60, prefer="any", target_dte=30)
    assert sel.expiry in (date(2024, 6, 21), date(2024, 7, 19))
    assert abs(sel.dte - 30) == min(abs(18 - 30), abs(46 - 30))


def test_p2_logs_the_single_chosen_expiry():
    sel = select_expiry(_chain(), dte_min=10, dte_max=60, prefer="any", target_dte=46)
    assert sel.expiry == date(2024, 7, 19)
    assert sel.n_candidates == 2


def test_p2_monthly_preference_restricts_to_standard_monthlies():
    df = _chain()
    df.loc[df["expiry"] == date(2024, 7, 19), "expiry"] = date(2024, 7, 17)  # a Wednesday
    sel = select_expiry(df, dte_min=10, dte_max=60, prefer="monthly")
    assert sel.expiry == date(2024, 6, 21)  # the third Friday
    assert sel.prefer == "monthly"


def test_p2_returns_none_when_the_window_is_empty():
    assert select_expiry(_chain(), dte_min=200, dte_max=300, prefer="any") is None


def test_p2_rejects_an_unknown_prefer_value():
    with pytest.raises(ValueError):
        select_expiry(_chain(), dte_min=10, dte_max=60, prefer="weekly")


# ==========================================================================
# P3 -- standard_exit
# ==========================================================================

def test_p3_registered_defaults():
    assert DEFAULT_PROFIT_TAKE == 0.50
    assert DEFAULT_DTE_EXIT == 21


def test_p3_exits_on_the_profit_take():
    d = standard_exit(entry_credit=2.00, current_value=1.00, dte=40)
    assert d.should_exit and d.reason == "profit_take"
    assert d.fraction_captured == pytest.approx(0.50)


def test_p3_does_not_exit_below_the_profit_take():
    d = standard_exit(entry_credit=2.00, current_value=1.20, dte=40)
    assert not d.should_exit and d.reason is None


def test_p3_exits_at_the_dte_floor():
    d = standard_exit(entry_credit=2.00, current_value=1.90, dte=21)
    assert d.should_exit and d.reason == "dte_exit"


def test_p3_profit_take_wins_when_both_fire():
    d = standard_exit(entry_credit=2.00, current_value=0.50, dte=10)
    assert d.reason == "profit_take"


def test_p3_requires_a_positive_credit():
    with pytest.raises(ValueError):
        standard_exit(entry_credit=0.0, current_value=1.0, dte=30)


def test_p3_unmarked_position_does_not_silently_exit():
    d = standard_exit(entry_credit=2.00, current_value=float("nan"), dte=40)
    assert not d.should_exit and d.reason is None
    assert d.unmarked is True


# ==========================================================================
# P4 -- sizers
# ==========================================================================

def test_p4_shared_vega_budget_is_not_registered_and_must_fail_loud():
    """The chain names a 'single shared budget constant' but never states its
    value. Substituting one silently would be an unlogged trial."""
    with pytest.raises(UnregisteredParameterError):
        shared_vega_budget()


def test_p4_sizes_to_the_vega_budget():
    structure = [
        {"side": "sell", "qty": 1, "vega": 0.30},
        {"side": "sell", "qty": 1, "vega": 0.30},
    ]
    r = size_by_vega_budget(structure, budget_vega=6000.0, nav=1_000_000.0)
    assert r.units == pytest.approx(100.0)
    assert r.net_vega_per_unit == pytest.approx(-60.0)


def test_p4_offsetting_legs_reduce_net_vega():
    structure = [
        {"side": "sell", "qty": 1, "vega": 0.30},
        {"side": "buy", "qty": 1, "vega": 0.20},
    ]
    r = size_by_vega_budget(structure, budget_vega=1000.0, nav=1_000_000.0)
    assert r.net_vega_per_unit == pytest.approx(-10.0)
    assert r.units == pytest.approx(100.0)


def test_p4_zero_net_vega_is_not_sizeable():
    structure = [
        {"side": "sell", "qty": 1, "vega": 0.30},
        {"side": "buy", "qty": 1, "vega": 0.30},
    ]
    with pytest.raises(ValueError):
        size_by_vega_budget(structure, budget_vega=1000.0, nav=1_000_000.0)


def test_p4_alternative_sizers_exist():
    assert size_by_debit(debit_per_unit=250.0, budget_usd=5000.0).units == pytest.approx(20.0)
    assert size_by_notional(
        notional_per_unit=53_000.0, budget_usd=530_000.0
    ).units == pytest.approx(10.0)


# ==========================================================================
# P5 -- regime_gate
# ==========================================================================

def test_p5_gate_allows_and_blocks():
    assert regime_gate("STRONG_BULL", {"STRONG_BULL", "WEAK_BULL"}) is True
    assert regime_gate("BEAR", {"STRONG_BULL", "WEAK_BULL"}) is False


def test_p5_unknown_state_fails_loud():
    with pytest.raises(ValueError):
        regime_gate("SUPER_BULL", {"STRONG_BULL"})


def test_p5_missing_state_blocks_rather_than_passing():
    assert regime_gate(None, {"STRONG_BULL"}) is False


# ==========================================================================
# P6 -- hedge_ledger
# ==========================================================================

def _minute_bars(sessions, first="09:30", last="15:59") -> pd.DataFrame:
    frames = []
    for i, s in enumerate(sessions):
        idx = pd.date_range(f"{s} {first}", f"{s} {last}", freq="1min", tz=EASTERN_TZ)
        px = 100.0 + i + np.linspace(0.0, 1.0, len(idx))
        frames.append(pd.DataFrame({"timestamp": idx.tz_convert("UTC"), "close": px,
                                    "open": px, "high": px, "low": px}))
    return pd.concat(frames, ignore_index=True)


def test_p6_requires_an_explicit_slippage_schedule():
    sessions = [date(2024, 6, 3), date(2024, 6, 4)]
    pos = pd.DataFrame({"session_date": sessions, "net_delta_shares": [-50.0, -60.0]})
    with pytest.raises(UnregisteredParameterError):
        hedge_ledger(pos, _minute_bars(sessions), mode=HedgeMode.DAILY_1545,
                     cost_bps_schedule=None)


def test_p6_daily_mode_hedges_once_per_session_at_1545():
    sessions = [date(2024, 6, 3), date(2024, 6, 4)]
    pos = pd.DataFrame({"session_date": sessions, "net_delta_shares": [-50.0, -60.0]})
    led = hedge_ledger(pos, _minute_bars(sessions), mode=HedgeMode.DAILY_1545,
                       cost_bps_schedule={"vol_low": 1.0, "vol_mid": 2.0,
                                          "vol_high": 5.0, "vol_unknown": 5.0})
    assert len(led) == 2
    assert set(led["ts_et"].dt.time) == {time(15, 45)}
    assert led.loc[0, "traded_shares"] == pytest.approx(50.0)
    assert led.loc[1, "traded_shares"] == pytest.approx(10.0)


def test_p6_slippage_is_tiered_by_realized_vol_percentile():
    sessions = [date(2024, 6, 3), date(2024, 6, 4)]
    pos = pd.DataFrame(
        {"session_date": sessions, "net_delta_shares": [-100.0, -200.0],
         "rv_percentile": [0.10, 0.95]}
    )
    sched = {"vol_low": 1.0, "vol_mid": 2.0, "vol_high": 10.0, "vol_unknown": 10.0}
    led = hedge_ledger(pos, _minute_bars(sessions), mode=HedgeMode.DAILY_1545,
                       cost_bps_schedule=sched)
    assert led.loc[0, "slippage_bps"] == pytest.approx(1.0)
    assert led.loc[1, "slippage_bps"] == pytest.approx(10.0)
    assert led.loc[1, "cost_usd"] > 0


def test_p6_band_mode_only_trades_on_a_breach():
    sessions = [date(2024, 6, 3)]
    pos = pd.DataFrame({"session_date": sessions, "net_delta_shares": [-100.0],
                        "sigma_daily": [0.01],
                        "gamma_shares_per_point": [500.0]})
    bars = _minute_bars(sessions)
    sched = {"vol_unknown": 1.0}
    tight = hedge_ledger(pos, bars, mode=HedgeMode.BAND, band=0.1,
                         cost_bps_schedule=sched)
    wide = hedge_ledger(pos, bars, mode=HedgeMode.BAND, band=100.0,
                        cost_bps_schedule=sched)
    assert len(tight) > len(wide)


def test_p6_ledger_records_every_fill():
    sessions = [date(2024, 6, 3), date(2024, 6, 4)]
    pos = pd.DataFrame({"session_date": sessions, "net_delta_shares": [-50.0, -60.0]})
    led = hedge_ledger(pos, _minute_bars(sessions), mode=HedgeMode.DAILY_1545,
                       cost_bps_schedule={"vol_unknown": 2.0})
    for col in ("session_date", "ts_et", "spot", "target_shares", "prior_shares",
                "traded_shares", "slippage_bps", "cost_usd", "mode", "trigger"):
        assert col in led.columns


# ==========================================================================
# P7 -- roll
# ==========================================================================

def test_p7_roll_trigger_on_dte():
    trig = RollTrigger(dte_at_or_below=21)
    assert should_roll(trig, dte=21, abs_delta=0.10)[0] is True
    assert should_roll(trig, dte=22, abs_delta=0.10)[0] is False


def test_p7_roll_trigger_on_delta():
    trig = RollTrigger(abs_delta_at_or_above=0.50)
    ok, reason = should_roll(trig, dte=45, abs_delta=0.55)
    assert ok and reason == "abs_delta"


def test_p7_roll_emits_two_cost_events():
    trig = RollTrigger(dte_at_or_below=21)
    ev = roll(
        current=dict(expiry=date(2024, 6, 21), strike=500.0, right="P", dte=21,
                     abs_delta=0.20),
        trigger=trig,
        new_selection=StrikeSelection(
            strike=495.0, expiry=date(2024, 7, 19), right="P", dte=46,
            realized_delta=-0.16, target_delta=-0.16,
            delta_source=DeltaSource.SHIPPED, extrapolated_fallback=False,
            mark=2.03, bid=2.00, ask=2.06, n_candidates=2,
            n_excluded_unusable=0, tie_break="none",
        ),
    )
    assert ev is not None
    assert len(ev.cost_events) == 2
    assert {e["action"] for e in ev.cost_events} == {"close", "open"}
    assert ev.reason == "dte"


def test_p7_no_roll_when_the_trigger_is_not_met():
    assert roll(current=dict(dte=45, abs_delta=0.10),
                trigger=RollTrigger(dte_at_or_below=21),
                new_selection=None) is None


# ==========================================================================
# P8 -- iv_rank / iv_percentile (strictly backward looking)
# ==========================================================================

def _iv_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n).date
    return pd.DataFrame(
        {
            "root": "SPY",
            "session_date": dates,
            "dte_bucket": 30,
            "atm_iv": 0.15 + 0.05 * rng.standard_normal(n).cumsum() / 30.0,
        }
    )


def test_p8_iv_rank_reads_the_derived_table():
    tab = _iv_table()
    v = iv_rank("SPY", tab["session_date"].iloc[500], window_years=1,
                dte_bucket=30, table=tab)
    assert np.isfinite(v)


def test_p8_iv_rank_may_leave_0_1_when_today_sets_a_new_extreme():
    """Documented, not clipped. The window is STRICTLY PRIOR, so a fresh high
    ranks above 1.0. Clipping would hide the very state the rank is read for,
    and the materialized `iv_rank_daily` already carries unclipped values."""
    s = pd.Series(np.concatenate([np.linspace(0.0, 1.0, 300), [5.0]]))
    out = trailing_rank(s, 252)
    assert out.iloc[300] > 1.0


def test_p8_iv_percentile_reads_the_derived_table():
    tab = _iv_table()
    v = iv_percentile("SPY", tab["session_date"].iloc[500], window_years=1,
                      dte_bucket=30, table=tab)
    assert 0.0 <= v <= 1.0


def test_p8_is_strictly_backward_looking():
    s = pd.Series(np.random.default_rng(3).standard_normal(400).cumsum())
    assert_backward_looking(lambda x: trailing_percentile(x, 252), s)
    assert_backward_looking(lambda x: trailing_rank(x, 252), s)


def test_p8_negative_control_full_sample_percentile_fails():
    """NEGATIVE CONTROL. A full-sample percentile uses future data; the
    backward-looking property test MUST reject it."""
    s = pd.Series(np.random.default_rng(3).standard_normal(400).cumsum())

    def full_sample_percentile(x: pd.Series) -> pd.Series:
        return x.rank(pct=True)

    with pytest.raises(LeakageError):
        assert_backward_looking(full_sample_percentile, s)


def test_p8_current_observation_is_excluded_from_its_own_window():
    s = pd.Series(np.arange(400, dtype=float))
    out = trailing_percentile(s, 252)
    assert out.iloc[300] == pytest.approx(1.0)


# ==========================================================================
# P9 -- Yang-Zhang realized vol
# ==========================================================================

def test_p9_returns_annualized_vol_per_session():
    sessions = pd.bdate_range("2024-01-02", periods=30).date
    rv = yang_zhang_rv_from_bars(_minute_bars(list(sessions)), window_days=10)
    assert isinstance(rv, pd.Series)
    assert len(rv) == 30
    assert np.isfinite(rv.iloc[-1])


def test_p9_enforces_the_1545_truncation_internally():
    """NEGATIVE CONTROL by construction: bars after 15:45 must not move it."""
    sessions = list(pd.bdate_range("2024-01-02", periods=30).date)
    base = yang_zhang_rv_from_bars(_minute_bars(sessions), window_days=10)
    tampered = _minute_bars(sessions).copy()
    et = tampered["timestamp"].dt.tz_convert(EASTERN_TZ)
    late = et.dt.time > time(15, 45)
    tampered.loc[late, ["close", "high", "open", "low"]] *= 5.0
    after = yang_zhang_rv_from_bars(tampered, window_days=10)
    pd.testing.assert_series_equal(base, after)


def test_p9_rejects_a_window_below_two():
    with pytest.raises(ValueError):
        yang_zhang_rv_from_bars(_minute_bars([date(2024, 1, 2)]), window_days=1)


# ==========================================================================
# P10 -- HAR-RV forecast
# ==========================================================================

def _rv_series(n=500) -> pd.Series:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=n)
    v = np.abs(1e-4 + 2e-5 * rng.standard_normal(n).cumsum())
    return pd.Series(v, index=idx)


def test_p10_spec_is_frozen_at_1_5_22():
    assert HAR_SPEC_FROZEN == (1, 5, 22)
    with pytest.raises(ValueError):
        har_rv_forecast(rv_daily=_rv_series(), spec=(1, 5, 10))
    with pytest.raises(ValueError):
        har_rv_forecast(rv_daily=_rv_series(), spec=(1, 10, 22))


def test_p10_returns_a_forecast_and_its_vix_correlation():
    rv = _rv_series()
    vix = pd.Series(15.0 + 3.0 * np.sin(np.arange(len(rv)) / 20.0), index=rv.index)
    out = har_rv_forecast(rv_daily=rv, min_train=252, vix=vix)
    assert isinstance(out, HarForecast)
    assert np.isfinite(out.forecast.dropna()).all()
    assert -1.0 <= out.vix_correlation <= 1.0
    assert out.n_forecast > 0


def test_p10_vix_correlation_is_nan_when_no_vix_supplied():
    out = har_rv_forecast(rv_daily=_rv_series(), min_train=252)
    assert np.isnan(out.vix_correlation)


def test_p10_uses_no_forward_data():
    rv = _rv_series()
    base = har_rv_forecast(rv_daily=rv, min_train=252).forecast
    tampered = rv.copy()
    tampered.iloc[400:] *= 50.0
    after = har_rv_forecast(rv_daily=tampered, min_train=252).forecast
    pd.testing.assert_series_equal(base.iloc[:400], after.iloc[:400])


def test_p10_horizon_must_be_one_under_the_frozen_spec():
    with pytest.raises(ValueError):
        har_rv_forecast(rv_daily=_rv_series(), horizon=5)
