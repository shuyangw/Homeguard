import numpy as np
import pandas as pd
import pytest

from src.strategies.advanced.fx_roro_regime_spread import FxRoroRegimeSpread
from src.strategies.registry import get_strategy_class

UNIVERSE = ["AUDJPY", "CHFJPY", "XAUUSD"]
UNIVERSE_4 = ["AUDJPY", "CHFJPY", "XAUUSD", "USDJPY"]

# All scenarios below build a perfectly flat (constant, zero-return) panel
# and inject deliberate level shifts. A perfectly flat leg makes its trailing
# 5-day-return z-score exactly NaN (0/0, since both the numerator and the
# rolling std are exactly zero) -- comparisons against NaN are always False,
# so the entry/exit state machine can never fire spuriously. This gives a
# provably clean "no entry before the injected event" baseline, which a
# randomized or smoothly-oscillating baseline cannot: any smooth or i.i.d.
# baseline's rolling z-score is, by construction, an O(1) (unit-scale)
# quantity regardless of the baseline's absolute amplitude, so it crosses a
# threshold as mild as 1.0 sigma routinely -- empirically confirmed while
# developing these tests (hundreds of random seeds were tried; essentially
# none gave a multi-day window free of a spurious pre-shock crossing).
#
# A single tiny SEED_BLIP at BLIP_DAY (well before any real event, and far
# enough back that it has "aged" out of being the current day) is applied to
# every leg once, purely so each leg's rolling std is nonzero and its z-score
# is defined and small (~0.1) by the time of the real, much larger injected
# event -- without this, the real event's "day before" reference score would
# itself be NaN and the crossing (`prev <= threshold < cur`) could never be
# detected, since NaN comparisons are always False.
BLIP_DAY = 200
ENTRY_DAY = 267  # after the 257-row zscore_window(252)+ret5(5) burn-in


def _flat_panel(n):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"AUDJPY": np.full(n, 90.0), "CHFJPY": np.full(n, 120.0),
                          "XAUUSD": np.full(n, 1500.0)}, index=idx)


def _apply_permanent_jump(panel, col, day_idx, factor):
    """Permanent level shift: multiplies `col` by `factor` from day_idx onward."""
    panel = panel.copy()
    idx_col = panel.columns.get_loc(col)
    panel.iloc[day_idx:, idx_col] = panel.iloc[day_idx:, idx_col] * factor
    return panel


def _apply_ramp(panel, col, day_idx, daily_drift, n_days):
    """Compounding daily drift for n_days starting at day_idx, then held flat
    at the final level (never reverses) -- keeps price moves strictly in one
    direction so a spread built from this leg never moves adversely."""
    panel = panel.copy()
    n = len(panel)
    mult = np.ones(n)
    level = 1.0
    end = min(day_idx + n_days, n)
    for t in range(day_idx, end):
        level *= (1.0 + daily_drift)
        mult[t] = level
    if end < n:
        mult[end:] = level
    idx_col = panel.columns.get_loc(col)
    panel.iloc[:, idx_col] = panel.iloc[:, idx_col].to_numpy() * mult
    return panel


def _seeded_flat_panel(n):
    panel = _flat_panel(n)
    for col in UNIVERSE:
        panel = _apply_permanent_jump(panel, col, BLIP_DAY, 1.0005)
    return panel


def test_registry_resolves():
    cls = get_strategy_class("FxRoroRegimeSpread")
    assert cls is FxRoroRegimeSpread


def test_xauusd_forecast_always_zero():
    panel = _seeded_flat_panel(400)
    panel = _apply_permanent_jump(panel, "AUDJPY", ENTRY_DAY, 1.08)
    panel = _apply_permanent_jump(panel, "CHFJPY", ENTRY_DAY, 0.92)
    strat = get_strategy_class("FxRoroRegimeSpread")(UNIVERSE)
    fc = strat.forecast_panel(panel)
    assert (fc["XAUUSD"] == 0.0).all()


def test_risk_on_ignition_entry_same_day_as_crossing():
    panel = _seeded_flat_panel(400)
    panel = _apply_permanent_jump(panel, "AUDJPY", ENTRY_DAY, 1.08)
    panel = _apply_permanent_jump(panel, "CHFJPY", ENTRY_DAY, 0.92)

    strat = FxRoroRegimeSpread(UNIVERSE)
    score = strat._composite_score(panel)
    scores = score.values
    assert scores[ENTRY_DAY - 1] <= strat.entry_threshold
    assert scores[ENTRY_DAY] > strat.entry_threshold

    fc = strat.forecast_panel(panel)
    assert not (fc.iloc[:ENTRY_DAY].abs() > 0).any().any()  # nothing before
    assert fc["AUDJPY"].iloc[ENTRY_DAY - 1] == 0.0
    assert fc["AUDJPY"].iloc[ENTRY_DAY] == pytest.approx(10.0)
    assert fc["CHFJPY"].iloc[ENTRY_DAY] < 0.0  # beta-weighted short leg


def test_risk_off_mirror_entry_same_day_as_crossing():
    panel = _seeded_flat_panel(400)
    panel = _apply_permanent_jump(panel, "AUDJPY", ENTRY_DAY, 0.92)
    panel = _apply_permanent_jump(panel, "CHFJPY", ENTRY_DAY, 1.08)

    strat = FxRoroRegimeSpread(UNIVERSE)
    score = strat._composite_score(panel)
    scores = score.values
    assert scores[ENTRY_DAY - 1] >= -strat.entry_threshold
    assert scores[ENTRY_DAY] < -strat.entry_threshold

    fc = strat.forecast_panel(panel)
    assert not (fc.iloc[:ENTRY_DAY].abs() > 0).any().any()
    assert fc["AUDJPY"].iloc[ENTRY_DAY - 1] == 0.0
    assert fc["AUDJPY"].iloc[ENTRY_DAY] == pytest.approx(-10.0)
    assert fc["CHFJPY"].iloc[ENTRY_DAY] > 0.0  # beta-weighted long leg


def test_score_recross_zero_exit_flattens_both_legs():
    # A single-day permanent jump produces a 5-day-window return spike that
    # naturally decays as the shock ages out of the trailing 5-day return,
    # so score recrosses 0 a few days after entry -- exercising the
    # signal_reversal exit without needing a second injected shock.
    panel = _seeded_flat_panel(400)
    panel = _apply_permanent_jump(panel, "AUDJPY", ENTRY_DAY, 1.08)
    panel = _apply_permanent_jump(panel, "CHFJPY", ENTRY_DAY, 0.92)

    strat = FxRoroRegimeSpread(UNIVERSE)
    score = strat._composite_score(panel)
    scores = score.values
    assert scores[ENTRY_DAY] > strat.entry_threshold

    exit_day = next(i for i in range(ENTRY_DAY + 1, len(scores)) if scores[i] <= 0.0)

    fc = strat.forecast_panel(panel)
    assert fc["AUDJPY"].iloc[exit_day - 1] == pytest.approx(10.0)  # still held prior day
    assert fc["AUDJPY"].iloc[exit_day] == 0.0
    assert fc["CHFJPY"].iloc[exit_day] == 0.0


def test_max_hold_20_days_exit_when_score_never_recrosses():
    # Sustained one-directional ramp keeps AUDJPY's 5-day return (and hence
    # the composite score) persistently elevated well past the 20-day hold,
    # and keeps the AUDJPY/CHFJPY spread moving only favorably, so neither
    # the recross-0 exit nor the ATR stop can fire -- isolating the
    # time-based exit.
    panel = _seeded_flat_panel(450)
    panel = _apply_ramp(panel, "AUDJPY", ENTRY_DAY, daily_drift=0.006, n_days=60)

    strat = FxRoroRegimeSpread(UNIVERSE)
    score = strat._composite_score(panel)
    scores = score.values
    d0 = next(i for i in range(1, len(scores)) if scores[i - 1] <= 1.0 and scores[i] > 1.0)
    assert d0 + strat.max_hold_days < ENTRY_DAY + 60  # exit must land inside the sustained ramp

    fc = strat.forecast_panel(panel)
    assert not (fc.iloc[:d0].abs() > 0).any().any()
    held = fc["AUDJPY"].iloc[d0:d0 + strat.max_hold_days]
    assert (held == 10.0).all()
    assert fc["AUDJPY"].iloc[d0 + strat.max_hold_days] == 0.0
    assert fc["CHFJPY"].iloc[d0 + strat.max_hold_days] == 0.0

    # Confirm the exit is the time-stop, not the other two exit types:
    assert scores[d0 + strat.max_hold_days] > 0.0  # score never recrossed 0
    log_spread = np.log(panel["AUDJPY"]) - np.log(panel["CHFJPY"])
    assert log_spread.iloc[d0 + strat.max_hold_days] >= log_spread.iloc[d0]  # spread only rose (favorable)


def test_atr_spread_stop_fires_on_adverse_move():
    # The ATR stop is coupled to the same AUDJPY/CHFJPY returns that drive
    # the composite score, so isolating it from natural price action would
    # also perturb the score (risking a same-day signal_reversal exit that
    # preempts it). Testing the state machine directly with hand-crafted
    # score/spread/beta/ATR arrays isolates the ATR-stop branch cleanly.
    strat = FxRoroRegimeSpread(UNIVERSE, atr_window=14, atr_stop_mult=2.5,
                                max_hold_days=20)
    n = 40
    score_vals = np.full(n, 0.0)
    score_vals[5:] = 5.0  # crosses above +1 at day 5 and stays elevated (no recross)
    beta_vals = np.full(n, 1.0)
    atr_vals = np.full(n, 0.01)  # constant ATR proxy -> stop distance = 2.5 * 0.01 = 0.025
    spread_vals = np.zeros(n)
    spread_vals[9] = -0.010   # adverse but within the stop distance
    spread_vals[10:] = -0.030  # adverse move breaches -0.025 threshold

    audjpy_fc, chfjpy_fc = strat._run_state_machine(score_vals, spread_vals, beta_vals, atr_vals)

    assert list(audjpy_fc[5:10]) == [10.0] * 5   # held through day 9
    assert audjpy_fc[10] == 0.0                   # ATR stop exit on day 10
    assert chfjpy_fc[10] == 0.0
    assert audjpy_fc[9] == 10.0 and chfjpy_fc[9] == -10.0


def test_beta_clipped_for_pathological_variance():
    # AUDJPY given a near-zero-variance baseline (amplitude 1e-8) so its
    # trailing-90-day return variance is pathologically small; the entry is
    # triggered purely by a CHFJPY shock (AUDJPY itself never jumps), so the
    # AUDJPY variance window at entry stays near-zero -- the definition of
    # the pathological beta-denominator case the clip guards against.
    n = 400
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    t = np.arange(n)
    audjpy = 90.0 * (1.0 + 1e-8 * np.sin(2 * np.pi * t / 23))
    chfjpy = 120.0 * (1.0 + 0.004 * np.sin(2 * np.pi * t / 17 + 0.5))
    xauusd = 1500.0 * (1.0 + 0.002 * np.sin(2 * np.pi * t / 19 + 1.0))
    panel = pd.DataFrame({"AUDJPY": audjpy, "CHFJPY": chfjpy, "XAUUSD": xauusd}, index=idx)

    D = 300
    panel = _apply_permanent_jump(panel, "CHFJPY", D, 0.90)

    strat = FxRoroRegimeSpread(UNIVERSE)
    score = strat._composite_score(panel)
    scores = score.values
    d0 = next((i for i in range(1, len(scores))
                if scores[i - 1] <= strat.entry_threshold and scores[i] > strat.entry_threshold), None)
    assert d0 is not None
    assert d0 >= strat.beta_window

    audjpy_ret = panel["AUDJPY"].pct_change()
    chfjpy_ret = panel["CHFJPY"].pct_change()
    window_a = audjpy_ret.iloc[d0 - strat.beta_window + 1: d0 + 1]
    window_c = chfjpy_ret.iloc[d0 - strat.beta_window + 1: d0 + 1]
    raw_beta = window_c.cov(window_a) / window_a.var()
    assert raw_beta > strat.beta_clip[1] or raw_beta < strat.beta_clip[0]
    expected_clip = min(max(raw_beta, strat.beta_clip[0]), strat.beta_clip[1])

    fc = strat.forecast_panel(panel)
    beta_used = -fc["CHFJPY"].iloc[d0] / strat.forecast_scale
    assert beta_used == pytest.approx(expected_clip, rel=1e-6)


def test_usdjpy_conversion_pair_never_traded_and_other_legs_unchanged():
    # USDJPY is included in the universe purely so the backtest engine can
    # derive the JPY->USD conversion rate for the AUDJPY/CHFJPY legs -- it
    # must never receive a nonzero forecast, and its presence must not alter
    # the AUDJPY/CHFJPY/XAUUSD forecasts versus the 3-column case for the
    # same underlying AUDJPY/CHFJPY/XAUUSD price paths.
    panel3 = _seeded_flat_panel(400)
    panel3 = _apply_permanent_jump(panel3, "AUDJPY", ENTRY_DAY, 1.08)
    panel3 = _apply_permanent_jump(panel3, "CHFJPY", ENTRY_DAY, 0.92)

    panel4 = panel3.copy()
    panel4["USDJPY"] = 110.0
    panel4 = _apply_permanent_jump(panel4, "USDJPY", BLIP_DAY, 1.0005)

    strat3 = FxRoroRegimeSpread(UNIVERSE)
    strat4 = FxRoroRegimeSpread(UNIVERSE_4)

    fc3 = strat3.forecast_panel(panel3)
    fc4 = strat4.forecast_panel(panel4)

    assert fc4.shape[1] == 4
    assert "USDJPY" in fc4.columns
    assert (fc4["USDJPY"] == 0.0).all()

    pd.testing.assert_series_equal(fc4["AUDJPY"], fc3["AUDJPY"], check_names=False)
    pd.testing.assert_series_equal(fc4["CHFJPY"], fc3["CHFJPY"], check_names=False)
    pd.testing.assert_series_equal(fc4["XAUUSD"], fc3["XAUUSD"], check_names=False)


def test_no_lookahead():
    panel = _seeded_flat_panel(450)
    panel = _apply_ramp(panel, "AUDJPY", ENTRY_DAY, daily_drift=0.006, n_days=60)
    strat = FxRoroRegimeSpread(UNIVERSE)

    fc_full = strat.forecast_panel(panel)
    cutoff = 400
    fc_trunc = strat.forecast_panel(panel.iloc[:cutoff])
    pd.testing.assert_frame_equal(fc_full.iloc[:cutoff], fc_trunc, check_names=False)
