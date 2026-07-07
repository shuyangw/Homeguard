import numpy as np
import pandas as pd

from src.backtesting.signals.carry_unwind import (
    compute_unwind_score, _trailing_zscore, currency_strength)


def _calm_panel(n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    pairs = ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    return pd.DataFrame(
        {p: 100.0 + np.cumsum(rng.normal(0, 0.05, n)) for p in pairs}, index=idx)


def test_trailing_zscore_is_causal_and_nan_free():
    s = pd.Series(np.arange(300, dtype=float))
    z = _trailing_zscore(s, 100)
    assert not z.isna().any()
    # truncating the future must not change past z-values (causality)
    z_trunc = _trailing_zscore(s.iloc[:200], 100)
    pd.testing.assert_series_equal(z.iloc[:200], z_trunc, check_names=False)


def test_currency_strength_rises_when_currency_appreciates():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # AUDJPY rising = AUD appreciating vs JPY -> AUD strength up, JPY down
    panel = pd.DataFrame({"AUDJPY": np.linspace(80, 90, 50)}, index=idx)
    strength = currency_strength(panel)
    assert strength["AUD"].iloc[-1] > strength["AUD"].iloc[0]
    assert strength["JPY"].iloc[-1] < strength["JPY"].iloc[0]


def test_score_is_high_on_a_risk_off_day():
    panel = _calm_panel(400)
    # Engineer a risk-off shock in the last 3 days: JPY and CHF appreciate
    # (their crosses fall), AUDJPY vol spikes, gold jumps.
    for p in ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "NZDJPY"]:
        panel.iloc[-3:, panel.columns.get_loc(p)] *= 0.90  # crosses crash -> JPY/CHF up
    panel.iloc[-3:, panel.columns.get_loc("XAUUSD")] *= 1.08  # gold bid
    score = compute_unwind_score(panel)
    assert score.iloc[-1] > score.iloc[:-10].mean() + 2.0


def test_score_is_causal_and_nan_free():
    panel = _calm_panel(400)
    score = compute_unwind_score(panel)
    assert not score.isna().any()
    score_trunc = compute_unwind_score(panel.iloc[:250])
    pd.testing.assert_series_equal(
        score.iloc[:250], score_trunc, check_names=False)


def test_score_handles_missing_inputs():
    # No XAUUSD, no CHF crosses -> those terms degrade to 0, no crash.
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    panel = pd.DataFrame({"AUDJPY": 80.0 + np.arange(300) * 0.01}, index=idx)
    score = compute_unwind_score(panel)
    assert not score.isna().any()
    assert len(score) == 300


def test_currency_strength_rises_when_chf_appreciates():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # EURCHF falling = fewer CHF per EUR = CHF appreciating vs EUR
    panel = pd.DataFrame({"EURCHF": np.linspace(1.10, 1.00, 50)}, index=idx)
    strength = currency_strength(panel)
    assert strength["CHF"].iloc[-1] > strength["CHF"].iloc[0]
    assert strength["EUR"].iloc[-1] < strength["EUR"].iloc[0]


def test_chf_appreciation_raises_score():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(7)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, 300))
    close[-3:] = close[-4] * np.array([0.97, 0.95, 0.93])  # EURCHF drops -> CHF up
    panel = pd.DataFrame({"EURCHF": close}, index=idx)
    score = compute_unwind_score(panel)
    assert score.iloc[-1] > 0.0
