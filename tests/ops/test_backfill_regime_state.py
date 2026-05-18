"""Tests for scripts/ops/backfill_regime_state.py."""
import pandas as pd
import numpy as np
import pytest
from scripts.ops.backfill_regime_state import format_regime_lines, classify_with_indicators


def test_format_regime_lines_produces_five_metrics():
    ts_ms = 1714435200000  # 2024-04-30 00:00:00 UTC
    lines = format_regime_lines(
        timestamp_ms=ts_ms,
        state_code=3,
        sma_20=432.18,
        sma_50=425.50,
        sma_200=410.00,
        time_in_state_seconds=86400.0,
    )
    assert lines == [
        'hg_regime_state_code{instance="127.0.0.1:8082",job="homeguard-ramp"} 3.0 1714435200000',
        'hg_regime_sma_signal{instance="127.0.0.1:8082",job="homeguard-ramp",period="20"} 432.18 1714435200000',
        'hg_regime_sma_signal{instance="127.0.0.1:8082",job="homeguard-ramp",period="50"} 425.5 1714435200000',
        'hg_regime_sma_signal{instance="127.0.0.1:8082",job="homeguard-ramp",period="200"} 410.0 1714435200000',
        'hg_regime_time_in_state_seconds{instance="127.0.0.1:8082",job="homeguard-ramp"} 86400.0 1714435200000',
    ]


def _synthetic_spy_vix(n: int = 300):
    """Generate enough SPY+VIX history for the detector's 252-day warmup."""
    idx = pd.date_range('2024-01-01', periods=n, freq='B', tz='America/New_York')
    spy = pd.Series(
        400.0 + np.arange(n, dtype=float) * 0.1,  # gentle uptrend
        index=idx, name='spy',
    )
    vix = pd.Series(
        15.0 + np.sin(np.arange(n) / 20.0),  # oscillating VIX
        index=idx, name='vix',
    )
    return spy, vix


def test_classify_with_indicators_raises_on_short_input():
    short_spy = pd.Series([1.0] * 100, index=pd.date_range('2024-01-01', periods=100, freq='B'))
    short_vix = pd.Series([15.0] * 100, index=pd.date_range('2024-01-01', periods=100, freq='B'))
    with pytest.raises(ValueError, match="252"):
        classify_with_indicators(None, short_spy, short_vix)


def test_classify_with_indicators_returns_known_keys():
    from src.strategies.advanced.ramp_strategy import RAMPSignals
    ramp = RAMPSignals(symbols=[])
    spy, vix = _synthetic_spy_vix()
    code, sma_20, sma_50, sma_200 = classify_with_indicators(ramp, spy, vix)
    assert isinstance(code, int)
    assert 0 <= code <= 5  # 0 fallback + 1..5 known regimes
    assert sma_20 > 0
    assert sma_50 > 0
    assert sma_200 > 0
