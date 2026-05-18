"""Tests for scripts/ops/backfill_regime_state.py."""
from datetime import datetime

import pandas as pd
import numpy as np
import pytest
from scripts.ops.backfill_regime_state import (
    format_regime_lines,
    classify_with_indicators,
    iter_regime_history,
    fetch_spy_vix,
)


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
    """Generate enough SPY+VIX history for the detector's 252-day warmup.

    Uses session-open-like timestamps (09:30 in NY tz) to match real Alpaca daily bars.
    """
    idx = pd.date_range('2024-01-01', periods=n, freq='B', tz='America/New_York') + pd.Timedelta(hours=9, minutes=30)
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


def test_iter_regime_history_skips_non_trading_days(monkeypatch):
    """Iterator emits one tuple per Alpaca-returned trading day, weekends skipped."""
    spy, vix = _synthetic_spy_vix(n=300)
    # Use a fixed-classifier mock so we focus on date iteration.
    def fake_classify(rs, s, v):
        return 3, 100.0, 99.0, 98.0
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.classify_with_indicators',
        fake_classify,
    )
    since = datetime(2024, 8, 1)
    until = datetime(2024, 8, 7)  # Thursday
    results = list(iter_regime_history(spy, vix, since, until))
    # Aug 1-7 has 5 trading days (no holidays in that window)
    assert len(results) == 5
    for ts_ms, code, sma_20, sma_50, sma_200, tis in results:
        assert code == 3
        assert sma_20 == 100.0
    # Timestamps are sorted ascending
    timestamps = [r[0] for r in results]
    assert timestamps == sorted(timestamps)


def test_iter_regime_history_includes_until_day_with_session_open_timestamps(monkeypatch):
    """Regression: Alpaca daily bars (09:30 ET timestamps) must include until day."""
    spy, vix = _synthetic_spy_vix(n=300)
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.classify_with_indicators',
        lambda rs, s, v: (3, 100.0, 99.0, 98.0),
    )
    since = datetime(2024, 8, 1)
    until = datetime(2024, 8, 7)
    results = list(iter_regime_history(spy, vix, since, until))
    # Aug 7 is Wednesday; with session-open timestamps the bar MUST be included.
    assert len(results) == 5
    last_ts_ms = results[-1][0]
    # The last timestamp should fall on Aug 7 (UTC): 2024-08-07 21:00 UTC = 1723064400 sec
    expected_last = int(pd.Timestamp('2024-08-07 21:00:00+00:00').timestamp() * 1000)
    assert last_ts_ms == expected_last


def test_time_in_state_resets_on_regime_change(monkeypatch):
    """time_in_state seconds reset to 0 when regime_name transitions."""
    spy, vix = _synthetic_spy_vix(n=300)
    # Inject a regime sequence: 3, 3, 3, 2, 2 - change on the 4th observation.
    sequence = iter([
        (3, 100.0, 99.0, 98.0),
        (3, 101.0, 99.0, 98.0),
        (3, 102.0, 99.0, 98.0),
        (2, 103.0, 99.0, 98.0),
        (2, 104.0, 99.0, 98.0),
    ])
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.classify_with_indicators',
        lambda rs, s, v: next(sequence),
    )
    since = datetime(2024, 8, 1)
    until = datetime(2024, 8, 7)
    results = list(iter_regime_history(spy, vix, since, until))
    assert len(results) == 5
    time_in_state = [r[5] for r in results]
    # Day 1: 0 (first observation), Day 2: 1 day, Day 3: 2 days,
    # Day 4: regime change -> 0, Day 5: 1 day
    one_day = 24 * 3600
    assert time_in_state[0] == 0
    assert time_in_state[1] == one_day
    assert time_in_state[2] == 2 * one_day
    assert time_in_state[3] == 0
    assert time_in_state[4] == one_day


def test_fetch_spy_vix_aborts_when_vix_missing(monkeypatch):
    """If neither Alpaca nor yfinance can serve VIX, abort with non-zero exit."""
    class FakeProvider:
        def get_historical_bars(self, symbol, start, end, timeframe='1D', **kw):
            if symbol == 'SPY':
                idx = pd.date_range('2024-01-01', periods=300, freq='B')
                return pd.DataFrame({'close': 400.0 + np.arange(300) * 0.1}, index=idx)
            elif symbol == '^VIX':
                return None  # explicit VIX-missing case
            else:
                raise AssertionError(f"Unexpected symbol: {symbol}")

    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state._build_provider',
        lambda: FakeProvider(),
    )
    with pytest.raises(SystemExit) as exc_info:
        fetch_spy_vix(datetime(2024, 1, 1), datetime(2024, 12, 31))
    assert exc_info.value.code != 0
