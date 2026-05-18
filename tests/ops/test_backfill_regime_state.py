"""Tests for scripts/ops/backfill_regime_state.py."""
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest
from scripts.ops.backfill_regime_state import (
    format_regime_lines,
    classify_with_indicators,
    iter_regime_history,
    fetch_spy_vix,
    query_vm_earliest_sample,
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


def test_query_vm_earliest_sample_parses_range_query_response():
    """Parse the timestamp of the earliest sample for a metric from VM."""
    fake_response = {
        'status': 'success',
        'data': {
            'resultType': 'matrix',
            'result': [{
                'metric': {'__name__': 'hg_regime_state_code'},
                'values': [
                    [1714435200, '3'],
                    [1714521600, '3'],
                    [1714608000, '2'],
                ],
            }],
        },
    }
    with patch('scripts.ops.backfill_regime_state.urllib.request.urlopen') as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_open.return_value.__enter__.return_value = mock_resp
        earliest = query_vm_earliest_sample('hg_regime_state_code')
    assert earliest == datetime.utcfromtimestamp(1714435200)


def test_query_vm_earliest_sample_returns_none_when_empty():
    """Returns None when VM has no samples for the metric."""
    fake_response = {'status': 'success', 'data': {'result': []}}
    with patch('scripts.ops.backfill_regime_state.urllib.request.urlopen') as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_open.return_value.__enter__.return_value = mock_resp
        assert query_vm_earliest_sample('hg_regime_state_code') is None


def test_query_vm_earliest_sample_aborts_on_connection_error():
    """Raises SystemExit when VM is unreachable."""
    import urllib.error
    with patch('scripts.ops.backfill_regime_state.urllib.request.urlopen',
               side_effect=urllib.error.URLError('Connection refused')):
        with pytest.raises(SystemExit) as exc_info:
            query_vm_earliest_sample('hg_regime_state_code')
        assert exc_info.value.code != 0


def test_query_vm_earliest_sample_picks_min_across_series():
    """Multi-series response: returns the earliest timestamp across all entries."""
    fake_response = {
        'status': 'success',
        'data': {
            'result': [
                {'metric': {'instance': 'old'}, 'values': [[1714608000, '3']]},
                {'metric': {'instance': 'new'}, 'values': [[1714435200, '3']]},  # earlier
            ],
        },
    }
    with patch('scripts.ops.backfill_regime_state.urllib.request.urlopen') as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_open.return_value.__enter__.return_value = mock_resp
        earliest = query_vm_earliest_sample('hg_regime_state_code')
    assert earliest == datetime.utcfromtimestamp(1714435200)


def test_main_dry_run_writes_to_stdout_no_post(monkeypatch, capsys):
    """--dry-run prints OpenMetrics body to stdout and skips the POST."""
    spy, vix = _synthetic_spy_vix(n=300)
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.fetch_spy_vix',
        lambda since, until: (spy, vix),
    )
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.classify_with_indicators',
        lambda rs, s, v: (3, 100.0, 99.0, 98.0),
    )
    posted = []
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state._post_openmetrics',
        lambda body: posted.append(body),
    )
    monkeypatch.setattr(
        'sys.argv',
        ['backfill_regime_state.py', '--since', '2024-08-01', '--until', '2024-08-07', '--dry-run'],
    )
    from scripts.ops.backfill_regime_state import main
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert 'hg_regime_state_code' in captured.out
    assert posted == []  # dry-run: no POST


def test_post_openmetrics_aborts_on_url_error(monkeypatch):
    """VM POST failure raises SystemExit with non-zero code."""
    import urllib.error
    from scripts.ops.backfill_regime_state import _post_openmetrics
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.urllib.request.urlopen',
        lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError('Connection refused')),
    )
    with pytest.raises(SystemExit) as exc_info:
        _post_openmetrics(b'fake metrics body\n')
    assert exc_info.value.code != 0


def test_main_aborts_when_no_since_and_vm_empty(monkeypatch):
    """If --since absent and VM has no samples, abort."""
    monkeypatch.setattr(
        'scripts.ops.backfill_regime_state.query_vm_earliest_sample',
        lambda name: None,
    )
    monkeypatch.setattr('sys.argv', ['backfill_regime_state.py'])
    from scripts.ops.backfill_regime_state import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
