#!/usr/bin/env python3
"""Backfill hg_regime_state_code (and companion gauges) into VictoriaMetrics.

Replays the production RAMP regime detector against Alpaca SPY+VIX history
and POSTs timestamped samples to VM's /api/v1/import/prometheus endpoint.

Idempotent: VM dedupes on (series, timestamp), so re-runs are safe.

Live emission reference: scripts/trading/run_live_paper_trading.py:725-774
Spec: docs/superpowers/specs/2026-05-16-grafana-gap-backfill-design.md
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Tuple

import pandas as pd

LABEL_BASE = 'instance="127.0.0.1:8082",job="homeguard-ramp"'
VM_URL = 'http://127.0.0.1:8428/api/v1/import/prometheus'
VM_QUERY_RANGE_URL = 'http://127.0.0.1:8428/api/v1/query_range'
VIX_SYMBOL = '^VIX'  # yfinance convention; CBOE volatility index
SPY_SYMBOL = 'SPY'


def format_regime_lines(
    timestamp_ms: int,
    state_code: int,
    sma_20: float,
    sma_50: float,
    sma_200: float,
    time_in_state_seconds: float,
) -> List[str]:
    """Produce five Prometheus exposition lines for one regime sample."""
    return [
        f'hg_regime_state_code{{{LABEL_BASE}}} {float(state_code)} {timestamp_ms}',
        f'hg_regime_sma_signal{{{LABEL_BASE},period="20"}} {sma_20} {timestamp_ms}',
        f'hg_regime_sma_signal{{{LABEL_BASE},period="50"}} {sma_50} {timestamp_ms}',
        f'hg_regime_sma_signal{{{LABEL_BASE},period="200"}} {sma_200} {timestamp_ms}',
        f'hg_regime_time_in_state_seconds{{{LABEL_BASE}}} {time_in_state_seconds} {timestamp_ms}',
    ]


def classify_with_indicators(
    ramp_signals,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
) -> Tuple[int, float, float, float]:
    """Run the live regime detector and return (state_code, sma_20, sma_50, sma_200).

    Mirrors the live emission path in scripts/trading/run_live_paper_trading.py:
    _emit_strategy_specific_metrics. state_code maps via MarketRegimeDetector.REGIMES
    (1-indexed; unknown regime falls back to 0).

    Precondition: both spy_prices and vix_prices must contain at least 252 rows;
    otherwise the underlying detector short-circuits and SMA indicators would be zero.
    """
    if len(spy_prices) < 252 or len(vix_prices) < 252:
        raise ValueError(
            f"classify_with_indicators requires >= 252 rows for both series "
            f"(got spy={len(spy_prices)}, vix={len(vix_prices)}; threshold=252)"
        )

    from src.strategies.advanced.market_regime_detector import MarketRegimeDetector

    regime_name, _ = ramp_signals.detect_regime(spy_prices, vix_prices)
    state_code = MarketRegimeDetector.REGIMES.get(regime_name, 0)

    detector = getattr(ramp_signals, 'regime_detector', None)
    indicators = getattr(detector, 'last_indicators', None) or {}
    sma_20 = float(indicators.get('sma_20') or 0.0)
    sma_50 = float(indicators.get('sma_50') or 0.0)
    sma_200 = float(indicators.get('sma_200') or 0.0)
    return state_code, sma_20, sma_50, sma_200


def iter_regime_history(
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    since: datetime,
    until: datetime,
) -> Iterator[Tuple[int, int, float, float, float, float]]:
    """Yield (timestamp_ms, state_code, sma_20, sma_50, sma_200, time_in_state).

    One sample per trading day in [since, until]. Trading days are inferred from
    the index of `spy_prices` (Alpaca-returned bars). Time-in-state seconds are
    tracked across the loop and reset to 0 on each regime transition.

    Timestamp for each sample is 21:00 UTC (NYSE close) on the trading day.
    """
    from src.strategies.advanced.ramp_strategy import RAMPSignals
    ramp = RAMPSignals(symbols=[])

    # Trading days = SPY index entries inside [since, until]
    since_ts = pd.Timestamp(since)
    if since_ts.tz is not None:
        since_ts = since_ts.tz_localize(None)
    until_ts = pd.Timestamp(until)
    if until_ts.tz is not None:
        until_ts = until_ts.tz_localize(None)
    spy_index_naive = spy_prices.index.tz_localize(None) if spy_prices.index.tz is not None else spy_prices.index
    # Upper bound is until_ts.normalize() + 1 day so Alpaca daily bars timestamped
    # at session-open (09:30 ET = 13:30 UTC) on the until day are included.
    mask = (spy_index_naive >= since_ts.normalize()) & (spy_index_naive <= until_ts.normalize() + pd.Timedelta(days=1))
    trading_days = spy_prices.index[mask]

    prev_state_code: int = -1
    prev_change_day: int = 0  # index into trading_days where last change happened
    for i, day in enumerate(trading_days):
        spy_slice = spy_prices.loc[:day]
        vix_slice = vix_prices.loc[:day]
        state_code, sma_20, sma_50, sma_200 = classify_with_indicators(
            ramp, spy_slice, vix_slice,
        )
        if state_code != prev_state_code:
            prev_change_day = i
            prev_state_code = state_code
        time_in_state = (i - prev_change_day) * 24 * 3600.0
        # 21:00 UTC = 16:00 ET = market close.
        # Build close_ts in UTC explicitly: tz-naive .timestamp() treats the value as
        # local wall-clock time (host-dependent). If `day` is tz-aware, convert to UTC
        # then normalize to UTC-midnight before adding 21h. If naive, localize to UTC.
        day_ts = pd.Timestamp(day)
        if day_ts.tz is not None:
            day_ts = day_ts.tz_convert('UTC')
        else:
            day_ts = day_ts.tz_localize('UTC')
        close_ts = day_ts.normalize() + pd.Timedelta(hours=21)
        ts_ms = int(close_ts.timestamp() * 1000)
        yield ts_ms, state_code, sma_20, sma_50, sma_200, time_in_state


def _build_provider():
    """Construct the data provider chain. Extracted for testability."""
    from src.data.providers.factory import create_data_provider
    return create_data_provider()  # No broker needed for historical bars


def fetch_spy_vix(since: datetime, until: datetime) -> Tuple[pd.Series, pd.Series]:
    """Fetch SPY + VIX daily close series with 280-day warmup buffer.

    Aborts the script (SystemExit) if either series is unavailable from both
    Alpaca and yfinance fallback. The detector needs 252 days of VIX history
    for percentile calculations, so a 280-trading-day warmup is requested.
    """
    from src.utils.logger import logger

    fetch_start = since - timedelta(days=400)  # 280 trading days = ~400 calendar
    provider = _build_provider()
    spy_df = provider.get_historical_bars(SPY_SYMBOL, fetch_start, until, timeframe='1D')
    vix_df = provider.get_historical_bars(VIX_SYMBOL, fetch_start, until, timeframe='1D')
    if spy_df is None or spy_df.empty:
        logger.error(f'[backfill] SPY data unavailable ({SPY_SYMBOL}) from all providers; aborting')
        raise SystemExit(2)
    if vix_df is None or vix_df.empty:
        logger.error(f'[backfill] VIX data unavailable ({VIX_SYMBOL}) from all providers; aborting')
        raise SystemExit(2)
    return spy_df['close'], vix_df['close']


def query_vm_earliest_sample(metric_name: str) -> Optional[datetime]:
    """Query VM for the earliest existing sample of `metric_name` from the
    RAMP scrape (job="homeguard-ramp"). Scans all returned series and returns
    the minimum timestamp across them, or None if no samples exist.

    Uses a 1-year range query at hourly resolution.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    selector = f'{metric_name}{{job="homeguard-ramp"}}'
    params = urllib.parse.urlencode({
        'query': selector,
        'start': int(start.timestamp()),
        'end': int(end.timestamp()),
        'step': '3600',
    })
    url = f'{VM_QUERY_RANGE_URL}?{params}'
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        from src.utils.logger import logger
        logger.error(f'[backfill] VM unreachable at {VM_QUERY_RANGE_URL}: {exc}')
        raise SystemExit(2)
    result = body.get('data', {}).get('result') or []
    if not result:
        return None
    earliest = None
    for entry in result:
        values = entry.get('values') or []
        if not values:
            continue
        ts = float(values[0][0])
        if earliest is None or ts < earliest:
            earliest = ts
    if earliest is None:
        return None
    return datetime.utcfromtimestamp(earliest)


def main() -> int:
    return 0


if __name__ == '__main__':
    sys.exit(main())
