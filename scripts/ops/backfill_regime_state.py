#!/usr/bin/env python3
"""Backfill hg_regime_state_code (and companion gauges) into VictoriaMetrics.

Replays the production RAMP regime detector against Alpaca SPY+VIX history
and POSTs timestamped samples to VM's /api/v1/import/prometheus endpoint.

Idempotent: VM dedupes on (series, timestamp), so re-runs are safe.

Live emission reference: scripts/trading/run_live_paper_trading.py:725-774
Spec: docs/superpowers/specs/2026-05-16-grafana-gap-backfill-design.md
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Iterator, List, Tuple

import pandas as pd

LABEL_BASE = 'instance="127.0.0.1:8082",job="homeguard-ramp"'
VM_URL = 'http://127.0.0.1:8428/api/v1/import/prometheus'


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
    since_ts = pd.Timestamp(since).tz_localize(None) if pd.Timestamp(since).tz is None else pd.Timestamp(since)
    until_ts = pd.Timestamp(until).tz_localize(None) if pd.Timestamp(until).tz is None else pd.Timestamp(until)
    spy_index_naive = spy_prices.index.tz_localize(None) if spy_prices.index.tz is not None else spy_prices.index
    mask = (spy_index_naive >= since_ts.normalize()) & (spy_index_naive <= until_ts.normalize())
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
        # 21:00 UTC = 16:00 ET = market close
        close_ts = pd.Timestamp(day).normalize() + pd.Timedelta(hours=21)
        ts_ms = int(close_ts.timestamp() * 1000)
        yield ts_ms, state_code, sma_20, sma_50, sma_200, time_in_state


def main() -> int:
    return 0


if __name__ == '__main__':
    sys.exit(main())
