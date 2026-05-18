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
from typing import List, Tuple

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


def main() -> int:
    return 0


if __name__ == '__main__':
    sys.exit(main())
