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
from typing import List

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


def main() -> int:
    return 0


if __name__ == '__main__':
    sys.exit(main())
