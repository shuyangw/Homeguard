"""Tests for scripts/ops/backfill_regime_state.py."""
from scripts.ops.backfill_regime_state import format_regime_lines


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
