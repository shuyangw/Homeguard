"""Tests for reports.py builder."""
from datetime import datetime
import pandas as pd
import pytest

from src.research.ramp_phase4.reports import build_variant_report
from src.research.ramp_phase4.reports import build_parity_report


def _fake_records(n=10, regime='STRONG_BULL', daily_return=0.001):
    out = []
    pv = 100000.0
    for i in range(n):
        pv *= (1 + daily_return)
        out.append(type('R', (), {
            'date': datetime(2024, 1, i + 1),
            'regime': regime,
            'portfolio_value': pv,
            'daily_return': daily_return,
            'turnover_usd': 5000.0,
            'cost_usd': 2.5,
            'target_weights': {},
            'realized_weights': {},
        })())
    return out


def test_build_variant_report_returns_markdown_string():
    records = _fake_records()
    md = build_variant_report(
        variant_id='V01',
        variant_description='Test',
        records_by_cost_bps={0.0: records, 5.0: records},
        git_commit='abc123',
        universe_csv='config/universes/sp500-2025.csv',
        timing_mode='near_close',
    )
    assert isinstance(md, str)
    assert '# Phase 4 V01' in md
    assert '0 bps' in md or '0.0 bps' in md
    assert '5 bps' in md or '5.0 bps' in md


def test_build_variant_report_includes_regime_attribution_section():
    records = _fake_records()
    md = build_variant_report(
        variant_id='V01',
        variant_description='Test',
        records_by_cost_bps={5.0: records},
        git_commit='abc123',
        universe_csv='config/universes/sp500-2025.csv',
        timing_mode='near_close',
    )
    assert '## Regime attribution' in md
    assert 'STRONG_BULL' in md


def test_build_parity_report_produces_side_by_side_table():
    v01_records = _fake_records(daily_return=0.001)
    v03_records = _fake_records(daily_return=0.0008)  # slightly worse
    md = build_parity_report(v01_records=v01_records, v03_records=v03_records, cost_bps=5.0)
    assert '# Phase 4 V01 vs V03' in md
    assert '| V01 |' in md
    assert '| V03 |' in md
    assert 'EXT-OOS Sharpe' in md or 'Sharpe' in md
