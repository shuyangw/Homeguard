"""Tests for reports.py builder."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.research.ramp_phase4.reports import build_variant_report
from src.research.ramp_phase4.reports import build_parity_report
from src.research.ramp_phase4.reports import _format_psr_gate


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


def test_build_variant_report_includes_per_period_decomposition():
    """The variant report must include a per-period sub-table under each cost
    tier with the five default period columns plus a 'Full' column.
    """
    # Records spanning 2017-01 through 2026-06 so every default period sees data.
    from src.research.ramp_phase4.reports import DEFAULT_PERIODS

    records = []
    pv = 100000.0
    # ~5 records per year so each default period has at least a handful.
    for year in range(2017, 2027):
        for month in (3, 6, 9, 12):
            pv *= 1.005
            records.append(type('R', (), {
                'date': datetime(year, month, 15),
                'regime': 'STRONG_BULL',
                'portfolio_value': pv,
                'daily_return': 0.005,
                'turnover_usd': 5000.0,
                'cost_usd': 2.5,
                'target_weights': {},
                'realized_weights': {},
            })())

    md = build_variant_report(
        variant_id='V01',
        variant_description='Test',
        records_by_cost_bps={5.0: records},
        git_commit='abc123',
        universe_csv='config/universes/sp500-2025.csv',
        timing_mode='near_close',
    )

    # Per-period heading and all five default period columns + Full must appear.
    assert 'per-period' in md
    for label, _, _ in DEFAULT_PERIODS:
        assert label in md, f'missing per-period column label: {label}'
    # The 'Full' column header on the per-period sub-table.
    assert '| Full |' in md


def _synthetic_records(n=250, mean=0.001, std=0.01, seed=42):
    """Build n synthetic DailyRecord-like objects with N(mean, std) returns."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mean, scale=std, size=n)
    out = []
    pv = 100000.0
    base = datetime(2024, 1, 1)
    for i in range(n):
        pv *= (1 + returns[i])
        out.append(type('R', (), {
            'date': base + timedelta(days=i),
            'regime': 'STRONG_BULL',
            'portfolio_value': pv,
            'daily_return': float(returns[i]),
            'turnover_usd': 5000.0,
            'cost_usd': 2.5,
            'target_weights': {},
            'realized_weights': {},
        })())
    return out


def test_psr_gate_renders_with_known_returns():
    """The PSR gate renders with daily-units sample stats and a PASS/FAIL verdict."""
    records = _synthetic_records(n=250, mean=0.001, std=0.01)
    table = _format_psr_gate(records)
    assert isinstance(table, str)
    assert 'PSR' in table
    assert 'daily Sharpe (formula input)' in table
    assert 'annualized Sharpe' in table
    # PSR row should contain PASS or FAIL
    psr_lines = [ln for ln in table.splitlines() if 'PSR' in ln]
    assert any(('PASS' in ln) or ('FAIL' in ln) for ln in psr_lines)


def test_psr_gate_insufficient_data():
    """Fewer than 30 observations -> friendly message, not crash."""
    records = _synthetic_records(n=10)
    msg = _format_psr_gate(records)
    assert msg == '_Insufficient data for statistical gate._'


def test_build_variant_report_includes_psr_section_not_dsr():
    """build_variant_report shows PSR per-variant; DSR is cross-variant only."""
    records = _synthetic_records(n=250)
    md = build_variant_report(
        variant_id='V01',
        variant_description='Test',
        records_by_cost_bps={5.0: records},
        git_commit='abc123',
        universe_csv='config/universes/sp500-2025.csv',
        timing_mode='near_close',
        n_trials=20,
    )
    assert '## PSR gate' in md
    assert 'PSR' in md
    # DSR must be flagged as cross-variant only, not computed inline.
    assert 'cross-variant' in md.lower()


def test_build_parity_report_produces_side_by_side_table():
    v01_records = _fake_records(daily_return=0.001)
    v03_records = _fake_records(daily_return=0.0008)  # slightly worse
    md = build_parity_report(v01_records=v01_records, v03_records=v03_records, cost_bps=5.0)
    assert '# Phase 4 V01 vs V03' in md
    assert '| V01 |' in md
    assert '| V03 |' in md
    assert 'EXT-OOS Sharpe' in md or 'Sharpe' in md
