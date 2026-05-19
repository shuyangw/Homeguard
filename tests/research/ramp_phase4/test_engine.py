"""Tests for engine: dataclasses + run_variant control flow."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytest

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import HarnessState, DailyRecord, run_variant


def _tiny_cfg(tmp_path):
    csv = tmp_path / 'u.csv'
    csv.write_text('symbol\nAAA\n')
    return HarnessConfig(
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 5),
        universe_csv=csv,
        initial_capital=100000.0,
        cost_bps_per_side=0.0,
    )


def _tiny_panel():
    idx = pd.date_range('2024-01-02', periods=4, freq='B')
    return pd.DataFrame({
        'AAA': [100.0, 101.0, 102.0, 103.0],
        'SPY': [400.0, 401.0, 402.0, 403.0],
        'VIX': [15.0, 15.1, 15.2, 15.3],
    }, index=idx)


def test_harness_state_initializes_empty():
    s = HarnessState(cash_usd=100000.0)
    assert s.cash_usd == 100000.0
    assert s.positions == {}
    assert s.realized_pnl_usd == 0.0


def test_daily_record_required_fields():
    r = DailyRecord(
        date=datetime(2024, 1, 2), regime='STRONG_BULL',
        target_weights={}, realized_weights={},
        turnover_usd=0.0, cost_usd=0.0,
        portfolio_value=100000.0, daily_return=0.0,
    )
    assert r.regime == 'STRONG_BULL'


def test_run_variant_empty_variant_returns_per_day_records(tmp_path, monkeypatch):
    """Stub variant returning empty dict on every day -- engine produces one record per trading day."""
    cfg = _tiny_cfg(tmp_path)
    panel = _tiny_panel()
    monkeypatch.setattr(
        'src.research.ramp_phase4.engine.load_universe_panel',
        lambda c, s, e: panel,
    )
    variant_spec = type('Spec', (), {
        'id': 'STUB',
        'plan_fn': staticmethod(lambda t, st, pn, cf: {}),
    })()
    records = run_variant(cfg, variant_spec)
    assert len(records) == 4
    assert all(r.portfolio_value == 100000.0 for r in records)
    assert all(r.daily_return == 0.0 for r in records)
