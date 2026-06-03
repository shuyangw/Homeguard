"""Tests for DailyRecord.trades chronicling (Part A deliverables).

Three test groups:
1. DailyRecord.trades is populated on a trading day, empty on SAFE_MODE / no-trade.
2. sum(abs(trade_value_usd)) over a day's trades == that day's turnover_usd.
3. Persisted holdings/ledger CSVs have the expected schema and row counts.
"""
from __future__ import annotations

import gzip
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import DailyRecord, HarnessState, run_variant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, symbols: List[str], n_days: int, timing_mode: str = 'near_close') -> HarnessConfig:
    csv = tmp_path / 'u.csv'
    csv.write_text('symbol\n' + '\n'.join(symbols) + '\n')
    idx = pd.date_range('2024-01-02', periods=n_days, freq='B')
    cfg = HarnessConfig(
        start_date=idx[0].to_pydatetime(),
        end_date=idx[-1].to_pydatetime(),
        universe_csv=csv,
        initial_capital=100_000.0,
        cost_bps_per_side=0.0,
        timing_mode=timing_mode,
    )
    return cfg


def _panel(symbols: List[str], n_days: int) -> pd.DataFrame:
    idx = pd.date_range('2024-01-02', periods=n_days, freq='B')
    data = {sym: [100.0 + i * 0.5 for i in range(n_days)] for sym in symbols}
    data['SPY'] = [400.0 + i for i in range(n_days)]
    data['VIX'] = [15.0 + 0.1 * i for i in range(n_days)]
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Group 1: trades field populated correctly
# ---------------------------------------------------------------------------


def test_daily_record_trades_populated_on_trading_day(tmp_path, monkeypatch):
    """On a rebalance day (near_close, non-SAFE_MODE), DailyRecord.trades is non-empty."""
    syms = ['AAA']
    n = 3
    cfg = _cfg(tmp_path, syms, n)
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    call = {'n': 0}

    def plan_fn(t, st, pn, cf):
        call['n'] += 1
        return {'AAA': 0.10}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    # Day 0 opens a position -> trades list must be non-empty.
    assert len(records[0].trades) > 0, 'Expected trades on first rebalance day'
    first_trade = records[0].trades[0]
    assert first_trade['symbol'] == 'AAA'
    assert first_trade['side'] in ('buy', 'sell')
    assert 'delta_shares' in first_trade
    assert 'trade_value_usd' in first_trade


def test_daily_record_trades_empty_on_safe_mode_day(tmp_path, monkeypatch):
    """SAFE_MODE days must have trades=[] (no execution)."""
    syms = ['AAA']
    n = 4
    cfg = _cfg(tmp_path, syms, n)
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    call = {'n': 0}

    def plan_fn(t, st, pn, cf):
        call['n'] += 1
        if call['n'] == 1:
            return {'AAA': 0.10}
        return {'__regime__': 'SAFE_MODE'}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    for i in range(1, n):
        assert records[i].regime == 'SAFE_MODE'
        assert records[i].trades == [], f'Expected empty trades on SAFE_MODE day {i}'


def test_daily_record_trades_empty_on_no_rebalance_day(tmp_path, monkeypatch):
    """When target weights exactly match holdings (no delta), trades=[]."""
    syms = ['AAA']
    n = 2
    cfg = _cfg(tmp_path, syms, n)
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    call = {'n': 0}

    def plan_fn(t, st, pn, cf):
        call['n'] += 1
        if call['n'] == 1:
            return {'AAA': 0.10}
        # Return exactly the current realized weight to suppress rebalance.
        cur_val = st.cash_usd
        for sym, sh in st.positions.items():
            px = pn.loc[t, sym]
            cur_val += sh * px
        if 'AAA' in st.positions and cur_val > 0:
            exact_w = (st.positions['AAA'] * pn.loc[t, 'AAA']) / cur_val
            return {'AAA': exact_w}
        return {}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    # Day 0 buys AAA. Day 1 target matches current weight (whole-share: no delta) -> empty.
    assert len(records[0].trades) > 0
    assert records[1].trades == []


def test_daily_record_trades_populated_one_day_lag(tmp_path, monkeypatch):
    """one_day_lag: plan on T executes on T+1 -> DailyRecord.trades on day 1 is non-empty."""
    syms = ['AAA']
    n = 3
    cfg = _cfg(tmp_path, syms, n, timing_mode='one_day_lag')
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    def plan_fn(t, st, pn, cf):
        return {'AAA': 0.10}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    # Day 0 (one_day_lag): no pending -> trades=[].
    assert records[0].trades == []
    # Day 1: yesterday's plan executes -> non-empty.
    assert len(records[1].trades) > 0


# ---------------------------------------------------------------------------
# Group 2: turnover consistency (sum(abs(trade_value_usd)) == turnover_usd)
# ---------------------------------------------------------------------------


def test_trades_sum_equals_turnover_usd_every_day(tmp_path, monkeypatch):
    """For every record, sum(|trade_value_usd|) must equal turnover_usd."""
    syms = ['AAA', 'BBB']
    n = 5
    cfg = _cfg(tmp_path, syms, n)
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    call = {'n': 0}

    def plan_fn(t, st, pn, cf):
        call['n'] += 1
        if call['n'] % 2 == 1:
            return {'AAA': 0.10, 'BBB': 0.10}
        # Rotate.
        return {'AAA': 0.05, 'BBB': 0.15}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    for i, rec in enumerate(records):
        computed = sum(abs(tr['trade_value_usd']) for tr in rec.trades)
        assert abs(computed - rec.turnover_usd) < 1e-6, (
            f'Day {i}: trades sum {computed:.6f} != turnover_usd {rec.turnover_usd:.6f}'
        )


def test_trades_sum_equals_turnover_one_day_lag(tmp_path, monkeypatch):
    """Same consistency check for one_day_lag timing mode."""
    syms = ['AAA']
    n = 4
    cfg = _cfg(tmp_path, syms, n, timing_mode='one_day_lag')
    panel = _panel(syms, n)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    def plan_fn(t, st, pn, cf):
        return {'AAA': 0.10}

    spec = type('S', (), {'id': 'T', 'plan_fn': staticmethod(plan_fn)})()
    records = run_variant(cfg, spec)

    for i, rec in enumerate(records):
        computed = sum(abs(tr['trade_value_usd']) for tr in rec.trades)
        assert abs(computed - rec.turnover_usd) < 1e-6, (
            f'Day {i}: trades sum {computed:.6f} != turnover_usd {rec.turnover_usd:.6f}'
        )


# ---------------------------------------------------------------------------
# Group 3: persisted CSV schema and row counts
# ---------------------------------------------------------------------------


def _make_records_with_known_trades() -> List[DailyRecord]:
    """Build two synthetic DailyRecords with known trade/holdings content."""
    r0 = DailyRecord(
        date=datetime(2024, 1, 2), regime='STRONG_BULL',
        target_weights={'AAA': 0.10},
        realized_weights={'AAA': 0.099},
        turnover_usd=9900.0, cost_usd=0.0,
        portfolio_value=100000.0, daily_return=0.0,
        trades=[
            {'symbol': 'AAA', 'delta_shares': 99, 'trade_value_usd': 9900.0, 'side': 'buy'},
        ],
    )
    r1 = DailyRecord(
        date=datetime(2024, 1, 3), regime='STRONG_BULL',
        target_weights={'AAA': 0.10, 'BBB': 0.05},
        realized_weights={'AAA': 0.099, 'BBB': 0.049},
        turnover_usd=4850.0, cost_usd=0.0,
        portfolio_value=100000.0, daily_return=0.0,
        trades=[
            {'symbol': 'BBB', 'delta_shares': 97, 'trade_value_usd': 4850.0, 'side': 'buy'},
        ],
    )
    return [r0, r1]


def _call_write_chronicles(records, tmp_path):
    """Import and call _write_chronicles with tmp_path as output_dir."""
    import sys
    import importlib.util
    spec_path = (
        Path(__file__).parents[3]
        / 'scripts' / 'backtest_scripts' / 'ramp_phase4_wave3_readiness.py'
    )
    loader_spec = importlib.util.spec_from_file_location('ramp_phase4_wave3_readiness', spec_path)
    mod = importlib.util.module_from_spec(loader_spec)
    # Register in sys.modules BEFORE exec so dataclass __module__ lookups succeed.
    sys.modules['ramp_phase4_wave3_readiness'] = mod
    try:
        loader_spec.loader.exec_module(mod)
        mod._write_chronicles(
            records,
            output_dir=tmp_path,
            variant_id='VTEST',
            timing_mode='near_close',
            cost_bps=5.0,
        )
    finally:
        sys.modules.pop('ramp_phase4_wave3_readiness', None)
    return tmp_path / 'holdings'


def test_holdings_csv_schema(tmp_path):
    """Holdings CSV must have columns: date, symbol, realized_weight."""
    records = _make_records_with_known_trades()
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_holdings.csv.gz'
    assert gz_path.exists(), f'Expected holdings file at {gz_path}'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert set(df.columns) == {'date', 'symbol', 'realized_weight'}


def test_trades_csv_schema(tmp_path):
    """Trade-ledger CSV must have columns: date, symbol, side, delta_shares, trade_value_usd."""
    records = _make_records_with_known_trades()
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_trades.csv.gz'
    assert gz_path.exists(), f'Expected trades file at {gz_path}'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert set(df.columns) == {'date', 'symbol', 'side', 'delta_shares', 'trade_value_usd'}


def test_holdings_row_count_matches_symbol_days(tmp_path):
    """Row count = sum of realized_weights sizes across all records."""
    records = _make_records_with_known_trades()
    expected_rows = sum(len(r.realized_weights) for r in records)  # 1 + 2 = 3
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_holdings.csv.gz'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert len(df) == expected_rows, f'Expected {expected_rows} holding rows, got {len(df)}'


def test_trades_row_count_matches_total_trades(tmp_path):
    """Row count = sum of len(r.trades) across all records."""
    records = _make_records_with_known_trades()
    expected_rows = sum(len(r.trades) for r in records)  # 1 + 1 = 2
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_trades.csv.gz'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert len(df) == expected_rows, f'Expected {expected_rows} trade rows, got {len(df)}'


def test_holdings_csv_empty_on_no_positions(tmp_path):
    """If every record has no realized_weights, holdings CSV has 0 data rows."""
    records = [
        DailyRecord(
            date=datetime(2024, 1, 2), regime='BEAR',
            target_weights={}, realized_weights={},
            turnover_usd=0.0, cost_usd=0.0,
            portfolio_value=100000.0, daily_return=0.0,
            trades=[],
        ),
    ]
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_holdings.csv.gz'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert len(df) == 0
    assert set(df.columns) == {'date', 'symbol', 'realized_weight'}


def test_trades_csv_empty_on_safe_mode_records(tmp_path):
    """If all records have trades=[], trade-ledger CSV has 0 data rows."""
    records = [
        DailyRecord(
            date=datetime(2024, 1, 2), regime='SAFE_MODE',
            target_weights={}, realized_weights={'AAA': 0.10},
            turnover_usd=0.0, cost_usd=0.0,
            portfolio_value=100000.0, daily_return=0.0,
            trades=[],
        ),
    ]
    holdings_dir = _call_write_chronicles(records, tmp_path)
    gz_path = holdings_dir / 'VTEST_near_close_5.0bps_trades.csv.gz'
    with gzip.open(gz_path, 'rt', encoding='utf-8') as fh:
        df = pd.read_csv(fh)
    assert len(df) == 0
    assert set(df.columns) == {'date', 'symbol', 'side', 'delta_shares', 'trade_value_usd'}
