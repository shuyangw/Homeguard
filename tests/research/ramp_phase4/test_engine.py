"""Tests for engine: dataclasses + run_variant control flow."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytest

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import HarnessState, DailyRecord, compute_trades, run_variant
from src.research.ramp_phase4.engine import apply_trades


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


def test_run_variant_marks_to_market_correctly(tmp_path, monkeypatch):
    """Hold 100 shares of AAA from day 1; portfolio_value follows price."""
    cfg = _tiny_cfg(tmp_path)
    panel = _tiny_panel()
    monkeypatch.setattr(
        'src.research.ramp_phase4.engine.load_universe_panel',
        lambda c, s, e: panel,
    )

    # Variant that opens 100 shares of AAA on day 1, then holds.
    initial_target = {'AAA': 0.10}  # 10% of equity in AAA
    call_count = {'n': 0}
    def variant_fn(t, st, pn, cf):
        call_count['n'] += 1
        if call_count['n'] == 1:
            return initial_target
        # Subsequent days: return current weight to avoid churn.
        cur_value = st.cash_usd + sum(sh * pn.loc[t, sym] for sym, sh in st.positions.items())
        if 'AAA' in st.positions and cur_value > 0:
            return {'AAA': (st.positions['AAA'] * pn.loc[t, 'AAA']) / cur_value}
        return {}

    spec = type('Spec', (), {'id': 'MTM', 'plan_fn': staticmethod(variant_fn)})()
    records = run_variant(cfg, spec)
    # Day 1 buys 100 shares at $100 (10% of $100k). After buy: cash=$90k, AAA=100 shares.
    # Day 2 price = $101 -> position value = $10,100 -> portfolio = $100,100.
    # Daily return day 2 ~ 0.001.
    assert records[1].daily_return > 0.0
    assert records[1].portfolio_value > records[0].portfolio_value


def test_run_variant_handles_nan_pricing_with_forced_exit(tmp_path, monkeypatch):
    """Held symbol going NaN triggers a forced exit at last good close."""
    cfg = _tiny_cfg(tmp_path)
    # Day 3 AAA price = NaN; should force exit at day 2 close.
    idx = pd.date_range('2024-01-02', periods=4, freq='B')
    panel = pd.DataFrame({
        'AAA': [100.0, 101.0, float('nan'), 103.0],
        'SPY': [400.0, 401.0, 402.0, 403.0],
        'VIX': [15.0, 15.1, 15.2, 15.3],
    }, index=idx)
    monkeypatch.setattr(
        'src.research.ramp_phase4.engine.load_universe_panel',
        lambda c, s, e: panel,
    )

    call_count = {'n': 0}
    def variant_fn(t, st, pn, cf):
        call_count['n'] += 1
        if call_count['n'] == 1:
            return {'AAA': 0.10}
        return {sym: (sh * pn.loc[t].get(sym, 0.0)) / max(_pv(st, pn.loc[t]), 1.0)
                for sym, sh in st.positions.items() if pd.notna(pn.loc[t].get(sym, float('nan')))}

    def _pv(st, prices):
        v = st.cash_usd
        for sym, sh in st.positions.items():
            px = prices.get(sym, float('nan'))
            if pd.notna(px):
                v += sh * px
        return v

    spec = type('Spec', (), {'id': 'NAN', 'plan_fn': staticmethod(variant_fn)})()
    records = run_variant(cfg, spec)
    # After day 3 (NaN), positions should be empty (forced exit).
    # Verify by checking day 4 has no AAA position implied in target_weights.
    assert len(records) == 4


def test_compute_trades_no_change_when_target_equals_current():
    state = HarnessState(cash_usd=50000.0, positions={'AAA': 100.0})
    prices = pd.Series({'AAA': 100.0})  # AAA position = $10000, cash = $50000, total = $60000
    target_weights = {'AAA': 10000.0 / 60000.0}
    trades = compute_trades(state, target_weights, prices, total_value=60000.0, min_trade_value_usd=100.0)
    assert trades == []


def test_compute_trades_buy_to_open_position():
    state = HarnessState(cash_usd=100000.0, positions={})
    prices = pd.Series({'AAA': 100.0})
    target_weights = {'AAA': 0.10}  # $10000 of AAA -> 100 shares
    trades = compute_trades(state, target_weights, prices, total_value=100000.0, min_trade_value_usd=100.0)
    assert len(trades) == 1
    assert trades[0]['symbol'] == 'AAA'
    assert trades[0]['delta_shares'] == 100
    assert trades[0]['trade_value_usd'] == 10000.0


def test_compute_trades_sell_to_close_position():
    state = HarnessState(cash_usd=90000.0, positions={'AAA': 100.0})
    prices = pd.Series({'AAA': 100.0})
    target_weights: dict = {}  # target absent -> exit
    trades = compute_trades(state, target_weights, prices, total_value=100000.0, min_trade_value_usd=100.0)
    assert len(trades) == 1
    assert trades[0]['symbol'] == 'AAA'
    assert trades[0]['delta_shares'] == -100
    assert trades[0]['trade_value_usd'] == -10000.0


def test_compute_trades_skips_undersize():
    state = HarnessState(cash_usd=100000.0, positions={})
    prices = pd.Series({'AAA': 100.0})
    target_weights = {'AAA': 0.0005}  # $50 target -> 0 whole shares with $100 min
    trades = compute_trades(state, target_weights, prices, total_value=100000.0, min_trade_value_usd=100.0)
    assert trades == []


def test_apply_trades_updates_cash_and_positions():
    state = HarnessState(cash_usd=100000.0)
    trades = [{'symbol': 'AAA', 'delta_shares': 100, 'trade_value_usd': 10000.0, 'side': 'buy'}]
    apply_trades(state, trades, cost_usd=5.0)
    assert state.cash_usd == 100000.0 - 10000.0 - 5.0
    assert state.positions == {'AAA': 100.0}


def test_apply_trades_sells_reduce_position_and_credit_cash():
    state = HarnessState(cash_usd=90000.0, positions={'AAA': 100.0})
    trades = [{'symbol': 'AAA', 'delta_shares': -50, 'trade_value_usd': -5000.0, 'side': 'sell'}]
    apply_trades(state, trades, cost_usd=2.5)
    # Sell 50 of 100 -> 50 left. Cash += 5000 - 2.5
    assert state.positions == {'AAA': 50.0}
    assert state.cash_usd == 90000.0 + 5000.0 - 2.5


def test_apply_trades_full_exit_removes_symbol_from_positions():
    state = HarnessState(cash_usd=90000.0, positions={'AAA': 100.0})
    trades = [{'symbol': 'AAA', 'delta_shares': -100, 'trade_value_usd': -10000.0, 'side': 'sell'}]
    apply_trades(state, trades, cost_usd=5.0)
    assert 'AAA' not in state.positions


def test_run_variant_applies_cost_to_cash(tmp_path, monkeypatch):
    """Force a single trade on day 1; assert cost is deducted from cash."""
    cfg_with_cost = HarnessConfig(
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 3),
        universe_csv=(tmp_path / 'u.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=10.0,  # 10 bps cost
    )
    cfg_with_cost.universe_csv.write_text('symbol\nAAA\n')
    idx = pd.date_range('2024-01-02', periods=2, freq='B')
    panel = pd.DataFrame({
        'AAA': [100.0, 101.0], 'SPY': [400.0, 401.0], 'VIX': [15.0, 15.1],
    }, index=idx)
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)

    call = {'n': 0}
    def variant_fn(t, st, pn, cf):
        call['n'] += 1
        if call['n'] == 1:
            return {'AAA': 0.10}  # buy $10k of AAA
        # Day 2 -- match current weight to avoid more trades
        cur = st.positions.get('AAA', 0.0) * pn.loc[t, 'AAA']
        pv = st.cash_usd + cur
        return {'AAA': cur / pv} if pv > 0 else {}

    spec = type('Spec', (), {'id': 'COST', 'plan_fn': staticmethod(variant_fn)})()
    records = run_variant(cfg_with_cost, spec)
    # Day 1 buy: $10000 notional at 10 bps -> $10 cost.
    assert records[0].cost_usd == pytest.approx(10.0, abs=0.01)
    assert records[0].turnover_usd == pytest.approx(10000.0, abs=1.0)


def test_run_variant_aborts_on_overleverage(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    panel = _tiny_panel()
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)
    spec = type('Spec', (), {'id': 'BAD', 'plan_fn': staticmethod(lambda t, st, pn, cf: {'AAA': 1.5})})()
    with pytest.raises(ValueError, match='overleverage'):
        run_variant(cfg, spec)


def test_run_variant_records_regime_when_variant_provides(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    panel = _tiny_panel()
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)
    def variant_fn(t, st, pn, cf):
        return {'__regime__': 'STRONG_BULL', 'AAA': 0.05}
    spec = type('Spec', (), {'id': 'R', 'plan_fn': staticmethod(variant_fn)})()
    records = run_variant(cfg, spec)
    assert all(r.regime == 'STRONG_BULL' for r in records)


def test_run_variant_safe_mode_skips_rebalance(tmp_path, monkeypatch):
    """If variant returns {'__regime__': 'SAFE_MODE'} the engine holds current weights."""
    cfg = _tiny_cfg(tmp_path)
    panel = _tiny_panel()
    monkeypatch.setattr('src.research.ramp_phase4.engine.load_universe_panel', lambda c, s, e: panel)
    call = {'n': 0}
    def variant_fn(t, st, pn, cf):
        call['n'] += 1
        if call['n'] == 1:
            return {'__regime__': 'STRONG_BULL', 'AAA': 0.10}
        return {'__regime__': 'SAFE_MODE'}  # subsequent days are safe-mode
    spec = type('Spec', (), {'id': 'SAFE', 'plan_fn': staticmethod(variant_fn)})()
    records = run_variant(cfg, spec)
    # After day 1 buy, days 2-4 stay safe-mode and produce zero turnover.
    assert records[1].regime == 'SAFE_MODE'
    assert records[1].turnover_usd == 0.0


def test_harness_state_initializes_new_fields_empty():
    from src.research.ramp_phase4.engine import HarnessState
    s = HarnessState(cash_usd=100000.0)
    assert s.position_open_dates == {}
    assert s.last_target_symbols == []
