"""Tests for flat_bps_cost."""
from src.research.ramp_phase4.costs import flat_bps_cost


def test_flat_bps_cost_zero_when_no_trades():
    assert flat_bps_cost([], bps=5.0) == 0.0


def test_flat_bps_cost_proportional_to_traded_notional():
    trades = [{'symbol': 'AAPL', 'trade_value_usd': 10000.0}]
    assert flat_bps_cost(trades, bps=5.0) == 5.0


def test_flat_bps_cost_handles_buys_and_sells_symmetrically():
    trades = [
        {'symbol': 'AAPL', 'trade_value_usd': 10000.0},
        {'symbol': 'MSFT', 'trade_value_usd': -10000.0},
    ]
    assert flat_bps_cost(trades, bps=5.0) == 10.0
