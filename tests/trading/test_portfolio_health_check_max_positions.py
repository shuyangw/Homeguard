"""Tests for HealthCheckResult.max_positions_exceeded.

The RAMP rebalance needs to distinguish "over the position cap" (which should
only block NEW entries, letting the rebalance SELL down toward top_n) from hard
failures (insufficient cash, etc., which must abort the whole rebalance). The
structured `max_positions_exceeded` flag lets the adapter make that distinction
without parsing error strings.
"""

from unittest.mock import MagicMock

from src.trading.utils.portfolio_health_check import PortfolioHealthChecker


def _broker_with(n_positions: int):
    broker = MagicMock()
    broker.get_account.return_value = {
        "buying_power": 1_000_000.0,
        "portfolio_value": 1_000_000.0,
        "cash": 1_000_000.0,
    }
    broker.get_positions.return_value = [
        {
            "symbol": f"SYM{i}",
            "quantity": 10,
            "current_price": 100.0,
            "avg_entry_price": 100.0,
        }
        for i in range(n_positions)
    ]
    broker.get_open_orders.return_value = []
    return broker


def _state_manager_with(n_positions: int):
    sm = MagicMock()
    sm.get_positions.return_value = {f"SYM{i}": {"qty": 10} for i in range(n_positions)}
    return sm


def test_max_positions_exceeded_flag_true_when_over_cap():
    checker = PortfolioHealthChecker(
        broker=_broker_with(33),
        max_positions=25,
        state_manager=_state_manager_with(33),
    )
    result = checker.check_before_entry(
        allow_existing_positions=True, strategy_name="ramp"
    )
    assert result.max_positions_exceeded is True
    assert any("Max positions reached" in e for e in result.errors)
    # Over-cap is the ONLY error (cash/portfolio are healthy).
    assert all(e.startswith("Max positions reached") for e in result.errors)


def test_max_positions_exceeded_flag_false_when_under_cap():
    checker = PortfolioHealthChecker(
        broker=_broker_with(10),
        max_positions=25,
        state_manager=_state_manager_with(10),
    )
    result = checker.check_before_entry(
        allow_existing_positions=True, strategy_name="ramp"
    )
    assert result.max_positions_exceeded is False
    assert result.passed is True
