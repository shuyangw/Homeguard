"""
Regression test for the 2026-04-24 "$1M on the Grafana board" mismatch.

The `hg_strategy_equity_usd{strategy="ramp"}` gauge was emitted as
`broker.get_account()['portfolio_value']` -- the whole IBKR paper
account (~$1.01M) -- even though RAMP's intended allocation is $100K.
The dashboard therefore showed $1M for RAMP regardless of position
sizing.

Fix: `LiveTradingRunner._compute_strategy_equity` computes
`initial_capital + sum(unrealized_pnl for strategy-tagged positions)`
and emits that as equity.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_runner_with_mocks(
    broker_positions=None,
    tagged_symbols=None,
    adapter_broker_raises=False,
):
    """Build just enough of LiveTradingRunner to exercise _compute_strategy_equity."""
    from scripts.trading.run_live_paper_trading import LiveTradingRunner

    runner = LiveTradingRunner.__new__(LiveTradingRunner)
    broker = MagicMock()
    if adapter_broker_raises:
        broker.get_positions.side_effect = RuntimeError("broker flaky")
    else:
        broker.get_positions.return_value = broker_positions or []

    state_manager = MagicMock()
    if tagged_symbols is None:
        state_manager.get_positions.return_value = {}
    else:
        state_manager.get_positions.return_value = {s: {} for s in tagged_symbols}

    adapter = SimpleNamespace(
        broker=broker,
        state_manager=state_manager,
    )

    registry = SimpleNamespace(strategy='ramp')
    runner.adapter = adapter
    runner.metrics_registry = registry
    return runner


class TestComputeStrategyEquity:
    def test_no_positions_returns_initial_capital(self):
        """With zero RAMP-tagged positions, equity == initial_capital.

        This is the immediate fix for the '$1M on dashboard' issue.
        """
        runner = _make_runner_with_mocks(broker_positions=[], tagged_symbols=set())
        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0,
            account={'portfolio_value': 1_014_176.25},
        )
        assert equity == 100_000.0

    def test_positions_with_positive_pnl_adds_to_equity(self):
        runner = _make_runner_with_mocks(
            broker_positions=[
                {'symbol': 'AAPL', 'unrealized_pnl': 500.0},
                {'symbol': 'MSFT', 'unrealized_pnl': 250.0},
                {'symbol': 'NVDA', 'unrealized_pnl': -100.0},
            ],
            tagged_symbols={'AAPL', 'MSFT', 'NVDA'},
        )
        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0,
            account={'portfolio_value': 1_014_176.25},  # account value ignored
        )
        assert equity == 100_000.0 + 500.0 + 250.0 - 100.0

    def test_ignores_other_strategy_positions(self):
        """Positions not owned by this strategy must be excluded from P&L."""
        runner = _make_runner_with_mocks(
            broker_positions=[
                {'symbol': 'AAPL', 'unrealized_pnl': 500.0},  # RAMP-tagged
                {'symbol': 'TSLA', 'unrealized_pnl': 10_000.0},  # OMR's, not RAMP
            ],
            tagged_symbols={'AAPL'},  # only AAPL is RAMP's
        )
        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0,
            account={'portfolio_value': 1_014_176.25},
        )
        assert equity == 100_000.0 + 500.0
        # Critically, TSLA's 10k gain must NOT inflate RAMP's equity
        assert equity != 100_000.0 + 500.0 + 10_000.0

    def test_broker_failure_falls_back_to_initial_capital(self):
        """If broker.get_positions raises, equity should degrade to the static
        initial_capital rather than the stale broker portfolio value."""
        runner = _make_runner_with_mocks(adapter_broker_raises=True)
        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0,
            account={'portfolio_value': 9_999_999.99},
        )
        assert equity == 100_000.0

    def test_state_manager_missing_falls_through_to_initial_capital(self):
        """If state_manager is not attached, return initial_capital (not the full
        broker account value)."""
        from scripts.trading.run_live_paper_trading import LiveTradingRunner
        runner = LiveTradingRunner.__new__(LiveTradingRunner)
        broker = MagicMock()
        broker.get_positions.return_value = [
            {'symbol': 'AAPL', 'unrealized_pnl': 500.0}
        ]
        runner.adapter = SimpleNamespace(broker=broker, state_manager=None)
        runner.metrics_registry = SimpleNamespace(strategy='ramp')

        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0, account={'portfolio_value': 1_000_000.0}
        )
        assert equity == 100_000.0

    def test_none_pnl_defaults_to_zero(self):
        runner = _make_runner_with_mocks(
            broker_positions=[{'symbol': 'AAPL', 'unrealized_pnl': None}],
            tagged_symbols={'AAPL'},
        )
        equity = runner._compute_strategy_equity(
            initial_capital=100_000.0,
            account={'portfolio_value': 1_014_176.25},
        )
        assert equity == 100_000.0
