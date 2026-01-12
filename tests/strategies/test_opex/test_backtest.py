"""Tests for OpEx Backtest Runner."""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.strategies.opex.backtest import (
    OpExBacktestRunner,
    OpExBacktestResults,
    OpExTradeRecord,
)
from src.strategies.opex.calendar import OpExPhase
from src.strategies.opex.signal_generator import SignalDirection
from src.data.options.options_schema import DailyGEXSummary


class TestOpExTradeRecord:
    """Tests for OpExTradeRecord dataclass."""

    def test_create_trade_record(self):
        """Test creating a trade record."""
        trade = OpExTradeRecord(
            symbol="SPY",
            entry_date=date(2024, 1, 17),
            exit_date=date(2024, 1, 18),
            entry_price=475.0,
            exit_price=478.0,
            direction=SignalDirection.LONG,
            phase=OpExPhase.PINNING,
            pin_strike=475.0,
            distance_to_pin=0.0,
            net_gex=5_000_000,
            pnl=300.0,
            pnl_pct=0.0063,
            shares=100,
        )

        assert trade.symbol == "SPY"
        assert trade.pnl == 300.0
        assert trade.phase == OpExPhase.PINNING


class TestOpExBacktestResults:
    """Tests for OpExBacktestResults dataclass."""

    def test_empty_results(self):
        """Test empty results initialization."""
        results = OpExBacktestResults()

        assert results.total_trades == 0
        assert results.win_rate == 0.0
        assert results.trades == []
        assert results.equity_curve is None

    def test_results_with_trades(self):
        """Test results with trade data."""
        trades = [
            OpExTradeRecord(
                symbol="SPY",
                entry_date=date(2024, 1, 17),
                exit_date=date(2024, 1, 18),
                entry_price=475.0,
                exit_price=478.0,
                direction=SignalDirection.LONG,
                phase=OpExPhase.PINNING,
                pin_strike=475.0,
                distance_to_pin=0.01,
                net_gex=5_000_000,
                pnl=300.0,
                pnl_pct=0.0063,
                shares=100,
            ),
        ]

        results = OpExBacktestResults(
            trades=trades,
            total_trades=1,
            winning_trades=1,
            win_rate=1.0,
            total_pnl=300.0,
        )

        assert results.total_trades == 1
        assert results.win_rate == 1.0
        assert len(results.trades) == 1


class TestOpExBacktestRunner:
    """Tests for OpExBacktestRunner."""

    def test_initialization_defaults(self):
        """Test runner initializes with defaults."""
        runner = OpExBacktestRunner()

        assert runner.initial_capital == 100_000
        assert runner.symbols == ['SPY', 'QQQ', 'IWM']
        assert runner.position_size_pct == 0.10
        assert runner.strategy is not None

    def test_initialization_custom(self):
        """Test runner with custom parameters."""
        runner = OpExBacktestRunner(
            initial_capital=50_000,
            symbols=['SPY'],
            position_size_pct=0.20,
        )

        assert runner.initial_capital == 50_000
        assert runner.symbols == ['SPY']
        assert runner.position_size_pct == 0.20

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with empty trades."""
        runner = OpExBacktestRunner()
        results = OpExBacktestResults()

        runner._calculate_metrics(results)

        assert results.total_trades == 0
        assert results.win_rate == 0.0
        assert results.sharpe_ratio == 0.0

    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trades."""
        runner = OpExBacktestRunner()

        trades = [
            OpExTradeRecord(
                symbol="SPY", entry_date=date(2024, 1, 17), exit_date=date(2024, 1, 18),
                entry_price=475.0, exit_price=480.0, direction=SignalDirection.LONG,
                phase=OpExPhase.PINNING, pin_strike=475.0, distance_to_pin=0.01,
                net_gex=5_000_000, pnl=500.0, pnl_pct=0.0105, shares=100,
            ),
            OpExTradeRecord(
                symbol="SPY", entry_date=date(2024, 1, 22), exit_date=date(2024, 1, 23),
                entry_price=480.0, exit_price=478.0, direction=SignalDirection.LONG,
                phase=OpExPhase.POST_OPEX, pin_strike=480.0, distance_to_pin=-0.01,
                net_gex=3_000_000, pnl=-200.0, pnl_pct=-0.0042, shares=100,
            ),
        ]

        results = OpExBacktestResults(trades=trades)
        runner._calculate_metrics(results)

        assert results.total_trades == 2
        assert results.winning_trades == 1
        assert results.losing_trades == 1
        assert results.win_rate == 0.5
        assert results.total_pnl == 300.0  # 500 - 200
        assert results.pinning_trades == 1
        assert results.post_opex_trades == 1

    def test_calculate_metrics_profit_factor(self):
        """Test profit factor calculation."""
        runner = OpExBacktestRunner()

        trades = [
            OpExTradeRecord(
                symbol="SPY", entry_date=date(2024, 1, 17), exit_date=date(2024, 1, 18),
                entry_price=475.0, exit_price=480.0, direction=SignalDirection.LONG,
                phase=OpExPhase.PINNING, pin_strike=475.0, distance_to_pin=0.01,
                net_gex=5_000_000, pnl=1000.0, pnl_pct=0.02, shares=100,
            ),
            OpExTradeRecord(
                symbol="SPY", entry_date=date(2024, 1, 22), exit_date=date(2024, 1, 23),
                entry_price=480.0, exit_price=478.0, direction=SignalDirection.LONG,
                phase=OpExPhase.PINNING, pin_strike=480.0, distance_to_pin=-0.01,
                net_gex=3_000_000, pnl=-500.0, pnl_pct=-0.01, shares=100,
            ),
        ]

        results = OpExBacktestResults(trades=trades)
        runner._calculate_metrics(results)

        # Profit factor = gross_profit / gross_loss = 1000 / 500 = 2.0
        assert results.profit_factor == 2.0


class TestOpExBacktestRunnerSimulation:
    """Tests for trading simulation."""

    def _create_price_data(self):
        """Create mock price data for testing."""
        dates = pd.date_range('2024-01-15', '2024-01-25', freq='D')
        data = {
            'open': [470, 472, 475, 478, 480, 479, 481, 482, 480, 478, 475],
            'high': [473, 476, 479, 481, 483, 482, 484, 485, 483, 481, 478],
            'low': [468, 470, 473, 476, 478, 477, 479, 480, 478, 476, 473],
            'close': [472, 475, 478, 480, 479, 481, 482, 480, 478, 475, 474],
            'volume': [1000000] * 11,
            'symbol': ['SPY'] * 11,
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_simulate_trading_no_data(self):
        """Test simulation with no data."""
        runner = OpExBacktestRunner()

        results = runner._simulate_trading(pd.DataFrame(), None)

        assert results.total_trades == 0
        assert len(results.trades) == 0

    @patch.object(OpExBacktestRunner, '_load_price_data')
    def test_run_backtest_mocked(self, mock_load):
        """Test running backtest with mocked data."""
        runner = OpExBacktestRunner(symbols=['SPY'])

        # Create mock price data
        dates = pd.date_range('2024-01-15', '2024-01-25', freq='D')
        mock_data = pd.DataFrame({
            'open': [470, 472, 475, 478, 480, 479, 481, 482, 480, 478, 475],
            'high': [473, 476, 479, 481, 483, 482, 484, 485, 483, 481, 478],
            'low': [468, 470, 473, 476, 478, 477, 479, 480, 478, 476, 473],
            'close': [472, 475, 478, 480, 479, 481, 482, 480, 478, 475, 474],
            'volume': [1000000] * 11,
            'symbol': ['SPY'] * 11,
        }, index=dates)
        mock_load.return_value = mock_data

        results = runner.run_backtest('2024-01-15', '2024-01-25', use_stored_gex=False)

        # Should have run successfully
        assert results is not None
        assert results.equity_curve is not None


class TestOpExBacktestRunnerWalkForward:
    """Tests for walk-forward validation."""

    @patch.object(OpExBacktestRunner, 'run_backtest')
    def test_walk_forward_structure(self, mock_backtest):
        """Test walk-forward returns correct structure."""
        runner = OpExBacktestRunner()

        # Mock backtest results
        mock_result = OpExBacktestResults(
            total_trades=10,
            win_rate=0.6,
            sharpe_ratio=1.2,
            total_pnl=5000,
        )
        mock_backtest.return_value = mock_result

        wf_results = runner.run_walk_forward(
            start_date='2020-01-01',
            end_date='2024-12-31',
            train_months=24,
            test_months=6,
        )

        assert 'in_sample' in wf_results
        assert 'out_of_sample' in wf_results
        assert 'is_avg_sharpe' in wf_results
        assert 'oos_avg_sharpe' in wf_results
        assert 'degradation' in wf_results

    @patch.object(OpExBacktestRunner, 'run_backtest')
    def test_walk_forward_degradation(self, mock_backtest):
        """Test walk-forward calculates degradation."""
        runner = OpExBacktestRunner()

        # Mock results: IS Sharpe = 1.5, OOS Sharpe = 1.0
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:  # Odd calls are training
                return OpExBacktestResults(sharpe_ratio=1.5, total_trades=10)
            else:  # Even calls are test
                return OpExBacktestResults(sharpe_ratio=1.0, total_trades=5)

        mock_backtest.side_effect = side_effect

        wf_results = runner.run_walk_forward(
            start_date='2020-01-01',
            end_date='2023-12-31',
            train_months=12,
            test_months=6,
        )

        # Degradation = (1.5 - 1.0) / 1.5 = 0.33
        assert wf_results['is_avg_sharpe'] > wf_results['oos_avg_sharpe']
        assert wf_results['degradation'] > 0
