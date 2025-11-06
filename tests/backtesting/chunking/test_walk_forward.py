"""
Tests for walk-forward validation module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtesting.chunking.walk_forward import (
    WalkForwardWindow,
    WalkForwardResults,
    WalkForwardValidator
)
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover


@pytest.fixture
def engine():
    """Create a BacktestEngine instance."""
    return BacktestEngine(initial_capital=10000, fees=0.001)


@pytest.fixture
def mock_data():
    """Create mock price data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    np.random.seed(42)

    # Generate trending price data
    trend = np.linspace(100, 150, 500)
    noise = np.random.randn(500) * 5
    close_prices = trend + noise

    df = pd.DataFrame({
        'open': close_prices + np.random.randn(500) * 0.5,
        'high': close_prices + np.abs(np.random.randn(500) * 1.0),
        'low': close_prices - np.abs(np.random.randn(500) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 500),
        'symbol': 'TEST'
    }, index=dates)

    df = df.set_index('symbol', append=True)
    df = df.swaplevel()

    return df


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, data):
        self.data = data

    def load_symbols(self, symbols, start, end):
        """Load mock data."""
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        # Use level 1 (the date level) instead of 'Date'
        mask = (self.data.index.get_level_values(1) >= start) & \
               (self.data.index.get_level_values(1) <= end)
        return self.data[mask]


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_window_creation(self):
        """Test creating a window."""
        window = WalkForwardWindow(
            train_start='2020-01-01',
            train_end='2020-06-30',
            test_start='2020-07-01',
            test_end='2020-09-30',
            window_number=1
        )

        assert window.train_start == '2020-01-01'
        assert window.train_end == '2020-06-30'
        assert window.test_start == '2020-07-01'
        assert window.test_end == '2020-09-30'
        assert window.window_number == 1


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator class."""

    def test_initialization(self, engine):
        """Test validator initialization."""
        validator = WalkForwardValidator(
            engine=engine,
            train_months=12,
            test_months=3,
            step_months=3
        )

        assert validator.engine == engine
        assert validator.train_months == 12
        assert validator.test_months == 3
        assert validator.step_months == 3

    def test_generate_windows(self, engine):
        """Test window generation."""
        validator = WalkForwardValidator(
            engine=engine,
            train_months=6,
            test_months=3,
            step_months=3
        )

        windows = validator.generate_windows(
            start_date='2020-01-01',
            end_date='2021-12-31'
        )

        # Should generate multiple windows
        assert len(windows) > 0

        # Check first window
        first = windows[0]
        assert first.window_number == 1
        assert first.train_start == '2020-01-01'
        # Train period is 6 months
        assert first.train_end == '2020-07-01'
        # Test starts immediately after train
        assert first.test_start == '2020-07-01'
        # Test period is 3 months
        assert first.test_end == '2020-10-01'

    def test_windows_dont_overlap_improperly(self, engine):
        """Test that windows step forward correctly."""
        validator = WalkForwardValidator(
            engine=engine,
            train_months=6,
            test_months=3,
            step_months=3
        )

        windows = validator.generate_windows(
            start_date='2020-01-01',
            end_date='2021-12-31'
        )

        # Each window should be numbered sequentially
        for i, window in enumerate(windows):
            assert window.window_number == i + 1

        # Check that test periods don't overlap
        # (train periods will overlap due to rolling window)
        if len(windows) > 1:
            for i in range(len(windows) - 1):
                test_end_i = pd.to_datetime(windows[i].test_end)
                test_start_next = pd.to_datetime(windows[i + 1].test_start)
                # Test periods should not overlap
                assert test_end_i <= test_start_next

    def test_validate_with_mock_data(self, engine, mock_data):
        """Test walk-forward validation with mock data."""
        # Use mock data loader
        engine.data_loader = MockDataLoader(mock_data)

        validator = WalkForwardValidator(
            engine=engine,
            train_months=6,
            test_months=3,
            step_months=3
        )

        param_grid = {
            'fast_window': [10, 20],
            'slow_window': [50]
        }

        results = validator.validate(
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            symbols='TEST',
            start_date='2020-01-01',
            end_date='2020-12-31',
            metric='sharpe_ratio'
        )

        # Check results structure
        assert isinstance(results, WalkForwardResults)
        assert results.in_sample_sharpe is not None
        assert results.out_of_sample_sharpe is not None
        assert results.in_sample_return is not None
        assert results.out_of_sample_return is not None
        assert results.degradation_pct is not None
        assert len(results.windows) > 0
        assert len(results.optimal_params_by_window) > 0
        assert isinstance(results.oos_returns, pd.Series)
        assert isinstance(results.is_returns, pd.Series)

    def test_calculate_sharpe(self, engine):
        """Test Sharpe ratio calculation."""
        validator = WalkForwardValidator(engine=engine)

        # Create returns with known properties
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)  # Daily returns

        sharpe = validator._calculate_sharpe(returns, periods_per_year=252)

        # Should return a float
        assert isinstance(sharpe, float)

        # Should not be NaN
        assert not np.isnan(sharpe)

    def test_calculate_sharpe_zero_std(self, engine):
        """Test Sharpe with zero standard deviation."""
        validator = WalkForwardValidator(engine=engine)

        # All zeros
        returns = pd.Series([0.0] * 100)
        sharpe = validator._calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_calculate_sharpe_empty(self, engine):
        """Test Sharpe with empty series."""
        validator = WalkForwardValidator(engine=engine)

        returns = pd.Series([])
        sharpe = validator._calculate_sharpe(returns)
        assert sharpe == 0.0


class TestWalkForwardResults:
    """Tests for WalkForwardResults dataclass."""

    def test_results_creation(self):
        """Test creating results object."""
        results = WalkForwardResults(
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.2,
            in_sample_return=25.0,
            out_of_sample_return=20.0,
            degradation_pct=-20.0,
            windows=[],
            optimal_params_by_window=[],
            oos_returns=pd.Series([0.01, 0.02]),
            is_returns=pd.Series([0.015, 0.025])
        )

        assert results.in_sample_sharpe == 1.5
        assert results.out_of_sample_sharpe == 1.2
        assert results.degradation_pct == -20.0

    def test_print_summary(self):
        """Test summary printing."""
        results = WalkForwardResults(
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.2,
            in_sample_return=25.0,
            out_of_sample_return=20.0,
            degradation_pct=-20.0,
            windows=[
                {
                    'window': 1,
                    'is_sharpe': 1.5,
                    'oos_sharpe': 1.2
                }
            ],
            optimal_params_by_window=[],
            oos_returns=pd.Series([0.01, 0.02]),
            is_returns=pd.Series([0.015, 0.025])
        )

        # Should not raise an error (logger outputs to custom logger, not stdout)
        results.print_summary()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
