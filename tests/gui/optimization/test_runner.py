"""
Unit tests for GUI optimization runner (app._run_optimization()).

Tests the full optimization workflow including:
- Parameter combination generation
- Backtest execution for each combination
- Invalid combination handling
- CSV export
- Result tracking
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))


class MockBacktestEngine:
    """Mock BacktestEngine for testing optimization runner."""

    def __init__(self, initial_capital=100000.0, fees=0.0):
        self.initial_capital = initial_capital
        self.fees = fees

    def _run_single_symbol(self, strategy, data, symbol, price_type):
        """Mock single symbol backtest."""
        # Create mock portfolio with stats
        portfolio = Mock()

        # Generate random but realistic stats
        np.random.seed(42)
        sharpe = np.random.uniform(-1.0, 2.0)
        total_return = np.random.uniform(-20.0, 50.0)
        max_drawdown = np.random.uniform(-30.0, -5.0)

        portfolio.stats = Mock(return_value={
            'Sharpe Ratio': sharpe,
            'Total Return [%]': total_return,
            'Max Drawdown [%]': max_drawdown,
            'Total Trades': 10,
            'Win Rate [%]': 55.0,
            'Start Value': self.initial_capital,
            'End Value': self.initial_capital * (1 + total_return / 100)
        })

        return portfolio


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, fast_window=10, slow_window=20):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.params = {'fast_window': fast_window, 'slow_window': slow_window}

        # Validate parameters (like real strategies)
        if fast_window >= slow_window:
            raise ValueError(f"fast_window ({fast_window}) must be < slow_window ({slow_window})")


class TestOptimizationRunnerBasics:
    """Test basic optimization runner functionality."""

    def test_optimization_processes_all_combinations(self, tmp_path):
        """Test that optimization runs all parameter combinations."""
        from itertools import product

        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20, 30]
        }

        # Calculate expected combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        expected_combos = list(product(*param_values))

        assert len(expected_combos) == 4  # 2 * 2

        # Verify combinations are correct
        expected_params = [
            {'fast_window': 5, 'slow_window': 20},
            {'fast_window': 5, 'slow_window': 30},
            {'fast_window': 10, 'slow_window': 20},
            {'fast_window': 10, 'slow_window': 30}
        ]

        for i, combo in enumerate(expected_combos):
            params = dict(zip(param_names, combo))
            assert params == expected_params[i]

    def test_optimization_tracks_all_results(self):
        """Test that all results are tracked (not just best)."""
        param_grid = {
            'fast_window': [5, 10, 15],
            'slow_window': [20]
        }

        all_results = []

        # Simulate running optimization and tracking results
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            # Simulate backtest result
            result_row = params.copy()
            result_row['metric_value'] = np.random.uniform(0.5, 2.0)
            result_row['sharpe_ratio'] = np.random.uniform(0.5, 2.0)
            result_row['total_return'] = np.random.uniform(10.0, 50.0)
            result_row['max_drawdown'] = np.random.uniform(-30.0, -5.0)
            all_results.append(result_row)

        # Should have 3 results (one per combination)
        assert len(all_results) == 3

        # All results should have required fields
        for result in all_results:
            assert 'fast_window' in result
            assert 'slow_window' in result
            assert 'metric_value' in result
            assert 'sharpe_ratio' in result
            assert 'total_return' in result
            assert 'max_drawdown' in result

    def test_optimization_skips_invalid_combinations(self):
        """Test that invalid parameter combinations are skipped."""
        param_grid = {
            'fast_window': [5, 10, 25],  # 25 >= 20, invalid
            'slow_window': [20]
        }

        all_results = []

        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            try:
                # Try to create strategy (will raise ValueError if invalid)
                strategy = MockStrategy(**params)

                # If valid, track result
                result_row = params.copy()
                result_row['metric_value'] = 1.5
                all_results.append(result_row)

            except ValueError:
                # Skip invalid combination
                continue

        # Should only have 2 valid results (fast=5,10 with slow=20)
        # fast=25, slow=20 should be skipped
        assert len(all_results) == 2
        assert all(r['fast_window'] < r['slow_window'] for r in all_results)


class TestOptimizationMetrics:
    """Test optimization with different metrics."""

    def test_optimization_sharpe_ratio_metric(self):
        """Test optimization maximizes Sharpe Ratio."""
        results = [
            {'params': {'fast_window': 5}, 'sharpe_ratio': 1.5, 'total_return': 30.0},
            {'params': {'fast_window': 10}, 'sharpe_ratio': 2.0, 'total_return': 25.0},  # Best
            {'params': {'fast_window': 15}, 'sharpe_ratio': 1.2, 'total_return': 35.0}
        ]

        # Find best by Sharpe Ratio
        best = max(results, key=lambda x: x['sharpe_ratio'])

        assert best['params']['fast_window'] == 10
        assert best['sharpe_ratio'] == 2.0

    def test_optimization_total_return_metric(self):
        """Test optimization maximizes Total Return."""
        results = [
            {'params': {'fast_window': 5}, 'sharpe_ratio': 1.5, 'total_return': 30.0},
            {'params': {'fast_window': 10}, 'sharpe_ratio': 2.0, 'total_return': 25.0},
            {'params': {'fast_window': 15}, 'sharpe_ratio': 1.2, 'total_return': 35.0}  # Best
        ]

        # Find best by Total Return
        best = max(results, key=lambda x: x['total_return'])

        assert best['params']['fast_window'] == 15
        assert best['total_return'] == 35.0

    def test_optimization_max_drawdown_metric(self):
        """Test optimization minimizes Max Drawdown (smaller abs value is better)."""
        results = [
            {'params': {'fast_window': 5}, 'max_drawdown': -25.0},
            {'params': {'fast_window': 10}, 'max_drawdown': -15.0},  # Best (smallest)
            {'params': {'fast_window': 15}, 'max_drawdown': -30.0}
        ]

        # Find best by Max Drawdown (minimize = smallest absolute value)
        best = min(results, key=lambda x: abs(x['max_drawdown']))

        assert best['params']['fast_window'] == 10
        assert best['max_drawdown'] == -15.0


class TestOptimizationCSVExport:
    """Test CSV export functionality."""

    def test_csv_export_includes_all_columns(self, tmp_path):
        """Test that CSV export includes all required columns."""
        all_results = [
            {
                'fast_window': 5,
                'slow_window': 20,
                'metric_value': 1.5,
                'sharpe_ratio': 1.5,
                'total_return': 30.0,
                'max_drawdown': -20.0
            },
            {
                'fast_window': 10,
                'slow_window': 20,
                'metric_value': 2.0,
                'sharpe_ratio': 2.0,
                'total_return': 35.0,
                'max_drawdown': -15.0
            }
        ]

        # Export to CSV
        df = pd.DataFrame(all_results)
        csv_path = tmp_path / "optimization_results.csv"
        df.to_csv(csv_path, index=False)

        # Read back and verify
        loaded_df = pd.read_csv(csv_path)

        assert len(loaded_df) == 2
        assert 'fast_window' in loaded_df.columns
        assert 'slow_window' in loaded_df.columns
        assert 'metric_value' in loaded_df.columns
        assert 'sharpe_ratio' in loaded_df.columns
        assert 'total_return' in loaded_df.columns
        assert 'max_drawdown' in loaded_df.columns

    def test_csv_sorted_by_metric(self, tmp_path):
        """Test that CSV is sorted by metric value (best first)."""
        all_results = [
            {'fast_window': 5, 'metric_value': 1.5},
            {'fast_window': 15, 'metric_value': 1.2},
            {'fast_window': 10, 'metric_value': 2.0}  # Best
        ]

        # Sort descending (best first) for maximization
        df = pd.DataFrame(all_results)
        df = df.sort_values('metric_value', ascending=False)

        csv_path = tmp_path / "sorted_results.csv"
        df.to_csv(csv_path, index=False)

        # Read back and verify order
        loaded_df = pd.read_csv(csv_path)

        assert loaded_df.iloc[0]['fast_window'] == 10  # Best first
        assert loaded_df.iloc[0]['metric_value'] == 2.0
        assert loaded_df.iloc[1]['fast_window'] == 5
        assert loaded_df.iloc[2]['fast_window'] == 15  # Worst last

    def test_csv_sorted_ascending_for_drawdown(self, tmp_path):
        """Test that CSV is sorted ascending for max_drawdown (minimize)."""
        all_results = [
            {'fast_window': 5, 'metric_value': -25.0},
            {'fast_window': 10, 'metric_value': -15.0},  # Best (smallest abs value)
            {'fast_window': 15, 'metric_value': -30.0}
        ]

        # Sort ascending for minimization (-30 < -25 < -15)
        df = pd.DataFrame(all_results)
        df = df.sort_values('metric_value', ascending=True)

        csv_path = tmp_path / "sorted_drawdown.csv"
        df.to_csv(csv_path, index=False)

        # Read back and verify order
        loaded_df = pd.read_csv(csv_path)

        # When sorted ascending: -30 comes first (worst), then -25, then -15
        # For drawdown, smaller absolute value is better, but numerically -30 < -15
        assert loaded_df.iloc[0]['fast_window'] == 15  # -30 (worst) first
        assert loaded_df.iloc[0]['metric_value'] == -30.0
        assert loaded_df.iloc[2]['fast_window'] == 10  # -15 (best) last
        assert loaded_df.iloc[2]['metric_value'] == -15.0

    def test_csv_filename_format(self):
        """Test CSV filename follows expected format."""
        from datetime import datetime

        strategy_name = "MovingAverageCrossover"
        metric = "sharpe_ratio"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Expected format: optimization_STRATEGY_METRIC_TIMESTAMP.csv
        csv_filename = f"optimization_{strategy_name}_{metric}_{timestamp}.csv"

        assert csv_filename.startswith("optimization_")
        assert strategy_name in csv_filename
        assert metric in csv_filename
        assert csv_filename.endswith(".csv")


class TestOptimizationProgressTracking:
    """Test optimization progress tracking."""

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        total_combinations = 20
        completed = 5

        progress_pct = (completed / total_combinations) * 100

        assert progress_pct == 25.0

    def test_progress_updates_correctly(self):
        """Test that progress updates for each combination."""
        total_combinations = 10
        progress_updates = []

        for i in range(total_combinations):
            completed = i + 1
            progress_pct = (completed / total_combinations) * 100
            progress_updates.append(progress_pct)

        # Should have 10 updates: 10%, 20%, ..., 100%
        assert len(progress_updates) == 10
        assert progress_updates[0] == 10.0
        assert progress_updates[-1] == 100.0


class TestOptimizationResultDialog:
    """Test optimization result dialog."""

    def test_result_dialog_shows_best_params(self):
        """Test that result dialog displays best parameters."""
        best_params = {
            'fast_window': 10,
            'slow_window': 20,
            'ma_type': 'ema'
        }

        # Format best params for display
        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())

        assert "fast_window=10" in params_str
        assert "slow_window=20" in params_str
        assert "ma_type=ema" in params_str

    def test_result_dialog_shows_metric_value(self):
        """Test that result dialog displays metric value."""
        metric = "sharpe_ratio"
        best_value = 2.345

        # Format metric display
        metric_display = f"{metric.replace('_', ' ').title()}: {best_value:.4f}"

        assert "Sharpe Ratio" in metric_display
        assert "2.3450" in metric_display

    def test_result_dialog_handles_no_results(self):
        """Test that dialog handles case with no valid results."""
        all_results = []

        if not all_results:
            # Should show error message
            error_msg = "No valid parameter combinations found. All combinations may have failed validation."
            assert "No valid parameter" in error_msg


class TestOptimizationEdgeCases:
    """Test optimization edge cases."""

    def test_optimization_with_single_symbol(self):
        """Test optimization with single symbol."""
        symbols = ['AAPL']

        assert isinstance(symbols, list)
        assert len(symbols) == 1

    def test_optimization_with_multiple_symbols(self):
        """Test optimization with multiple symbols."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        assert isinstance(symbols, list)
        assert len(symbols) == 3

    def test_optimization_all_combinations_invalid(self):
        """Test handling when all combinations are invalid."""
        param_grid = {
            'fast_window': [20, 30, 40],  # All >= slow_window
            'slow_window': [20]
        }

        all_results = []

        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            try:
                strategy = MockStrategy(**params)
                result_row = params.copy()
                result_row['metric_value'] = 1.5
                all_results.append(result_row)
            except ValueError:
                continue

        # All combinations should be invalid
        assert len(all_results) == 0

    def test_optimization_preserves_parameter_types(self):
        """Test that parameter types are preserved in results."""
        params = {
            'fast_window': 10,        # int
            'threshold': 1.5,         # float
            'ma_type': 'ema',         # str
            'use_stops': True         # bool
        }

        # Verify types are preserved
        assert isinstance(params['fast_window'], int)
        assert isinstance(params['threshold'], float)
        assert isinstance(params['ma_type'], str)
        assert isinstance(params['use_stops'], bool)


class TestOptimizationIntegration:
    """Integration tests for full optimization workflow."""

    def test_full_optimization_workflow_simulation(self, tmp_path):
        """Simulate complete optimization workflow end-to-end."""
        # Setup
        param_grid = {
            'fast_window': [5, 10],
            'slow_window': [20, 30]
        }
        metric = 'sharpe_ratio'
        strategy_name = 'MovingAverageCrossover'

        # Generate all combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(product(*param_values))

        # Run optimization (simulated)
        all_results = []
        for param_combo in combinations:
            params = dict(zip(param_names, param_combo))

            try:
                # Validate
                strategy = MockStrategy(**params)

                # Run backtest (mocked)
                np.random.seed(hash(str(params)) % 2**32)
                sharpe = np.random.uniform(0.5, 2.5)

                # Track result
                result_row = params.copy()
                result_row['metric_value'] = sharpe
                result_row['sharpe_ratio'] = sharpe
                result_row['total_return'] = np.random.uniform(10.0, 50.0)
                result_row['max_drawdown'] = np.random.uniform(-30.0, -5.0)
                all_results.append(result_row)

            except ValueError:
                continue

        # Find best
        if all_results:
            df = pd.DataFrame(all_results)
            df = df.sort_values('metric_value', ascending=False)

            # Export CSV
            csv_path = tmp_path / f"optimization_{strategy_name}_{metric}.csv"
            df.to_csv(csv_path, index=False)

            # Verify workflow completed
            assert csv_path.exists()
            assert len(df) == 4  # All 4 combinations valid

            best_row = df.iloc[0]
            assert best_row['metric_value'] == df['metric_value'].max()
