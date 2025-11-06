"""
Grid search parameter optimization for backtesting strategies.

Exhaustively tests all parameter combinations to find optimal values
based on specified metrics (Sharpe ratio, total return, max drawdown).
"""

import pandas as pd
from itertools import product
from typing import Dict, List, Any, Union, TYPE_CHECKING

from utils import logger

if TYPE_CHECKING:
    from backtesting.engine.backtest_engine import BacktestEngine


class GridSearchOptimizer:
    """
    Grid search optimizer for strategy parameters.

    Performs exhaustive search over all parameter combinations
    and tracks the best performing set based on the chosen metric.
    """

    def __init__(self, engine: 'BacktestEngine'):
        """
        Initialize grid search optimizer.

        Args:
            engine: BacktestEngine instance to use for running backtests
        """
        self.engine = engine

    def optimize(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters over a grid.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary mapping parameter names to lists of values
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            Dictionary with best parameters and results:
            {
                'best_params': Dict[str, Any],
                'best_value': float,
                'best_portfolio': Portfolio,
                'metric': str
            }

        Raises:
            ValueError: If unknown metric is specified
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate metric before starting optimization
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Load data for the optimization period
        data = self.engine.data_loader.load_symbols(symbols, start_date, end_date)

        # Log optimization header
        logger.blank()
        logger.separator()
        logger.header(f"Optimizing {strategy_class.__name__}")
        logger.info(f"Parameter grid: {param_grid}")
        logger.separator()
        logger.blank()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Initialize best tracking
        best_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        best_portfolio = None

        # Test each parameter combination
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            try:
                # Create strategy instance with these parameters
                strategy = strategy_class(**params)

                # Run backtest
                if len(symbols) == 1:
                    portfolio = self.engine._run_single_symbol(strategy, data, symbols[0], 'close')
                else:
                    portfolio = self.engine._run_multiple_symbols(strategy, data, symbols, 'close')

                # Get performance statistics
                stats = portfolio.stats()

                if stats is None:
                    continue

                # Extract the metric value
                value = self._extract_metric_value(stats, metric)

                # Check if this is the best so far
                if self._is_better(value, best_value, metric):
                    best_value = value
                    best_params = params
                    best_portfolio = portfolio

                # Log this combination's result
                logger.metric(f"Params: {params} -> {metric}: {value:.4f}")

            except (ValueError, TypeError) as e:
                # Skip invalid parameter combinations (e.g., fast_window >= slow_window)
                logger.warning(f"Skipping invalid combination {params}: {e}")
                continue

        # Log best results
        logger.blank()
        logger.separator()
        logger.success(f"Best parameters: {best_params}")
        logger.profit(f"Best {metric}: {best_value:.4f}")
        logger.separator()
        logger.blank()

        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_portfolio': best_portfolio,
            'metric': metric
        }

    def _extract_metric_value(self, stats: Dict[str, Any], metric: str) -> float:
        """
        Extract metric value from portfolio statistics.

        Args:
            stats: Portfolio statistics dictionary
            metric: Metric name ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            Metric value as float

        Raises:
            ValueError: If unknown metric is specified
        """
        if metric == 'sharpe_ratio':
            return float(stats.get('Sharpe Ratio', float('-inf')))  # type: ignore[arg-type]
        elif metric == 'total_return':
            return float(stats.get('Total Return [%]', float('-inf')))  # type: ignore[arg-type]
        elif metric == 'max_drawdown':
            return float(stats.get('Max Drawdown [%]', float('inf')))  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _is_better(self, value: float, best_value: float, metric: str) -> bool:
        """
        Determine if a value is better than the current best.

        Args:
            value: New value to compare
            best_value: Current best value
            metric: Metric being optimized

        Returns:
            True if value is better than best_value
        """
        if metric == 'max_drawdown':
            # For drawdown, smaller absolute value is better
            return value < best_value
        else:
            # For other metrics, larger is better
            return value > best_value
