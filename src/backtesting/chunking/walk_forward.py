"""
Walk-forward validation for backtesting strategies.

Prevents overfitting by testing on truly out-of-sample data using
rolling train/test windows.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import BaseStrategy
from utils import logger


@dataclass
class WalkForwardWindow:
    """Single train/test window for walk-forward validation."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_number: int


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation."""

    # Overall metrics
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    degradation_pct: float

    # Per-window results
    windows: List[Dict[str, Any]]

    # Best parameters per window
    optimal_params_by_window: List[Dict[str, Any]]

    # Aggregated returns
    oos_returns: pd.Series
    is_returns: pd.Series

    def print_summary(self):
        """Print walk-forward validation summary."""
        logger.blank()
        logger.separator()
        logger.header("WALK-FORWARD VALIDATION RESULTS")
        logger.separator()
        logger.blank()

        logger.info(f"Total Windows Tested: {len(self.windows)}")
        logger.blank()

        logger.header("IN-SAMPLE PERFORMANCE (Training Period)")
        logger.metric(f"  Sharpe Ratio:  {self.in_sample_sharpe:.2f}")
        logger.metric(f"  Total Return:  {self.in_sample_return:.1f}%")
        logger.blank()

        logger.header("OUT-OF-SAMPLE PERFORMANCE (Testing Period - True Performance)")
        logger.metric(f"  Sharpe Ratio:  {self.out_of_sample_sharpe:.2f}")
        logger.metric(f"  Total Return:  {self.out_of_sample_return:.1f}%")
        logger.blank()

        # Degradation warning
        if abs(self.degradation_pct) > 20:
            logger.warning(f"Performance Degradation: {self.degradation_pct:.1f}%")
            logger.warning("Strategy may be overfit to in-sample period!")
        elif abs(self.degradation_pct) > 10:
            logger.info(f"Performance Degradation: {self.degradation_pct:.1f}%")
            logger.info("Moderate degradation - within acceptable range")
        else:
            logger.success(f"Performance Degradation: {self.degradation_pct:.1f}%")
            logger.success("Low degradation - strategy appears robust")

        logger.blank()
        logger.separator()


class WalkForwardValidator:
    """
    Walk-forward validation for strategy optimization.

    Prevents overfitting by:
    1. Splitting data into rolling train/test windows
    2. Optimizing parameters on training period
    3. Testing with optimal parameters on out-of-sample test period
    4. Only reporting out-of-sample performance

    This simulates real-world usage where you optimize on past data
    and trade on future (unseen) data.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3
    ):
        """
        Initialize walk-forward validator.

        Args:
            engine: BacktestEngine instance
            train_months: Training period length in months (default: 12)
            test_months: Testing period length in months (default: 3)
            step_months: Step size between windows in months (default: 3)
        """
        self.engine = engine
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def generate_windows(
        self,
        start_date: str,
        end_date: str
    ) -> List[WalkForwardWindow]:
        """
        Generate rolling train/test windows.

        Args:
            start_date: Overall start date (YYYY-MM-DD)
            end_date: Overall end date (YYYY-MM-DD)

        Returns:
            List of WalkForwardWindow objects
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        windows = []
        window_num = 1
        current_start = start

        while True:
            # Calculate train period
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_months)

            # Calculate test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Check if we've exceeded the end date
            if test_end > end:
                break

            window = WalkForwardWindow(
                train_start=train_start.strftime('%Y-%m-%d'),
                train_end=train_end.strftime('%Y-%m-%d'),
                test_start=test_start.strftime('%Y-%m-%d'),
                test_end=test_end.strftime('%Y-%m-%d'),
                window_number=window_num
            )
            windows.append(window)

            # Step forward
            current_start += pd.DateOffset(months=self.step_months)
            window_num += 1

        return windows

    def validate(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio'
    ) -> WalkForwardResults:
        """
        Run walk-forward validation.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Parameter grid for optimization
            symbols: Symbol or list of symbols
            start_date: Overall start date
            end_date: Overall end date
            metric: Optimization metric

        Returns:
            WalkForwardResults object
        """
        logger.blank()
        logger.separator()
        logger.header("WALK-FORWARD VALIDATION")
        logger.separator()
        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {symbols if isinstance(symbols, list) else [symbols]}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Train: {self.train_months} months, Test: {self.test_months} months, Step: {self.step_months} months")
        logger.separator()
        logger.blank()

        # Generate windows
        windows = self.generate_windows(start_date, end_date)
        logger.info(f"Generated {len(windows)} walk-forward windows")
        logger.blank()

        # Track results
        window_results = []
        optimal_params_by_window = []
        oos_returns_list = []
        is_returns_list = []

        for window in windows:
            logger.info(f"Window {window.window_number}/{len(windows)}")
            logger.info(f"  Train: {window.train_start} to {window.train_end}")
            logger.info(f"  Test:  {window.test_start} to {window.test_end}")

            # 1. Optimize on training period
            from backtesting.optimization import GridSearchOptimizer
            optimizer = GridSearchOptimizer(self.engine)

            train_result = optimizer.optimize(
                strategy_class=strategy_class,
                param_grid=param_grid,
                symbols=symbols,
                start_date=window.train_start,
                end_date=window.train_end,
                metric=metric
            )

            best_params = train_result['best_params']
            optimal_params_by_window.append(best_params)

            # Get in-sample stats
            is_stats = train_result['best_portfolio'].stats()
            is_sharpe = float(is_stats.get('Sharpe Ratio', 0))
            is_return = float(is_stats.get('Total Return [%]', 0))

            # Collect in-sample returns
            is_returns = train_result['best_portfolio'].returns()
            is_returns_list.append(is_returns)

            logger.success(f"  Optimal params: {best_params}")
            logger.metric(f"  In-sample Sharpe: {is_sharpe:.2f}")

            # 2. Test on out-of-sample period with optimal params
            test_strategy = strategy_class(**best_params)

            # Load test data
            test_data = self.engine.data_loader.load_symbols(
                symbols if isinstance(symbols, list) else [symbols],
                window.test_start,
                window.test_end
            )

            # Run backtest on test period
            if isinstance(symbols, str):
                symbols = [symbols]

            if len(symbols) == 1:
                test_portfolio = self.engine._run_single_symbol(
                    test_strategy, test_data, symbols[0], 'close'
                )
            else:
                test_portfolio = self.engine._run_multiple_symbols(
                    test_strategy, test_data, symbols, 'close'
                )

            # Get out-of-sample stats
            oos_stats = test_portfolio.stats()
            oos_sharpe = float(oos_stats.get('Sharpe Ratio', 0))
            oos_return = float(oos_stats.get('Total Return [%]', 0))

            # Collect out-of-sample returns
            oos_returns = test_portfolio.returns()
            oos_returns_list.append(oos_returns)

            # Calculate degradation
            degradation = ((oos_sharpe - is_sharpe) / is_sharpe * 100) if is_sharpe != 0 else 0

            logger.metric(f"  Out-of-sample Sharpe: {oos_sharpe:.2f}")
            if abs(degradation) > 20:
                logger.warning(f"  Degradation: {degradation:.1f}%")
            else:
                logger.info(f"  Degradation: {degradation:.1f}%")

            # Store window results
            window_results.append({
                'window': window.window_number,
                'train_period': f"{window.train_start} to {window.train_end}",
                'test_period': f"{window.test_start} to {window.test_end}",
                'best_params': best_params,
                'is_sharpe': is_sharpe,
                'is_return': is_return,
                'oos_sharpe': oos_sharpe,
                'oos_return': oos_return,
                'degradation_pct': degradation
            })

            logger.blank()

        # Aggregate results
        oos_returns_combined = pd.concat(oos_returns_list)
        is_returns_combined = pd.concat(is_returns_list)

        # Calculate overall metrics
        overall_is_sharpe = self._calculate_sharpe(is_returns_combined)
        overall_oos_sharpe = self._calculate_sharpe(oos_returns_combined)
        overall_is_return = (1 + is_returns_combined).prod() - 1
        overall_oos_return = (1 + oos_returns_combined).prod() - 1
        overall_degradation = ((overall_oos_sharpe - overall_is_sharpe) / overall_is_sharpe * 100) if overall_is_sharpe != 0 else 0

        results = WalkForwardResults(
            in_sample_sharpe=overall_is_sharpe,
            out_of_sample_sharpe=overall_oos_sharpe,
            in_sample_return=overall_is_return * 100,
            out_of_sample_return=overall_oos_return * 100,
            degradation_pct=overall_degradation,
            windows=window_results,
            optimal_params_by_window=optimal_params_by_window,
            oos_returns=oos_returns_combined,
            is_returns=is_returns_combined
        )

        results.print_summary()

        return results

    def _calculate_sharpe(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)
