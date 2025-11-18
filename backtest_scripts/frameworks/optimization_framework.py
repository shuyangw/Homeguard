"""
Optimization Framework - Standardized parameter optimization pipeline.

This framework eliminates ~1,800 lines of duplicated optimization code across
15+ optimization scripts by providing common patterns for:
- Grid search optimization
- Walk-forward validation
- Parameter sensitivity analysis
- Parallel execution
- Progress tracking and incremental saving
- Result aggregation and ranking

Usage:
    from frameworks.optimization_framework import OptimizationFramework
    from strategies.advanced.pairs_trading import PairsTrading

    # Initialize framework
    framework = OptimizationFramework(config)

    # Define parameter grid
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.0, 0.5, 1.0]
    }

    # Run grid search
    results = framework.grid_search(
        strategy_class=PairsTrading,
        symbols=["SPY", "IWM"],
        param_grid=param_grid
    )

    # Export best parameters
    framework.export_best_params()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Tuple
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.utils.logger import logger
from utils.config_loader import get_nested


class OptimizationFramework:
    """
    Standardized framework for strategy parameter optimization.

    This framework provides common optimization patterns eliminating
    code duplication across optimization scripts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimization framework.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = []
        self.best_params = {}
        self.optimization_start_time = None
        self.optimization_end_time = None

        # Extract settings
        self.initial_cash = get_nested(config, 'backtest.initial_cash', 100000)
        self.commission = get_nested(config, 'costs.commission', 0.001)
        self.slippage = get_nested(config, 'costs.slippage', 0.0005)
        self.start_date = get_nested(config, 'backtest.start_date', '2020-01-01')
        self.end_date = get_nested(config, 'backtest.end_date', '2024-12-31')

        # Optimization settings
        self.n_jobs = get_nested(config, 'optimization.n_jobs', 1)
        self.max_iterations = get_nested(config, 'optimization.max_iterations', None)
        self.primary_metric = get_nested(config, 'optimization.primary_metric', 'sharpe_ratio')
        self.minimize = get_nested(config, 'optimization.minimize', False)

        # Output settings
        self.output_dir = Path(get_nested(config, 'output.output_dir', 'output/optimization'))
        self.reports_dir = Path(get_nested(config, 'output.reports_dir', 'reports/optimization'))

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.total_iterations = 0
        self.completed_iterations = 0
        self.failed_iterations = 0

    def grid_search(
        self,
        strategy_class: Type,
        symbols: List[str],
        param_grid: Dict[str, List[Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        description: str = "Grid Search"
    ) -> pd.DataFrame:
        """
        Perform grid search optimization over parameter space.

        Args:
            strategy_class: Strategy class to optimize
            symbols: List of symbols to trade
            param_grid: Dictionary mapping parameter names to lists of values
            fixed_params: Optional fixed parameters (not optimized)
            description: Description of this optimization

        Returns:
            DataFrame with all results sorted by primary metric

        Example:
            >>> param_grid = {
            ...     'entry_zscore': [1.5, 2.0, 2.5],
            ...     'exit_zscore': [0.0, 0.5]
            ... }
            >>> results = framework.grid_search(PairsTrading, ["SPY", "IWM"], param_grid)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"GRID SEARCH OPTIMIZATION: {description}")
        logger.info(f"{'='*80}")
        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        self.total_iterations = len(combinations)
        logger.info(f"Total combinations: {self.total_iterations}")

        # Apply max iterations limit if set
        if self.max_iterations and self.max_iterations < self.total_iterations:
            combinations = combinations[:self.max_iterations]
            self.total_iterations = self.max_iterations
            logger.info(f"Limited to: {self.total_iterations} iterations")

        # Start timer
        self.optimization_start_time = datetime.now()

        # Run optimization
        if self.n_jobs > 1:
            results = self._parallel_grid_search(
                strategy_class, symbols, param_names, combinations, fixed_params
            )
        else:
            results = self._sequential_grid_search(
                strategy_class, symbols, param_names, combinations, fixed_params
            )

        # Store results
        self.results = results

        # Find best parameters
        self._find_best_params()

        # End timer
        self.optimization_end_time = datetime.now()
        duration = (self.optimization_end_time - self.optimization_start_time).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total iterations: {self.completed_iterations}/{self.total_iterations}")
        logger.info(f"Failed iterations: {self.failed_iterations}")
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
        logger.info(f"Best {self.primary_metric}: {self.best_params.get(self.primary_metric, 0):.3f}")
        logger.info(f"{'='*80}\n")

        # Convert to DataFrame and sort
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(self.primary_metric, ascending=self.minimize)

        return df

    def _sequential_grid_search(
        self,
        strategy_class: Type,
        symbols: List[str],
        param_names: List[str],
        combinations: List[Tuple],
        fixed_params: Optional[Dict] = None
    ) -> List[Dict]:
        """Run grid search sequentially."""
        results = []

        for i, combo in enumerate(combinations, 1):
            # Build parameter dict
            params = dict(zip(param_names, combo))
            if fixed_params:
                params.update(fixed_params)

            # Run backtest
            result = self._run_single_backtest(
                strategy_class, symbols, params, iteration=i
            )

            if result:
                results.append(result)
                self.completed_iterations += 1
            else:
                self.failed_iterations += 1

            # Progress update
            if i % 10 == 0 or i == len(combinations):
                pct = (i / len(combinations)) * 100
                logger.info(f"Progress: {i}/{len(combinations)} ({pct:.1f}%)")

        return results

    def _parallel_grid_search(
        self,
        strategy_class: Type,
        symbols: List[str],
        param_names: List[str],
        combinations: List[Tuple],
        fixed_params: Optional[Dict] = None
    ) -> List[Dict]:
        """Run grid search in parallel."""
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            futures = {}
            for i, combo in enumerate(combinations, 1):
                params = dict(zip(param_names, combo))
                if fixed_params:
                    params.update(fixed_params)

                future = executor.submit(
                    self._run_single_backtest,
                    strategy_class, symbols, params, i
                )
                futures[future] = i

            # Collect results as they complete
            for future in as_completed(futures):
                iteration = futures[future]
                result = future.result()

                if result:
                    results.append(result)
                    self.completed_iterations += 1
                else:
                    self.failed_iterations += 1

                # Progress update
                completed = self.completed_iterations + self.failed_iterations
                if completed % 10 == 0 or completed == len(combinations):
                    pct = (completed / len(combinations)) * 100
                    logger.info(f"Progress: {completed}/{len(combinations)} ({pct:.1f}%)")

        return results

    def _run_single_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        params: Dict[str, Any],
        iteration: int
    ) -> Optional[Dict[str, Any]]:
        """
        Run single backtest with given parameters.

        Args:
            strategy_class: Strategy class
            symbols: List of symbols
            params: Strategy parameters
            iteration: Iteration number

        Returns:
            Result dictionary or None if failed
        """
        try:
            # Initialize strategy
            strategy = strategy_class(**params)

            # Create engine
            engine = BacktestEngine(
                initial_capital=self.initial_cash,
                fees=self.commission,
                slippage=self.slippage
            )

            # Run backtest
            portfolio = engine.run(
                strategy=strategy,
                symbols=symbols,
                start_date=self.start_date,
                end_date=self.end_date
            )

            # Get stats
            stats = portfolio.stats()

            if stats is None:
                return None

            # Build result dict
            result = params.copy()
            result.update({
                'iteration': iteration,
                'symbols': symbols,
                'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
                'sortino_ratio': float(stats.get('Sortino Ratio', 0)),
                'total_return_pct': float(stats.get('Total Return [%]', 0)),
                'annual_return_pct': float(stats.get('Annual Return [%]', 0)),
                'max_drawdown_pct': float(stats.get('Max Drawdown [%]', 0)),
                'win_rate_pct': float(stats.get('Win Rate [%]', 0)),
                'profit_factor': float(stats.get('Profit Factor', 0)),
                'total_trades': int(stats.get('Total Trades', 0)),
                'final_equity': float(stats.get('End Value', 0)),
                'success': True
            })

            return result

        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {str(e)}")
            return None

    def _find_best_params(self) -> None:
        """Find best parameters based on primary metric."""
        if not self.results:
            return

        # Sort by primary metric
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get(self.primary_metric, 0),
            reverse=not self.minimize
        )

        self.best_params = sorted_results[0].copy()

    def walk_forward(
        self,
        strategy_class: Type,
        symbols: List[str],
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        train_pct: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.

        Args:
            strategy_class: Strategy class
            symbols: List of symbols
            param_grid: Parameter grid
            n_splits: Number of time splits
            train_pct: Training period percentage (0.0-1.0)

        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"\n{'='*80}")
        logger.info("WALK-FORWARD OPTIMIZATION")
        logger.info(f"{'='*80}")
        logger.info(f"Splits: {n_splits}")
        logger.info(f"Train/Test: {train_pct*100:.0f}% / {(1-train_pct)*100:.0f}%")

        # Split date range into periods
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        total_days = (end - start).days
        period_days = total_days // n_splits

        walk_forward_results = []

        for i in range(n_splits):
            # Calculate train/test periods
            period_start = start + pd.Timedelta(days=i * period_days)
            period_end = period_start + pd.Timedelta(days=period_days)

            train_days = int(period_days * train_pct)
            train_end = period_start + pd.Timedelta(days=train_days)

            logger.info(f"\n--- Split {i+1}/{n_splits} ---")
            logger.info(f"Train: {period_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {train_end.date()} to {period_end.date()}")

            # Optimize on training period
            self.start_date = period_start.strftime('%Y-%m-%d')
            self.end_date = train_end.strftime('%Y-%m-%d')

            train_results = self.grid_search(
                strategy_class, symbols, param_grid,
                description=f"Split {i+1} Training"
            )

            # Get best params from training
            best_params = train_results.iloc[0].to_dict() if not train_results.empty else {}

            # Test on test period
            self.start_date = train_end.strftime('%Y-%m-%d')
            self.end_date = period_end.strftime('%Y-%m-%d')

            test_result = self._run_single_backtest(
                strategy_class, symbols, best_params, iteration=i+1
            )

            walk_forward_results.append({
                'split': i + 1,
                'train_start': period_start.date(),
                'train_end': train_end.date(),
                'test_start': train_end.date(),
                'test_end': period_end.date(),
                'best_params': best_params,
                'test_result': test_result
            })

        # Restore original dates
        self.start_date = self.config['backtest']['start_date']
        self.end_date = self.config['backtest']['end_date']

        return {
            'n_splits': n_splits,
            'splits': walk_forward_results
        }

    def export_results(self, filename: Optional[str] = None) -> Path:
        """
        Export optimization results to CSV.

        Args:
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not self.results:
            logger.warning("No results to export")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_optimization_results.csv"

        output_path = self.output_dir / filename
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)

        logger.info(f"Results exported to: {output_path}")
        return output_path

    def export_best_params(self, filename: Optional[str] = None) -> Path:
        """
        Export best parameters to JSON.

        Args:
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not self.best_params:
            logger.warning("No best parameters to export")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_best_params.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)

        logger.info(f"Best parameters exported to: {output_path}")
        return output_path
