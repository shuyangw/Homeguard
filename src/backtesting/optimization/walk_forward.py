"""
Walk-Forward Validation Framework

Implements rolling window optimization to test parameter stability and prevent overfitting.

Walk-forward analysis splits data into multiple train/test windows:
- Train window: Optimize strategy parameters
- Test window: Validate parameters on unseen data
- Roll forward and repeat

This reveals:
1. Parameter stability (do same params work across periods?)
2. Overfitting risk (train Sharpe >> test Sharpe = overfitted)
3. Performance degradation (how much worse on out-of-sample data?)

Example:
    optimizer = WalkForwardOptimizer(engine)
    results = optimizer.analyze(
        strategy_class=MovingAverageCrossover,
        param_grid={'fast': [10, 20, 30], 'slow': [50, 100, 150]},
        symbols='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        train_months=12,
        test_months=6,
        step_months=6
    )

This creates rolling windows:
- Window 1: Train 2020-01 to 2020-12, Test 2021-01 to 2021-06
- Window 2: Train 2020-07 to 2021-06, Test 2021-07 to 2021-12
- Window 3: Train 2021-01 to 2021-12, Test 2022-01 to 2022-06
- ... etc
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from pathlib import Path

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.optimization.base_optimizer import BaseOptimizer
from src.utils import logger


class WalkForwardOptimizer:
    """
    Walk-forward optimization for parameter stability analysis.

    Splits data into rolling train/test windows and optimizes on train,
    validates on test. Measures parameter stability and overfitting risk.
    """

    def __init__(self, engine: BacktestEngine, base_optimizer: BaseOptimizer):
        """
        Initialize walk-forward optimizer.

        Args:
            engine: BacktestEngine instance
            base_optimizer: Underlying optimizer to use (Grid, Random, Bayesian, etc.)
        """
        self.engine = engine
        self.base_optimizer = base_optimizer

    def analyze(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],  # param_grid for Grid, param_ranges for others
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        train_months: int = 12,
        test_months: int = 6,
        step_months: int = 6,
        metric: str = 'sharpe_ratio',
        export_results: bool = True,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis with rolling windows.

        Args:
            strategy_class: Strategy to optimize
            param_space: Parameter grid/ranges/space for optimizer
            symbols: Symbol(s) to test
            start_date: Start of entire period (YYYY-MM-DD)
            end_date: End of entire period (YYYY-MM-DD)
            train_months: Training window size in months
            test_months: Test window size in months
            step_months: Step size for rolling window (overlap if < test_months)
            metric: Optimization metric
            export_results: Export detailed results to CSV
            **optimizer_kwargs: Additional args for underlying optimizer

        Returns:
            Dictionary with analysis results:
                - windows: List of train/test windows
                - train_results: Optimization results for each window
                - test_results: Validation results for each window
                - degradation: Sharpe degradation (train - test)
                - avg_degradation: Average degradation across windows
                - best_stable_params: Parameters with lowest degradation
                - summary: Performance summary
        """
        logger.blank()
        logger.separator()
        logger.header("WALK-FORWARD VALIDATION")
        logger.separator()
        logger.blank()

        # Generate rolling windows
        windows = self._generate_windows(
            start_date, end_date, train_months, test_months, step_months
        )

        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Total Period: {start_date} to {end_date}")
        logger.info(f"Train window: {train_months} months")
        logger.info(f"Test window: {test_months} months")
        logger.info(f"Step size: {step_months} months")
        logger.info(f"Total windows: {len(windows)}")
        logger.blank()

        # Run optimization on each window
        train_results = []
        test_results = []
        all_params = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            logger.separator()
            logger.header(f"WINDOW {i}/{len(windows)}")
            logger.separator()
            logger.info(f"Train: {train_start} to {train_end}")
            logger.info(f"Test: {test_start} to {test_end}")
            logger.blank()

            # Optimize on training window
            logger.info("Optimizing on training period...")
            try:
                train_result = self._optimize_on_window(
                    strategy_class=strategy_class,
                    param_space=param_space,
                    symbols=symbols,
                    start_date=train_start,
                    end_date=train_end,
                    metric=metric,
                    **optimizer_kwargs
                )

                train_results.append({
                    'window': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'best_params': train_result['best_params'],
                    'best_value': train_result['best_value'],
                    'metric': metric
                })

                logger.success(f"Train {metric}: {train_result['best_value']:.4f}")
                logger.metric(f"Best parameters: {train_result['best_params']}")
                logger.blank()

                # Validate on test window
                logger.info("Validating on test period...")
                test_result = self._validate_on_window(
                    strategy_class=strategy_class,
                    params=train_result['best_params'],
                    symbols=symbols,
                    start_date=test_start,
                    end_date=test_end,
                    metric=metric
                )

                test_results.append({
                    'window': i,
                    'test_start': test_start,
                    'test_end': test_end,
                    'params': train_result['best_params'],
                    'test_value': test_result['value'],
                    'metric': metric
                })

                degradation = train_result['best_value'] - test_result['value']

                logger.success(f"Test {metric}: {test_result['value']:.4f}")
                if degradation > 0.5:
                    logger.error(f"Degradation: {degradation:.4f} ⚠️  HIGH (overfitting risk)")
                elif degradation > 0.3:
                    logger.warning(f"Degradation: {degradation:.4f} ⚠️  MODERATE")
                else:
                    logger.profit(f"Degradation: {degradation:.4f} ✅ LOW (robust)")

                all_params.append(train_result['best_params'])

            except Exception as e:
                logger.error(f"Window {i} failed: {e}")
                train_results.append({
                    'window': i,
                    'error': str(e)
                })
                test_results.append({
                    'window': i,
                    'error': str(e)
                })

            logger.blank()

        # Analyze results
        analysis = self._analyze_results(
            windows, train_results, test_results, all_params, metric
        )

        # Export if requested
        if export_results:
            self._export_results(
                strategy_class, symbols, windows, train_results, test_results, analysis
            )

        # Print summary
        self._print_summary(analysis, metric)

        return analysis

    def _generate_windows(
        self,
        start_date: str,
        end_date: str,
        train_months: int,
        test_months: int,
        step_months: int
    ) -> List[Tuple[str, str, str, str]]:
        """
        Generate rolling train/test windows.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        windows = []
        current_start = start

        while True:
            train_end = current_start + relativedelta(months=train_months)
            test_start = train_end + relativedelta(days=1)
            test_end = test_start + relativedelta(months=test_months)

            # Check if window fits in data range
            if test_end > end:
                break

            windows.append((
                current_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))

            # Step forward
            current_start += relativedelta(months=step_months)

        return windows

    def _optimize_on_window(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run optimization on training window."""
        return self.base_optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_space if hasattr(self.base_optimizer, 'optimize_parallel') else None,
            param_ranges=param_space if not hasattr(self.base_optimizer, 'optimize_parallel') else None,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            export_results=False,  # Don't export intermediate results
            **kwargs
        )

    def _validate_on_window(
        self,
        strategy_class: type,
        params: Dict[str, Any],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str
    ) -> Dict[str, Any]:
        """Validate parameters on test window."""
        # Create strategy with optimized parameters
        strategy = strategy_class(**params)

        # Run backtest
        result = self.engine.run(
            strategy=strategy,
            symbols=symbols if isinstance(symbols, list) else [symbols],
            start_date=start_date,
            end_date=end_date,
            portfolio_mode='single'  # Single symbol sweep
        )

        # Extract metric
        stats = result.stats()
        metric_value = self._extract_metric(stats, metric)

        return {
            'value': metric_value,
            'params': params
        }

    def _extract_metric(self, stats: Dict[str, Any], metric: str) -> float:
        """Extract metric value from backtest stats."""
        metric_mapping = {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': 'Total Return [%]',
            'annual_return': 'Annual Return [%]',
            'max_drawdown': 'Max Drawdown [%]',
            'win_rate': 'Win Rate [%]'
        }

        key = metric_mapping.get(metric, metric)
        return stats.get(key, 0.0)

    def _analyze_results(
        self,
        windows: List[Tuple[str, str, str, str]],
        train_results: List[Dict],
        test_results: List[Dict],
        all_params: List[Dict],
        metric: str
    ) -> Dict[str, Any]:
        """Analyze walk-forward results."""
        # Filter out failed windows
        valid_indices = [
            i for i in range(len(train_results))
            if 'error' not in train_results[i] and 'error' not in test_results[i]
        ]

        if not valid_indices:
            return {
                'error': 'All windows failed',
                'valid_windows': 0
            }

        # Calculate degradation for each window
        degradations = []
        for i in valid_indices:
            train_val = train_results[i]['best_value']
            test_val = test_results[i]['test_value']
            degradations.append(train_val - test_val)

        avg_degradation = np.mean(degradations)
        std_degradation = np.std(degradations)

        # Find most stable parameters (lowest degradation)
        if degradations:
            best_window_idx = valid_indices[np.argmin(degradations)]
            best_stable_params = train_results[best_window_idx]['best_params']
            min_degradation = degradations[np.argmin(degradations)]
        else:
            best_stable_params = None
            min_degradation = None

        # Parameter stability (how often do params repeat?)
        if all_params:
            param_stability = self._measure_parameter_stability(all_params)
        else:
            param_stability = {}

        return {
            'windows': windows,
            'train_results': train_results,
            'test_results': test_results,
            'degradations': degradations,
            'avg_degradation': avg_degradation,
            'std_degradation': std_degradation,
            'min_degradation': min_degradation,
            'max_degradation': max(degradations) if degradations else None,
            'best_stable_params': best_stable_params,
            'param_stability': param_stability,
            'valid_windows': len(valid_indices),
            'failed_windows': len(train_results) - len(valid_indices),
            'metric': metric
        }

    def _measure_parameter_stability(self, all_params: List[Dict]) -> Dict[str, Any]:
        """Measure how stable parameters are across windows."""
        if not all_params:
            return {}

        # Convert to DataFrame for analysis
        param_df = pd.DataFrame(all_params)

        stability = {}
        for col in param_df.columns:
            if pd.api.types.is_numeric_dtype(param_df[col]):
                stability[col] = {
                    'mean': param_df[col].mean(),
                    'std': param_df[col].std(),
                    'min': param_df[col].min(),
                    'max': param_df[col].max(),
                    'coefficient_of_variation': param_df[col].std() / param_df[col].mean() if param_df[col].mean() != 0 else np.inf
                }
            else:
                # Categorical parameter
                value_counts = param_df[col].value_counts()
                stability[col] = {
                    'most_common': value_counts.index[0],
                    'frequency': value_counts.iloc[0] / len(param_df),
                    'unique_values': len(value_counts)
                }

        return stability

    def _export_results(
        self,
        strategy_class: type,
        symbols: Union[str, List[str]],
        windows: List[Tuple],
        train_results: List[Dict],
        test_results: List[Dict],
        analysis: Dict[str, Any]
    ):
        """Export walk-forward results to CSV."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_str = symbols if isinstance(symbols, str) else '_'.join(symbols)
        output_dir = Path(f"C:/Users/qwqw1/Dropbox/cs/stonk/logs/{timestamp}_WalkForward_{strategy_class.__name__}_{symbol_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export window results
        results_data = []
        for i, (train, test) in enumerate(zip(train_results, test_results)):
            if 'error' not in train and 'error' not in test:
                row = {
                    'window': i + 1,
                    'train_start': train['train_start'],
                    'train_end': train['train_end'],
                    'test_start': test['test_start'],
                    'test_end': test['test_end'],
                    'train_sharpe': train['best_value'],
                    'test_sharpe': test['test_value'],
                    'degradation': train['best_value'] - test['test_value']
                }
                # Add parameters
                for param, value in train['best_params'].items():
                    row[f'param_{param}'] = value

                results_data.append(row)

        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_dir / 'walk_forward_results.csv', index=False)

        # Export summary
        with open(output_dir / 'walk_forward_summary.txt', 'w') as f:
            f.write("WALK-FORWARD VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Strategy: {strategy_class.__name__}\n")
            f.write(f"Symbols: {symbols}\n")
            f.write(f"Total Windows: {len(windows)}\n")
            f.write(f"Valid Windows: {analysis['valid_windows']}\n")
            f.write(f"Failed Windows: {analysis['failed_windows']}\n\n")
            f.write(f"Average Degradation: {analysis['avg_degradation']:.4f}\n")
            f.write(f"Std Degradation: {analysis['std_degradation']:.4f}\n")
            f.write(f"Min Degradation: {analysis['min_degradation']:.4f}\n")
            f.write(f"Max Degradation: {analysis['max_degradation']:.4f}\n\n")
            f.write("Best Stable Parameters:\n")
            for param, value in analysis['best_stable_params'].items():
                f.write(f"  {param}: {value}\n")

        logger.info(f"Results exported to: {output_dir}")

    def _print_summary(self, analysis: Dict[str, Any], metric: str):
        """Print walk-forward analysis summary."""
        logger.blank()
        logger.separator()
        logger.header("WALK-FORWARD ANALYSIS SUMMARY")
        logger.separator()
        logger.blank()

        if 'error' in analysis:
            logger.error(f"Analysis failed: {analysis['error']}")
            return

        logger.info(f"Valid Windows: {analysis['valid_windows']}")
        logger.info(f"Failed Windows: {analysis['failed_windows']}")
        logger.blank()

        logger.header("PERFORMANCE DEGRADATION")
        logger.blank()
        logger.metric(f"Average: {analysis['avg_degradation']:.4f}")
        logger.metric(f"Std Dev: {analysis['std_degradation']:.4f}")
        logger.metric(f"Minimum: {analysis['min_degradation']:.4f}")
        logger.metric(f"Maximum: {analysis['max_degradation']:.4f}")
        logger.blank()

        # Interpretation
        avg_deg = analysis['avg_degradation']
        if avg_deg < 0.3:
            logger.profit("✅ EXCELLENT: Parameters are robust across time periods")
        elif avg_deg < 0.5:
            logger.success("✅ GOOD: Acceptable degradation")
        elif avg_deg < 0.8:
            logger.warning("⚠️  CONCERNING: Significant overfitting risk")
        else:
            logger.error("❌ SEVERE: Parameters highly overfit to training data")

        logger.blank()

        logger.header("MOST STABLE PARAMETERS")
        logger.blank()
        for param, value in analysis['best_stable_params'].items():
            logger.metric(f"{param}: {value}")

        logger.blank()

        logger.header("PARAMETER STABILITY")
        logger.blank()
        for param, stability in analysis['param_stability'].items():
            if 'mean' in stability:
                logger.metric(f"{param}:")
                logger.info(f"  Mean: {stability['mean']:.2f}")
                logger.info(f"  Std: {stability['std']:.2f}")
                logger.info(f"  Range: [{stability['min']:.2f}, {stability['max']:.2f}]")
                logger.info(f"  CV: {stability['coefficient_of_variation']:.2f}")
            else:
                logger.metric(f"{param}: {stability['most_common']} ({stability['frequency']*100:.1f}%)")

        logger.blank()
        logger.separator()
