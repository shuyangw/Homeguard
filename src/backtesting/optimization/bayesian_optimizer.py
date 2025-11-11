"""
Bayesian optimization using Gaussian Processes for parameter tuning.

Uses scikit-optimize library for intelligent parameter selection based on
previous results. Much more efficient than grid/random search for expensive
objective functions.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path

from utils import logger
from backtesting.optimization.base_optimizer import BaseOptimizer

# Conditional import of scikit-optimize
try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real, Integer, Categorical
    from skopt import dump, load
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    SkoptOptimizer = None
    Real = None
    Integer = None
    Categorical = None


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Gaussian Processes.

    Uses scikit-optimize (skopt) for intelligent parameter selection.
    Builds a surrogate model (Gaussian Process) to predict performance
    and uses acquisition functions to select the most promising parameters.

    Features:
    - 5-20x fewer iterations than Random Search
    - Intelligent exploration vs exploitation balance
    - Convergence detection and early stopping
    - Plots for convergence analysis
    - Full cache integration (inherited)

    Example:
        from skopt.space import Integer, Real

        optimizer = BayesianOptimizer(engine)
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_space=[
                Integer(5, 30, name='fast_window'),
                Integer(40, 120, name='slow_window'),
                Real(0.01, 0.10, prior='log-uniform', name='threshold')
            ],
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            n_iterations=50,
            acquisition_func='EI'
        )

    Note:
        Requires scikit-optimize: pip install scikit-optimize
    """

    def __init__(self, engine: 'BacktestEngine'):
        """
        Initialize Bayesian optimizer.

        Args:
            engine: BacktestEngine instance

        Raises:
            ImportError: If scikit-optimize is not installed
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "Bayesian optimization requires scikit-optimize. "
                "Install it with: pip install scikit-optimize"
            )

        super().__init__(engine)
        self._convergence_history = []
        self._best_value_history = []

    def optimize(
        self,
        strategy_class: type,
        param_space: List[Any],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        n_iterations: int = 50,
        n_initial_points: int = 10,
        acquisition_func: str = 'EI',
        max_workers: Optional[int] = None,
        price_type: str = 'close',
        use_cache: bool = True,
        cache_config: Optional[Any] = None,
        export_results: bool = True,
        output_dir: Optional[Any] = None,
        random_seed: Optional[int] = None,
        enable_plots: bool = True,
        convergence_tolerance: Optional[float] = None,
        convergence_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Bayesian optimization.

        Args:
            strategy_class: Strategy class to optimize
            param_space: List of parameter dimensions from skopt.space:
                - Integer(low, high, name='param')
                - Real(low, high, prior='uniform'|'log-uniform', name='param')
                - Categorical(categories, name='param')
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            n_iterations: Total optimization iterations (including initial)
            n_initial_points: Random points before Bayesian selection starts
            acquisition_func: 'EI' (Expected Improvement), 'LCB' (Lower Confidence Bound),
                            'PI' (Probability of Improvement)
            max_workers: Maximum parallel workers (default: min(4, cpu_count))
            price_type: Price column to use ('close', 'open', etc.)
            use_cache: If True, use result cache (default: True)
            cache_config: Optional cache configuration
            export_results: If True, export all results to CSV (default: True)
            output_dir: Optional custom output directory for CSV export
            random_seed: Random seed for reproducibility (default: None)
            enable_plots: If True, generate convergence plots (default: True)
            convergence_tolerance: Early stop if best value doesn't improve by this amount
            convergence_patience: Number of iterations without improvement before early stop

        Returns:
            Dictionary with optimization results:
            {
                'best_params': Dict[str, Any],
                'best_value': float,
                'best_portfolio': Portfolio,
                'metric': str,
                'all_results': List[Dict],
                'convergence_data': Dict,
                'n_iterations': int,
                'early_stopped': bool,
                'total_time': float
            }

        Raises:
            ValueError: If unknown metric or invalid param_space
        """
        import os

        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate metric
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Validate param_space
        if not param_space or not isinstance(param_space, list):
            raise ValueError("param_space must be a non-empty list of skopt.space dimensions")

        # Extract parameter names from param_space
        param_names = [dim.name for dim in param_space]
        if None in param_names:
            raise ValueError("All dimensions in param_space must have a 'name' attribute")

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        # Load data once (shared across iterations)
        logger.info("Loading market data for optimization period...")
        data = self.engine.data_loader.load_symbols(symbols, start_date, end_date)

        # Initialize Bayesian optimizer
        logger.blank()
        logger.separator()
        logger.header(f"Optimizing {strategy_class.__name__} (BAYESIAN OPTIMIZATION)")
        logger.info(f"Parameter space: {[str(dim) for dim in param_space]}")
        logger.info(f"Total iterations: {n_iterations}")
        logger.info(f"Initial random points: {n_initial_points}")
        logger.info(f"Acquisition function: {acquisition_func}")
        logger.info(f"Metric: {metric}")
        logger.separator()
        logger.blank()

        # Map acquisition function names
        acq_func_map = {
            'EI': 'EI',
            'LCB': 'LCB',
            'PI': 'PI',
            'gp_hedge': 'gp_hedge'
        }
        acq_func = acq_func_map.get(acquisition_func, 'EI')

        # Initialize scikit-optimize Optimizer
        bayes_opt = SkoptOptimizer(
            dimensions=param_space,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=random_seed
        )

        # Initialize tracking
        best_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        best_portfolio = None
        all_results = []
        self._convergence_history = []
        self._best_value_history = []

        # Initialize cache if enabled
        cache = None
        cache_hits = 0
        cache_misses = 0
        if use_cache:
            from backtesting.optimization.result_cache import ResultCache
            cache = ResultCache(cache_config)
            logger.info("Result cache enabled")

        # Progress tracking
        start_time = time.time()
        iteration_times = []
        early_stopped = False
        no_improvement_count = 0

        # Bayesian optimization loop
        for iteration in range(n_iterations):
            iteration_start = time.time()

            # Ask optimizer for next point to evaluate
            next_params_list = bayes_opt.ask()
            params = dict(zip(param_names, next_params_list))

            # Check cache first
            cached_result = None
            cache_key = None
            if cache:
                cache_key = cache.generate_cache_key(
                    strategy_class=strategy_class,
                    params=params,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    price_type=price_type,
                    engine_config=self._engine_config,
                    metric=metric
                )
                cached_result = cache.get(cache_key)

            if cached_result:
                # Cache hit!
                cache_hits += 1
                metric_value = cached_result['value']
                stats = cached_result['stats']
                error = cached_result['error']

                logger.metric(
                    f"[{iteration+1}/{n_iterations} | {(iteration+1)/n_iterations*100:.1f}%] "
                    f"CACHED: {params} -> {metric}: {metric_value:.4f}"
                )
            else:
                # Cache miss - run backtest
                cache_misses += 1

                try:
                    # Create strategy instance
                    strategy = strategy_class(**params)

                    # Run backtest
                    if len(symbols) == 1:
                        portfolio = self.engine._run_single_symbol(
                            strategy, data, symbols[0], price_type
                        )
                    else:
                        portfolio = self.engine._run_multiple_symbols(
                            strategy, data, symbols, price_type
                        )

                    # Extract stats
                    stats = portfolio.stats()
                    metric_value = self._extract_metric_value(stats, metric)
                    error = None

                except Exception as e:
                    logger.warning(f"Error testing {params}: {e}")
                    metric_value = float('-inf') if metric != 'max_drawdown' else float('inf')
                    stats = None
                    error = str(e)

                # Store in cache
                if cache and cache_key:
                    cache.put(
                        cache_key=cache_key,
                        params=params,
                        metric_value=metric_value,
                        stats=stats,
                        error=error
                    )

            # Tell optimizer the result (negative for minimization)
            # skopt minimizes, so we negate for maximization
            if metric == 'max_drawdown':
                # Drawdown: minimize (less negative is better), already minimizing
                bayes_opt.tell(next_params_list, -metric_value)
            else:
                # Sharpe/Return: maximize, need to negate for skopt
                bayes_opt.tell(next_params_list, -metric_value)

            # Store result
            result = {
                'params': params,
                'value': metric_value,
                'stats': stats,
                'error': error
            }
            all_results.append(result)

            # Update best if better
            previous_best = best_value
            if error is None and stats is not None:
                if self._is_better(metric_value, best_value, metric):
                    best_value = metric_value
                    best_params = params

                    # Re-run best to get portfolio object
                    best_strategy = strategy_class(**best_params)
                    if len(symbols) == 1:
                        best_portfolio = self.engine._run_single_symbol(
                            best_strategy, data, symbols[0], price_type
                        )
                    else:
                        best_portfolio = self.engine._run_multiple_symbols(
                            best_strategy, data, symbols, price_type
                        )

            # Track convergence
            self._best_value_history.append(best_value)

            # Calculate ETA
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            avg_time = sum(iteration_times) / len(iteration_times)
            remaining = n_iterations - (iteration + 1)
            eta_seconds = remaining * avg_time
            eta_mins = eta_seconds / 60

            # Log progress
            improvement_marker = " (NEW BEST!)" if best_value != previous_best else ""
            phase = "RANDOM" if iteration < n_initial_points else "BAYESIAN"

            if not cached_result:  # Only show ETA for non-cached results
                logger.metric(
                    f"[{iteration+1}/{n_iterations} | {(iteration+1)/n_iterations*100:.1f}%] "
                    f"{phase}: {params} -> {metric}: {metric_value:.4f} "
                    f"(Best: {best_value:.4f}){improvement_marker} [ETA: {eta_mins:.1f}m]"
                )

            # Check for convergence (early stopping)
            if convergence_tolerance is not None and iteration >= n_initial_points:
                if best_value == previous_best:
                    no_improvement_count += 1
                else:
                    improvement = abs(best_value - previous_best)
                    if improvement < convergence_tolerance:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

                if no_improvement_count >= convergence_patience:
                    logger.warning(
                        f"Early stopping: No significant improvement for {convergence_patience} iterations"
                    )
                    early_stopped = True
                    break

        # Calculate final statistics
        total_time = time.time() - start_time
        total_mins = total_time / 60
        actual_iterations = iteration + 1

        # Extract convergence data
        convergence_data = {
            'iterations': list(range(actual_iterations)),
            'best_values': self._best_value_history,
            'early_stopped': early_stopped,
            'convergence_iteration': actual_iterations
        }

        # Log best results
        logger.blank()
        logger.separator()
        logger.success(f"Best parameters: {best_params}")
        logger.profit(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Iterations: {actual_iterations}/{n_iterations}")

        if early_stopped:
            logger.info(f"Early stopped at iteration {actual_iterations}")

        # Cache statistics
        if cache:
            total_requests = cache_hits + cache_misses
            logger.info(f"Cache hits: {cache_hits}/{total_requests} ({cache_hits/total_requests*100:.1f}%)")
            logger.info(f"Cache misses: {cache_misses}/{total_requests} ({cache_misses/total_requests*100:.1f}%)")
            logger.info(f"Tests executed: {cache_misses}/{n_iterations}")

        logger.info(f"Total time: {total_mins:.2f} minutes ({total_time:.1f}s)")
        logger.info(f"Average time per iteration: {total_time/actual_iterations:.2f}s")
        logger.separator()
        logger.blank()

        # Export results to CSV if requested
        csv_path = None
        if export_results and all_results and best_params is not None:
            csv_path = self._export_results_to_csv(
                all_results=all_results,
                best_params=best_params,
                best_value=best_value,
                metric=metric,
                strategy_name=strategy_class.__name__,
                symbols=symbols,
                output_dir=output_dir,
                method_name="BayesianOptimization",
                convergence_data=convergence_data
            )

        # Export convergence plots if requested
        plots_path = None
        if enable_plots and all_results:
            plots_path = self._export_convergence_plots(
                convergence_data=convergence_data,
                metric=metric,
                strategy_name=strategy_class.__name__,
                symbols=symbols,
                output_dir=output_dir
            )

        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_portfolio': best_portfolio,
            'metric': metric,
            'all_results': all_results,
            'convergence_data': convergence_data,
            'n_iterations': actual_iterations,
            'early_stopped': early_stopped,
            'total_time': total_time,
            'avg_time_per_iteration': total_time / actual_iterations if actual_iterations > 0 else 0,
            'cache_hits': cache_hits if cache else 0,
            'cache_misses': cache_misses if cache else 0,
            'method': 'bayesian_optimization',
            'csv_path': csv_path,
            'plots_path': plots_path
        }

    def _export_results_to_csv(
        self,
        all_results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_value: float,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None,
        method_name: str = "BayesianOptimization",
        convergence_data: Optional[Dict] = None
    ) -> Path:
        """Export optimization results to CSV file."""
        from datetime import datetime
        from config import get_log_output_dir

        # Determine output directory
        if output_dir is None:
            base_dir = get_log_output_dir()
        else:
            base_dir = Path(output_dir)

        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbols_str = '_'.join(symbols)
        dir_name = f"{timestamp}_{strategy_name}_{symbols_str}_{method_name}"
        output_path = base_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data for CSV
        rows = []
        for i, result in enumerate(all_results):
            row = {
                'iteration': i + 1,
                'method': method_name
            }
            # Add individual parameter columns
            for param_name, param_value in result['params'].items():
                row[f'param_{param_name}'] = param_value

            row[metric] = result['value']
            row['error'] = result['error'] if result['error'] else ''
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by metric (best first)
        ascending = (metric == 'max_drawdown')
        df = df.sort_values(by=metric, ascending=ascending)

        # Add distance from best
        df['distance_from_best'] = abs(df[metric] - best_value)

        # Export optimization_results.csv
        csv_path = output_path / 'optimization_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported optimization results to: {csv_path}")

        # Export summary
        summary_path = output_path / 'optimization_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Bayesian Optimization Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Symbols: {', '.join(symbols)}\n")
            f.write(f"Metric: {metric}\n")
            f.write(f"Iterations: {len(all_results)}\n")
            if convergence_data and convergence_data.get('early_stopped'):
                f.write(f"Early stopped: Yes (iteration {convergence_data['convergence_iteration']})\n")
            f.write(f"\nBest Parameters:\n")
            for param_name, param_value in best_params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write(f"\nBest {metric}: {best_value:.4f}\n")

        logger.info(f"Exported optimization summary to: {summary_path}")

        return csv_path

    def _export_convergence_plots(
        self,
        convergence_data: Dict,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None
    ) -> Optional[Path]:
        """Export convergence plot showing best value over iterations."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available - skipping plot generation")
            return None

        from datetime import datetime
        from config import get_log_output_dir

        # Determine output directory
        if output_dir is None:
            base_dir = get_log_output_dir()
        else:
            base_dir = Path(output_dir)

        # Use same directory as CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbols_str = '_'.join(symbols)
        dir_name = f"{timestamp}_{strategy_name}_{symbols_str}_BayesianOptimization"
        output_path = base_dir / dir_name

        # Create convergence plot
        iterations = convergence_data['iterations']
        best_values = convergence_data['best_values']

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_values, 'b-', linewidth=2, label='Best Value')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(f'Best {metric}', fontsize=12)
        plt.title(f'Bayesian Optimization Convergence - {strategy_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        if convergence_data.get('early_stopped'):
            plt.axvline(
                x=convergence_data['convergence_iteration'],
                color='r',
                linestyle='--',
                label='Early Stop'
            )

        plot_path = output_path / 'convergence_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Exported convergence plot to: {plot_path}")

        return plot_path


def is_bayesian_available() -> bool:
    """Check if Bayesian optimization is available (scikit-optimize installed)."""
    return SKOPT_AVAILABLE
