"""
Random search parameter optimization.

Randomly samples parameter combinations from continuous/discrete ranges.
Much faster than grid search for large parameter spaces while still finding
good solutions.
"""

import numpy as np
import pandas as pd
import time
from itertools import product
from typing import Dict, List, Any, Union, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import logger
from src.backtesting.optimization.base_optimizer import BaseOptimizer, _EngineConfig


def _test_single_params(
    param_combo: Tuple[Any, ...],
    param_names: List[str],
    strategy_class: type,
    data: pd.DataFrame,
    symbols: List[str],
    price_type: str,
    engine_config: _EngineConfig,
    metric: str
) -> Dict[str, Any]:
    """
    Test a single parameter combination (standalone function for multiprocessing).

    Args:
        param_combo: Tuple of parameter values
        param_names: List of parameter names
        strategy_class: Strategy class to instantiate
        data: Pre-loaded market data
        symbols: List of symbols
        price_type: Price column to use ('close', 'open', etc.)
        engine_config: Engine configuration
        metric: Metric to optimize

    Returns:
        Dictionary with params, metric value, and stats
    """
    from backtesting.engine.portfolio_simulator import from_signals
    from backtesting.utils.risk_config import RiskConfig

    # Convert tuple to dict
    params = dict(zip(param_names, param_combo))

    try:
        # Create strategy instance
        strategy = strategy_class(**params)

        # Reconstruct risk config from dict
        if engine_config.risk_config_dict:
            risk_config = RiskConfig(**engine_config.risk_config_dict)
        else:
            risk_config = RiskConfig.moderate()

        # Run backtest (simplified version of _run_single_symbol)
        if len(symbols) == 1:
            symbol_data = data.xs(symbols[0], level='symbol')
            entries, exits = strategy.generate_signals(symbol_data)
            price = symbol_data[price_type]

            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)

            portfolio = from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=engine_config.initial_capital,
                fees=engine_config.fees,
                slippage=engine_config.slippage,
                freq=engine_config.freq,
                market_hours_only=engine_config.market_hours_only,
                risk_config=risk_config,
                price_data=symbol_data
            )
        else:
            # Multi-symbol: Use first symbol for now (matches current behavior)
            symbol_data = data.xs(symbols[0], level='symbol')
            entries, exits = strategy.generate_signals(symbol_data)
            price = symbol_data[price_type]

            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)

            portfolio = from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=engine_config.initial_capital,
                fees=engine_config.fees,
                slippage=engine_config.slippage,
                freq=engine_config.freq,
                market_hours_only=engine_config.market_hours_only,
                risk_config=risk_config,
                price_data=symbol_data
            )

        # Get stats
        stats = portfolio.stats()

        if stats is None:
            return {
                'params': params,
                'value': float('-inf') if metric != 'max_drawdown' else float('inf'),
                'stats': None,
                'error': 'No stats generated'
            }

        # Extract metric value
        if metric == 'sharpe_ratio':
            value = float(stats.get('Sharpe Ratio', float('-inf')))
        elif metric == 'total_return':
            value = float(stats.get('Total Return [%]', float('-inf')))
        elif metric == 'max_drawdown':
            value = float(stats.get('Max Drawdown [%]', float('inf')))
        else:
            value = float('-inf')

        return {
            'params': params,
            'value': value,
            'stats': stats,
            'error': None
        }

    except (ValueError, TypeError) as e:
        # Invalid parameter combination
        return {
            'params': params,
            'value': float('-inf') if metric != 'max_drawdown' else float('inf'),
            'stats': None,
            'error': str(e)
        }


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random search optimizer for strategy parameters.

    Randomly samples N parameter combinations from specified ranges.
    Much faster than exhaustive grid search for large parameter spaces.

    Example:
        optimizer = RandomSearchOptimizer(engine)
        result = optimizer.optimize(
            strategy_class=MovingAverageCrossover,
            param_ranges={
                'fast_window': (5, 30),      # Uniform sampling [5, 30]
                'slow_window': (40, 120),    # Uniform sampling [40, 120]
                'threshold': (0.01, 0.10, 'log')  # Log-uniform sampling
            },
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            n_iterations=100  # Test 100 random combinations
        )

    Speedup: Typically 10-100x faster than grid search for large spaces
    Quality: Usually finds 80-95% optimal solution
    """

    def optimize(
        self,
        strategy_class: type,
        param_ranges: Dict[str, Union[Tuple, List]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        n_iterations: int = 100,
        max_workers: Optional[int] = None,
        price_type: str = 'close',
        use_cache: bool = True,
        cache_config: Optional[Any] = None,
        export_results: bool = True,
        output_dir: Optional[Any] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using random search.

        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Dictionary mapping parameter names to ranges
                Format: {
                    'param_name': (min, max),              # Uniform int/float
                    'param_name': (min, max, 'log'),       # Log-uniform
                    'param_name': [value1, value2, ...],   # Discrete choice
                }
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            n_iterations: Number of random samples to test
            max_workers: Maximum number of parallel workers (default: min(4, cpu_count))
            price_type: Price column to use ('close', 'open', etc.')
            use_cache: If True, use result cache (default: True)
            cache_config: Optional cache configuration
            export_results: If True, export all results to CSV (default: True)
            output_dir: Optional custom output directory for CSV export
            random_seed: Random seed for reproducibility (default: None)

        Returns:
            Dictionary with best parameters and results:
            {
                'best_params': Dict[str, Any],
                'best_value': float,
                'best_portfolio': Portfolio,
                'metric': str,
                'all_results': List[Dict],
                'total_time': float,
                'avg_time_per_test': float,
                'cache_hits': int,
                'cache_misses': int
            }

        Raises:
            ValueError: If unknown metric is specified or invalid param_ranges
        """
        import os

        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate metric before starting optimization
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        # Load data for the optimization period (once, shared across workers)
        data = self.engine.data_loader.load_symbols(symbols, start_date, end_date)

        # Generate random parameter samples
        param_samples = self._generate_random_samples(param_ranges, n_iterations)
        param_names = list(param_ranges.keys())

        # Log optimization header
        logger.blank()
        logger.separator()
        logger.header(f"Optimizing {strategy_class.__name__} (RANDOM SEARCH)")
        logger.info(f"Parameter ranges: {param_ranges}")
        logger.info(f"Random samples: {n_iterations}")
        logger.info(f"Workers: {max_workers}")
        logger.separator()
        logger.blank()

        # Initialize tracking
        best_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        best_portfolio = None
        all_results = []
        completed_count = 0

        # Progress tracking
        start_time = time.time()
        test_times = []

        # Initialize cache if enabled
        cache = None
        cache_hits = 0
        cache_misses = 0
        if use_cache:
            from backtesting.optimization.result_cache import ResultCache, CacheConfig
            cache = ResultCache(cache_config)
            logger.info("Result cache enabled")

        # Check cache and determine which jobs to run
        jobs_to_run = []
        for param_combo in param_samples:
            params_dict = dict(zip(param_names, param_combo))

            # Generate cache key
            if cache:
                cache_key = cache.generate_cache_key(
                    strategy_class=strategy_class,
                    params=params_dict,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    price_type=price_type,
                    engine_config=self._engine_config,
                    metric=metric
                )

                # Check cache
                cached_result = cache.get(cache_key)
                if cached_result:
                    # Cache hit!
                    cache_hits += 1
                    all_results.append(cached_result)
                    completed_count += 1

                    # Update best if this is better
                    if cached_result['error'] is None and cached_result['stats'] is not None:
                        if self._is_better(cached_result['value'], best_value, metric):
                            best_value = cached_result['value']
                            best_params = cached_result['params']

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

                    logger.metric(
                        f"[{completed_count}/{n_iterations} | {(completed_count/n_iterations*100):.1f}%] "
                        f"CACHED: {params_dict} -> {metric}: {cached_result['value']:.4f}"
                    )
                    continue
                else:
                    # Cache miss - need to run this
                    cache_misses += 1
                    jobs_to_run.append((param_combo, cache_key))
            else:
                # Cache disabled - run all jobs
                jobs_to_run.append((param_combo, None))

        # Log cache statistics
        if cache:
            logger.info(f"Cache hits: {cache_hits}/{n_iterations} ({cache_hits/n_iterations*100:.1f}%)")
            logger.info(f"Jobs to run: {len(jobs_to_run)}")
            logger.blank()

        # Parallel execution (only for cache misses or if cache disabled)
        if jobs_to_run:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs
                future_to_info = {}
                for param_combo, cache_key in jobs_to_run:
                    future = executor.submit(
                        _test_single_params,
                        param_combo,
                        param_names,
                        strategy_class,
                        data,
                        symbols,
                        price_type,
                        self._engine_config,
                        metric
                    )
                    future_to_info[future] = (param_combo, cache_key)

                # Collect results as they complete
                for future in as_completed(future_to_info):
                    iteration_start = time.time()
                    result = future.result()
                    completed_count += 1
                    param_combo, cache_key = future_to_info[future]

                    # Store all results
                    all_results.append(result)

                    # Store in cache if enabled
                    if cache and cache_key:
                        cache.put(
                            cache_key=cache_key,
                            params=result['params'],
                            metric_value=result['value'],
                            stats=result['stats'],
                            error=result['error']
                        )

                    # Track time for ETA calculation
                    iteration_time = time.time() - iteration_start
                    test_times.append(iteration_time)

                    # Update best if this is better
                    if result['error'] is None and result['stats'] is not None:
                        if self._is_better(result['value'], best_value, metric):
                            best_value = result['value']
                            best_params = result['params']

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

                        # Progress logging with ETA
                        avg_time = sum(test_times) / len(test_times) if test_times else 0
                        remaining = len(jobs_to_run) - len(test_times)
                        eta_seconds = remaining * avg_time / max_workers if max_workers > 0 else 0
                        eta_mins = eta_seconds / 60

                        progress_pct = (completed_count / n_iterations) * 100

                        logger.metric(
                            f"[{completed_count}/{n_iterations} | {progress_pct:.1f}%] "
                            f"Params: {result['params']} -> {metric}: {result['value']:.4f} "
                            f"(Best: {best_value:.4f}) [ETA: {eta_mins:.1f}m]"
                        )
                    elif result['error']:
                        # Calculate ETA even for errors
                        avg_time = sum(test_times) / len(test_times) if test_times else 0
                        remaining = len(jobs_to_run) - len(test_times)
                        eta_seconds = remaining * avg_time / max_workers if max_workers > 0 else 0
                        eta_mins = eta_seconds / 60
                        progress_pct = (completed_count / n_iterations) * 100

                        logger.warning(
                            f"[{completed_count}/{n_iterations} | {progress_pct:.1f}%] "
                            f"Skipping invalid combination {result['params']}: {result['error']} "
                            f"[ETA: {eta_mins:.1f}m]"
                        )

        # Calculate final statistics
        total_time = time.time() - start_time
        total_mins = total_time / 60

        # Log best results
        logger.blank()
        logger.separator()
        logger.success(f"Best parameters: {best_params}")
        logger.profit(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Tested {n_iterations} random samples using {max_workers} workers")

        # Cache statistics
        if cache:
            logger.info(f"Cache hits: {cache_hits} ({cache_hits/n_iterations*100:.1f}%)")
            logger.info(f"Cache misses: {cache_misses} ({cache_misses/n_iterations*100:.1f}%)")
            logger.info(f"Tests executed: {len(jobs_to_run)}/{n_iterations}")

        logger.info(f"Total time: {total_mins:.2f} minutes ({total_time:.1f}s)")
        logger.info(f"Average time per test: {total_time/n_iterations:.2f}s")
        logger.separator()
        logger.blank()

        # Export results to CSV if requested
        if export_results and all_results and best_params is not None:
            self._export_results_to_csv(
                all_results=all_results,
                best_params=best_params,
                best_value=best_value,
                metric=metric,
                strategy_name=strategy_class.__name__,
                symbols=symbols,
                output_dir=output_dir,
                method_name="RandomSearch"
            )

        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_portfolio': best_portfolio,
            'metric': metric,
            'all_results': all_results,
            'total_time': total_time,
            'avg_time_per_test': total_time / n_iterations if n_iterations > 0 else 0,
            'cache_hits': cache_hits if cache else 0,
            'cache_misses': cache_misses if cache else 0,
            'method': 'random_search',
            'n_iterations': n_iterations
        }

    def _generate_random_samples(
        self,
        param_ranges: Dict[str, Union[Tuple, List]],
        n_samples: int
    ) -> List[Tuple]:
        """
        Generate random parameter samples from specified ranges.

        Args:
            param_ranges: Dictionary mapping parameter names to ranges
            n_samples: Number of random samples to generate

        Returns:
            List of parameter tuples (each tuple is one sample)
        """
        param_names = list(param_ranges.keys())
        samples = []

        for _ in range(n_samples):
            sample = []
            for param_name in param_names:
                param_spec = param_ranges[param_name]

                if isinstance(param_spec, list):
                    # Discrete choice
                    value = np.random.choice(param_spec)
                elif isinstance(param_spec, tuple):
                    if len(param_spec) == 2:
                        # Uniform sampling
                        min_val, max_val = param_spec
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            value = np.random.randint(min_val, max_val + 1)
                        else:
                            value = np.random.uniform(min_val, max_val)
                    elif len(param_spec) == 3 and param_spec[2] == 'log':
                        # Log-uniform sampling
                        min_val, max_val, _ = param_spec
                        value = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            value = int(value)
                    else:
                        raise ValueError(f"Invalid param_spec for {param_name}: {param_spec}")
                else:
                    raise ValueError(f"Invalid param_spec for {param_name}: {param_spec}")

                sample.append(value)

            samples.append(tuple(sample))

        return samples

    def _export_results_to_csv(
        self,
        all_results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_value: float,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None,
        method_name: str = "RandomSearch"
    ) -> None:
        """
        Export optimization results to CSV file.

        Args:
            all_results: List of all tested parameter combinations
            best_params: Best parameters found
            best_value: Best metric value
            metric: Optimization metric used
            strategy_name: Name of strategy class
            symbols: List of symbols tested
            output_dir: Optional output directory (uses default if None)
            method_name: Optimization method name for file naming
        """
        from pathlib import Path
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
        for result in all_results:
            row = {'method': method_name}
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
            f.write(f"Optimization Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"Symbols: {', '.join(symbols)}\n")
            f.write(f"Metric: {metric}\n")
            f.write(f"Samples tested: {len(all_results)}\n\n")
            f.write(f"Best Parameters:\n")
            for param_name, param_value in best_params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write(f"\nBest {metric}: {best_value:.4f}\n")

        logger.info(f"Exported optimization summary to: {summary_path}")
