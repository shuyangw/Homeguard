"""
Grid search parameter optimization for backtesting strategies.

Exhaustively tests all parameter combinations to find optimal values
based on specified metrics (Sharpe ratio, total return, max drawdown).
"""

import pandas as pd
from itertools import product
from typing import Dict, List, Any, Union, Optional, Tuple, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from utils import logger

if TYPE_CHECKING:
    from backtesting.engine.backtest_engine import BacktestEngine
    from backtesting.engine.portfolio_simulator import Portfolio


@dataclass
class _EngineConfig:
    """Configuration for backtest engine (pickleable for multiprocessing)."""
    initial_capital: float
    fees: float
    slippage: float
    freq: str
    market_hours_only: bool
    # Risk config as dict for pickling
    risk_config_dict: Optional[Dict[str, Any]]
    enable_regime_analysis: bool


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
    from backtesting.base.strategy import MultiSymbolStrategy
    from backtesting.base.pairs_strategy import PairsStrategy
    from backtesting.engine.pairs_portfolio import PairsPortfolio

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

        # Detect strategy type and route to appropriate execution
        if isinstance(strategy, MultiSymbolStrategy):
            # Multi-symbol strategy execution
            if isinstance(strategy, PairsStrategy) and len(symbols) == 2:
                # PairsStrategy - use PairsPortfolio for synchronized execution
                data_dict = {}

                for symbol in symbols:
                    symbol_data = data.xs(symbol, level='symbol')
                    data_dict[symbol] = symbol_data

                # Generate signals once with complete data_dict
                signals_dict = strategy.generate_signals_multi(data_dict)

                symbol1, symbol2 = symbols
                long_entries1, long_exits1, short_entries1, short_exits1 = signals_dict[symbol1]

                portfolio = PairsPortfolio(
                    symbols=(symbol1, symbol2),
                    prices1=data_dict[symbol1][price_type],
                    prices2=data_dict[symbol2][price_type],
                    entries=short_entries1,
                    exits=short_exits1,
                    short_entries=long_entries1,
                    short_exits=long_exits1,
                    init_cash=engine_config.initial_capital,
                    fees=engine_config.fees,
                    slippage=engine_config.slippage,
                    freq=engine_config.freq,
                    market_hours_only=engine_config.market_hours_only,
                    risk_config=risk_config
                )
            else:
                # Other multi-symbol strategies - use first symbol fallback
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
            # Single-symbol strategy
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
                # Multi-symbol with single-symbol strategy - use first symbol
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
        self._prepare_engine_config()

    def _prepare_engine_config(self) -> _EngineConfig:
        """Prepare pickleable engine configuration for multiprocessing."""
        # Convert risk config to dict for pickling
        risk_dict = None
        if self.engine.risk_config:
            risk_dict = {
                'position_sizing_method': self.engine.risk_config.position_sizing_method,
                'position_size_pct': self.engine.risk_config.position_size_pct,
                'use_stop_loss': self.engine.risk_config.use_stop_loss,
                'stop_loss_type': self.engine.risk_config.stop_loss_type,
                'stop_loss_pct': self.engine.risk_config.stop_loss_pct,
                'atr_multiplier': self.engine.risk_config.atr_multiplier,
                'atr_lookback': self.engine.risk_config.atr_lookback,
                'max_holding_bars': self.engine.risk_config.max_holding_bars,
                'take_profit_pct': self.engine.risk_config.take_profit_pct,
                'max_positions': self.engine.risk_config.max_positions,
                'max_single_position_pct': self.engine.risk_config.max_single_position_pct,
                'max_portfolio_heat': self.engine.risk_config.max_portfolio_heat,
                'risk_per_trade_pct': self.engine.risk_config.risk_per_trade_pct,
                'kelly_win_rate': self.engine.risk_config.kelly_win_rate,
            }

        self._engine_config = _EngineConfig(
            initial_capital=self.engine.initial_capital,
            fees=self.engine.fees,
            slippage=self.engine.slippage,
            freq=self.engine.freq,
            market_hours_only=self.engine.market_hours_only,
            risk_config_dict=risk_dict,
            enable_regime_analysis=self.engine.enable_regime_analysis
        )
        return self._engine_config

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
                # Detect if this is a multi-symbol strategy
                from backtesting.base.strategy import MultiSymbolStrategy

                if isinstance(strategy, MultiSymbolStrategy):
                    # Use proper multi-symbol execution
                    portfolio = self.engine._run_multi_symbol_strategy(strategy, data, symbols, 'close')
                elif len(symbols) == 1:
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

    def optimize_parallel(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        max_workers: Optional[int] = None,
        price_type: str = 'close',
        export_results: bool = True,
        output_dir: Optional[Any] = None,
        use_cache: bool = True,
        cache_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters over a grid using parallel processing.

        This method runs multiple parameter combinations in parallel using
        multiprocessing, providing significant speedup for large parameter grids.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary mapping parameter names to lists of values
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            max_workers: Maximum number of parallel workers (default: min(4, cpu_count))
            price_type: Price column to use ('close', 'open', etc.')
            export_results: If True, export all results to CSV (default: True) [Phase 2]
            output_dir: Optional custom output directory for CSV export [Phase 2]
            use_cache: If True, use result cache (default: True) [Phase 3]
            cache_config: Optional cache configuration [Phase 3]

        Returns:
            Dictionary with best parameters and results:
            {
                'best_params': Dict[str, Any],
                'best_value': float,
                'best_portfolio': Portfolio,
                'metric': str,
                'all_results': List[Dict],  # All tested combinations
                'total_time': float,  # Total optimization time in seconds [Phase 2]
                'avg_time_per_test': float,  # Average time per test in seconds [Phase 2]
                'cache_hits': int,  # Number of cache hits [Phase 3]
                'cache_misses': int  # Number of cache misses [Phase 3]
            }

        Raises:
            ValueError: If unknown metric is specified

        Note:
            For small parameter grids (< 10 combinations), sequential optimization
            may be faster due to multiprocessing overhead.

        Phase 2 Features:
            - Enhanced progress tracking with ETA
            - Automatic CSV export of all results
            - Parameter sensitivity analysis
            - Detailed timing statistics

        Phase 3 Features:
            - Smart result caching (memory + disk)
            - Automatic cache invalidation (TTL-based)
            - Major speedup for walk-forward validation
        """
        import os

        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate metric before starting optimization
        valid_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        # Load data for the optimization period (once, shared across workers)
        data = self.engine.data_loader.load_symbols(symbols, start_date, end_date)

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combos = list(product(*param_values))

        # Log optimization header
        logger.blank()
        logger.separator()
        logger.header(f"Optimizing {strategy_class.__name__} (PARALLEL)")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Total combinations: {len(param_combos)}")
        logger.info(f"Workers: {max_workers}")
        logger.separator()
        logger.blank()

        # Use sequential for very small grids (overhead not worth it)
        if len(param_combos) < 10:
            logger.info("Small grid detected - using sequential optimization")
            return self.optimize(
                strategy_class=strategy_class,
                param_grid=param_grid,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                metric=metric
            )

        # Initialize tracking
        best_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = None
        best_portfolio = None
        all_results = []
        completed_count = 0

        # Phase 2: Enhanced progress tracking
        import time
        start_time = time.time()
        test_times = []  # Track time per test for ETA calculation

        # Phase 3: Initialize cache if enabled
        cache = None
        cache_hits = 0
        cache_misses = 0
        if use_cache:
            from backtesting.optimization.result_cache import ResultCache, CacheConfig
            cache = ResultCache(cache_config)
            logger.info("Result cache enabled")

        # Phase 3: Check cache and determine which jobs to run
        jobs_to_run = []  # (param_combo, cache_key) tuples
        for param_combo in param_combos:
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
                            from backtesting.base.strategy import MultiSymbolStrategy
                            best_strategy = strategy_class(**best_params)
                            if isinstance(best_strategy, MultiSymbolStrategy):
                                best_portfolio = self.engine._run_multi_symbol_strategy(
                                    best_strategy, data, symbols, price_type
                                )
                            elif len(symbols) == 1:
                                best_portfolio = self.engine._run_single_symbol(
                                    best_strategy, data, symbols[0], price_type
                                )
                            else:
                                best_portfolio = self.engine._run_multiple_symbols(
                                    best_strategy, data, symbols, price_type
                                )

                    logger.metric(
                        f"[{completed_count}/{len(param_combos)} | {(completed_count/len(param_combos)*100):.1f}%] "
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
            logger.info(f"Cache hits: {cache_hits}/{len(param_combos)} ({cache_hits/len(param_combos)*100:.1f}%)")
            logger.info(f"Jobs to run: {len(jobs_to_run)}")
            logger.blank()

        # Parallel execution (only for cache misses or if cache disabled)
        if jobs_to_run:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs
                future_to_info = {}  # future -> (param_combo, cache_key)
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

                    # Phase 3: Store in cache if enabled
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

                            # Re-run best to get portfolio object (stats don't include portfolio)
                            # This is necessary because we can't pickle Portfolio objects
                            from backtesting.base.strategy import MultiSymbolStrategy
                            best_strategy = strategy_class(**best_params)
                            if isinstance(best_strategy, MultiSymbolStrategy):
                                best_portfolio = self.engine._run_multi_symbol_strategy(
                                    best_strategy, data, symbols, price_type
                                )
                            elif len(symbols) == 1:
                                best_portfolio = self.engine._run_single_symbol(
                                    best_strategy, data, symbols[0], price_type
                                )
                            else:
                                best_portfolio = self.engine._run_multiple_symbols(
                                    best_strategy, data, symbols, price_type
                                )

                        # Phase 2: Enhanced progress logging with ETA
                        avg_time = sum(test_times) / len(test_times) if test_times else 0
                        remaining = len(jobs_to_run) - len(test_times)  # Remaining jobs to complete
                        eta_seconds = remaining * avg_time / max_workers if max_workers > 0 else 0
                        eta_mins = eta_seconds / 60

                        progress_pct = (completed_count / len(param_combos)) * 100

                        logger.metric(
                            f"[{completed_count}/{len(param_combos)} | {progress_pct:.1f}%] "
                            f"Params: {result['params']} -> {metric}: {result['value']:.4f} "
                            f"(Best: {best_value:.4f}) [ETA: {eta_mins:.1f}m]"
                        )
                    elif result['error']:
                        # Calculate ETA even for errors
                        avg_time = sum(test_times) / len(test_times) if test_times else 0
                        remaining = len(jobs_to_run) - len(test_times)
                        eta_seconds = remaining * avg_time / max_workers if max_workers > 0 else 0
                        eta_mins = eta_seconds / 60
                        progress_pct = (completed_count / len(param_combos)) * 100

                        logger.warning(
                            f"[{completed_count}/{len(param_combos)} | {progress_pct:.1f}%] "
                            f"Skipping invalid combination {result['params']}: {result['error']} "
                            f"[ETA: {eta_mins:.1f}m]"
                        )

        # Phase 2: Calculate final statistics
        total_time = time.time() - start_time
        total_mins = total_time / 60

        # Log best results
        logger.blank()
        logger.separator()
        logger.success(f"Best parameters: {best_params}")
        logger.profit(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Tested {len(param_combos)} combinations using {max_workers} workers")

        # Phase 3: Cache statistics
        if cache:
            logger.info(f"Cache hits: {cache_hits} ({cache_hits/len(param_combos)*100:.1f}%)")
            logger.info(f"Cache misses: {cache_misses} ({cache_misses/len(param_combos)*100:.1f}%)")
            logger.info(f"Tests executed: {len(jobs_to_run)}/{len(param_combos)}")

        logger.info(f"Total time: {total_mins:.2f} minutes ({total_time:.1f}s)")
        logger.info(f"Average time per test: {total_time/len(param_combos):.2f}s")
        logger.separator()
        logger.blank()

        # Phase 2: Export results to CSV if requested
        if export_results and all_results and best_params is not None:
            self._export_results_to_csv(
                all_results=all_results,
                best_params=best_params,
                best_value=best_value,
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
            'all_results': all_results,  # Phase 1: full results for analysis
            'total_time': total_time,  # Phase 2: timing info
            'avg_time_per_test': total_time / len(param_combos) if param_combos else 0,
            'cache_hits': cache_hits if cache else 0,  # Phase 3: cache statistics
            'cache_misses': cache_misses if cache else 0
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

    def _export_results_to_csv(
        self,
        all_results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_value: float,
        metric: str,
        strategy_name: str,
        symbols: List[str],
        output_dir: Optional[Any] = None
    ) -> None:
        """
        Export optimization results to CSV file (Phase 2).

        Args:
            all_results: List of all tested parameter combinations
            best_params: Best parameters found
            best_value: Best metric value
            metric: Optimization metric used
            strategy_name: Name of strategy class
            symbols: List of symbols tested
            output_dir: Optional output directory (uses default if None)
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
        symbols_str = '_'.join(symbols[:3])  # First 3 symbols
        if len(symbols) > 3:
            symbols_str += f'_and_{len(symbols)-3}_more'

        dir_name = f"{timestamp}_{strategy_name}_{symbols_str}_optimization"
        output_path = base_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare CSV data
        csv_rows = []
        for result in all_results:
            row = {
                'params': str(result['params']),
                metric: result['value'],
                'error': result['error'] if result['error'] else '',
                'is_best': result['params'] == best_params,
                'distance_from_best': abs(result['value'] - best_value) if result['value'] != float('-inf') and result['value'] != float('inf') else None
            }

            # Add individual parameter columns
            if result['params']:
                for param_name, param_value in result['params'].items():
                    row[f'param_{param_name}'] = param_value

            csv_rows.append(row)

        # Convert to DataFrame and sort by metric
        import pandas as pd
        df = pd.DataFrame(csv_rows)

        # Sort by metric value (descending for sharpe/return, ascending for drawdown)
        ascending = (metric == 'max_drawdown')
        df = df.sort_values(by=metric, ascending=ascending)

        # Export to CSV
        csv_path = output_path / 'optimization_results.csv'
        df.to_csv(csv_path, index=False)

        logger.blank()
        logger.success(f"Optimization results exported to: {csv_path}")
        logger.info(f"Total combinations tested: {len(csv_rows)}")
        logger.info(f"Valid results: {len([r for r in all_results if not r['error']])}")
        logger.info(f"Invalid combinations: {len([r for r in all_results if r['error']])}")

        # Phase 2: Export parameter sensitivity analysis
        sensitivity_path = output_path / 'parameter_sensitivity.csv'
        self._export_sensitivity_analysis(df, metric, sensitivity_path)
        logger.info(f"Parameter sensitivity analysis: {sensitivity_path}")

        logger.blank()

    def _export_sensitivity_analysis(
        self,
        results_df: pd.DataFrame,
        metric: str,
        output_path: Any
    ) -> None:
        """
        Analyze and export parameter sensitivity (Phase 2).

        Shows how each parameter affects the optimization metric.

        Args:
            results_df: DataFrame with all optimization results
            metric: Optimization metric column name
            output_path: Path to save sensitivity analysis CSV
        """
        import pandas as pd
        import numpy as np

        # Find parameter columns (start with 'param_')
        param_cols = [col for col in results_df.columns if col.startswith('param_')]

        if not param_cols:
            return

        sensitivity_rows = []

        for param_col in param_cols:
            param_name = param_col.replace('param_', '')

            # Filter out rows with errors
            valid_df = results_df[results_df['error'] == ''].copy()

            if len(valid_df) == 0:
                continue

            # Group by parameter value and calculate statistics
            grouped = valid_df.groupby(param_col)[metric].agg(['mean', 'std', 'min', 'max', 'count'])

            # Calculate overall impact (range of means)
            impact = grouped['mean'].max() - grouped['mean'].min()

            # Calculate correlation if numeric
            try:
                param_values = valid_df[param_col].astype(float)
                metric_values = valid_df[metric].astype(float)
                correlation = param_values.corr(metric_values)
            except (ValueError, TypeError):
                correlation = np.nan

            sensitivity_rows.append({
                'parameter': param_name,
                'impact_range': impact,
                'correlation': correlation,
                'unique_values': len(grouped),
                'best_value': grouped['mean'].idxmax(),
                'best_avg_score': grouped['mean'].max(),
                'worst_value': grouped['mean'].idxmin(),
                'worst_avg_score': grouped['mean'].min()
            })

        # Create sensitivity DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_rows)

        # Sort by impact (most impactful first)
        sensitivity_df = sensitivity_df.sort_values('impact_range', ascending=False)

        # Export
        sensitivity_df.to_csv(output_path, index=False)

        # Log top insights
        if len(sensitivity_df) > 0:
            logger.blank()
            logger.header("PARAMETER SENSITIVITY ANALYSIS")
            logger.info("Parameters ranked by impact on performance:")
            for rank, (_, row) in enumerate(sensitivity_df.head(3).iterrows(), 1):
                logger.metric(
                    f"  {rank}. {row['parameter']}: "
                    f"Impact range = {row['impact_range']:.4f}, "
                    f"Best value = {row['best_value']}"
                )
            logger.blank()
