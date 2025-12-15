"""
Sweep runner for executing backtests across multiple symbols and parameters.

Includes:
- SweepRunner: Symbol-based sweep runner using BacktestEngine
- run_generic_parameter_sweep: Generic parameter sweep for any backtest function
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product

from src.backtesting.base.strategy import BaseStrategy
from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.engine.results_aggregator import ResultsAggregator
from src.backtesting.engine.trade_logger import TradeLogger
from src.backtesting.optimization.data_loader import get_default_workers
from src.utils import logger


class SweepRunner:
    """
    Coordinates running backtests across multiple symbols and aggregating results.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        max_workers: int = 4,
        show_progress: bool = True,
        on_symbol_start: Optional[Callable[[str], None]] = None,
        on_symbol_progress: Optional[Callable[[str, str, float], None]] = None,
        on_symbol_complete: Optional[Callable[[str, Portfolio, pd.Series], None]] = None,
        on_symbol_error: Optional[Callable[[str, Exception], None]] = None
    ):
        """
        Initialize sweep runner.

        Args:
            engine: BacktestEngine instance to use for backtests
            max_workers: Maximum number of parallel workers (default: 4)
            show_progress: Show progress during sweep (default: True)
            on_symbol_start: Optional callback when symbol starts (symbol)
            on_symbol_progress: Optional callback for progress (symbol, message, progress)
            on_symbol_complete: Optional callback when complete (symbol, portfolio, stats)
            on_symbol_error: Optional callback on error (symbol, exception)
        """
        self.engine = engine
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.on_symbol_start = on_symbol_start
        self.on_symbol_progress = on_symbol_progress
        self.on_symbol_complete = on_symbol_complete
        self.on_symbol_error = on_symbol_error
        self._portfolios: Dict[str, Portfolio] = {}
        self._cancelled = False

    def run_sweep(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        parallel: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Run backtest for same strategy across multiple symbols.

        Args:
            strategy: Strategy instance to test
            symbols: List of symbols to test
            start_date: Start date
            end_date: End date
            parallel: Run in parallel if True (default: False)

        Returns:
            Dictionary of {symbol: portfolio_stats}
        """
        # Clear previous portfolios and reset cancellation flag
        self._portfolios.clear()
        self._cancelled = False

        logger.blank()
        logger.separator()
        logger.header(f"RUNNING SWEEP: {strategy}")

        # Format symbol list display
        if len(symbols) == 1:
            symbol_display = symbols[0]
        elif len(symbols) == 2:
            symbol_display = f"{symbols[0]}, {symbols[1]}"
        else:
            symbol_display = f"{symbols[0]}, {symbols[1]}, ... {symbols[-1]}"

        logger.info(f"Symbols: {len(symbols)} ({symbol_display})")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        logger.separator()
        logger.blank()

        results = {}

        if parallel:
            results = self._run_parallel(strategy, symbols, start_date, end_date)
        else:
            results = self._run_sequential(strategy, symbols, start_date, end_date)

        successful = sum(1 for v in results.values() if v is not None)
        failed = len(symbols) - successful

        logger.blank()
        logger.separator()
        logger.success(f"Sweep complete: {successful} successful, {failed} failed")
        logger.separator()
        logger.blank()

        return results

    def _run_sequential(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Run backtests sequentially (one at a time).
        """
        results = {}

        for i, symbol in enumerate(symbols, 1):
            # Check for cancellation
            if self._cancelled:
                logger.warning("Sweep cancelled by user")
                break

            # Callback: symbol started
            if self.on_symbol_start:
                self.on_symbol_start(symbol)

            if self.show_progress and not self.on_symbol_start:
                logger.info(f"[{i}/{len(symbols)}] Testing {symbol}...")

            try:
                # Callback: loading data
                if self.on_symbol_progress:
                    self.on_symbol_progress(symbol, "Loading data...", 0.2)

                portfolio = self.engine.run(
                    strategy=strategy,
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date
                )

                # Callback: computing metrics
                if self.on_symbol_progress:
                    self.on_symbol_progress(symbol, "Computing metrics...", 0.8)

                stats = portfolio.stats()
                results[symbol] = stats

                # Store portfolio for GUI access
                self._portfolios[symbol] = portfolio

                # Callback: completed
                if self.on_symbol_complete:
                    self.on_symbol_complete(symbol, portfolio, stats)

                # Only show console progress if not using GUI callbacks
                if self.show_progress and not self.on_symbol_complete:
                    return_pct = stats.get('Total Return [%]', 0) if stats is not None else 0
                    sharpe = stats.get('Sharpe Ratio', 0) if stats is not None else 0

                    if return_pct >= 0:
                        logger.profit(f"  -> Return: {return_pct:.2f}%, Sharpe: {sharpe:.2f}")
                    else:
                        logger.loss(f"  -> Return: {return_pct:.2f}%, Sharpe: {sharpe:.2f}")

            except Exception as e:
                # Callback: error
                if self.on_symbol_error:
                    self.on_symbol_error(symbol, e)

                # Only log to console if not using GUI callbacks
                if not self.on_symbol_error:
                    logger.error(f"  -> Error testing {symbol}: {e}")

                results[symbol] = None

        return results

    def _run_parallel(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Run backtests in parallel (multiple workers).
        """
        results = {}
        completed_count = 0

        def run_single_backtest(symbol: str) -> tuple:
            # Check for cancellation before starting
            if self._cancelled:
                return (symbol, None, None)

            try:
                # Callback: symbol started
                if self.on_symbol_start:
                    self.on_symbol_start(symbol)

                # Callback: loading data
                if self.on_symbol_progress:
                    self.on_symbol_progress(symbol, "Loading data...", 0.2)

                portfolio = self.engine.run(
                    strategy=strategy,
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date
                )

                # Callback: computing metrics
                if self.on_symbol_progress:
                    self.on_symbol_progress(symbol, "Computing metrics...", 0.8)

                stats = portfolio.stats()

                # Callback: completed
                if self.on_symbol_complete:
                    self.on_symbol_complete(symbol, portfolio, stats)

                return (symbol, stats, portfolio)

            except Exception as e:
                # Callback: error
                if self.on_symbol_error:
                    self.on_symbol_error(symbol, e)

                # Only log to console if not using GUI callbacks
                if not self.on_symbol_error:
                    logger.error(f"Error testing {symbol}: {e}")

                return (symbol, None, None)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(run_single_backtest, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol, stats, portfolio = future.result()
                results[symbol] = stats

                # Store portfolio for GUI access
                if portfolio is not None:
                    self._portfolios[symbol] = portfolio

                completed_count += 1

                # Only show console progress if not using GUI callbacks
                if self.show_progress and not self.on_symbol_complete:
                    logger.info(f"[{completed_count}/{len(symbols)}] Completed {symbol}")

        return results

    def run_and_report(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_dir: Optional[Path | str] = None,
        sort_by: str = 'Sharpe Ratio',
        top_n: Optional[int] = None,
        export_csv: bool = True,
        export_html: bool = True,
        parallel: bool = False
    ) -> pd.DataFrame:
        """
        Run sweep and generate reports.

        Args:
            strategy: Strategy instance
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            output_dir: Output directory for reports
            sort_by: Column to sort results by (default: 'Sharpe Ratio')
            top_n: Only show top N results (default: None = show all)
            export_csv: Export results to CSV (default: True)
            export_html: Export results to HTML (default: True)
            parallel: Run in parallel (default: False)

        Returns:
            DataFrame with aggregated results
        """
        results = self.run_sweep(strategy, symbols, start_date, end_date, parallel)

        df = ResultsAggregator.aggregate_results(results, sort_by=sort_by)

        if df.empty:
            logger.error("No results to aggregate")
            return df

        if top_n:
            df_display = df.head(top_n)
            logger.info(f"Showing top {top_n} results (sorted by {sort_by}):")
        else:
            df_display = df
            logger.info(f"All results (sorted by {sort_by}):")

        logger.blank()
        print(df_display.to_string(index=False))
        logger.blank()

        summary = ResultsAggregator.calculate_summary_stats(df)

        # Display comprehensive summary statistics
        ResultsAggregator.display_summary_stats(summary)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = strategy.__class__.__name__

            if export_csv:
                csv_path = output_dir / f"{timestamp}_{strategy_name}_sweep_results.csv"
                ResultsAggregator.export_to_csv(df, csv_path, include_summary=True)

            if export_html:
                html_path = output_dir / f"{timestamp}_{strategy_name}_sweep_results.html"
                title = f"Backtest Sweep: {strategy_name} ({len(symbols)} symbols)"
                # Pass portfolios, data_loader, and date range for advanced metrics and benchmark calculations
                ResultsAggregator.export_to_html(
                    df,
                    html_path,
                    title=title,
                    portfolios=self._portfolios,
                    data_loader=self.engine.data_loader,
                    start_date=start_date,
                    end_date=end_date,
                    include_benchmarks=True
                )

            # Export detailed trade logs and equity curves for each symbol
            if self._portfolios:
                logger.blank()
                logger.info(f"Exporting detailed trade logs and equity curves for {len(self._portfolios)} symbols...")
                trades_dir = output_dir / "trades"
                trades_dir.mkdir(exist_ok=True)

                for symbol, portfolio in self._portfolios.items():
                    logger.info(f"Processing {symbol}...")
                    if portfolio is None:
                        continue

                    symbol_prefix = f"{timestamp}_{symbol}"

                    # Export trades CSV
                    trades_csv = trades_dir / f"{symbol_prefix}_trades.csv"
                    TradeLogger.export_trades_csv(portfolio, trades_csv, symbol=symbol)

                    # Export equity curve (bar-by-bar portfolio value)
                    equity_csv = trades_dir / f"{symbol_prefix}_equity_curve.csv"
                    TradeLogger.export_equity_curve_csv(portfolio, equity_csv, symbol=symbol)

                    # Export comprehensive portfolio state
                    state_csv = trades_dir / f"{symbol_prefix}_portfolio_state.csv"
                    TradeLogger.export_portfolio_state_csv(portfolio, state_csv, symbol=symbol)

                logger.success(f"Trade logs exported to: {trades_dir}")
                logger.blank()

        return df

    def optimize_across_universe(
        self,
        strategy_class: type,
        symbols: List[str],
        param_grid: Dict[str, List[Any]],
        start_date: str,
        end_date: str,
        metric: str = 'median_sharpe',
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters across a universe of symbols.

        This finds parameters that work best ACROSS ALL symbols, not just one.

        Args:
            strategy_class: Strategy class (not instance)
            symbols: List of symbols
            param_grid: Parameter grid for optimization
            start_date: Start date
            end_date: End date
            metric: Aggregation metric ('median_sharpe', 'mean_sharpe', 'median_return', etc.)
            parallel: Run in parallel (default: False)

        Returns:
            Dictionary with best parameters and performance
        """
        from itertools import product

        logger.blank()
        logger.separator()
        logger.header("UNIVERSE-WIDE OPTIMIZATION")
        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Optimization metric: {metric}")
        logger.separator()
        logger.blank()

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations across {len(symbols)} symbols")
        logger.info(f"Total backtests: {len(param_combinations) * len(symbols)}")
        logger.blank()

        best_score = float('-inf')
        best_params = None
        all_results = []

        for i, param_combo in enumerate(param_combinations, 1):
            params_dict = dict(zip(param_names, param_combo))

            logger.info(f"[{i}/{len(param_combinations)}] Testing params: {params_dict}")

            try:
                strategy = strategy_class(**params_dict)

                results = self.run_sweep(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    parallel=parallel
                )

                df = ResultsAggregator.aggregate_results(results)

                if metric == 'median_sharpe':
                    score = df['Sharpe Ratio'].median()
                elif metric == 'mean_sharpe':
                    score = df['Sharpe Ratio'].mean()
                elif metric == 'median_return':
                    score = df['Total Return [%]'].median()
                elif metric == 'mean_return':
                    score = df['Total Return [%]'].mean()
                elif metric == 'win_rate':
                    score = (df['Total Return [%]'] > 0).sum() / len(df) * 100
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                logger.metric(f"  -> {metric}: {score:.4f}")

                all_results.append({
                    'params': params_dict,
                    'score': score,
                    'results': df
                })

                if score > best_score:
                    best_score = score
                    best_params = params_dict
                    logger.success(f"  -> New best! {metric}: {best_score:.4f}")

            except Exception as e:
                logger.error(f"Error with params {params_dict}: {e}")

        logger.blank()
        logger.separator()
        logger.success("OPTIMIZATION COMPLETE")
        logger.metric(f"Best parameters: {best_params}")
        logger.metric(f"Best {metric}: {best_score:.4f}")
        logger.separator()
        logger.blank()

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }

    def get_portfolios(self) -> Dict[str, Portfolio]:
        """
        Get Portfolio objects from last parallel run.

        Returns:
            Dictionary mapping symbol -> Portfolio object

        Note:
            Only available after running parallel backtests.
            Used by GUI to access Portfolio objects for chart generation.
        """
        return self._portfolios.copy()

    def cancel(self):
        """
        Request cancellation of running sweep.

        Sets a flag that will be checked before starting each new symbol.
        Already-running symbols will complete, but no new symbols will start.

        Note:
            This is a cooperative cancellation - the sweep will stop at the
            next safe checkpoint (before starting a new symbol backtest).
        """
        self._cancelled = True
        logger.warning("Cancellation requested - sweep will stop after current symbols complete")


def run_generic_parameter_sweep(
    backtest_fn: Callable[..., Dict[str, float]],
    param_space: Dict[str, List[Any]],
    shared_data: Optional[Dict[str, Any]] = None,
    n_workers: Optional[int] = None,
    metric: str = 'sharpe',
    minimize: bool = False,
    progress_interval: int = 10
) -> Dict[str, Any]:
    """Run parameter sweep on ANY existing backtest function.

    This is a generic function that works with any callable that returns
    a metrics dictionary. It uses ThreadPoolExecutor to run trials in parallel,
    sharing data across threads (no memory duplication).

    Args:
        backtest_fn: Any callable that accepts **kwargs and returns a dict of metrics.
                     Example: {'sharpe': 1.5, 'cagr': 0.20, 'max_dd': -0.15}
        param_space: Parameter names and their possible values to sweep.
                     Example: {'stop_loss': [0.05, 0.10], 'top_n': [10, 20]}
        shared_data: Pre-loaded data to pass to backtest_fn on every call.
                     This data is shared (read-only) across all threads.
        n_workers: Number of worker threads. Default: 80% of CPU cores.
        metric: Which metric to optimize (default: 'sharpe').
        minimize: If True, find minimum instead of maximum (e.g., for max_dd).
        progress_interval: Log progress every N completions.

    Returns:
        Dictionary with:
            'best_params': Dict of best parameter values
            'best_metrics': Dict of metrics for best params
            'all_results': List of {'params': {...}, 'metrics': {...}, 'error': str|None}

    Example:
        >>> from src.backtesting.optimization.sweep_runner import run_generic_parameter_sweep
        >>> from src.backtesting.optimization.data_loader import load_minute_cache_polars
        >>> from backtest_scripts.ramp_minute_parallel import simulate_all_days
        >>>
        >>> # Load data once
        >>> minute_cache = load_minute_cache_polars('cache.parquet')
        >>>
        >>> # Define parameter space
        >>> param_space = {
        ...     'stop_loss_pct': [None, 0.05, 0.10],
        ...     'top_n': [10, 15, 20]
        ... }
        >>>
        >>> # Run sweep
        >>> results = run_generic_parameter_sweep(
        ...     backtest_fn=simulate_all_days,
        ...     param_space=param_space,
        ...     shared_data={'minute_cache': minute_cache},
        ...     metric='sharpe'
        ... )
        >>> print(f"Best params: {results['best_params']}")
        >>> print(f"Best Sharpe: {results['best_metrics']['sharpe']}")

    Note:
        - Uses ThreadPoolExecutor (not ProcessPoolExecutor) to share data
        - NumPy releases GIL, so threading works well for array operations
        - All threads share the same shared_data reference (no copying)
    """
    if n_workers is None:
        n_workers = get_default_workers()

    if shared_data is None:
        shared_data = {}

    # Generate all parameter combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    combinations = list(product(*param_values))

    total_combos = len(combinations)
    logger.info(f"Running {total_combos} parameter combinations with {n_workers} threads")

    results: List[Dict[str, Any]] = []

    def run_trial(params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single parameter combination."""
        try:
            # Merge shared data with trial-specific params
            call_kwargs = {**shared_data, **params}
            metrics = backtest_fn(**call_kwargs)
            return {'params': params, 'metrics': metrics, 'error': None}
        except Exception as e:
            logger.error(f"Trial failed for {params}: {e}")
            return {'params': params, 'metrics': None, 'error': str(e)}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {}
        for combo in combinations:
            params = dict(zip(param_names, combo))
            future = executor.submit(run_trial, params)
            futures[future] = params

        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            if (i + 1) % progress_interval == 0 or (i + 1) == total_combos:
                logger.info(f"Completed {i + 1}/{total_combos} trials")

    # Find best result
    valid_results = [r for r in results if r['metrics'] is not None]

    if not valid_results:
        logger.error("No valid results found in parameter sweep")
        return {
            'best_params': None,
            'best_metrics': None,
            'all_results': results
        }

    def get_metric_value(r: Dict[str, Any]) -> float:
        """Extract metric value with fallback."""
        val = r['metrics'].get(metric)
        if val is None:
            return float('inf') if minimize else float('-inf')
        return val

    if minimize:
        best = min(valid_results, key=get_metric_value)
    else:
        best = max(valid_results, key=get_metric_value)

    logger.success(f"Best {metric}: {best['metrics'].get(metric)}")
    logger.info(f"Best params: {best['params']}")

    return {
        'best_params': best['params'],
        'best_metrics': best['metrics'],
        'all_results': results
    }
