"""
Main interface for running backtests with trading strategies.

Usage:
    python backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01
    python backtest_runner.py --strategy MeanReversion --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.engine.metrics import PerformanceMetrics
from src.backtesting.engine.data_loader import DataLoader
from src.backtesting.optimization.sweep_runner import SweepRunner
from src.backtesting.utils.universe import UniverseManager
from src.strategies.base_strategies.moving_average import MovingAverageCrossover, TripleMovingAverage
from src.strategies.base_strategies.mean_reversion import MeanReversion, RSIMeanReversion
from src.strategies.base_strategies.momentum import MomentumStrategy, BreakoutStrategy
from src.visualization.config import VisualizationConfig
from src.visualization.integration import BacktestVisualizer
from src.config import get_log_output_dir
from src.utils import logger


STRATEGY_REGISTRY = {
    'MovingAverageCrossover': MovingAverageCrossover,
    'TripleMovingAverage': TripleMovingAverage,
    'MeanReversion': MeanReversion,
    'RSIMeanReversion': RSIMeanReversion,
    'MomentumStrategy': MomentumStrategy,
    'BreakoutStrategy': BreakoutStrategy,
}

try:
    from strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
    from strategies.advanced.overnight_mean_reversion import OvernightMeanReversion
    from strategies.advanced.cross_sectional_momentum import CrossSectionalMomentum
    from strategies.advanced.pairs_trading import PairsTrading

    STRATEGY_REGISTRY['VolatilityTargetedMomentum'] = VolatilityTargetedMomentum
    STRATEGY_REGISTRY['OvernightMeanReversion'] = OvernightMeanReversion
    STRATEGY_REGISTRY['CrossSectionalMomentum'] = CrossSectionalMomentum
    STRATEGY_REGISTRY['PairsTrading'] = PairsTrading
except ImportError:
    pass


def parse_strategy_params(params_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse strategy parameters from command line string.

    Args:
        params_str: String like "fast_window=10,slow_window=30"

    Returns:
        Dictionary of parameter names and values
    """
    if not params_str:
        return {}

    params = {}
    for pair in params_str.split(','):
        key, value = pair.split('=')
        key = key.strip()

        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value.strip()

    return params


def run_backtest(
    strategy_name: str,
    symbols: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    fees: float = 0.001,
    strategy_params: Optional[str] = None,
    save_report: bool = False,
    show_plots: bool = False,
    visualize: bool = False,
    quantstats: bool = False,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    verbosity: int = 1
) -> None:
    """
    Run a backtest with specified parameters.

    Args:
        strategy_name: Name of strategy from STRATEGY_REGISTRY
        symbols: Comma-separated list of symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital
        fees: Trading fees as decimal
        strategy_params: Strategy parameters as key=value pairs
        save_report: Save performance report to file
        show_plots: Display performance plots
        visualize: Enable visualization engine (charts, logs, reports)
        quantstats: Generate QuantStats tearsheet report (recommended)
        output_dir: Output directory for reports (auto-generated if not specified)
        run_name: Custom name for this test run (used in output directory, default: strategy_name)
        verbosity: Logging verbosity (0=minimal, 1=normal, 2=detailed, 3=verbose)
    """
    if strategy_name not in STRATEGY_REGISTRY:
        logger.error(f"Unknown strategy '{strategy_name}'")
        logger.info(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    strategy_class = STRATEGY_REGISTRY[strategy_name]

    params = parse_strategy_params(strategy_params)
    strategy = strategy_class(**params)

    symbol_list = [s.strip() for s in symbols.split(',')]

    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees
    )

    # Generate output directory for results (use configured output_dir from settings.ini)
    if output_dir is None and (quantstats or visualize):
        from datetime import datetime
        from src.config import get_backtest_results_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = '_'.join(symbol_list)
        results_dir = get_backtest_results_dir()
        # Use run_name if provided, otherwise fall back to strategy_name
        display_name = run_name if run_name else strategy_name
        output_dir = str(results_dir / f"{timestamp}_{display_name}_{symbols_str}")

    output_path = Path(output_dir) if output_dir else None

    # Run backtest with QuantStats reporting (includes static charts)
    if quantstats and output_path:
        # Generate QuantStats tearsheet
        portfolio = engine.run_and_report(
            strategy=strategy,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_path
        )

        # Also generate static charts in the same directory
        if visualize:
            viz_config = VisualizationConfig.from_args(
                verbosity=verbosity,
                enable_charts=True,
                enable_logs=True
            )

            visualizer = BacktestVisualizer(viz_config)

            loader = DataLoader()
            if len(symbol_list) == 1:
                price_data = loader.load_single_symbol(symbol_list[0], start_date, end_date)
            else:
                price_data = loader.load_symbols(symbol_list, start_date, end_date)

            visualizer.visualize_backtest(
                portfolio=portfolio,
                strategy_name=strategy_name,
                symbols=symbol_list,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                fees=fees,
                price_data=price_data,
                output_dir=output_path
            )
    elif visualize and output_path:
        # Visualization only (without QuantStats)
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date
        )

        viz_config = VisualizationConfig.from_args(
            verbosity=verbosity,
            enable_charts=True,
            enable_logs=True
        )

        visualizer = BacktestVisualizer(viz_config)

        loader = DataLoader()
        if len(symbol_list) == 1:
            price_data = loader.load_single_symbol(symbol_list[0], start_date, end_date)
        else:
            price_data = loader.load_symbols(symbol_list, start_date, end_date)

        visualizer.visualize_backtest(
            portfolio=portfolio,
            strategy_name=strategy_name,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            fees=fees,
            price_data=price_data,
            output_dir=output_path
        )
    else:
        # Standard backtest without visualization
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date
        )

    if save_report:
        report_path = Path(f"backtest_report_{strategy_name}_{symbols.replace(',', '_')}.csv")
        PerformanceMetrics.generate_report(portfolio, report_path)  # type: ignore[arg-type]

    if show_plots:
        PerformanceMetrics.plot_equity_curve(portfolio, title=f"{strategy_name} - Equity Curve")  # type: ignore[arg-type]
        PerformanceMetrics.plot_drawdown(portfolio, title=f"{strategy_name} - Drawdown")  # type: ignore[arg-type]


def optimize_strategy(
    strategy_name: str,
    symbols: str,
    start_date: str,
    end_date: str,
    param_grid_str: str,
    metric: str = 'sharpe_ratio',
    initial_capital: float = 100000.0,
    fees: float = 0.001
) -> None:
    """
    Optimize strategy parameters.

    Args:
        strategy_name: Name of strategy from STRATEGY_REGISTRY
        symbols: Comma-separated list of symbols
        start_date: Start date
        end_date: End date
        param_grid_str: Parameter grid like "fast_window=10,20,30;slow_window=40,50,60"
        metric: Optimization metric
        initial_capital: Starting capital
        fees: Trading fees
    """
    if strategy_name not in STRATEGY_REGISTRY:
        logger.error(f"Unknown strategy '{strategy_name}'")
        sys.exit(1)

    strategy_class = STRATEGY_REGISTRY[strategy_name]

    param_grid = {}
    for param_spec in param_grid_str.split(';'):
        param_name, values_str = param_spec.split('=')
        param_name = param_name.strip()

        values = []
        for v in values_str.split(','):
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v.strip())

        param_grid[param_name] = values

    symbol_list = [s.strip() for s in symbols.split(',')]

    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees
    )

    results = engine.optimize(
        strategy_class=strategy_class,
        param_grid=param_grid,
        symbols=symbol_list,
        start_date=start_date,
        end_date=end_date,
        metric=metric
    )

    logger.blank()
    logger.success("Optimization complete!")
    logger.metric(f"Best parameters: {results['best_params']}")
    logger.metric(f"Best {metric}: {results['best_value']:.4f}")


def sweep_backtest(
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    fees: float = 0.001,
    strategy_params: Optional[str] = None,
    universe: Optional[str] = None,
    symbols_file: Optional[str] = None,
    symbols_list: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    sort_by: str = 'Sharpe Ratio',
    top_n: Optional[int] = None,
    parallel: bool = False,
    max_workers: int = 4,
    verbosity: int = 1
) -> None:
    """
    Run backtest sweep across multiple symbols.

    Args:
        strategy_name: Name of strategy from STRATEGY_REGISTRY
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        fees: Trading fees
        strategy_params: Strategy parameters string
        universe: Predefined universe name (e.g., 'DOW30', 'FAANG')
        symbols_file: Path to file with symbols
        symbols_list: Comma-separated symbol list
        output_dir: Output directory for reports
        run_name: Custom name for this sweep
        sort_by: Column to sort results by
        top_n: Show only top N results
        parallel: Run in parallel
        max_workers: Max parallel workers
        verbosity: Logging verbosity
    """
    if strategy_name not in STRATEGY_REGISTRY:
        logger.error(f"Unknown strategy '{strategy_name}'")
        logger.info(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    strategy_class = STRATEGY_REGISTRY[strategy_name]

    params = parse_strategy_params(strategy_params)
    strategy = strategy_class(**params)

    try:
        if symbols_list:
            symbols = [s.strip() for s in symbols_list.split(',')]
        else:
            symbols = UniverseManager.get_symbols(
                universe=universe,
                symbols_file=symbols_file
            )
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        sys.exit(1)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        fees=fees
    )

    sweep_runner = SweepRunner(
        engine=engine,
        max_workers=max_workers,
        show_progress=(verbosity >= 1)
    )

    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = run_name if run_name else f"{strategy_name}_sweep"
        log_base_dir = get_log_output_dir()
        output_dir = str(log_base_dir / f"{timestamp}_{display_name}")

    sweep_runner.run_and_report(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        sort_by=sort_by,
        top_n=top_n,
        export_csv=True,
        export_html=True,
        parallel=parallel
    )


def main():
    """
    Main command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies using VectorBT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple backtest
  python backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01

  # Backtest with custom parameters
  python backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01 --params "fast_window=10,slow_window=30"

  # Multiple symbols
  python backtest_runner.py --strategy MeanReversion --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01

  # Optimize parameters
  python backtest_runner.py --optimize --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01 --param-grid "fast_window=10,20,30;slow_window=40,50,60"

Available strategies:
  - MovingAverageCrossover
  - TripleMovingAverage
  - MeanReversion
  - RSIMeanReversion
  - MomentumStrategy
  - BreakoutStrategy
        """
    )

    parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--fees', type=float, default=0.001, help='Trading fees as decimal (default: 0.001)')
    parser.add_argument('--params', type=str, help='Strategy parameters (e.g., "fast_window=10,slow_window=30")')
    parser.add_argument('--save-report', action='store_true', help='Save performance report to CSV')
    parser.add_argument('--show-plots', action='store_true', help='Display performance plots')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization engine (charts, logs, reports)')
    parser.add_argument('--quantstats', action='store_true', help='Generate QuantStats tearsheet report (recommended)')
    parser.add_argument('--output-dir', type=str, help='Output directory for reports (auto-generated if not specified)')
    parser.add_argument('--run-name', type=str, help='Custom name for this test run (used in log directory, default: strategy name)')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Logging verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose (default: 1)')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--param-grid', type=str, help='Parameter grid for optimization')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'total_return', 'max_drawdown'],
                       help='Optimization metric (default: sharpe_ratio)')

    parser.add_argument('--sweep', action='store_true', help='Run backtest sweep across multiple symbols')
    parser.add_argument('--universe', type=str, help='Predefined universe (DOW30, NASDAQ100, FAANG, etc.)')
    parser.add_argument('--symbols-file', type=str, help='Path to file with symbols (one per line or CSV)')
    parser.add_argument('--sort-by', type=str, default='Sharpe Ratio', help='Sort sweep results by column (default: Sharpe Ratio)')
    parser.add_argument('--top-n', type=int, help='Show only top N results in sweep')
    parser.add_argument('--parallel', action='store_true', help='Run sweep in parallel')
    parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers for sweep (default: 4)')

    args = parser.parse_args()

    if args.sweep:
        if not args.symbols and not args.universe and not args.symbols_file:
            logger.error("Sweep mode requires --symbols, --universe, or --symbols-file")
            sys.exit(1)

        sweep_backtest(
            strategy_name=args.strategy,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            fees=args.fees,
            strategy_params=args.params,
            universe=args.universe,
            symbols_file=args.symbols_file,
            symbols_list=args.symbols,
            output_dir=args.output_dir,
            run_name=args.run_name,
            sort_by=args.sort_by,
            top_n=args.top_n,
            parallel=args.parallel,
            max_workers=args.max_workers,
            verbosity=args.verbosity
        )
    elif args.optimize:
        if not args.param_grid:
            logger.error("--param-grid required for optimization")
            sys.exit(1)

        if not args.symbols:
            logger.error("--symbols required for optimization")
            sys.exit(1)

        optimize_strategy(
            strategy_name=args.strategy,
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            param_grid_str=args.param_grid,
            metric=args.metric,
            initial_capital=args.capital,
            fees=args.fees
        )
    else:
        if not args.symbols:
            logger.error("--symbols required (or use --sweep with --universe/--symbols-file)")
            sys.exit(1)

        run_backtest(
            strategy_name=args.strategy,
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            fees=args.fees,
            strategy_params=args.params,
            save_report=args.save_report,
            show_plots=args.show_plots,
            visualize=args.visualize,
            quantstats=args.quantstats,
            output_dir=args.output_dir,
            run_name=args.run_name,
            verbosity=args.verbosity
        )


if __name__ == '__main__':
    main()
