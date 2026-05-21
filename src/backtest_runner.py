"""
Main interface for running backtests with trading strategies.

Usage:
    # Traditional CLI usage:
    python backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL --start 2023-01-01 --end 2024-01-01
    python backtest_runner.py --strategy MeanReversion --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01

    # Config-driven usage (NEW):
    python -m src.backtest_runner --config config/backtesting/omr_backtest.yaml
    python -m src.backtest_runner --config config/backtesting/ma_sweep.yaml --mode sweep
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.engine.metrics import PerformanceMetrics
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader as DataLoader
from src.backtesting.engine.trade_logger import TradeLogger
from src.backtesting.optimization.sweep_runner import SweepRunner
from src.backtesting.utils.universe import UniverseManager
from src.strategies.research.moving_average import MovingAverageCrossover, TripleMovingAverage
from src.strategies.research.mean_reversion import MeanReversion, RSIMeanReversion
from src.strategies.research.momentum import MomentumStrategy, BreakoutStrategy
from src.visualization.config import VisualizationConfig
from src.visualization.integration import BacktestVisualizer
from src.settings import get_log_output_dir, get_backtest_results_dir
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
    from src.strategies.research.volatility_targeted_momentum import VolatilityTargetedMomentum
    from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversionStrategy
    from src.strategies.research.cross_sectional_momentum import CrossSectionalMomentum
    from src.strategies.research.pairs_trading import PairsTrading

    STRATEGY_REGISTRY['VolatilityTargetedMomentum'] = VolatilityTargetedMomentum
    STRATEGY_REGISTRY['OvernightMeanReversion'] = OvernightMeanReversionStrategy
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
        from src.settings import get_backtest_results_dir
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


# ============================================================================
# Config-driven backtest functions
# ============================================================================

def _resolve_symbols(config: 'BacktestConfig') -> List[str]:
    """
    Resolve symbols from config (list, universe reference, or file).

    Args:
        config: BacktestConfig object

    Returns:
        List of symbol strings
    """
    from src.settings import get_symbol_universe

    symbols_config = config.symbols

    if symbols_config.list:
        return symbols_config.list

    if symbols_config.universe:
        return get_symbol_universe(symbols_config.universe)

    if symbols_config.file:
        file_path = Path(symbols_config.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Symbols file not found: {file_path}")

        # Check if it's a CSV file
        if file_path.suffix.lower() == '.csv':
            import csv
            symbols = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names for symbol
                    symbol = row.get('Symbol') or row.get('symbol') or row.get('SYMBOL') or row.get('Ticker') or row.get('ticker')
                    if symbol:
                        symbols.append(symbol.strip())
            return symbols
        else:
            # Plain text file with one symbol per line
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return symbols

    raise ValueError("No symbols source specified in config")


def _resolve_dates(config: 'BacktestConfig') -> tuple:
    """
    Resolve start and end dates from config (explicit or preset).

    Args:
        config: BacktestConfig object

    Returns:
        Tuple of (start_date, end_date) strings
    """
    from src.settings import get_date_preset

    dates_config = config.dates

    if dates_config.preset:
        preset = get_date_preset(dates_config.preset)
        return preset['start'], preset['end']

    return dates_config.start, dates_config.end


def _get_strategy_instance(config: 'BacktestConfig'):
    """
    Get strategy instance from config.

    Handles both parameter styles:
    - **kwargs style (most strategies): strategy_cls(fast_window=10, slow_window=50)
    - params dict style (OMR, etc.): strategy_cls(params={'key': value})

    Args:
        config: BacktestConfig object

    Returns:
        Instantiated strategy object
    """
    from src.strategies.registry import get_strategy_class

    strategy_cls = get_strategy_class(config.strategy.name)
    params = config.strategy.parameters

    if not params:
        return strategy_cls()

    try:
        return strategy_cls(**params)
    except TypeError:
        return strategy_cls(params=params)


def _create_output_dir(config: 'BacktestConfig', mode_suffix: str = "") -> Optional[Path]:
    """
    Create output directory for results.

    Args:
        config: BacktestConfig object
        mode_suffix: Optional suffix for mode (e.g., "_sweep")

    Returns:
        Path to output directory, or None if output disabled
    """
    if config.output.directory:
        output_dir = Path(config.output.directory)
    elif config.output.quantstats or config.output.save_reports:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = get_backtest_results_dir()
        output_dir = results_dir / f"{timestamp}_{config.strategy.name}{mode_suffix}"
    else:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_costs(config: 'BacktestConfig') -> tuple:
    """Return (fees, slippage, stop_slippage_multiplier) honoring `costs.tier`
    per methodology Section 4.

    Hard-fail when both `costs.tier` AND `costs.bps_override` are None
    (Section 4 governance). Strategies that don't fit the bps-tier model
    -- e.g., options strategies that need the alpha-of-half-spread fill
    model from Section 4.5 -- declare an explicit `costs.bps_override`
    as the opt-out, with a comment pointing at the deferred wiring.

    stop_slippage_multiplier is sourced from `costs.stop_slippage_multiplier`
    (default 1.5 per methodology Section 11.5) and applied to slippage on
    stop-loss exits only.
    """
    tier = getattr(getattr(config, 'costs', None), 'tier', None)
    override_bps = getattr(getattr(config, 'costs', None), 'bps_override', None)

    if tier is None and override_bps is None:
        raise ValueError(
            f"[costs] {config.strategy.name}: backtest config has neither "
            "costs.tier nor costs.bps_override set. Section 4 requires every "
            "backtest to declare its cost basis. Pick a tier from "
            "{large_cap_liquid, mid_cap, leveraged_etf, small_cap, "
            "crypto_major, crypto_alt} OR set costs.bps_override with a "
            "comment naming the asset-class model the bps approximates. "
            "Reference: docs/methodology/backtesting.md Section 4."
        )

    if override_bps is not None:
        round_trip_bps = float(override_bps)
    else:
        # Equity / leveraged-ETF / small-cap tiers come from costs.equities;
        # crypto from costs.crypto. Other asset classes (FX, futures, options)
        # use specialized models and are out of scope for this fees/slippage path.
        equity_tiers = {'large_cap_liquid', 'mid_cap', 'leveraged_etf', 'small_cap'}
        crypto_tiers = {'crypto_major', 'crypto_alt'}
        if tier in equity_tiers:
            from src.backtesting.costs import equities_round_trip_bps
            round_trip_bps = equities_round_trip_bps(tier)
        elif tier in crypto_tiers:
            from src.backtesting.costs import crypto_round_trip_bps
            pair_tier = 'major' if tier == 'crypto_major' else 'altcoin'
            round_trip_bps = crypto_round_trip_bps(pair_tier)
        else:
            raise ValueError(
                f"costs.tier='{tier}' is not yet supported by run_*_from_config. "
                "Supported: large_cap_liquid, mid_cap, leveraged_etf, small_cap, "
                "crypto_major, crypto_alt. Use bps_override for other asset classes."
            )

    # The engine charges `fees` on each side (entry and exit) and applies `slippage`
    # on each fill. Put the entire round-trip in per-side fees; leave slippage at 0
    # since the tier round-trip already encompasses spread.
    per_side_fees = round_trip_bps / 10_000.0 / 2.0
    stop_slippage_multiplier = getattr(
        getattr(config, 'costs', None), 'stop_slippage_multiplier', 1.0
    ) or 1.0
    logger.info(
        f"[costs] {config.strategy.name}: tier={tier} -> "
        f"round_trip={round_trip_bps:.1f} bps, per_side_fees={per_side_fees:.5f}, "
        f"stop_slippage_multiplier={stop_slippage_multiplier:.2f}"
    )
    return per_side_fees, 0.0, stop_slippage_multiplier


def _append_to_registry(
    config: 'BacktestConfig',
    portfolio,
    symbols: List[str],
    start_date: str,
    end_date: str,
    wall_clock_start: datetime,
    *,
    agent_name: str = 'backtest-runner',
    phase: str = 'initial',
    parent_run_id: Optional[str] = None,
    combinations_in_run: int = 1,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """Append a run row to the experiment registry per methodology Section 9.

    Fatal on failure (Section 9.3 -- "if the append fails, the run fails.
    no silent success"). Transient Windows file-lock errors from Dropbox
    contention are retried inside `_connect_with_retry`; anything that
    survives the retries propagates to the caller.
    """
    from src.experiments import append_run, n_trials_project_wide
    try:
        stats = portfolio.stats() if hasattr(portfolio, 'stats') else None
        metrics: Dict[str, Any] = {}
        if stats:
            metrics = {
                'sharpe': float(stats.get('Sharpe Ratio', 0.0) or 0.0),
                'total_return_pct': float(stats.get('Total Return [%]', 0.0) or 0.0),
                'annual_return_pct': float(stats.get('Annual Return [%]', 0.0) or 0.0),
                'max_drawdown_pct': float(stats.get('Max Drawdown [%]', 0.0) or 0.0),
                'win_rate_pct': float(stats.get('Win Rate [%]', 0.0) or 0.0),
                'trade_count': int(stats.get('Total Trades', 0) or 0),
                'start_value': float(stats.get('Start Value', 0.0) or 0.0),
                'end_value': float(stats.get('End Value', 0.0) or 0.0),
            }

        params: Dict[str, Any] = {}
        try:
            if config.strategy and getattr(config.strategy, 'parameters', None):
                params = dict(config.strategy.parameters)
        except Exception:
            params = {}

        # Cost-tier provenance: populate the registry's cost_tier_used and
        # cost_bps columns so DSR / cost-sensitivity follow-ups can read
        # what was actually used at runtime (not just what was in config).
        cost_tier_used: Optional[str] = None
        cost_bps_val: Optional[float] = None
        try:
            costs_cfg = getattr(config, 'costs', None)
            if costs_cfg is not None and getattr(costs_cfg, 'tier', None):
                cost_tier_used = costs_cfg.tier
                # Resolve the round-trip bps the same way _resolve_costs does.
                override = getattr(costs_cfg, 'bps_override', None)
                if override is not None:
                    cost_bps_val = float(override)
                else:
                    equity_tiers = {'large_cap_liquid', 'mid_cap', 'leveraged_etf', 'small_cap'}
                    crypto_tiers = {'crypto_major', 'crypto_alt'}
                    if cost_tier_used in equity_tiers:
                        from src.backtesting.costs import equities_round_trip_bps
                        cost_bps_val = float(equities_round_trip_bps(cost_tier_used))
                    elif cost_tier_used in crypto_tiers:
                        from src.backtesting.costs import crypto_round_trip_bps
                        pair_tier = 'major' if cost_tier_used == 'crypto_major' else 'altcoin'
                        cost_bps_val = float(crypto_round_trip_bps(pair_tier))
        except Exception:
            pass  # best-effort; missing provenance does not fail the append

        return append_run(
            run_id=run_id,
            strategy_name=config.strategy.name,
            agent_name=agent_name,
            phase=phase,
            parent_run_id=parent_run_id,
            params=params,
            universe_name=str(config.symbols.universe) if getattr(config, 'symbols', None) and getattr(config.symbols, 'universe', None) else None,
            asset_class=None,  # populated by Phase 7 cost-tier wiring
            cost_tier_used=cost_tier_used,
            cost_bps=cost_bps_val,
            data_frequency=getattr(config.backtest, 'timeframe', None),
            window_start=start_date,
            window_end=end_date,
            metrics=metrics,
            combinations_in_run=combinations_in_run,
            combinations_project=n_trials_project_wide(),
            config_payload=getattr(config, 'model_dump', lambda: None)() if hasattr(config, 'model_dump') else None,
            wall_clock_start=wall_clock_start,
            wall_clock_end=datetime.now(tz=timezone.utc),
            verdict='PENDING',
        )
    except Exception as e:  # noqa
        # Section 9.3: "if the append fails, the run fails. no silent success."
        # Transient Dropbox-style file locks are retried inside
        # _connect_with_retry; anything reaching here is a real failure.
        logger.error(f"[registry] append failed (fatal per Section 9.3): {e}")
        raise


def run_single_from_config(config: 'BacktestConfig') -> None:
    """
    Run single backtest from config.

    Args:
        config: Validated BacktestConfig object
    """
    from src.backtesting.utils.risk_config import RiskConfig

    wall_clock_start = datetime.now(tz=timezone.utc)

    symbols = _resolve_symbols(config)
    start_date, end_date = _resolve_dates(config)
    strategy = _get_strategy_instance(config)

    # Create RiskConfig from YAML config
    risk_config = None
    if config.risk and config.risk.enabled:
        # Map schema enum to RiskConfig literal
        sizing_method_map = {
            'fixed_percent': 'fixed_percentage',
            'kelly': 'kelly_criterion',
            'volatility_target': 'volatility_based',
            'equal_weight': 'risk_parity',
            'custom': 'fixed_percentage',
        }
        sizing_method = config.risk.position_sizing_method.value if hasattr(config.risk.position_sizing_method, 'value') else str(config.risk.position_sizing_method)
        mapped_method = sizing_method_map.get(sizing_method, 'fixed_percentage')

        risk_config = RiskConfig(
            position_size_pct=config.risk.position_size_pct,
            position_sizing_method=mapped_method,
            use_stop_loss=config.risk.stop_loss_pct is not None,
            stop_loss_pct=config.risk.stop_loss_pct if config.risk.stop_loss_pct else 0.02,
            max_positions=config.risk.max_positions,
        )

    fees, slippage, stop_slip_mult = _resolve_costs(config)

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        fees=fees,
        slippage=slippage,
        allow_shorts=config.backtest.allow_shorts,
        risk_config=risk_config,
        timeframe=config.backtest.timeframe,
        market_hours_only=config.backtest.market_hours_only,
        stop_slippage_multiplier=stop_slip_mult,
    )

    output_dir = _create_output_dir(config)

    # Get portfolio mode from config (default to 'single')
    portfolio_mode = getattr(config.backtest, 'portfolio_mode', 'single')

    if config.output.quantstats and output_dir:
        portfolio = engine.run_and_report(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            portfolio_mode=portfolio_mode
        )
    else:
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            portfolio_mode=portfolio_mode
        )

    if config.output.visualize and output_dir:
        viz_config = VisualizationConfig.from_args(
            verbosity=config.output.verbosity,
            enable_charts=True,
            enable_logs=True
        )

        visualizer = BacktestVisualizer(viz_config)
        loader = DataLoader()

        if len(symbols) == 1:
            price_data = loader.load_single_symbol(symbols[0], start_date, end_date)
        else:
            price_data = loader.load_symbols(symbols, start_date, end_date)

        visualizer.visualize_backtest(
            portfolio=portfolio,
            strategy_name=config.strategy.name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.backtest.initial_capital,
            fees=config.backtest.fees,
            price_data=price_data,
            output_dir=output_dir
        )

    # Export trades if save_trades is enabled
    if config.output.save_trades and output_dir:
        trades_dir = output_dir / "trades"
        trades_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_csv = trades_dir / f"{timestamp}_all_trades.csv"

        try:
            TradeLogger.export_trades_csv(portfolio, trades_csv)
            logger.success(f"Trade log exported: {trades_csv}")

            # Also print a trade-count summary to console. The V1 Portfolio
            # exposes `trades` as a list of dicts (entry + exit rows);
            # MultiAssetPortfolio exposes a DataFrame of round-trips. Handle
            # both without trying to .iterrows() over a list.
            trades_obj = portfolio.trades
            if isinstance(trades_obj, list):
                total_rows = len(trades_obj)
                exit_count = sum(
                    1 for t in trades_obj
                    if t.get('type') in ('exit', 'cover_short')
                )
                if total_rows > 0:
                    logger.blank()
                    logger.metric(
                        f"Total Trades: {exit_count} round-trips "
                        f"({total_rows} entry+exit rows)"
                    )
            elif isinstance(trades_obj, pd.DataFrame):
                if len(trades_obj) > 0:
                    logger.blank()
                    logger.metric(f"Total Trades: {len(trades_obj)}")
        except Exception as e:
            logger.warning(f"Could not export trades: {e}")

    logger.success(f"Backtest complete: {config.strategy.name}")
    if output_dir:
        logger.info(f"Results saved to: {output_dir}")

    # Append to the experiment registry per methodology Section 9. Best-effort
    # during initial rollout: log-and-continue on failure so existing backtests
    # aren't broken by a registry outage.
    run_id = _append_to_registry(
        config=config,
        portfolio=portfolio,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        wall_clock_start=wall_clock_start,
        agent_name='backtest-runner',
        phase='initial',
    )
    if run_id:
        logger.info(f"[registry] appended run_id={run_id}")


def run_sweep_from_config(config: 'BacktestConfig') -> None:
    """
    Run sweep backtest from config.

    Args:
        config: Validated BacktestConfig object
    """
    wall_clock_start = datetime.now(tz=timezone.utc)

    symbols = _resolve_symbols(config)
    start_date, end_date = _resolve_dates(config)
    strategy = _get_strategy_instance(config)

    fees, slippage, stop_slip_mult = _resolve_costs(config)

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        fees=fees,
        slippage=slippage,
        allow_shorts=config.backtest.allow_shorts,
        timeframe=config.backtest.timeframe,
        market_hours_only=config.backtest.market_hours_only,
        stop_slippage_multiplier=stop_slip_mult,
    )

    sweep_runner = SweepRunner(
        engine=engine,
        max_workers=config.sweep.max_workers,
        show_progress=(config.output.verbosity >= 1)
    )

    output_dir = _create_output_dir(config, "_sweep")
    output_str = str(output_dir) if output_dir else None

    sweep_runner.run_and_report(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_str,
        sort_by=config.sweep.sort_by,
        top_n=config.sweep.top_n,
        export_csv=config.sweep.export_csv,
        export_html=config.sweep.export_html,
        parallel=config.sweep.parallel
    )

    logger.success(f"Sweep complete: {config.strategy.name}")
    if output_dir:
        logger.info(f"Results saved to: {output_dir}")

    # Aggregate sweep row: per-symbol detail is in SweepRunner's CSV/HTML.
    # The registry gets one parent row; per-config wiring (per-symbol or
    # per-param row) is a follow-up that needs SweepRunner to expose results.
    run_id = _append_to_registry(
        config=config,
        portfolio=None,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        wall_clock_start=wall_clock_start,
        agent_name='backtest-runner',
        phase='sweep',
        combinations_in_run=len(symbols),
    )
    if run_id:
        logger.info(f"[registry] appended sweep run_id={run_id}")


def run_optimize_from_config(config: 'BacktestConfig') -> None:
    """
    Run optimization from config.

    Args:
        config: Validated BacktestConfig object
    """
    from src.strategies.registry import get_strategy_class

    wall_clock_start = datetime.now(tz=timezone.utc)

    symbols = _resolve_symbols(config)
    start_date, end_date = _resolve_dates(config)
    strategy_cls = get_strategy_class(config.strategy.name)

    fees, slippage, stop_slip_mult = _resolve_costs(config)

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        fees=fees,
        slippage=slippage,
        allow_shorts=config.backtest.allow_shorts,
        timeframe=config.backtest.timeframe,
        market_hours_only=config.backtest.market_hours_only,
        stop_slippage_multiplier=stop_slip_mult,
    )

    if not config.optimization.param_grid:
        logger.error("No param_grid specified in optimization config")
        sys.exit(1)

    # Build the per-trial registry callback once and pass to the optimizer.
    # Counts every combo toward project-wide N for DSR (methodology 9.4).
    n_combos = 1
    for v in (config.optimization.param_grid or {}).values():
        try:
            n_combos *= max(1, len(v))
        except TypeError:
            pass
    sweep_parent_id = str(uuid.uuid4())
    from src.experiments import make_trial_callback
    trial_cb = make_trial_callback(
        parent_run_id=sweep_parent_id,
        strategy_name=config.strategy.name,
        combinations_in_run=n_combos,
        start_date=start_date,
        end_date=end_date,
        wall_clock_start=wall_clock_start,
        cost_tier_used=getattr(engine, 'cost_tier_used', None),
        cost_bps=getattr(engine, 'cost_bps', None),
        phase='optimization_grid_search',
    )

    results = engine.optimize(
        strategy_class=strategy_cls,
        param_grid=config.optimization.param_grid,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        metric=config.optimization.metric,
        on_trial_complete=trial_cb,
    )

    logger.blank()
    logger.success("Optimization complete!")
    logger.metric(f"Best parameters: {results['best_params']}")
    logger.metric(f"Best {config.optimization.metric}: {results['best_value']:.4f}")

    # Parent row uses sweep_parent_id so the per-trial children appended
    # by trial_cb link to it via parent_run_id.
    run_id = _append_to_registry(
        config=config,
        portfolio=results.get('best_portfolio') if isinstance(results, dict) else None,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        wall_clock_start=wall_clock_start,
        agent_name='backtest-optimizer',
        phase='optimization',
        combinations_in_run=n_combos,
        run_id=sweep_parent_id,
    )
    if run_id:
        logger.info(f"[registry] appended optimization run_id={run_id} (combinations={n_combos})")


def run_walk_forward_from_config(config: 'BacktestConfig') -> None:
    """
    Run walk-forward validation from config.

    Args:
        config: Validated BacktestConfig object
    """
    from src.backtesting.optimization.walk_forward import WalkForwardOptimizer
    from src.backtesting.optimization.grid_search import GridSearchOptimizer
    from src.backtesting.engine.backtest_engine import BacktestEngine
    from src.strategies.registry import get_strategy_class

    wall_clock_start = datetime.now(tz=timezone.utc)

    symbols = _resolve_symbols(config)
    start_date, end_date = _resolve_dates(config)
    strategy_cls = get_strategy_class(config.strategy.name)

    if not config.optimization.param_grid:
        logger.error("No param_grid specified for walk-forward (needed for optimization)")
        sys.exit(1)

    output_dir = _create_output_dir(config, "_walk_forward")

    fees, slippage, stop_slip_mult = _resolve_costs(config)

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        fees=fees,
        slippage=slippage,
        allow_shorts=config.backtest.allow_shorts,
        timeframe=config.backtest.timeframe,
        market_hours_only=config.backtest.market_hours_only,
        stop_slippage_multiplier=stop_slip_mult,
    )

    base_optimizer = GridSearchOptimizer(engine)

    walk_forward = WalkForwardOptimizer(engine, base_optimizer)

    results = walk_forward.analyze(
        strategy_class=strategy_cls,
        param_space=config.optimization.param_grid,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_months=config.walk_forward.train_months,
        test_months=config.walk_forward.test_months,
        step_months=config.walk_forward.step_months,
        metric=config.optimization.metric,
        export_results=config.walk_forward.export_results
    )

    if output_dir:
        logger.info(f"Walk-forward results saved to: {output_dir}")

    logger.success("Walk-forward validation complete!")

    n_combos = 1
    for v in (config.optimization.param_grid or {}).values():
        try:
            n_combos *= max(1, len(v))
        except TypeError:
            pass
    run_id = _append_to_registry(
        config=config,
        portfolio=None,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        wall_clock_start=wall_clock_start,
        agent_name='backtest-optimizer',
        phase='walk_forward',
        combinations_in_run=n_combos,
    )
    if run_id:
        logger.info(f"[registry] appended walk-forward run_id={run_id} (combinations={n_combos})")


def run_from_config(config: 'BacktestConfig') -> None:
    """
    Run backtest based on config mode.

    Args:
        config: Validated BacktestConfig object
    """
    from src.settings import BacktestMode

    mode = config.mode

    if mode == BacktestMode.SINGLE:
        run_single_from_config(config)
    elif mode == BacktestMode.SWEEP:
        run_sweep_from_config(config)
    elif mode == BacktestMode.OPTIMIZE:
        run_optimize_from_config(config)
    elif mode == BacktestMode.WALK_FORWARD:
        run_walk_forward_from_config(config)
    else:
        logger.error(f"Unknown backtest mode: {mode}")
        sys.exit(1)


def main():
    """
    Main command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies using VectorBT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config-driven backtest (RECOMMENDED)
  python -m src.backtest_runner --config configs/examples/omr_backtest.yaml
  python -m src.backtest_runner --config configs/ma_sweep.yaml --mode sweep

  # Simple backtest (traditional)
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
  - VolatilityTargetedMomentum
  - OvernightMeanReversion
  - CrossSectionalMomentum
  - PairsTrading
        """
    )

    # Config-driven mode (NEW)
    parser.add_argument('--config', type=str, help='Path to YAML config file (enables config-driven mode)')
    parser.add_argument('--mode', type=str, choices=['single', 'sweep', 'optimize', 'walk_forward'],
                       help='Override mode from config file')

    # Traditional mode (strategy, symbols, dates required unless using --config)
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
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

    # =========== Config-driven mode ===========
    if args.config:
        from src.settings import load_config, BacktestMode

        try:
            config = load_config(args.config)

            # Override mode if specified on command line
            if args.mode:
                config.mode = BacktestMode(args.mode)

            logger.info(f"Running config-driven backtest: {args.config}")
            logger.info(f"Mode: {config.mode.value}")
            logger.info(f"Strategy: {config.strategy.name}")

            run_from_config(config)
            return

        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    # =========== Traditional CLI mode ===========
    # Validate required arguments for traditional mode
    if not args.strategy:
        logger.error("--strategy required (or use --config for config-driven mode)")
        sys.exit(1)
    if not args.start:
        logger.error("--start required (or use --config for config-driven mode)")
        sys.exit(1)
    if not args.end:
        logger.error("--end required (or use --config for config-driven mode)")
        sys.exit(1)

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
