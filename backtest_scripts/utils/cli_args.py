"""
Common CLI argument parser for backtest scripts.

This module provides a standardized argument parser that all backtest scripts
can use and extend. It provides common arguments for configuration, date ranges,
symbols, output settings, etc.

Usage:
    from utils.cli_args import create_base_parser, parse_with_config

    # Create parser with common arguments
    parser = create_base_parser(description="My backtest script")

    # Add script-specific arguments
    parser.add_argument('--my-param', type=float, default=1.0)

    # Parse arguments and load config
    args, config = parse_with_config(parser)
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from utils.config_loader import load_config, get_nested


def create_base_parser(description: str = "Backtest script") -> argparse.ArgumentParser:
    """
    Create base argument parser with common backtest arguments.

    All backtest scripts should use this as a starting point and add
    script-specific arguments as needed.

    Args:
        description: Description of the script for help text

    Returns:
        ArgumentParser with common arguments

    Example:
        >>> parser = create_base_parser("Pairs trading validation")
        >>> parser.add_argument('--entry-z', type=float, default=2.0)
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (relative to config/ or absolute)'
    )

    # Date range arguments
    date_group = parser.add_argument_group('Date Range')
    date_group.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Backtest start date (YYYY-MM-DD). Overrides config value.'
    )
    date_group.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Backtest end date (YYYY-MM-DD). Overrides config value.'
    )
    date_group.add_argument(
        '--date-range',
        type=str,
        default=None,
        choices=['max_history', 'five_years', 'three_years', 'one_year',
                 'bull_2019_2021', 'bear_2022', 'covid_crash', 'covid_recovery'],
        help='Predefined date range from config/date_ranges.yaml'
    )

    # Symbol arguments
    symbol_group = parser.add_argument_group('Symbols')
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='List of symbols to backtest. Overrides config value.'
    )
    symbol_group.add_argument(
        '--universe',
        type=str,
        default=None,
        help='Symbol universe from config/universes.yaml (e.g., "indices.all_indices")'
    )

    # Risk management
    risk_group = parser.add_argument_group('Risk Management')
    risk_group.add_argument(
        '--initial-cash',
        type=float,
        default=None,
        help='Initial portfolio cash. Overrides config value.'
    )
    risk_group.add_argument(
        '--position-size',
        type=float,
        default=None,
        help='Position size as percentage (0.0-1.0). Overrides config value.'
    )
    risk_group.add_argument(
        '--commission',
        type=float,
        default=None,
        help='Commission rate as decimal (e.g., 0.001 = 0.1%%). Overrides config value.'
    )

    # Output settings
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for CSV files. Overrides config value.'
    )
    output_group.add_argument(
        '--reports-dir',
        type=str,
        default=None,
        help='Reports directory for markdown files. Overrides config value.'
    )
    output_group.add_argument(
        '--save-trades',
        action='store_true',
        default=None,
        help='Save trade logs to CSV'
    )
    output_group.add_argument(
        '--no-save-trades',
        action='store_true',
        default=None,
        help='Do not save trade logs'
    )
    output_group.add_argument(
        '--save-report',
        action='store_true',
        default=None,
        help='Generate markdown report'
    )
    output_group.add_argument(
        '--no-save-report',
        action='store_true',
        default=None,
        help='Do not generate markdown report'
    )

    # Logging and debugging
    debug_group = parser.add_argument_group('Logging & Debugging')
    debug_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    debug_group.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal logging (WARNING level only)'
    )
    debug_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    # Optimization-specific arguments
    opt_group = parser.add_argument_group('Optimization (optional)')
    opt_group.add_argument(
        '--optimize',
        action='store_true',
        help='Run parameter optimization instead of single backtest'
    )
    opt_group.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for optimization (default: 1)'
    )
    opt_group.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum optimization iterations'
    )

    return parser


def parse_with_config(
    parser: argparse.ArgumentParser,
    args: Optional[list] = None
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse arguments and load configuration with CLI overrides.

    This function:
    1. Parses command-line arguments
    2. Loads config file if specified
    3. Applies CLI argument overrides to config
    4. Returns both args and merged config

    Args:
        parser: ArgumentParser (from create_base_parser)
        args: Optional list of arguments (for testing). Defaults to sys.argv

    Returns:
        Tuple of (args, config) where:
            - args: Parsed arguments namespace
            - config: Configuration dictionary with overrides applied

    Example:
        >>> parser = create_base_parser()
        >>> args, config = parse_with_config(parser)
        >>> start_date = config['backtest']['start_date']
    """
    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Load base config
    if parsed_args.config:
        config = load_config(parsed_args.config)
    else:
        # Load default config if no config specified
        config = load_config("default_backtest.yaml")

    # Build overrides from CLI arguments
    overrides = {}

    # Date range overrides
    if parsed_args.start_date is not None:
        overrides['backtest.start_date'] = parsed_args.start_date
    if parsed_args.end_date is not None:
        overrides['backtest.end_date'] = parsed_args.end_date

    # Handle predefined date ranges
    if parsed_args.date_range is not None:
        date_ranges = load_config("date_ranges.yaml")
        if parsed_args.date_range in date_ranges.get('full_periods', {}):
            range_config = date_ranges['full_periods'][parsed_args.date_range]
            overrides['backtest.start_date'] = range_config['start']
            overrides['backtest.end_date'] = range_config['end']
        elif parsed_args.date_range in date_ranges.get('regimes', {}):
            range_config = date_ranges['regimes'][parsed_args.date_range]
            overrides['backtest.start_date'] = range_config['start']
            overrides['backtest.end_date'] = range_config['end']

    # Symbol overrides
    if parsed_args.symbols is not None:
        overrides['symbols'] = parsed_args.symbols

    # Handle universe lookup
    if parsed_args.universe is not None:
        universes = load_config("universes.yaml")
        universe_symbols = get_nested(universes, parsed_args.universe)
        if universe_symbols:
            overrides['symbols'] = universe_symbols

    # Risk management overrides
    if parsed_args.initial_cash is not None:
        overrides['backtest.initial_cash'] = parsed_args.initial_cash
    if parsed_args.position_size is not None:
        overrides['risk.position_size_pct'] = parsed_args.position_size
    if parsed_args.commission is not None:
        overrides['costs.commission'] = parsed_args.commission

    # Output overrides
    if parsed_args.output_dir is not None:
        overrides['output.output_dir'] = parsed_args.output_dir
    if parsed_args.reports_dir is not None:
        overrides['output.reports_dir'] = parsed_args.reports_dir

    # Boolean output overrides (handle --save-trades vs --no-save-trades)
    if parsed_args.save_trades:
        overrides['output.save_trades'] = True
    elif parsed_args.no_save_trades:
        overrides['output.save_trades'] = False

    if parsed_args.save_report:
        overrides['output.save_reports'] = True
    elif parsed_args.no_save_report:
        overrides['output.save_reports'] = False

    # Logging overrides
    if parsed_args.verbose:
        overrides['logging.level'] = 'DEBUG'
    elif parsed_args.quiet:
        overrides['logging.level'] = 'WARNING'

    if parsed_args.no_progress:
        overrides['logging.show_progress'] = False

    # Apply all overrides
    from utils.config_loader import apply_overrides
    config = apply_overrides(config, overrides)

    return parsed_args, config


def get_output_filename(
    script_name: str,
    config: Dict[str, Any],
    suffix: str = "results.csv"
) -> Path:
    """
    Generate standardized output filename with timestamp.

    Args:
        script_name: Name of the script (e.g., "validate_pairs_comprehensive")
        config: Configuration dictionary
        suffix: File suffix (default: "results.csv")

    Returns:
        Path to output file with timestamp

    Example:
        >>> config = {"output": {"output_dir": "output/validation", "timestamp_files": True}}
        >>> get_output_filename("validate_pairs", config)
        Path("output/validation/20251118_143022_validate_pairs_results.csv")
    """
    output_dir = Path(get_nested(config, 'output.output_dir', 'output'))
    timestamp_files = get_nested(config, 'output.timestamp_files', True)

    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp_files:
        timestamp = datetime.now().strftime(get_nested(config, 'output.timestamp_format', '%Y%m%d_%H%M%S'))
        filename = f"{timestamp}_{script_name}_{suffix}"
    else:
        filename = f"{script_name}_{suffix}"

    return output_dir / filename
