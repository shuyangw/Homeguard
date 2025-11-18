"""
Example: Strategy Validation Using ValidationFramework

This example demonstrates how to use the ValidationFramework to validate
a strategy with minimal code. Compare this to the old approach which required
200-300 lines per script.

This script replaces the need for:
- validate_top_10_pairs.py (~250 lines)
- validate_all_50_pairs.py (~280 lines)
- validate_discovered_pairs.py (~300 lines)

With just ~40 lines using the framework!

Usage:
    # Using default config
    python example_validation_with_framework.py

    # Using custom config with overrides
    python example_validation_with_framework.py \
        --config config/pairs_trading.yaml \
        --start-date 2023-01-01 \
        --symbols SPY IWM QQQ DIA
"""

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

from frameworks.validation_framework import ValidationFramework
from strategies.advanced.pairs_trading import PairsTrading
from utils.cli_args import create_base_parser, parse_with_config
from utils.config_loader import get_nested


def main():
    """Main entry point."""
    # Parse arguments and load config
    parser = create_base_parser(description="Example validation with framework")
    args, config = parse_with_config(parser)

    # Initialize validation framework
    framework = ValidationFramework(config)

    # Get symbols (from config or CLI)
    symbols = get_nested(config, 'symbols', ['SPY', 'IWM'])
    pairs = get_nested(config, 'pairs', None)

    if pairs:
        # Validate multiple pairs
        symbol_groups = [
            (pair, f"{pair[0]}/{pair[1]}")
            for pair in pairs
        ]

        results = framework.run_multiple_validations(
            strategy_class=PairsTrading,
            symbol_groups=symbol_groups
        )
    else:
        # Single validation
        result = framework.run_validation(
            strategy_class=PairsTrading,
            symbols=symbols,
            description=f"Pairs Trading: {'/'.join(symbols)}"
        )

    # Export results
    if framework.results:
        framework.export_results(format='both')  # CSV and JSON
        framework.generate_report()


if __name__ == '__main__':
    main()
