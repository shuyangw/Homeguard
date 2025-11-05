"""
Example: Using cache to retrieve and re-run previous backtest settings.

This demonstrates how to:
1. Retrieve the most recent backtest settings from cache
2. Re-run a backtest with the same configuration
3. List all cached backtests with their settings
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.cache_manager import CacheManager
from utils import logger


def main():
    """Demonstrate cache usage for backtest settings."""

    # Initialize cache manager
    cache = CacheManager()

    logger.header("BACKTEST CACHE EXAMPLE")
    logger.blank()

    # Example 1: Get last run settings
    logger.info("Example 1: Retrieving last run settings")
    logger.separator()

    last_settings = cache.get_last_run_settings()

    if last_settings:
        logger.success("Found previous backtest settings!")
        logger.blank()
        logger.info("Strategy Configuration:")
        logger.info(f"  Strategy: {last_settings['strategy']}")
        logger.info(f"  Parameters: {last_settings['strategy_params']}")
        logger.blank()

        logger.info("Market Data:")
        logger.info(f"  Symbols: {', '.join(last_settings['symbols'])}")
        logger.info(f"  Period: {last_settings['start_date']} to {last_settings['end_date']}")
        logger.blank()

        logger.info("Execution Settings:")
        logger.info(f"  Initial Capital: ${last_settings['initial_capital']:,.0f}")
        logger.info(f"  Fees: {last_settings['fees']:.3%}")
        logger.info(f"  Risk Profile: {last_settings['risk_profile']}")
        logger.blank()

        logger.info("Portfolio Settings:")
        logger.info(f"  Mode: {last_settings['portfolio_mode']}")
        logger.info(f"  Position Sizing: {last_settings['position_sizing_method']}")
        logger.info(f"  Rebalancing: {last_settings['rebalancing_frequency']}")
        logger.info(f"  Rebalancing Threshold: {last_settings['rebalancing_threshold_pct']:.1%}")
        logger.blank()

        logger.info(f"Run Timestamp: {last_settings['timestamp']}")
    else:
        logger.warning("No previous backtest found in cache")

    logger.blank()
    logger.separator()

    # Example 2: List all cached runs
    logger.info("Example 2: Listing all cached backtests")
    logger.separator()

    runs = cache.list_cached_runs(limit=10)

    if runs:
        logger.success(f"Found {len(runs)} cached backtest(s)")
        logger.blank()

        for i, run in enumerate(runs, 1):
            logger.info(f"Run #{i}:")
            logger.info(f"  Strategy: {run['strategy']}")
            logger.info(f"  Symbols: {', '.join(run['symbols'][:3])}" +
                       (f" (+{len(run['symbols']) - 3} more)" if len(run['symbols']) > 3 else ""))
            logger.info(f"  Date Range: {run['date_range']}")
            logger.info(f"  Mode: {run['portfolio_mode']}")
            logger.info(f"  Risk: {run['risk_profile']}")
            logger.info(f"  Timestamp: {run['timestamp']}")
            logger.blank()
    else:
        logger.warning("No cached backtests found")

    logger.blank()
    logger.separator()

    # Example 3: Cache statistics
    logger.info("Example 3: Cache statistics")
    logger.separator()

    stats = cache.get_cache_size()
    logger.info(f"Cache Directory: {stats['cache_dir']}")
    logger.info(f"Total Size: {stats['total_size_mb']:.2f} MB")
    logger.info(f"Total Files: {stats['file_count']}")
    logger.info(f"Cached Runs: {stats['num_cached_runs']}")

    logger.blank()
    logger.separator()
    logger.header("EXAMPLE COMPLETE")
    logger.blank()

    # Show how to use these settings in a new backtest
    if last_settings:
        logger.blank()
        logger.info("To re-run this backtest, use these settings in your GUI or script:")
        logger.info("  - Same strategy and parameters")
        logger.info("  - Same symbols and date range")
        logger.info("  - Same risk and portfolio settings")
        logger.blank()


if __name__ == '__main__':
    main()
