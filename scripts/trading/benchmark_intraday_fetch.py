"""
Benchmark intraday data fetching times.

Compares:
1. Full day fetch (9:30 AM - 3:50 PM) - 380 bars × N symbols
2. Last 5 minutes fetch (3:45-3:50 PM) - 5 bars × N symbols

This helps determine if 3:45 PM pre-fetching provides meaningful speedup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd

from src.trading.brokers import AlpacaBroker
from src.strategies.universe import ETFUniverse
from src.utils.logger import logger


def benchmark_full_day_fetch(broker: AlpacaBroker, symbols: list) -> Dict:
    """
    Benchmark fetching full intraday data (9:30 AM - 3:50 PM).

    This simulates what happens without pre-fetching.
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK: Full Day Fetch (WITHOUT Pre-Fetching)")
    logger.info("=" * 80)

    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_window = now.replace(hour=15, minute=50, second=0, microsecond=0)

    results = {
        'mode': 'full_day',
        'symbols_attempted': len(symbols),
        'symbols_succeeded': 0,
        'total_bars': 0,
        'total_time': 0,
        'per_symbol_times': {},
        'errors': []
    }

    logger.info(f"Fetching {len(symbols)} symbols from {market_open.strftime('%H:%M')} to {market_close_window.strftime('%H:%M')}")
    logger.info(f"Expected: ~380 bars per symbol")
    logger.info("")

    overall_start = time.time()

    for symbol in symbols:
        try:
            symbol_start = time.time()

            df = broker.get_historical_bars(
                symbol=symbol,
                start=market_open,
                end=market_close_window,
                timeframe='1Min'
            )

            symbol_time = time.time() - symbol_start

            if df is not None and not df.empty:
                bars = len(df)
                results['symbols_succeeded'] += 1
                results['total_bars'] += bars
                results['per_symbol_times'][symbol] = symbol_time

                logger.info(f"  ✓ {symbol:6s}: {bars:4d} bars in {symbol_time:6.2f}s")
            else:
                results['errors'].append(f"{symbol}: No data returned")
                logger.warning(f"  ✗ {symbol:6s}: No data")

        except Exception as e:
            results['errors'].append(f"{symbol}: {str(e)}")
            logger.error(f"  ✗ {symbol:6s}: {str(e)}")

    overall_time = time.time() - overall_start
    results['total_time'] = overall_time

    logger.info("")
    logger.info(f"Total time: {overall_time:.2f}s")
    logger.info(f"Success rate: {results['symbols_succeeded']}/{results['symbols_attempted']}")
    logger.info(f"Total bars fetched: {results['total_bars']:,}")
    logger.info(f"Average per symbol: {overall_time / len(symbols):.2f}s")

    return results


def benchmark_last_5min_fetch(broker: AlpacaBroker, symbols: list) -> Dict:
    """
    Benchmark fetching only last 5 minutes (3:45-3:50 PM).

    This simulates what happens WITH pre-fetching (just the final update).
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK: Last 5 Minutes Fetch (WITH Pre-Fetching)")
    logger.info("=" * 80)

    now = datetime.now()
    start_time = now.replace(hour=15, minute=45, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=50, second=0, microsecond=0)

    results = {
        'mode': 'last_5min',
        'symbols_attempted': len(symbols),
        'symbols_succeeded': 0,
        'total_bars': 0,
        'total_time': 0,
        'per_symbol_times': {},
        'errors': []
    }

    logger.info(f"Fetching {len(symbols)} symbols from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
    logger.info(f"Expected: ~5 bars per symbol")
    logger.info("")

    overall_start = time.time()

    for symbol in symbols:
        try:
            symbol_start = time.time()

            df = broker.get_historical_bars(
                symbol=symbol,
                start=start_time,
                end=end_time,
                timeframe='1Min'
            )

            symbol_time = time.time() - symbol_start

            if df is not None and not df.empty:
                bars = len(df)
                results['symbols_succeeded'] += 1
                results['total_bars'] += bars
                results['per_symbol_times'][symbol] = symbol_time

                logger.info(f"  ✓ {symbol:6s}: {bars:4d} bars in {symbol_time:6.2f}s")
            else:
                results['errors'].append(f"{symbol}: No data returned")
                logger.warning(f"  ✗ {symbol:6s}: No data")

        except Exception as e:
            results['errors'].append(f"{symbol}: {str(e)}")
            logger.error(f"  ✗ {symbol:6s}: {str(e)}")

    overall_time = time.time() - overall_start
    results['total_time'] = overall_time

    logger.info("")
    logger.info(f"Total time: {overall_time:.2f}s")
    logger.info(f"Success rate: {results['symbols_succeeded']}/{results['symbols_attempted']}")
    logger.info(f"Total bars fetched: {results['total_bars']:,}")
    logger.info(f"Average per symbol: {overall_time / len(symbols):.2f}s")

    return results


def print_comparison(full_day: Dict, last_5min: Dict):
    """Print comparison between both approaches."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    # Time comparison
    speedup = full_day['total_time'] / last_5min['total_time'] if last_5min['total_time'] > 0 else 0
    time_saved = full_day['total_time'] - last_5min['total_time']

    logger.info("Execution Time:")
    logger.info(f"  Full day fetch:   {full_day['total_time']:6.2f}s  (WITHOUT pre-fetching)")
    logger.info(f"  Last 5 min fetch: {last_5min['total_time']:6.2f}s  (WITH pre-fetching)")
    logger.info(f"  Time saved:       {time_saved:6.2f}s  ({speedup:.1f}x faster)")
    logger.info("")

    # Data volume comparison
    data_reduction = (1 - last_5min['total_bars'] / full_day['total_bars']) * 100 if full_day['total_bars'] > 0 else 0

    logger.info("Data Volume:")
    logger.info(f"  Full day bars:    {full_day['total_bars']:6,} bars")
    logger.info(f"  Last 5 min bars:  {last_5min['total_bars']:6,} bars")
    logger.info(f"  Reduction:        {data_reduction:6.1f}% fewer bars")
    logger.info("")

    # Critical window analysis
    logger.info("Critical 3:50 PM Window:")
    logger.info(f"  WITHOUT pre-fetch: {full_day['total_time']:.2f}s execution time")
    logger.info(f"  WITH pre-fetch:    {last_5min['total_time']:.2f}s execution time")
    logger.info("")

    # Risk assessment
    if full_day['total_time'] > 60:
        logger.warning(f"  ⚠️  Full day fetch takes >{full_day['total_time']:.0f}s (may miss 4:00 PM close!)")
    if full_day['total_time'] > 600:
        logger.error(f"  ❌  Full day fetch takes >{full_day['total_time']:.0f}s (execution window exceeded!)")

    if last_5min['total_time'] < 10:
        logger.success(f"  ✅  Last 5 min fetch completes in <10s (safe execution)")

    logger.info("")

    # Per-symbol analysis
    logger.info("Slowest Symbols (Full Day):")
    sorted_full = sorted(full_day['per_symbol_times'].items(), key=lambda x: x[1], reverse=True)[:5]
    for symbol, time_taken in sorted_full:
        logger.info(f"  {symbol:6s}: {time_taken:6.2f}s")

    logger.info("")
    logger.info("Slowest Symbols (Last 5 Min):")
    sorted_last = sorted(last_5min['per_symbol_times'].items(), key=lambda x: x[1], reverse=True)[:5]
    for symbol, time_taken in sorted_last:
        logger.info(f"  {symbol:6s}: {time_taken:6.2f}s")

    logger.info("")
    logger.info("=" * 80)

    # Recommendation
    if speedup > 10:
        logger.success(f"RECOMMENDATION: Pre-fetching provides significant benefit ({speedup:.1f}x faster)")
    elif speedup > 3:
        logger.info(f"RECOMMENDATION: Pre-fetching provides moderate benefit ({speedup:.1f}x faster)")
    else:
        logger.info(f"RECOMMENDATION: Pre-fetching provides minimal benefit ({speedup:.1f}x faster)")

    logger.info("=" * 80)


def main():
    """Run benchmark."""
    logger.info("=" * 80)
    logger.info("INTRADAY DATA FETCH BENCHMARK")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Initialize broker
        logger.info("Connecting to Alpaca (paper trading)...")
        broker = AlpacaBroker(paper=True)
        logger.success("✓ Connected to Alpaca")
        logger.info("")

        # Get OMR symbols
        symbols = ETFUniverse.OPTIMAL_OMR
        logger.info(f"Testing with {len(symbols)} OMR symbols:")
        logger.info(f"  {', '.join(symbols)}")
        logger.info("")

        # Run benchmarks
        full_day_results = benchmark_full_day_fetch(broker, symbols)
        time.sleep(2)  # Brief pause between tests
        last_5min_results = benchmark_last_5min_fetch(broker, symbols)

        # Print comparison
        print_comparison(full_day_results, last_5min_results)

        return 0

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
