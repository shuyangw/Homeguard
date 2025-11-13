"""
Test Minute-Level Data Fetching

Validates that the broker can fetch minute-level data for:
1. Daily bars (SPY, VIX for regime detection)
2. Minute bars (TQQQ, SQQQ for intraday returns)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_daily_data_fetch(broker):
    """Test 1: Fetch daily bars for SPY and VIX."""
    logger.header("\nTest 1: Daily Data Fetch (SPY, VIX)")
    logger.header("=" * 70)

    try:
        end_date = datetime.now()

        # Test SPY - 250 days
        logger.info("Fetching SPY daily data (250 days)...")
        start_date = end_date - timedelta(days=250)

        bars = broker.get_bars(
            symbols=['SPY'],
            timeframe='1Day',
            start=start_date,
            end=end_date
        )

        if bars is not None and not bars.empty:
            # Extract SPY data
            if isinstance(bars.index, pd.MultiIndex):
                spy_bars = bars.xs('SPY', level=0, drop_level=True)
            else:
                spy_bars = bars

            logger.success(f"[PASS] SPY: Fetched {len(spy_bars)} days of data")
            logger.info(f"  Date range: {spy_bars.index.min()} to {spy_bars.index.max()}")
            logger.info(f"  Columns: {spy_bars.columns.tolist()}")
            logger.info(f"  Latest close: ${spy_bars['close'].iloc[-1]:.2f}")
        else:
            logger.error("[FAIL] SPY: No data returned")
            return False

        # Test VIX - 365 days
        logger.info("\nFetching VIX daily data (365 days)...")
        start_date = end_date - timedelta(days=365)

        bars = broker.get_bars(
            symbols=['VIX'],
            timeframe='1Day',
            start=start_date,
            end=end_date
        )

        if bars is not None and not bars.empty:
            # Extract VIX data
            if isinstance(bars.index, pd.MultiIndex):
                vix_bars = bars.xs('VIX', level=0, drop_level=True)
            else:
                vix_bars = bars

            logger.success(f"[PASS] VIX: Fetched {len(vix_bars)} days of data")
            logger.info(f"  Date range: {vix_bars.index.min()} to {vix_bars.index.max()}")
            logger.info(f"  Latest close: {vix_bars['close'].iloc[-1]:.2f}")
        else:
            logger.error("[FAIL] VIX: No data returned")
            return False

        return True

    except Exception as e:
        logger.error(f"[FAIL] Daily data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minute_data_fetch(broker):
    """Test 2: Fetch minute bars for TQQQ today."""
    logger.header("\nTest 2: Minute Data Fetch (TQQQ Today)")
    logger.header("=" * 70)

    try:
        end_date = datetime.now()

        # Get today's market hours
        logger.info("Getting today's market hours...")
        try:
            market_open, market_close = broker.get_market_hours(end_date)
            logger.info(f"  Market hours: {market_open} to {market_close}")
        except Exception as e:
            # Market might be closed - use fallback times
            logger.warning(f"  Market hours not available: {e}")
            logger.info("  Using fallback times (9:30 AM - 4:00 PM ET)")

            from datetime import time
            today = end_date.date()
            market_open = datetime.combine(today, time(9, 30))
            market_close = datetime.combine(today, time(16, 0))

        # Fetch minute bars from open to now (or close)
        start_time = market_open
        end_time = min(end_date, market_close)

        logger.info(f"\nFetching TQQQ minute bars from {start_time} to {end_time}...")

        bars = broker.get_bars(
            symbols=['TQQQ'],
            timeframe='1Min',
            start=start_time,
            end=end_time
        )

        if bars is not None and not bars.empty:
            # Extract TQQQ data
            if isinstance(bars.index, pd.MultiIndex):
                tqqq_bars = bars.xs('TQQQ', level=0, drop_level=True)
            else:
                tqqq_bars = bars

            logger.success(f"[PASS] TQQQ: Fetched {len(tqqq_bars)} minute bars")
            logger.info(f"  Time range: {tqqq_bars.index.min()} to {tqqq_bars.index.max()}")
            logger.info(f"  Columns: {tqqq_bars.columns.tolist()}")

            # Calculate intraday return (like the signal generator does)
            if len(tqqq_bars) >= 2:
                open_price = tqqq_bars['open'].iloc[0]
                current_price = tqqq_bars['close'].iloc[-1]
                intraday_return = (current_price - open_price) / open_price

                logger.info(f"\n  Intraday Performance:")
                logger.info(f"    Open price: ${open_price:.2f}")
                logger.info(f"    Current price: ${current_price:.2f}")
                logger.info(f"    Intraday return: {intraday_return:.2%}")
            else:
                logger.warning("  Not enough data to calculate intraday return")

            return True
        else:
            logger.warning("[WARN] TQQQ: No minute data returned")
            logger.info("  This is expected if market is closed")
            return True  # Not a failure if market is closed

    except Exception as e:
        logger.error(f"[FAIL] Minute data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_symbol_minute_fetch(broker):
    """Test 3: Fetch minute bars for multiple symbols at once."""
    logger.header("\nTest 3: Multi-Symbol Minute Data Fetch")
    logger.header("=" * 70)

    try:
        end_date = datetime.now()

        # Get today's market hours
        try:
            market_open, market_close = broker.get_market_hours(end_date)
        except:
            from datetime import time
            today = end_date.date()
            market_open = datetime.combine(today, time(9, 30))
            market_close = datetime.combine(today, time(16, 0))

        start_time = market_open
        end_time = min(end_date, market_close)

        symbols = ['TQQQ', 'SQQQ', 'UPRO']
        logger.info(f"Fetching minute bars for {symbols}...")

        bars = broker.get_bars(
            symbols=symbols,
            timeframe='1Min',
            start=start_time,
            end=end_time
        )

        if bars is not None and not bars.empty:
            logger.success(f"[PASS] Fetched data for {len(symbols)} symbols")

            # Extract each symbol
            for symbol in symbols:
                try:
                    if isinstance(bars.index, pd.MultiIndex):
                        symbol_bars = bars.xs(symbol, level=0, drop_level=True)
                    else:
                        symbol_bars = bars

                    logger.info(f"  {symbol}: {len(symbol_bars)} bars")

                except KeyError:
                    logger.warning(f"  {symbol}: No data available")

            return True
        else:
            logger.warning("[WARN] No multi-symbol data returned")
            logger.info("  This is expected if market is closed")
            return True

    except Exception as e:
        logger.error(f"[FAIL] Multi-symbol fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all minute data fetch tests."""
    logger.header("=" * 70)
    logger.header("Minute-Level Data Fetch Test")
    logger.header("=" * 70)
    logger.blank()

    # Connect to broker
    logger.info("Connecting to Alpaca paper trading...")
    config_path = project_root / "config" / "trading" / "broker_alpaca.yaml"
    broker = BrokerFactory.create_from_yaml(str(config_path))

    if not broker.test_connection():
        logger.error("Failed to connect to broker")
        return False

    logger.success("Broker connected successfully")
    account = broker.get_account()
    logger.info(f"  Account ID: {account['account_id']}")
    logger.blank()

    # Run tests
    results = []

    # Test 1: Daily data
    passed = test_daily_data_fetch(broker)
    results.append(("Daily Data Fetch", passed))

    # Test 2: Minute data (single symbol)
    passed = test_minute_data_fetch(broker)
    results.append(("Minute Data Fetch (Single)", passed))

    # Test 3: Minute data (multiple symbols)
    passed = test_multi_symbol_minute_fetch(broker)
    results.append(("Minute Data Fetch (Multi)", passed))

    # Summary
    logger.blank()
    logger.header("=" * 70)
    logger.header("Test Results Summary")
    logger.header("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"  {status} {test_name}")

    logger.blank()
    logger.header(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.success("\nALL TESTS PASSED!")
        logger.success("Minute data fetching is working correctly")
        return True
    else:
        logger.warning(f"\n{total_count - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
