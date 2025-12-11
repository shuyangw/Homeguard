"""
Diagnostic script to check streaming buffer status.

This script connects to the live data provider and reports which symbols
are actually receiving bar data from the WebSocket stream.

Usage:
    python scripts/debug_streaming_buffer.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import time
from dotenv import load_dotenv

from src.streaming import LiveDataProvider
from src.trading.config import load_omr_config
from src.utils.logger import logger

def main():
    """Main diagnostic function."""
    load_dotenv()

    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')
    streaming_feed = os.getenv('STREAMING_FEED', 'iex')

    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found")
        return 1

    logger.info("=" * 80)
    logger.info("STREAMING BUFFER DIAGNOSTIC")
    logger.info("=" * 80)

    # Load OMR symbols
    omr_config = load_omr_config()
    symbols = omr_config.symbols
    logger.info(f"Testing with {len(symbols)} OMR symbols:")
    logger.info(f"  {', '.join(symbols)}")
    logger.info("")

    # Create LiveDataProvider
    logger.info(f"Creating LiveDataProvider with {streaming_feed.upper()} feed...")
    provider = LiveDataProvider(
        api_key=api_key,
        secret_key=secret_key,
        feed=streaming_feed,
        max_bars_per_symbol=500,
        fallback_enabled=True
    )

    # Start streaming
    logger.info(f"Starting WebSocket and subscribing to {len(symbols)} symbols...")
    provider.start(symbols)
    logger.info("WebSocket started. Waiting for data...")
    logger.info("")

    # Wait for data to accumulate
    wait_time = 60  # Wait 1 minute
    logger.info(f"Waiting {wait_time} seconds for bars to accumulate...")
    for i in range(wait_time, 0, -10):
        logger.info(f"  {i} seconds remaining...")
        time.sleep(10)
    logger.info("")

    # Check buffer status
    logger.info("=" * 80)
    logger.info("BUFFER STATUS CHECK")
    logger.info("=" * 80)

    symbols_with_data = []
    symbols_without_data = []

    for symbol in symbols:
        bars_df = provider.get_bars(symbol)
        bar_count = len(bars_df)

        if bar_count > 0:
            symbols_with_data.append(symbol)
            latest_bar = bars_df.iloc[-1]
            logger.success(
                f"✓ {symbol}: {bar_count} bars "
                f"(latest: {latest_bar.name.strftime('%H:%M:%S')} @ ${latest_bar['close']:.2f})"
            )
        else:
            symbols_without_data.append(symbol)
            logger.warning(f"✗ {symbol}: NO DATA")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Symbols with data: {len(symbols_with_data)}/{len(symbols)}")
    logger.info(f"Symbols without data: {len(symbols_without_data)}/{len(symbols)}")

    if symbols_with_data:
        logger.info("")
        logger.success(f"Symbols receiving data: {', '.join(symbols_with_data)}")

    if symbols_without_data:
        logger.info("")
        logger.warning(f"Symbols NOT receiving data: {', '.join(symbols_without_data)}")
        logger.info("")
        logger.info("Possible reasons:")
        logger.info("  1. IEX feed doesn't include these symbols")
        logger.info("  2. No trades occurred on IEX exchange during wait period")
        logger.info("  3. WebSocket subscription failed silently")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check logs for WebSocket connection errors")
        logger.info("  2. Try longer wait time during market hours")
        logger.info("  3. Test with STREAMING_FEED=sip (requires paid subscription)")
        logger.info("  4. Use yfinance fallback for these symbols")

    # Cleanup
    logger.info("")
    logger.info("Stopping WebSocket...")
    provider.stop()
    logger.success("Diagnostic complete!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
