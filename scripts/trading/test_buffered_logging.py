"""
Test Buffered Logging Functionality

Demonstrates how the TradingLogger buffers logs during the trading day
and flushes them to disk at market close.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_trading_logger

def main():
    """Test buffered logging."""

    # Create log directory
    log_dir = Path('logs/test_buffered')
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TESTING BUFFERED LOGGING")
    print("="*80)
    print(f"Log directory: {log_dir}")
    print()

    # Create trading logger with buffering enabled
    logger = get_trading_logger('TestStrategy', log_dir, buffer_logs=True)

    print("Logging messages (buffered in memory)...")
    print("-" * 80)

    # Simulate trading day activity
    logger.info("Trading day started")
    logger.separator()
    logger.blank()

    logger.info("Checking market status...")
    logger.success("Market is open")
    logger.blank()

    logger.info("Generating signals...")
    logger.success("Found 3 signals")
    logger.blank()

    logger.info("Placing orders...")
    logger.success("Order placed: TQQQ BUY 100 @ $45.23")
    logger.success("Order placed: UPRO BUY 150 @ $55.67")
    logger.warning("Order rejected: SOXL BUY 200 @ $30.12 (insufficient buying power)")
    logger.blank()

    logger.error("Connection timeout while fetching data")
    logger.info("Retrying...")
    logger.success("Connection restored")
    logger.blank()

    logger.info("Market check complete")
    logger.separator()

    print()
    print("All messages logged to console and buffered in memory.")
    print(f"Buffer contains {len(logger.log_buffer)} log entries")
    print()

    # Show what's in the buffer (first 5 entries)
    print("Buffer preview (first 5 entries):")
    print("-" * 80)
    for entry in logger.log_buffer[:5]:
        print(entry)
    print("...")
    print()

    # Simulate market close
    print("Simulating market close (4:00 PM ET)...")
    print("-" * 80)
    logger.flush_to_disk(reason="Market closed (4:00 PM ET)")
    print()

    # Verify file was created
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        log_file = log_files[0]
        print(f"Log file created: {log_file}")
        print(f"File size: {log_file.stat().st_size} bytes")
        print()

        # Show file contents
        print("Log file contents:")
        print("-" * 80)
        with open(log_file, 'r') as f:
            print(f.read())
        print("-" * 80)
    else:
        print("ERROR: No log file found!")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()

if __name__ == "__main__":
    main()
