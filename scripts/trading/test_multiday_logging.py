"""
Test Multi-Day Buffered Logging

Tests the following scenarios:
1. Each trading day gets its own log file
2. Periodic flushing works for long-running sessions
3. Log file paths update correctly based on current date
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_trading_logger


def test_multiday_logging():
    """Test that each day gets its own log file."""

    print("\n" + "=" * 80)
    print("TEST 1: MULTI-DAY LOG FILE SEPARATION")
    print("=" * 80)
    print()

    # Create log directory
    log_dir = Path('logs/test_multiday')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing log files
    for f in log_dir.glob("*.log"):
        f.unlink()

    # Create trading logger with buffering
    logger = get_trading_logger('TestStrategy', log_dir, buffer_logs=True)

    # Day 1: Log some messages
    print("Day 1 (Today): Logging messages...")
    logger.info("Day 1: Market opened")
    logger.success("Day 1: Signal found")
    logger.success("Day 1: Order placed: TQQQ BUY 100")
    print(f"  Buffer contains {len(logger.log_buffer)} entries")
    print()

    # Flush Day 1 logs
    print("Day 1: Flushing logs at market close...")
    logger.flush_to_disk(reason="Market closed (4:00 PM ET)")

    # Check Day 1 log file
    day1_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}_TestStrategy.log"
    if day1_file.exists():
        print(f"[PASS] Day 1 log file created: {day1_file.name}")
        with open(day1_file, 'r') as f:
            lines = f.readlines()
            print(f"       Contains {len(lines)} lines")
    else:
        print(f"[FAIL] ERROR: Day 1 log file not found!")
    print()

    # Simulate Day 2 by mocking the date
    print("Day 2 (Tomorrow): Simulating next trading day...")

    # Log more messages (buffer should be empty after Day 1 flush)
    logger.info("Day 2: Market opened")
    logger.success("Day 2: New signal found")
    logger.success("Day 2: Order placed: UPRO BUY 150")
    print(f"  Buffer contains {len(logger.log_buffer)} entries")
    print()

    # Mock tomorrow's date for flushing
    tomorrow = datetime.now() + timedelta(days=1)
    with patch('src.utils.logger.datetime') as mock_datetime:
        mock_datetime.now.return_value = tomorrow
        mock_datetime.strftime = datetime.strftime  # Preserve strftime

        print("Day 2: Flushing logs at market close...")
        logger.flush_to_disk(reason="Market closed (4:00 PM ET)")

    # Check Day 2 log file
    day2_file = log_dir / f"{tomorrow.strftime('%Y%m%d')}_TestStrategy.log"
    if day2_file.exists():
        print(f"[PASS] Day 2 log file created: {day2_file.name}")
        with open(day2_file, 'r') as f:
            lines = f.readlines()
            print(f"       Contains {len(lines)} lines")
    else:
        print(f"[FAIL] ERROR: Day 2 log file not found!")
    print()

    # Verify both files exist
    log_files = list(log_dir.glob("*.log"))
    print(f"Total log files: {len(log_files)}")
    for f in log_files:
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
    print()

    if len(log_files) == 2:
        print("[PASS] TEST PASSED: Each day got its own log file!")
    else:
        print("[FAIL] TEST FAILED: Expected 2 log files!")
    print()


def test_periodic_flushing():
    """Test periodic flushing for long-running sessions."""

    print("\n" + "=" * 80)
    print("TEST 2: PERIODIC FLUSHING")
    print("=" * 80)
    print()

    # Create log directory
    log_dir = Path('logs/test_periodic')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing log files
    for f in log_dir.glob("*.log"):
        f.unlink()

    # Create trading logger with 1-second periodic flush (for testing)
    # In production, use flush_interval_hours=24 for daily flushing
    logger = get_trading_logger('TestStrategy', log_dir, buffer_logs=True, flush_interval_hours=1/3600)

    print("Creating logger with 1-second periodic flush interval (for testing)...")
    print()

    # Log some messages
    logger.info("Session started")
    logger.success("First trade executed")
    print(f"Buffer contains {len(logger.log_buffer)} entries")
    print(f"Should periodic flush? {logger.should_periodic_flush()}")
    print()

    # Wait 1.5 seconds
    print("Waiting 1.5 seconds...")
    time.sleep(1.5)

    # Check if periodic flush is needed
    should_flush = logger.should_periodic_flush()
    print(f"Should periodic flush? {should_flush}")

    if should_flush:
        print("[PASS] Periodic flush check working correctly!")
        logger.flush_to_disk(reason="Periodic flush (test)")
    else:
        print("[FAIL] ERROR: Periodic flush check failed!")
    print()

    # Log more messages after flush
    logger.info("Second batch of messages")
    logger.success("Another trade executed")
    print(f"Buffer contains {len(logger.log_buffer)} entries after flush")
    print()

    # Verify log file exists
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        print(f"[PASS] Log file created: {log_files[0].name}")
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            print(f"       Contains {len(lines)} lines from first flush")
    else:
        print("[FAIL] ERROR: No log file found!")
    print()


def test_no_periodic_flush_default():
    """Test that periodic flushing is disabled by default."""

    print("\n" + "=" * 80)
    print("TEST 3: NO PERIODIC FLUSH BY DEFAULT")
    print("=" * 80)
    print()

    # Create log directory
    log_dir = Path('logs/test_no_periodic')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create trading logger without periodic flushing (default)
    logger = get_trading_logger('TestStrategy', log_dir, buffer_logs=True)

    print("Creating logger without periodic flush interval (default)...")
    print()

    # Check periodic flush
    should_flush = logger.should_periodic_flush()
    print(f"Should periodic flush? {should_flush}")

    if not should_flush:
        print("[PASS] Periodic flush disabled by default (as expected)")
    else:
        print("[FAIL] ERROR: Periodic flush should be disabled by default!")
    print()


def main():
    """Run all tests."""

    print("\n" + "=" * 80)
    print("MULTI-DAY BUFFERED LOGGING TESTS")
    print("=" * 80)

    # Test 1: Multi-day log file separation
    test_multiday_logging()

    # Test 2: Periodic flushing
    test_periodic_flushing()

    # Test 3: No periodic flush by default
    test_no_periodic_flush_default()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
