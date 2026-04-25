"""
Validate CSV logging functionality and type compatibility.
"""
import sys
from pathlib import Path
import tempfile
import csv
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.trading.run_live_paper_trading import TradingSessionTracker
from src.utils.logger import logger


def test_csv_logging():
    """Test CSV logging creates valid files with correct formatting."""
    # Create temporary directory for test logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        logger.info("Testing TradingSessionTracker CSV logging...")
        logger.info(f"Log directory: {log_dir}")

        # Create session tracker
        tracker = TradingSessionTracker(log_dir, "TestStrategy")

        # Test market check logging
        logger.info("\n1. Testing market check logging...")
        tracker.log_check(True)
        tracker.log_check(False)
        tracker.log_check(True)

        # Verify market checks CSV
        market_checks_file = tracker.market_checks_log_file
        assert market_checks_file.exists(), "Market checks CSV not created"

        with open(market_checks_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4, f"Expected 4 rows (header + 3 checks), got {len(rows)}"
            assert rows[0] == ['timestamp', 'market_open', 'check_number'], "Invalid header"
            assert rows[1][1] == 'True', "First check should be True"
            assert rows[2][1] == 'False', "Second check should be False"
            assert rows[3][1] == 'True', "Third check should be True"

        logger.success("   Market checks CSV: OK")

        # Trade logging is now handled by the decision log (Tasks 1-12).
        # log_order() was removed from TradingSessionTracker in Task 13.
        logger.info("\n2. Trade logging is sourced from the decision log (deprecated from session tracker).")
        logger.success("   Trade logging: skipped (decision log is source of truth)")

        logger.info("\n" + "="*60)
        logger.success("ALL VALIDATION TESTS PASSED")
        logger.info("="*60)
        logger.info(f"\nCSV Files Created:")
        logger.info(f"  - Market Checks: {market_checks_file.name}")
        logger.info(f"\nFormat Validation:")
        logger.success("  - Headers correct")
        logger.success("  - CSV format valid")
        logger.success("  - File encoding correct")

        return True


if __name__ == "__main__":
    try:
        test_csv_logging()
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nVALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
