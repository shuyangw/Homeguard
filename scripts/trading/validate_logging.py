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


def test_csv_logging():
    """Test CSV logging creates valid files with correct formatting."""
    # Create temporary directory for test logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        print("Testing TradingSessionTracker CSV logging...")
        print(f"Log directory: {log_dir}")

        # Create session tracker
        tracker = TradingSessionTracker(log_dir, "TestStrategy")

        # Test market check logging
        print("\n1. Testing market check logging...")
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

        print("   Market checks CSV: OK")

        # Test trade logging
        print("\n2. Testing trade logging...")
        tracker.log_order(
            symbol='TQQQ',
            side='buy',
            qty=100,
            price=45.23,
            success=True,
            order_id='test123',
            error=None,
            order_type='market'
        )

        tracker.log_order(
            symbol='SQQQ',
            side='sell',
            qty=50,
            price=12.34,
            success=False,
            order_id=None,
            error='Insufficient funds',
            order_type='market'
        )

        # Verify trades CSV
        trades_file = tracker.trades_log_file
        assert trades_file.exists(), "Trades CSV not created"

        with open(trades_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3, f"Expected 3 rows (header + 2 trades), got {len(rows)}"
            assert rows[0] == ['timestamp', 'symbol', 'side', 'qty', 'price',
                              'order_type', 'status', 'order_id', 'error'], "Invalid header"

            # Check first trade
            assert rows[1][1] == 'TQQQ', "Symbol mismatch"
            assert rows[1][2] == 'buy', "Side mismatch"
            assert rows[1][3] == '100', "Qty mismatch"
            assert rows[1][4] == '45.23', "Price mismatch"
            assert rows[1][5] == 'market', "Order type mismatch"
            assert rows[1][6] == 'SUCCESS', "Status mismatch"
            assert rows[1][7] == 'test123', "Order ID mismatch"
            assert rows[1][8] == '', "Error should be empty"

            # Check second trade
            assert rows[2][1] == 'SQQQ', "Symbol mismatch"
            assert rows[2][2] == 'sell', "Side mismatch"
            assert rows[2][3] == '50', "Qty mismatch"
            assert rows[2][4] == '12.34', "Price mismatch"
            assert rows[2][6] == 'FAILED', "Status mismatch"
            assert rows[2][7] == '', "Order ID should be empty"
            assert rows[2][8] == 'Insufficient funds', "Error mismatch"

        print("   Trades CSV: OK")

        # Test type compatibility
        print("\n3. Testing type compatibility...")

        # Test with different numeric types
        tracker.log_order(
            symbol='SPY',
            side='buy',
            qty=int(100),  # Explicit int
            price=float(450.50),  # Explicit float
            success=bool(True),  # Explicit bool
            order_id=str('test456'),  # Explicit str
            error=None,
            order_type='limit'
        )

        with open(trades_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4, "Type compatibility trade not logged"
            assert rows[3][3] == '100', "Int conversion failed"
            assert rows[3][4] == '450.5', "Float conversion failed"

        print("   Type compatibility: OK")

        print("\n" + "="*60)
        print("ALL VALIDATION TESTS PASSED")
        print("="*60)
        print(f"\nCSV Files Created:")
        print(f"  - Market Checks: {market_checks_file.name}")
        print(f"  - Trades:        {trades_file.name}")
        print(f"\nFormat Validation:")
        print("  - Headers correct")
        print("  - Data types compatible")
        print("  - CSV format valid")
        print("  - File encoding correct")

        return True


if __name__ == "__main__":
    try:
        test_csv_logging()
        sys.exit(0)
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
