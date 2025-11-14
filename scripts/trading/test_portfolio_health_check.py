"""
Test Portfolio Health Check with Mock Broker Data

Simulates the live trading flow to verify portfolio health checks work correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.utils.logger import logger


class MockBroker:
    """Mock broker that returns dicts like AlpacaBroker."""

    def __init__(self, scenario: str = "healthy"):
        """
        Initialize mock broker with different scenarios.

        Args:
            scenario: "healthy", "low_buying_power", "max_positions", "stale_orders"
        """
        self.scenario = scenario

    def get_account(self):
        """Return account info as dict."""
        if self.scenario == "low_buying_power":
            return {
                'account_id': 'TEST123',
                'buying_power': 500.0,  # Below minimum of 1000
                'cash': 500.0,
                'portfolio_value': 5000.0,
                'equity': 5000.0,
                'currency': 'USD',
            }
        else:
            return {
                'account_id': 'TEST123',
                'buying_power': 25000.0,
                'cash': 10000.0,
                'portfolio_value': 30000.0,
                'equity': 30000.0,
                'currency': 'USD',
            }

    def get_positions(self):
        """Return positions as list of dicts."""
        if self.scenario == "max_positions":
            # Return 5 positions (max is 5)
            return [
                {
                    'symbol': f'STOCK{i}',
                    'quantity': 100,
                    'avg_entry_price': 50.0,
                    'current_price': 52.0,
                    'market_value': 5200.0,
                    'unrealized_pnl': 200.0,
                    'unrealized_pnl_pct': 0.04,
                    'side': 'long',
                    'created_at': datetime.now() - timedelta(hours=15)  # 15 hours ago (overnight)
                }
                for i in range(5)
            ]
        elif self.scenario == "stale_position":
            return [
                {
                    'symbol': 'TQQQ',
                    'quantity': 100,
                    'avg_entry_price': 45.0,
                    'current_price': 46.5,
                    'market_value': 4650.0,
                    'unrealized_pnl': 150.0,
                    'unrealized_pnl_pct': 0.033,
                    'side': 'long',
                    'created_at': datetime.now() - timedelta(hours=50)  # 50 hours (STALE!)
                }
            ]
        elif self.scenario == "healthy_overnight":
            return [
                {
                    'symbol': 'TQQQ',
                    'quantity': 100,
                    'avg_entry_price': 45.0,
                    'current_price': 46.5,
                    'market_value': 4650.0,
                    'unrealized_pnl': 150.0,
                    'unrealized_pnl_pct': 0.033,
                    'side': 'long',
                    'created_at': datetime.now() - timedelta(hours=15)  # 15 hours (good)
                },
                {
                    'symbol': 'SQQQ',
                    'quantity': 50,
                    'avg_entry_price': 12.0,
                    'current_price': 12.3,
                    'market_value': 615.0,
                    'unrealized_pnl': 15.0,
                    'unrealized_pnl_pct': 0.025,
                    'side': 'long',
                    'created_at': datetime.now() - timedelta(hours=15)
                }
            ]
        else:
            # No positions
            return []

    def get_open_orders(self):
        """Return open orders as list of dicts."""
        if self.scenario == "stale_orders":
            return [
                {
                    'order_id': 'ORDER123',
                    'symbol': 'SPY',
                    'quantity': 10,
                    'side': 'buy',
                    'order_type': 'limit',
                    'status': 'pending',
                    'limit_price': 450.0,
                    'created_at': datetime.now() - timedelta(hours=3)  # 3 hours old (STALE!)
                }
            ]
        else:
            # No pending orders
            return []


def test_scenario(scenario_name: str, broker: MockBroker, checker: PortfolioHealthChecker):
    """Test a specific scenario."""
    logger.info("\n" + "=" * 80)
    logger.info(f"TESTING SCENARIO: {scenario_name}")
    logger.info("=" * 80)

    result = checker.check_before_entry(
        required_capital=None,
        allow_existing_positions=True
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Warnings: {len(result.warnings)}")
    logger.info("=" * 80)

    return result


def test_exit_scenario(scenario_name: str, broker: MockBroker, checker: PortfolioHealthChecker):
    """Test exit scenario."""
    logger.info("\n" + "=" * 80)
    logger.info(f"TESTING EXIT SCENARIO: {scenario_name}")
    logger.info("=" * 80)

    result = checker.check_before_exit()

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Warnings: {len(result.warnings)}")
    logger.info("=" * 80)

    return result


def main():
    """Run all test scenarios."""
    logger.info("=" * 80)
    logger.info("PORTFOLIO HEALTH CHECK SIMULATION")
    logger.info("=" * 80)
    logger.info("Testing with mock broker data (dicts, not objects)")
    logger.info("")

    # Test 1: Healthy scenario (should PASS)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 1: Healthy Portfolio (should PASS)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="healthy")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        min_portfolio_value=5000.0,
        max_positions=5,
        max_position_age_hours=48
    )
    result = test_scenario("Healthy Portfolio", broker, checker)
    assert result.passed, "Healthy scenario should pass"
    logger.success("[PASS] Test 1 PASSED")

    # Test 2: Low buying power (should FAIL)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 2: Low Buying Power (should FAIL)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="low_buying_power")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        min_portfolio_value=5000.0,
        max_positions=5
    )
    result = test_scenario("Low Buying Power", broker, checker)
    assert not result.passed, "Low buying power should fail"
    assert len(result.errors) > 0, "Should have errors"
    logger.success("[PASS] Test 2 PASSED (correctly failed health check)")

    # Test 3: Max positions reached (should FAIL)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 3: Max Positions Reached (should FAIL)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="max_positions")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        max_positions=5
    )
    result = test_scenario("Max Positions", broker, checker)
    assert not result.passed, "Max positions should fail"
    logger.success("[PASS] Test 3 PASSED (correctly failed health check)")

    # Test 4: Stale orders (should PASS with warnings)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 4: Stale Orders (should PASS with warnings)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="stale_orders")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        max_positions=5
    )
    result = test_scenario("Stale Orders", broker, checker)
    assert result.passed, "Stale orders should pass (just warnings)"
    assert len(result.warnings) > 0, "Should have warnings"
    logger.success("[PASS] Test 4 PASSED (passed with warnings)")

    # Test 5: Exit check - healthy overnight positions (should PASS)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 5: Exit Check - Healthy Overnight Positions (should PASS)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="healthy_overnight")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        max_positions=5,
        max_position_age_hours=48
    )
    result = test_exit_scenario("Healthy Overnight", broker, checker)
    assert result.passed, "Healthy overnight positions should pass"
    logger.success("[PASS] Test 5 PASSED")

    # Test 6: Exit check - stale position (should FAIL)
    logger.info("\n" + "#" * 80)
    logger.info("# TEST 6: Exit Check - Stale Position (should FAIL)")
    logger.info("#" * 80)
    broker = MockBroker(scenario="stale_position")
    checker = PortfolioHealthChecker(
        broker=broker,
        min_buying_power=1000.0,
        max_positions=5,
        max_position_age_hours=48
    )
    result = test_exit_scenario("Stale Position", broker, checker)
    assert not result.passed, "Stale position should fail"
    assert len(result.errors) > 0, "Should have errors for 50h old position"
    logger.success("[PASS] Test 6 PASSED (correctly detected stale position)")

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 80)
    logger.success("Portfolio health check correctly handles dict-based broker responses")
    logger.success("All scenarios validated:")
    logger.success("  [OK] Healthy portfolio passes")
    logger.success("  [OK] Low buying power fails")
    logger.success("  [OK] Max positions fails")
    logger.success("  [OK] Stale orders warn but pass")
    logger.success("  [OK] Healthy overnight positions pass exit check")
    logger.success("  [OK] Stale positions fail exit check")
    logger.info("")
    logger.info("Ready for live trading!")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
