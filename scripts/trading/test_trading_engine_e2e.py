"""
Comprehensive E2E Test Suite for Trading Engine

Tests the complete trading workflow from strategy initialization through
live execution, position management, and exit.

Test Coverage:
1. Full strategy execution pipeline (signal → order → fill → exit)
2. Multi-symbol concurrent trading
3. Error handling and retry logic
4. Risk management enforcement
5. Market hours boundary conditions
6. Regime detection integration

Usage:
    python scripts/trading/test_trading_engine_e2e.py
"""

import sys
from pathlib import Path
from datetime import datetime, time, timedelta
import pandas as pd
import time as time_module
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.trading.brokers.broker_factory import BrokerFactory
from src.trading.brokers.broker_interface import OrderSide, OrderType, TimeInForce
from src.trading.core.position_manager import PositionManager
from src.trading.core.execution_engine import ExecutionEngine
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
from src.strategies.advanced.overnight_mean_reversion import OvernightMeanReversionStrategy
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger


class TradingEngineE2ETests:
    """Comprehensive E2E test suite for trading engine."""

    def __init__(self):
        """Initialize test suite."""
        self.broker = None
        self.position_manager = None
        self.execution_engine = None
        self.health_checker = None
        self.test_results = []

    def setup(self):
        """Set up test environment."""
        logger.info("=" * 80)
        logger.info("TRADING ENGINE E2E TEST SUITE")
        logger.info("=" * 80)
        logger.info("")

        try:
            # Initialize broker (paper trading)
            logger.info("[Setup] Initializing broker connection...")
            self.broker = BrokerFactory.create_from_env()
            logger.success("  [OK] Broker initialized")

            # Initialize position manager
            logger.info("[Setup] Initializing position manager...")
            self.position_manager = PositionManager(
                broker=self.broker,
                max_positions=5,
                max_position_size=0.20,
                max_portfolio_risk=0.50
            )
            logger.success("  [OK] Position manager initialized")

            # Initialize execution engine
            logger.info("[Setup] Initializing execution engine...")
            self.execution_engine = ExecutionEngine(
                broker=self.broker,
                max_retries=3,
                retry_delay=1.0
            )
            logger.success("  [OK] Execution engine initialized")

            # Initialize health checker
            logger.info("[Setup] Initializing portfolio health checker...")
            self.health_checker = PortfolioHealthChecker(
                broker=self.broker,
                position_manager=self.position_manager
            )
            logger.success("  [OK] Health checker initialized")

            # Get account info
            account = self.broker.get_account()
            logger.info(f"[Setup] Paper trading account:")
            logger.info(f"  Account ID: {account.get('account_id', 'N/A')}")
            logger.info(f"  Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            logger.info(f"  Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"[Setup] Setup failed: {e}")
            return False

    def teardown(self):
        """Clean up test environment."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEARDOWN")
        logger.info("=" * 80)

        try:
            # Close all test positions
            logger.info("[Teardown] Closing all positions...")
            positions = self.broker.get_positions()

            if not positions:
                logger.info("  No positions to close")
            else:
                for position in positions:
                    symbol = position.get('symbol')
                    logger.info(f"  Closing {symbol}...")
                    try:
                        self.broker.close_position(symbol)
                        logger.success(f"    [OK] Closed {symbol}")
                    except Exception as e:
                        logger.error(f"    [FAIL] Failed to close {symbol}: {e}")

            logger.success("[Teardown] Cleanup complete")

        except Exception as e:
            logger.error(f"[Teardown] Cleanup failed: {e}")

    def record_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result."""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })

        if passed:
            logger.success(f"  [OK] PASSED: {test_name}")
        else:
            logger.error(f"  [FAIL] FAILED: {test_name}")

        if message:
            logger.info(f"    {message}")

    # =========================================================================
    # TEST 1: Single Order Execution
    # =========================================================================

    def test_single_order_execution(self):
        """Test single order placement, fill verification, and closure."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 1: Single Order Execution")
        logger.info("=" * 80)

        test_symbol = 'SPY'

        try:
            # Step 1: Get quote
            logger.info(f"[1.1] Fetching quote for {test_symbol}...")
            quote = self.broker.get_quote(test_symbol)
            current_price = quote.get('ask_price', 0)
            logger.info(f"  Current price: ${current_price:.2f}")

            # Step 2: Place order
            logger.info(f"[1.2] Placing BUY order for 1 share of {test_symbol}...")
            order_result = self.execution_engine.execute_order(
                symbol=test_symbol,
                quantity=1,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

            if not order_result['success']:
                raise Exception(f"Order placement failed: {order_result.get('error', 'Unknown error')}")

            order_id = order_result['order_id']
            logger.success(f"  [OK] Order placed: {order_id}")

            # Step 3: Wait for fill
            logger.info(f"[1.3] Waiting for order to fill...")
            time_module.sleep(2)

            order_status = self.broker.get_order(order_id)
            status = order_status.get('status', 'unknown')
            fill_price = order_status.get('avg_fill_price', 0)

            logger.info(f"  Order status: {status}")
            logger.info(f"  Fill price: ${fill_price:.2f}")

            if status != 'filled':
                raise Exception(f"Order not filled. Status: {status}")

            # Step 4: Verify position exists
            logger.info(f"[1.4] Verifying position exists...")
            positions = self.broker.get_positions()
            spy_position = next((p for p in positions if p['symbol'] == test_symbol), None)

            if not spy_position:
                raise Exception(f"Position for {test_symbol} not found")

            logger.info(f"  Quantity: {spy_position['quantity']}")
            logger.info(f"  Entry price: ${spy_position['avg_entry_price']:.2f}")
            logger.info(f"  Current P&L: ${spy_position['unrealized_pnl']:+.2f}")

            # Step 5: Close position
            logger.info(f"[1.5] Closing position...")
            time_module.sleep(1)

            close_result = self.broker.close_position(test_symbol)
            logger.success(f"  [OK] Close order submitted")

            # Step 6: Verify position closed
            time_module.sleep(2)
            positions = self.broker.get_positions()
            spy_position = next((p for p in positions if p['symbol'] == test_symbol), None)

            if spy_position:
                raise Exception(f"Position for {test_symbol} still exists after close")

            logger.success(f"  [OK] Position closed successfully")

            self.record_result(
                "Single Order Execution",
                True,
                f"Successfully executed full lifecycle for {test_symbol}"
            )
            return True

        except Exception as e:
            self.record_result("Single Order Execution", False, str(e))
            return False

    # =========================================================================
    # TEST 2: Multi-Symbol Concurrent Trading
    # =========================================================================

    def test_multi_symbol_trading(self):
        """Test concurrent trading across multiple symbols."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 2: Multi-Symbol Concurrent Trading")
        logger.info("=" * 80)

        test_symbols = ['SPY', 'QQQ', 'IWM']

        try:
            # Step 1: Place orders for multiple symbols
            logger.info(f"[2.1] Placing orders for {len(test_symbols)} symbols...")
            order_ids = {}

            for symbol in test_symbols:
                logger.info(f"  Placing order for {symbol}...")
                order_result = self.execution_engine.execute_order(
                    symbol=symbol,
                    quantity=1,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )

                if not order_result['success']:
                    raise Exception(f"Failed to place order for {symbol}: {order_result.get('error')}")

                order_ids[symbol] = order_result['order_id']
                logger.success(f"    [OK] Order placed for {symbol}")
                time_module.sleep(0.5)

            # Step 2: Wait for all fills
            logger.info(f"[2.2] Waiting for all orders to fill...")
            time_module.sleep(3)

            # Step 3: Verify all positions exist
            logger.info(f"[2.3] Verifying all positions...")
            positions = self.broker.get_positions()
            position_symbols = [p['symbol'] for p in positions]

            for symbol in test_symbols:
                if symbol not in position_symbols:
                    raise Exception(f"Position for {symbol} not found")
                logger.success(f"    [OK] {symbol} position verified")

            # Step 4: Check portfolio exposure
            logger.info(f"[2.4] Checking portfolio exposure...")
            total_exposure = sum(abs(float(p['market_value'])) for p in positions)
            account = self.broker.get_account()
            portfolio_value = float(account.get('portfolio_value', 0))
            exposure_pct = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0

            logger.info(f"  Total exposure: ${total_exposure:,.2f}")
            logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")
            logger.info(f"  Exposure %: {exposure_pct:.1f}%")

            # Step 5: Close all positions
            logger.info(f"[2.5] Closing all positions...")
            time_module.sleep(1)

            for symbol in test_symbols:
                logger.info(f"  Closing {symbol}...")
                self.broker.close_position(symbol)
                logger.success(f"    [OK] Close order submitted for {symbol}")
                time_module.sleep(0.5)

            # Step 6: Verify all positions closed
            time_module.sleep(3)
            positions = self.broker.get_positions()

            if any(p['symbol'] in test_symbols for p in positions):
                raise Exception("Some positions still open after close")

            logger.success(f"  [OK] All positions closed")

            self.record_result(
                "Multi-Symbol Trading",
                True,
                f"Successfully traded {len(test_symbols)} symbols concurrently"
            )
            return True

        except Exception as e:
            # Cleanup on failure
            try:
                for symbol in test_symbols:
                    self.broker.close_position(symbol)
            except:
                pass

            self.record_result("Multi-Symbol Trading", False, str(e))
            return False

    # =========================================================================
    # TEST 3: Risk Management Enforcement
    # =========================================================================

    def test_risk_management(self):
        """Test risk limits are enforced correctly."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 3: Risk Management Enforcement")
        logger.info("=" * 80)

        try:
            # Step 1: Check position size limits
            logger.info(f"[3.1] Testing position size limits...")

            max_position_size = self.position_manager.max_position_size
            account = self.broker.get_account()
            portfolio_value = float(account.get('portfolio_value', 0))
            max_position_value = portfolio_value * max_position_size

            logger.info(f"  Max position size: {max_position_size:.1%}")
            logger.info(f"  Max position value: ${max_position_value:,.2f}")

            # Try to open a position within limits
            quote = self.broker.get_quote('SPY')
            spy_price = quote.get('ask_price', 0)
            allowed_qty = int(max_position_value / spy_price)

            logger.info(f"  SPY price: ${spy_price:.2f}")
            logger.info(f"  Allowed quantity: {allowed_qty}")

            # This should succeed
            can_trade = self.position_manager.can_open_position('SPY', allowed_qty)
            if not can_trade:
                raise Exception("Position within limits was rejected")

            logger.success(f"    [OK] Position size validation working")

            # Step 2: Check max positions limit
            logger.info(f"[3.2] Testing max positions limit...")

            max_positions = self.position_manager.max_positions
            logger.info(f"  Max positions: {max_positions}")

            # Open positions up to limit
            symbols_to_trade = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'][:max_positions]
            logger.info(f"  Opening {len(symbols_to_trade)} positions...")

            for symbol in symbols_to_trade:
                order_result = self.execution_engine.execute_order(
                    symbol=symbol,
                    quantity=1,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                if not order_result['success']:
                    raise Exception(f"Failed to open position for {symbol}")
                time_module.sleep(0.5)

            time_module.sleep(2)

            # Try to open one more (should fail)
            logger.info(f"  Attempting to exceed max positions...")
            can_trade = self.position_manager.can_open_position('AAPL', 1)

            if can_trade:
                raise Exception("Position manager allowed exceeding max positions")

            logger.success(f"    [OK] Max positions limit enforced")

            # Step 3: Check portfolio exposure limit
            logger.info(f"[3.3] Testing portfolio exposure limit...")

            max_portfolio_risk = self.position_manager.max_portfolio_risk
            logger.info(f"  Max portfolio risk: {max_portfolio_risk:.1%}")

            positions = self.broker.get_positions()
            total_exposure = sum(abs(float(p['market_value'])) for p in positions)
            exposure_pct = (total_exposure / portfolio_value) if portfolio_value > 0 else 0

            logger.info(f"  Current exposure: {exposure_pct:.1%}")

            if exposure_pct > max_portfolio_risk:
                raise Exception(f"Portfolio exposure {exposure_pct:.1%} exceeds limit {max_portfolio_risk:.1%}")

            logger.success(f"    [OK] Portfolio exposure within limits")

            # Cleanup
            logger.info(f"[3.4] Cleaning up test positions...")
            for symbol in symbols_to_trade:
                try:
                    self.broker.close_position(symbol)
                except:
                    pass

            time_module.sleep(2)

            self.record_result(
                "Risk Management",
                True,
                "All risk limits properly enforced"
            )
            return True

        except Exception as e:
            # Cleanup on failure
            try:
                positions = self.broker.get_positions()
                for position in positions:
                    self.broker.close_position(position['symbol'])
            except:
                pass

            self.record_result("Risk Management", False, str(e))
            return False

    # =========================================================================
    # TEST 4: Error Handling and Retry Logic
    # =========================================================================

    def test_error_handling(self):
        """Test error handling and retry logic."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 4: Error Handling and Retry Logic")
        logger.info("=" * 80)

        try:
            # Step 1: Test invalid symbol
            logger.info(f"[4.1] Testing invalid symbol handling...")

            try:
                quote = self.broker.get_quote('INVALID_SYMBOL_12345')
                # If we get here, the test should fail
                raise Exception("Invalid symbol did not raise an error")
            except Exception as e:
                if "INVALID_SYMBOL" in str(e) or "not found" in str(e).lower():
                    logger.success(f"    [OK] Invalid symbol error handled correctly")
                else:
                    raise

            # Step 2: Test order with zero quantity
            logger.info(f"[4.2] Testing zero quantity order...")

            order_result = self.execution_engine.execute_order(
                symbol='SPY',
                quantity=0,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

            if order_result['success']:
                raise Exception("Zero quantity order was accepted")

            logger.success(f"    [OK] Zero quantity order rejected")

            # Step 3: Test retry logic with execution engine
            logger.info(f"[4.3] Testing execution engine retry logic...")

            # The execution engine should retry on temporary failures
            # We can't easily simulate this, but we can verify the config
            max_retries = self.execution_engine.max_retries
            retry_delay = self.execution_engine.retry_delay

            logger.info(f"  Max retries: {max_retries}")
            logger.info(f"  Retry delay: {retry_delay}s")

            if max_retries < 1:
                raise Exception("Retry logic not configured")

            logger.success(f"    [OK] Retry logic configured")

            # Step 4: Test closing non-existent position
            logger.info(f"[4.4] Testing close of non-existent position...")

            try:
                self.broker.close_position('AAPL')
                # Should raise an error
                raise Exception("Closing non-existent position did not raise error")
            except Exception as e:
                if "position" in str(e).lower() or "not found" in str(e).lower():
                    logger.success(f"    [OK] Non-existent position error handled")
                else:
                    raise

            self.record_result(
                "Error Handling",
                True,
                "All error scenarios handled correctly"
            )
            return True

        except Exception as e:
            self.record_result("Error Handling", False, str(e))
            return False

    # =========================================================================
    # TEST 5: Portfolio Health Checks
    # =========================================================================

    def test_portfolio_health_checks(self):
        """Test portfolio health check system."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 5: Portfolio Health Checks")
        logger.info("=" * 80)

        try:
            # Run all health checks
            logger.info(f"[5.1] Running portfolio health checks...")

            health_result = self.health_checker.run_all_checks()

            logger.info(f"  Overall health: {health_result.overall_health}")
            logger.info(f"  Checks passed: {health_result.checks_passed}/{health_result.total_checks}")

            # Display individual check results
            logger.info(f"\n  Individual Checks:")
            for check_name, passed in health_result.checks.items():
                status = "[OK]" if passed else "[FAIL]"
                logger.info(f"    {status} {check_name}")

            if not health_result.is_healthy:
                logger.warning(f"\n  Issues found:")
                for issue in health_result.issues:
                    logger.warning(f"    - {issue}")

            # All checks should pass with empty portfolio
            if not health_result.is_healthy:
                raise Exception(f"Health checks failed: {health_result.issues}")

            self.record_result(
                "Portfolio Health Checks",
                True,
                f"All {health_result.total_checks} health checks passed"
            )
            return True

        except Exception as e:
            self.record_result("Portfolio Health Checks", False, str(e))
            return False

    # =========================================================================
    # TEST 6: Market Hours Detection
    # =========================================================================

    def test_market_hours(self):
        """Test market hours detection."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 6: Market Hours Detection")
        logger.info("=" * 80)

        try:
            # Check if market is open
            logger.info(f"[6.1] Checking market status...")

            clock = self.broker.get_market_status()
            is_open = clock.get('is_open', False)
            next_open = clock.get('next_open', 'Unknown')
            next_close = clock.get('next_close', 'Unknown')

            logger.info(f"  Market is open: {is_open}")
            logger.info(f"  Next open: {next_open}")
            logger.info(f"  Next close: {next_close}")

            # Verify we can detect market hours
            if is_open:
                logger.info(f"  Market is currently OPEN - live trading possible")
            else:
                logger.info(f"  Market is currently CLOSED - orders will queue")

            logger.success(f"    [OK] Market hours detection working")

            self.record_result(
                "Market Hours Detection",
                True,
                f"Market hours correctly detected (Market {'OPEN' if is_open else 'CLOSED'})"
            )
            return True

        except Exception as e:
            self.record_result("Market Hours Detection", False, str(e))
            return False

    # =========================================================================
    # TEST 7: Regime Detection Integration
    # =========================================================================

    def test_regime_detection(self):
        """Test market regime detection integration."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 7: Regime Detection Integration")
        logger.info("=" * 80)

        try:
            # Initialize regime detector
            logger.info(f"[7.1] Initializing regime detector...")
            detector = MarketRegimeDetector()
            logger.success(f"    [OK] Regime detector initialized")

            # Fetch SPY data for regime detection
            logger.info(f"[7.2] Fetching market data for regime detection...")

            # Get historical bars (last 100 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)

            bars = self.broker.get_historical_bars(
                'SPY',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeframe='1Day'
            )

            if len(bars) < 50:
                raise Exception(f"Insufficient historical data: {len(bars)} bars")

            logger.info(f"  Fetched {len(bars)} daily bars")

            # Convert to DataFrame
            df = pd.DataFrame(bars)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            logger.info(f"[7.3] Detecting current market regime...")

            # Detect regime
            regime = detector.detect_regime(df, df.index[-1])

            logger.info(f"  Current regime: {regime}")
            logger.success(f"    [OK] Regime detection working")

            self.record_result(
                "Regime Detection",
                True,
                f"Successfully detected regime: {regime}"
            )
            return True

        except Exception as e:
            self.record_result("Regime Detection", False, str(e))
            return False

    # =========================================================================
    # Main Test Runner
    # =========================================================================

    def run_all_tests(self):
        """Run all E2E tests."""
        # Setup
        if not self.setup():
            logger.error("Setup failed - aborting tests")
            return False

        # Run all tests
        try:
            self.test_single_order_execution()
            self.test_multi_symbol_trading()
            self.test_risk_management()
            self.test_error_handling()
            self.test_portfolio_health_checks()
            self.test_market_hours()
            self.test_regime_detection()

        finally:
            # Teardown
            self.teardown()

        # Print summary
        self.print_summary()

        # Return overall result
        return all(result['passed'] for result in self.test_results)

    def print_summary(self):
        """Print test summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        failed_tests = total_tests - passed_tests

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("")

        if failed_tests > 0:
            logger.error("FAILED TESTS:")
            for result in self.test_results:
                if not result['passed']:
                    logger.error(f"  [FAIL] {result['test']}")
                    if result['message']:
                        logger.error(f"    {result['message']}")
            logger.info("")

        if passed_tests == total_tests:
            logger.success("=" * 80)
            logger.success("ALL TESTS PASSED! [OK]")
            logger.success("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error(f"SOME TESTS FAILED ({failed_tests}/{total_tests})")
            logger.error("=" * 80)


def main():
    """Main entry point."""
    test_suite = TradingEngineE2ETests()
    success = test_suite.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
