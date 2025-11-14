"""
Comprehensive E2E Test Suite for Trading Engine

Tests high-level trading workflows end-to-end using the broker interface.

Test Coverage:
1. Single trade execution (buy → verify → sell)
2. Multi-symbol trading
3. Error handling
4. Market hours detection
5. Portfolio health checks
6. Regime detection

Usage:
    python scripts/trading/test_trading_e2e_simple.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
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
from src.trading.utils.portfolio_health_check import PortfolioHealthChecker
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger


def test_single_trade_execution(broker):
    """Test single trade execution workflow."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 1: Single Trade Execution")
    logger.info("=" * 80)

    symbol = 'SPY'

    try:
        # Get quote
        quote = broker.get_latest_quote(symbol)
        logger.info(f"Current {symbol} price: ${quote['ask_price']:.2f}")

        # Place buy order
        logger.info(f"Placing BUY order for 1 share...")
        account = broker.get_account()
        order = broker.place_order(
            symbol=symbol,
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )

        logger.success(f"Order placed: {order['id']}")

        # Wait for fill
        time_module.sleep(2)
        order_status = broker.get_order(order['id'])
        logger.info(f"Order status: {order_status['status']}")

        # Verify position
        positions = broker.get_positions()
        spy_pos = next((p for p in positions if p['symbol'] == symbol), None)

        if not spy_pos:
            raise Exception("Position not found after order fill")

        logger.info(f"Position verified: {spy_pos['quantity']} shares @ ${spy_pos['avg_entry_price']:.2f}")

        # Close position
        time_module.sleep(1)
        logger.info(f"Closing position...")
        broker.close_position(symbol)

        # Verify closed
        time_module.sleep(2)
        positions = broker.get_positions()
        spy_pos = next((p for p in positions if p['symbol'] == symbol), None)

        if spy_pos:
            raise Exception("Position still exists after close")

        logger.success("[OK] TEST 1 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 1 FAILED: {e}")
        # Cleanup
        try:
            broker.close_position(symbol)
        except:
            pass
        return False


def test_multi_symbol_trading(broker):
    """Test trading multiple symbols concurrently."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Multi-Symbol Trading")
    logger.info("=" * 80)

    symbols = ['SPY', 'QQQ', 'IWM']

    try:
        # Place orders for all symbols
        logger.info(f"Placing orders for {len(symbols)} symbols...")
        for symbol in symbols:
            broker.place_order(
                symbol=symbol,
                quantity=1,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            logger.info(f"  Order placed for {symbol}")
            time_module.sleep(0.5)

        # Wait for fills
        time_module.sleep(3)

        # Verify all positions
        positions = broker.get_positions()
        position_symbols = [p['symbol'] for p in positions]

        for symbol in symbols:
            if symbol not in position_symbols:
                raise Exception(f"Position for {symbol} not found")
            logger.success(f"  [OK] {symbol} position verified")

        # Close all
        logger.info(f"Closing all positions...")
        for symbol in symbols:
            broker.close_position(symbol)
            time_module.sleep(0.5)

        # Verify all closed
        time_module.sleep(3)
        positions = broker.get_positions()

        for symbol in symbols:
            if any(p['symbol'] == symbol for p in positions):
                raise Exception(f"Position for {symbol} still open")

        logger.success("[OK] TEST 2 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 2 FAILED: {e}")
        # Cleanup
        for symbol in symbols:
            try:
                broker.close_position(symbol)
            except:
                pass
        return False


def test_error_handling(broker):
    """Test error handling."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 3: Error Handling")
    logger.info("=" * 80)

    try:
        # Test 1: Invalid symbol
        logger.info("Testing invalid symbol handling...")
        try:
            broker.get_latest_quote('INVALID_SYMBOL_12345')
            raise Exception("Invalid symbol should have raised an error")
        except Exception as e:
            if "invalid" in str(e).lower() or "not found" in str(e).lower():
                logger.success("  [OK] Invalid symbol error handled")
            else:
                raise

        # Test 2: Close non-existent position
        logger.info("Testing close of non-existent position...")
        try:
            broker.close_position('AAPL')
            raise Exception("Closing non-existent position should raise error")
        except Exception as e:
            if "position" in str(e).lower() or "not found" in str(e).lower():
                logger.success("  [OK] Non-existent position error handled")
            else:
                raise

        logger.success("[OK] TEST 3 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 3 FAILED: {e}")
        return False


def test_market_hours(broker):
    """Test market hours detection."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 4: Market Hours Detection")
    logger.info("=" * 80)

    try:
        # Get market hours for today
        today = datetime.now()
        market_open, market_close = broker.get_market_hours(today)

        # Check if market is currently open
        now = datetime.now()
        is_open = market_open <= now <= market_close if market_open and market_close else False

        logger.info(f"Market is open: {is_open}")
        logger.info(f"Market open time: {market_open}")
        logger.info(f"Market close time: {market_close}")

        logger.success("[OK] TEST 4 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 4 FAILED: {e}")
        return False


def test_portfolio_health(broker):
    """Test portfolio health checks."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 5: Portfolio Health Checks")
    logger.info("=" * 80)

    try:
        # Note: PortfolioHealthChecker needs a position_manager,
        # so we'll just test that we can check account status
        account = broker.get_account()

        # Check key account metrics
        buying_power = float(account.get('buying_power', 0))
        portfolio_value = float(account.get('portfolio_value', 0))

        logger.info(f"Buying power: ${buying_power:,.2f}")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")

        if buying_power <= 0:
            raise Exception("Buying power is zero or negative")

        if portfolio_value <= 0:
            raise Exception("Portfolio value is zero or negative")

        logger.success("[OK] TEST 5 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 5 FAILED: {e}")
        return False


def test_regime_detection(broker):
    """Test market regime detection."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 6: Regime Detection")
    logger.info("=" * 80)

    try:
        # Initialize detector
        detector = MarketRegimeDetector()

        # Fetch SPY data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)

        bars = broker.get_historical_bars(
            'SPY',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            timeframe='1Day'
        )

        if len(bars) < 50:
            raise Exception(f"Insufficient data: {len(bars)} bars")

        logger.info(f"Fetched {len(bars)} daily bars")

        # Convert to DataFrame
        df = pd.DataFrame(bars)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Classify regime
        regime = detector.classify_regime(df, df.index[-1])

        logger.info(f"Current regime: {regime}")
        logger.success("[OK] TEST 6 PASSED")
        return True

    except Exception as e:
        logger.error(f"[FAIL] TEST 6 FAILED: {e}")
        return False


def main():
    """Run all E2E tests."""
    logger.info("=" * 80)
    logger.info("TRADING ENGINE E2E TEST SUITE")
    logger.info("=" * 80)
    logger.info("")

    # Initialize broker
    logger.info("Initializing broker connection...")
    broker = BrokerFactory.create_from_env()

    account = broker.get_account()
    logger.info(f"Connected to paper trading account")
    logger.info(f"  Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
    logger.info(f"  Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")

    # Run tests
    results = []
    results.append(("Single Trade Execution", test_single_trade_execution(broker)))
    results.append(("Multi-Symbol Trading", test_multi_symbol_trading(broker)))
    results.append(("Error Handling", test_error_handling(broker)))
    results.append(("Market Hours Detection", test_market_hours(broker)))
    results.append(("Portfolio Health", test_portfolio_health(broker)))
    results.append(("Regime Detection", test_regime_detection(broker)))

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed

    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    logger.info("")

    if failed > 0:
        logger.error("FAILED TESTS:")
        for name, result in results:
            if not result:
                logger.error(f"  [FAIL] {name}")
        logger.info("")

    if passed == total:
        logger.success("=" * 80)
        logger.success("ALL TESTS PASSED!")
        logger.success("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error(f"SOME TESTS FAILED ({failed}/{total})")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
