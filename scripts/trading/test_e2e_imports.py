"""
E2E Import Test for Trading Module

Tests all imports across the trading system to ensure no broken dependencies.

Usage:
    python scripts/trading/test_e2e_imports.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_core_packages():
    """Test core Python packages."""
    print("=" * 80)
    print("TESTING CORE PACKAGES")
    print("=" * 80)

    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("dotenv", "python-dotenv"),
        ("yaml", "pyyaml"),
        ("schedule", "schedule"),
        ("yfinance", "yfinance"),
        ("alpaca", "alpaca-py"),
        ("pickle", "built-in"),
        ("datetime", "built-in"),
        ("logging", "built-in"),
        ("csv", "built-in"),
    ]

    failed = []
    for module, package in packages:
        try:
            __import__(module)
            print(f"  [OK] {module:20s} ({package})")
        except ImportError as e:
            print(f"  [FAIL] {module:20s} ({package}) - {e}")
            failed.append((module, package))

    print()
    return failed


def test_trading_brokers():
    """Test broker module imports."""
    print("=" * 80)
    print("TESTING TRADING BROKERS")
    print("=" * 80)

    tests = [
        ("src.trading.brokers.broker_interface", ["BrokerInterface", "OrderSide", "OrderType", "OrderStatus", "TimeInForce"]),
        ("src.trading.brokers.alpaca_broker", ["AlpacaBroker"]),
        ("src.trading.brokers.broker_factory", ["BrokerFactory"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_trading_core():
    """Test core trading module imports."""
    print("=" * 80)
    print("TESTING TRADING CORE")
    print("=" * 80)

    tests = [
        ("src.trading.core.execution_engine", ["ExecutionEngine"]),
        ("src.trading.core.position_manager", ["PositionManager"]),
        ("src.trading.core.paper_trading_bot", ["PaperTradingBot"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_trading_adapters():
    """Test strategy adapter imports."""
    print("=" * 80)
    print("TESTING STRATEGY ADAPTERS")
    print("=" * 80)

    tests = [
        ("src.trading.adapters.strategy_adapter", ["StrategyAdapter"]),
        ("src.trading.adapters.omr_live_adapter", ["OMRLiveAdapter"]),
        ("src.trading.adapters.ma_live_adapter", ["MACrossoverLiveAdapter"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_trading_strategies():
    """Test live strategy imports."""
    print("=" * 80)
    print("TESTING LIVE STRATEGIES")
    print("=" * 80)

    tests = [
        ("src.trading.strategies.omr_live_strategy", ["OMRLiveStrategy"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_trading_utils():
    """Test trading utility imports."""
    print("=" * 80)
    print("TESTING TRADING UTILS")
    print("=" * 80)

    tests = [
        ("src.trading.utils.portfolio_health_check", ["PortfolioHealthChecker", "HealthCheckResult"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_advanced_strategies():
    """Test advanced strategy imports."""
    print("=" * 80)
    print("TESTING ADVANCED STRATEGIES")
    print("=" * 80)

    tests = [
        ("src.strategies.advanced.bayesian_reversion_model", ["BayesianReversionModel"]),
        ("src.strategies.advanced.market_regime_detector", ["MarketRegimeDetector"]),
        ("src.strategies.advanced.overnight_mean_reversion", ["OvernightMeanReversionStrategy"]),
        ("src.strategies.advanced.overnight_signal_generator", ["OvernightReversionSignals"]),
    ]

    failed = []
    for module_path, classes in tests:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if not hasattr(module, cls):
                    raise AttributeError(f"Class {cls} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def test_logger():
    """Test logger import."""
    print("=" * 80)
    print("TESTING LOGGER")
    print("=" * 80)

    tests = [
        ("src.utils.logger", ["logger", "TradingLogger", "CSVLogger"]),
    ]

    failed = []
    for module_path, items in tests:
        try:
            module = __import__(module_path, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    raise AttributeError(f"Item {item} not found in {module_path}")
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path} - {e}")
            failed.append((module_path, str(e)))

    print()
    return failed


def main():
    """Run all import tests."""
    print()
    print("=" * 80)
    print("E2E IMPORT TEST FOR TRADING MODULE")
    print("=" * 80)
    print()

    all_failures = []

    # Run all test suites
    all_failures.extend(test_core_packages())
    all_failures.extend(test_trading_brokers())
    all_failures.extend(test_trading_core())
    all_failures.extend(test_trading_adapters())
    all_failures.extend(test_trading_strategies())
    all_failures.extend(test_trading_utils())
    all_failures.extend(test_advanced_strategies())
    all_failures.extend(test_logger())

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if not all_failures:
        print()
        print("  [SUCCESS] All imports passed!")
        print()
        print("  Trading module is ready to use.")
        print()
        return 0
    else:
        print()
        print(f"  [FAILURE] {len(all_failures)} import(s) failed:")
        print()
        for item, error in all_failures:
            print(f"    - {item}")
            print(f"      Error: {error}")
        print()
        print("  Please install missing packages:")
        print("    pip install -r requirements.txt")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
