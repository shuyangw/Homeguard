"""
Phase 1 Validation Script

Validates that all Phase 1 components are correctly implemented and working.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_file_structure() -> Tuple[bool, List[str]]:
    """Validate that all required files exist."""
    logger.info("Validating file structure...")

    required_files = [
        # Core implementation
        "src/trading/__init__.py",
        "src/trading/brokers/__init__.py",
        "src/trading/brokers/broker_interface.py",
        "src/trading/brokers/broker_factory.py",
        "src/trading/brokers/alpaca_broker.py",
        "src/trading/core/__init__.py",
        "src/trading/core/position_manager.py",

        # Configuration
        "config/trading/broker_alpaca.yaml",
        "config/trading/omr_trading_config.yaml",

        # Tests
        "tests/trading/__init__.py",
        "tests/trading/mock_broker.py",
        "tests/trading/test_broker_interface.py",
        "tests/trading/test_position_manager.py",

        # Scripts
        "scripts/trading/test_alpaca_connection.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False, missing_files

    logger.success(f"[OK] All {len(required_files)} required files exist")
    return True, []


def validate_imports() -> Tuple[bool, List[str]]:
    """Validate that all modules can be imported."""
    logger.info("Validating imports...")

    imports_to_test = [
        ("BrokerInterface", "src.trading.brokers.broker_interface"),
        ("OrderSide", "src.trading.brokers.broker_interface"),
        ("OrderType", "src.trading.brokers.broker_interface"),
        ("BrokerFactory", "src.trading.brokers.broker_factory"),
        ("AlpacaBroker", "src.trading.brokers.alpaca_broker"),
        ("PositionManager", "src.trading.core.position_manager"),
        ("MockBroker", "tests.trading.mock_broker"),
    ]

    failed_imports = []
    for name, module in imports_to_test:
        try:
            exec(f"from {module} import {name}")
            logger.info(f"  [OK] {module}.{name}")
        except Exception as e:
            logger.error(f"  [FAILED] {module}.{name}: {e}")
            failed_imports.append(f"{module}.{name}")

    if failed_imports:
        return False, failed_imports

    logger.success(f"[OK] All {len(imports_to_test)} imports successful")
    return True, []


def validate_broker_interface() -> Tuple[bool, str]:
    """Validate BrokerInterface implementation."""
    logger.info("Validating BrokerInterface...")

    try:
        from src.trading.brokers.broker_interface import BrokerInterface
        from abc import abstractmethod

        # Check that it's an abstract class
        required_methods = [
            'get_account', 'get_positions', 'get_position',
            'place_order', 'cancel_order', 'get_order', 'get_orders',
            'close_position', 'close_all_positions',
            'get_latest_quote', 'get_latest_trade', 'get_bars',
            'is_market_open', 'get_market_hours', 'test_connection'
        ]

        for method in required_methods:
            if not hasattr(BrokerInterface, method):
                return False, f"Missing method: {method}"

        logger.success(f"[OK] BrokerInterface has all {len(required_methods)} required methods")
        return True, ""

    except Exception as e:
        return False, str(e)


def validate_mock_broker() -> Tuple[bool, str]:
    """Validate MockBroker implementation."""
    logger.info("Validating MockBroker...")

    try:
        from tests.trading.mock_broker import MockBroker
        from src.trading.brokers.broker_interface import OrderSide, OrderType

        # Create mock broker
        broker = MockBroker(initial_cash=100000.0)

        # Test basic operations
        account = broker.get_account()
        assert account['cash'] == 100000.0

        # Test order placement
        broker.set_quote('TEST', bid=100.0, ask=100.05)
        order = broker.place_order('TEST', 10, OrderSide.BUY)
        assert order['symbol'] == 'TEST'

        # Test position
        position = broker.get_position('TEST')
        assert position is not None

        logger.success("[OK] MockBroker working correctly")
        return True, ""

    except Exception as e:
        return False, str(e)


def validate_position_manager() -> Tuple[bool, str]:
    """Validate PositionManager implementation."""
    logger.info("Validating PositionManager...")

    try:
        from src.trading.core.position_manager import PositionManager
        from datetime import datetime

        config = {
            'max_position_size_pct': 0.15,
            'max_concurrent_positions': 3,
            'max_total_exposure_pct': 0.45,
            'stop_loss_pct': -0.02,
        }

        manager = PositionManager(config)

        # Test adding position
        pos_id = manager.add_position(
            symbol='TEST',
            entry_price=100.0,
            qty=10,
            timestamp=datetime.now(),
            order_id='test_order'
        )

        assert len(manager.positions) == 1

        # Test P&L calculation
        pnl = manager.calculate_pnl(manager.positions[pos_id], 105.0)
        assert pnl == 50.0

        logger.success("[OK] PositionManager working correctly")
        return True, ""

    except Exception as e:
        return False, str(e)


def validate_broker_factory() -> Tuple[bool, str]:
    """Validate BrokerFactory implementation."""
    logger.info("Validating BrokerFactory...")

    try:
        from src.trading.brokers.broker_factory import BrokerFactory

        # Test listing supported brokers
        brokers = BrokerFactory.list_supported_brokers()
        assert 'alpaca' in brokers

        logger.success(f"[OK] BrokerFactory supports {len(brokers)} broker(s): {', '.join(brokers)}")
        return True, ""

    except Exception as e:
        return False, str(e)


def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate configuration files."""
    logger.info("Validating configuration files...")

    try:
        import yaml

        # Validate broker config
        broker_config_path = project_root / "config" / "trading" / "broker_alpaca.yaml"
        with open(broker_config_path) as f:
            broker_config = yaml.safe_load(f)

        assert 'broker' in broker_config
        assert broker_config['broker']['type'] == 'alpaca'
        logger.info("  [OK] broker_alpaca.yaml")

        # Validate strategy config
        strategy_config_path = project_root / "config" / "trading" / "omr_trading_config.yaml"
        with open(strategy_config_path) as f:
            strategy_config = yaml.safe_load(f)

        assert 'strategy' in strategy_config
        assert strategy_config['strategy']['name'] == 'OMR'
        logger.info("  [OK] omr_trading_config.yaml")

        logger.success("[OK] Configuration files valid")
        return True, []

    except Exception as e:
        return False, [str(e)]


def run_unit_tests() -> Tuple[bool, str]:
    """Run all unit tests."""
    logger.info("Running unit tests...")

    try:
        import pytest

        test_dir = project_root / "tests" / "trading"

        # Run tests with pytest
        result = pytest.main([
            str(test_dir),
            "-v",
            "--tb=short",
            "-q"
        ])

        if result == 0:
            logger.success("[OK] All unit tests passed")
            return True, ""
        else:
            return False, f"Tests failed with exit code {result}"

    except Exception as e:
        return False, str(e)


def count_lines_of_code() -> dict:
    """Count lines of code in Phase 1 implementation."""
    logger.info("Counting lines of code...")

    file_groups = {
        "Core Implementation": [
            "src/trading/brokers/broker_interface.py",
            "src/trading/brokers/broker_factory.py",
            "src/trading/brokers/alpaca_broker.py",
            "src/trading/core/position_manager.py",
        ],
        "Tests": [
            "tests/trading/mock_broker.py",
            "tests/trading/test_broker_interface.py",
            "tests/trading/test_position_manager.py",
        ],
        "Scripts": [
            "scripts/trading/test_alpaca_connection.py",
        ]
    }

    totals = {}
    for group, files in file_groups.items():
        total_lines = 0
        for file_path in files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
        totals[group] = total_lines

    return totals


def main():
    """Run all validation checks."""
    logger.header("=" * 70)
    logger.header("Phase 1 Validation")
    logger.header("=" * 70)
    logger.blank()

    all_passed = True

    # 1. File Structure
    passed, missing = validate_file_structure()
    if not passed:
        all_passed = False
        logger.error(f"File structure validation failed: {missing}")
    logger.blank()

    # 2. Imports
    passed, failed = validate_imports()
    if not passed:
        all_passed = False
        logger.error(f"Import validation failed: {failed}")
    logger.blank()

    # 3. BrokerInterface
    passed, error = validate_broker_interface()
    if not passed:
        all_passed = False
        logger.error(f"BrokerInterface validation failed: {error}")
    logger.blank()

    # 4. MockBroker
    passed, error = validate_mock_broker()
    if not passed:
        all_passed = False
        logger.error(f"MockBroker validation failed: {error}")
    logger.blank()

    # 5. PositionManager
    passed, error = validate_position_manager()
    if not passed:
        all_passed = False
        logger.error(f"PositionManager validation failed: {error}")
    logger.blank()

    # 6. BrokerFactory
    passed, error = validate_broker_factory()
    if not passed:
        all_passed = False
        logger.error(f"BrokerFactory validation failed: {error}")
    logger.blank()

    # 7. Configuration
    passed, errors = validate_configuration()
    if not passed:
        all_passed = False
        logger.error(f"Configuration validation failed: {errors}")
    logger.blank()

    # 8. Unit Tests
    passed, error = run_unit_tests()
    if not passed:
        all_passed = False
        logger.error(f"Unit tests failed: {error}")
    logger.blank()

    # 9. Lines of Code
    loc = count_lines_of_code()
    logger.info("Lines of Code:")
    for group, count in loc.items():
        logger.info(f"  {group}: {count:,} lines")
    logger.info(f"  Total: {sum(loc.values()):,} lines")
    logger.blank()

    # Final Summary
    logger.header("=" * 70)
    if all_passed:
        logger.success("PHASE 1 VALIDATION PASSED")
        logger.success("All components implemented correctly and working!")
    else:
        logger.error("PHASE 1 VALIDATION FAILED")
        logger.error("Some components have issues - see details above")
    logger.header("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
