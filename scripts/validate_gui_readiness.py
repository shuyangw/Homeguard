"""
Validate that the GUI is ready to run out-of-the-box.

Checks:
1. Database exists and has data
2. Strategies are loadable
3. Backtesting engine works
4. GUI components can be instantiated
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import logger

def check_database():
    """Check if database exists and has data."""
    try:
        from backtesting.engine.data_loader import DataLoader
        loader = DataLoader()

        # Try to get date range for a common symbol
        date_range = loader.get_date_range('AAPL')  # Fixed: pass string not list

        if date_range:
            logger.success(f"[OK] Database accessible with data from {date_range[0]} to {date_range[1]}")
            return True
        else:
            logger.warning("[WARN] Database accessible but no data for AAPL")
            return False

    except Exception as e:
        logger.error(f"[FAIL] Database check failed: {str(e)}")
        return False


def check_strategies():
    """Check if strategies can be loaded."""
    try:
        from strategies import get_all_strategies
        strategies = get_all_strategies()

        if strategies:
            logger.success(f"[OK] Loaded {len(strategies)} strategies")
            for name in list(strategies.keys())[:3]:
                logger.info(f"  - {name}")
            return True
        else:
            logger.error("[FAIL] No strategies found")
            return False

    except Exception as e:
        logger.error(f"[FAIL] Strategy loading failed: {str(e)}")
        return False


def check_backtest_engine():
    """Check if backtest engine works."""
    try:
        from backtesting.engine.backtest_engine import BacktestEngine
        from strategies import MovingAverageCrossover

        engine = BacktestEngine(
            strategy=MovingAverageCrossover(),
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-31',  # Short test period
            initial_capital=100000,
            fees=0.0
        )

        logger.success("✓ Backtest engine instantiates successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Backtest engine check failed: {e}")
        return False


def check_gui_components():
    """Check if GUI components can be instantiated."""
    try:
        from gui.views.setup_view import SetupView
        from gui.views.run_view import RunView

        # Try to create views (without page)
        setup_view = SetupView(on_run_clicked=lambda config: None)
        run_view = RunView()

        logger.success("✓ GUI components instantiate successfully")
        return True

    except Exception as e:
        logger.error(f"✗ GUI component check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.blank()
    logger.separator("=")
    logger.header("GUI READINESS VALIDATION")
    logger.separator("=")
    logger.blank()

    checks = [
        ("Database", check_database),
        ("Strategies", check_strategies),
        ("Backtest Engine", check_backtest_engine),
        ("GUI Components", check_gui_components)
    ]

    results = {}
    for name, check_func in checks:
        logger.info(f"Checking {name}...")
        results[name] = check_func()
        logger.blank()

    # Summary
    logger.separator("=")
    passed = sum(results.values())
    total = len(results)

    if passed == total:
        logger.success(f"ALL CHECKS PASSED ({passed}/{total})")
        logger.success("GUI is ready to run out-of-the-box!")
    else:
        logger.warning(f"SOME CHECKS FAILED ({passed}/{total} passed)")
        logger.blank()
        logger.info("Failed checks:")
        for name, passed in results.items():
            if not passed:
                logger.error(f"  ✗ {name}")

        logger.blank()
        logger.info("RECOMMENDATIONS:")
        if not results.get("Database"):
            logger.warning("  → Run data ingestion to populate the database:")
            logger.info("     python src/data_engine/api/alpaca_client.py")

        if not results.get("Strategies"):
            logger.warning("  → Check that strategies are properly registered")

        if not results.get("Backtest Engine"):
            logger.warning("  → Verify backtest engine dependencies are installed")

        if not results.get("GUI Components"):
            logger.warning("  → Check Flet installation: pip install flet==0.28.3")

    logger.separator("=")
    logger.blank()

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
