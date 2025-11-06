"""
Comprehensive validation test for Level 4 regime analysis implementation.

Tests all phases:
- Phase 1: Data storage and retrieval
- Phase 2: File export (CSV, HTML, JSON)
- End-to-end integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import time
from gui.workers.gui_controller import GUIBacktestController
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def test_level4_complete_integration():
    """
    Comprehensive test of Level 4 implementation.

    Validates:
    1. Regime results stored in controller
    2. Regime results retrievable via get_regime_results()
    3. Regime results exported to CSV/HTML/JSON
    4. Portfolio objects have regime_analysis attribute
    """
    logger.blank()
    logger.separator()
    logger.header("LEVEL 4 COMPREHENSIVE VALIDATION TEST")
    logger.separator()
    logger.blank()

    # Test parameters
    symbols = ['AAPL']
    start_date = '2024-01-01'
    end_date = '2024-01-31'  # Short test period

    # Create strategy
    strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

    logger.info("=" * 80)
    logger.info("TEST: Level 4 Integration - Regime Analysis GUI & Export")
    logger.info("=" * 80)
    logger.blank()

    #
    # Phase 1 Test: Data Storage and Retrieval
    #
    logger.header("PHASE 1 TEST: Data Storage & Retrieval")
    logger.info("Testing regime results storage in GUIBacktestController...")
    logger.blank()

    controller = GUIBacktestController(max_workers=1)

    try:
        # Run backtest with regime analysis enabled
        controller.start_backtests(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            fees=0.001,
            risk_profile='Moderate',
            generate_full_output=True,  # Enable file export
            portfolio_mode='Single-Symbol',
            enable_regime_analysis=True  # ENABLE REGIME ANALYSIS
        )

        # Wait for completion
        logger.info("Waiting for backtest to complete...")
        while controller.is_running():
            time.sleep(0.5)

        # Check results
        results = controller.get_results()
        portfolios = controller.get_portfolios()
        regime_results = controller.get_regime_results()  # Phase 1: New method

        logger.blank()
        logger.info("PHASE 1 RESULTS:")
        logger.info(f"  Backtest results: {len(results)}")
        logger.info(f"  Portfolio objects: {len(portfolios)}")
        logger.info(f"  Regime results: {len(regime_results)}")  # Should be > 0
        logger.blank()

        # Validation checks
        phase1_passed = True

        # Check 1: Backtest completed successfully
        if len(results) == 0:
            logger.error("✗ FAILED: No backtest results returned")
            phase1_passed = False
        else:
            logger.success("✓ PASSED: Backtest results present")

        # Check 2: Portfolios stored
        if len(portfolios) == 0:
            logger.error("✗ FAILED: No portfolio objects stored")
            phase1_passed = False
        else:
            logger.success("✓ PASSED: Portfolio objects stored")

        # Check 3: Regime results stored (CRITICAL FOR LEVEL 4)
        if len(regime_results) == 0:
            logger.error("✗ FAILED: No regime results stored in controller")
            logger.error("  This means Phase 1 data storage is NOT working")
            phase1_passed = False
        else:
            logger.success(f"✓ PASSED: Regime results stored for {len(regime_results)} symbol(s)")

        # Check 4: Portfolio has regime_analysis attribute
        for symbol, portfolio in portfolios.items():
            if hasattr(portfolio, 'regime_analysis') and portfolio.regime_analysis is not None:
                logger.success(f"✓ PASSED: Portfolio[{symbol}].regime_analysis exists")
            else:
                logger.error(f"✗ FAILED: Portfolio[{symbol}].regime_analysis missing")
                phase1_passed = False

        # Check 5: Regime results retrievable
        for symbol in symbols:
            if symbol in regime_results:
                regime_result = regime_results[symbol]
                logger.success(f"✓ PASSED: Regime results retrievable for {symbol}")
                logger.info(f"    Robustness Score: {regime_result.robustness_score:.1f}/100")
                logger.info(f"    Best Regime: {regime_result.best_regime}")
                logger.info(f"    Worst Regime: {regime_result.worst_regime}")
            else:
                logger.error(f"✗ FAILED: Regime results NOT found for {symbol}")
                phase1_passed = False

        logger.blank()
        logger.separator()
        if phase1_passed:
            logger.success("PHASE 1: ✓ ALL CHECKS PASSED")
        else:
            logger.error("PHASE 1: ✗ SOME CHECKS FAILED")
        logger.separator()
        logger.blank()

        #
        # Phase 2 Test: File Export Validation
        #
        logger.header("PHASE 2 TEST: File Export (CSV/HTML/JSON)")
        logger.info("Checking for exported regime analysis files...")
        logger.blank()

        # Find output directory (most recent)
        from config import get_log_output_dir
        log_dir = get_log_output_dir()

        # Find most recent GUI output directory
        gui_dirs = sorted(log_dir.glob("*_GUI"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not gui_dirs:
            logger.error("✗ FAILED: No GUI output directory found")
            logger.error("  generate_full_output may not be working")
            phase2_passed = False
        else:
            output_dir = gui_dirs[0]
            logger.info(f"Output directory: {output_dir.name}")
            logger.blank()

            # Check for regime_analysis subdirectory
            regime_dir = output_dir / "regime_analysis"

            phase2_passed = True

            if not regime_dir.exists():
                logger.error("✗ FAILED: regime_analysis/ directory NOT created")
                logger.error("  Phase 2 export integration is NOT working")
                phase2_passed = False
            else:
                logger.success("✓ PASSED: regime_analysis/ directory exists")

                # Check for CSV files (4 expected per symbol: summary, trend, volatility, drawdown)
                csv_files = list(regime_dir.glob("*_regime_summary.csv"))
                if len(csv_files) >= len(symbols):
                    logger.success(f"✓ PASSED: CSV summary files found ({len(csv_files)})")
                else:
                    logger.error(f"✗ FAILED: Expected {len(symbols)} CSV files, found {len(csv_files)}")
                    phase2_passed = False

                # Check for additional CSV files
                for suffix in ['trend', 'volatility', 'drawdown']:
                    csv_files = list(regime_dir.glob(f"*_regime_{suffix}.csv"))
                    if len(csv_files) >= len(symbols):
                        logger.success(f"✓ PASSED: CSV {suffix} files found ({len(csv_files)})")
                    else:
                        logger.warning(f"⚠ WARNING: CSV {suffix} files may be missing")

                # Check for HTML files
                html_files = list(regime_dir.glob("*_regime.html"))
                if len(html_files) >= len(symbols):
                    logger.success(f"✓ PASSED: HTML files found ({len(html_files)})")
                    # Check HTML file size (should be > 1KB)
                    for html_file in html_files:
                        size_kb = html_file.stat().st_size / 1024
                        if size_kb > 1:
                            logger.success(f"  {html_file.name}: {size_kb:.1f} KB")
                        else:
                            logger.warning(f"  {html_file.name}: {size_kb:.1f} KB (seems small)")
                else:
                    logger.error(f"✗ FAILED: Expected {len(symbols)} HTML files, found {len(html_files)}")
                    phase2_passed = False

                # Check for JSON files
                json_files = list(regime_dir.glob("*_regime.json"))
                if len(json_files) >= len(symbols):
                    logger.success(f"✓ PASSED: JSON files found ({len(json_files)})")
                    # Try to load and parse JSON
                    import json
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            if 'robustness_score' in data and 'overall_sharpe' in data:
                                logger.success(f"  {json_file.name}: Valid JSON structure")
                            else:
                                logger.warning(f"  {json_file.name}: Missing expected fields")
                        except Exception as e:
                            logger.error(f"  {json_file.name}: Failed to parse - {e}")
                            phase2_passed = False
                else:
                    logger.error(f"✗ FAILED: Expected {len(symbols)} JSON files, found {len(json_files)}")
                    phase2_passed = False

            logger.blank()
            logger.separator()
            if phase2_passed:
                logger.success("PHASE 2: ✓ ALL CHECKS PASSED")
            else:
                logger.error("PHASE 2: ✗ SOME CHECKS FAILED")
            logger.separator()
            logger.blank()

        #
        # Final Summary
        #
        logger.blank()
        logger.separator("=", 80)
        logger.header("LEVEL 4 VALIDATION SUMMARY")
        logger.separator("=", 80)
        logger.blank()

        all_passed = phase1_passed and phase2_passed

        logger.info("Test Results:")
        logger.info(f"  Phase 1 (Data Storage): {'✓ PASSED' if phase1_passed else '✗ FAILED'}")
        logger.info(f"  Phase 2 (File Export):  {'✓ PASSED' if phase2_passed else '✗ FAILED'}")
        logger.blank()

        if all_passed:
            logger.success("=" * 80)
            logger.success("ALL LEVEL 4 TESTS PASSED ✓")
            logger.success("=" * 80)
            logger.blank()
            logger.success("Level 4 is fully functional:")
            logger.success("  ✓ Regime results stored in controller")
            logger.success("  ✓ Regime results exported to CSV/HTML/JSON")
            logger.success("  ✓ Portfolio objects have regime_analysis attribute")
            logger.success("  ✓ Ready for GUI display (Phase 3-4)")
            logger.blank()
            return True
        else:
            logger.error("=" * 80)
            logger.error("SOME LEVEL 4 TESTS FAILED ✗")
            logger.error("=" * 80)
            logger.blank()
            logger.error("Please review failures above and fix implementation")
            logger.blank()
            return False

    except Exception as e:
        logger.error("=" * 80)
        logger.error("TEST FAILED WITH EXCEPTION")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_level4_complete_integration()
    sys.exit(0 if success else 1)
