"""
PHASE 2: VolatilityTargetedMomentum Bull Market Proof-of-Concept (2019-2021)

CRITICAL TEST: Determine if volatility targeting has ANY fundamental edge.

Period: 2019-2021 (300%+ bull market for AAPL)
Symbols: AAPL, MSFT
Method: Grid Search (324 combinations, exhaustive)
Objective: Find if ANY parameters yield Sharpe > 0.3 (Bollinger Bands baseline)

Decision Criteria:
- Sharpe > 0.3: PROCEED to Phase 3 (promising)
- Sharpe 0.2-0.3: PROCEED with caution (marginal)
- Sharpe 0.0-0.2: WEAK (barely profitable)
- Sharpe < 0.0: STOP (fundamentally broken)
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def optimize_bull_market():
    """
    Optimize VolatilityTargetedMomentum on 2019-2021 bull market.

    This is the critical proof-of-concept test to determine if there's
    any fundamental edge before expanding to comprehensive validation.
    """
    logger.blank()
    logger.separator()
    logger.header("VOLATILITY TARGETED MOMENTUM - BULL MARKET PROOF-OF-CONCEPT")
    logger.separator()
    logger.blank()

    # Test symbols
    symbols = ['AAPL', 'MSFT']

    # Parameter space for Grid Search optimization
    # Reduced from full space to make grid search feasible
    # Based on analysis in VOLATILITY_TARGETED_ANALYSIS.md
    param_grid = {
        'lookback_period': [150, 200, 250],                 # Momentum lookback (3 values)
        'ma_window': [150, 200, 250],                       # Trend filter sensitivity (3 values)
        'vol_window': [15, 20, 25],                         # Vol estimation window (3 values)
        'target_vol': [0.12, 0.15, 0.18, 0.22],             # Target volatility (KEY) (4 values)
        'max_leverage': [1.5, 2.0, 2.5],                    # Max position size (3 values)
        'use_ma_filter': [True],                            # Enable MA signal (fixed - most important)
        'use_return_filter': [False],                       # Disable return signal (fixed)
        'combine_filters': ['or']                           # Signal logic (fixed)
    }

    logger.info("TEST DESIGN")
    logger.info("-" * 60)
    logger.info(f"Period: 2019-01-01 to 2021-12-31 (Bull Market)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Baseline: Bollinger Bands Sharpe = +0.33")
    logger.info(f"Target: Find parameters with Sharpe > 0.3")
    logger.blank()

    logger.info("PARAMETER SPACE")
    logger.info("-" * 60)
    logger.info(f"  lookback_period: [150, 200, 250] (3 values)")
    logger.info(f"  ma_window: [150, 200, 250] (3 values)")
    logger.info(f"  vol_window: [15, 20, 25] (3 values)")
    logger.info(f"  target_vol: [0.12, 0.15, 0.18, 0.22] (4 values - KEY PARAMETER)")
    logger.info(f"  max_leverage: [1.5, 2.0, 2.5] (3 values)")
    logger.info(f"  use_ma_filter: [True] (fixed)")
    logger.info(f"  use_return_filter: [False] (fixed)")
    logger.info(f"  combine_filters: ['or'] (fixed)")
    logger.blank()

    # Calculate total combinations
    total_combinations = (
        len(param_grid['lookback_period']) *
        len(param_grid['ma_window']) *
        len(param_grid['vol_window']) *
        len(param_grid['target_vol']) *
        len(param_grid['max_leverage']) *
        len(param_grid['use_ma_filter']) *
        len(param_grid['use_return_filter']) *
        len(param_grid['combine_filters'])
    )

    logger.info(f"Total combinations: {total_combinations} (Grid Search)")
    logger.info(f"Method: Exhaustive grid search (tests ALL combinations)")
    logger.blank()

    logger.warning("CRITICAL CONCERNS")
    logger.info("-" * 60)
    logger.info("  1. Long-only momentum has FAILED before (MA Crossover: -1.16, Breakout: -6.59)")
    logger.info("  2. Volatility targeting is a WRAPPER - needs good signals")
    logger.info("  3. This strategy may fail for the same reasons")
    logger.info("  4. But: Better signals (200MA) + vol targeting may work")
    logger.blank()

    # Create engine with moderate risk (10% base position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,        # 0.1% per trade
        slippage=0.0005    # 0.05% slippage
    )

    engine.risk_config = RiskConfig.moderate()

    logger.info("RISK CONFIGURATION")
    logger.info("-" * 60)
    logger.info(f"Base Position Sizing: 10% (moderate risk profile)")
    logger.info(f"Volatility Scaling: Applied ON TOP of base 10%")
    logger.info(f"  Example: 10% × 1.5x vol scaling = 15% actual position")
    logger.info(f"Maximum Exposure: 10% × 2.5 max leverage = 25% per trade")
    logger.info(f"Initial Capital: ${engine.initial_capital:,.0f}")
    logger.info(f"Trading Fees: {engine.fees*100:.2f}%")
    logger.info(f"Slippage: {engine.slippage*100:.3f}%")
    logger.blank()

    # Create optimizer
    optimizer = GridSearchOptimizer(engine)

    # Run optimization for each symbol
    results = {}
    start_time = time.time()

    for symbol in symbols:
        logger.separator()
        logger.header(f"OPTIMIZING {symbol} - GRID SEARCH ({total_combinations} combinations)")
        logger.separator()
        logger.blank()

        symbol_start = time.time()

        try:
            result = optimizer.optimize(
                strategy_class=VolatilityTargetedMomentum,
                param_grid=param_grid,
                symbols=symbol,
                start_date='2019-01-01',
                end_date='2021-12-31',  # Bull market only
                metric='sharpe_ratio'
            )

            symbol_elapsed = time.time() - symbol_start

            results[symbol] = {
                'best_params': result['best_params'],
                'best_sharpe': result['best_value'],
                'time_taken': symbol_elapsed,
                'combinations_tested': total_combinations
            }

            logger.blank()
            logger.success(f"[SUCCESS] {symbol} optimization complete!")
            logger.blank()

            # Performance assessment
            sharpe = result['best_value']
            logger.profit(f"Best Sharpe Ratio: {sharpe:.4f}")
            logger.blank()

            if sharpe > 0.4:
                logger.success("EXCEPTIONAL: Sharpe > 0.4 - BEATS Bollinger Bands!")
                logger.success("  → Proceed to Phase 3 with HIGH confidence")
            elif sharpe > 0.3:
                logger.success("EXCELLENT: Sharpe > 0.3 - MATCHES Bollinger Bands")
                logger.success("  → Proceed to Phase 3 with confidence")
            elif sharpe > 0.2:
                logger.info("GOOD: Sharpe > 0.2 - Profitable but inferior")
                logger.info("  → Proceed to Phase 3 with caution")
            elif sharpe > 0.0:
                logger.warning("MARGINAL: Sharpe > 0 but < 0.2")
                logger.warning("  → Consider limited Phase 3 expansion")
            else:
                logger.error("FAILED: Sharpe < 0 - Negative returns")
                logger.error("  → STOP - Fundamentally broken")

            logger.blank()
            logger.header("BEST PARAMETERS")
            logger.info("-" * 60)
            for param, value in result['best_params'].items():
                if isinstance(value, float):
                    logger.info(f"  {param}: {value:.4f}")
                else:
                    logger.info(f"  {param}: {value}")
            logger.blank()

            # Highlight key parameters
            target_vol = result['best_params'].get('target_vol', 0)
            max_lev = result['best_params'].get('max_leverage', 0)
            vol_window = result['best_params'].get('vol_window', 0)

            logger.header("KEY INSIGHTS")
            logger.info("-" * 60)
            logger.info(f"Target Volatility: {target_vol:.2%} annualized")
            logger.info(f"  → Portfolio aims for {target_vol:.0%} vol regardless of market")
            logger.info(f"Max Leverage: {max_lev:.2f}x")
            logger.info(f"  → Maximum position size: {max_lev*10:.0f}% of capital")
            logger.info(f"Volatility Window: {vol_window} days")
            logger.info(f"  → How quickly position sizing adapts to vol changes")
            logger.info(f"MA Filter: {result['best_params'].get('use_ma_filter', False)}")
            logger.info(f"Return Filter: {result['best_params'].get('use_return_filter', False)}")
            logger.info(f"Combine Logic: {result['best_params'].get('combine_filters', 'N/A')}")
            logger.blank()

            logger.info(f"Optimization Stats:")
            logger.info(f"  Combinations tested: {total_combinations}")
            logger.info(f"  Coverage: 100% (exhaustive grid search)")
            logger.info(f"  Time taken: {symbol_elapsed/60:.2f} minutes")
            logger.blank()

        except Exception as e:
            logger.error(f"[FAILED] {symbol} optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = {'error': str(e)}

    # Summary
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("PHASE 2 SUMMARY - BULL MARKET PROOF-OF-CONCEPT")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.info(f"Optimization method: Grid Search (exhaustive)")
    logger.info(f"Test period: 2019-2021 bull market")
    logger.info(f"Combinations tested: {total_combinations} (100% coverage)")
    logger.blank()

    logger.header("RESULTS BY SYMBOL")
    logger.info("-" * 60)
    logger.blank()

    for symbol, result in results.items():
        if 'error' in result:
            logger.error(f"{symbol}: FAILED - {result['error']}")
        else:
            logger.success(f"{symbol}:")
            logger.profit(f"  Sharpe Ratio: {result['best_sharpe']:.4f}")
            logger.info(f"  Target Vol: {result['best_params'].get('target_vol', 0):.2%}")
            logger.info(f"  Max Leverage: {result['best_params'].get('max_leverage', 0):.2f}x")
            logger.info(f"  Combinations tested: {result['combinations_tested']}")
            logger.info(f"  Time: {result['time_taken']/60:.2f} min")
        logger.blank()

    # Overall assessment
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_symbol = max(valid_results.keys(), key=lambda k: valid_results[k]['best_sharpe'])
        best_sharpe = valid_results[best_symbol]['best_sharpe']

        logger.separator()
        logger.header("PHASE 2 DECISION POINT")
        logger.separator()
        logger.blank()

        logger.success(f"Best Result: {best_symbol} with Sharpe = {best_sharpe:.4f}")
        logger.blank()

        if best_sharpe > 0.3:
            logger.success("DECISION: PROCEED TO PHASE 3")
            logger.success("  Rationale: Sharpe > 0.3 indicates fundamental edge")
            logger.success("  Next steps:")
            logger.success("    1. Test on 5 additional symbols (GOOGL, AMZN, META, NVDA, TSLA)")
            logger.success("    2. Test on 5 time periods (different vol regimes)")
            logger.success("    3. Total: 35 comprehensive tests")
            logger.success("    4. Compare consistency to Bollinger Bands")
        elif best_sharpe > 0.2:
            logger.info("DECISION: PROCEED TO PHASE 3 WITH CAUTION")
            logger.info("  Rationale: Profitable but inferior to Bollinger Bands")
            logger.info("  Next steps:")
            logger.info("    1. Limited symbol expansion (3-4 symbols)")
            logger.info("    2. Test on bear market 2022 (key test for vol-targeting)")
            logger.info("    3. Assess if complexity is justified")
        elif best_sharpe > 0.0:
            logger.warning("DECISION: MARGINAL - CONSIDER STOPPING")
            logger.warning("  Rationale: Barely profitable, unlikely to beat Bollinger Bands")
            logger.warning("  Options:")
            logger.warning("    1. Limited Phase 3 (2022 bear market test only)")
            logger.warning("    2. Skip to next strategy (simpler alternatives)")
        else:
            logger.error("DECISION: STOP")
            logger.error("  Rationale: Negative Sharpe indicates fundamental issues")
            logger.error("  Root cause: Long-only momentum curse (same as previous failures)")
            logger.error("  Next action: Move to next strategy")

        logger.blank()
        logger.separator()
        logger.header("VOLATILITY TARGETING ASSESSMENT")
        logger.separator()
        logger.blank()

        logger.info("Expected benefits of volatility targeting:")
        logger.info("  ✓ Consistent risk across different market regimes")
        logger.info("  ✓ Automatic leverage reduction in high-vol periods")
        logger.info("  ✓ Automatic leverage increase in low-vol periods")
        logger.info("  ✓ Better risk-adjusted returns than fixed sizing")
        logger.blank()

        if best_sharpe > 0.2:
            logger.success("Initial evidence: Volatility targeting adds value")
            logger.info("  → Need Phase 3 to confirm across regimes")
        else:
            logger.warning("Initial evidence: Volatility targeting insufficient")
            logger.info("  → Underlying signals may be the problem")

        logger.blank()
    else:
        logger.error("CRITICAL FAILURE: All symbols failed optimization")
        logger.error("DECISION: STOP - Strategy is fundamentally broken")

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_bull_market()
        if results is not None:
            # Check if we should proceed
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_sharpe = max(v['best_sharpe'] for v in valid_results.values())
                if best_sharpe > 0.0:
                    logger.success("[SUCCESS] Strategy shows positive edge!")
                    sys.exit(0)
                else:
                    logger.error("[FAILED] Strategy has negative Sharpe")
                    sys.exit(1)
            else:
                logger.error("[FAILED] All optimizations failed")
                sys.exit(1)
        else:
            logger.error("[FAILED] Optimization returned None")
            sys.exit(1)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
