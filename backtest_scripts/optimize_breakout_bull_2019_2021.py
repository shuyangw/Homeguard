"""
Optimization script for BreakoutStrategy (Donchian Channel) on BULL MARKET period (2019-2021).

Phase 2 of comprehensive testing: Proof-of-concept on proven profitable period.

This tests the core breakout mechanism WITHOUT optional filters to understand
fundamental strategy behavior. If positive results found, Phase 3 will test
across multiple symbols and time periods.

Strategy: Donchian Channel Breakout (Turtle Trading)
- Entry: Price breaks above N-period high
- Exit: Price breaks below M-period low
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.momentum import BreakoutStrategy
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def optimize_breakout_bull():
    """Optimize BreakoutStrategy on bull market period (2019-2021)."""
    logger.blank()
    logger.separator()
    logger.header("BREAKOUT STRATEGY OPTIMIZATION - BULL MARKET (2019-2021)")
    logger.separator()
    logger.blank()

    # Symbols: AAPL and MSFT (baseline comparison symbols)
    symbols = ['AAPL', 'MSFT']

    # Parameter grid: Core parameters only (no filters for simplicity)
    param_grid = {
        'breakout_window': [10, 15, 20, 30, 40, 50],  # Lookback for breakout high
        'exit_window': [5, 10, 15, 20],               # Lookback for exit low
        # All optional filters disabled (default False)
    }

    # Calculate total combinations
    total_combos = (len(param_grid['breakout_window']) *
                   len(param_grid['exit_window']))

    logger.info(f"Test Period: 2019-01-01 to 2021-12-31 (BULL MARKET)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Strategy: Donchian Channel Breakout (Turtle Trading)")
    logger.blank()
    logger.info(f"Parameter grid:")
    logger.info(f"  breakout_window: {param_grid['breakout_window']}")
    logger.info(f"  exit_window: {param_grid['exit_window']}")
    logger.info(f"  (all filters disabled for initial test)")
    logger.blank()
    logger.info(f"Total combinations: {total_combos} per symbol")
    logger.info(f"Total tests: {total_combos * len(symbols)}")
    logger.blank()
    logger.warning("Phase 2 Decision Point:")
    logger.warning("  Proceed to Phase 3 if: Sharpe > 0.2 on at least one symbol")
    logger.warning("  Stop if: Sharpe < 0.0 on both symbols")
    logger.blank()

    # Create engine with moderate risk (10% position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )

    engine.risk_config = RiskConfig.moderate()

    logger.info(f"Risk Profile: Moderate (10% position sizing)")
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
        logger.header(f"OPTIMIZING {symbol}")
        logger.separator()
        logger.blank()

        symbol_start = time.time()

        try:
            result = optimizer.optimize_parallel(
                strategy_class=BreakoutStrategy,
                param_grid=param_grid,
                symbols=symbol,
                start_date='2019-01-01',
                end_date='2021-12-31',
                metric='sharpe_ratio',
                max_workers=4,
                export_results=True
            )

            symbol_elapsed = time.time() - symbol_start

            results[symbol] = {
                'best_params': result['best_params'],
                'best_sharpe': result['best_value'],
                'time_taken': symbol_elapsed,
                'valid_combos': result.get('total_tested', 0),
                'cache_hits': result.get('cache_hits', 0)
            }

            logger.blank()
            logger.success(f"[SUCCESS] {symbol} optimization complete!")
            logger.profit(f"Best Sharpe Ratio: {result['best_value']:.4f}")
            logger.metric(f"Best Parameters: {result['best_params']}")
            logger.info(f"Valid combinations: {result.get('total_tested', 0)}")
            logger.info(f"Cache hits: {result.get('cache_hits', 0)}")
            logger.info(f"Time taken: {symbol_elapsed/60:.2f} minutes")
            logger.blank()

            # Interpret results
            if result['best_value'] >= 1.0:
                logger.success(f"EXCEPTIONAL: Sharpe >= 1.0 - Strategy shows strong edge!")
            elif result['best_value'] >= 0.5:
                logger.success(f"EXCELLENT: Sharpe >= 0.5 - Strong performance!")
            elif result['best_value'] >= 0.2:
                logger.info(f"GOOD: Sharpe >= 0.2 - Proceed to Phase 3")
            elif result['best_value'] > 0.0:
                logger.warning(f"MARGINAL: Sharpe > 0 but < 0.2 - Investigate further")
            elif result['best_value'] == 0.0:
                logger.error(f"NO TRADES: Sharpe = 0.0 - No signals generated")
            else:
                logger.error(f"NEGATIVE: Sharpe < 0 - Strategy loses money")

        except Exception as e:
            logger.error(f"[FAILED] {symbol} optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = {
                'error': str(e)
            }

    # Summary
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("OPTIMIZATION SUMMARY - BULL MARKET (2019-2021)")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.blank()

    logger.header("RESULTS BY SYMBOL")
    logger.blank()

    for symbol, result in results.items():
        if 'error' in result:
            logger.error(f"{symbol}: FAILED - {result['error']}")
        else:
            logger.success(f"{symbol}:")
            logger.profit(f"  Sharpe Ratio: {result['best_sharpe']:.4f}")
            logger.metric(f"  Parameters: {result['best_params']}")
            logger.info(f"  Time: {result['time_taken']/60:.2f} min")
        logger.blank()

    # Find best overall
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_symbol = max(valid_results.keys(), key=lambda k: valid_results[k]['best_sharpe'])
        logger.blank()
        logger.separator()
        logger.header("BEST CONFIGURATION")
        logger.separator()
        logger.blank()
        logger.success(f"Symbol: {best_symbol}")
        logger.profit(f"Sharpe Ratio: {valid_results[best_symbol]['best_sharpe']:.4f}")
        logger.metric(f"Parameters: {valid_results[best_symbol]['best_params']}")
        logger.blank()

        # Phase 3 decision
        logger.separator()
        logger.header("PHASE 3 DECISION")
        logger.separator()
        logger.blank()

        best_sharpe = valid_results[best_symbol]['best_sharpe']

        if best_sharpe >= 0.2:
            logger.success("DECISION: PROCEED TO PHASE 3")
            logger.success(f"Sharpe {best_sharpe:.4f} >= 0.2 threshold")
            logger.success("Strategy shows fundamental edge on bull market")
            logger.blank()
            logger.info("Next steps:")
            logger.info("1. Test optimal parameters across 7 symbols")
            logger.info("2. Test across 5 time periods (different market regimes)")
            logger.info("3. Generate 35-test comprehensive validation")
            logger.info("4. Compare against Bollinger Bands performance")
        elif best_sharpe > 0.0:
            logger.warning("DECISION: MARGINAL - INVESTIGATE FURTHER")
            logger.warning(f"Sharpe {best_sharpe:.4f} is positive but < 0.2")
            logger.warning("Consider testing with filters enabled")
            logger.blank()
            logger.info("Options:")
            logger.info("1. Try volatility filter to reduce false breakouts")
            logger.info("2. Try volume confirmation to improve signal quality")
            logger.info("3. Test on more symbols before making decision")
        else:
            logger.error("DECISION: STOP - STRATEGY NOT VIABLE")
            logger.error(f"Sharpe {best_sharpe:.4f} <= 0.0 on proven profitable period")
            logger.error("Strategy lacks fundamental edge")
            logger.blank()
            logger.info("Reasons for failure:")
            logger.info("- Breakouts may be too late (momentum already priced in)")
            logger.info("- False breakouts generate too many losses")
            logger.info("- Transaction costs too high relative to gains")
            logger.info("- Exit signals too quick, cutting profits short")

        logger.blank()

        # Comparison to Bollinger Bands
        logger.separator()
        logger.header("COMPARISON TO BOLLINGER BANDS")
        logger.separator()
        logger.blank()
        logger.info("Bollinger Bands (Mean Reversion) Results:")
        logger.info("  Period: 2019-2021 (same)")
        logger.info("  Median Sharpe: 0.33")
        logger.info("  Success Rate: 100% (9/9 symbols positive)")
        logger.info("  Strategy Type: Mean reversion (buy dips)")
        logger.blank()
        logger.info(f"Breakout Strategy Results:")
        logger.info(f"  Period: 2019-2021 (same)")
        logger.info(f"  Best Sharpe: {best_sharpe:.4f}")
        logger.info(f"  Symbols tested: {len(valid_results)}")
        logger.info(f"  Strategy Type: Momentum (buy breakouts)")
        logger.blank()

        if best_sharpe >= 0.33:
            logger.success("Breakout outperforms Bollinger Bands!")
            logger.success("Strong candidate for deployment")
        elif best_sharpe >= 0.2:
            logger.info("Breakout underperforms Bollinger Bands")
            logger.info("May still be viable as complementary strategy")
        else:
            logger.warning("Breakout significantly underperforms Bollinger Bands")

    logger.blank()
    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_breakout_bull()
        logger.success("[SUCCESS] Script completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
