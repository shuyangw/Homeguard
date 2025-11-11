"""
Optimization script for MomentumStrategy (MACD) on BOTH bull (2019-2021) and bear (2022-2024) periods.

This is an UNTESTED strategy. Testing on both regimes to understand its behavior.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.momentum import MomentumStrategy
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def optimize_macd():
    """Optimize MomentumStrategy (MACD) on both bull and bear periods."""
    logger.blank()
    logger.separator()
    logger.header("MACD MOMENTUM STRATEGY OPTIMIZATION - BOTH REGIMES")
    logger.separator()
    logger.blank()

    # Symbols: Liquid tech stocks + QQQ
    symbols = ['AAPL', 'NVDA', 'QQQ']

    # Parameter grid
    param_grid = {
        'fast': [8, 10, 12, 15, 18, 20],
        'slow': [20, 23, 26, 30, 35, 40],
        'signal': [5, 7, 9, 12, 15]
    }

    # Calculate total combinations
    total_combos = (len(param_grid['fast']) *
                   len(param_grid['slow']) *
                   len(param_grid['signal']))

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Total combinations: {total_combos} per symbol per period")
    logger.blank()
    logger.warning("UNTESTED STRATEGY - First comprehensive test")
    logger.blank()

    # Test periods
    test_periods = [
        ('2019-01-01', '2021-12-31', 'BULL MARKET'),
        ('2022-01-01', '2024-01-01', 'BEAR MARKET')
    ]

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

    # Run optimization for each period and symbol
    results = {}
    start_time = time.time()

    for start_date, end_date, regime_name in test_periods:
        logger.blank()
        logger.separator()
        logger.header(f"TESTING PERIOD: {regime_name} ({start_date} to {end_date})")
        logger.separator()
        logger.blank()

        regime_results = {}

        for symbol in symbols:
            logger.separator()
            logger.header(f"OPTIMIZING {symbol} - {regime_name}")
            logger.separator()
            logger.blank()

            symbol_start = time.time()

            try:
                result = optimizer.optimize_parallel(
                    strategy_class=MomentumStrategy,
                    param_grid=param_grid,
                    symbols=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    metric='sharpe_ratio',
                    max_workers=4,
                    export_results=True
                )

                symbol_elapsed = time.time() - symbol_start

                regime_results[symbol] = {
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

                if result['best_value'] > 1.0:
                    logger.success(f"EXCELLENT: Sharpe > 1.0")
                elif result['best_value'] > 0.5:
                    logger.info(f"GOOD: Sharpe > 0.5")
                elif result['best_value'] > 0.0:
                    logger.warning(f"MARGINAL: Sharpe > 0 but < 0.5")
                else:
                    logger.error(f"NEGATIVE: Sharpe < 0")

            except Exception as e:
                logger.error(f"[FAILED] {symbol} optimization failed: {e}")
                import traceback
                traceback.print_exc()
                regime_results[symbol] = {
                    'error': str(e)
                }

        results[regime_name] = regime_results

    # Summary
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("COMPREHENSIVE OPTIMIZATION SUMMARY - MACD MOMENTUM")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.blank()

    # Print results by regime
    for regime_name, regime_results in results.items():
        logger.blank()
        logger.separator()
        logger.header(f"{regime_name} RESULTS")
        logger.separator()
        logger.blank()

        for symbol, result in regime_results.items():
            if 'error' in result:
                logger.error(f"{symbol}: FAILED - {result['error']}")
            else:
                logger.success(f"{symbol}:")
                logger.profit(f"  Sharpe Ratio: {result['best_sharpe']:.4f}")
                logger.metric(f"  Parameters: {result['best_params']}")
                logger.info(f"  Time: {result['time_taken']/60:.2f} min")
            logger.blank()

        # Find best in this regime
        valid_results = {k: v for k, v in regime_results.items() if 'error' not in v}
        if valid_results:
            best_symbol = max(valid_results.keys(), key=lambda k: valid_results[k]['best_sharpe'])
            logger.info(f"Best in {regime_name}: {best_symbol} (Sharpe: {valid_results[best_symbol]['best_sharpe']:.4f})")
            logger.blank()

    # Regime comparison
    logger.blank()
    logger.separator()
    logger.header("REGIME COMPARISON ANALYSIS")
    logger.separator()
    logger.blank()

    if 'BULL MARKET' in results and 'BEAR MARKET' in results:
        bull_results = {k: v for k, v in results['BULL MARKET'].items() if 'error' not in v}
        bear_results = {k: v for k, v in results['BEAR MARKET'].items() if 'error' not in v}

        if bull_results and bear_results:
            avg_bull_sharpe = sum(r['best_sharpe'] for r in bull_results.values()) / len(bull_results)
            avg_bear_sharpe = sum(r['best_sharpe'] for r in bear_results.values()) / len(bear_results)

            logger.info(f"Average Bull Market Sharpe: {avg_bull_sharpe:.4f}")
            logger.info(f"Average Bear Market Sharpe: {avg_bear_sharpe:.4f}")
            logger.info(f"Difference: {avg_bull_sharpe - avg_bear_sharpe:.4f}")
            logger.blank()

            if avg_bull_sharpe > avg_bear_sharpe:
                logger.success("MACD performs better in BULL markets")
            else:
                logger.warning("MACD performs better in BEAR markets (surprising!)")

            logger.blank()

            # Parameter stability
            logger.header("PARAMETER STABILITY ACROSS REGIMES")
            logger.blank()
            common_symbols = set(bull_results.keys()) & set(bear_results.keys())
            for symbol in common_symbols:
                bull_params = bull_results[symbol]['best_params']
                bear_params = bear_results[symbol]['best_params']
                logger.info(f"{symbol}:")
                logger.info(f"  Bull params: {bull_params}")
                logger.info(f"  Bear params: {bear_params}")
                if bull_params == bear_params:
                    logger.success(f"  STABLE: Same parameters work in both regimes!")
                else:
                    logger.warning(f"  UNSTABLE: Different parameters needed per regime")
                logger.blank()

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_macd()
        logger.success("[SUCCESS] Script completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
