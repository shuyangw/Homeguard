"""
Optimization script for RSIMeanReversion strategy on BULL MARKET period (2019-2021).

Previous testing (2022-2024) showed Sharpe -0.80 on NVDA.
Testing on bull market to see if mean reversion works better in trending environments.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.mean_reversion import RSIMeanReversion
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def optimize_rsi_bull():
    """Optimize RSIMeanReversion strategy on bull market period."""
    logger.blank()
    logger.separator()
    logger.header("RSI MEAN REVERSION OPTIMIZATION - BULL MARKET (2019-2021)")
    logger.separator()
    logger.blank()

    # Symbols: Volatile tech stocks (same as previous test for comparison)
    symbols = ['AAPL', 'MSFT', 'SPY']

    # Parameter grid (same as previous test)
    param_grid = {
        'rsi_window': [7, 10, 14, 17, 21],
        'oversold': [20, 25, 30, 35],
        'overbought': [65, 70, 75, 80]
    }

    # Calculate total combinations
    total_combos = (len(param_grid['rsi_window']) *
                   len(param_grid['oversold']) *
                   len(param_grid['overbought']))

    logger.info(f"Test Period: 2019-01-01 to 2021-12-31 (BULL MARKET)")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Total combinations: {total_combos} per symbol")
    logger.info(f"Total tests: {total_combos * len(symbols)}")
    logger.blank()
    logger.warning("Hypothesis: RSI mean reversion may work differently in bull vs bear markets")
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
                strategy_class=RSIMeanReversion,
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

        logger.separator()
        logger.header("COMPARISON WITH PREVIOUS BEAR MARKET (2022-2024)")
        logger.separator()
        logger.blank()
        logger.info(f"Previous Bear Market Sharpe (NVDA): -0.80")
        logger.info(f"Current Bull Market Sharpe ({best_symbol}): {valid_results[best_symbol]['best_sharpe']:.4f}")
        improvement = valid_results[best_symbol]['best_sharpe'] - (-0.80)
        logger.profit(f"Improvement: {improvement:.4f} Sharpe points")
        logger.blank()

    logger.separator()

    return results


if __name__ == '__main__':
    try:
        results = optimize_rsi_bull()
        logger.success("[SUCCESS] Script completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAILED] Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
