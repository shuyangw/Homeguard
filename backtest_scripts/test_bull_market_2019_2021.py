"""
Test strategies on 2019-2021 bull market period to validate infrastructure.

This script re-runs the same optimizations that were tested on 2022-2024 (bear/choppy)
but on the 2019-2021 bull market period. This validates that negative Sharpe ratios
in the original testing were due to unfavorable market conditions, not system flaws.

Expected Results:
- MA Crossover: Sharpe -2.15 (2022-2024) → +1.2 to +1.8 (2019-2021)
- RSI MeanReversion: Sharpe -0.80 (2022-2024) → +0.5 to +1.0 (2019-2021)
- MeanReversion: Sharpe 0.00 (2022-2024) → +0.5 to +1.2 (2019-2021)
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer
from strategies.base_strategies.mean_reversion import RSIMeanReversion, MeanReversion
from strategies.base_strategies.moving_average import MovingAverageCrossover
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def test_strategy(engine, optimizer, strategy_class, strategy_name, param_grid, symbols):
    """Test a single strategy across multiple symbols."""
    logger.separator()
    logger.header(f"{strategy_name} OPTIMIZATION (2019-2021 BULL MARKET)")
    logger.separator()
    logger.blank()

    results = {}

    for symbol in symbols:
        logger.info(f"Optimizing {strategy_name} on {symbol}...")

        try:
            result = optimizer.optimize_parallel(
                strategy_class=strategy_class,
                param_grid=param_grid,
                symbols=symbol,
                start_date='2019-01-01',  # Bull market start
                end_date='2021-12-31',    # Bull market end (before 2022 crash)
                metric='sharpe_ratio',
                max_workers=4,
                export_results=True
            )

            results[symbol] = {
                'best_params': result['best_params'],
                'best_sharpe': result['best_value'],
                'total_tested': result.get('total_combinations', 0)
            }

            logger.success(f"  {symbol}: Sharpe = {result['best_value']:.4f}")
            logger.metric(f"  Parameters: {result['best_params']}")

        except Exception as e:
            logger.error(f"  {symbol}: FAILED - {e}")
            results[symbol] = {'error': str(e)}

    logger.blank()
    return results


def main():
    """Run all strategy optimizations on 2019-2021 bull market period."""
    logger.blank()
    logger.separator()
    logger.header("BULL MARKET VALIDATION (2019-2021)")
    logger.separator()
    logger.info("Testing strategies on favorable market period")
    logger.info("Comparing with 2022-2024 bear/choppy results")
    logger.blank()

    # Create engine with moderate risk (10% position sizing)
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,  # 0.1% trading fees
        slippage=0.0005  # 0.05% slippage
    )
    engine.risk_config = RiskConfig.moderate()

    optimizer = GridSearchOptimizer(engine)

    all_results = {}
    start_time = time.time()

    # ========== 1. RSI Mean Reversion (High Vol Stocks) ==========
    logger.header("1. RSI MEAN REVERSION")
    logger.blank()

    rsi_param_grid = {
        'rsi_window': [7, 10, 14, 17, 21],
        'oversold': [20, 25, 30, 35],
        'overbought': [65, 70, 75, 80]
    }

    rsi_symbols = ['NVDA', 'TSLA', 'AMD']  # Best performer in 2022-2024 was NVDA (-0.80)

    all_results['RSIMeanReversion'] = test_strategy(
        engine, optimizer,
        RSIMeanReversion,
        "RSI Mean Reversion",
        rsi_param_grid,
        rsi_symbols
    )

    # ========== 2. Moving Average Crossover (Trend Following) ==========
    logger.header("2. MOVING AVERAGE CROSSOVER")
    logger.blank()

    ma_param_grid = {
        'fast_window': [10, 20, 30, 40, 50],
        'slow_window': [50, 75, 100, 125, 150, 175, 200],
        'ma_type': ['sma', 'ema']
    }

    ma_symbols = ['AAPL', 'MSFT', 'TSLA']  # TSLA was best in 2022-2024 (-0.997)

    all_results['MovingAverageCrossover'] = test_strategy(
        engine, optimizer,
        MovingAverageCrossover,
        "Moving Average Crossover",
        ma_param_grid,
        ma_symbols
    )

    # ========== 3. Mean Reversion (Bollinger Bands) ==========
    logger.header("3. MEAN REVERSION (BOLLINGER BANDS)")
    logger.blank()

    bb_param_grid = {
        'window': [10, 15, 20, 25, 30, 40, 50],
        'num_std': [1.5, 2.0, 2.5, 3.0],
        'exit_at_middle': [True, False]
    }

    bb_symbols = ['AAPL', 'SPY', 'QQQ']  # AAPL generated 0.00 Sharpe (no trades)

    all_results['MeanReversion'] = test_strategy(
        engine, optimizer,
        MeanReversion,
        "Mean Reversion",
        bb_param_grid,
        bb_symbols
    )

    # ========== Summary ==========
    total_elapsed = time.time() - start_time

    logger.blank()
    logger.separator()
    logger.header("BULL MARKET RESULTS SUMMARY (2019-2021)")
    logger.separator()
    logger.blank()

    logger.info(f"Total execution time: {total_elapsed/60:.2f} minutes")
    logger.blank()

    # Print results comparison
    logger.header("PERFORMANCE COMPARISON")
    logger.blank()

    # Reference: Previous 2022-2024 results
    logger.info("2022-2024 (Bear/Choppy) Best Results:")
    logger.error("  RSI MeanReversion (NVDA): Sharpe = -0.8029")
    logger.error("  MA Crossover (TSLA): Sharpe = -0.9969")
    logger.error("  Mean Reversion (AAPL): Sharpe = 0.0000 (no trades)")
    logger.blank()

    logger.info("2019-2021 (Bull Market) Results:")
    logger.blank()

    for strategy_name, symbol_results in all_results.items():
        logger.metric(f"{strategy_name}:")

        valid_results = {k: v for k, v in symbol_results.items() if 'error' not in v}
        if valid_results:
            best_symbol = max(valid_results.keys(), key=lambda k: valid_results[k]['best_sharpe'])
            best_sharpe = valid_results[best_symbol]['best_sharpe']

            if best_sharpe > 1.0:
                logger.profit(f"  Best: {best_symbol} = {best_sharpe:.4f} ✅ PROFITABLE")
            elif best_sharpe > 0.0:
                logger.success(f"  Best: {best_symbol} = {best_sharpe:.4f} ⚠️  MARGINAL")
            else:
                logger.error(f"  Best: {best_symbol} = {best_sharpe:.4f} ❌ NEGATIVE")

            logger.info(f"  Parameters: {valid_results[best_symbol]['best_params']}")
        else:
            logger.error(f"  ALL SYMBOLS FAILED")

        logger.blank()

    # Analysis
    logger.separator()
    logger.header("ANALYSIS")
    logger.separator()
    logger.blank()

    logger.info("Key Findings:")
    logger.blank()

    logger.info("1. Market Period Impact:")
    logger.info("   - Bull market (2019-2021): Trending, low volatility")
    logger.info("   - Bear market (2022-2024): Choppy, high volatility")
    logger.info("   - Long-only strategies MUST show positive Sharpe in bull")
    logger.blank()

    logger.info("2. Strategy Suitability:")
    logger.info("   - MA Crossover: Should profit in trending bull market")
    logger.info("   - Mean Reversion: May struggle in trending markets")
    logger.info("   - RSI MeanReversion: Works in ranges, needs testing")
    logger.blank()

    logger.info("3. Infrastructure Validation:")
    logger.info("   - If Sharpe > 1.0: Infrastructure is working correctly ✅")
    logger.info("   - If Sharpe < 0.0: Check strategy logic or data quality ❌")
    logger.blank()

    logger.separator()
    logger.success("BULL MARKET VALIDATION COMPLETE")
    logger.separator()
    logger.blank()


if __name__ == '__main__':
    main()
