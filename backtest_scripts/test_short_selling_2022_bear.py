"""
Test Short Selling Capability on 2022 Bear Market

This script validates the short selling implementation by comparing:
1. Long-only strategies (expected: negative Sharpe in bear market)
2. Long/short strategies (expected: positive Sharpe in bear market)

The 2022 bear market provides the perfect test case for short selling:
- SPY dropped ~25% from Jan to Oct 2022
- Long-only strategies should lose money
- Short-enabled strategies should profit from downtrends

Expected Impact: +0.5 to +1.0 Sharpe improvement
"""

import sys
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import RSIMeanReversion
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def test_ma_crossover_long_vs_short():
    """
    Test MA Crossover: Long-only vs Long/Short on 2022 bear market.

    Expected:
    - Long-only: Negative Sharpe (can't profit from downtrends)
    - Long/short: Positive Sharpe (profits from both up and down trends)
    """
    logger.blank()
    logger.separator()
    logger.header("TEST 1: MA CROSSOVER - LONG-ONLY VS LONG/SHORT")
    logger.separator()
    logger.blank()

    logger.info("Period: 2022-01-01 to 2022-12-31 (Bear Market)")
    logger.info("Symbol: AAPL")
    logger.info("Strategy: Moving Average Crossover (20/100)")
    logger.blank()

    # Strategy parameters
    strategy = MovingAverageCrossover(fast_window=20, slow_window=100, ma_type='ema')

    # Test 1: Long-only (traditional approach)
    logger.header("VERSION 1: LONG-ONLY (Traditional)")
    logger.blank()
    logger.warning("Can only profit when price goes up")
    logger.warning("Must hold cash during downtrends")
    logger.blank()

    engine_long_only = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=False  # Long-only
    )
    engine_long_only.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long_only.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_long = portfolio_long.stats()
    logger.blank()
    logger.separator()
    logger.blank()

    # Test 2: Long/short (new capability)
    logger.header("VERSION 2: LONG/SHORT (Short Selling Enabled)")
    logger.blank()
    logger.success("Can profit from both uptrends AND downtrends")
    logger.success("Short when fast MA < slow MA")
    logger.blank()

    engine_long_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=True  # Enable shorts
    )
    engine_long_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_long_short.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_short = portfolio_short.stats()
    logger.blank()
    logger.separator()
    logger.blank()

    # Compare results
    logger.header("RESULTS COMPARISON")
    logger.separator()
    logger.blank()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Sharpe Ratio: {stats_long['Sharpe Ratio']:.4f}")
    logger.info(f"  Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.info(f"  Max Drawdown: {stats_long['Max Drawdown [%]']:.2f}%")
    logger.info(f"  Total Trades: {stats_long['Total Trades']}")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Sharpe Ratio: {stats_short['Sharpe Ratio']:.4f}")
    logger.info(f"  Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.info(f"  Max Drawdown: {stats_short['Max Drawdown [%]']:.2f}%")
    logger.info(f"  Total Trades: {stats_short['Total Trades']}")
    logger.blank()

    # Calculate improvement
    sharpe_improvement = stats_short['Sharpe Ratio'] - stats_long['Sharpe Ratio']
    return_improvement = stats_short['Total Return [%]'] - stats_long['Total Return [%]']

    logger.separator()
    logger.header("IMPROVEMENT FROM SHORT SELLING")
    logger.separator()
    logger.blank()

    if sharpe_improvement > 0:
        logger.profit(f"Sharpe Improvement: +{sharpe_improvement:.4f}")
    else:
        logger.error(f"Sharpe Improvement: {sharpe_improvement:.4f}")

    if return_improvement > 0:
        logger.profit(f"Return Improvement: +{return_improvement:.2f}%")
    else:
        logger.error(f"Return Improvement: {return_improvement:.2f}%")

    logger.blank()

    if sharpe_improvement >= 0.5:
        logger.profit("✅ TARGET MET: Sharpe improvement >= +0.5")
        logger.profit("   Short selling capability working as expected!")
    elif sharpe_improvement >= 0.3:
        logger.success("✅ GOOD: Sharpe improvement >= +0.3")
        logger.success("   Short selling providing significant benefit")
    elif sharpe_improvement > 0:
        logger.warning("⚠️  MARGINAL: Sharpe improvement positive but < +0.3")
        logger.warning("   Short selling helps but not dramatically")
    else:
        logger.error("❌ ISSUE: Short selling not improving performance")
        logger.error("   May need to investigate implementation")

    logger.blank()
    logger.separator()

    return {
        'long_only': stats_long,
        'long_short': stats_short,
        'improvement': sharpe_improvement
    }


def test_rsi_with_shorts():
    """
    Test RSI Mean Reversion with short selling on 2022 bear market.

    RSI Mean Reversion should work well with shorts:
    - Short when RSI > 70 (overbought)
    - Cover when RSI < 50
    """
    logger.blank()
    logger.separator()
    logger.header("TEST 2: RSI MEAN REVERSION WITH SHORT SELLING")
    logger.separator()
    logger.blank()

    logger.info("Period: 2022-01-01 to 2022-12-31 (Bear Market)")
    logger.info("Symbol: NVDA (high volatility)")
    logger.info("Strategy: RSI Mean Reversion (14, 30, 70)")
    logger.blank()

    strategy = RSIMeanReversion(rsi_window=14, oversold=30, overbought=70)

    # Long-only
    logger.header("VERSION 1: LONG-ONLY")
    logger.blank()

    engine_long = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=False
    )
    engine_long.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long.run(
        strategy=strategy,
        symbols='NVDA',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_long = portfolio_long.stats()
    logger.blank()
    logger.separator()
    logger.blank()

    # Long/short
    logger.header("VERSION 2: LONG/SHORT")
    logger.blank()

    engine_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=True
    )
    engine_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_short.run(
        strategy=strategy,
        symbols='NVDA',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_short = portfolio_short.stats()
    logger.blank()
    logger.separator()
    logger.blank()

    # Compare
    logger.header("RESULTS COMPARISON")
    logger.separator()
    logger.blank()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Sharpe Ratio: {stats_long['Sharpe Ratio']:.4f}")
    logger.info(f"  Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.info(f"  Total Trades: {stats_long['Total Trades']}")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Sharpe Ratio: {stats_short['Sharpe Ratio']:.4f}")
    logger.info(f"  Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.info(f"  Total Trades: {stats_short['Total Trades']}")
    logger.blank()

    sharpe_improvement = stats_short['Sharpe Ratio'] - stats_long['Sharpe Ratio']

    if sharpe_improvement > 0:
        logger.profit(f"Sharpe Improvement: +{sharpe_improvement:.4f}")
    else:
        logger.error(f"Sharpe Improvement: {sharpe_improvement:.4f}")

    logger.blank()
    logger.separator()

    return {
        'long_only': stats_long,
        'long_short': stats_short,
        'improvement': sharpe_improvement
    }


def main():
    """Run all short selling validation tests."""
    logger.blank()
    logger.separator()
    logger.header("SHORT SELLING VALIDATION TEST SUITE")
    logger.info("Testing on 2022 Bear Market (-25% SPY drop)")
    logger.separator()
    logger.blank()

    # Run tests
    results_ma = test_ma_crossover_long_vs_short()
    logger.blank()
    logger.blank()

    results_rsi = test_rsi_with_shorts()
    logger.blank()
    logger.blank()

    # Summary
    logger.separator()
    logger.header("OVERALL SHORT SELLING VALIDATION SUMMARY")
    logger.separator()
    logger.blank()

    avg_improvement = (results_ma['improvement'] + results_rsi['improvement']) / 2

    logger.metric(f"MA Crossover Sharpe Improvement: {results_ma['improvement']:+.4f}")
    logger.metric(f"RSI Mean Reversion Sharpe Improvement: {results_rsi['improvement']:+.4f}")
    logger.metric(f"Average Sharpe Improvement: {avg_improvement:+.4f}")
    logger.blank()

    if avg_improvement >= 0.5:
        logger.profit("✅ SUCCESS: Short selling working as expected!")
        logger.profit("   Average improvement >= +0.5 Sharpe")
        logger.profit("   Ready for production use")
    elif avg_improvement >= 0.3:
        logger.success("✅ GOOD: Short selling provides significant benefit")
        logger.success("   Average improvement >= +0.3 Sharpe")
    elif avg_improvement > 0:
        logger.warning("⚠️  MARGINAL: Short selling helps but not dramatically")
        logger.warning("   Consider parameter tuning")
    else:
        logger.error("❌ VALIDATION FAILED: Short selling not improving performance")
        logger.error("   Implementation needs review")

    logger.blank()
    logger.separator()


if __name__ == '__main__':
    main()
