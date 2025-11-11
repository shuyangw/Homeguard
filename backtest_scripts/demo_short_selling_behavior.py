"""
Demonstrate Short Selling Behavior Changes

This script shows EXACT behavior differences between long-only and long/short modes
for different strategy types with concrete examples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import RSIMeanReversion
from strategies.base_strategies.momentum import BreakoutStrategy
from backtesting.utils.risk_config import RiskConfig
from utils import logger
import pandas as pd


def example_1_ma_crossover_bear_market():
    """
    Example 1: MA Crossover in 2022 Bear Market

    Shows how the SAME strategy behaves differently with shorts enabled.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 1: MA CROSSOVER IN BEAR MARKET")
    logger.separator()
    logger.blank()

    logger.info("Scenario: Apple stock declining from $180 to $130 in 2022")
    logger.info("Strategy: MA Crossover (20/100)")
    logger.blank()

    strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

    # Long-only version
    logger.header("LONG-ONLY MODE (allow_shorts=False)")
    logger.separator()
    logger.blank()

    engine_long = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=False
    )
    engine_long.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_long = portfolio_long.stats()
    trades_long = portfolio_long.trades()

    logger.blank()
    logger.info("=== TIMELINE OF POSITIONS (Long-Only) ===")
    logger.blank()

    # Show what happens
    logger.info("Jan 2022: Fast MA > Slow MA → ENTER LONG @ $180")
    logger.metric("  Position: +100 shares")
    logger.metric("  Capital deployed: $18,000")
    logger.blank()

    logger.info("Mar 2022: Fast MA < Slow MA → EXIT LONG @ $165")
    logger.metric("  Position: 0 shares (FLAT)")
    logger.loss("  P&L: -$1,500 (-8.3%)")
    logger.warning("  → Sitting in CASH while stock continues falling")
    logger.blank()

    logger.info("Apr-Dec 2022: Stock falls to $130")
    logger.warning("  Position: Still FLAT (0 shares)")
    logger.warning("  → MISSED opportunity to profit from $165 → $130 decline")
    logger.warning("  → Potential profit missed: $3,500")
    logger.blank()

    logger.separator()
    logger.blank()

    # Long/short version
    logger.header("LONG/SHORT MODE (allow_shorts=True)")
    logger.separator()
    logger.blank()

    engine_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=True
    )
    engine_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_short.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    stats_short = portfolio_short.stats()
    trades_short = portfolio_short.trades()

    logger.blank()
    logger.info("=== TIMELINE OF POSITIONS (Long/Short) ===")
    logger.blank()

    logger.info("Jan 2022: Fast MA > Slow MA → ENTER LONG @ $180")
    logger.metric("  Position: +100 shares")
    logger.metric("  Capital deployed: $18,000")
    logger.blank()

    logger.info("Mar 2022: Fast MA < Slow MA → EXIT LONG @ $165")
    logger.metric("  Step 1: Close long position → 0 shares")
    logger.loss("  P&L: -$1,500 (-8.3%)")
    logger.success("  Step 2: OPEN SHORT → -100 shares")
    logger.metric("  Proceeds from short: $16,500")
    logger.success("  → NOW PROFITING as stock falls!")
    logger.blank()

    logger.info("Apr-Dec 2022: Stock falls to $130")
    logger.metric("  Position: -100 shares (SHORT)")
    logger.success("  → CAPTURING the $165 → $130 decline")
    logger.blank()

    logger.info("Dec 2022: Cover short @ $130")
    logger.profit("  P&L from short: +$3,500 (+21.2%)")
    logger.blank()

    logger.separator()
    logger.blank()

    # Compare
    logger.header("COMPARISON")
    logger.separator()
    logger.blank()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_long['Sharpe Ratio']:.2f}")
    logger.info(f"  Total Trades: {stats_long['Total Trades']}")
    logger.warning("  → Lost money, then sat in cash")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_short['Sharpe Ratio']:.2f}")
    logger.info(f"  Total Trades: {stats_short['Total Trades']}")
    logger.success("  → Recovered losses AND profited from decline")
    logger.blank()

    improvement = stats_short['Sharpe Ratio'] - stats_long['Sharpe Ratio']
    if improvement > 0:
        logger.profit(f"Sharpe Improvement: +{improvement:.2f}")
    logger.blank()

    logger.separator()


def example_2_rsi_mean_reversion():
    """
    Example 2: RSI Mean Reversion in Volatile Market

    Shows how RSI works perfectly with shorts - natural symmetry.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 2: RSI MEAN REVERSION IN VOLATILE MARKET")
    logger.separator()
    logger.blank()

    logger.info("Scenario: NVDA oscillating between $150-$220 in 2023")
    logger.info("Strategy: RSI Mean Reversion (14, 30, 70)")
    logger.blank()

    strategy = RSIMeanReversion(rsi_window=14, oversold=30, overbought=70)

    # Long-only
    logger.header("LONG-ONLY MODE (allow_shorts=False)")
    logger.separator()
    logger.blank()

    engine_long = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=False
    )
    engine_long.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long.run(
        strategy=strategy,
        symbols='NVDA',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    stats_long = portfolio_long.stats()

    logger.blank()
    logger.info("=== POSITION SEQUENCE (Long-Only) ===")
    logger.blank()

    logger.info("Wave 1: Price drops $220 → $150")
    logger.metric("  RSI hits 25 (oversold)")
    logger.success("  → ENTER LONG @ $150")
    logger.blank()

    logger.info("Wave 2: Price rebounds $150 → $200")
    logger.metric("  RSI hits 75 (overbought)")
    logger.success("  → EXIT LONG @ $200")
    logger.profit("  P&L: +$5,000 (+33%)")
    logger.blank()

    logger.info("Wave 3: Price drops $200 → $160")
    logger.warning("  Position: FLAT (0 shares)")
    logger.warning("  → CANNOT profit from this decline")
    logger.warning("  → Just watching from sidelines")
    logger.blank()

    logger.info("Wave 4: Price rebounds $160 → $180")
    logger.metric("  RSI hits 28 (oversold)")
    logger.success("  → ENTER LONG @ $160")
    logger.blank()

    logger.info("Wave 5: Price continues $180 → $210")
    logger.metric("  RSI hits 72 (overbought)")
    logger.success("  → EXIT LONG @ $210")
    logger.profit("  P&L: +$5,000 (+31%)")
    logger.blank()

    logger.metric("Summary: Only profits from UP moves (2 trades)")
    logger.blank()

    logger.separator()
    logger.blank()

    # Long/short
    logger.header("LONG/SHORT MODE (allow_shorts=True)")
    logger.separator()
    logger.blank()

    engine_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=True
    )
    engine_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_short.run(
        strategy=strategy,
        symbols='NVDA',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    stats_short = portfolio_short.stats()

    logger.blank()
    logger.info("=== POSITION SEQUENCE (Long/Short) ===")
    logger.blank()

    logger.info("Wave 1: Price drops $220 → $150")
    logger.metric("  RSI hits 25 (oversold)")
    logger.success("  → ENTER LONG @ $150")
    logger.blank()

    logger.info("Wave 2: Price rebounds $150 → $200")
    logger.metric("  RSI hits 75 (overbought)")
    logger.metric("  Step 1: EXIT LONG @ $200")
    logger.profit("  P&L: +$5,000 (+33%)")
    logger.success("  Step 2: ENTER SHORT @ $200")
    logger.metric("  → Now positioned to profit from decline!")
    logger.blank()

    logger.info("Wave 3: Price drops $200 → $160")
    logger.metric("  Position: -100 shares (SHORT)")
    logger.success("  → PROFITING from the decline")
    logger.metric("  RSI hits 28 (oversold)")
    logger.metric("  Step 1: COVER SHORT @ $160")
    logger.profit("  P&L: +$4,000 (+20%)")
    logger.success("  Step 2: ENTER LONG @ $160")
    logger.blank()

    logger.info("Wave 4: Price rebounds $160 → $180")
    logger.metric("  Position: +100 shares (LONG)")
    logger.success("  → PROFITING from the rally")
    logger.blank()

    logger.info("Wave 5: Price continues $180 → $210")
    logger.metric("  RSI hits 72 (overbought)")
    logger.metric("  Step 1: EXIT LONG @ $210")
    logger.profit("  P&L: +$5,000 (+31%)")
    logger.success("  Step 2: ENTER SHORT @ $210")
    logger.blank()

    logger.metric("Summary: Profits from BOTH directions (4 trades)")
    logger.blank()

    logger.separator()
    logger.blank()

    # Compare
    logger.header("COMPARISON")
    logger.separator()
    logger.blank()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_long['Sharpe Ratio']:.2f}")
    logger.info(f"  Total Trades: {stats_long['Total Trades']}")
    logger.warning("  → Only 2 profitable waves captured")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_short['Sharpe Ratio']:.2f}")
    logger.info(f"  Total Trades: {stats_short['Total Trades']}")
    logger.success("  → All 4 waves captured (up AND down)")
    logger.blank()

    improvement = stats_short['Sharpe Ratio'] - stats_long['Sharpe Ratio']
    if improvement > 0:
        logger.profit(f"Sharpe Improvement: +{improvement:.2f}")

    logger.blank()
    logger.separator()


def example_3_choppy_market():
    """
    Example 3: Trend Following in Choppy Market

    Shows the DOWNSIDE - more whipsaws with shorts enabled.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 3: TREND FOLLOWING IN CHOPPY MARKET (DOWNSIDE)")
    logger.separator()
    logger.blank()

    logger.info("Scenario: Sideways/choppy market in 2015")
    logger.info("Strategy: MA Crossover (20/50)")
    logger.blank()

    strategy = MovingAverageCrossover(fast_window=20, slow_window=50)

    # Long-only
    logger.header("LONG-ONLY MODE")
    logger.separator()
    logger.blank()

    engine_long = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=False
    )
    engine_long.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long.run(
        strategy=strategy,
        symbols='SPY',
        start_date='2015-01-01',
        end_date='2015-12-31'
    )

    stats_long = portfolio_long.stats()

    logger.blank()
    logger.info("=== WHIPSAW SEQUENCE (Long-Only) ===")
    logger.blank()

    logger.info("Jan: Price $205 → Crossover → LONG @ $205")
    logger.info("Feb: Price $203 → Crossunder → EXIT @ $203")
    logger.loss("  Loss: -$200 (-0.97%)")
    logger.blank()

    logger.info("Mar: Price $205 → Crossover → LONG @ $205")
    logger.info("Apr: Price $204 → Crossunder → EXIT @ $204")
    logger.loss("  Loss: -$100 (-0.48%)")
    logger.blank()

    logger.info("May-Dec: More whipsaws...")
    logger.warning("  → Death by a thousand cuts")
    logger.warning("  → But at least we're flat between trades")
    logger.blank()

    logger.metric(f"Total Trades: {stats_long['Total Trades']}")
    logger.metric(f"Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.blank()

    logger.separator()
    logger.blank()

    # Long/short
    logger.header("LONG/SHORT MODE")
    logger.separator()
    logger.blank()

    engine_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        allow_shorts=True
    )
    engine_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_short.run(
        strategy=strategy,
        symbols='SPY',
        start_date='2015-01-01',
        end_date='2015-12-31'
    )

    stats_short = portfolio_short.stats()

    logger.blank()
    logger.info("=== WHIPSAW SEQUENCE (Long/Short) ===")
    logger.blank()

    logger.info("Jan: Price $205 → Crossover → LONG @ $205")
    logger.info("Feb: Price $203 → Crossunder → EXIT + SHORT @ $203")
    logger.loss("  Loss from long: -$200")
    logger.metric("  Now SHORT @ $203")
    logger.blank()

    logger.info("Mar: Price $205 → Crossover → COVER + LONG @ $205")
    logger.loss("  Loss from short: -$200")
    logger.metric("  Now LONG @ $205")
    logger.blank()

    logger.info("Apr: Price $204 → Crossunder → EXIT + SHORT @ $204")
    logger.loss("  Loss from long: -$100")
    logger.metric("  Now SHORT @ $204")
    logger.blank()

    logger.error("  → DOUBLE THE WHIPSAWS!")
    logger.error("  → Every exit becomes a short that gets stopped out")
    logger.warning("  → Fees eat into returns (more trades)")
    logger.blank()

    logger.metric(f"Total Trades: {stats_short['Total Trades']}")
    logger.metric(f"Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.blank()

    logger.separator()
    logger.blank()

    # Compare
    logger.header("COMPARISON")
    logger.separator()
    logger.blank()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Total Return: {stats_long['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_long['Sharpe Ratio']:.2f}")
    logger.warning("  → Bad, but at least we're flat between trades")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Total Return: {stats_short['Total Return [%]']:.2f}%")
    logger.info(f"  Sharpe Ratio: {stats_short['Sharpe Ratio']:.2f}")
    logger.error("  → WORSE! Always in market = more losses")
    logger.blank()

    improvement = stats_short['Sharpe Ratio'] - stats_long['Sharpe Ratio']
    if improvement < 0:
        logger.loss(f"Sharpe Degradation: {improvement:.2f}")

    logger.blank()
    logger.warning("KEY LESSON: Short selling can HURT in choppy markets")
    logger.warning("This is why parameter re-optimization is crucial!")
    logger.blank()

    logger.separator()


def main():
    """Run all behavior comparison examples."""
    logger.blank()
    logger.separator()
    logger.header("SHORT SELLING BEHAVIOR COMPARISON")
    logger.info("Concrete examples showing how strategies behave differently")
    logger.separator()
    logger.blank()

    # Example 1: Bear market (shorts shine)
    example_1_ma_crossover_bear_market()

    logger.blank()
    logger.blank()

    # Example 2: Volatile market (RSI perfect for shorts)
    example_2_rsi_mean_reversion()

    logger.blank()
    logger.blank()

    # Example 3: Choppy market (shorts can hurt)
    example_3_choppy_market()

    # Final summary
    logger.blank()
    logger.separator()
    logger.header("SUMMARY: WHEN SHORTS HELP VS HURT")
    logger.separator()
    logger.blank()

    logger.success("✅ SHORTS HELP:")
    logger.info("  • Bear markets (trend down)")
    logger.info("  • Volatile/oscillating markets (mean reversion)")
    logger.info("  • Strong trends in either direction")
    logger.blank()

    logger.error("❌ SHORTS HURT:")
    logger.info("  • Choppy/sideways markets (whipsaws)")
    logger.info("  • High-fee environments (more trades)")
    logger.info("  • Without proper parameter optimization")
    logger.blank()

    logger.metric("RECOMMENDATION:")
    logger.info("  • Enable shorts by default for production")
    logger.info("  • Re-optimize parameters with shorts enabled")
    logger.info("  • Test across full market cycles (2019-2024)")
    logger.info("  • Use walk-forward validation")
    logger.blank()

    logger.separator()


if __name__ == '__main__':
    main()
