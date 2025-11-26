"""
Detailed analysis of BreakoutStrategy failure.

Run a single backtest with "best" parameters to understand:
- Trade statistics (count, win rate, avg win/loss)
- Entry/exit behavior
- Why strategy loses money consistently
"""

import sys
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.momentum import BreakoutStrategy
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def analyze_breakout_failure():
    """Analyze BreakoutStrategy failure in detail."""
    logger.blank()
    logger.separator()
    logger.header("BREAKOUT STRATEGY FAILURE ANALYSIS")
    logger.separator()
    logger.blank()

    # Best parameters from optimization (still terrible)
    params = {
        'breakout_window': 50,
        'exit_window': 20
    }

    logger.info(f"Testing 'best' parameters: {params}")
    logger.info(f"Period: 2019-01-01 to 2021-12-31 (bull market)")
    logger.info(f"Symbol: AAPL")
    logger.blank()

    # Create engine and strategy
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )
    engine.risk_config = RiskConfig.moderate()

    strategy = BreakoutStrategy(**params)

    # Run backtest
    logger.info("Running backtest...")
    portfolio = engine.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2019-01-01',
        end_date='2021-12-31'
    )

    # Get statistics
    stats = portfolio.stats()

    # Display results
    logger.blank()
    logger.separator()
    logger.header("PERFORMANCE METRICS")
    logger.separator()
    logger.blank()

    sharpe = stats.get('Sharpe Ratio', 0)
    total_return = stats.get('Total Return [%]', 0)
    max_dd = stats.get('Max Drawdown [%]', 0)

    logger.profit(f"Total Return: {total_return:.2f}%")
    logger.profit(f"Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"Max Drawdown: {max_dd:.2f}%")
    logger.blank()

    # Trade statistics
    logger.separator()
    logger.header("TRADE STATISTICS")
    logger.separator()
    logger.blank()

    total_trades = stats.get('Total Trades', 0)
    win_rate = stats.get('Win Rate [%]', 0)
    avg_trade = stats.get('Avg Trade [%]', 0)
    best_trade = stats.get('Best Trade [%]', 0)
    worst_trade = stats.get('Worst Trade [%]', 0)

    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Average Trade: {avg_trade:.2f}%")
    logger.success(f"Best Trade: {best_trade:.2f}%")
    logger.error(f"Worst Trade: {worst_trade:.2f}%")
    logger.blank()

    # Calculate wins and losses
    if total_trades > 0:
        wins = int(total_trades * win_rate / 100)
        losses = total_trades - wins
        logger.info(f"Winning Trades: {wins}")
        logger.info(f"Losing Trades: {losses}")

    logger.blank()

    # Analysis
    logger.separator()
    logger.header("FAILURE ANALYSIS")
    logger.separator()
    logger.blank()

    logger.error("Why This Strategy Failed:")
    logger.blank()

    if total_trades > 0:
        if win_rate < 30:
            logger.error("1. POOR WIN RATE")
            logger.error(f"   Win rate {win_rate:.1f}% is far too low")
            logger.error("   Most breakouts are FALSE BREAKOUTS")
            logger.error("   Price breaks out then quickly reverses")
            logger.blank()

        if avg_trade < 0:
            logger.error("2. NEGATIVE AVERAGE TRADE")
            logger.error(f"   Avg trade {avg_trade:.2f}% is negative")
            logger.error("   Losses consistently outweigh wins")
            logger.error("   No positive expectancy")
            logger.blank()

        logger.error("3. LONG-ONLY LIMITATION")
        logger.error("   Strategy buys breakouts (new highs)")
        logger.error("   But EXITS on breakdown (new lows)")
        logger.error("   In trending market, should stay IN, not exit")
        logger.error("   Exit mechanism is fundamentally flawed")
        logger.blank()

        logger.error("4. LATE ENTRY")
        logger.error("   Breakout signal comes AFTER move has started")
        logger.error("   Buy at the top of the breakout")
        logger.error("   Immediate drawdown is common")
        logger.blank()

        logger.error("5. TRANSACTION COSTS")
        logger.error(f"   {total_trades} trades × 0.15% cost = {total_trades*0.0015:.2%} total")
        logger.error("   High trade frequency erodes returns")
        logger.blank()

    else:
        logger.error("NO TRADES GENERATED")
        logger.error("Parameters are so conservative, no signals triggered")

    logger.blank()

    # Comparison
    logger.separator()
    logger.header("WHY BOLLINGER BANDS WORKS BUT BREAKOUT DOESN'T")
    logger.separator()
    logger.blank()

    logger.success("Bollinger Bands (Mean Reversion):")
    logger.success("  - Buys DIPS (price below band) = good entry")
    logger.success("  - Exits at MIDDLE (fair value) = take profit")
    logger.success("  - Works in OSCILLATING markets")
    logger.success("  - Lower trade frequency = lower costs")
    logger.blank()

    logger.error("Breakout Strategy (Momentum):")
    logger.error("  - Buys HIGHS (price above channel) = bad entry")
    logger.error("  - Exits at LOWS (price breakdown) = late exit")
    logger.error("  - Needs TRENDING markets, but exits kill trends")
    logger.error("  - Higher trade frequency = higher costs")
    logger.blank()

    logger.warning("FUNDAMENTAL PROBLEM:")
    logger.warning("Long-only breakout strategies are COUNTERPRODUCTIVE")
    logger.warning("They buy strength and sell weakness")
    logger.warning("In bull market, should HOLD strength, not sell it")
    logger.blank()

    logger.separator()

    return stats


if __name__ == '__main__':
    try:
        result = analyze_breakout_failure()
        logger.success("[SUCCESS] Analysis complete!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAILED] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
