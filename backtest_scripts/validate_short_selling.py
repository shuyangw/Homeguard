"""
Quick validation script to test short selling capability on AAPL 2022 bear market.

This script compares:
1. Long-only Bollinger Bands mean reversion
2. Long-short Bollinger Bands mean reversion (flip-flop model)

Expected result: Long-short should dramatically outperform in bear markets.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.mean_reversion import MeanReversion
from strategies.base_strategies.mean_reversion_long_short import MeanReversionLongShort
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def main():
    """
    Test Bollinger Bands with and without short selling on AAPL 2022.
    """
    logger.blank()
    logger.header("=" * 80)
    logger.header("SHORT SELLING VALIDATION TEST")
    logger.header("Testing: AAPL 2022 Bear Market (-27% year)")
    logger.header("=" * 80)
    logger.blank()

    # Optimal parameters from validation testing
    params = {
        'window': 15,
        'num_std': 3.0,
        'exit_at_middle': False
    }

    test_config = {
        'symbol': 'AAPL',
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'initial_capital': 100000,
        'fees': 0.001
    }

    risk_config = RiskConfig(
        enabled=True,
        position_sizing_method='fixed_percentage',
        position_size_pct=0.10  # 10% per trade (moderate risk)
    )

    # ========================================================================
    # TEST 1: LONG-ONLY (Baseline)
    # ========================================================================
    logger.blank()
    logger.separator()
    logger.header("TEST 1: LONG-ONLY (Baseline)")
    logger.separator()
    logger.blank()

    strategy_long_only = MeanReversion(**params)
    engine_long_only = BacktestEngine(
        initial_capital=test_config['initial_capital'],
        fees=test_config['fees'],
        risk_config=risk_config,
        allow_shorts=False  # LONG-ONLY
    )

    portfolio_long_only = engine_long_only.run(
        strategy=strategy_long_only,
        symbols=test_config['symbol'],
        start_date=test_config['start_date'],
        end_date=test_config['end_date']
    )

    # ========================================================================
    # TEST 2: LONG-SHORT (With Short Selling)
    # ========================================================================
    logger.blank()
    logger.separator()
    logger.header("TEST 2: LONG-SHORT (With Short Selling)")
    logger.separator()
    logger.blank()

    strategy_long_short = MeanReversionLongShort(**params)
    engine_long_short = BacktestEngine(
        initial_capital=test_config['initial_capital'],
        fees=test_config['fees'],
        risk_config=risk_config,
        allow_shorts=True  # ENABLE SHORTS
    )

    portfolio_long_short = engine_long_short.run(
        strategy=strategy_long_short,
        symbols=test_config['symbol'],
        start_date=test_config['start_date'],
        end_date=test_config['end_date']
    )

    # ========================================================================
    # COMPARISON ANALYSIS
    # ========================================================================
    logger.blank()
    logger.separator()
    logger.header("COMPARISON RESULTS")
    logger.separator()
    logger.blank()

    # Extract metrics from stats dictionaries
    stats_long_only = portfolio_long_only.stats()
    stats_long_short = portfolio_long_short.stats()

    long_only_return = stats_long_only.get('Total Return [%]', 0)
    long_short_return = stats_long_short.get('Total Return [%]', 0)

    long_only_sharpe = stats_long_only.get('Sharpe Ratio', 0)
    long_short_sharpe = stats_long_short.get('Sharpe Ratio', 0)

    long_only_max_dd = stats_long_only.get('Max Drawdown [%]', 0)
    long_short_max_dd = stats_long_short.get('Max Drawdown [%]', 0)

    long_only_trades = stats_long_only.get('Total Trades', 0)
    long_short_trades = stats_long_short.get('Total Trades', 0)

    long_only_win_rate = stats_long_only.get('Win Rate [%]', 0)
    long_short_win_rate = stats_long_short.get('Win Rate [%]', 0)

    # Calculate improvements
    return_improvement = ((long_short_return - long_only_return) / abs(long_only_return) * 100) if long_only_return != 0 else float('inf')
    sharpe_improvement = ((long_short_sharpe - long_only_sharpe) / abs(long_only_sharpe) * 100) if long_only_sharpe != 0 else float('inf')
    dd_improvement = ((long_short_max_dd - long_only_max_dd) / abs(long_only_max_dd) * 100) if long_only_max_dd != 0 else 0

    # Display comparison table
    logger.info("Metric                  Long-Only        Long-Short       Improvement")
    logger.info("-" * 75)
    logger.metric(f"Total Return            {long_only_return:>7.2f}%         {long_short_return:>7.2f}%         {return_improvement:>7.1f}%")
    logger.metric(f"Sharpe Ratio            {long_only_sharpe:>7.2f}          {long_short_sharpe:>7.2f}          {sharpe_improvement:>7.1f}%")
    logger.metric(f"Max Drawdown            {long_only_max_dd:>7.2f}%         {long_short_max_dd:>7.2f}%         {dd_improvement:>7.1f}%")
    logger.metric(f"Number of Trades        {long_only_trades:>7}          {long_short_trades:>7}          {long_short_trades - long_only_trades:>7}")
    logger.metric(f"Win Rate                {long_only_win_rate:>7.1f}%         {long_short_win_rate:>7.1f}%         {long_short_win_rate - long_only_win_rate:>7.1f}%")

    logger.blank()

    # Verdict
    if long_short_return > long_only_return and long_short_sharpe > long_only_sharpe:
        logger.success("=" * 75)
        logger.success("VALIDATION SUCCESSFUL: Short selling improves performance!")
        logger.success("=" * 75)
        logger.success(f"Return improved by {return_improvement:.1f}%")
        logger.success(f"Sharpe improved by {sharpe_improvement:.1f}%")
    elif long_short_return > long_only_return:
        logger.warning("=" * 75)
        logger.warning("MIXED RESULTS: Higher returns but lower risk-adjusted returns")
        logger.warning("=" * 75)
    else:
        logger.error("=" * 75)
        logger.error("VALIDATION FAILED: Short selling did not improve performance")
        logger.error("=" * 75)

    logger.blank()


if __name__ == '__main__':
    main()
