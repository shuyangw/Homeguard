"""
Bollinger Bands Comprehensive Expansion Testing
==============================================
Tests optimal parameters (window=15, std=3.0, exit_middle=False) across:
- Multiple symbols (tech mega caps)
- Multiple time periods (different market regimes)

This validates the first profitable long-only strategy discovered.

OPTIMAL PARAMETERS (proven on AAPL 2019-2021):
    window=15, num_std=3.0, exit_at_middle=False
    Sharpe: +0.60 (AAPL), +0.12 (MSFT)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.mean_reversion import MeanReversion
from backtesting.utils.risk_config import RiskConfig
import pandas as pd
from datetime import datetime
from utils import logger
from config import get_backtest_results_dir

# OPTIMAL PARAMETERS (proven on AAPL 2019-2021)
OPTIMAL_PARAMS = {
    'window': 15,
    'num_std': 3.0,
    'exit_at_middle': False
}

# Phase 1: Symbol expansion (bull market period that worked for AAPL)
PHASE1_SYMBOLS = ['GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
PHASE1_PERIOD = ('2019-01-01', '2021-12-31', 'BULL_2019-2021')

# Phase 2: Time period expansion (test AAPL across different regimes)
PHASE2_SYMBOL = 'AAPL'
PHASE2_PERIODS = [
    ('2022-01-01', '2022-12-31', 'BEAR_2022'),
    ('2023-01-01', '2024-10-31', 'RECOVERY_2023-2024'),
    ('2017-01-01', '2018-12-31', 'BULL_2017-2018'),
    ('2015-01-01', '2016-12-31', 'CHOPPY_2015-2016'),
]


def run_backtest(symbol, start, end, period_name):
    """
    Run single backtest with optimal params.

    Args:
        symbol: Stock ticker
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        period_name: Descriptive name for the period

    Returns:
        Dictionary with results or error info
    """
    try:
        # Configure moderate risk (10% per trade)
        risk_config = RiskConfig.moderate()

        # Create engine
        engine = BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            risk_config=risk_config
        )

        # Create strategy with optimal params
        strategy = MeanReversion(**OPTIMAL_PARAMS)

        # Run backtest
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbol,
            start_date=start,
            end_date=end
        )

        # Extract stats
        stats = portfolio.stats()

        return {
            'symbol': symbol,
            'period': period_name,
            'start': start,
            'end': end,
            'sharpe': stats.get('Sharpe Ratio', 0),
            'total_return': stats.get('Total Return [%]', 0),
            'annual_return': stats.get('Annual Return [%]', 0),
            'max_drawdown': stats.get('Max Drawdown [%]', 0),
            'total_trades': stats.get('Total Trades', 0),
            'win_rate': stats.get('Win Rate [%]', 0),
            'final_value': stats.get('End Value', 0),
            'status': 'SUCCESS'
        }

    except Exception as e:
        logger.error(f"Backtest failed for {symbol} {period_name}: {e}")
        return {
            'symbol': symbol,
            'period': period_name,
            'start': start,
            'end': end,
            'sharpe': None,
            'total_return': None,
            'annual_return': None,
            'max_drawdown': None,
            'total_trades': None,
            'win_rate': None,
            'final_value': None,
            'status': 'FAILED',
            'error': str(e)
        }


def main():
    """Run comprehensive expansion tests."""

    logger.blank()
    logger.separator('=', 80)
    logger.header("BOLLINGER BANDS EXPANSION TESTING")
    logger.info("Testing optimal params: window=15, std=3.0, exit_middle=False")
    logger.separator('=', 80)
    logger.blank()

    all_results = []
    start_time = datetime.now()

    # ========================================================================
    # PHASE 1: SYMBOL EXPANSION (2019-2021 BULL MARKET)
    # ========================================================================

    logger.separator('=', 80)
    logger.header("PHASE 1: SYMBOL EXPANSION (2019-2021 BULL MARKET)")
    logger.info(f"Testing {len(PHASE1_SYMBOLS)} tech mega caps in proven bull market period")
    logger.separator('=', 80)
    logger.blank()

    phase1_results = []

    for i, symbol in enumerate(PHASE1_SYMBOLS, 1):
        logger.separator('-', 80)
        logger.info(f"[{i}/{len(PHASE1_SYMBOLS)}] Testing {symbol} on {PHASE1_PERIOD[2]}...")
        logger.separator('-', 80)

        result = run_backtest(symbol, PHASE1_PERIOD[0], PHASE1_PERIOD[1], PHASE1_PERIOD[2])
        phase1_results.append(result)
        all_results.append(result)

        if result['status'] == 'SUCCESS':
            sharpe = result['sharpe']
            status_icon = "PROFITABLE" if sharpe > 0 else "UNPROFITABLE"
            logger.blank()
            logger.metric(f"Result: {status_icon} (Sharpe {sharpe:.2f})")
            logger.metric(f"  Return: {result['total_return']:.1f}%")
            logger.metric(f"  Trades: {result['total_trades']}")
            logger.metric(f"  Win Rate: {result['win_rate']:.1f}%")
        else:
            logger.error(f"  FAILED: {result['error']}")

        logger.blank()

    # ========================================================================
    # PHASE 2: TIME PERIOD EXPANSION (AAPL)
    # ========================================================================

    logger.separator('=', 80)
    logger.header("PHASE 2: TIME PERIOD EXPANSION (AAPL)")
    logger.info(f"Testing AAPL (proven winner) across {len(PHASE2_PERIODS)} different market regimes")
    logger.separator('=', 80)
    logger.blank()

    phase2_results = []

    for i, (start, end, period_name) in enumerate(PHASE2_PERIODS, 1):
        logger.separator('-', 80)
        logger.info(f"[{i}/{len(PHASE2_PERIODS)}] Testing {PHASE2_SYMBOL} on {period_name}...")
        logger.separator('-', 80)

        result = run_backtest(PHASE2_SYMBOL, start, end, period_name)
        phase2_results.append(result)
        all_results.append(result)

        if result['status'] == 'SUCCESS':
            sharpe = result['sharpe']
            status_icon = "PROFITABLE" if sharpe > 0 else "UNPROFITABLE"
            logger.blank()
            logger.metric(f"Result: {status_icon} (Sharpe {sharpe:.2f})")
            logger.metric(f"  Return: {result['total_return']:.1f}%")
            logger.metric(f"  Trades: {result['total_trades']}")
            logger.metric(f"  Win Rate: {result['win_rate']:.1f}%")
        else:
            logger.error(f"  FAILED: {result['error']}")

        logger.blank()

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================

    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = get_backtest_results_dir() / f"bollinger_bands_expansion_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    # ========================================================================
    # COMPREHENSIVE SUMMARY ANALYSIS
    # ========================================================================

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60

    logger.separator('=', 80)
    logger.header("COMPREHENSIVE SUMMARY ANALYSIS")
    logger.separator('=', 80)
    logger.blank()

    # Phase 1 Summary (Symbol Expansion)
    logger.header("PHASE 1: SYMBOL EXPANSION RESULTS")
    logger.separator('-', 80)
    phase1_df = df[df['symbol'].isin(PHASE1_SYMBOLS)]
    phase1_success = phase1_df[phase1_df['status'] == 'SUCCESS']

    if len(phase1_success) > 0:
        phase1_positive = (phase1_success['sharpe'] > 0).sum()
        phase1_total = len(phase1_success)
        phase1_pct = 100 * phase1_positive / phase1_total

        logger.metric(f"Symbols tested: {phase1_total}")
        logger.metric(f"Positive Sharpe: {phase1_positive}/{phase1_total} ({phase1_pct:.1f}%)")
        logger.metric(f"Median Sharpe: {phase1_success['sharpe'].median():.2f}")
        logger.metric(f"Mean Sharpe: {phase1_success['sharpe'].mean():.2f}")

        best_idx = phase1_success['sharpe'].idxmax()
        worst_idx = phase1_success['sharpe'].idxmin()

        logger.success(f"Best performer: {phase1_success.loc[best_idx, 'symbol']} "
                      f"(Sharpe {phase1_success.loc[best_idx, 'sharpe']:.2f}, "
                      f"Return {phase1_success.loc[best_idx, 'total_return']:.1f}%)")
        logger.warning(f"Worst performer: {phase1_success.loc[worst_idx, 'symbol']} "
                      f"(Sharpe {phase1_success.loc[worst_idx, 'sharpe']:.2f}, "
                      f"Return {phase1_success.loc[worst_idx, 'total_return']:.1f}%)")
    else:
        logger.error("No successful Phase 1 backtests")

    logger.blank()

    # Phase 2 Summary (Time Period Expansion)
    logger.header("PHASE 2: TIME PERIOD EXPANSION RESULTS")
    logger.separator('-', 80)
    phase2_df = df[df['symbol'] == PHASE2_SYMBOL]
    phase2_success = phase2_df[phase2_df['status'] == 'SUCCESS']

    if len(phase2_success) > 0:
        phase2_positive = (phase2_success['sharpe'] > 0).sum()
        phase2_total = len(phase2_success)
        phase2_pct = 100 * phase2_positive / phase2_total

        logger.metric(f"Periods tested: {phase2_total}")
        logger.metric(f"Positive Sharpe: {phase2_positive}/{phase2_total} ({phase2_pct:.1f}%)")
        logger.metric(f"Median Sharpe: {phase2_success['sharpe'].median():.2f}")
        logger.metric(f"Mean Sharpe: {phase2_success['sharpe'].mean():.2f}")

        best_idx = phase2_success['sharpe'].idxmax()
        worst_idx = phase2_success['sharpe'].idxmin()

        logger.success(f"Best period: {phase2_success.loc[best_idx, 'period']} "
                      f"(Sharpe {phase2_success.loc[best_idx, 'sharpe']:.2f}, "
                      f"Return {phase2_success.loc[best_idx, 'total_return']:.1f}%)")
        logger.warning(f"Worst period: {phase2_success.loc[worst_idx, 'period']} "
                      f"(Sharpe {phase2_success.loc[worst_idx, 'sharpe']:.2f}, "
                      f"Return {phase2_success.loc[worst_idx, 'total_return']:.1f}%)")
    else:
        logger.error("No successful Phase 2 backtests")

    logger.blank()

    # Overall Summary
    logger.header("OVERALL RESULTS")
    logger.separator('-', 80)
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        overall_positive = (success_df['sharpe'] > 0).sum()
        overall_total = len(success_df)
        overall_pct = 100 * overall_positive / overall_total

        logger.metric(f"Total tests: {overall_total}")
        logger.metric(f"Positive Sharpe: {overall_positive}/{overall_total} ({overall_pct:.1f}%)")
        logger.metric(f"Median Sharpe: {success_df['sharpe'].median():.2f}")
        logger.metric(f"Mean Sharpe: {success_df['sharpe'].mean():.2f}")
        logger.metric(f"Median Return: {success_df['total_return'].median():.1f}%")
        logger.metric(f"Median Win Rate: {success_df['win_rate'].median():.1f}%")

        # Tier classification
        logger.blank()
        logger.header("STRATEGY CLASSIFICATION")
        logger.separator('-', 80)

        median_sharpe = success_df['sharpe'].median()

        if overall_pct >= 80 and median_sharpe > 0.5:
            tier = "TIER 1: EXCEPTIONAL"
            tier_color = logger.success
        elif overall_pct >= 60 and median_sharpe > 0.3:
            tier = "TIER 2: EXCELLENT"
            tier_color = logger.success
        elif overall_pct >= 50 and median_sharpe > 0.2:
            tier = "TIER 3: GOOD"
            tier_color = logger.metric
        else:
            tier = "TIER 4: NEEDS IMPROVEMENT"
            tier_color = logger.warning

        tier_color(f"Classification: {tier}")
        logger.blank()

        # Recommendations
        logger.header("RECOMMENDATIONS")
        logger.separator('-', 80)

        if overall_pct >= 60:
            logger.success("Strategy shows ROBUST profitability across symbols/periods")
            logger.success("RECOMMEND: Proceed with live testing / paper trading")
            logger.info("Next steps: Parameter sensitivity analysis, transaction cost analysis")
        elif overall_pct >= 40:
            logger.metric("Strategy shows MODERATE profitability")
            logger.metric("RECOMMEND: Further testing on additional symbols/periods")
            logger.info("Next steps: Identify specific conditions where strategy works best")
        else:
            logger.warning("Strategy shows LIMITED profitability")
            logger.warning("RECOMMEND: Re-examine strategy logic or test different parameters")
            logger.info("Next steps: Analyze failure cases, consider regime-specific parameters")

    else:
        logger.error("No successful backtests - all tests failed")

    logger.blank()
    logger.separator('=', 80)
    logger.metric(f"Total elapsed time: {elapsed:.1f} minutes")
    logger.success(f"Results exported to: {output_path}")
    logger.separator('=', 80)
    logger.blank()


if __name__ == '__main__':
    main()
