"""
Bollinger Bands Long-Short Comprehensive Testing
================================================
Tests long-short version (with short selling) across all 35 combinations:
- 7 symbols: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- 5 time periods: 2019-2021, 2022, 2023-2024, 2017-2018, 2015-2016

Compares against long-only baseline to quantify short selling benefits.

OPTIMAL PARAMETERS (proven on AAPL long-only):
    window=15, num_std=3.0, exit_at_middle=False
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.mean_reversion_long_short import MeanReversionLongShort
from backtesting.utils.risk_config import RiskConfig
import pandas as pd
from datetime import datetime
from utils import logger

# OPTIMAL PARAMETERS (proven on long-only testing)
OPTIMAL_PARAMS = {
    'window': 15,
    'num_std': 3.0,
    'exit_at_middle': False
}

# ALL SYMBOLS (7 total)
ALL_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

# ALL TIME PERIODS (5 distinct regimes)
ALL_PERIODS = [
    ('2019-01-01', '2021-12-31', 'BULL_2019-2021'),
    ('2022-01-01', '2022-12-31', 'BEAR_2022'),
    ('2023-01-01', '2024-10-31', 'RECOVERY_2023-2024'),
    ('2017-01-01', '2018-12-31', 'BULL_2017-2018'),
    ('2015-01-01', '2016-12-31', 'CHOPPY_2015-2016'),
]

# LONG-ONLY BASELINE RESULTS (from validation report)
# These are used for comparison to quantify short selling improvement
BASELINE_RESULTS = {
    # Original discovery tests
    ('AAPL', 'BULL_2019-2021'): {'sharpe': 0.60, 'return': None},  # Not in report
    ('MSFT', 'BULL_2019-2021'): {'sharpe': 0.12, 'return': None},  # Not in report

    # Phase 1: Symbol expansion (2019-2021)
    ('GOOGL', 'BULL_2019-2021'): {'sharpe': 0.33, 'return': -3.5},
    ('AMZN', 'BULL_2019-2021'): {'sharpe': 0.33, 'return': 1.4},
    ('META', 'BULL_2019-2021'): {'sharpe': 0.36, 'return': -1.5},
    ('NVDA', 'BULL_2019-2021'): {'sharpe': 0.33, 'return': 2.1},
    ('TSLA', 'BULL_2019-2021'): {'sharpe': 0.36, 'return': 1.2},

    # Phase 2: Time period expansion (AAPL only)
    ('AAPL', 'BEAR_2022'): {'sharpe': 0.31, 'return': 1.5},
    ('AAPL', 'RECOVERY_2023-2024'): {'sharpe': 0.26, 'return': -2.9},
    ('AAPL', 'BULL_2017-2018'): {'sharpe': 0.28, 'return': -4.8},
    ('AAPL', 'CHOPPY_2015-2016'): {'sharpe': 0.38, 'return': 7.1},
}


def run_backtest(symbol, start, end, period_name):
    """
    Run single backtest with long-short strategy.

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

        # Create engine with SHORT SELLING ENABLED
        engine = BacktestEngine(
            initial_capital=100000,
            fees=0.001,
            risk_config=risk_config,
            allow_shorts=True  # ENABLE SHORT SELLING
        )

        # Create long-short strategy with optimal params
        strategy = MeanReversionLongShort(**OPTIMAL_PARAMS)

        # Run backtest
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbol,
            start_date=start,
            end_date=end
        )

        # Extract stats
        stats = portfolio.stats()

        # Get baseline for comparison (if available)
        baseline_key = (symbol, period_name)
        baseline = BASELINE_RESULTS.get(baseline_key, {'sharpe': None, 'return': None})

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
            'baseline_sharpe': baseline['sharpe'],
            'baseline_return': baseline['return'],
            'status': 'SUCCESS'
        }

    except Exception as e:
        logger.error(f"Backtest failed for {symbol} {period_name}: {e}")
        baseline_key = (symbol, period_name)
        baseline = BASELINE_RESULTS.get(baseline_key, {'sharpe': None, 'return': None})

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
            'baseline_sharpe': baseline['sharpe'],
            'baseline_return': baseline['return'],
            'status': 'FAILED',
            'error': str(e)
        }


def main():
    """Run comprehensive long-short testing across all 35 combinations."""

    logger.blank()
    logger.separator('=', 80)
    logger.header("BOLLINGER BANDS LONG-SHORT COMPREHENSIVE TESTING")
    logger.info("Testing 7 symbols × 5 periods = 35 total tests")
    logger.info("Parameters: window=15, std=3.0, exit_middle=False")
    logger.success("SHORT SELLING: ENABLED")
    logger.separator('=', 80)
    logger.blank()

    all_results = []
    start_time = datetime.now()

    # ========================================================================
    # RUN ALL 35 TESTS (7 symbols × 5 periods)
    # ========================================================================

    total_tests = len(ALL_SYMBOLS) * len(ALL_PERIODS)
    test_num = 0

    for symbol in ALL_SYMBOLS:
        logger.blank()
        logger.separator('=', 80)
        logger.header(f"TESTING SYMBOL: {symbol}")
        logger.separator('=', 80)
        logger.blank()

        for start, end, period_name in ALL_PERIODS:
            test_num += 1

            logger.separator('-', 80)
            logger.info(f"[{test_num}/{total_tests}] {symbol} on {period_name}")
            logger.separator('-', 80)

            result = run_backtest(symbol, start, end, period_name)
            all_results.append(result)

            # Print immediate comparison if baseline available
            if result['baseline_sharpe'] is not None and result['status'] == 'SUCCESS':
                sharpe_improvement = ((result['sharpe'] - result['baseline_sharpe']) /
                                     abs(result['baseline_sharpe']) * 100) if result['baseline_sharpe'] != 0 else float('inf')

                logger.blank()
                logger.metric(f"Long-Short Sharpe: {result['sharpe']:.2f}")
                logger.metric(f"Long-Only Sharpe:  {result['baseline_sharpe']:.2f}")
                logger.success(f"Improvement: {sharpe_improvement:+.1f}%")
                logger.blank()

    # ========================================================================
    # SAVE RESULTS TO CSV
    # ========================================================================

    logger.blank()
    logger.separator('=', 80)
    logger.header("SAVING RESULTS")
    logger.separator('=', 80)
    logger.blank()

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Calculate improvement columns
    df['sharpe_improvement_pct'] = ((df['sharpe'] - df['baseline_sharpe']) /
                                    df['baseline_sharpe'].abs() * 100).round(1)

    df['return_improvement_pct'] = ((df['total_return'] - df['baseline_return']) /
                                   df['baseline_return'].abs() * 100).round(1)

    # Save to CSV
    output_file = Path(__file__).parent.parent / 'logs' / 'bollinger_bands_long_short_results.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.success(f"Results saved to: {output_file}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    logger.blank()
    logger.separator('=', 80)
    logger.header("SUMMARY STATISTICS")
    logger.separator('=', 80)
    logger.blank()

    successful_tests = df[df['status'] == 'SUCCESS']
    num_success = len(successful_tests)
    num_failed = len(df) - num_success

    logger.info(f"Total Tests:        {len(df)}")
    logger.success(f"Successful:         {num_success}")
    if num_failed > 0:
        logger.error(f"Failed:             {num_failed}")
    logger.blank()

    if num_success > 0:
        # Long-short statistics
        positive_sharpe = (successful_tests['sharpe'] > 0).sum()
        median_sharpe = successful_tests['sharpe'].median()
        mean_sharpe = successful_tests['sharpe'].mean()
        min_sharpe = successful_tests['sharpe'].min()
        max_sharpe = successful_tests['sharpe'].max()

        median_return = successful_tests['total_return'].median()
        mean_return = successful_tests['total_return'].mean()

        logger.header("Long-Short Performance:")
        logger.metric(f"Positive Sharpe:    {positive_sharpe}/{num_success} ({positive_sharpe/num_success*100:.1f}%)")
        logger.metric(f"Median Sharpe:      {median_sharpe:.2f}")
        logger.metric(f"Mean Sharpe:        {mean_sharpe:.2f}")
        logger.metric(f"Sharpe Range:       {min_sharpe:.2f} to {max_sharpe:.2f}")
        logger.metric(f"Median Return:      {median_return:+.2f}%")
        logger.metric(f"Mean Return:        {mean_return:+.2f}%")
        logger.blank()

        # Comparison with baseline (only where baseline exists)
        has_baseline = successful_tests[successful_tests['baseline_sharpe'].notna()]

        if len(has_baseline) > 0:
            logger.header("Improvement vs Long-Only Baseline:")

            median_sharpe_improvement = has_baseline['sharpe_improvement_pct'].median()
            mean_sharpe_improvement = has_baseline['sharpe_improvement_pct'].mean()

            improved_sharpe = (has_baseline['sharpe'] > has_baseline['baseline_sharpe']).sum()

            logger.metric(f"Tests with Baseline: {len(has_baseline)}")
            logger.success(f"Better Sharpe:      {improved_sharpe}/{len(has_baseline)} ({improved_sharpe/len(has_baseline)*100:.1f}%)")
            logger.metric(f"Median Improvement: {median_sharpe_improvement:+.1f}%")
            logger.metric(f"Mean Improvement:   {mean_sharpe_improvement:+.1f}%")
            logger.blank()

            # By regime analysis
            logger.header("Performance by Regime:")
            for period_name in ['BULL_2019-2021', 'BEAR_2022', 'RECOVERY_2023-2024', 'BULL_2017-2018', 'CHOPPY_2015-2016']:
                regime_tests = has_baseline[has_baseline['period'] == period_name]
                if len(regime_tests) > 0:
                    median_improvement = regime_tests['sharpe_improvement_pct'].median()
                    logger.metric(f"{period_name:20s}: {median_improvement:+7.1f}% improvement")
            logger.blank()

    # Execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    logger.info(f"Execution time: {duration:.1f} minutes")
    logger.blank()

    logger.separator('=', 80)
    logger.success("COMPREHENSIVE TESTING COMPLETE")
    logger.separator('=', 80)
    logger.blank()


if __name__ == '__main__':
    main()
