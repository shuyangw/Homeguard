"""
5-Year Backtest: 5x15% vs 3x15% Position Limit Comparison

Tests two configurations over 5 years with detailed monthly tracking:
- Current Production: 3 positions × 15% = 45% exposure
- Original Spec: 5 positions × 15% = 75% exposure

Generates comprehensive monthly performance reports.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
warnings.filterwarnings('ignore')

from src.utils.logger import logger
from src.trading.config import get_production_symbols

# Import components from existing validation script
from validate_overnight_strategy_v3_full_universe import (
    SimpleRegimeDetector,
    SimpleBayesianModel,
    load_data as load_data_base,
    VIX_THRESHOLD,
    MAX_LOSS_PER_TRADE
)

DATA_DIR = Path('data/leveraged_etfs')

# Test configurations
CONFIGS = [
    {
        'name': 'Current Production (3x15%)',
        'max_positions': 3,
        'position_size': 0.15,
        'max_exposure': 0.45
    },
    {
        'name': 'Original Spec (5x15%)',
        'max_positions': 5,
        'position_size': 0.15,
        'max_exposure': 0.75
    }
]


def load_data(symbols):
    """Load historical data for all symbols."""
    logger.info(f"Loading data for {len(symbols)} symbols...")

    data = {}

    # Load SPY and VIX
    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    data['SPY'] = pd.read_parquet(spy_path)
    data['^VIX'] = pd.read_parquet(vix_path)

    # Load symbol data
    loaded = 0
    for symbol in symbols:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'

        if file_path.exists():
            data[symbol] = pd.read_parquet(file_path)
            loaded += 1
        else:
            logger.warning(f"  {symbol} data not found")

    logger.success(f"Loaded {loaded}/{len(symbols)} symbols")
    return data


def run_backtest(data, spy_data, vix_data, regime_detector, bayesian_model,
                 start_date, end_date, symbols, max_positions, position_size, name=""):
    """
    Run backtest with specified position limit configuration.

    Args:
        data: Historical price data
        spy_data: SPY data for regime detection
        vix_data: VIX data for regime detection
        regime_detector: Regime classification model
        bayesian_model: Probability model
        start_date: Backtest start date
        end_date: Backtest end date
        symbols: List of symbols to trade
        max_positions: Maximum concurrent positions
        position_size: Position size as percentage (e.g., 0.15 = 15%)
        name: Configuration name for logging

    Returns:
        DataFrame of trades
    """
    logger.info(f"\nBacktesting {name}")
    logger.info(f"  Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Max Positions: {max_positions}")
    logger.info(f"  Position Size: {position_size:.1%}")
    logger.info(f"  Max Exposure: {max_positions * position_size:.1%}")

    test_dates = spy_data[
        (spy_data.index >= start_date) & (spy_data.index <= end_date)
    ].index

    trades = []
    skipped_bear = 0
    skipped_vix = 0
    stopped_out = 0

    for date in test_dates:
        regime, confidence, vix_current = regime_detector.classify_regime(spy_data, vix_data, date)

        # Skip bear markets
        if regime == 'BEAR':
            skipped_bear += 1
            continue

        # Skip high VIX
        if vix_current > VIX_THRESHOLD:
            skipped_vix += 1
            continue

        # Collect all valid signals for this day
        daily_signals = []

        for symbol in symbols:
            if symbol not in data:
                continue

            symbol_data = data[symbol]

            if date not in symbol_data.index:
                continue

            today = symbol_data.loc[date]

            if isinstance(today, pd.Series):
                today_open = float(today['Open'])
                today_close = float(today['Close'])
            else:
                today_open = float(today['Open'].iloc[0])
                today_close = float(today['Close'].iloc[0])

            # Intraday return (open to close)
            intraday_return = (today_close - today_open) / today_open

            if abs(intraday_return) < 0.005:
                continue

            prob_data = bayesian_model.get_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            # Quality filters
            if (prob_data['probability'] < 0.58 or
                prob_data['expected_return'] < 0.002 or
                prob_data['sample_size'] < 15):
                continue

            # Calculate overnight return: close to next open
            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

            # Store signal with score (probability * expected_return)
            signal_score = prob_data['probability'] * prob_data['expected_return']

            daily_signals.append({
                'symbol': symbol,
                'score': signal_score,
                'regime': regime,
                'vix': vix_current,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'today_close': today_close,
                'next_open': next_open
            })

        # Select top N signals based on score
        if len(daily_signals) > 0:
            # Sort by score (descending) and take top max_positions
            daily_signals.sort(key=lambda x: x['score'], reverse=True)
            selected_signals = daily_signals[:max_positions]

            for signal in selected_signals:
                overnight_return = signal['actual_return']

                # Apply stop-loss
                stopped_out_flag = False
                if overnight_return < MAX_LOSS_PER_TRADE:
                    overnight_return = MAX_LOSS_PER_TRADE
                    stopped_out_flag = True
                    stopped_out += 1

                trades.append({
                    'date': date,
                    'symbol': signal['symbol'],
                    'regime': signal['regime'],
                    'vix': signal['vix'],
                    'intraday_return': signal['intraday_return'],
                    'expected_return': signal['expected_return'],
                    'probability': signal['probability'],
                    'actual_return': overnight_return,
                    'position_size': position_size,
                    'stopped_out': stopped_out_flag,
                    'profitable': overnight_return > 0
                })

    logger.info(f"  Generated {len(trades)} trades")
    logger.info(f"  Skipped {skipped_bear} days (BEAR), {skipped_vix} days (high VIX)")
    logger.info(f"  Stopped out: {stopped_out} trades")

    return pd.DataFrame(trades)


def calculate_metrics(trades_df):
    """Calculate overall performance metrics."""

    if trades_df.empty:
        return None

    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()

    # Portfolio returns (position-size weighted)
    trades_df['portfolio_return'] = trades_df['actual_return'] * trades_df['position_size']
    total_return = trades_df['portfolio_return'].sum()

    # Daily aggregated returns
    daily_returns = trades_df.groupby('date')['portfolio_return'].sum()

    # Sharpe ratio
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Stop-out rate
    stopped_out_count = trades_df['stopped_out'].sum()
    stopped_out_pct = stopped_out_count / total_trades

    # Monthly metrics
    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly_returns = trades_df.groupby('month')['portfolio_return'].sum()
    monthly_win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'monthly_win_rate': monthly_win_rate,
        'stopped_out_pct': stopped_out_pct,
        'daily_returns': daily_returns,
        'monthly_returns': monthly_returns
    }


def generate_monthly_report(trades_df, config_name):
    """Generate detailed month-by-month performance report."""

    if trades_df.empty:
        return None

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    trades_df['portfolio_return'] = trades_df['actual_return'] * trades_df['position_size']

    monthly_stats = []

    for month in trades_df['month'].unique():
        month_data = trades_df[trades_df['month'] == month]

        total_trades = len(month_data)
        wins = month_data['profitable'].sum()
        win_rate = month_data['profitable'].mean()

        monthly_return = month_data['portfolio_return'].sum()
        avg_return_per_trade = month_data['actual_return'].mean()

        # Best and worst trades
        best_trade = month_data['actual_return'].max()
        worst_trade = month_data['actual_return'].min()

        # Most traded symbols
        symbol_counts = month_data['symbol'].value_counts()
        top_symbol = symbol_counts.index[0] if len(symbol_counts) > 0 else 'N/A'

        monthly_stats.append({
            'Month': str(month),
            'Trades': total_trades,
            'Wins': wins,
            'Win Rate': win_rate,
            'Monthly Return': monthly_return,
            'Avg Return/Trade': avg_return_per_trade,
            'Best Trade': best_trade,
            'Worst Trade': worst_trade,
            'Top Symbol': top_symbol,
            'Top Symbol Count': symbol_counts.iloc[0] if len(symbol_counts) > 0 else 0
        })

    return pd.DataFrame(monthly_stats)


def compare_configurations(results):
    """Compare results between configurations."""

    print("\n" + "="*100)
    print("5-YEAR BACKTEST COMPARISON: 3x15% vs 5x15%")
    print("="*100)

    # Overall performance comparison
    comparison = []
    for result in results:
        config = result['config']
        metrics = result['metrics']

        comparison.append({
            'Configuration': config['name'],
            'Max Positions': config['max_positions'],
            'Position Size': f"{config['position_size']:.1%}",
            'Total Exposure': f"{config['max_exposure']:.1%}",
            'Total Trades': f"{metrics['total_trades']:,}",
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'Total Return': f"{metrics['total_return']:.1%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.1%}",
            'Monthly Win Rate': f"{metrics['monthly_win_rate']:.1%}",
            'Stop-Out Rate': f"{metrics['stopped_out_pct']:.1%}"
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    # Winner analysis
    print("\n" + "="*100)
    print("WINNER ANALYSIS")
    print("="*100)

    config_3x = results[0]
    config_5x = results[1]

    metrics_3x = config_3x['metrics']
    metrics_5x = config_5x['metrics']

    print(f"\nSharpe Ratio:")
    print(f"  3x15%: {metrics_3x['sharpe_ratio']:.2f}")
    print(f"  5x15%: {metrics_5x['sharpe_ratio']:.2f}")
    if metrics_5x['sharpe_ratio'] > metrics_3x['sharpe_ratio']:
        improvement = ((metrics_5x['sharpe_ratio'] - metrics_3x['sharpe_ratio']) /
                      metrics_3x['sharpe_ratio'] * 100)
        print(f"  Winner: 5x15% (+{improvement:.1f}%)")
    else:
        decline = ((metrics_3x['sharpe_ratio'] - metrics_5x['sharpe_ratio']) /
                  metrics_3x['sharpe_ratio'] * 100)
        print(f"  Winner: 3x15% (5x15% is {decline:.1f}% worse)")

    print(f"\nTotal Return:")
    print(f"  3x15%: {metrics_3x['total_return']:.1%}")
    print(f"  5x15%: {metrics_5x['total_return']:.1%}")
    diff = metrics_5x['total_return'] - metrics_3x['total_return']
    print(f"  Difference: {diff:+.1%}")

    print(f"\nMax Drawdown (lower is better):")
    print(f"  3x15%: {metrics_3x['max_drawdown']:.1%}")
    print(f"  5x15%: {metrics_5x['max_drawdown']:.1%}")
    if abs(metrics_5x['max_drawdown']) < abs(metrics_3x['max_drawdown']):
        print(f"  Winner: 5x15% (better risk control)")
    else:
        print(f"  Winner: 3x15% (better risk control)")

    print(f"\nWin Rate:")
    print(f"  3x15%: {metrics_3x['win_rate']:.1%}")
    print(f"  5x15%: {metrics_5x['win_rate']:.1%}")
    if metrics_5x['win_rate'] > metrics_3x['win_rate']:
        print(f"  Winner: 5x15%")
    else:
        print(f"  Winner: 3x15%")

    return df


def main():
    """Run 5-year backtest comparison."""

    print("\n" + "="*100)
    print("5-YEAR BACKTEST: 3x15% vs 5x15% POSITION LIMIT ANALYSIS")
    print("="*100)
    print("Testing period: Approximately 5 years (based on available data)")
    print("="*100)

    # Load production symbols
    symbols = get_production_symbols()
    logger.info(f"\nUsing {len(symbols)} production symbols: {symbols}")

    # Load data
    data = load_data(symbols + ['SPY', '^VIX'])

    if data is None:
        logger.error("Failed to load data!")
        return 1

    spy_data = data['SPY']
    vix_data = data['^VIX']

    # Determine date range (last 5 years of available data)
    max_date = spy_data.index.max()
    min_date = spy_data.index.min()

    # Use last 5 years or all available data (whichever is shorter)
    start_date = max(min_date, max_date - pd.DateOffset(years=5))
    end_date = max_date

    logger.info(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {(end_date - start_date).days}")

    # Train models (use first 60% of data for training)
    train_end_date = start_date + (end_date - start_date) * 0.6
    test_start_date = train_end_date + pd.DateOffset(days=1)

    logger.info(f"\nTraining period: {start_date.date()} to {train_end_date.date()}")
    logger.info(f"Testing period: {test_start_date.date()} to {end_date.date()}")

    # Initialize models
    regime_detector = SimpleRegimeDetector()
    bayesian_model = SimpleBayesianModel()

    # Train Bayesian model
    bayesian_model.train(
        historical_data=data,
        regime_detector=regime_detector,
        spy_data=spy_data,
        vix_data=vix_data,
        train_end_date=train_end_date,
        symbols=symbols
    )

    # Run backtests for both configurations
    results = []

    for config in CONFIGS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Configuration: {config['name']}")
        logger.info(f"{'='*80}")

        trades_df = run_backtest(
            data=data,
            spy_data=spy_data,
            vix_data=vix_data,
            regime_detector=regime_detector,
            bayesian_model=bayesian_model,
            start_date=test_start_date,
            end_date=end_date,
            symbols=symbols,
            max_positions=config['max_positions'],
            position_size=config['position_size'],
            name=config['name']
        )

        if trades_df.empty:
            logger.error(f"No trades generated for {config['name']}!")
            continue

        # Calculate metrics
        metrics = calculate_metrics(trades_df)

        # Generate monthly report
        monthly_report = generate_monthly_report(trades_df, config['name'])

        results.append({
            'config': config,
            'trades': trades_df,
            'metrics': metrics,
            'monthly_report': monthly_report
        })

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY: {config['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"  Total Trades: {metrics['total_trades']:,}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
        logger.info(f"  Total Return: {metrics['total_return']:.1%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
        logger.info(f"  Monthly Win Rate: {metrics['monthly_win_rate']:.1%}")

    if len(results) < 2:
        logger.error("Insufficient results for comparison!")
        return 1

    # Compare configurations
    comparison_df = compare_configurations(results)

    # Save detailed reports
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save comparison summary
    comparison_file = output_dir / f'{timestamp}_5YEAR_POSITION_LIMIT_COMPARISON.csv'
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nComparison saved to: {comparison_file}")

    # Save monthly reports
    for result in results:
        config_name = result['config']['name'].replace(' ', '_').replace('(', '').replace(')', '')
        monthly_file = output_dir / f'{timestamp}_MONTHLY_{config_name}.csv'
        result['monthly_report'].to_csv(monthly_file, index=False)
        logger.info(f"Monthly report saved to: {monthly_file}")

    # Generate detailed markdown report
    report_file = output_dir / f'{timestamp}_5YEAR_BACKTEST_REPORT.md'
    generate_markdown_report(results, report_file, start_date, end_date, test_start_date, train_end_date)
    logger.info(f"Detailed report saved to: {report_file}")

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)

    return 0


def generate_markdown_report(results, output_file, start_date, end_date, test_start_date, train_end_date):
    """Generate comprehensive markdown report."""

    config_3x = results[0]
    config_5x = results[1]

    metrics_3x = config_3x['metrics']
    metrics_5x = config_5x['metrics']

    monthly_3x = config_3x['monthly_report']
    monthly_5x = config_5x['monthly_report']

    report = f"""# 5-Year Backtest Report: 3x15% vs 5x15% Position Limits

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

**Period Analyzed**: {start_date.date()} to {end_date.date()}
**Training Period**: {start_date.date()} to {train_end_date.date()}
**Testing Period**: {test_start_date.date()} to {end_date.date()}
**Total Days**: {(end_date - start_date).days}

### Configuration Comparison

| Metric | 3x15% (Current) | 5x15% (Original) | Winner |
|--------|-----------------|------------------|--------|
| **Max Positions** | 3 | 5 | - |
| **Position Size** | 15% | 15% | - |
| **Total Exposure** | 45% | 75% | - |
| **Total Trades** | {metrics_3x['total_trades']:,} | {metrics_5x['total_trades']:,} | {'5x15%' if metrics_5x['total_trades'] > metrics_3x['total_trades'] else '3x15%'} |
| **Win Rate** | {metrics_3x['win_rate']:.1%} | {metrics_5x['win_rate']:.1%} | {'5x15%' if metrics_5x['win_rate'] > metrics_3x['win_rate'] else '3x15%'} |
| **Total Return** | {metrics_3x['total_return']:.1%} | {metrics_5x['total_return']:.1%} | {'5x15%' if metrics_5x['total_return'] > metrics_3x['total_return'] else '3x15%'} |
| **Sharpe Ratio** | {metrics_3x['sharpe_ratio']:.2f} | {metrics_5x['sharpe_ratio']:.2f} | {'5x15%' if metrics_5x['sharpe_ratio'] > metrics_3x['sharpe_ratio'] else '3x15%'} |
| **Max Drawdown** | {metrics_3x['max_drawdown']:.1%} | {metrics_5x['max_drawdown']:.1%} | {'3x15%' if abs(metrics_3x['max_drawdown']) < abs(metrics_5x['max_drawdown']) else '5x15%'} (lower is better) |
| **Monthly Win Rate** | {metrics_3x['monthly_win_rate']:.1%} | {metrics_5x['monthly_win_rate']:.1%} | {'5x15%' if metrics_5x['monthly_win_rate'] > metrics_3x['monthly_win_rate'] else '3x15%'} |

### Recommendation

"""

    # Determine recommendation
    sharpe_winner = '5x15%' if metrics_5x['sharpe_ratio'] > metrics_3x['sharpe_ratio'] else '3x15%'
    sharpe_diff = abs(metrics_5x['sharpe_ratio'] - metrics_3x['sharpe_ratio'])
    return_diff = metrics_5x['total_return'] - metrics_3x['total_return']
    dd_worse = abs(metrics_5x['max_drawdown']) > abs(metrics_3x['max_drawdown'])

    if sharpe_winner == '5x15%' and sharpe_diff > 0.2:
        if dd_worse and abs(metrics_5x['max_drawdown']) > 0.10:
            report += f"**CONDITIONAL UPGRADE**: 5x15% offers better Sharpe ({metrics_5x['sharpe_ratio']:.2f} vs {metrics_3x['sharpe_ratio']:.2f}) "
            report += f"and higher returns (+{return_diff:.1%}), but drawdown is worse ({metrics_5x['max_drawdown']:.1%} vs {metrics_3x['max_drawdown']:.1%}). "
            report += f"Suitable for higher risk tolerance.\n\n"
        else:
            report += f"**RECOMMENDED UPGRADE**: 5x15% outperforms on Sharpe ({metrics_5x['sharpe_ratio']:.2f} vs {metrics_3x['sharpe_ratio']:.2f}) "
            report += f"with acceptable drawdown ({metrics_5x['max_drawdown']:.1%}). Additional return: +{return_diff:.1%}\n\n"
    else:
        report += f"**KEEP CURRENT**: 3x15% maintains better risk-adjusted returns (Sharpe {metrics_3x['sharpe_ratio']:.2f} vs {metrics_5x['sharpe_ratio']:.2f}). "
        report += f"While 5x15% may offer {'higher' if return_diff > 0 else 'lower'} absolute returns ({return_diff:+.1%}), "
        report += f"the risk-reward tradeoff favors current configuration.\n\n"

    report += f"""
---

## Monthly Performance: 3x15% (Current Production)

{monthly_3x.to_markdown(index=False)}

**Statistics**:
- Best Month: {monthly_3x['Monthly Return'].max():.2%}
- Worst Month: {monthly_3x['Monthly Return'].min():.2%}
- Avg Monthly Return: {monthly_3x['Monthly Return'].mean():.2%}
- Positive Months: {(monthly_3x['Monthly Return'] > 0).sum()}/{len(monthly_3x)}

---

## Monthly Performance: 5x15% (Original Spec)

{monthly_5x.to_markdown(index=False)}

**Statistics**:
- Best Month: {monthly_5x['Monthly Return'].max():.2%}
- Worst Month: {monthly_5x['Monthly Return'].min():.2%}
- Avg Monthly Return: {monthly_5x['Monthly Return'].mean():.2%}
- Positive Months: {(monthly_5x['Monthly Return'] > 0).sum()}/{len(monthly_5x)}

---

## Detailed Analysis

### Trade Volume Comparison

3x15% generated {metrics_3x['total_trades']:,} trades over the testing period, averaging {metrics_3x['total_trades'] / ((end_date - test_start_date).days / 365.25):.0f} trades per year.

5x15% generated {metrics_5x['total_trades']:,} trades, averaging {metrics_5x['total_trades'] / ((end_date - test_start_date).days / 365.25):.0f} trades per year.

The {'higher' if metrics_5x['total_trades'] > metrics_3x['total_trades'] else 'lower'} trade count for 5x15% reflects {'increased' if metrics_5x['total_trades'] > metrics_3x['total_trades'] else 'decreased'} diversification across more simultaneous positions.

### Risk-Adjusted Returns

**Sharpe Ratio**: The primary metric for risk-adjusted returns.

- 3x15%: {metrics_3x['sharpe_ratio']:.2f}
- 5x15%: {metrics_5x['sharpe_ratio']:.2f}

{'5x15% delivers superior risk-adjusted returns.' if metrics_5x['sharpe_ratio'] > metrics_3x['sharpe_ratio'] else '3x15% maintains better risk-adjusted performance.'}

### Drawdown Analysis

**Max Drawdown**: Worst peak-to-trough decline.

- 3x15%: {metrics_3x['max_drawdown']:.1%}
- 5x15%: {metrics_5x['max_drawdown']:.1%}

{'5x15% exhibits worse drawdown, indicating higher volatility during adverse periods.' if abs(metrics_5x['max_drawdown']) > abs(metrics_3x['max_drawdown']) else '5x15% maintains better drawdown control despite higher exposure.'}

---

## Conclusion

Based on {((end_date - test_start_date).days / 365.25):.1f} years of out-of-sample testing:

"""

    if sharpe_winner == '5x15%' and sharpe_diff > 0.2:
        report += f"**5x15% (Original Spec) is the recommended configuration**, offering:\n"
        report += f"- +{((metrics_5x['sharpe_ratio'] - metrics_3x['sharpe_ratio']) / metrics_3x['sharpe_ratio'] * 100):.1f}% better Sharpe ratio\n"
        report += f"- +{return_diff:.1%} additional total return\n"
        report += f"- {metrics_5x['total_trades'] - metrics_3x['total_trades']:,} more trading opportunities\n\n"
        if dd_worse:
            report += f"**Caveat**: Drawdown increases from {metrics_3x['max_drawdown']:.1f}% to {metrics_5x['max_drawdown']:.1f}%. "
            report += f"Ensure risk tolerance supports this level.\n"
    else:
        report += f"**3x15% (Current Production) should be maintained**, providing:\n"
        report += f"- Better risk-adjusted returns (Sharpe {metrics_3x['sharpe_ratio']:.2f})\n"
        report += f"- Lower drawdown ({metrics_3x['max_drawdown']:.1%})\n"
        report += f"- More conservative capital deployment (45% vs 75%)\n\n"
        report += f"While 5x15% offers {'higher' if return_diff > 0 else 'similar'} absolute returns, "
        report += f"the risk-reward profile of 3x15% is superior for long-term consistency.\n"

    report += f"""
---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Period**: {test_start_date.date()} to {end_date.date()}
**Total Testing Days**: {(end_date - test_start_date).days}
"""

    with open(output_file, 'w') as f:
        f.write(report)


if __name__ == "__main__":
    sys.exit(main())
