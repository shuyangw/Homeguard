"""
OMR Strategy VIX Filter Validation

CRITICAL MISSION: Definitively determine if VIX filter was active in previous backtests

This script:
1. Runs backtest WITH VIX filter (threshold=35)
2. Runs backtest WITHOUT VIX filter
3. Provides month-by-month breakdown
4. Explicitly tracks VIX filtering statistics
5. Compares against previous OMR reports to determine which scenario matches

Timeline: ~1 hour
Outputs: Comprehensive CSV and markdown reports

Author: Homeguard Quantitative Research
Date: 2025-11-17
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

# Add src to path
from src.utils.logger import logger
from src.config import get_backtest_results_dir

# Using production modules for exact parity
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel

# Data directory
DATA_DIR = Path('data/leveraged_etfs')

# Base configuration (matches previous backtests)
BASE_CONFIG = {
    'position_size': 0.15,
    'stop_loss': -0.02,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'max_positions': 3,
    'skip_regimes': ['BEAR'],
    'symbols': [
        'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
        'DFEN', 'WEBL', 'UCO', 'NAIL', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL', 'CUT'
    ]
}

# Test period: 2017-11-17 to present (matches previous reports)
TEST_START = '2017-11-17'
TEST_END = '2025-11-17'

# Training period: 2 years before test start
TRAIN_START = '2015-11-17'
TRAIN_END = '2017-11-16'


def load_data():
    """Load all downloaded data."""
    logger.info("Loading leveraged ETF data...")

    data = {}

    # Load SPY and VIX
    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    spy_df = pd.read_parquet(spy_path)
    vix_df = pd.read_parquet(vix_path)

    # Flatten multi-index columns to simple DataFrame
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    # Normalize column names to lowercase (production expects lowercase)
    spy_df.columns = [col.lower() for col in spy_df.columns]
    vix_df.columns = [col.lower() for col in vix_df.columns]

    data['SPY'] = spy_df
    data['^VIX'] = vix_df

    logger.info(f"[OK] Loaded SPY: {len(data['SPY'])} days")
    logger.info(f"[OK] Loaded VIX: {len(data['^VIX'])} days")

    # Load all symbols
    loaded_count = 0
    for symbol in BASE_CONFIG['symbols']:
        path = DATA_DIR / f'{symbol}_1d.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            # Normalize column names to lowercase (production expects lowercase)
            df.columns = [col.lower() for col in df.columns]
            data[symbol] = df
            loaded_count += 1

    logger.info(f"[OK] Loaded {loaded_count}/{len(BASE_CONFIG['symbols'])} symbols")

    return data


def run_backtest(data, config, vix_filter_enabled=True):
    """
    Run backtest with explicit VIX filter control.

    Args:
        data: Market data dictionary
        config: Backtest configuration
        vix_filter_enabled: If True, apply VIX threshold filter

    Returns:
        tuple: (trades_df, daily_returns, portfolio_value, vix_stats)
    """
    logger.info(f"\nRunning backtest - VIX Filter: {'ENABLED' if vix_filter_enabled else 'DISABLED'}")
    if vix_filter_enabled:
        logger.info(f"VIX Threshold: {config.get('vix_threshold', 35)}")

    # Get market data
    spy_data = data['SPY']
    vix_data = data['^VIX']

    # Initialize models
    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel(data_frequency='daily')

    # Train on training period
    logger.info(f"Training models: {TRAIN_START} to {TRAIN_END}")
    train_start_ts = pd.Timestamp(TRAIN_START)
    train_end_ts = pd.Timestamp(TRAIN_END)

    train_data = {}
    for symbol, df in data.items():
        if symbol in ['SPY', '^VIX']:
            train_data[symbol] = df
        else:
            train_data[symbol] = df.loc[train_start_ts:train_end_ts]

    bayesian_model.train(
        historical_data=train_data,
        regime_detector=regime_detector,
        spy_data=spy_data,
        vix_data=vix_data
    )
    logger.info("[OK] Model training complete")

    # Test period
    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts = pd.Timestamp(TEST_END)

    test_dates = pd.date_range(test_start_ts, test_end_ts, freq='D')
    test_dates = test_dates[test_dates.isin(spy_data.index)]

    logger.info(f"Testing: {TEST_START} to {TEST_END} ({len(test_dates)} trading days)")

    # Track statistics
    vix_stats = {
        'total_days_evaluated': 0,
        'days_vix_filtered': 0,
        'days_bear_filtered': 0,
        'vix_values_when_filtered': [],
        'trading_days': 0
    }

    # Run backtest
    trades = []
    portfolio_value = [100000]  # Start with $100k
    daily_returns = []

    for date in test_dates:
        vix_stats['total_days_evaluated'] += 1

        # Classify regime
        regime, confidence = regime_detector.classify_regime(spy_data, vix_data, date)

        # Skip bear regime (always applied)
        if regime in config['skip_regimes']:
            vix_stats['days_bear_filtered'] += 1
            continue

        # Check VIX (EXPLICIT CONTROL)
        vix_value = float(vix_data[vix_data.index <= date]['close'].iloc[-1])

        if vix_filter_enabled and vix_value > config.get('vix_threshold', 35):
            vix_stats['days_vix_filtered'] += 1
            vix_stats['vix_values_when_filtered'].append(vix_value)
            continue

        vix_stats['trading_days'] += 1

        # Evaluate each symbol
        day_trades = []
        for symbol in config['symbols']:
            if symbol not in data:
                continue

            symbol_data = data[symbol]

            if date not in symbol_data.index:
                continue

            today = symbol_data.loc[date]

            if isinstance(today, pd.Series):
                today_open = float(today['open'])
                today_close = float(today['close'])
            else:
                today_open = float(today['open'].iloc[0])
                today_close = float(today['close'].iloc[0])

            intraday_return = (today_close - today_open) / today_open

            if abs(intraday_return) < 0.005:
                continue

            # Get probability from model
            prob_data = bayesian_model.get_reversion_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            # Filter by quality
            if (prob_data['probability'] < config['min_win_rate'] or
                prob_data['expected_return'] < config['min_expected_return'] or
                prob_data['sample_size'] < config['min_sample_size']):
                continue

            # Calculate actual overnight return
            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['open'])
            overnight_return = (next_open - today_close) / today_close

            # Apply stop-loss
            if overnight_return < config['stop_loss']:
                overnight_return = config['stop_loss']

            day_trades.append({
                'date': date,
                'symbol': symbol,
                'regime': regime,
                'vix': vix_value,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'profitable': overnight_return > 0
            })

        # Limit to max positions
        if len(day_trades) > config['max_positions']:
            day_trades = sorted(day_trades, key=lambda x: x['probability'], reverse=True)
            day_trades = day_trades[:config['max_positions']]

        # Record trades
        trades.extend(day_trades)

        # Calculate daily portfolio return
        if day_trades:
            daily_return = sum(t['actual_return'] for t in day_trades) * config['position_size']
            daily_returns.append(daily_return)
            portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
        else:
            daily_returns.append(0)
            portfolio_value.append(portfolio_value[-1])

    return pd.DataFrame(trades), pd.Series(daily_returns), pd.Series(portfolio_value), vix_stats


def analyze_results(trades_df, daily_returns, portfolio_value):
    """Analyze backtest results."""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'final_value': 100000
        }

    # Overall metrics
    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()

    # Total return
    final_value = portfolio_value.iloc[-1]
    initial_value = portfolio_value.iloc[0]
    total_return = (final_value - initial_value) / initial_value

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cumulative = portfolio_value / initial_value
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': final_value
    }


def analyze_monthly_performance(trades_df):
    """Break down performance by month."""
    if trades_df.empty:
        return pd.DataFrame()

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')

    monthly = trades_df.groupby('month').agg({
        'actual_return': ['count', 'mean', 'sum'],
        'profitable': 'mean',
        'vix': 'mean'
    }).reset_index()

    monthly.columns = ['month', 'num_trades', 'avg_return', 'total_return', 'win_rate', 'avg_vix']
    monthly['month'] = monthly['month'].astype(str)

    return monthly


def generate_report(results_with_vix, results_without_vix, vix_stats_with, vix_stats_without):
    """Generate comprehensive comparison report."""

    report_lines = []
    report_lines.append("# OMR Strategy VIX Filter Validation Report")
    report_lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Test Period**: {TEST_START} to {TEST_END}")
    report_lines.append(f"**Training Period**: {TRAIN_START} to {TRAIN_END}")
    report_lines.append("\n---\n")

    report_lines.append("## Executive Summary\n")
    report_lines.append("This report definitively determines whether the VIX threshold filter (VIX > 35)")
    report_lines.append("was active in previous OMR backtests by comparing performance with and without the filter.\n")

    # VIX Filter Impact
    report_lines.append("\n## VIX Filter Impact Statistics\n")
    report_lines.append("### WITH VIX Filter (Threshold = 35)\n")
    report_lines.append(f"- **Total Days Evaluated**: {vix_stats_with['total_days_evaluated']}")
    report_lines.append(f"- **Days Filtered by BEAR Regime**: {vix_stats_with['days_bear_filtered']}")
    report_lines.append(f"- **Days Filtered by VIX > 35**: {vix_stats_with['days_vix_filtered']}")
    report_lines.append(f"- **Trading Days Remaining**: {vix_stats_with['trading_days']}")

    if vix_stats_with['vix_values_when_filtered']:
        avg_vix_filtered = np.mean(vix_stats_with['vix_values_when_filtered'])
        report_lines.append(f"- **Average VIX When Filtered**: {avg_vix_filtered:.2f}")

    vix_filter_pct = (vix_stats_with['days_vix_filtered'] / vix_stats_with['total_days_evaluated']) * 100
    report_lines.append(f"- **% Days Filtered by VIX**: {vix_filter_pct:.1f}%\n")

    report_lines.append("### WITHOUT VIX Filter\n")
    report_lines.append(f"- **Total Days Evaluated**: {vix_stats_without['total_days_evaluated']}")
    report_lines.append(f"- **Days Filtered by BEAR Regime**: {vix_stats_without['days_bear_filtered']}")
    report_lines.append(f"- **Days Filtered by VIX > 35**: 0 (filter disabled)")
    report_lines.append(f"- **Trading Days Remaining**: {vix_stats_without['trading_days']}\n")

    # Performance Comparison
    report_lines.append("\n## Performance Comparison\n")
    report_lines.append("| Metric | WITH VIX Filter | WITHOUT VIX Filter | Difference |")
    report_lines.append("|--------|----------------|-------------------|------------|")

    metrics = ['total_trades', 'win_rate', 'avg_return', 'total_return', 'sharpe_ratio', 'max_drawdown', 'final_value']

    for metric in metrics:
        with_val = results_with_vix[metric]
        without_val = results_without_vix[metric]

        if metric == 'win_rate':
            with_str = f"{with_val*100:.1f}%"
            without_str = f"{without_val*100:.1f}%"
            diff = f"{(with_val - without_val)*100:+.1f}pp"
        elif metric == 'total_return' or metric == 'max_drawdown':
            with_str = f"{with_val*100:.1f}%"
            without_str = f"{without_val*100:.1f}%"
            diff = f"{(with_val - without_val)*100:+.1f}pp"
        elif metric == 'avg_return':
            with_str = f"{with_val*100:.2f}%"
            without_str = f"{without_val*100:.2f}%"
            diff = f"{(with_val - without_val)*100:+.2f}%"
        elif metric == 'final_value':
            with_str = f"${with_val:,.0f}"
            without_str = f"${without_val:,.0f}"
            diff = f"${with_val - without_val:+,.0f}"
        elif metric == 'sharpe_ratio':
            with_str = f"{with_val:.2f}"
            without_str = f"{without_val:.2f}"
            diff = f"{with_val - without_val:+.2f}"
        else:
            with_str = f"{with_val:,}"
            without_str = f"{without_val:,}"
            diff = f"{with_val - without_val:+,}"

        metric_name = metric.replace('_', ' ').title()
        report_lines.append(f"| {metric_name} | {with_str} | {without_str} | {diff} |")

    report_lines.append("\n---\n")

    # Conclusion
    report_lines.append("\n## Comparison Against Previous Reports\n")
    report_lines.append("\n### Previous OMR Backtest Results (from 20251113_BEAR_REGIME_FILTER_IMPLEMENTATION.md)\n")
    report_lines.append("**Period 8 (2024-05-17 to 2025-11-10)**:\n")
    report_lines.append("- Total Return: 78.0%\n")
    report_lines.append("- Sharpe Ratio: 4.23\n")
    report_lines.append("- Win Rate: 59.7%\n")
    report_lines.append("- Total Trades: 745\n")
    report_lines.append("\n**8-Period Average (2017-2025)**:\n")
    report_lines.append("- Average Sharpe: 3.40\n")
    report_lines.append("- Average Return: 21.1%\n")
    report_lines.append("- Average Win Rate: 56.7%\n")

    report_lines.append("\n### Which Scenario Matches?\n")

    # Compare Sharpe ratios
    sharpe_diff_with = abs(results_with_vix['sharpe_ratio'] - 3.40)
    sharpe_diff_without = abs(results_without_vix['sharpe_ratio'] - 3.40)

    # Compare win rates
    wr_diff_with = abs(results_with_vix['win_rate'] - 0.567)
    wr_diff_without = abs(results_without_vix['win_rate'] - 0.567)

    if sharpe_diff_with < sharpe_diff_without and wr_diff_with < wr_diff_without:
        conclusion = "**WITH VIX FILTER**"
        confidence = "HIGH"
    elif sharpe_diff_without < sharpe_diff_with and wr_diff_without < wr_diff_with:
        conclusion = "**WITHOUT VIX FILTER**"
        confidence = "HIGH"
    else:
        conclusion = "**MIXED RESULTS**"
        confidence = "MEDIUM"

    report_lines.append(f"\n**CONCLUSION**: Previous backtests most closely match scenario {conclusion}")
    report_lines.append(f"**Confidence**: {confidence}\n")

    report_lines.append("\n---\n")

    return "\n".join(report_lines)


def main():
    """Main execution."""
    logger.info("\n" + "="*80)
    logger.info("OMR STRATEGY VIX FILTER VALIDATION")
    logger.info("="*80)
    logger.info("\nObjective: Definitively determine if VIX filter was active in previous backtests")
    logger.info(f"Test Period: {TEST_START} to {TEST_END}")
    logger.info("")

    start_time = datetime.now()

    # Load data
    data = load_data()
    if data is None:
        logger.error("Failed to load data")
        return

    # Run WITH VIX filter
    logger.info("\n" + "="*80)
    logger.info("SCENARIO 1: WITH VIX FILTER (threshold=35)")
    logger.info("="*80)

    config_with_vix = BASE_CONFIG.copy()
    config_with_vix['vix_threshold'] = 35

    trades_with, returns_with, portfolio_with, vix_stats_with = run_backtest(
        data, config_with_vix, vix_filter_enabled=True
    )
    results_with = analyze_results(trades_with, returns_with, portfolio_with)
    monthly_with = analyze_monthly_performance(trades_with)

    logger.info(f"\n[RESULTS WITH VIX FILTER]")
    logger.info(f"Total Trades: {results_with['total_trades']}")
    logger.info(f"Win Rate: {results_with['win_rate']*100:.1f}%")
    logger.info(f"Total Return: {results_with['total_return']*100:.1f}%")
    logger.info(f"Sharpe Ratio: {results_with['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results_with['max_drawdown']*100:.1f}%")

    # Run WITHOUT VIX filter
    logger.info("\n" + "="*80)
    logger.info("SCENARIO 2: WITHOUT VIX FILTER")
    logger.info("="*80)

    trades_without, returns_without, portfolio_without, vix_stats_without = run_backtest(
        data, BASE_CONFIG, vix_filter_enabled=False
    )
    results_without = analyze_results(trades_without, returns_without, portfolio_without)
    monthly_without = analyze_monthly_performance(trades_without)

    logger.info(f"\n[RESULTS WITHOUT VIX FILTER]")
    logger.info(f"Total Trades: {results_without['total_trades']}")
    logger.info(f"Win Rate: {results_without['win_rate']*100:.1f}%")
    logger.info(f"Total Return: {results_without['total_return']*100:.1f}%")
    logger.info(f"Sharpe Ratio: {results_without['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results_without['max_drawdown']*100:.1f}%")

    # Generate report
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("="*80)

    report = generate_report(results_with, results_without, vix_stats_with, vix_stats_without)

    # Save results
    results_dir = Path('reports')
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')

    # Save trades
    trades_with.to_csv(results_dir / f'{timestamp}_omr_vix_validation_WITH_filter_trades.csv', index=False)
    trades_without.to_csv(results_dir / f'{timestamp}_omr_vix_validation_WITHOUT_filter_trades.csv', index=False)

    # Save monthly breakdowns
    monthly_with.to_csv(results_dir / f'{timestamp}_omr_vix_validation_WITH_filter_monthly.csv', index=False)
    monthly_without.to_csv(results_dir / f'{timestamp}_omr_vix_validation_WITHOUT_filter_monthly.csv', index=False)

    # Save report
    report_path = Path('docs/reports') / f'{timestamp}_OMR_VIX_FILTER_VALIDATION.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"\n[OK] Report saved: {report_path}")
    logger.info(f"[OK] Trades saved: reports/{timestamp}_omr_vix_validation_*_trades.csv")
    logger.info(f"[OK] Monthly saved: reports/{timestamp}_omr_vix_validation_*_monthly.csv")

    elapsed = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed}")

    # Print key findings
    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS")
    logger.info("="*80)
    logger.info(report.split("## Comparison Against Previous Reports")[1].split("---")[0])


if __name__ == '__main__':
    main()
