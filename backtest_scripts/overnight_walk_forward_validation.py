"""
Walk-Forward Validation for Overnight Mean Reversion Strategy

CRITICAL MISSION: Test strategy robustness across multiple time periods

This script:
1. Tests optimal config on 8 rolling train/test windows (2017-2025)
2. Calculates consistency metrics across all periods
3. Identifies failure modes and optimal conditions
4. Validates that 81% return is not just lucky timing

Timeline: ~2 hours (8 periods × 15 min each)
Reports progress every 20 minutes

Author: Homeguard Quantitative Research
Date: 2025-11-12
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

# Data directory
DATA_DIR = Path('data/leveraged_etfs')

# Optimal configuration from Phase 4 optimization
OPTIMAL_CONFIG = {
    'position_size': 0.15,
    'stop_loss': -0.02,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'vix_threshold': 35,
    'max_positions': 3,
    'skip_regimes': ['BEAR'],
    'symbols': [
        'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
        'DFEN', 'WEBL', 'UCO', 'NAIL', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL', 'CUT'
    ]
}

# Walk-forward periods: 2-year train, 6-month test, rolling
WALK_FORWARD_PERIODS = [
    {
        'name': 'Period 1',
        'train_start': '2015-11-16',
        'train_end': '2017-11-16',
        'test_start': '2017-11-17',
        'test_end': '2018-05-16'
    },
    {
        'name': 'Period 2',
        'train_start': '2016-05-16',
        'train_end': '2018-05-16',
        'test_start': '2018-05-17',
        'test_end': '2018-11-16'
    },
    {
        'name': 'Period 3',
        'train_start': '2017-11-16',
        'train_end': '2019-11-16',
        'test_start': '2019-11-17',
        'test_end': '2020-05-16'
    },
    {
        'name': 'Period 4',
        'train_start': '2018-05-16',
        'train_end': '2020-05-16',
        'test_start': '2020-05-17',
        'test_end': '2020-11-16'
    },
    {
        'name': 'Period 5',
        'train_start': '2019-11-16',
        'train_end': '2021-11-16',
        'test_start': '2021-11-17',
        'test_end': '2022-05-16'
    },
    {
        'name': 'Period 6',
        'train_start': '2020-05-16',
        'train_end': '2022-05-16',
        'test_start': '2022-05-17',
        'test_end': '2022-11-16'
    },
    {
        'name': 'Period 7',
        'train_start': '2021-11-16',
        'train_end': '2023-11-16',
        'test_start': '2023-11-17',
        'test_end': '2024-05-16'
    },
    {
        'name': 'Period 8',
        'train_start': '2023-05-16',
        'train_end': '2025-05-16',
        'test_start': '2024-05-17',
        'test_end': '2025-11-10'
    }
]


class SimpleRegimeDetector:
    """Simplified regime detector for daily data."""

    def classify_regime(self, spy_data, vix_data, date):
        """Classify market regime based on momentum and volatility."""
        spy = spy_data[spy_data.index <= date].copy()
        vix = vix_data[vix_data.index <= date].copy()

        if len(spy) < 200:
            return 'SIDEWAYS', 0.5

        # Calculate indicators
        spy['sma_20'] = spy['Close'].rolling(20).mean()
        spy['sma_50'] = spy['Close'].rolling(50).mean()
        spy['sma_200'] = spy['Close'].rolling(200).mean()

        current_price = float(spy['Close'].iloc[-1])
        sma_20 = float(spy['sma_20'].iloc[-1])
        sma_50 = float(spy['sma_50'].iloc[-1])
        sma_200 = float(spy['sma_200'].iloc[-1])

        # Momentum
        if len(spy) >= 40:
            sma_20_prev = spy['sma_20'].iloc[-20]
            momentum = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev != 0 else 0
        else:
            momentum = 0

        # VIX percentile
        vix_current = float(vix['Close'].iloc[-1])
        vix_lookback = vix['Close'].iloc[-252:] if len(vix) >= 252 else vix['Close']
        vix_percentile = float((vix_lookback < vix_current).sum() / len(vix_lookback) * 100)

        # Classify regime
        above_all = current_price > sma_20 and current_price > sma_50 and current_price > sma_200
        below_all = current_price < sma_20 and current_price < sma_50 and current_price < sma_200

        if above_all and momentum > 0.02 and vix_percentile < 30:
            regime = 'STRONG_BULL'
            confidence = 0.8
        elif above_all and vix_percentile < 50:
            regime = 'WEAK_BULL'
            confidence = 0.7
        elif below_all and vix_percentile > 70:
            regime = 'BEAR'
            confidence = 0.8
        elif vix_percentile > 60:
            regime = 'UNPREDICTABLE'
            confidence = 0.6
        else:
            regime = 'SIDEWAYS'
            confidence = 0.6

        return regime, confidence


class SimpleBayesianModel:
    """Simplified Bayesian model for daily data."""

    def __init__(self):
        self.probabilities = {}
        self.trained = False

    def train(self, historical_data, regime_detector, spy_data, vix_data, train_start, train_end):
        """Train on historical data."""
        logger.info(f"Training Bayesian model from {train_start} to {train_end}...")

        # Convert dates to timestamps
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)

        for symbol in historical_data.keys():
            if symbol in ['SPY', '^VIX']:
                continue

            df = historical_data[symbol].copy()

            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"  Skipping {symbol}: index is not DatetimeIndex (type: {type(df.index)})")
                continue

            # Filter to training period
            df = df.loc[train_start_ts:train_end_ts]

            if len(df) < 100:
                continue

            # Calculate overnight returns
            df['overnight_return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
            df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']

            # Classify regimes
            regimes = []
            for date in df.index:
                regime, _ = regime_detector.classify_regime(spy_data, vix_data, date)
                regimes.append(regime)
            df['regime'] = regimes

            # Calculate probabilities by regime and intraday move
            self.probabilities[symbol] = {}

            for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
                regime_data = df[df['regime'] == regime].copy()

                if len(regime_data) < 30:
                    continue

                buckets = [
                    ('large_down', -1.0, -0.03),
                    ('medium_down', -0.03, -0.015),
                    ('small_down', -0.015, -0.005),
                    ('flat', -0.005, 0.005),
                    ('small_up', 0.005, 0.015),
                    ('medium_up', 0.015, 0.03),
                    ('large_up', 0.03, 1.0)
                ]

                self.probabilities[symbol][regime] = {}

                for bucket_name, min_val, max_val in buckets:
                    bucket_data = regime_data[
                        (regime_data['intraday_return'] >= min_val) &
                        (regime_data['intraday_return'] < max_val)
                    ]

                    if len(bucket_data) < 10:
                        continue

                    overnight_returns = bucket_data['overnight_return'].dropna()

                    if len(overnight_returns) < 10:
                        continue

                    win_rate = (overnight_returns > 0).sum() / len(overnight_returns)
                    expected_return = overnight_returns.mean()

                    self.probabilities[symbol][regime][bucket_name] = {
                        'probability': win_rate,
                        'expected_return': expected_return,
                        'sample_size': len(overnight_returns),
                        'std': overnight_returns.std()
                    }

        self.trained = True
        logger.success(f"  Training complete! Analyzed {len(self.probabilities)} symbols")

    def get_probability(self, symbol, regime, intraday_return):
        """Get probability for given conditions."""
        if symbol not in self.probabilities:
            return None
        if regime not in self.probabilities[symbol]:
            return None

        buckets = [
            ('large_down', -1.0, -0.03),
            ('medium_down', -0.03, -0.015),
            ('small_down', -0.015, -0.005),
            ('flat', -0.005, 0.005),
            ('small_up', 0.005, 0.015),
            ('medium_up', 0.015, 0.03),
            ('large_up', 0.03, 1.0)
        ]

        for bucket_name, min_val, max_val in buckets:
            if min_val <= intraday_return < max_val:
                return self.probabilities[symbol][regime].get(bucket_name)

        return None


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

    data['SPY'] = spy_df
    data['^VIX'] = vix_df

    logger.success(f"  Loaded SPY: {len(data['SPY'])} bars ({data['SPY'].index[0]} to {data['SPY'].index[-1]})")
    logger.success(f"  Loaded ^VIX: {len(data['^VIX'])} bars")

    # Load optimal symbols
    loaded = 0
    for symbol in OPTIMAL_CONFIG['symbols']:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            data[symbol] = df
            loaded += 1

    logger.success(f"  Loaded {loaded}/{len(OPTIMAL_CONFIG['symbols'])} optimal symbols")

    return data


def backtest_period(data, regime_detector, bayesian_model, test_start, test_end, config):
    """Backtest strategy on test period."""
    spy_data = data['SPY']
    vix_data = data['^VIX']

    # Convert dates to timestamps
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Get trading days in test period
    test_dates = spy_data[(spy_data.index >= test_start_ts) & (spy_data.index <= test_end_ts)].index

    trades = []
    portfolio_value = [100000]  # Start with $100k
    daily_returns = []

    for date in test_dates:
        # Classify regime
        regime, confidence = regime_detector.classify_regime(spy_data, vix_data, date)

        # Skip bear regime
        if regime in config['skip_regimes']:
            continue

        # Check VIX
        vix_value = float(vix_data[vix_data.index <= date]['Close'].iloc[-1])
        if vix_value > config['vix_threshold']:
            continue

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
                today_open = float(today['Open'])
                today_close = float(today['Close'])
            else:
                today_open = float(today['Open'].iloc[0])
                today_close = float(today['Close'].iloc[0])

            intraday_return = (today_close - today_open) / today_open

            if abs(intraday_return) < 0.005:
                continue

            # Get probability from model
            prob_data = bayesian_model.get_probability(symbol, regime, intraday_return)

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

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

            # Apply stop-loss
            if overnight_return < config['stop_loss']:
                overnight_return = config['stop_loss']

            day_trades.append({
                'date': date,
                'symbol': symbol,
                'regime': regime,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'profitable': overnight_return > 0
            })

        # Limit to max positions
        if len(day_trades) > config['max_positions']:
            # Sort by probability and take top N
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

    return pd.DataFrame(trades), pd.Series(daily_returns), pd.Series(portfolio_value)


def analyze_results(trades_df, daily_returns, portfolio_value):
    """Analyze backtest results."""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
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


def run_walk_forward_validation():
    """Main walk-forward validation."""
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION - OVERNIGHT MEAN REVERSION STRATEGY")
    logger.info("="*80)
    logger.info("\nEstimated Runtime: 2 hours (8 periods × 15 min each)")
    logger.info("Progress will be reported every period")
    logger.info("")

    start_time = datetime.now()

    # Load data
    data = load_data()
    if data is None:
        return None

    # Initialize components
    regime_detector = SimpleRegimeDetector()

    # Results storage
    all_results = []

    # Process each period
    for i, period in enumerate(WALK_FORWARD_PERIODS, 1):
        period_start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING {period['name']} ({i}/8)")
        logger.info("="*80)
        logger.info(f"Train: {period['train_start']} to {period['train_end']}")
        logger.info(f"Test:  {period['test_start']} to {period['test_end']}")

        # Train Bayesian model on training period
        bayesian_model = SimpleBayesianModel()
        bayesian_model.train(
            data, regime_detector,
            data['SPY'], data['^VIX'],
            period['train_start'], period['train_end']
        )

        # Backtest on test period
        logger.info(f"\nBacktesting {period['name']}...")
        trades_df, daily_returns, portfolio_value = backtest_period(
            data, regime_detector, bayesian_model,
            period['test_start'], period['test_end'],
            OPTIMAL_CONFIG
        )

        # Analyze results
        results = analyze_results(trades_df, daily_returns, portfolio_value)
        results['period'] = period['name']
        results['train_start'] = period['train_start']
        results['train_end'] = period['train_end']
        results['test_start'] = period['test_start']
        results['test_end'] = period['test_end']

        all_results.append(results)

        # Log results
        period_elapsed = (datetime.now() - period_start_time).total_seconds() / 60

        logger.info(f"\n{period['name']} Results:")
        logger.info(f"  Return:     {results['total_return']*100:.1f}%")
        logger.info(f"  Sharpe:     {results['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate:   {results['win_rate']*100:.1f}%")
        logger.info(f"  Max DD:     {results['max_drawdown']*100:.1f}%")
        logger.info(f"  Trades:     {results['total_trades']}")
        logger.info(f"  Time:       {period_elapsed:.1f} minutes")

        # Progress summary
        logger.info("\n" + "-"*80)
        logger.info(f"PROGRESS: {i}/8 periods complete ({i/8*100:.0f}%)")

        if i > 0:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
            avg_return = np.mean([r['total_return'] for r in all_results])
            worst_period = min(all_results, key=lambda x: x['sharpe_ratio'])

            logger.info(f"Running Average Sharpe: {avg_sharpe:.2f}")
            logger.info(f"Running Average Return: {avg_return*100:.1f}%")
            logger.info(f"Worst Period So Far: {worst_period['period']} (Sharpe: {worst_period['sharpe_ratio']:.2f})")

            elapsed = (datetime.now() - start_time).total_seconds() / 60
            remaining = (8 - i) * (elapsed / i)
            logger.info(f"Elapsed: {elapsed:.0f} min, Estimated Remaining: {remaining:.0f} min")

        logger.info("-"*80)

    # Final analysis
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION COMPLETE")
    logger.info("="*80)

    results_df = pd.DataFrame(all_results)

    # Aggregate statistics
    logger.info("\nAGGREGATE STATISTICS:")
    logger.info(f"  Average Sharpe:  {results_df['sharpe_ratio'].mean():.2f}")
    logger.info(f"  Sharpe Std Dev:  {results_df['sharpe_ratio'].std():.2f}")
    logger.info(f"  Min Sharpe:      {results_df['sharpe_ratio'].min():.2f} ({results_df.loc[results_df['sharpe_ratio'].idxmin(), 'period']})")
    logger.info(f"  Max Sharpe:      {results_df['sharpe_ratio'].max():.2f} ({results_df.loc[results_df['sharpe_ratio'].idxmax(), 'period']})")
    logger.info("")
    logger.info(f"  Average Return:  {results_df['total_return'].mean()*100:.1f}%")
    logger.info(f"  Return Std Dev:  {results_df['total_return'].std()*100:.1f}%")
    logger.info(f"  Min Return:      {results_df['total_return'].min()*100:.1f}% ({results_df.loc[results_df['total_return'].idxmin(), 'period']})")
    logger.info(f"  Max Return:      {results_df['total_return'].max()*100:.1f}% ({results_df.loc[results_df['total_return'].idxmax(), 'period']})")
    logger.info("")
    logger.info(f"  Average Win Rate: {results_df['win_rate'].mean()*100:.1f}%")
    logger.info(f"  Win Rate Range:   {results_df['win_rate'].min()*100:.1f}% - {results_df['win_rate'].max()*100:.1f}%")
    logger.info("")
    logger.info(f"  Average Max DD:   {results_df['max_drawdown'].mean()*100:.1f}%")
    logger.info(f"  Worst DD:         {results_df['max_drawdown'].min()*100:.1f}% ({results_df.loc[results_df['max_drawdown'].idxmin(), 'period']})")

    # Success criteria assessment
    logger.info("\n" + "="*80)
    logger.info("SUCCESS CRITERIA ASSESSMENT")
    logger.info("="*80)

    avg_sharpe = results_df['sharpe_ratio'].mean()
    periods_above_55_wr = (results_df['win_rate'] > 0.55).sum()
    periods_below_neg15 = (results_df['total_return'] < -0.15).sum()
    periods_max_dd_ok = (results_df['max_drawdown'] > -0.10).sum()

    logger.info("\nTarget: Average Sharpe > 3.0")
    if avg_sharpe > 3.0:
        logger.success(f"  ✓ PASSED: {avg_sharpe:.2f} > 3.0")
    else:
        logger.error(f"  ✗ FAILED: {avg_sharpe:.2f} < 3.0")

    logger.info("\nTarget: Win Rate > 55% in at least 7/8 periods")
    if periods_above_55_wr >= 7:
        logger.success(f"  ✓ PASSED: {periods_above_55_wr}/8 periods")
    else:
        logger.error(f"  ✗ FAILED: {periods_above_55_wr}/8 periods")

    logger.info("\nTarget: No period with < -15% return")
    if periods_below_neg15 == 0:
        logger.success(f"  ✓ PASSED: All periods > -15%")
    else:
        logger.error(f"  ✗ FAILED: {periods_below_neg15} periods < -15%")

    logger.info("\nTarget: Max DD < -10% in all periods")
    if periods_max_dd_ok == 8:
        logger.success(f"  ✓ PASSED: All periods have DD > -10%")
    else:
        logger.error(f"  ✗ FAILED: {8 - periods_max_dd_ok} periods have DD < -10%")

    # Comparison to 2024-2025 baseline
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TO BASELINE (2024-2025)")
    logger.info("="*80)
    baseline_sharpe = 4.75
    baseline_return = 0.81  # 81% over 22 months

    logger.info(f"\n2024-2025 Baseline:")
    logger.info(f"  Return: {baseline_return*100:.1f}%")
    logger.info(f"  Sharpe: {baseline_sharpe:.2f}")
    logger.info(f"\nWalk-Forward Average:")
    logger.info(f"  Return: {results_df['total_return'].mean()*100:.1f}%")
    logger.info(f"  Sharpe: {avg_sharpe:.2f}")
    logger.info(f"\nInterpretation:")

    if avg_sharpe >= baseline_sharpe * 0.7:
        logger.success(f"  ✓ ROBUST: Average Sharpe is {avg_sharpe/baseline_sharpe*100:.0f}% of baseline")
        logger.success(f"  Strategy performance is consistent across time")
    else:
        logger.error(f"  ✗ OVERFITTED: Average Sharpe only {avg_sharpe/baseline_sharpe*100:.0f}% of baseline")
        logger.error(f"  2024-2025 results may have been lucky")

    # Save results
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)

    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = output_dir / '20251112_WALK_FORWARD_RESULTS.csv'
    results_df.to_csv(csv_path, index=False)
    logger.success(f"  Saved CSV: {csv_path}")

    # Save detailed report
    report_path = output_dir / '20251112_WALK_FORWARD_RESULTS.md'
    with open(report_path, 'w') as f:
        f.write("# Walk-Forward Validation Results - Overnight Mean Reversion Strategy\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Runtime**: {(datetime.now() - start_time).total_seconds() / 60:.0f} minutes\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Average Sharpe**: {avg_sharpe:.2f}\n")
        f.write(f"- **Average Return**: {results_df['total_return'].mean()*100:.1f}%\n")
        f.write(f"- **Baseline Comparison**: {avg_sharpe/baseline_sharpe*100:.0f}% of 2024-2025 Sharpe\n")
        f.write(f"- **Consistency**: Sharpe std dev = {results_df['sharpe_ratio'].std():.2f}\n\n")

        f.write("## Period-by-Period Results\n\n")
        f.write("| Period | Test Dates | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|--------|------------|--------|--------|----------|--------|--------|\n")

        for _, row in results_df.iterrows():
            f.write(f"| {row['period']} | {row['test_start']} to {row['test_end']} | "
                   f"{row['total_return']*100:.1f}% | {row['sharpe_ratio']:.2f} | "
                   f"{row['win_rate']*100:.1f}% | {row['max_drawdown']*100:.1f}% | "
                   f"{row['total_trades']} |\n")

        f.write("\n## Key Findings\n\n")

        if avg_sharpe >= 3.0:
            f.write("✓ **PASSED**: Strategy is robust across time periods\n\n")
        else:
            f.write("✗ **FAILED**: Strategy shows significant performance degradation\n\n")

        f.write(f"- Best period: {results_df.loc[results_df['sharpe_ratio'].idxmax(), 'period']} "
               f"(Sharpe: {results_df['sharpe_ratio'].max():.2f})\n")
        f.write(f"- Worst period: {results_df.loc[results_df['sharpe_ratio'].idxmin(), 'period']} "
               f"(Sharpe: {results_df['sharpe_ratio'].min():.2f})\n")
        f.write(f"- Performance consistency: {'High' if results_df['sharpe_ratio'].std() < 1.5 else 'Low'}\n")

    logger.success(f"  Saved report: {report_path}")

    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"\n✓ Total execution time: {total_time:.0f} minutes")

    return results_df


if __name__ == "__main__":
    results = run_walk_forward_validation()
