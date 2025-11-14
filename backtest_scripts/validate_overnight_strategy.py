"""
Validate the overnight mean reversion strategy.

This script:
1. Loads downloaded leveraged ETF data
2. Tests the market regime detector
3. Trains a simplified Bayesian model for daily data
4. Backtests the strategy on 2024 data
5. Reports performance metrics
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.config import get_backtest_results_dir

# Data directory
DATA_DIR = Path('data/leveraged_etfs')

# Trading universe
SYMBOLS = ['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SOXL', 'SOXS', 'FAS', 'FAZ', 'TECL', 'TECS']


class SimpleRegimeDetector:
    """Simplified regime detector for daily data."""

    def classify_regime(self, spy_data, vix_data, date):
        """Classify market regime based on momentum and volatility."""

        # Get data up to date
        spy = spy_data[spy_data.index <= date].copy()
        vix = vix_data[vix_data.index <= date].copy()

        if len(spy) < 200:
            return 'SIDEWAYS', 0.5

        # Calculate indicators
        spy['sma_20'] = spy['Close'].rolling(20).mean()
        spy['sma_50'] = spy['Close'].rolling(50).mean()
        spy['sma_200'] = spy['Close'].rolling(200).mean()

        # Ensure we get scalar values
        current_price = float(spy['Close'].iloc[-1])
        sma_20 = float(spy['sma_20'].iloc[-1])
        sma_50 = float(spy['sma_50'].iloc[-1])
        sma_200 = float(spy['sma_200'].iloc[-1])

        # Momentum slope (20-day change in 20-day SMA)
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

    def train(self, historical_data, regime_detector, spy_data, vix_data, train_end_date):
        """Train on historical data."""

        logger.info("Training Bayesian model on historical patterns...")

        for symbol in historical_data.keys():
            if symbol in ['SPY', '^VIX']:
                continue

            logger.info(f"  Processing {symbol}...")

            df = historical_data[symbol].copy()
            df = df[df.index <= train_end_date]

            if len(df) < 100:
                continue

            # Calculate overnight returns (close to next open)
            df['overnight_return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']

            # Calculate intraday return (open to close)
            df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']

            # Classify each day's regime
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

                # Buckets: large_down, medium_down, small_down, flat, small_up, medium_up, large_up
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

                    # Calculate metrics
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
        logger.success(f"Training complete! Analyzed {len(self.probabilities)} symbols")

    def get_probability(self, symbol, regime, intraday_return):
        """Get probability for given conditions."""

        if symbol not in self.probabilities:
            return None

        if regime not in self.probabilities[symbol]:
            return None

        # Find appropriate bucket
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

    logger.info("Loading downloaded data...")

    data = {}

    # Load SPY and VIX first
    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    data['SPY'] = pd.read_parquet(spy_path)
    data['^VIX'] = pd.read_parquet(vix_path)

    logger.success(f"  Loaded SPY: {len(data['SPY'])} bars")
    logger.success(f"  Loaded ^VIX: {len(data['^VIX'])} bars")

    # Load leveraged ETFs
    loaded = 0
    for symbol in SYMBOLS:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'

        if file_path.exists():
            data[symbol] = pd.read_parquet(file_path)
            loaded += 1
        else:
            logger.warning(f"  {symbol} data not found")

    logger.success(f"Loaded {loaded} leveraged ETF symbols")

    return data


def backtest_strategy(data, regime_detector, bayesian_model, start_date, end_date):
    """Backtest the overnight mean reversion strategy."""

    logger.info(f"\nBacktesting from {start_date} to {end_date}...")

    spy_data = data['SPY']
    vix_data = data['^VIX']

    # Get trading days in test period
    test_dates = spy_data[
        (spy_data.index >= start_date) & (spy_data.index <= end_date)
    ].index

    trades = []

    for date in test_dates:
        # Classify regime
        regime, confidence = regime_detector.classify_regime(spy_data, vix_data, date)

        # Evaluate each symbol
        for symbol in SYMBOLS:
            if symbol not in data:
                continue

            symbol_data = data[symbol]

            # Get today's data
            if date not in symbol_data.index:
                continue

            today = symbol_data.loc[date]

            # Calculate intraday return (ensure scalar)
            if isinstance(today, pd.Series):
                today_open = float(today['Open'])
                today_close = float(today['Close'])
            else:
                today_open = float(today['Open'].iloc[0])
                today_close = float(today['Close'].iloc[0])

            intraday_return = (today_close - today_open) / today_open

            # Skip if flat day
            if abs(intraday_return) < 0.005:
                continue

            # Get probability from model
            prob_data = bayesian_model.get_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            # Filter by minimum criteria
            if (prob_data['probability'] < 0.55 or
                prob_data['expected_return'] < 0.001 or
                prob_data['sample_size'] < 10):
                continue

            # Calculate actual overnight return
            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

            # Record trade
            trades.append({
                'date': date,
                'symbol': symbol,
                'regime': regime,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'profitable': overnight_return > 0
            })

    return pd.DataFrame(trades)


def analyze_results(trades_df):
    """Analyze backtest results."""

    if trades_df.empty:
        logger.error("No trades generated!")
        return None

    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)

    # Overall metrics
    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()
    total_return = trades_df['actual_return'].sum()

    # Sharpe ratio (annualized)
    daily_returns = trades_df.groupby('date')['actual_return'].sum()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total Trades: {total_trades:,}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  Avg Return per Trade: {avg_return:.3%}")
    logger.info(f"  Total Return: {total_return:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Max Drawdown: {max_dd:.1%}")

    # Performance by regime
    logger.info(f"\nPerformance by Regime:")
    regime_stats = trades_df.groupby('regime').agg({
        'profitable': ['mean', 'count'],
        'actual_return': 'mean'
    })

    for regime in regime_stats.index:
        win_rate_r = regime_stats.loc[regime, ('profitable', 'mean')]
        count_r = regime_stats.loc[regime, ('profitable', 'count')]
        avg_ret_r = regime_stats.loc[regime, ('actual_return', 'mean')]

        logger.info(f"  {regime:15} Win Rate: {win_rate_r:.1%}  Avg Return: {avg_ret_r:.3%}  Trades: {int(count_r)}")

    # Top performing symbols
    logger.info(f"\nTop Performing Symbols:")
    symbol_stats = trades_df.groupby('symbol').agg({
        'profitable': 'mean',
        'actual_return': ['mean', 'count']
    }).sort_values(('actual_return', 'mean'), ascending=False).head(10)

    for symbol in symbol_stats.index:
        win_rate_s = symbol_stats.loc[symbol, ('profitable', 'mean')]
        avg_ret_s = symbol_stats.loc[symbol, ('actual_return', 'mean')]
        count_s = symbol_stats.loc[symbol, ('actual_return', 'count')]

        logger.info(f"  {symbol:8} Win Rate: {win_rate_s:.1%}  Avg Return: {avg_ret_s:.3%}  Trades: {int(count_s)}")

    # Monthly performance
    logger.info(f"\nMonthly Returns:")
    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly = trades_df.groupby('month')['actual_return'].sum()

    for month, ret in monthly.items():
        logger.info(f"  {month}: {ret:.2%}")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def main():
    """Main validation function."""

    logger.info("\n" + "="*80)
    logger.info("OVERNIGHT MEAN REVERSION STRATEGY VALIDATION")
    logger.info("="*80)

    # Step 1: Load data
    data = load_data()
    if data is None:
        return

    # Step 2: Initialize components
    logger.info("\n[Step 1] Initializing regime detector...")
    regime_detector = SimpleRegimeDetector()

    # Test regime detector on recent data
    recent_date = data['SPY'].index[-1]
    regime, confidence = regime_detector.classify_regime(
        data['SPY'], data['^VIX'], recent_date
    )
    logger.success(f"  Current regime: {regime} (confidence: {confidence:.2f})")

    # Step 3: Train Bayesian model
    logger.info("\n[Step 2] Training Bayesian model...")

    # Train on 2015-2023 data, test on 2024
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end)

    # Step 4: Backtest on 2024 data
    logger.info("\n[Step 3] Running backtest on 2024 data...")

    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end
    )

    # Step 5: Analyze results
    logger.info("\n[Step 4] Analyzing results...")

    results = analyze_results(trades_df)

    # Step 6: Save results
    if results and not trades_df.empty:
        logger.info("\n[Step 5] Saving results...")

        # Save trades to CSV
        output_path = get_backtest_results_dir() / 'overnight_validation_trades.csv'
        output_path.parent.mkdir(exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        logger.success(f"  Saved trades to {output_path}")

        # Generate summary report
        report_path = get_backtest_results_dir() / '20251112_OVERNIGHT_VALIDATION_RESULTS.md'

        with open(report_path, 'w') as f:
            f.write("# Overnight Mean Reversion Strategy - Validation Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Test Parameters\n\n")
            f.write(f"- **Training Period**: 2015-11-16 to 2023-12-31\n")
            f.write(f"- **Test Period**: 2024-01-01 to {test_end.strftime('%Y-%m-%d')}\n")
            f.write(f"- **Symbols Tested**: {len(SYMBOLS)}\n\n")
            f.write("## Performance Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Trades | {results['total_trades']:,} |\n")
            f.write(f"| Win Rate | {results['win_rate']:.1%} |\n")
            f.write(f"| Avg Return per Trade | {results['avg_return']:.3%} |\n")
            f.write(f"| Total Return | {results['total_return']:.1%} |\n")
            f.write(f"| Sharpe Ratio | {results['sharpe_ratio']:.2f} |\n")
            f.write(f"| Max Drawdown | {results['max_drawdown']:.1%} |\n\n")

            f.write("## Comparison to Target\n\n")
            f.write(f"| Metric | Target | Actual | Status |\n")
            f.write(f"|--------|--------|--------|--------|\n")
            f.write(f"| Win Rate | 60-65% | {results['win_rate']:.1%} | ")
            f.write(f"{'[PASS]' if results['win_rate'] >= 0.60 else '[FAIL]'} |\n")
            f.write(f"| Sharpe Ratio | 2.5-3.5 | {results['sharpe_ratio']:.2f} | ")
            f.write(f"{'[PASS]' if results['sharpe_ratio'] >= 2.5 else '[REVIEW]'} |\n\n")

            f.write("## Conclusion\n\n")
            if results['win_rate'] >= 0.55 and results['sharpe_ratio'] > 1.5:
                f.write("[PASS] **Strategy shows promise!** Metrics are within acceptable range.\n")
            else:
                f.write("[REVIEW] **Strategy needs refinement.** Consider adjusting parameters or filters.\n")

        logger.success(f"  Saved report to {report_path}")

    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
