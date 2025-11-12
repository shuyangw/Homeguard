"""
Comprehensive optimization framework for overnight mean reversion strategy.

This script systematically tests parameter combinations to find optimal configurations
that maximize risk-adjusted returns while maintaining acceptable risk levels.

Phases:
1. Parameter grid search
2. Symbol universe expansion
3. Advanced position sizing methods
4. Short selling experiments
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

DATA_DIR = Path('data/leveraged_etfs')
REPORTS_DIR = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

# All available symbols
BASELINE_SYMBOLS = [
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW',
    'TNA', 'TZA', 'SOXL', 'SOXS', 'FAS', 'FAZ',
    'LABU', 'LABD', 'TECL', 'TECS',
    'QLD', 'QID', 'SSO', 'SDS',
    'UVXY', 'SVXY', 'VIXY'
]

ADDITIONAL_SYMBOLS = [
    'CURE', 'CUT', 'ERX', 'ERY', 'RETL', 'WEBL',
    'DPST', 'DFEN', 'NAIL', 'DUST',
    'UWM', 'TWM', 'UGL', 'GLL', 'UYG', 'SKF', 'USD',
    'UCO', 'SCO', 'BIB', 'BIS'
]

ALL_SYMBOLS = BASELINE_SYMBOLS + ADDITIONAL_SYMBOLS


class SimpleRegimeDetector:
    """Regime detector for daily data."""

    def classify_regime(self, spy_data, vix_data, date):
        spy = spy_data[spy_data.index <= date].copy()
        vix = vix_data[vix_data.index <= date].copy()

        if len(spy) < 200:
            return 'SIDEWAYS', 0.5, 20.0

        spy['sma_20'] = spy['Close'].rolling(20).mean()
        spy['sma_50'] = spy['Close'].rolling(50).mean()
        spy['sma_200'] = spy['Close'].rolling(200).mean()

        current_price = float(spy['Close'].iloc[-1])
        sma_20 = float(spy['sma_20'].iloc[-1])
        sma_50 = float(spy['sma_50'].iloc[-1])
        sma_200 = float(spy['sma_200'].iloc[-1])

        if len(spy) >= 40:
            sma_20_prev = spy['sma_20'].iloc[-20]
            momentum = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev != 0 else 0
        else:
            momentum = 0

        vix_current = float(vix['Close'].iloc[-1])
        vix_lookback = vix['Close'].iloc[-252:] if len(vix) >= 252 else vix['Close']
        vix_percentile = float((vix_lookback < vix_current).sum() / len(vix_lookback) * 100)

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

        return regime, confidence, vix_current


class SimpleBayesianModel:
    """Bayesian model for daily data."""

    def __init__(self):
        self.probabilities = {}
        self.trained = False

    def train(self, historical_data, regime_detector, spy_data, vix_data, train_end_date, symbols):
        logger.info(f"Training Bayesian model on {len(symbols)} symbols...")

        for symbol in symbols:
            if symbol not in historical_data or symbol in ['SPY', '^VIX']:
                continue

            df = historical_data[symbol].copy()
            df = df[df.index <= train_end_date]

            if len(df) < 100:
                continue

            df['overnight_return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
            df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']

            regimes = []
            for date in df.index:
                result = regime_detector.classify_regime(spy_data, vix_data, date)
                regime = result[0] if isinstance(result, tuple) else result
                regimes.append(regime)

            df['regime'] = regimes
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
        logger.success(f"Training complete! Analyzed {len(self.probabilities)} symbols")

    def get_probability(self, symbol, regime, intraday_return):
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


def load_data(symbols):
    """Load data for specified symbols."""
    logger.info(f"Loading data for {len(symbols)} symbols...")

    data = {}

    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    spy_df = pd.read_parquet(spy_path)
    vix_df = pd.read_parquet(vix_path)

    # Set timestamp as index
    if 'timestamp' in spy_df.columns:
        spy_df = spy_df.set_index('timestamp')
    if 'timestamp' in vix_df.columns:
        vix_df = vix_df.set_index('timestamp')

    # Remove timezone if present (for consistent comparison)
    if spy_df.index.tz is not None:
        spy_df.index = spy_df.index.tz_localize(None)
    if vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)

    data['SPY'] = spy_df
    data['^VIX'] = vix_df

    loaded = 0
    for symbol in symbols:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'

        if file_path.exists():
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            # Remove timezone if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[symbol] = df
            loaded += 1

    logger.success(f"Loaded {loaded}/{len(symbols)} symbols")
    return data


def backtest_strategy(data, regime_detector, bayesian_model, start_date, end_date,
                      symbols, config, name=""):
    """Backtest with configurable parameters."""

    spy_data = data['SPY']
    vix_data = data['^VIX']

    test_dates = spy_data[
        (spy_data.index >= start_date) & (spy_data.index <= end_date)
    ].index

    trades = []
    skipped_bear = 0
    skipped_vix = 0
    skipped_regime = 0
    stopped_out = 0

    for date in test_dates:
        regime, confidence, vix_current = regime_detector.classify_regime(spy_data, vix_data, date)

        # Skip regimes based on config
        if regime in config.get('skip_regimes', ['BEAR']):
            skipped_regime += 1
            continue

        # Skip high VIX
        vix_threshold = config.get('vix_threshold', 35)
        if vix_threshold > 0 and vix_current > vix_threshold:
            skipped_vix += 1
            continue

        # Position limit check
        max_positions = config.get('max_concurrent_positions', 3)
        if len([t for t in trades if t['date'] == date]) >= max_positions:
            continue

        for symbol in symbols:
            if symbol not in data:
                continue

            if len([t for t in trades if t['date'] == date]) >= max_positions:
                break

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

            prob_data = bayesian_model.get_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            # Quality filters from config
            min_win_rate = config.get('min_win_rate', 0.58)
            min_expected_return = config.get('min_expected_return', 0.002)
            min_sample_size = config.get('min_sample_size', 15)

            if (prob_data['probability'] < min_win_rate or
                prob_data['expected_return'] < min_expected_return or
                prob_data['sample_size'] < min_sample_size):
                continue

            # Calculate overnight return
            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

            # Determine position size
            sizing_method = config.get('sizing_method', 'fixed')
            base_size = config.get('max_position_size', 0.10)

            if sizing_method == 'fixed':
                position_size = base_size
            elif sizing_method == 'probability_weighted':
                # Scale by probability (58% -> 0.5x, 70% -> 1.5x)
                prob = prob_data['probability']
                multiplier = ((prob - min_win_rate) / (0.70 - min_win_rate)) * 1.0 + 0.5
                multiplier = max(0.5, min(2.0, multiplier))
                position_size = base_size * multiplier
            elif sizing_method == 'expected_return_weighted':
                # Scale by expected return
                exp_ret = prob_data['expected_return']
                multiplier = (exp_ret / 0.005) * 0.5 + 0.5  # 0.5% -> 1x
                multiplier = max(0.5, min(2.0, multiplier))
                position_size = base_size * multiplier
            elif sizing_method == 'kelly':
                # Kelly Criterion: f = (p*b - q) / b
                # where p=win_rate, q=1-p, b=avg_win/avg_loss ratio
                # Simplified: use expected return as proxy
                p = prob_data['probability']
                exp_ret = prob_data['expected_return']
                # Estimate b from historical data (assume avg_win/avg_loss ~ 1.2)
                b = 1.2
                kelly_f = (p * b - (1 - p)) / b
                kelly_f = max(0, min(0.25, kelly_f))  # Cap at 25%
                # Use half-Kelly for safety
                position_size = kelly_f * 0.5
            else:
                position_size = base_size

            # Cap position size
            max_pos = config.get('max_position_size', 0.10)
            position_size = min(position_size, max_pos * 2)  # Allow up to 2x base

            # Apply stop-loss
            stopped_out_flag = False
            max_loss = config.get('max_loss_per_trade', -0.03)
            if max_loss < 0 and overnight_return < max_loss:
                overnight_return = max_loss
                stopped_out_flag = True
                stopped_out += 1

            trades.append({
                'date': date,
                'symbol': symbol,
                'regime': regime,
                'vix': vix_current,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'position_size': position_size,
                'stopped_out': stopped_out_flag,
                'profitable': overnight_return > 0
            })

    return pd.DataFrame(trades)


def analyze_results(trades_df, name="", config=None):
    """Analyze backtest results."""

    if trades_df.empty:
        return None

    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()

    trades_df['portfolio_return'] = trades_df['actual_return'] * trades_df['position_size']
    total_return = trades_df['portfolio_return'].sum()

    daily_returns = trades_df.groupby('date')['portfolio_return'].sum()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    stopped_out_count = trades_df['stopped_out'].sum()
    stopped_out_pct = stopped_out_count / total_trades if total_trades > 0 else 0

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly = trades_df.groupby('month')['portfolio_return'].sum()
    monthly_win_rate = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0

    return {
        'name': name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'monthly_win_rate': monthly_win_rate,
        'stopped_out_pct': stopped_out_pct,
        'config': config,
        'trades_df': trades_df
    }


def parameter_grid_search(data, regime_detector, bayesian_model, test_start, test_end, symbols):
    """Phase 2: Systematic parameter grid search."""

    logger.info("\n" + "="*80)
    logger.info("PHASE 2: PARAMETER GRID SEARCH")
    logger.info("="*80)

    # Define parameter grid
    param_grid = {
        'max_position_size': [0.05, 0.10, 0.15, 0.20],
        'max_loss_per_trade': [-0.02, -0.03, -0.04, -0.05, 0],  # 0 = no stop
        'min_win_rate': [0.55, 0.58, 0.60],
        'vix_threshold': [30, 35, 40, 0],  # 0 = no filter
        'max_concurrent_positions': [2, 3, 4, 5]
    }

    # Generate all combinations (but test subset first)
    all_combinations = list(product(
        param_grid['max_position_size'],
        param_grid['max_loss_per_trade'],
        param_grid['min_win_rate'],
        param_grid['vix_threshold'],
        param_grid['max_concurrent_positions']
    ))

    logger.info(f"Total possible combinations: {len(all_combinations)}")
    logger.info(f"Testing priority subset first...")

    # Priority combinations (most promising based on domain knowledge)
    priority_tests = [
        # Baseline
        (0.10, -0.03, 0.58, 35, 3),
        # More aggressive
        (0.15, -0.03, 0.58, 35, 3),
        (0.20, -0.03, 0.58, 35, 4),
        # No stop-loss
        (0.10, 0, 0.58, 35, 3),
        (0.15, 0, 0.60, 35, 3),
        # Higher quality threshold
        (0.10, -0.03, 0.60, 35, 3),
        (0.15, -0.03, 0.60, 40, 3),
        # More positions
        (0.10, -0.03, 0.58, 35, 5),
        (0.08, -0.03, 0.58, 35, 5),
        # Looser VIX
        (0.10, -0.03, 0.58, 40, 3),
        (0.15, -0.03, 0.58, 0, 3),
        # Tighter stop
        (0.15, -0.02, 0.58, 35, 3),
        (0.20, -0.02, 0.60, 35, 4),
    ]

    results = []

    for i, params in enumerate(priority_tests, 1):
        pos_size, stop_loss, min_wr, vix_thresh, max_pos = params

        config = {
            'max_position_size': pos_size,
            'max_total_exposure': 0.50,
            'max_loss_per_trade': stop_loss,
            'max_concurrent_positions': max_pos,
            'vix_threshold': vix_thresh,
            'min_win_rate': min_wr,
            'min_expected_return': 0.002,
            'min_sample_size': 15,
            'skip_regimes': ['BEAR'],
            'sizing_method': 'fixed'
        }

        name = f"Pos{int(pos_size*100)}%_Stop{abs(int(stop_loss*100)) if stop_loss < 0 else 'None'}_WR{int(min_wr*100)}_VIX{int(vix_thresh) if vix_thresh > 0 else 'Off'}_Max{max_pos}"

        logger.info(f"\n[{i}/{len(priority_tests)}] Testing: {name}")

        trades_df = backtest_strategy(
            data, regime_detector, bayesian_model,
            test_start, test_end, symbols, config, name
        )

        result = analyze_results(trades_df, name, config)

        if result:
            results.append(result)
            logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                       f"Win Rate: {result['win_rate']:.1%}, Max DD: {result['max_drawdown']:.1%}, "
                       f"Trades: {result['total_trades']}")

            # Highlight if beats baseline
            if result['total_return'] > 0.358 and result['sharpe_ratio'] > 3.64:
                logger.success(f"  *** BEATS V3 BASELINE! ***")

        # Save progress every 5 tests
        if i % 5 == 0:
            save_progress(results, "phase2_grid_search_progress.csv")

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    logger.info("\n" + "="*80)
    logger.info("TOP 5 CONFIGURATIONS BY SHARPE RATIO")
    logger.info("="*80)

    for i, r in enumerate(results[:5], 1):
        logger.success(f"{i}. {r['name']}")
        logger.info(f"   Return: {r['total_return']:.1%}, Sharpe: {r['sharpe_ratio']:.2f}, "
                   f"Win Rate: {r['win_rate']:.1%}, Max DD: {r['max_drawdown']:.1%}")

    return results


def save_progress(results, filename):
    """Save intermediate results to CSV."""
    if not results:
        return

    rows = []
    for r in results:
        row = {
            'name': r['name'],
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'total_return': r['total_return'],
            'sharpe_ratio': r['sharpe_ratio'],
            'max_drawdown': r['max_drawdown'],
            'monthly_win_rate': r['monthly_win_rate'],
            'stopped_out_pct': r['stopped_out_pct']
        }
        if r['config']:
            for key, val in r['config'].items():
                if key not in ['skip_regimes']:
                    row[f'config_{key}'] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = REPORTS_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"  Progress saved to {output_path}")


def main():
    logger.info("\n" + "="*80)
    logger.info("OVERNIGHT MEAN REVERSION - COMPREHENSIVE OPTIMIZATION")
    logger.info("="*80)

    # Load data for baseline symbols first
    data = load_data(BASELINE_SYMBOLS)
    if data is None:
        return

    # Initialize models
    logger.info("\nInitializing regime detector...")
    regime_detector = SimpleRegimeDetector()

    # Train model
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    logger.info(f"\nTraining period: 2015 - 2023")
    logger.info(f"Testing period: 2024 - {test_end.date()}")

    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end, BASELINE_SYMBOLS)

    # Phase 2: Parameter grid search
    results = parameter_grid_search(
        data, regime_detector, bayesian_model,
        test_start, test_end, BASELINE_SYMBOLS
    )

    # Save final results
    save_progress(results, "20251112_phase2_final_results.csv")

    # Save detailed results
    summary_path = REPORTS_DIR / "20251112_OPTIMIZATION_PHASE2_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write("# Overnight Mean Reversion - Phase 2 Parameter Optimization\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Symbols Tested**: {len(BASELINE_SYMBOLS)} (baseline universe)\n")
        f.write(f"**Test Period**: 2024-01-01 to {test_end.date()}\n")
        f.write(f"**Configurations Tested**: {len(results)}\n\n")

        f.write("## Top 10 Configurations by Sharpe Ratio\n\n")
        f.write("| Rank | Config | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|------|--------|--------|--------|----------|--------|--------|\n")

        for i, r in enumerate(results[:10], 1):
            f.write(f"| {i} | {r['name']} | {r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | "
                   f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['total_trades']} |\n")

        # Top by return
        results_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)
        f.write("\n## Top 10 Configurations by Total Return\n\n")
        f.write("| Rank | Config | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|------|--------|--------|--------|----------|--------|--------|\n")

        for i, r in enumerate(results_by_return[:10], 1):
            f.write(f"| {i} | {r['name']} | {r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | "
                   f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['total_trades']} |\n")

    logger.success(f"\nSaved summary to {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
