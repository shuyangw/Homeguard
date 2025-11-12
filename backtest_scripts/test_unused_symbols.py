"""
Test All 26 Unused ETF Symbols for Overnight Mean Reversion Strategy

MISSION: Expand symbol universe from 20 to 25-35 symbols
- Test each of 26 unused symbols individually
- Measure: Sharpe ratio, win rate, return, max drawdown
- Identify symbols with Sharpe > 3.0 and win rate > 55%
- Test combinations of top performers
- Find optimal expanded universe

Timeline: ~3-4 hours total
- Phase 1: Individual testing (1.5-2 hours)
- Phase 2: Combination testing (1-1.5 hours)
- Phase 3: Portfolio optimization (30-45 minutes)

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
from typing import List, Dict
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

# Data directory
DATA_DIR = Path('data/leveraged_etfs')
PROGRESS_FILE = Path('docs/agent-learnings/20251112_unused_symbols_testing_progress.md')

# Current optimal configuration
OPTIMAL_CONFIG = {
    'position_size': 0.15,
    'stop_loss': -0.02,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'vix_threshold': 35,
    'max_positions': 3,
    'skip_regimes': ['BEAR']
}

# Current 20 optimal symbols
CURRENT_SYMBOLS = [
    'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
    'DFEN', 'WEBL', 'UCO', 'NAIL', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL', 'CUT'
]

# 26 unused symbols to test
UNUSED_SYMBOLS = [
    # High priority (5)
    'ERX', 'RETL', 'NVDL', 'TSLL', 'UVXY',
    # Bear ETFs (5)
    'TZA', 'SOXS', 'LABD', 'SDOW', 'TECS',
    # Remaining (16)
    'BIB', 'BIS', 'CURE', 'CUT', 'DPST', 'DUST', 'ERY', 'GLL',
    'QID', 'SCO', 'SDS', 'SKF', 'TWM', 'UGL', 'UWM', 'VIXY'
]

# Remove duplicates (ERX, RETL, CUT already in CURRENT_SYMBOLS)
UNUSED_SYMBOLS = [s for s in UNUSED_SYMBOLS if s not in CURRENT_SYMBOLS]

# Walk-forward periods
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
        # Convert dates to timestamps
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)

        for symbol in historical_data.keys():
            if symbol in ['SPY', '^VIX']:
                continue

            df = historical_data[symbol].copy()

            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
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


def update_progress(message: str):
    """Append message to progress file with timestamp."""
    timestamp = datetime.now().strftime('%H:%M')
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"\n### [{timestamp}] {message}\n")
    logger.info(f"[{timestamp}] {message}")


def load_data(symbols: List[str]) -> Dict:
    """Load data for specified symbols plus SPY and VIX."""
    data = {}

    # Load SPY and VIX
    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    spy_df = pd.read_parquet(spy_path)
    vix_df = pd.read_parquet(vix_path)

    # Flatten multi-index columns
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    data['SPY'] = spy_df
    data['^VIX'] = vix_df

    # Load specified symbols
    loaded = 0
    for symbol in symbols:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            data[symbol] = df
            loaded += 1

    logger.success(f"Loaded {loaded}/{len(symbols)} symbols")
    return data


def backtest_period(data, regime_detector, bayesian_model, test_start, test_end, config):
    """Backtest strategy on test period."""
    spy_data = data['SPY']
    vix_data = data['^VIX']

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    test_dates = spy_data[(spy_data.index >= test_start_ts) & (spy_data.index <= test_end_ts)].index

    trades = []
    portfolio_value = [100000]
    daily_returns = []

    for date in test_dates:
        regime, confidence = regime_detector.classify_regime(spy_data, vix_data, date)

        if regime in config['skip_regimes']:
            continue

        vix_value = float(vix_data[vix_data.index <= date]['Close'].iloc[-1])
        if vix_value > config['vix_threshold']:
            continue

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

            prob_data = bayesian_model.get_probability(symbol, regime, intraday_return)

            if prob_data is None:
                continue

            if (prob_data['probability'] < config['min_win_rate'] or
                prob_data['expected_return'] < config['min_expected_return'] or
                prob_data['sample_size'] < config['min_sample_size']):
                continue

            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

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

        if len(day_trades) > config['max_positions']:
            day_trades = sorted(day_trades, key=lambda x: x['probability'], reverse=True)
            day_trades = day_trades[:config['max_positions']]

        trades.extend(day_trades)

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

    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()

    final_value = portfolio_value.iloc[-1]
    initial_value = portfolio_value.iloc[0]
    total_return = (final_value - initial_value) / initial_value

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

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


def test_single_symbol(symbol: str, data: Dict) -> Dict:
    """Test a single symbol across all walk-forward periods."""
    logger.info(f"\nTesting {symbol}...")

    regime_detector = SimpleRegimeDetector()
    all_results = []

    for period in WALK_FORWARD_PERIODS:
        # Train model
        bayesian_model = SimpleBayesianModel()
        bayesian_model.train(
            data, regime_detector,
            data['SPY'], data['^VIX'],
            period['train_start'], period['train_end']
        )

        # Test config with single symbol
        test_config = OPTIMAL_CONFIG.copy()
        test_config['symbols'] = [symbol]

        # Backtest
        trades_df, daily_returns, portfolio_value = backtest_period(
            data, regime_detector, bayesian_model,
            period['test_start'], period['test_end'],
            test_config
        )

        results = analyze_results(trades_df, daily_returns, portfolio_value)
        all_results.append(results)

    # Aggregate across periods
    sharpe_values = [r['sharpe_ratio'] for r in all_results if r['total_trades'] > 0]
    win_rates = [r['win_rate'] for r in all_results if r['total_trades'] > 0]
    returns = [r['total_return'] for r in all_results if r['total_trades'] > 0]
    drawdowns = [r['max_drawdown'] for r in all_results if r['total_trades'] > 0]
    total_trades_sum = sum(r['total_trades'] for r in all_results)

    if not sharpe_values:
        return {
            'symbol': symbol,
            'avg_sharpe': 0,
            'avg_win_rate': 0,
            'avg_return': 0,
            'avg_max_dd': 0,
            'total_trades': 0,
            'viable': False
        }

    return {
        'symbol': symbol,
        'avg_sharpe': np.mean(sharpe_values),
        'std_sharpe': np.std(sharpe_values),
        'min_sharpe': np.min(sharpe_values),
        'max_sharpe': np.max(sharpe_values),
        'avg_win_rate': np.mean(win_rates),
        'avg_return': np.mean(returns),
        'avg_max_dd': np.mean(drawdowns),
        'total_trades': total_trades_sum,
        'viable': np.mean(sharpe_values) > 3.0 and np.mean(win_rates) > 0.55
    }


def test_symbol_combination(symbols: List[str], data: Dict) -> Dict:
    """Test a combination of symbols."""
    logger.info(f"\nTesting combination: {len(symbols)} symbols")

    regime_detector = SimpleRegimeDetector()
    all_results = []

    for period in WALK_FORWARD_PERIODS:
        bayesian_model = SimpleBayesianModel()
        bayesian_model.train(
            data, regime_detector,
            data['SPY'], data['^VIX'],
            period['train_start'], period['train_end']
        )

        test_config = OPTIMAL_CONFIG.copy()
        test_config['symbols'] = symbols

        trades_df, daily_returns, portfolio_value = backtest_period(
            data, regime_detector, bayesian_model,
            period['test_start'], period['test_end'],
            test_config
        )

        results = analyze_results(trades_df, daily_returns, portfolio_value)
        all_results.append(results)

    sharpe_values = [r['sharpe_ratio'] for r in all_results]
    win_rates = [r['win_rate'] for r in all_results if r['total_trades'] > 0]
    returns = [r['total_return'] for r in all_results]
    total_trades_sum = sum(r['total_trades'] for r in all_results)

    return {
        'num_symbols': len(symbols),
        'avg_sharpe': np.mean(sharpe_values),
        'std_sharpe': np.std(sharpe_values),
        'avg_win_rate': np.mean(win_rates) if win_rates else 0,
        'avg_return': np.mean(returns),
        'total_trades': total_trades_sum
    }


def main():
    """Main testing workflow."""
    start_time = datetime.now()

    logger.info("\n" + "="*80)
    logger.info("TESTING 26 UNUSED ETF SYMBOLS - OVERNIGHT MEAN REVERSION STRATEGY")
    logger.info("="*80)
    logger.info("\nEstimated Runtime: 3-4 hours")
    logger.info("Phase 1: Individual testing (1.5-2 hours)")
    logger.info("Phase 2: Combination testing (1-1.5 hours)")
    logger.info("Phase 3: Portfolio optimization (30-45 minutes)")
    logger.info("")

    update_progress("Testing session started")
    update_progress(f"Testing {len(UNUSED_SYMBOLS)} unused symbols")

    # Phase 1: Test each symbol individually
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: INDIVIDUAL SYMBOL TESTING")
    logger.info("="*80)

    update_progress(f"Phase 1 started: Testing {len(UNUSED_SYMBOLS)} symbols individually")

    # Load data for all symbols
    logger.info("\nLoading data for all symbols...")
    all_symbols = UNUSED_SYMBOLS.copy()
    data = load_data(all_symbols)

    if data is None:
        logger.error("Failed to load data!")
        return

    # Test each symbol
    individual_results = []
    for i, symbol in enumerate(UNUSED_SYMBOLS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing symbol {i}/{len(UNUSED_SYMBOLS)}: {symbol}")
        logger.info(f"{'='*80}")

        result = test_single_symbol(symbol, data)
        individual_results.append(result)

        logger.info(f"\n{symbol} Results:")
        logger.info(f"  Avg Sharpe:    {result['avg_sharpe']:.2f}")
        logger.info(f"  Avg Win Rate:  {result['avg_win_rate']*100:.1f}%")
        logger.info(f"  Avg Return:    {result['avg_return']*100:.1f}%")
        logger.info(f"  Total Trades:  {result['total_trades']}")
        logger.info(f"  Viable:        {'✓ YES' if result['viable'] else '✗ NO'}")

        # Update progress every 5 symbols
        if i % 5 == 0 or i == len(UNUSED_SYMBOLS):
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            remaining = (len(UNUSED_SYMBOLS) - i) * (elapsed / i) if i > 0 else 0
            update_progress(
                f"Phase 1: {i}/{len(UNUSED_SYMBOLS)} complete ({i/len(UNUSED_SYMBOLS)*100:.0f}%). "
                f"Elapsed: {elapsed:.0f}min, Est remaining: {remaining:.0f}min"
            )

    # Save Phase 1 results
    results_df = pd.DataFrame(individual_results)
    results_df = results_df.sort_values('avg_sharpe', ascending=False)

    phase1_csv = Path('reports/20251112_INDIVIDUAL_SYMBOL_RESULTS.csv')
    phase1_csv.parent.mkdir(exist_ok=True)
    results_df.to_csv(phase1_csv, index=False)

    logger.info("\n" + "="*80)
    logger.info("PHASE 1 COMPLETE - TOP 10 PERFORMERS")
    logger.info("="*80)

    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        logger.info(
            f"{row['symbol']:6s} - Sharpe: {row['avg_sharpe']:5.2f}, "
            f"Win Rate: {row['avg_win_rate']*100:5.1f}%, "
            f"Return: {row['avg_return']*100:6.1f}%, "
            f"Trades: {row['total_trades']:4.0f}"
        )

    update_progress(f"Phase 1 complete! Top performers identified. Found {len(results_df[results_df['viable']])} viable symbols")

    # Phase 2: Test combinations
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: COMBINATION TESTING")
    logger.info("="*80)

    update_progress("Phase 2 started: Testing symbol combinations")

    # Get viable symbols
    viable_symbols = results_df[results_df['viable']]['symbol'].tolist()
    logger.info(f"\nViable symbols for combination testing: {len(viable_symbols)}")
    logger.info(f"Symbols: {', '.join(viable_symbols)}")

    # Test combinations: Current + Top N
    combination_results = []

    # Current 20 baseline
    logger.info("\nTesting current 20 symbols baseline...")
    current_data = load_data(CURRENT_SYMBOLS)
    baseline_result = test_symbol_combination(CURRENT_SYMBOLS, current_data)
    baseline_result['label'] = 'Current 20'
    combination_results.append(baseline_result)

    logger.info(f"\nCurrent 20 Baseline: Sharpe {baseline_result['avg_sharpe']:.2f}")

    # Test adding top performers
    for n in [5, 10, 15, 20]:
        if n > len(viable_symbols):
            break

        top_n = viable_symbols[:n]
        expanded_symbols = list(set(CURRENT_SYMBOLS + top_n))

        logger.info(f"\nTesting Current 20 + Top {n} ({len(expanded_symbols)} total symbols)...")

        # Load data for expanded universe
        expanded_data = load_data(expanded_symbols)
        result = test_symbol_combination(expanded_symbols, expanded_data)
        result['label'] = f'Current 20 + Top {n} ({len(expanded_symbols)} total)'
        combination_results.append(result)

        logger.info(f"  Sharpe: {result['avg_sharpe']:.2f} (vs {baseline_result['avg_sharpe']:.2f} baseline)")
        logger.info(f"  Improvement: {(result['avg_sharpe'] - baseline_result['avg_sharpe']) / baseline_result['avg_sharpe'] * 100:+.1f}%")

        update_progress(
            f"Phase 2: Tested combination with {len(expanded_symbols)} symbols. "
            f"Sharpe: {result['avg_sharpe']:.2f}"
        )

    # Save combination results
    combo_df = pd.DataFrame(combination_results)
    combo_csv = Path('reports/20251112_COMBINATION_RESULTS.csv')
    combo_df.to_csv(combo_csv, index=False)

    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE - COMBINATION RESULTS")
    logger.info("="*80)

    for _, row in combo_df.iterrows():
        logger.info(
            f"{row['label']:30s} - Sharpe: {row['avg_sharpe']:5.2f}, "
            f"Win Rate: {row['avg_win_rate']*100:5.1f}%, "
            f"Trades: {row['total_trades']:5.0f}"
        )

    update_progress("Phase 2 complete! All combinations tested")

    # Final recommendations
    logger.info("\n" + "="*80)
    logger.info("FINAL RECOMMENDATIONS")
    logger.info("="*80)

    best_combo = combo_df.sort_values('avg_sharpe', ascending=False).iloc[0]

    logger.info(f"\nBest Configuration: {best_combo['label']}")
    logger.info(f"  Sharpe Ratio:    {best_combo['avg_sharpe']:.2f}")
    logger.info(f"  vs Baseline:     {(best_combo['avg_sharpe'] - baseline_result['avg_sharpe']) / baseline_result['avg_sharpe'] * 100:+.1f}%")
    logger.info(f"  Win Rate:        {best_combo['avg_win_rate']*100:.1f}%")
    logger.info(f"  Avg Return:      {best_combo['avg_return']*100:.1f}%")

    logger.info("\nTop 10 Individual Symbols to Add:")
    for i, row in enumerate(top_10.iterrows(), 1):
        idx, data_row = row
        logger.info(
            f"  {i:2d}. {data_row['symbol']:6s} - Sharpe: {data_row['avg_sharpe']:5.2f}, "
            f"Win Rate: {data_row['avg_win_rate']*100:5.1f}%"
        )

    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"\n✓ Total execution time: {total_time:.0f} minutes")

    update_progress(f"Testing complete! Total time: {total_time:.0f} minutes")
    update_progress(f"Best configuration: {best_combo['label']} with Sharpe {best_combo['avg_sharpe']:.2f}")

    # Generate detailed report
    report_path = Path('reports/20251112_UNUSED_SYMBOLS_TESTING_REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Testing 26 Unused ETF Symbols - Results Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Runtime**: {total_time:.0f} minutes\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Symbols Tested**: {len(UNUSED_SYMBOLS)}\n")
        f.write(f"- **Viable Symbols Found**: {len(viable_symbols)} (Sharpe > 3.0, Win Rate > 55%)\n")
        f.write(f"- **Best Configuration**: {best_combo['label']}\n")
        f.write(f"- **Sharpe Improvement**: {(best_combo['avg_sharpe'] - baseline_result['avg_sharpe']) / baseline_result['avg_sharpe'] * 100:+.1f}%\n\n")

        f.write("## Individual Symbol Results\n\n")
        f.write("| Rank | Symbol | Avg Sharpe | Win Rate | Avg Return | Trades | Viable |\n")
        f.write("|------|--------|-----------|----------|-----------|--------|--------|\n")

        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            f.write(
                f"| {i} | {row['symbol']} | {row['avg_sharpe']:.2f} | "
                f"{row['avg_win_rate']*100:.1f}% | {row['avg_return']*100:.1f}% | "
                f"{row['total_trades']:.0f} | {'✓' if row['viable'] else '✗'} |\n"
            )

        f.write("\n## Combination Results\n\n")
        f.write("| Configuration | Symbols | Sharpe | Win Rate | Trades | vs Baseline |\n")
        f.write("|---------------|---------|--------|----------|--------|-------------|\n")

        for _, row in combo_df.iterrows():
            improvement = ((row['avg_sharpe'] - baseline_result['avg_sharpe']) /
                         baseline_result['avg_sharpe'] * 100) if row['label'] != 'Current 20' else 0
            f.write(
                f"| {row['label']} | {row['num_symbols']} | {row['avg_sharpe']:.2f} | "
                f"{row['avg_win_rate']*100:.1f}% | {row['total_trades']:.0f} | "
                f"{improvement:+.1f}% |\n"
            )

        f.write("\n## Recommendations\n\n")
        f.write(f"**Recommended Configuration**: {best_combo['label']}\n\n")
        f.write(f"**Top Symbols to Add**:\n")
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"{i}. {row['symbol']} (Sharpe: {row['avg_sharpe']:.2f}, Win Rate: {row['avg_win_rate']*100:.1f}%)\n")

    logger.success(f"\n✓ Detailed report saved: {report_path}")

    return results_df, combo_df


if __name__ == "__main__":
    individual_results, combination_results = main()
