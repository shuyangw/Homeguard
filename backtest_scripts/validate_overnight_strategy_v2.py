"""
Validate the overnight mean reversion strategy - VERSION 2 (IMPROVED).

Improvements from V1:
1. Stop-loss protection (-3% max loss per trade)
2. Reduced position sizing (10% per trade, max 30% total)
3. Bear market filtering (skip BEAR regime)
4. VIX filter (skip when VIX > 35)
5. Focus on best performing symbols
6. Maximum 3 concurrent positions
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

# Data directory
DATA_DIR = Path('data/leveraged_etfs')

# Best performing symbols from V1 validation
SYMBOLS = ['UPRO', 'TECL', 'FAS', 'TQQQ', 'SOXL']  # Top 5 performers only

# Risk management parameters
MAX_POSITION_SIZE = 0.10  # 10% per position
MAX_TOTAL_EXPOSURE = 0.30  # 30% total
MAX_LOSS_PER_TRADE = -0.03  # -3% stop loss
MAX_CONCURRENT_POSITIONS = 3
VIX_THRESHOLD = 35  # Skip trading when VIX > 35


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

        if len(spy) >= 40:
            sma_20_prev = spy['sma_20'].iloc[-20]
            momentum = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev != 0 else 0
        else:
            momentum = 0

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

        return regime, confidence, vix_current


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

    logger.info("Loading downloaded data...")

    data = {}

    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    data['SPY'] = pd.read_parquet(spy_path)
    data['^VIX'] = pd.read_parquet(vix_path)

    logger.success(f"  Loaded SPY: {len(data['SPY'])} bars")
    logger.success(f"  Loaded ^VIX: {len(data['^VIX'])} bars")

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


def backtest_strategy_v2(data, regime_detector, bayesian_model, start_date, end_date):
    """Backtest the improved overnight mean reversion strategy with risk management."""

    logger.info(f"\nBacktesting V2 (with risk management) from {start_date} to {end_date}...")

    spy_data = data['SPY']
    vix_data = data['^VIX']

    test_dates = spy_data[
        (spy_data.index >= start_date) & (spy_data.index <= end_date)
    ].index

    trades = []
    skipped_bear = 0
    skipped_vix = 0
    stopped_out = 0

    for date in test_dates:
        # Classify regime
        regime, confidence, vix_current = regime_detector.classify_regime(spy_data, vix_data, date)

        # IMPROVEMENT 1: Skip bear markets
        if regime == 'BEAR':
            skipped_bear += 1
            continue

        # IMPROVEMENT 2: Skip high VIX periods
        if vix_current > VIX_THRESHOLD:
            skipped_vix += 1
            continue

        # IMPROVEMENT 3: Check position limits (count trades for this date)
        positions_today = len([t for t in trades if t['date'] == date])
        if positions_today * MAX_POSITION_SIZE >= MAX_TOTAL_EXPOSURE:
            continue

        # Evaluate each symbol
        for symbol in SYMBOLS:
            if symbol not in data:
                continue

            # IMPROVEMENT 4: Max concurrent positions (recount for each symbol)
            positions_today = len([t for t in trades if t['date'] == date])
            if positions_today >= MAX_CONCURRENT_POSITIONS:
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

            # IMPROVEMENT 5: Higher quality threshold
            if (prob_data['probability'] < 0.58 or  # Increased from 0.55
                prob_data['expected_return'] < 0.002 or  # Increased from 0.001
                prob_data['sample_size'] < 15):  # Increased from 10
                continue

            # Calculate actual overnight return
            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['Open'])
            overnight_return = (next_open - today_close) / today_close

            # IMPROVEMENT 6: Apply stop-loss
            stopped_out_flag = False
            if overnight_return < MAX_LOSS_PER_TRADE:
                overnight_return = MAX_LOSS_PER_TRADE
                stopped_out_flag = True
                stopped_out += 1

            # Record trade
            trades.append({
                'date': date,
                'symbol': symbol,
                'regime': regime,
                'vix': vix_current,
                'intraday_return': intraday_return,
                'expected_return': prob_data['expected_return'],
                'probability': prob_data['probability'],
                'actual_return': overnight_return,
                'position_size': MAX_POSITION_SIZE,
                'stopped_out': stopped_out_flag,
                'profitable': overnight_return > 0
            })

    logger.info(f"  Skipped {skipped_bear} days (BEAR regime)")
    logger.info(f"  Skipped {skipped_vix} days (high VIX)")
    logger.info(f"  Stopped out: {stopped_out} trades")

    return pd.DataFrame(trades)


def analyze_results_v2(trades_df):
    """Analyze backtest results with improvements."""

    if trades_df.empty:
        logger.error("No trades generated!")
        return None

    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS V2 (WITH RISK MANAGEMENT)")
    logger.info("="*80)

    # Overall metrics
    total_trades = len(trades_df)
    win_rate = trades_df['profitable'].mean()
    avg_return = trades_df['actual_return'].mean()

    # Portfolio returns (with position sizing)
    trades_df['portfolio_return'] = trades_df['actual_return'] * trades_df['position_size']
    total_return = trades_df['portfolio_return'].sum()

    # Daily portfolio returns
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

    # Stopped out stats
    stopped_out_count = trades_df['stopped_out'].sum()
    stopped_out_pct = stopped_out_count / total_trades

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total Trades: {total_trades:,}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  Avg Return per Trade: {avg_return:.3%}")
    logger.info(f"  Total Return: {total_return:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Max Drawdown: {max_dd:.1%}")
    logger.info(f"  Stopped Out: {stopped_out_count} ({stopped_out_pct:.1%})")

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

    # Symbol performance
    logger.info(f"\nPerformance by Symbol:")
    symbol_stats = trades_df.groupby('symbol').agg({
        'profitable': 'mean',
        'actual_return': ['mean', 'count'],
        'portfolio_return': 'sum'
    }).sort_values(('portfolio_return', 'sum'), ascending=False)

    for symbol in symbol_stats.index:
        win_rate_s = symbol_stats.loc[symbol, ('profitable', 'mean')]
        avg_ret_s = symbol_stats.loc[symbol, ('actual_return', 'mean')]
        count_s = symbol_stats.loc[symbol, ('actual_return', 'count')]
        total_contrib = symbol_stats.loc[symbol, ('portfolio_return', 'sum')]

        logger.info(
            f"  {symbol:8} Win Rate: {win_rate_s:.1%}  "
            f"Avg Return: {avg_ret_s:.3%}  "
            f"Trades: {int(count_s):3}  "
            f"Contribution: {total_contrib:.1%}"
        )

    # Monthly performance
    logger.info(f"\nMonthly Returns:")
    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly = trades_df.groupby('month')['portfolio_return'].sum()

    positive_months = (monthly > 0).sum()
    total_months = len(monthly)

    for month, ret in monthly.items():
        status = "[+]" if ret > 0 else "[-]"
        logger.info(f"  {status} {month}: {ret:.2%}")

    logger.info(f"\nMonthly Win Rate: {positive_months}/{total_months} ({positive_months/total_months:.1%})")

    # Compare to V1
    logger.info(f"\n" + "="*80)
    logger.info("COMPARISON TO V1 (No Risk Management)")
    logger.info("="*80)
    logger.info(f"                      V1          V2          Improvement")
    logger.info(f"  Win Rate:         60.7%       {win_rate:.1%}")
    logger.info(f"  Sharpe Ratio:     1.76        {sharpe:.2f}")
    logger.info(f"  Max Drawdown:     -93.8%      {max_dd:.1%}")
    logger.info(f"  Total Return:     406.4%      {total_return:.1%}")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'stopped_out_pct': stopped_out_pct,
        'monthly_win_rate': positive_months / total_months
    }


def main():
    """Main validation function."""

    logger.info("\n" + "="*80)
    logger.info("OVERNIGHT MEAN REVERSION STRATEGY VALIDATION V2")
    logger.info("="*80)
    logger.info("\nImprovements:")
    logger.info("  [1] Stop-loss: -3% max loss per trade")
    logger.info("  [2] Position sizing: 10% per trade, 30% max total")
    logger.info("  [3] Max concurrent positions: 3")
    logger.info("  [4] Skip BEAR regime entirely")
    logger.info("  [5] Skip when VIX > 35")
    logger.info("  [6] Higher quality thresholds (58% win rate, 0.2% expected return)")
    logger.info("  [7] Best symbols only (UPRO, TECL, FAS, TQQQ, SOXL)")

    # Load data
    data = load_data()
    if data is None:
        return

    # Initialize components
    logger.info("\n[Step 1] Initializing regime detector...")
    regime_detector = SimpleRegimeDetector()

    recent_date = data['SPY'].index[-1]
    regime, confidence, vix = regime_detector.classify_regime(
        data['SPY'], data['^VIX'], recent_date
    )
    logger.success(f"  Current regime: {regime} (confidence: {confidence:.2f}, VIX: {vix:.1f})")

    # Train model
    logger.info("\n[Step 2] Training Bayesian model...")
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end)

    # Backtest V2
    logger.info("\n[Step 3] Running backtest V2 with risk management...")

    trades_df = backtest_strategy_v2(
        data, regime_detector, bayesian_model,
        test_start, test_end
    )

    # Analyze
    logger.info("\n[Step 4] Analyzing results...")

    results = analyze_results_v2(trades_df)

    # Save results
    if results and not trades_df.empty:
        logger.info("\n[Step 5] Saving results...")

        output_path = Path('reports/overnight_validation_trades_v2.csv')
        output_path.parent.mkdir(exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        logger.success(f"  Saved trades to {output_path}")

        # Quality assessment
        logger.info("\n" + "="*80)
        logger.info("STRATEGY QUALITY ASSESSMENT")
        logger.info("="*80)

        quality_score = 0
        max_score = 7

        # Win rate
        if results['win_rate'] >= 0.60:
            logger.success("  [PASS] Win Rate >= 60%")
            quality_score += 1
        else:
            logger.warning(f"  [FAIL] Win Rate < 60% ({results['win_rate']:.1%})")

        # Sharpe ratio
        if results['sharpe_ratio'] >= 2.0:
            logger.success(f"  [PASS] Sharpe Ratio >= 2.0 ({results['sharpe_ratio']:.2f})")
            quality_score += 1
        elif results['sharpe_ratio'] >= 1.5:
            logger.info(f"  [OK] Sharpe Ratio >= 1.5 ({results['sharpe_ratio']:.2f})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Sharpe Ratio < 1.5 ({results['sharpe_ratio']:.2f})")

        # Max drawdown
        if results['max_drawdown'] > -0.20:
            logger.success(f"  [PASS] Max Drawdown > -20% ({results['max_drawdown']:.1%})")
            quality_score += 1
        elif results['max_drawdown'] > -0.30:
            logger.info(f"  [OK] Max Drawdown > -30% ({results['max_drawdown']:.1%})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Max Drawdown < -30% ({results['max_drawdown']:.1%})")

        # Total return
        if results['total_return'] > 0.50:
            logger.success(f"  [PASS] Total Return > 50% ({results['total_return']:.1%})")
            quality_score += 1
        elif results['total_return'] > 0.20:
            logger.info(f"  [OK] Total Return > 20% ({results['total_return']:.1%})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Total Return < 20% ({results['total_return']:.1%})")

        # Monthly win rate
        if results['monthly_win_rate'] >= 0.70:
            logger.success(f"  [PASS] Monthly Win Rate >= 70% ({results['monthly_win_rate']:.1%})")
            quality_score += 1
        elif results['monthly_win_rate'] >= 0.60:
            logger.info(f"  [OK] Monthly Win Rate >= 60% ({results['monthly_win_rate']:.1%})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Monthly Win Rate < 60% ({results['monthly_win_rate']:.1%})")

        # Trade volume
        if results['total_trades'] >= 200:
            logger.success(f"  [PASS] Total Trades >= 200 ({results['total_trades']:,})")
            quality_score += 1
        elif results['total_trades'] >= 100:
            logger.info(f"  [OK] Total Trades >= 100 ({results['total_trades']:,})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Total Trades < 100 ({results['total_trades']:,})")

        # Stop-out rate
        if results['stopped_out_pct'] < 0.10:
            logger.success(f"  [PASS] Stop-out Rate < 10% ({results['stopped_out_pct']:.1%})")
            quality_score += 1
        elif results['stopped_out_pct'] < 0.20:
            logger.info(f"  [OK] Stop-out Rate < 20% ({results['stopped_out_pct']:.1%})")
            quality_score += 0.5
        else:
            logger.warning(f"  [FAIL] Stop-out Rate >= 20% ({results['stopped_out_pct']:.1%})")

        # Final score
        quality_percentage = (quality_score / max_score) * 100

        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL QUALITY SCORE: {quality_score:.1f}/{max_score} ({quality_percentage:.0f}%)")
        logger.info(f"{'='*80}")

        if quality_percentage >= 80:
            logger.success("\n[EXCELLENT] Strategy is ready for paper trading!")
        elif quality_percentage >= 60:
            logger.info("\n[GOOD] Strategy shows promise but needs minor refinements")
        elif quality_percentage >= 40:
            logger.warning("\n[FAIR] Strategy needs significant improvements")
        else:
            logger.error("\n[POOR] Strategy not ready for production")

    logger.info("\n" + "="*80)
    logger.info("VALIDATION V2 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
