"""
Simulate overnight mean reversion strategy for the last month.

Period: October 12, 2025 to November 12, 2025 (1 month)
Training: September 12, 2023 to October 12, 2025 (2 years)
Configuration: Optimal settings from walk-forward validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.logger import logger

DATA_DIR = Path('data/leveraged_etfs')

# Optimal configuration from optimization
OPTIMAL_CONFIG = {
    'symbols': [
        'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL',
        'UPRO', 'SVXY', 'TQQQ', 'SSO', 'DFEN', 'WEBL',
        'UCO', 'FAS', 'TNA', 'LABU', 'SPXU', 'QLD', 'SQQQ', 'NAIL'
    ],
    'max_position_size': 0.15,
    'stop_loss': -0.02,
    'vix_threshold': 35,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'skip_regimes': ['BEAR'],
    'max_concurrent_positions': 3
}


class MarketRegimeDetector:
    """Detect market regimes using technical indicators."""

    REGIMES = {
        'STRONG_BULL': {'name': 'Strong Bull', 'risk': 'low'},
        'WEAK_BULL': {'name': 'Weak Bull', 'risk': 'moderate'},
        'SIDEWAYS': {'name': 'Sideways', 'risk': 'moderate'},
        'UNPREDICTABLE': {'name': 'Unpredictable', 'risk': 'high'},
        'BEAR': {'name': 'Bear', 'risk': 'very_high'}
    }

    def detect_regime(self, spy_data, vix_data, date):
        """Detect current market regime."""
        try:
            # Get data up to current date
            spy_df = spy_data[spy_data.index <= date].tail(60)

            if len(spy_df) < 50:
                return 'UNPREDICTABLE', 0.5

            # Calculate indicators (convert to float to avoid Series comparison issues)
            current_price = float(spy_df['Close'].iloc[-1])
            sma_20 = float(spy_df['Close'].tail(20).mean())
            sma_50 = float(spy_df['Close'].tail(50).mean())
            momentum_20 = float((current_price - spy_df['Close'].iloc[-20]) / spy_df['Close'].iloc[-20])

            # VIX percentile
            vix_subset = vix_data[vix_data.index <= date]
            if len(vix_subset) == 0:
                return 'UNPREDICTABLE', 0.5
            vix_current = float(vix_subset['Close'].iloc[-1])
            vix_60d = vix_subset.tail(60)['Close']
            vix_percentile = float((vix_60d < vix_current).sum()) / len(vix_60d)

            # Determine regime
            if current_price > sma_50 and sma_20 > sma_50 and momentum_20 > 0.05:
                return 'STRONG_BULL', 0.85
            elif current_price > sma_50 and momentum_20 > 0:
                return 'WEAK_BULL', 0.75
            elif abs(momentum_20) < 0.02 and vix_percentile < 0.5:
                return 'SIDEWAYS', 0.70
            elif current_price < sma_50 and momentum_20 < -0.05:
                return 'BEAR', 0.80
            else:
                return 'UNPREDICTABLE', 0.65

        except Exception as e:
            logger.warning(f"Regime detection error: {str(e)}")
            return 'UNPREDICTABLE', 0.5


class BayesianOvernightModel:
    """Bayesian probability model for overnight returns."""

    def __init__(self):
        self.stats = {}

    def train(self, historical_data, regime_detector, spy_data, vix_data, train_start, train_end):
        """Train on historical data."""
        logger.info(f"Training Bayesian model from {train_start} to {train_end}...")

        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)

        symbols_trained = 0

        for symbol in historical_data.keys():
            if symbol in ['SPY', '^VIX']:
                continue

            df = historical_data[symbol].copy()

            if not isinstance(df.index, pd.DatetimeIndex):
                continue

            df = df.loc[train_start_ts:train_end_ts]

            if len(df) < 100:
                continue

            # Calculate returns
            df['overnight_return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
            df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']

            # Classify regimes
            df['regime'] = None
            for date in df.index:
                regime, _ = regime_detector.detect_regime(spy_data, vix_data, date)
                df.loc[date, 'regime'] = regime

            # Bucket intraday moves
            df['intraday_bucket'] = pd.cut(
                df['intraday_return'],
                bins=[-np.inf, -0.03, -0.01, 0, 0.01, 0.03, np.inf],
                labels=['large_down', 'small_down', 'flat_down', 'flat_up', 'small_up', 'large_up']
            )

            # Calculate statistics
            for regime in ['STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR']:
                for bucket in ['large_down', 'small_down', 'flat_down', 'flat_up', 'small_up', 'large_up']:
                    mask = (df['regime'] == regime) & (df['intraday_bucket'] == bucket)
                    subset = df[mask]['overnight_return'].dropna()

                    if len(subset) >= 5:
                        key = (symbol, regime, bucket)
                        self.stats[key] = {
                            'win_rate': (subset > 0).mean(),
                            'avg_return': subset.mean(),
                            'sample_size': len(subset)
                        }

            symbols_trained += 1

        logger.success(f"  Training complete! Analyzed {symbols_trained} symbols")

    def predict(self, symbol, regime, intraday_return):
        """Predict overnight return probability."""
        # Bucket the intraday return
        if intraday_return < -0.03:
            bucket = 'large_down'
        elif intraday_return < -0.01:
            bucket = 'small_down'
        elif intraday_return < 0:
            bucket = 'flat_down'
        elif intraday_return < 0.01:
            bucket = 'flat_up'
        elif intraday_return < 0.03:
            bucket = 'small_up'
        else:
            bucket = 'large_up'

        key = (symbol, regime, bucket)

        if key in self.stats:
            return self.stats[key]
        else:
            return {'win_rate': 0.5, 'avg_return': 0, 'sample_size': 0}


def load_data():
    """Load all required data."""
    logger.info("Loading data...")

    data = {}

    # Load SPY and VIX
    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    data['SPY'] = pd.read_parquet(spy_path)
    data['^VIX'] = pd.read_parquet(vix_path)

    logger.success(f"  Loaded SPY: {len(data['SPY'])} bars")
    logger.success(f"  Loaded ^VIX: {len(data['^VIX'])} bars")

    # Load optimal symbols
    loaded = 0
    for symbol in OPTIMAL_CONFIG['symbols']:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            data[symbol] = df
            loaded += 1

    logger.success(f"  Loaded {loaded}/{len(OPTIMAL_CONFIG['symbols'])} optimal symbols")

    return data


def backtest_period(data, regime_detector, bayesian_model, test_start, test_end, config):
    """Backtest strategy on test period."""
    logger.info(f"\nBacktesting {test_start} to {test_end}...")

    spy_data = data['SPY']
    vix_data = data['^VIX']

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Get trading days
    test_dates = spy_data[(spy_data.index >= test_start_ts) & (spy_data.index <= test_end_ts)].index

    trades = []
    portfolio_value = [100000]
    daily_returns = []

    for date in test_dates:
        # Detect regime
        regime, confidence = regime_detector.detect_regime(spy_data, vix_data, date)

        # Skip BEAR regime
        if regime in config['skip_regimes']:
            continue

        # Check VIX threshold
        vix_subset = vix_data[vix_data.index <= date]
        if len(vix_subset) == 0:
            continue
        vix_value = float(vix_subset['Close'].iloc[-1])
        if vix_value > config['vix_threshold']:
            continue

        # Analyze each symbol
        signals = []

        for symbol in config['symbols']:
            if symbol not in data:
                continue

            symbol_data = data[symbol]
            symbol_data = symbol_data[symbol_data.index <= date]

            if len(symbol_data) < 2:
                continue

            current_row = symbol_data.iloc[-1]

            # Calculate intraday return (convert to float)
            intraday_return = float((current_row['Close'] - current_row['Open']) / current_row['Open'])

            # Get prediction
            pred = bayesian_model.predict(symbol, regime, intraday_return)

            # Quality filters
            if pred['win_rate'] < config['min_win_rate']:
                continue
            if pred['avg_return'] < config['min_expected_return']:
                continue
            if pred['sample_size'] < config['min_sample_size']:
                continue

            signals.append({
                'symbol': symbol,
                'date': date,
                'entry_price': current_row['Close'],
                'win_rate': pred['win_rate'],
                'expected_return': pred['avg_return'],
                'sample_size': pred['sample_size'],
                'regime': regime,
                'confidence': confidence,
                'intraday_return': intraday_return,
                'vix': vix_value
            })

        # Sort by expected return
        signals.sort(key=lambda x: x['expected_return'], reverse=True)

        # Limit concurrent positions
        positions_today = len([t for t in trades if t['date'] == date])
        max_new_positions = config['max_concurrent_positions'] - positions_today

        # Take top signals
        for signal in signals[:max_new_positions]:
            symbol = signal['symbol']
            symbol_data = data[symbol]

            # Get next day's open
            symbol_future = symbol_data[symbol_data.index > date]
            if len(symbol_future) == 0:
                continue

            next_row = symbol_future.iloc[0]
            exit_price = next_row['Open']

            # Calculate return
            entry_price = signal['entry_price']
            raw_return = (exit_price - entry_price) / entry_price

            # Apply stop-loss
            actual_return = max(raw_return, config['stop_loss'])
            stopped_out = raw_return < config['stop_loss']

            # Record trade
            trade = {
                'date': signal['date'],
                'exit_date': next_row.name,
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'raw_return': raw_return,
                'actual_return': actual_return,
                'position_size': config['max_position_size'],
                'profit': actual_return * config['max_position_size'] * portfolio_value[-1],
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'stopped_out': stopped_out,
                'win_rate_pred': signal['win_rate'],
                'expected_return_pred': signal['expected_return'],
                'intraday_return': signal['intraday_return'],
                'vix': signal['vix']
            }

            trades.append(trade)

            # Update portfolio
            new_value = portfolio_value[-1] + trade['profit']
            portfolio_value.append(new_value)

    # Calculate metrics
    if len(trades) == 0:
        logger.warning("No trades executed!")
        return None

    trades_df = pd.DataFrame(trades)

    total_return = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0]

    # Calculate Sharpe
    daily_pnl = trades_df.groupby('date')['actual_return'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    # Calculate max drawdown
    portfolio_series = pd.Series(portfolio_value)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate
    win_rate = (trades_df['actual_return'] > 0).mean()

    # Stop-out rate
    stop_out_rate = trades_df['stopped_out'].mean()

    results = {
        'test_start': test_start,
        'test_end': test_end,
        'total_return': total_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'num_trades': len(trades),
        'stop_out_rate': stop_out_rate,
        'trades': trades_df
    }

    return results


def main():
    """Run last month simulation."""
    logger.info("="*80)
    logger.info("OVERNIGHT MEAN REVERSION - LAST MONTH SIMULATION")
    logger.info("="*80)

    # Define periods
    today = datetime.now()
    test_end = today.strftime('%Y-%m-%d')
    test_start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    train_end = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    train_start = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years

    logger.info(f"\nTraining Period: {train_start} to {train_end}")
    logger.info(f"Testing Period: {test_start} to {test_end}")
    logger.info(f"Configuration: Optimal (15% position, -2% stop, 20 symbols)")

    # Load data
    data = load_data()

    # Initialize components
    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianOvernightModel()

    # Train model
    logger.info("\n" + "="*80)
    logger.info("TRAINING PHASE")
    logger.info("="*80)
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_start, train_end)

    # Backtest
    logger.info("\n" + "="*80)
    logger.info("BACKTESTING PHASE")
    logger.info("="*80)
    results = backtest_period(data, regime_detector, bayesian_model, test_start, test_end, OPTIMAL_CONFIG)

    if results is None:
        logger.error("No results generated!")
        return

    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESULTS - LAST MONTH")
    logger.info("="*80)

    logger.info(f"\nPeriod: {results['test_start']} to {results['test_end']}")
    logger.info(f"Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']*100:.1f}%")
    logger.info(f"Max Drawdown: {results['max_dd']*100:.2f}%")
    logger.info(f"Total Trades: {results['num_trades']}")
    logger.info(f"Stop-Out Rate: {results['stop_out_rate']*100:.1f}%")

    # Save trades
    trades_file = Path('reports/20251112_LAST_MONTH_TRADES.csv')
    results['trades'].to_csv(trades_file, index=False)
    logger.success(f"\nSaved trades to: {trades_file}")

    # Show top trades
    logger.info("\n" + "="*80)
    logger.info("TOP 10 TRADES")
    logger.info("="*80)

    top_trades = results['trades'].nlargest(10, 'actual_return')
    logger.info(f"\n{'Date':<12} {'Symbol':<8} {'Entry':<10} {'Exit':<10} {'Return':<10} {'Regime':<15}")
    logger.info("-" * 75)
    for _, trade in top_trades.iterrows():
        logger.info(
            f"{trade['date'].strftime('%Y-%m-%d'):<12} "
            f"{trade['symbol']:<8} "
            f"${trade['entry_price']:<9.2f} "
            f"${trade['exit_price']:<9.2f} "
            f"{trade['actual_return']*100:>8.2f}% "
            f"{trade['regime']:<15}"
        )

    # Show worst trades
    logger.info("\n" + "="*80)
    logger.info("WORST 10 TRADES")
    logger.info("="*80)

    worst_trades = results['trades'].nsmallest(10, 'actual_return')
    logger.info(f"\n{'Date':<12} {'Symbol':<8} {'Entry':<10} {'Exit':<10} {'Return':<10} {'Stop':<6}")
    logger.info("-" * 65)
    for _, trade in worst_trades.iterrows():
        stopped = "YES" if trade['stopped_out'] else "NO"
        logger.info(
            f"{trade['date'].strftime('%Y-%m-%d'):<12} "
            f"{trade['symbol']:<8} "
            f"${trade['entry_price']:<9.2f} "
            f"${trade['exit_price']:<9.2f} "
            f"{trade['actual_return']*100:>8.2f}% "
            f"{stopped:<6}"
        )

    # Daily performance
    logger.info("\n" + "="*80)
    logger.info("DAILY PERFORMANCE")
    logger.info("="*80)

    daily_pnl = results['trades'].groupby('date').agg({
        'actual_return': 'sum',
        'symbol': 'count'
    }).rename(columns={'symbol': 'num_trades'})

    logger.info(f"\n{'Date':<12} {'Trades':<8} {'Return':<10}")
    logger.info("-" * 35)
    for date, row in daily_pnl.iterrows():
        logger.info(
            f"{date.strftime('%Y-%m-%d'):<12} "
            f"{int(row['num_trades']):<8} "
            f"{row['actual_return']*100:>8.2f}%"
        )

    # Performance by regime
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE BY REGIME")
    logger.info("="*80)

    regime_perf = results['trades'].groupby('regime').agg({
        'actual_return': ['count', 'mean', lambda x: (x > 0).mean()]
    }).round(4)

    logger.info(f"\n{'Regime':<15} {'Trades':<10} {'Avg Return':<15} {'Win Rate':<10}")
    logger.info("-" * 55)
    for regime, row in regime_perf.iterrows():
        logger.info(
            f"{regime:<15} "
            f"{int(row[('actual_return', 'count')]):<10} "
            f"{row[('actual_return', 'mean')]*100:>12.2f}% "
            f"{row[('actual_return', '<lambda_0>')]*100:>8.1f}%"
        )

    logger.info("\n" + "="*80)
    logger.success("SIMULATION COMPLETE")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    main()
