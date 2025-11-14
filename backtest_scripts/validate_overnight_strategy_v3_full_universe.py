"""
Validate overnight mean reversion strategy - V3 FULL UNIVERSE TEST.

This version tests ALL 23 leveraged ETFs to find optimal symbol selection.
Ensures alignment with Reddit strategy:
- Entry: 3:50 PM (simulated as daily close price)
- Exit: Next day open (9:31 AM)
- Holding period: Overnight only

Tests multiple symbol groups:
1. All 23 leveraged ETFs
2. 3x leveraged only
3. 2x leveraged only
4. Long vs Short comparison
5. Sector-specific (tech, financials, etc.)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.config import get_backtest_results_dir

DATA_DIR = Path('data/leveraged_etfs')

# ALL available leveraged ETFs
ALL_ETFS = [
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW',  # Broad market 3x
    'TNA', 'TZA',  # Small cap 3x
    'SOXL', 'SOXS',  # Semiconductors 3x
    'FAS', 'FAZ',  # Financials 3x
    'LABU', 'LABD',  # Biotech 3x
    'TECL', 'TECS',  # Technology 3x
    'QLD', 'QID', 'SSO', 'SDS',  # 2x leveraged
    'UVXY', 'SVXY', 'VIXY'  # Volatility
]

# Risk management (from V2)
MAX_POSITION_SIZE = 0.10
MAX_TOTAL_EXPOSURE = 0.30
MAX_LOSS_PER_TRADE = -0.03
MAX_CONCURRENT_POSITIONS = 3
VIX_THRESHOLD = 35


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

            # CRITICAL: Overnight return = (next_open - today_close) / today_close
            # This matches Reddit strategy: enter at close (3:50 PM), exit at next open
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
    logger.info(f"Loading data for {len(symbols)} symbols...")

    data = {}

    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found!")
        return None

    data['SPY'] = pd.read_parquet(spy_path)
    data['^VIX'] = pd.read_parquet(vix_path)

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


def backtest_strategy(data, regime_detector, bayesian_model, start_date, end_date, symbols, name=""):
    """Backtest with risk management."""

    logger.info(f"\nBacktesting {name} ({len(symbols)} symbols) from {start_date.date()} to {end_date.date()}...")

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
        regime, confidence, vix_current = regime_detector.classify_regime(spy_data, vix_data, date)

        # Skip bear markets
        if regime == 'BEAR':
            skipped_bear += 1
            continue

        # Skip high VIX
        if vix_current > VIX_THRESHOLD:
            skipped_vix += 1
            continue

        # Position limit check
        if len([t for t in trades if t['date'] == date]) >= MAX_CONCURRENT_POSITIONS:
            continue

        for symbol in symbols:
            if symbol not in data:
                continue

            if len([t for t in trades if t['date'] == date]) >= MAX_CONCURRENT_POSITIONS:
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

            # Apply stop-loss
            stopped_out_flag = False
            if overnight_return < MAX_LOSS_PER_TRADE:
                overnight_return = MAX_LOSS_PER_TRADE
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
                'position_size': MAX_POSITION_SIZE,
                'stopped_out': stopped_out_flag,
                'profitable': overnight_return > 0
            })

    logger.info(f"  Generated {len(trades)} trades")
    logger.info(f"  Skipped {skipped_bear} days (BEAR), {skipped_vix} days (high VIX)")
    logger.info(f"  Stopped out: {stopped_out} trades")

    return pd.DataFrame(trades)


def analyze_results(trades_df, name=""):
    """Analyze backtest results."""

    if trades_df.empty:
        logger.error(f"No trades for {name}!")
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
    stopped_out_pct = stopped_out_count / total_trades

    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly = trades_df.groupby('month')['portfolio_return'].sum()
    monthly_win_rate = (monthly > 0).sum() / len(monthly)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {name}")
    logger.info(f"{'='*80}")
    logger.info(f"  Total Trades: {total_trades:,}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    logger.info(f"  Avg Return/Trade: {avg_return:.3%}")
    logger.info(f"  Total Return: {total_return:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Max Drawdown: {max_dd:.1%}")
    logger.info(f"  Monthly Win Rate: {monthly_win_rate:.1%}")
    logger.info(f"  Stop-Out Rate: {stopped_out_pct:.1%}")

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
        'trades_df': trades_df
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("OVERNIGHT MEAN REVERSION - V3 FULL UNIVERSE TEST")
    logger.info("="*80)
    logger.info("\nExecution Timing (matching Reddit strategy):")
    logger.info("  Entry: 3:50 PM (simulated as daily CLOSE price)")
    logger.info("  Exit: Next day 9:31 AM (actual OPEN price)")
    logger.info("  Holding Period: Overnight only (~16 hours)")
    logger.info("\nTesting Symbol Groups:")
    logger.info(f"  [1] All 23 leveraged ETFs")
    logger.info(f"  [2] Top 5 from V2 (baseline)")
    logger.info(f"  [3] 3x Leveraged Long only")
    logger.info(f"  [4] 3x Leveraged Short only")
    logger.info(f"  [5] 2x Leveraged only")
    logger.info(f"  [6] Tech sector (TQQQ, SQQQ, SOXL, SOXS, TECL, TECS)")
    logger.info(f"  [7] Broad market (UPRO, SPXU, UDOW, SDOW, SSO, SDS)")

    # Define symbol groups
    symbol_groups = {
        'All 23 ETFs': ALL_ETFS,
        'Top 5 (V2 Baseline)': ['UPRO', 'TECL', 'FAS', 'TQQQ', 'SOXL'],
        '3x Long Only': ['TQQQ', 'UPRO', 'UDOW', 'TNA', 'SOXL', 'FAS', 'LABU', 'TECL'],
        '3x Short Only': ['SQQQ', 'SPXU', 'SDOW', 'TZA', 'SOXS', 'FAZ', 'LABD', 'TECS'],
        '2x Leveraged': ['QLD', 'QID', 'SSO', 'SDS'],
        'Tech Sector': ['TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'TECL', 'TECS'],
        'Broad Market': ['UPRO', 'SPXU', 'UDOW', 'SDOW', 'SSO', 'SDS', 'TNA', 'TZA']
    }

    # Load data for all symbols
    data = load_data(ALL_ETFS)
    if data is None:
        return

    # Initialize models
    logger.info("\n[Step 1] Initializing regime detector...")
    regime_detector = SimpleRegimeDetector()

    recent_date = data['SPY'].index[-1]
    regime, confidence, vix = regime_detector.classify_regime(
        data['SPY'], data['^VIX'], recent_date
    )
    logger.success(f"  Current regime: {regime} (confidence: {confidence:.2f}, VIX: {vix:.1f})")

    # Train model
    logger.info("\n[Step 2] Training Bayesian model on ALL symbols...")
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end, ALL_ETFS)

    # Test each group
    logger.info("\n[Step 3] Testing symbol groups...")
    logger.info(f"  Training: 2015-2023 (8 years)")
    logger.info(f"  Testing: 2024-{test_end.year} ({(test_end - test_start).days / 30:.1f} months)")

    results = []

    for group_name, symbols in symbol_groups.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {group_name}")
        logger.info(f"{'='*80}")

        trades_df = backtest_strategy(
            data, regime_detector, bayesian_model,
            test_start, test_end, symbols, group_name
        )

        result = analyze_results(trades_df, group_name)
        if result:
            results.append(result)

            # Save trades
            output_path = get_backtest_results_dir() / f'overnight_v3_{group_name.replace(" ", "_").lower()}_trades.csv'
            output_path.parent.mkdir(exist_ok=True)
            trades_df.to_csv(output_path, index=False)
            logger.success(f"  Saved to {output_path}")

    # Comparison table
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: ALL SYMBOL GROUPS")
    logger.info("="*80)
    logger.info(f"{'Group':<25} {'Trades':<8} {'Win%':<7} {'Return':<9} {'Sharpe':<7} {'MaxDD':<8}")
    logger.info("-"*80)

    for r in results:
        logger.info(
            f"{r['name']:<25} {r['total_trades']:<8} "
            f"{r['win_rate']:<7.1%} {r['total_return']:<9.1%} "
            f"{r['sharpe_ratio']:<7.2f} {r['max_drawdown']:<8.1%}"
        )

    # Find best group
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    best_return = max(results, key=lambda x: x['total_return'])
    best_winrate = max(results, key=lambda x: x['win_rate'])

    logger.info("\n" + "="*80)
    logger.info("BEST PERFORMERS")
    logger.info("="*80)
    logger.success(f"  Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})")
    logger.success(f"  Best Return: {best_return['name']} ({best_return['total_return']:.1%})")
    logger.success(f"  Best Win Rate: {best_winrate['name']} ({best_winrate['win_rate']:.1%})")

    # Save summary
    summary_path = get_backtest_results_dir() / '20251112_V3_FULL_UNIVERSE_COMPARISON.md'
    with open(summary_path, 'w') as f:
        f.write("# Overnight Mean Reversion - Full Universe Test (V3)\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Execution Timing\n\n")
        f.write("**Entry**: 3:50 PM EST (simulated as daily close)\n")
        f.write("**Exit**: Next day 9:31 AM EST (actual open price)\n")
        f.write("**Holding**: Overnight only (~16 hours)\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Group | Trades | Win Rate | Total Return | Sharpe | Max DD |\n")
        f.write("|-------|--------|----------|--------------|--------|--------|\n")

        for r in results:
            f.write(
                f"| {r['name']} | {r['total_trades']} | {r['win_rate']:.1%} | "
                f"{r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | {r['max_drawdown']:.1%} |\n"
            )

        f.write("\n## Best Performers\n\n")
        f.write(f"- **Best Sharpe Ratio**: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})\n")
        f.write(f"- **Best Total Return**: {best_return['name']} ({best_return['total_return']:.1%})\n")
        f.write(f"- **Best Win Rate**: {best_winrate['name']} ({best_winrate['win_rate']:.1%})\n")

    logger.success(f"\nSaved summary to {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("VALIDATION V3 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
