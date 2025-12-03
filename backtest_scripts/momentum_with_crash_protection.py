"""
Backtest Momentum Strategy with Crash Protection.

Compares:
1. Baseline momentum (always invested)
2. Momentum with crash prediction (reduce exposure when crash probability high)

Uses walk-forward approach: train crash predictor, then use in next period.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

import yfinance as yf

from src.utils.logger import logger
from src.strategies.advanced.momentum_crash_predictor import MomentumCrashPredictor


def calculate_momentum_returns(prices_df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Calculate daily returns of a simple momentum strategy.

    Long top N stocks by 12-1 month momentum.
    """
    # 12-1 month momentum
    returns_12m = prices_df.pct_change(252)
    returns_1m = prices_df.pct_change(21)
    momentum = returns_12m - returns_1m

    daily_returns = prices_df.pct_change()

    strategy_returns = []

    for i in range(253, len(prices_df)):
        date = prices_df.index[i]
        prev_date = prices_df.index[i-1]

        # Get momentum scores from previous day
        scores = momentum.iloc[i-1].dropna()
        if len(scores) < top_n:
            continue

        # Select top N
        top_stocks = scores.nlargest(top_n).index

        # Equal weight returns
        ret = daily_returns.loc[date, top_stocks].mean()
        strategy_returns.append({'date': date, 'return': ret})

    return pd.DataFrame(strategy_returns).set_index('date')['return']


def backtest_with_crash_protection(
    prices_df: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    train_end: str,
    test_start: str,
    test_end: str,
    top_n: int = 10,
    crash_threshold: float = 0.25
) -> dict:
    """
    Run backtest comparing baseline vs crash-protected momentum.

    Args:
        prices_df: Stock prices DataFrame
        spy_prices: SPY prices
        vix_prices: VIX prices
        train_end: End date for crash predictor training
        test_start: Start of test period
        test_end: End of test period
        top_n: Number of stocks to hold
        crash_threshold: Probability above which to reduce exposure

    Returns:
        Dictionary with backtest results
    """
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Initialize crash predictor
    predictor = MomentumCrashPredictor(
        crash_threshold=-0.03,  # -3% over 5 days = crash
        crash_horizon=5
    )

    # Calculate momentum factor returns for training
    train_prices = prices_df[prices_df.index <= train_end_ts]
    factor_returns = predictor.calculate_momentum_factor_returns(train_prices)

    if len(factor_returns) < 500:
        logger.warning("Insufficient data for crash predictor training")
        return None

    # Identify crash events
    crash_labels = predictor.identify_crash_events(factor_returns)

    # Compute features
    train_spy = spy_prices[spy_prices.index <= train_end_ts]
    train_vix = vix_prices[vix_prices.index <= train_end_ts]

    features = predictor.compute_crash_features(
        factor_returns, train_spy, train_vix, train_prices
    )

    # Train crash predictor
    stats = predictor.train(features, crash_labels, eval_split=0.2)

    logger.info(f"Crash predictor - Train AUC: {stats['train_auc']:.3f}, Val AUC: {stats['val_auc']:.3f}")

    # Now run test period
    test_prices = prices_df[(prices_df.index >= test_start_ts) & (prices_df.index <= test_end_ts)]

    # Calculate baseline momentum returns for test period
    full_prices = prices_df[prices_df.index <= test_end_ts]
    all_mom_returns = calculate_momentum_returns(full_prices, top_n)
    test_mom_returns = all_mom_returns[(all_mom_returns.index >= test_start_ts) &
                                        (all_mom_returns.index <= test_end_ts)]

    # Calculate factor returns for test period (for crash prediction)
    test_factor_returns = predictor.calculate_momentum_factor_returns(full_prices)
    test_factor_returns = test_factor_returns[(test_factor_returns.index >= test_start_ts) &
                                               (test_factor_returns.index <= test_end_ts)]

    # Compute features for test period
    test_spy = spy_prices[(spy_prices.index >= test_start_ts) & (spy_prices.index <= test_end_ts)]
    test_vix = vix_prices[(vix_prices.index >= test_start_ts) & (vix_prices.index <= test_end_ts)]

    # We need rolling features - use full history
    full_factor = predictor.calculate_momentum_factor_returns(full_prices)
    full_features = predictor.compute_crash_features(
        full_factor, spy_prices, vix_prices, full_prices
    )
    test_features = full_features[(full_features.index >= test_start_ts) &
                                   (full_features.index <= test_end_ts)]

    # Predict crash probability
    crash_prob = predictor.predict_crash_probability(test_features)

    # Align returns and crash probability
    common_dates = test_mom_returns.index.intersection(crash_prob.index)
    baseline_returns = test_mom_returns.loc[common_dates]
    crash_prob_aligned = crash_prob.loc[common_dates]

    # Protected strategy: reduce exposure when crash probability > threshold
    # Scale: 100% exposure at prob=0, 0% exposure at prob=threshold*2
    exposure = np.clip(1 - (crash_prob_aligned / (crash_threshold * 2)), 0, 1)
    protected_returns = baseline_returns * exposure

    # Calculate metrics
    def calc_metrics(returns: pd.Series) -> dict:
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        max_dd = drawdown.min()

        return {
            'total_return': total_ret,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_days': len(returns)
        }

    baseline_metrics = calc_metrics(baseline_returns)
    protected_metrics = calc_metrics(protected_returns)

    # Count days with reduced exposure
    reduced_exposure_days = (exposure < 1).sum()
    avg_exposure = exposure.mean()

    # Crash prediction accuracy during test
    actual_crashes = predictor.identify_crash_events(full_factor)
    actual_crashes_test = actual_crashes.reindex(common_dates).fillna(0)

    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(actual_crashes_test, crash_prob_aligned)

    return {
        'baseline': baseline_metrics,
        'protected': protected_metrics,
        'test_auc': test_auc,
        'reduced_exposure_days': int(reduced_exposure_days),
        'avg_exposure': float(avg_exposure),
        'crash_prob_mean': float(crash_prob_aligned.mean()),
        'crash_prob_max': float(crash_prob_aligned.max()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Momentum with crash protection backtest')
    parser.add_argument('--top-n', type=int, default=10, help='Number of stocks to hold')
    parser.add_argument('--threshold', type=float, default=0.25, help='Crash probability threshold')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MOMENTUM WITH CRASH PROTECTION BACKTEST")
    logger.info("=" * 70)

    # Load S&P 500 symbols
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500_top100-2025.csv'
    symbols = pd.read_csv(csv_path)['Symbol'].tolist()

    logger.info(f"Loading data for {len(symbols)} symbols...")

    # Download data (start 2014 to have enough history for 2017 test)
    prices = yf.download(symbols, start='2014-01-01', end='2024-12-31', progress=True)['Close']
    spy = yf.download('SPY', start='2014-01-01', end='2024-12-31', progress=False)['Close']
    vix = yf.download('^VIX', start='2014-01-01', end='2024-12-31', progress=False)['Close']

    # Handle MultiIndex
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]

    logger.info(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")

    # Walk-forward periods (from 2017 onwards)
    periods = [
        {'train_end': '2016-12-31', 'test_start': '2017-01-01', 'test_end': '2017-12-31'},
        {'train_end': '2017-12-31', 'test_start': '2018-01-01', 'test_end': '2018-12-31'},
        {'train_end': '2018-12-31', 'test_start': '2019-01-01', 'test_end': '2019-12-31'},
        {'train_end': '2019-12-31', 'test_start': '2020-01-01', 'test_end': '2020-12-31'},
        {'train_end': '2020-12-31', 'test_start': '2021-01-01', 'test_end': '2021-12-31'},
        {'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
        {'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-11-29'},
    ]

    results = []

    for period in periods:
        year = period['test_start'][:4]
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {year}: Train until {period['train_end']}")
        logger.info(f"{'='*60}")

        result = backtest_with_crash_protection(
            prices, spy, vix,
            period['train_end'], period['test_start'], period['test_end'],
            top_n=args.top_n,
            crash_threshold=args.threshold
        )

        if result is None:
            continue

        result['year'] = year
        results.append(result)

        logger.info(f"Test AUC: {result['test_auc']:.3f}")
        logger.info(f"Baseline:  Return={result['baseline']['total_return']*100:+.1f}%  "
                   f"Sharpe={result['baseline']['sharpe']:.2f}  "
                   f"MaxDD={result['baseline']['max_drawdown']*100:.1f}%")
        logger.info(f"Protected: Return={result['protected']['total_return']*100:+.1f}%  "
                   f"Sharpe={result['protected']['sharpe']:.2f}  "
                   f"MaxDD={result['protected']['max_drawdown']*100:.1f}%")
        logger.info(f"Avg Exposure: {result['avg_exposure']:.1%}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    print("\n{:<6} {:>12} {:>12} {:>10} {:>10} {:>10}".format(
        "Year", "Base Ret", "Prot Ret", "Base DD", "Prot DD", "Test AUC"
    ))
    print("-" * 65)

    total_base = 1.0
    total_prot = 1.0

    for r in results:
        total_base *= (1 + r['baseline']['total_return'])
        total_prot *= (1 + r['protected']['total_return'])

        print("{:<6} {:>11.1f}% {:>11.1f}% {:>9.1f}% {:>9.1f}% {:>10.3f}".format(
            r['year'],
            r['baseline']['total_return'] * 100,
            r['protected']['total_return'] * 100,
            r['baseline']['max_drawdown'] * 100,
            r['protected']['max_drawdown'] * 100,
            r['test_auc']
        ))

    print("-" * 65)

    avg_base_sharpe = np.mean([r['baseline']['sharpe'] for r in results])
    avg_prot_sharpe = np.mean([r['protected']['sharpe'] for r in results])
    avg_auc = np.mean([r['test_auc'] for r in results])

    print(f"\nCumulative Baseline:  {(total_base-1)*100:+.1f}%")
    print(f"Cumulative Protected: {(total_prot-1)*100:+.1f}%")
    print(f"\nAvg Baseline Sharpe:  {avg_base_sharpe:.2f}")
    print(f"Avg Protected Sharpe: {avg_prot_sharpe:.2f}")
    print(f"Avg Test AUC: {avg_auc:.3f}")

    # Improvement analysis
    improvement = (total_prot - total_base) / abs(total_base - 1) * 100 if total_base != 1 else 0
    print(f"\nReturn Improvement: {improvement:+.1f}%")

    return results


if __name__ == "__main__":
    results = main()
