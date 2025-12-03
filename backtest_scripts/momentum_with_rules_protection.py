"""
Backtest Momentum Strategy with Crash Protection.

Two modes:
1. Rule-based (default): Simple rules to reduce exposure
2. ML-based (--ml flag): LightGBM crash predictor

Rules (when not using ML):
1. VIX > 25 (high fear)
2. VIX spike > 20% in 5 days
3. SPY drawdown > 5%
4. Momentum volatility in top 10% of past year
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


def calculate_momentum_returns(
    prices_df: pd.DataFrame,
    top_n: int = 10,
    slippage_per_share: float = 0.0
) -> pd.Series:
    """
    Calculate daily returns of a simple momentum strategy.
    Long top N stocks by 12-1 month momentum.

    Args:
        prices_df: DataFrame with stock prices
        top_n: Number of stocks to hold
        slippage_per_share: Slippage cost in dollars per share (e.g., 0.01)
    """
    returns_12m = prices_df.pct_change(252)
    returns_1m = prices_df.pct_change(21)
    momentum = returns_12m - returns_1m

    daily_returns = prices_df.pct_change()

    strategy_returns = []
    prev_holdings = set()

    for i in range(253, len(prices_df)):
        date = prices_df.index[i]
        prev_date = prices_df.index[i-1]

        scores = momentum.iloc[i-1].dropna()
        if len(scores) < top_n:
            continue

        top_stocks = scores.nlargest(top_n).index
        current_holdings = set(top_stocks)

        # Calculate base return
        ret = daily_returns.loc[date, top_stocks].mean()

        # Apply slippage for trades
        if slippage_per_share > 0 and prev_holdings:
            stocks_bought = current_holdings - prev_holdings
            stocks_sold = prev_holdings - current_holdings

            # Slippage cost = $0.01 / price = percentage cost
            slippage_cost = 0.0
            for stock in stocks_bought:
                if stock in prices_df.columns:
                    price = prices_df.loc[prev_date, stock]
                    if pd.notna(price) and price > 0:
                        # Cost to buy (pay more) - applied to 1/top_n of portfolio
                        slippage_cost += (slippage_per_share / price) / top_n

            for stock in stocks_sold:
                if stock in prices_df.columns:
                    price = prices_df.loc[prev_date, stock]
                    if pd.notna(price) and price > 0:
                        # Cost to sell (receive less) - applied to 1/top_n of portfolio
                        slippage_cost += (slippage_per_share / price) / top_n

            ret -= slippage_cost

        prev_holdings = current_holdings
        strategy_returns.append({'date': date, 'return': ret})

    return pd.DataFrame(strategy_returns).set_index('date')['return']


def calculate_momentum_factor_returns(prices_df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Calculate momentum factor returns (for volatility calculation)."""
    returns_12m = prices_df.pct_change(252)
    returns_1m = prices_df.pct_change(21)
    momentum = returns_12m - returns_1m

    daily_returns = prices_df.pct_change()

    factor_returns = []

    for i in range(253, len(prices_df)):
        date = prices_df.index[i]

        scores = momentum.iloc[i-1].dropna()
        if len(scores) < top_n * 2:
            continue

        top_stocks = scores.nlargest(top_n).index
        bottom_stocks = scores.nsmallest(top_n).index

        long_ret = daily_returns.loc[date, top_stocks].mean()
        short_ret = daily_returns.loc[date, bottom_stocks].mean()
        factor_ret = long_ret - short_ret

        factor_returns.append({'date': date, 'return': factor_ret})

    return pd.DataFrame(factor_returns).set_index('date')['return']


def compute_risk_signals(
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    mom_factor_returns: pd.Series,
    vix_threshold: float = 25,
    vix_spike_threshold: float = 0.20,
    spy_dd_threshold: float = -0.05,
    mom_vol_percentile: float = 0.90
) -> pd.DataFrame:
    """
    Compute rule-based risk signals.

    Returns DataFrame with columns:
    - high_vix: VIX > threshold
    - vix_spike: VIX up > 20% in 5 days
    - spy_drawdown: SPY in drawdown > threshold
    - high_mom_vol: Momentum volatility in top percentile
    - reduce_exposure: Any signal triggered
    """
    signals = pd.DataFrame(index=vix_prices.index)

    # Rule 1: High VIX
    signals['high_vix'] = vix_prices > vix_threshold

    # Rule 2: VIX spike
    vix_change_5d = vix_prices.pct_change(5)
    signals['vix_spike'] = vix_change_5d > vix_spike_threshold

    # Rule 3: SPY drawdown
    spy_cummax = spy_prices.expanding().max()
    spy_drawdown = (spy_prices / spy_cummax) - 1
    signals['spy_drawdown'] = spy_drawdown < spy_dd_threshold

    # Rule 4: High momentum volatility
    mom_vol_21d = mom_factor_returns.rolling(21).std() * np.sqrt(252)
    mom_vol_threshold = mom_vol_21d.rolling(252).quantile(mom_vol_percentile)
    signals['high_mom_vol'] = mom_vol_21d > mom_vol_threshold

    # Align indices
    signals = signals.reindex(mom_factor_returns.index)

    # Combined signal: reduce if ANY rule triggers
    signals['reduce_exposure'] = (
        signals['high_vix'].fillna(False) |
        signals['vix_spike'].fillna(False) |
        signals['spy_drawdown'].fillna(False) |
        signals['high_mom_vol'].fillna(False)
    )

    return signals


def backtest_with_rules(
    prices_df: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    test_start: str,
    test_end: str,
    top_n: int = 10,
    reduced_exposure: float = 0.5,  # 50% exposure when risk is high
    vix_threshold: float = 25,
    vix_spike_threshold: float = 0.20,
    spy_dd_threshold: float = -0.05,
    mom_vol_percentile: float = 0.90,
    slippage_per_share: float = 0.0
) -> dict:
    """
    Run backtest comparing baseline vs rule-protected momentum.
    """
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Calculate momentum returns
    full_prices = prices_df[prices_df.index <= test_end_ts]
    all_mom_returns = calculate_momentum_returns(full_prices, top_n, slippage_per_share)
    test_mom_returns = all_mom_returns[
        (all_mom_returns.index >= test_start_ts) &
        (all_mom_returns.index <= test_end_ts)
    ]

    # Calculate momentum factor returns (for volatility)
    mom_factor = calculate_momentum_factor_returns(full_prices, top_n)

    # Compute risk signals
    risk_signals = compute_risk_signals(
        spy_prices, vix_prices, mom_factor,
        vix_threshold=vix_threshold,
        vix_spike_threshold=vix_spike_threshold,
        spy_dd_threshold=spy_dd_threshold,
        mom_vol_percentile=mom_vol_percentile
    )

    # Get test period signals
    test_signals = risk_signals[
        (risk_signals.index >= test_start_ts) &
        (risk_signals.index <= test_end_ts)
    ]

    # Align
    common_dates = test_mom_returns.index.intersection(test_signals.index)
    baseline_returns = test_mom_returns.loc[common_dates]
    risk_on = test_signals.loc[common_dates, 'reduce_exposure']

    # Protected: reduce exposure when risk signals fire
    exposure = np.where(risk_on, reduced_exposure, 1.0)
    protected_returns = baseline_returns * exposure

    # Calculate metrics
    def calc_metrics(returns: pd.Series) -> dict:
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
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

    # Signal statistics
    n_risk_days = risk_on.sum()
    avg_exposure = exposure.mean()

    # Per-rule stats
    rule_stats = {
        'high_vix_days': test_signals['high_vix'].sum(),
        'vix_spike_days': test_signals['vix_spike'].sum(),
        'spy_dd_days': test_signals['spy_drawdown'].sum(),
        'high_mom_vol_days': test_signals['high_mom_vol'].sum(),
    }

    return {
        'baseline': baseline_metrics,
        'protected': protected_metrics,
        'n_risk_days': int(n_risk_days),
        'avg_exposure': float(avg_exposure),
        'rule_stats': rule_stats,
    }


def backtest_with_ml(
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
    Run backtest with ML-based crash protection (walk-forward).
    """
    from sklearn.metrics import roc_auc_score

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    predictor = MomentumCrashPredictor(
        crash_threshold=-0.03,
        crash_horizon=5
    )

    # Train on data up to train_end
    train_prices = prices_df[prices_df.index <= train_end_ts]
    factor_returns = predictor.calculate_momentum_factor_returns(train_prices)

    if len(factor_returns) < 500:
        logger.warning("Insufficient data for crash predictor training")
        return None

    crash_labels = predictor.identify_crash_events(factor_returns)
    train_spy = spy_prices[spy_prices.index <= train_end_ts]
    train_vix = vix_prices[vix_prices.index <= train_end_ts]

    features = predictor.compute_crash_features(
        factor_returns, train_spy, train_vix, train_prices
    )

    stats = predictor.train(features, crash_labels, eval_split=0.2)
    logger.info(f"ML Model - Train AUC: {stats['train_auc']:.3f}, Val AUC: {stats['val_auc']:.3f}")

    # Test period
    full_prices = prices_df[prices_df.index <= test_end_ts]
    all_mom_returns = calculate_momentum_returns(full_prices, top_n)
    test_mom_returns = all_mom_returns[
        (all_mom_returns.index >= test_start_ts) &
        (all_mom_returns.index <= test_end_ts)
    ]

    full_factor = predictor.calculate_momentum_factor_returns(full_prices)
    full_features = predictor.compute_crash_features(
        full_factor, spy_prices, vix_prices, full_prices
    )
    test_features = full_features[
        (full_features.index >= test_start_ts) &
        (full_features.index <= test_end_ts)
    ]

    crash_prob = predictor.predict_crash_probability(test_features)

    common_dates = test_mom_returns.index.intersection(crash_prob.index)
    baseline_returns = test_mom_returns.loc[common_dates]
    crash_prob_aligned = crash_prob.loc[common_dates]

    exposure = np.clip(1 - (crash_prob_aligned / (crash_threshold * 2)), 0, 1)
    protected_returns = baseline_returns * exposure

    def calc_metrics(returns: pd.Series) -> dict:
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
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

    avg_exposure = exposure.mean()

    actual_crashes = predictor.identify_crash_events(full_factor)
    actual_crashes_test = actual_crashes.reindex(common_dates).fillna(0)
    test_auc = roc_auc_score(actual_crashes_test, crash_prob_aligned)

    return {
        'baseline': baseline_metrics,
        'protected': protected_metrics,
        'n_risk_days': int((exposure < 1).sum()),
        'avg_exposure': float(avg_exposure),
        'test_auc': float(test_auc),
        'train_auc': stats['train_auc'],
        'val_auc': stats['val_auc'],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Momentum with rule-based protection')
    parser.add_argument('--top-n', type=int, default=10, help='Number of stocks to hold')
    parser.add_argument('--reduced-exposure', type=float, default=0.5, help='Exposure when risk high (0-1)')
    parser.add_argument('--vix-threshold', type=float, default=25, help='VIX level threshold')
    parser.add_argument('--vix-spike', type=float, default=0.20, help='VIX 5-day spike threshold')
    parser.add_argument('--spy-dd', type=float, default=-0.05, help='SPY drawdown threshold')
    parser.add_argument('--mom-vol-pct', type=float, default=0.90, help='Momentum vol percentile')
    parser.add_argument('--ml', action='store_true', help='Use ML-based crash prediction instead of rules')
    parser.add_argument('--crash-threshold', type=float, default=0.25, help='ML crash probability threshold')
    parser.add_argument('--slippage', type=float, default=0.0, help='Slippage per share in dollars (e.g., 0.01)')
    args = parser.parse_args()

    logger.info("=" * 70)
    if args.ml:
        logger.info("MOMENTUM WITH ML-BASED CRASH PROTECTION")
        logger.info("=" * 70)
        logger.info(f"  Crash threshold: {args.crash_threshold}")
        logger.info(f"  Top N stocks: {args.top_n}")
    else:
        logger.info("MOMENTUM WITH RULE-BASED PROTECTION")
        logger.info("=" * 70)
        logger.info(f"Rules:")
        logger.info(f"  1. VIX > {args.vix_threshold}")
        logger.info(f"  2. VIX 5-day spike > {args.vix_spike:.0%}")
        logger.info(f"  3. SPY drawdown > {abs(args.spy_dd):.0%}")
        logger.info(f"  4. Momentum vol > {args.mom_vol_pct:.0%} percentile")
        logger.info(f"  Reduced exposure: {args.reduced_exposure:.0%}")
    if args.slippage > 0:
        logger.info(f"  Slippage: ${args.slippage:.2f} per share")
    logger.info("=" * 70)

    # Load symbols
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    symbols = pd.read_csv(csv_path)['Symbol'].tolist()

    logger.info(f"Loading data for {len(symbols)} symbols...")

    # Download data
    prices = yf.download(symbols, start='2014-01-01', end='2024-12-31', progress=True)['Close']
    spy = yf.download('SPY', start='2014-01-01', end='2024-12-31', progress=False)['Close']
    vix = yf.download('^VIX', start='2014-01-01', end='2024-12-31', progress=False)['Close']

    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]

    logger.info(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")

    # Test periods (ML mode needs train_end for walk-forward)
    if args.ml:
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
    else:
        periods = [
            {'test_start': '2017-01-01', 'test_end': '2017-12-31'},
            {'test_start': '2018-01-01', 'test_end': '2018-12-31'},
            {'test_start': '2019-01-01', 'test_end': '2019-12-31'},
            {'test_start': '2020-01-01', 'test_end': '2020-12-31'},
            {'test_start': '2021-01-01', 'test_end': '2021-12-31'},
            {'test_start': '2022-01-01', 'test_end': '2022-12-31'},
            {'test_start': '2023-01-01', 'test_end': '2023-12-31'},
            {'test_start': '2024-01-01', 'test_end': '2024-11-29'},
        ]

    results = []

    for period in periods:
        year = period['test_start'][:4]
        logger.info(f"\nTesting {year}...")

        if args.ml:
            result = backtest_with_ml(
                prices, spy, vix,
                period['train_end'], period['test_start'], period['test_end'],
                top_n=args.top_n,
                crash_threshold=args.crash_threshold
            )
            if result is None:
                continue
        else:
            result = backtest_with_rules(
                prices, spy, vix,
                period['test_start'], period['test_end'],
                top_n=args.top_n,
                reduced_exposure=args.reduced_exposure,
                vix_threshold=args.vix_threshold,
                vix_spike_threshold=args.vix_spike,
                spy_dd_threshold=args.spy_dd,
                mom_vol_percentile=args.mom_vol_pct,
                slippage_per_share=args.slippage
            )

        result['year'] = year
        results.append(result)

        logger.info(f"Baseline:  Return={result['baseline']['total_return']*100:+.1f}%  "
                   f"Sharpe={result['baseline']['sharpe']:.2f}  "
                   f"MaxDD={result['baseline']['max_drawdown']*100:.1f}%")
        logger.info(f"Protected: Return={result['protected']['total_return']*100:+.1f}%  "
                   f"Sharpe={result['protected']['sharpe']:.2f}  "
                   f"MaxDD={result['protected']['max_drawdown']*100:.1f}%")
        if args.ml:
            logger.info(f"Test AUC: {result['test_auc']:.3f}, Avg Exposure: {result['avg_exposure']:.1%}")
        else:
            logger.info(f"Risk days: {result['n_risk_days']} ({result['avg_exposure']:.1%} avg exposure)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    print("\n{:<6} {:>12} {:>12} {:>10} {:>10} {:>12}".format(
        "Year", "Base Ret", "Prot Ret", "Base DD", "Prot DD", "Avg Exposure"
    ))
    print("-" * 70)

    total_base = 1.0
    total_prot = 1.0

    for r in results:
        total_base *= (1 + r['baseline']['total_return'])
        total_prot *= (1 + r['protected']['total_return'])

        print("{:<6} {:>11.1f}% {:>11.1f}% {:>9.1f}% {:>9.1f}% {:>11.1f}%".format(
            r['year'],
            r['baseline']['total_return'] * 100,
            r['protected']['total_return'] * 100,
            r['baseline']['max_drawdown'] * 100,
            r['protected']['max_drawdown'] * 100,
            r['avg_exposure'] * 100
        ))

    print("-" * 70)

    avg_base_sharpe = np.mean([r['baseline']['sharpe'] for r in results])
    avg_prot_sharpe = np.mean([r['protected']['sharpe'] for r in results])
    avg_exposure = np.mean([r['avg_exposure'] for r in results])

    print(f"\nCumulative Baseline:  {(total_base-1)*100:+.1f}%")
    print(f"Cumulative Protected: {(total_prot-1)*100:+.1f}%")
    print(f"\nAvg Baseline Sharpe:  {avg_base_sharpe:.2f}")
    print(f"Avg Protected Sharpe: {avg_prot_sharpe:.2f}")
    print(f"Avg Exposure: {avg_exposure:.1%}")

    if args.ml:
        avg_test_auc = np.mean([r['test_auc'] for r in results])
        print(f"Avg Test AUC: {avg_test_auc:.3f}")

    # Return sacrifice
    sacrifice = (total_prot - total_base) / (total_base - 1) * 100 if total_base != 1 else 0
    print(f"\nReturn vs Baseline: {sacrifice:+.1f}%")

    return results


if __name__ == "__main__":
    results = main()
