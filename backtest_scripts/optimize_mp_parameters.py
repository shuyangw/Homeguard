"""
MP (Momentum Protection) Strategy Parameter Optimization.

Comprehensive grid search over:
- Momentum lookback periods (1m-1w, 3m-1m, 6m-1m, 12m-1m, 1m-1d, 2w-1w)
- Number of holdings (top_n)
- Position sizes
- VIX thresholds
- SPY drawdown thresholds

Tests MP standalone (not combined with OMR) to isolate parameter effects.

Usage:
    python backtest_scripts/optimize_mp_parameters.py
    python backtest_scripts/optimize_mp_parameters.py --quick  # Faster test with fewer combinations
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import argparse
from itertools import product
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

import yfinance as yf

from src.utils.logger import logger

# ============================================================================
# Parameter Grid Definition
# ============================================================================

# Momentum lookback periods: (long_period, short_period, name)
MOMENTUM_PERIODS = [
    (21, 5, '1m-1w'),     # Current production: 1-month minus 1-week
    (63, 21, '3m-1m'),    # Original: 3-month minus 1-month
    (126, 21, '6m-1m'),   # 6-month minus 1-month
    (252, 21, '12m-1m'),  # Classic 12-month momentum
    (21, 1, '1m-1d'),     # 1-month minus 1-day (ultra-short)
    (10, 5, '2w-1w'),     # 2-week minus 1-week (very short)
]

# Number of holdings
TOP_N_VALUES = [5, 10, 15, 20, 25]

# Position size per holding (as percentage)
POSITION_SIZES = [0.05, 0.065, 0.08, 0.10]

# VIX threshold (reduce exposure when VIX exceeds this)
VIX_THRESHOLDS = [20, 25, 30, 35]

# SPY drawdown threshold (reduce exposure when SPY DD exceeds this)
SPY_DD_THRESHOLDS = [-0.03, -0.05, -0.07, -0.10]

# Fixed parameters (not optimized)
REDUCED_EXPOSURE = 0.50  # 50% exposure when risk signals trigger
VIX_SPIKE_THRESHOLD = 0.20  # 20% VIX spike
MOM_VOL_PERCENTILE = 0.90  # 90th percentile momentum volatility

# Backtest period
START_DATE = '2017-01-01'
END_DATE = '2024-12-31'
INITIAL_CAPITAL = 100000


# ============================================================================
# Data Loading
# ============================================================================

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbols from CSV."""
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    try:
        df = pd.read_csv(csv_path)
        return df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Failed to load S&P 500 symbols: {e}")
        # Fallback to top stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'AVGO', 'JPM', 'V', 'UNH', 'MA', 'JNJ', 'XOM', 'PG']


def download_price_data(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download price data for all symbols including SPY and VIX."""
    logger.info(f"Downloading price data for {len(symbols)} symbols...")

    try:
        data = yf.download(
            symbols + ['SPY', '^VIX'],
            start=start,
            end=end,
            progress=False,
            auto_adjust=True
        )

        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data

        logger.info(f"  Downloaded {len(prices)} trading days")
        return prices
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return pd.DataFrame()


# ============================================================================
# MP Backtest Engine (Vectorized for Speed)
# ============================================================================

def run_mp_backtest_vectorized(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    long_period: int,
    short_period: int,
    top_n: int,
    position_size: float,
    vix_threshold: float,
    spy_dd_threshold: float,
    test_start: str,
    test_end: str
) -> pd.Series:
    """
    Vectorized MP backtest for speed.

    Returns daily returns series.

    CRITICAL: Properly handles lookahead bias with shift(1) on all signals.
    """

    symbols = [c for c in prices.columns if c not in ['SPY', '^VIX']]

    # Pre-compute momentum scores
    returns_long = prices.pct_change(long_period, fill_method=None)
    returns_short = prices.pct_change(short_period, fill_method=None)
    momentum = returns_long - returns_short

    # Pre-compute risk signals
    high_vix = vix_prices > vix_threshold

    # VIX spike
    vix_5d_change = vix_prices.pct_change(5)
    vix_spike = vix_5d_change > VIX_SPIKE_THRESHOLD

    # SPY drawdown
    spy_rolling_max = spy_prices.expanding().max()
    spy_drawdown = (spy_prices - spy_rolling_max) / spy_rolling_max
    spy_dd_trigger = spy_drawdown < spy_dd_threshold

    # Momentum volatility (use cross-sectional momentum volatility)
    mom_mean = momentum.mean(axis=1)
    mom_vol = mom_mean.rolling(252).std()
    mom_vol_90pct = mom_vol.expanding().quantile(MOM_VOL_PERCENTILE)
    high_mom_vol = mom_vol > mom_vol_90pct

    # Daily returns (for calculating portfolio return)
    daily_returns = prices.pct_change()

    # Get test period indices
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    try:
        start_idx = prices.index.get_loc(test_start_ts)
    except KeyError:
        start_idx = prices.index.searchsorted(test_start_ts)

    # Need enough history for long period + buffer
    start_idx = max(long_period + 10, start_idx)

    portfolio_returns = []

    for i in range(start_idx, len(prices)):
        date = prices.index[i]

        if date < test_start_ts or date > test_end_ts:
            continue

        # CRITICAL: Use previous day's signals to avoid lookahead bias
        prev_date = prices.index[i - 1]

        # Get momentum scores from previous day
        scores = momentum.loc[prev_date].dropna()

        if len(scores) < top_n:
            portfolio_returns.append({'date': date, 'return': 0.0})
            continue

        # Check risk signals from previous day
        reduce_exposure = (
            high_vix.get(prev_date, False) or
            vix_spike.get(prev_date, False) or
            spy_dd_trigger.get(prev_date, False) or
            high_mom_vol.get(prev_date, False)
        )

        exposure = REDUCED_EXPOSURE if reduce_exposure else 1.0

        # Select top N by momentum
        top_stocks = scores.nlargest(top_n).index.tolist()

        # Calculate return on current day for top stocks
        day_rets = daily_returns.loc[date, top_stocks].dropna()

        if len(day_rets) > 0:
            avg_return = day_rets.mean()

            # Portfolio return = avg return × position size × number of positions × exposure
            portfolio_return = avg_return * position_size * len(day_rets) * exposure
            portfolio_returns.append({'date': date, 'return': portfolio_return})
        else:
            portfolio_returns.append({'date': date, 'return': 0.0})

    return pd.DataFrame(portfolio_returns).set_index('date')['return']


# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if len(returns) == 0 or returns.sum() == 0:
        return {
            'cumulative_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'n_days': len(returns)
        }

    returns = returns.dropna()

    # Cumulative and annualized returns
    cum_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Win rate and average win/loss
    winning = returns[returns > 0]
    losing = returns[returns < 0]

    win_rate = len(winning) / len(returns) if len(returns) > 0 else 0
    avg_win = winning.mean() if len(winning) > 0 else 0
    avg_loss = losing.mean() if len(losing) > 0 else 0

    # Profit factor
    total_wins = winning.sum()
    total_losses = abs(losing.sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    return {
        'cumulative_return': cum_return,
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'n_days': len(returns)
    }


# ============================================================================
# Optimization Engine
# ============================================================================

def run_optimization(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    quick_mode: bool = False
) -> pd.DataFrame:
    """
    Run comprehensive parameter optimization.

    Returns DataFrame with all tested combinations and their metrics.
    """

    logger.info("=" * 80)
    logger.info("MP STRATEGY PARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    logger.info("")

    # Calculate SPY benchmark
    spy_returns = spy_prices.pct_change().loc[START_DATE:END_DATE]
    spy_metrics = calculate_metrics(spy_returns)

    logger.info("SPY Benchmark:")
    logger.info(f"  Cumulative Return: {spy_metrics['cumulative_return']:+.1%}")
    logger.info(f"  Annual Return:     {spy_metrics['annual_return']:+.1%}")
    logger.info(f"  Sharpe Ratio:      {spy_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:      {spy_metrics['max_drawdown']:.1%}")
    logger.info("")

    # Define parameter grid
    if quick_mode:
        # Reduced grid for faster testing
        momentum_periods = [(21, 5, '1m-1w'), (63, 21, '3m-1m'), (252, 21, '12m-1m')]
        top_n_values = [10, 20]
        position_sizes = [0.065, 0.10]
        vix_thresholds = [25, 30]
        spy_dd_thresholds = [-0.05, -0.07]
    else:
        # Full grid
        momentum_periods = MOMENTUM_PERIODS
        top_n_values = TOP_N_VALUES
        position_sizes = POSITION_SIZES
        vix_thresholds = VIX_THRESHOLDS
        spy_dd_thresholds = SPY_DD_THRESHOLDS

    # Generate all combinations
    param_combinations = list(product(
        momentum_periods,
        top_n_values,
        position_sizes,
        vix_thresholds,
        spy_dd_thresholds
    ))

    total_combinations = len(param_combinations)
    logger.info(f"Testing {total_combinations} parameter combinations...")
    logger.info(f"Estimated time: {total_combinations * 0.5 / 60:.1f} minutes")
    logger.info("")

    results = []

    for idx, (mom_params, top_n, pos_size, vix_thresh, spy_dd_thresh) in enumerate(param_combinations, 1):
        long_period, short_period, mom_name = mom_params

        if idx % 20 == 0 or idx == 1:
            logger.info(f"Progress: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%)")

        try:
            # Run backtest
            returns = run_mp_backtest_vectorized(
                prices=prices,
                spy_prices=spy_prices,
                vix_prices=vix_prices,
                long_period=long_period,
                short_period=short_period,
                top_n=top_n,
                position_size=pos_size,
                vix_threshold=vix_thresh,
                spy_dd_threshold=spy_dd_thresh,
                test_start=START_DATE,
                test_end=END_DATE
            )

            # Calculate metrics
            metrics = calculate_metrics(returns)

            # Store results
            results.append({
                'momentum': mom_name,
                'long_period': long_period,
                'short_period': short_period,
                'top_n': top_n,
                'position_size': pos_size,
                'vix_threshold': vix_thresh,
                'spy_dd_threshold': spy_dd_thresh,
                **metrics,
                'alpha_vs_spy': metrics['annual_return'] - spy_metrics['annual_return']
            })

        except Exception as e:
            logger.error(f"Error with params {mom_name}, top_n={top_n}, pos={pos_size}: {e}")
            continue

    logger.info("")
    logger.info(f"Optimization complete. Tested {len(results)}/{total_combinations} combinations successfully.")
    logger.info("")

    return pd.DataFrame(results)


# ============================================================================
# Results Analysis
# ============================================================================

def analyze_results(results_df: pd.DataFrame, spy_metrics: Dict[str, float]):
    """Analyze and display optimization results."""

    logger.info("=" * 80)
    logger.info("OPTIMIZATION RESULTS ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    # Top 10 by Sharpe Ratio
    logger.info("TOP 10 CONFIGURATIONS BY SHARPE RATIO:")
    logger.info("-" * 80)
    top_sharpe = results_df.nlargest(10, 'sharpe_ratio')

    for idx, row in top_sharpe.iterrows():
        logger.info(f"\nRank #{idx+1}:")
        logger.info(f"  Momentum:     {row['momentum']} ({row['long_period']}-{row['short_period']} days)")
        logger.info(f"  Top N:        {row['top_n']}")
        logger.info(f"  Position:     {row['position_size']:.1%}")
        logger.info(f"  VIX Thresh:   {row['vix_threshold']:.0f}")
        logger.info(f"  SPY DD Thresh: {row['spy_dd_threshold']:.1%}")
        logger.info(f"  Sharpe:       {row['sharpe_ratio']:.2f}")
        logger.info(f"  Annual Return: {row['annual_return']:+.1%}")
        logger.info(f"  Max Drawdown: {row['max_drawdown']:.1%}")
        logger.info(f"  Alpha vs SPY: {row['alpha_vs_spy']:+.1%}")

    logger.info("")
    logger.info("=" * 80)

    # Top 10 by Cumulative Return
    logger.info("\nTOP 10 CONFIGURATIONS BY CUMULATIVE RETURN:")
    logger.info("-" * 80)
    top_return = results_df.nlargest(10, 'cumulative_return')

    for idx, row in top_return.iterrows():
        logger.info(f"\nRank #{idx+1}:")
        logger.info(f"  Momentum:     {row['momentum']} ({row['long_period']}-{row['short_period']} days)")
        logger.info(f"  Top N:        {row['top_n']}")
        logger.info(f"  Position:     {row['position_size']:.1%}")
        logger.info(f"  VIX Thresh:   {row['vix_threshold']:.0f}")
        logger.info(f"  SPY DD Thresh: {row['spy_dd_threshold']:.1%}")
        logger.info(f"  Cum Return:   {row['cumulative_return']:+.1%}")
        logger.info(f"  Annual Return: {row['annual_return']:+.1%}")
        logger.info(f"  Sharpe:       {row['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {row['max_drawdown']:.1%}")

    logger.info("")
    logger.info("=" * 80)

    # Top 10 by Alpha vs SPY
    logger.info("\nTOP 10 CONFIGURATIONS BY ALPHA VS SPY:")
    logger.info("-" * 80)
    top_alpha = results_df.nlargest(10, 'alpha_vs_spy')

    for idx, row in top_alpha.iterrows():
        logger.info(f"\nRank #{idx+1}:")
        logger.info(f"  Momentum:     {row['momentum']} ({row['long_period']}-{row['short_period']} days)")
        logger.info(f"  Top N:        {row['top_n']}")
        logger.info(f"  Position:     {row['position_size']:.1%}")
        logger.info(f"  VIX Thresh:   {row['vix_threshold']:.0f}")
        logger.info(f"  SPY DD Thresh: {row['spy_dd_threshold']:.1%}")
        logger.info(f"  Alpha:        {row['alpha_vs_spy']:+.1%}")
        logger.info(f"  Annual Return: {row['annual_return']:+.1%}")
        logger.info(f"  Sharpe:       {row['sharpe_ratio']:.2f}")

    logger.info("")
    logger.info("=" * 80)


def save_results(results_df: pd.DataFrame, spy_metrics: Dict[str, float]):
    """Save results to CSV and markdown report."""

    timestamp = datetime.now().strftime('%Y%m%d')

    # Save full results to CSV
    csv_path = PROJECT_ROOT / 'docs' / 'reports' / f'{timestamp}_MP_OPTIMIZATION_FULL_RESULTS.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Full results saved to: {csv_path}")

    # Generate markdown report
    md_path = PROJECT_ROOT / 'docs' / 'reports' / f'{timestamp}_MP_OPTIMIZATION_RESULTS.md'

    with open(md_path, 'w') as f:
        f.write(f"# MP Strategy Parameter Optimization Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n\n")
        f.write(f"**Combinations Tested:** {len(results_df)}\n\n")

        f.write("## SPY Benchmark\n\n")
        f.write(f"- Cumulative Return: {spy_metrics['cumulative_return']:+.2%}\n")
        f.write(f"- Annual Return: {spy_metrics['annual_return']:+.2%}\n")
        f.write(f"- Sharpe Ratio: {spy_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- Max Drawdown: {spy_metrics['max_drawdown']:.2%}\n\n")

        # Top 3 by Sharpe
        f.write("## Top 3 Configurations by Sharpe Ratio\n\n")
        top_sharpe = results_df.nlargest(3, 'sharpe_ratio')
        for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
            f.write(f"### #{i} - Sharpe {row['sharpe_ratio']:.2f}\n\n")
            f.write(f"- **Momentum:** {row['momentum']} ({row['long_period']}-{row['short_period']} days)\n")
            f.write(f"- **Top N:** {row['top_n']}\n")
            f.write(f"- **Position Size:** {row['position_size']:.1%}\n")
            f.write(f"- **VIX Threshold:** {row['vix_threshold']:.0f}\n")
            f.write(f"- **SPY DD Threshold:** {row['spy_dd_threshold']:.1%}\n\n")
            f.write(f"**Performance:**\n")
            f.write(f"- Annual Return: {row['annual_return']:+.2%}\n")
            f.write(f"- Cumulative Return: {row['cumulative_return']:+.2%}\n")
            f.write(f"- Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
            f.write(f"- Sortino Ratio: {row['sortino_ratio']:.2f}\n")
            f.write(f"- Max Drawdown: {row['max_drawdown']:.2%}\n")
            f.write(f"- Win Rate: {row['win_rate']:.1%}\n")
            f.write(f"- Alpha vs SPY: {row['alpha_vs_spy']:+.2%}\n\n")

        # Top 3 by Cumulative Return
        f.write("## Top 3 Configurations by Cumulative Return\n\n")
        top_return = results_df.nlargest(3, 'cumulative_return')
        for i, (_, row) in enumerate(top_return.iterrows(), 1):
            f.write(f"### #{i} - Cum Return {row['cumulative_return']:+.2%}\n\n")
            f.write(f"- **Momentum:** {row['momentum']} ({row['long_period']}-{row['short_period']} days)\n")
            f.write(f"- **Top N:** {row['top_n']}\n")
            f.write(f"- **Position Size:** {row['position_size']:.1%}\n")
            f.write(f"- **VIX Threshold:** {row['vix_threshold']:.0f}\n")
            f.write(f"- **SPY DD Threshold:** {row['spy_dd_threshold']:.1%}\n\n")
            f.write(f"**Performance:**\n")
            f.write(f"- Cumulative Return: {row['cumulative_return']:+.2%}\n")
            f.write(f"- Annual Return: {row['annual_return']:+.2%}\n")
            f.write(f"- Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
            f.write(f"- Max Drawdown: {row['max_drawdown']:.2%}\n")
            f.write(f"- Alpha vs SPY: {row['alpha_vs_spy']:+.2%}\n\n")

        # Recommendations
        f.write("## Recommended Configuration\n\n")
        # Best balance of Sharpe and return
        results_df['score'] = results_df['sharpe_ratio'] * 0.6 + results_df['cumulative_return'] * 0.4
        best = results_df.nlargest(1, 'score').iloc[0]

        f.write(f"**Based on balanced scoring (60% Sharpe, 40% Return):**\n\n")
        f.write(f"- **Momentum:** {best['momentum']} ({best['long_period']}-{best['short_period']} days)\n")
        f.write(f"- **Top N:** {int(best['top_n'])}\n")
        f.write(f"- **Position Size:** {best['position_size']:.1%}\n")
        f.write(f"- **VIX Threshold:** {best['vix_threshold']:.0f}\n")
        f.write(f"- **SPY DD Threshold:** {best['spy_dd_threshold']:.1%}\n\n")
        f.write(f"**Expected Performance:**\n")
        f.write(f"- Annual Return: {best['annual_return']:+.2%}\n")
        f.write(f"- Sharpe Ratio: {best['sharpe_ratio']:.2f}\n")
        f.write(f"- Max Drawdown: {best['max_drawdown']:.2%}\n")
        f.write(f"- Alpha vs SPY: {best['alpha_vs_spy']:+.2%}\n\n")

        # Parameter insights
        f.write("## Parameter Insights\n\n")
        f.write("### Momentum Period Analysis\n\n")
        mom_perf = results_df.groupby('momentum').agg({
            'sharpe_ratio': 'mean',
            'annual_return': 'mean',
            'max_drawdown': 'mean'
        }).sort_values('sharpe_ratio', ascending=False)

        f.write("| Momentum | Avg Sharpe | Avg Annual Return | Avg Max DD |\n")
        f.write("|----------|-----------|-------------------|------------|\n")
        for mom, row in mom_perf.iterrows():
            f.write(f"| {mom} | {row['sharpe_ratio']:.2f} | {row['annual_return']:+.2%} | {row['max_drawdown']:.2%} |\n")

        f.write("\n### Top N Analysis\n\n")
        topn_perf = results_df.groupby('top_n').agg({
            'sharpe_ratio': 'mean',
            'annual_return': 'mean',
            'max_drawdown': 'mean'
        }).sort_values('sharpe_ratio', ascending=False)

        f.write("| Top N | Avg Sharpe | Avg Annual Return | Avg Max DD |\n")
        f.write("|-------|-----------|-------------------|------------|\n")
        for topn, row in topn_perf.iterrows():
            f.write(f"| {int(topn)} | {row['sharpe_ratio']:.2f} | {row['annual_return']:+.2%} | {row['max_drawdown']:.2%} |\n")

    logger.info(f"Markdown report saved to: {md_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(description='MP Strategy Parameter Optimization')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer combinations')
    args = parser.parse_args()

    # Load S&P 500 symbols
    symbols = load_sp500_symbols()

    # Download data
    prices = download_price_data(symbols, START_DATE, END_DATE)

    if prices.empty:
        logger.error("Failed to download price data")
        return

    spy_prices = prices['SPY'] if 'SPY' in prices.columns else None
    vix_prices = prices['^VIX'] if '^VIX' in prices.columns else None

    if spy_prices is None or vix_prices is None:
        logger.error("Missing SPY or VIX data")
        return

    # Filter to only S&P 500 stocks (exclude SPY and VIX)
    stock_symbols = [s for s in symbols if s in prices.columns]
    prices_filtered = prices[stock_symbols]

    logger.info(f"Loaded {len(stock_symbols)} S&P 500 stocks")
    logger.info(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    logger.info("")

    # Calculate SPY metrics for reference
    spy_returns = spy_prices.pct_change().loc[START_DATE:END_DATE]
    spy_metrics = calculate_metrics(spy_returns)

    # Run optimization
    results_df = run_optimization(
        prices=prices_filtered,
        spy_prices=spy_prices,
        vix_prices=vix_prices,
        quick_mode=args.quick
    )

    # Analyze results
    analyze_results(results_df, spy_metrics)

    # Save results
    save_results(results_df, spy_metrics)

    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
