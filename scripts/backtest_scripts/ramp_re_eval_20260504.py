"""
RAMP Strategy Re-Evaluation 2026-05-04

Diagnostic re-evaluation of the RAMP 0.846 OOS Sharpe baseline.
Three periods:
  - IS (Training):        2017-01-01 to 2021-12-31 (matches baseline)
  - OOS (Original Test):  2022-01-01 to 2024-12-31 (compare to 0.846 baseline)
  - EXTENDED-OOS (New):   2025-01-01 to 2026-04-30 (truly out-of-sample)

Methodology:
  - Identical to ramp_walk_forward_validation.py
  - Yahoo Finance split-adjusted data (auto_adjust=True)
  - 0% transaction costs (research-grade; apples-to-apples vs 0.846)
  - 1/N equal weight within regime-selected top-N
  - Production REGIME_PARAMS used verbatim (no optimization)
  - Return cap +/-20% on individual stocks (data quality filter)

Usage:
    python scripts/backtest_scripts/ramp_re_eval_20260504.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import yfinance as yf

from src.settings import get_local_storage_dir
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# RAMP Parameters - Production values, verbatim from ramp_strategy.py:44-50
# =============================================================================

REGIME_PARAMS = {
    'STRONG_BULL': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 20},
    'WEAK_BULL': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10},
    'SIDEWAYS': {'long_p': 21, 'short_p': 5, 'long_w': 0.5, 'pen_w': 2.0, 'top_n': 5},
    'UNPREDICTABLE': {'long_p': 42, 'short_p': 21, 'long_w': 0.5, 'pen_w': 4.0, 'top_n': 10},
    'BEAR': {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 3.0, 'top_n': 10}
}

DEFAULT_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 4.0, 'top_n': 10}

# Risk management thresholds (production values)
VIX_THRESHOLD = 25.0
SPY_DD_THRESHOLD = -0.05
REDUCED_EXPOSURE = 0.5

# Return cap to filter data errors
MAX_DAILY_RETURN = 0.20  # 20% cap on individual stock returns


# =============================================================================
# Data Loading
# =============================================================================

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbols from canonical universe file."""
    csv_path = Path("config/universes/sp500-2025.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].astype(str).tolist()
        elif 'symbol' in df.columns:
            symbols = df['symbol'].astype(str).tolist()
        else:
            col = df.columns[0]
            symbols = df[col].astype(str).tolist()
        # Filter out any numeric-only strings (likely bad data)
        symbols = [s for s in symbols if not s.isdigit()]
        return symbols
    logger.error(f"Universe file not found: {csv_path}")
    return []


def load_daily_prices_yf(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load daily prices from Yahoo Finance (split-adjusted)."""
    logger.info(f"Downloading {len(symbols)} symbols from Yahoo Finance ({start_date} to {end_date})...")

    yf_symbols = [s.replace('.', '-') for s in symbols]

    # Buffer for momentum lookback
    buffer_start = (pd.to_datetime(start_date) - pd.Timedelta(days=150)).strftime('%Y-%m-%d')

    df = yf.download(
        yf_symbols,
        start=buffer_start,
        end=end_date,
        progress=True,
        auto_adjust=True,
        threads=True
    )

    if df.empty:
        logger.error("No data downloaded from yfinance")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close']
    else:
        closes = df[['Close']]
        closes.columns = [yf_symbols[0]]

    # Convert column names back (BRK-B -> BRK.B)
    closes.columns = [c.replace('-', '.') for c in closes.columns]

    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)

    logger.info(f"Loaded {len(closes)} days, {len(closes.columns)} symbols")
    return closes


def load_market_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPY and VIX data."""
    buffer_start = (pd.to_datetime(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

    spy = yf.download('SPY', start=buffer_start, end=end_date, progress=False, auto_adjust=True)
    vix = yf.download('^VIX', start=buffer_start, end=end_date, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0].lower() for c in spy.columns]
    else:
        spy.columns = [c.lower() for c in spy.columns]

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0].lower() for c in vix.columns]
    else:
        vix.columns = [c.lower() for c in vix.columns]

    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)

    return spy, vix


# =============================================================================
# RAMP Simulation
# =============================================================================

def calculate_momentum(prices_df: pd.DataFrame, params: Dict) -> pd.Series:
    """Calculate RAMP momentum scores."""
    long_p = params['long_p']
    short_p = params['short_p']
    long_w = params['long_w']
    pen_w = params['pen_w']

    long_ret = prices_df.pct_change(long_p)
    short_ret = prices_df.pct_change(short_p)

    momentum = (long_w * long_ret) - (pen_w * short_ret)
    return momentum.iloc[-1].dropna()


def run_backtest(
    prices_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    cap_returns: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Run RAMP backtest on specified period."""

    logger.info(f"Running backtest: {start_date} to {end_date}")

    prices = prices_df[start_date:end_date]
    trading_days = prices.index.tolist()

    logger.info(f"Trading days in period: {len(trading_days)}")

    detector = MarketRegimeDetector()

    spy_close = spy_df['close']
    spy_cummax = spy_close.cummax()
    spy_dd = (spy_close - spy_cummax) / spy_cummax

    results = []

    for i, date in enumerate(trading_days[:-1]):
        # Use only data available up to and including the current date (no lookahead)
        hist_prices = prices_df[prices_df.index <= date]

        if len(hist_prices) < 50:
            continue

        spy_subset = spy_df[spy_df.index <= date]
        vix_subset = vix_df[vix_df.index <= date]

        if len(spy_subset) < 252 or len(vix_subset) < 252:
            regime = 'SIDEWAYS'
        else:
            regime, _ = detector.classify_regime(spy_subset, vix_subset, date)

        params = REGIME_PARAMS.get(regime, DEFAULT_PARAMS)
        top_n = params['top_n']

        momentum = calculate_momentum(hist_prices, params)

        if momentum.empty:
            continue

        top_stocks = momentum.nlargest(top_n).index.tolist()

        vix_value = vix_df.loc[:date, 'close'].iloc[-1] if date in vix_df.index or len(vix_df.loc[:date]) > 0 else 20.0
        spy_dd_value = spy_dd.loc[:date].iloc[-1] if date in spy_dd.index or len(spy_dd.loc[:date]) > 0 else 0.0

        exposure = 1.0
        if vix_value > VIX_THRESHOLD or spy_dd_value < SPY_DD_THRESHOLD:
            exposure = REDUCED_EXPOSURE

        weight = exposure / len(top_stocks) if top_stocks else 0.0

        next_date = trading_days[i + 1]
        today_prices = prices.loc[date]
        next_prices = prices.loc[next_date]

        daily_returns = []
        for symbol in top_stocks:
            if symbol in today_prices.index and symbol in next_prices.index:
                p0 = today_prices[symbol]
                p1 = next_prices[symbol]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = (p1 - p0) / p0
                    if cap_returns:
                        ret = np.clip(ret, -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
                    daily_returns.append(ret)

        if daily_returns:
            portfolio_return = np.mean(daily_returns) * exposure
            results.append({
                'date': next_date,
                'regime': regime,
                'return': portfolio_return,
                'positions': len(daily_returns),
                'exposure': exposure
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df, {}

    returns = results_df['return']
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    n_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'n_days': len(returns),
        'avg_daily_return': returns.mean(),
        'std_daily_return': returns.std()
    }

    return results_df, metrics


def compute_regime_breakdown(results_df: pd.DataFrame) -> Dict:
    """Compute per-regime statistics from backtest results."""
    if results_df.empty:
        return {}

    breakdown = {}
    total_days = len(results_df)

    for regime in results_df['regime'].unique():
        subset = results_df[results_df['regime'] == regime]['return']
        days = len(subset)
        pct_time = days / total_days
        contrib = subset.sum()  # Sum of daily returns = cumulative contribution approx
        avg_ret = subset.mean()
        sharpe_r = (subset.mean() / subset.std()) * np.sqrt(252) if subset.std() > 0 else 0.0

        breakdown[regime] = {
            'days': days,
            'pct_time': pct_time,
            'total_contribution': contrib,
            'avg_daily_return': avg_ret,
            'sharpe': sharpe_r
        }

    return breakdown


def run_re_evaluation(cap_returns: bool = True):
    """Run the three-period RAMP re-evaluation."""

    logger.info("=" * 80)
    logger.info("RAMP RE-EVALUATION 2026-05-04")
    logger.info("=" * 80)
    logger.info("")
    logger.info("METHODOLOGY:")
    logger.info("  - IS Training:       2017-01-01 to 2021-12-31")
    logger.info("  - OOS Test:          2022-01-01 to 2024-12-31  (compare to 0.846 baseline)")
    logger.info("  - EXTENDED-OOS:      2025-01-01 to 2026-04-30  (truly out-of-sample)")
    logger.info("  - Data Source:       Yahoo Finance (split-adjusted, auto_adjust=True)")
    logger.info("  - Transaction Costs: 0% (research-grade; matches 0.846 baseline methodology)")
    logger.info(f"  - Return Cap:        {'+/-' + str(int(MAX_DAILY_RETURN*100)) + '%' if cap_returns else 'None'}")
    logger.info("  - Parameters:        Production REGIME_PARAMS verbatim, no optimization")
    logger.info("")

    symbols = load_sp500_symbols()
    if not symbols:
        logger.error("BLOCKED: No symbols loaded. Check config/universes/sp500-2025.csv")
        sys.exit(1)
    logger.info(f"Universe: {len(symbols)} S&P 500 symbols")

    logger.info("Loading price data (2017-2026, single download)...")
    start_time = time.time()

    # Single download spanning all three periods
    prices_df = load_daily_prices_yf(symbols, '2017-01-01', '2026-04-30')
    spy_df, vix_df = load_market_data('2017-01-01', '2026-04-30')

    if prices_df.empty:
        logger.error("BLOCKED: Price data download failed.")
        sys.exit(1)

    load_time = time.time() - start_time
    logger.info(f"Data loaded in {load_time:.1f}s")

    # IS period
    logger.info("")
    logger.info("-" * 80)
    logger.info("PERIOD 1: IN-SAMPLE (TRAINING) 2017-2021")
    logger.info("-" * 80)
    is_results, is_metrics = run_backtest(
        prices_df, spy_df, vix_df,
        '2017-01-01', '2021-12-31',
        cap_returns=cap_returns
    )

    # OOS period (matches baseline)
    logger.info("")
    logger.info("-" * 80)
    logger.info("PERIOD 2: OUT-OF-SAMPLE 2022-2024 (compare to 0.846 baseline)")
    logger.info("-" * 80)
    oos_results, oos_metrics = run_backtest(
        prices_df, spy_df, vix_df,
        '2022-01-01', '2024-12-31',
        cap_returns=cap_returns
    )

    # EXTENDED-OOS (new, never validated)
    logger.info("")
    logger.info("-" * 80)
    logger.info("PERIOD 3: EXTENDED-OOS 2025-2026 (truly out-of-sample)")
    logger.info("-" * 80)
    ext_results, ext_metrics = run_backtest(
        prices_df, spy_df, vix_df,
        '2025-01-01', '2026-04-30',
        cap_returns=cap_returns
    )

    # Regime breakdown for EXTENDED-OOS
    ext_regime_breakdown = compute_regime_breakdown(ext_results)

    # Print results
    print()
    print("=" * 80)
    print("RAMP RE-EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Universe: {len(symbols)} symbols | Data: yfinance split-adjusted | Costs: 0%")
    print()

    header = f"{'Metric':<25} {'IS 2017-2021':<18} {'OOS 2022-2024':<18} {'EXT-OOS 2025-2026':<20}"
    print(header)
    print("-" * 80)

    metrics_to_show = [
        ('Total Return', 'total_return', '{:.1%}'),
        ('CAGR', 'cagr', '{:.1%}'),
        ('Sharpe Ratio', 'sharpe', '{:.3f}'),
        ('Max Drawdown', 'max_drawdown', '{:.1%}'),
        ('Win Rate', 'win_rate', '{:.1%}'),
        ('Trading Days', 'n_days', '{:.0f}'),
    ]

    for name, key, fmt in metrics_to_show:
        is_val = is_metrics.get(key, 0) if is_metrics else 0
        oos_val = oos_metrics.get(key, 0) if oos_metrics else 0
        ext_val = ext_metrics.get(key, 0) if ext_metrics else 0
        print(f"{name:<25} {fmt.format(is_val):<18} {fmt.format(oos_val):<18} {fmt.format(ext_val):<20}")

    print()

    # Assessment
    print("=" * 80)
    print("BASELINE COMPARISON (OOS 2022-2024)")
    print("=" * 80)
    oos_sharpe = oos_metrics.get('sharpe', 0) if oos_metrics else 0
    baseline = 0.846
    delta_baseline = oos_sharpe - baseline
    print(f"Reference baseline (2025-12-12): {baseline:.3f}")
    print(f"This run OOS Sharpe:             {oos_sharpe:.3f}")
    print(f"Delta:                           {delta_baseline:+.3f}")
    if abs(delta_baseline) <= 0.1:
        print("Assessment: MATCHES BASELINE (within +/-0.1)")
    elif delta_baseline > 0.1:
        print(f"Assessment: ABOVE BASELINE by {delta_baseline:.3f}")
    else:
        print(f"Assessment: BELOW BASELINE by {abs(delta_baseline):.3f}")

    print()
    print("=" * 80)
    print("ALPHA DECAY ASSESSMENT (EXTENDED-OOS 2025-2026)")
    print("=" * 80)
    ext_sharpe = ext_metrics.get('sharpe', 0) if ext_metrics else 0
    decay = baseline - ext_sharpe
    print(f"Baseline Sharpe (2022-2024):     {baseline:.3f}")
    print(f"EXTENDED-OOS Sharpe:             {ext_sharpe:.3f}")
    print(f"Decay vs baseline:               {decay:+.3f}")
    if ext_sharpe < 0:
        print("Status: SEVERE DECAY - strategy may be broken")
    elif decay > 0.3:
        print("Status: MATERIAL DECAY - warrants investigation")
    elif decay > 0.1:
        print("Status: MILD DEGRADATION - worth monitoring")
    else:
        print("Status: HEALTHY - no significant decay detected")

    print()
    print("=" * 80)
    print("REGIME BREAKDOWN (EXTENDED-OOS 2025-2026)")
    print("=" * 80)
    if ext_regime_breakdown:
        print(f"{'Regime':<20} {'Days':<8} {'% Time':<10} {'Cum Contrib':<14} {'Avg Daily':<12} {'Sharpe':<10}")
        print("-" * 75)
        for regime, stats in sorted(ext_regime_breakdown.items(), key=lambda x: -x[1]['days']):
            print(
                f"{regime:<20} {stats['days']:<8} {stats['pct_time']:.1%}       "
                f"{stats['total_contribution']:.3f}        {stats['avg_daily_return']:.4f}      "
                f"{stats['sharpe']:.3f}"
            )

    # Save results
    output_dir = Path('logs/backtesting/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not is_results.empty:
        is_results.to_csv(output_dir / f'{timestamp}_ramp_reeval_in_sample.csv', index=False)
    if not oos_results.empty:
        oos_results.to_csv(output_dir / f'{timestamp}_ramp_reeval_oos.csv', index=False)
    if not ext_results.empty:
        ext_results.to_csv(output_dir / f'{timestamp}_ramp_reeval_ext_oos.csv', index=False)

    summary = {
        'timestamp': timestamp,
        'script': 'ramp_re_eval_20260504.py',
        'universe_file': 'config/universes/sp500-2025.csv',
        'universe_size': len(symbols),
        'data_source': 'Yahoo Finance (split-adjusted, auto_adjust=True)',
        'transaction_costs': 0,
        'return_cap': MAX_DAILY_RETURN if cap_returns else None,
        'regime_params': REGIME_PARAMS,
        'reference_baseline_sharpe': baseline,
        'periods': {
            'in_sample': {
                'range': '2017-01-01 to 2021-12-31',
                'metrics': is_metrics
            },
            'out_of_sample': {
                'range': '2022-01-01 to 2024-12-31',
                'metrics': oos_metrics
            },
            'extended_oos': {
                'range': '2025-01-01 to 2026-04-30',
                'metrics': ext_metrics,
                'regime_breakdown': ext_regime_breakdown
            }
        }
    }

    summary_path = output_dir / f'{timestamp}_ramp_reeval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_dir}")
    print(f"JSON summary: {summary_path}")

    return is_metrics, oos_metrics, ext_metrics, ext_regime_breakdown, len(symbols)


def main():
    parser = argparse.ArgumentParser(description="RAMP Re-Evaluation 2026-05-04")
    parser.add_argument('--no-cap', action='store_true', help='Disable return capping')
    args = parser.parse_args()

    run_re_evaluation(cap_returns=not args.no_cap)


if __name__ == '__main__':
    main()
