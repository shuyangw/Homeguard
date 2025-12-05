"""
Breakout Strategies Backtest.

Compares two breakout strategies:
1. Volume Breakout - 4x volume spike with price confirmation
2. ATR Squeeze Breakout - Volatility compression -> expansion

Both use weekly rebalancing with top 10 stock selection.

Usage:
    python backtest_scripts/breakout_strategies_backtest.py --start 2017 --end 2024
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TOP_N = 10
VOLUME_MULTIPLIER = 4.0  # Reset to baseline
VOLUME_LOOKBACK = 5
MIN_PRICE_STRENGTH = 0.7
ATR_PERIOD = 14
SQUEEZE_PERCENTILE = 30  # OPTIMIZED: increased from 20 for looser squeeze
SQUEEZE_LOOKBACK = 100
CHANNEL_PERIOD = 20
RISK_FREE_RATE = 0.04


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbols from CSV."""
    csv_path = Path(__file__).parent.parent / "backtest_lists" / "sp500-2025.csv"
    df = pd.read_csv(csv_path)
    symbols = df['Symbol'].tolist()
    symbols = [s.replace('.', '-') for s in symbols]
    logger.info(f"[BRK] Loaded {len(symbols)} S&P 500 symbols")
    return symbols


def download_ohlcv_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV data from yfinance.

    Returns:
        Tuple of (close_df, high_df, low_df, volume_df)
    """
    logger.info(f"[BRK] Downloading OHLCV data for {len(symbols)} symbols...")

    # Extra lookback for indicators
    lookback_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

    data = yf.download(
        symbols,
        start=lookback_start,
        end=end_date,
        progress=True,
        threads=True,
        auto_adjust=True
    )

    # Extract OHLCV
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data['Close']
        high_df = data['High']
        low_df = data['Low']
        volume_df = data['Volume']
    else:
        close_df = data[['Close']].rename(columns={'Close': symbols[0]})
        high_df = data[['High']].rename(columns={'High': symbols[0]})
        low_df = data[['Low']].rename(columns={'Low': symbols[0]})
        volume_df = data[['Volume']].rename(columns={'Volume': symbols[0]})

    # Clean data
    close_df = close_df.dropna(axis=1, how='all').ffill(limit=5)
    high_df = high_df.dropna(axis=1, how='all').ffill(limit=5)
    low_df = low_df.dropna(axis=1, how='all').ffill(limit=5)
    volume_df = volume_df.dropna(axis=1, how='all').ffill(limit=5)

    logger.info(f"[BRK] Downloaded {len(close_df.columns)} symbols with data")
    if len(close_df) > 0:
        logger.info(f"[BRK]   Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")

    return close_df, high_df, low_df, volume_df


def download_spy_benchmark(start_date: str, end_date: str) -> pd.Series:
    """Download SPY for benchmark comparison."""
    lookback_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    spy = yf.download('SPY', start=lookback_start, end=end_date, progress=False, auto_adjust=True)

    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']

    spy_close = pd.Series(spy_close.values, index=pd.DatetimeIndex(spy.index), name='SPY')
    return spy_close


# ============================================================================
# VOLUME BREAKOUT STRATEGY
# ============================================================================

def calculate_volume_scores(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    volume_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate volume breakout scores for ranking.

    Score = volume_ratio * price_strength (where conditions met)
    """
    # Average volume (shifted to avoid lookahead)
    avg_volume = volume_df.rolling(window=VOLUME_LOOKBACK).mean().shift(1)
    volume_ratio = volume_df / avg_volume

    # Price strength: where close is in daily range
    daily_range = high_df - low_df
    price_strength = (close_df - low_df) / daily_range.replace(0, np.nan)
    price_strength = price_strength.fillna(0.5)

    # Valid signals: volume spike AND bullish price action
    valid_mask = (volume_ratio >= VOLUME_MULTIPLIER) & (price_strength >= MIN_PRICE_STRENGTH)

    # Score
    score = (volume_ratio * price_strength).where(valid_mask, 0)

    return score


def select_volume_breakout_stocks(scores: pd.Series, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """Select top N stocks by volume breakout score."""
    valid_scores = scores[scores > 0].dropna()
    if len(valid_scores) == 0:
        return []
    ranked = valid_scores.sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


# ============================================================================
# ATR SQUEEZE STRATEGY
# ============================================================================

def calculate_atr(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.DataFrame:
    """Calculate Average True Range."""
    high_low = high_df - low_df
    high_close = abs(high_df - close_df.shift(1))
    low_close = abs(low_df - close_df.shift(1))

    true_range = pd.DataFrame(index=high_df.index, columns=high_df.columns)
    for col in high_df.columns:
        tr = pd.concat([high_low[col], high_close[col], low_close[col]], axis=1).max(axis=1)
        true_range[col] = tr

    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_atr_percentile(atr: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ATR percentile (normalized) over lookback period."""
    atr_normalized = atr / close_df

    def percentile_rank(x):
        if len(x.dropna()) < 10:
            return np.nan
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100

    atr_percentile = atr_normalized.rolling(window=SQUEEZE_LOOKBACK).apply(
        percentile_rank, raw=False
    )
    return atr_percentile


def calculate_atr_squeeze_scores(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate ATR squeeze breakout scores.

    Score = (100 - atr_percentile) * breakout_strength (where conditions met)
    """
    # ATR and percentile
    atr = calculate_atr(high_df, low_df, close_df)
    atr_percentile = calculate_atr_percentile(atr, close_df)

    # Squeeze condition (shifted for lookahead prevention)
    in_squeeze = (atr_percentile < SQUEEZE_PERCENTILE).shift(1)

    # Channel breakout
    channel_high = high_df.rolling(window=CHANNEL_PERIOD).max().shift(1)
    breakout = close_df > channel_high

    # ATR expanding
    atr_expanding = atr > atr.shift(1)

    # Momentum positive (10-day ROC > 0)
    roc = close_df.pct_change(10)
    momentum_positive = roc > 0

    # Valid breakout: all conditions
    valid_breakout = in_squeeze & breakout & atr_expanding & momentum_positive

    # Breakout strength
    breakout_strength = (close_df - channel_high) / channel_high
    breakout_strength = breakout_strength.clip(lower=0)

    # Score
    score = ((100 - atr_percentile) * breakout_strength).where(valid_breakout, 0)

    return score


def select_atr_squeeze_stocks(scores: pd.Series, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """Select top N stocks by ATR squeeze score."""
    valid_scores = scores[scores > 0].dropna()
    if len(valid_scores) == 0:
        return []
    ranked = valid_scores.sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def is_first_trading_day_of_week(
    date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex
) -> bool:
    """Check if date is the first trading day of its week."""
    week_start = date - pd.Timedelta(days=date.dayofweek)
    week_end = week_start + pd.Timedelta(days=6)

    week_dates = trading_dates[
        (trading_dates >= week_start) &
        (trading_dates <= week_end)
    ]

    if len(week_dates) == 0:
        return False

    return date == week_dates[0]


def run_backtest(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    strategy: str = 'volume',
    top_n: int = DEFAULT_TOP_N
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run weekly rebalancing backtest.

    Args:
        close_df, high_df, low_df, volume_df: OHLCV data
        start_date: Backtest start
        end_date: Backtest end
        strategy: 'volume' or 'atr_squeeze'
        top_n: Number of positions

    Returns:
        Tuple of (portfolio_value_series, trades_df)
    """
    logger.info(f"[BRK] Running {strategy.upper()} backtest: {start_date} to {end_date}")

    # Calculate scores based on strategy
    if strategy == 'volume':
        all_scores = calculate_volume_scores(close_df, high_df, low_df, volume_df)
    else:
        all_scores = calculate_atr_squeeze_scores(close_df, high_df, low_df)

    # Filter to backtest period
    mask = (close_df.index >= start_date) & (close_df.index <= end_date)
    backtest_dates = close_df.index[mask]

    # Initialize
    portfolio_value = 100000.0
    cash = portfolio_value
    positions: Dict[str, float] = {}

    portfolio_history = []
    trades_log = []

    last_rebalance_week = None

    for date in backtest_dates:
        current_week = (date.year, date.isocalendar()[1])

        # Weekly rebalance check
        should_rebalance = (
            is_first_trading_day_of_week(date, backtest_dates) and
            current_week != last_rebalance_week
        )

        if should_rebalance:
            # Get current scores
            date_scores = all_scores.loc[date].dropna()

            # Select top stocks
            if strategy == 'volume':
                new_holdings = select_volume_breakout_stocks(date_scores, top_n)
            else:
                new_holdings = select_atr_squeeze_stocks(date_scores, top_n)

            if len(new_holdings) > 0:
                # Calculate current portfolio value
                current_prices = close_df.loc[date]
                port_value = cash
                for sym, shares in positions.items():
                    if sym in current_prices.index:
                        price = current_prices[sym]
                        if pd.notna(price):
                            port_value += shares * price

                # Sell all positions
                for sym, shares in list(positions.items()):
                    if sym in current_prices.index:
                        price = current_prices[sym]
                        if pd.notna(price):
                            sell_value = shares * price
                            cash += sell_value
                            trades_log.append({
                                'date': date,
                                'symbol': sym,
                                'action': 'sell',
                                'shares': shares,
                                'price': price,
                                'value': sell_value
                            })
                positions.clear()

                # Buy new positions
                target_value = cash / len(new_holdings)

                for sym in new_holdings:
                    if sym in current_prices.index:
                        price = current_prices[sym]
                        if pd.notna(price) and price > 0:
                            shares = int(target_value / price)
                            if shares > 0:
                                cost = shares * price
                                cash -= cost
                                positions[sym] = shares
                                trades_log.append({
                                    'date': date,
                                    'symbol': sym,
                                    'action': 'buy',
                                    'shares': shares,
                                    'price': price,
                                    'value': cost,
                                    'score': date_scores.get(sym, 0)
                                })

                last_rebalance_week = current_week
                logger.debug(f"[BRK] {strategy.upper()} rebalanced on {date.date()}: {len(positions)} positions")

        # Calculate daily portfolio value
        current_prices = close_df.loc[date]
        port_value = cash
        for sym, shares in positions.items():
            if sym in current_prices.index:
                price = current_prices[sym]
                if pd.notna(price):
                    port_value += shares * price

        portfolio_history.append({
            'date': date,
            'portfolio_value': port_value,
            'cash': cash,
            'num_positions': len(positions)
        })

    # Create output
    portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    return portfolio_df['portfolio_value'], trades_df


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, float]:
    """Calculate performance metrics."""
    if len(returns) == 0:
        return {
            'total_return': 0,
            'annual_return': 0,
            'annual_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }

    total_return = (returns + 1).prod() - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }


def calculate_yearly_metrics(
    portfolio_values: pd.Series,
    spy_values: pd.Series
) -> pd.DataFrame:
    """Calculate metrics for each year."""
    results = []

    # Remove timezone if present
    if portfolio_values.index.tz is not None:
        portfolio_values.index = portfolio_values.index.tz_localize(None)
    if spy_values.index.tz is not None:
        spy_values.index = spy_values.index.tz_localize(None)

    years = portfolio_values.index.year.unique()

    for year in years:
        year_port = portfolio_values[portfolio_values.index.year == year]
        year_spy = spy_values[spy_values.index.year == year]

        if len(year_port) < 20:
            continue

        port_returns = year_port.pct_change().dropna()
        spy_returns = year_spy.pct_change().dropna()

        if len(port_returns) == 0 or len(spy_returns) == 0:
            continue

        port_metrics = calculate_metrics(port_returns)
        spy_metrics = calculate_metrics(spy_returns)

        results.append({
            'year': year,
            'strategy_return': port_metrics['total_return'],
            'spy_return': spy_metrics['total_return'],
            'alpha': port_metrics['total_return'] - spy_metrics['total_return'],
            'strategy_sharpe': port_metrics['sharpe_ratio'],
            'max_drawdown': port_metrics['max_drawdown'],
            'beat_spy': port_metrics['total_return'] > spy_metrics['total_return']
        })

    return pd.DataFrame(results)


# ============================================================================
# REPORTING
# ============================================================================

def print_strategy_results(
    strategy_name: str,
    yearly_results: pd.DataFrame,
    full_metrics: Dict[str, float]
):
    """Print results for one strategy."""
    print(f"\n{strategy_name.upper()} (Weekly Rebalance)")
    print("-" * 70)
    print(f"{'Year':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>10} {'Sharpe':>8} {'Max DD':>10} {'Beat SPY':>10}")
    print("-" * 70)

    for _, row in yearly_results.iterrows():
        beat = "YES" if row['beat_spy'] else "no"
        print(
            f"{int(row['year']):<6} "
            f"{row['strategy_return']:>+9.1%} "
            f"{row['spy_return']:>+9.1%} "
            f"{row['alpha']:>+9.1%} "
            f"{row['strategy_sharpe']:>8.2f} "
            f"{row['max_drawdown']:>9.1%} "
            f"{beat:>10}"
        )

    print("-" * 70)
    print(f"Full Period: Return {full_metrics['total_return']:+.1%}, "
          f"Sharpe {full_metrics['sharpe_ratio']:.2f}, "
          f"Max DD {full_metrics['max_drawdown']:.1%}")


def print_comparison(results: Dict[str, Dict]):
    """Print comparison of all strategies."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'CAGR':>10} {'Sharpe':>10} {'Max DD':>10} {'Beat SPY':>10}")
    print("-" * 70)

    for name, data in results.items():
        metrics = data['full_metrics']
        yearly = data['yearly_results']
        beat_count = yearly['beat_spy'].sum() if len(yearly) > 0 else 0
        total_years = len(yearly)
        beat_str = f"{beat_count}/{total_years}" if total_years > 0 else "N/A"

        print(
            f"{name:<20} "
            f"{metrics['annual_return']:>+9.1%} "
            f"{metrics['sharpe_ratio']:>10.2f} "
            f"{metrics['max_drawdown']:>9.1%} "
            f"{beat_str:>10}"
        )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Breakout Strategies Backtest")
    parser.add_argument('--start', type=int, default=2017, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N, help='Number of positions')
    args = parser.parse_args()

    logger.info("[BRK] Starting Breakout Strategies Backtest")
    logger.info(f"[BRK]   Period: {args.start} - {args.end}")
    logger.info(f"[BRK]   Top N: {args.top_n}")

    # Load symbols
    symbols = load_sp500_symbols()

    # Download data
    start_date = f"{args.start - 1}-01-01"
    end_date = f"{args.end}-12-31"

    close_df, high_df, low_df, volume_df = download_ohlcv_data(symbols, start_date, end_date)
    spy_prices = download_spy_benchmark(start_date, end_date)

    # Backtest dates
    full_start = f"{args.start}-01-01"
    full_end = f"{args.end}-12-31"

    results = {}

    # Run Volume Breakout
    print("\n" + "=" * 70)
    print("BREAKOUT STRATEGIES BACKTEST COMPARISON")
    print("=" * 70)

    logger.info("[BRK] Running Volume Breakout strategy...")
    vol_portfolio, vol_trades = run_backtest(
        close_df, high_df, low_df, volume_df,
        full_start, full_end,
        strategy='volume',
        top_n=args.top_n
    )

    vol_returns = vol_portfolio.pct_change().dropna()
    vol_metrics = calculate_metrics(vol_returns)
    vol_yearly = calculate_yearly_metrics(vol_portfolio, spy_prices)

    results['Volume Breakout'] = {
        'full_metrics': vol_metrics,
        'yearly_results': vol_yearly
    }

    print_strategy_results('Volume Breakout', vol_yearly, vol_metrics)

    # Run ATR Squeeze Breakout
    logger.info("[BRK] Running ATR Squeeze strategy...")
    atr_portfolio, atr_trades = run_backtest(
        close_df, high_df, low_df, volume_df,
        full_start, full_end,
        strategy='atr_squeeze',
        top_n=args.top_n
    )

    atr_returns = atr_portfolio.pct_change().dropna()
    atr_metrics = calculate_metrics(atr_returns)
    atr_yearly = calculate_yearly_metrics(atr_portfolio, spy_prices)

    results['ATR Squeeze'] = {
        'full_metrics': atr_metrics,
        'yearly_results': atr_yearly
    }

    print_strategy_results('ATR Squeeze', atr_yearly, atr_metrics)

    # Add High52 reference (from previous backtest)
    results['High52 Monthly'] = {
        'full_metrics': {
            'annual_return': 0.133,
            'sharpe_ratio': 0.52,
            'max_drawdown': -0.291
        },
        'yearly_results': pd.DataFrame([
            {'beat_spy': True}, {'beat_spy': True}, {'beat_spy': True}, {'beat_spy': False},
            {'beat_spy': False}, {'beat_spy': True}, {'beat_spy': False}, {'beat_spy': False}
        ])
    }

    # Print comparison
    print_comparison(results)

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")

    vol_yearly.to_csv(output_dir / f"{timestamp}_VOLUME_BREAKOUT_RESULTS.csv", index=False)
    atr_yearly.to_csv(output_dir / f"{timestamp}_ATR_SQUEEZE_RESULTS.csv", index=False)

    logger.info(f"[BRK] Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
