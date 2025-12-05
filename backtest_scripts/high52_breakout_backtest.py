"""
52-Week High Monthly Breakout Backtest.

Research-backed strategy:
- Stocks near 52-week highs outperform due to anchoring bias
- Monthly rebalancing is CRITICAL (weekly loses money due to churn)
- Top 10 stocks closest to highs, equal weighted

Usage:
    python backtest_scripts/high52_breakout_backtest.py --start 2017 --end 2024
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TOP_N = 10
DEFAULT_MAX_DISTANCE_PCT = 0.05  # 5% from 52-week high
LOOKBACK_DAYS = 252  # Trading days in a year
RISK_FREE_RATE = 0.04


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sp500_symbols() -> List[str]:
    """Load S&P 500 symbols from CSV."""
    csv_path = Path(__file__).parent.parent / "backtest_lists" / "sp500-2025.csv"
    df = pd.read_csv(csv_path)
    symbols = df['Symbol'].tolist()

    # Clean symbols (handle BRK.B -> BRK-B for yfinance)
    symbols = [s.replace('.', '-') for s in symbols]

    logger.info(f"[H52] Loaded {len(symbols)} S&P 500 symbols")
    return symbols


def download_price_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.

    Args:
        symbols: List of symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with adjusted close prices (columns=symbols, index=dates)
    """
    logger.info(f"[H52] Downloading price data for {len(symbols)} symbols...")
    logger.info(f"[H52]   Period: {start_date} to {end_date}")

    # Need extra data for 52-week lookback
    lookback_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

    try:
        # yfinance now uses auto_adjust=True by default, so 'Close' is adjusted
        data = yf.download(
            symbols,
            start=lookback_start,
            end=end_date,
            progress=True,
            threads=True,
            auto_adjust=True  # Explicit for clarity
        )

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, columns are (Price, Ticker)
            if 'Close' in data.columns.get_level_values(0):
                prices = data['Close']
            else:
                # Fallback to first level
                prices = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]]
        else:
            # Single ticker case
            prices = data[['Close']].rename(columns={'Close': symbols[0]})

        # Drop columns with all NaN
        prices = prices.dropna(axis=1, how='all')

        # Forward fill missing values (within reason)
        prices = prices.ffill(limit=5)

        logger.info(f"[H52] Downloaded {len(prices.columns)} symbols with data")
        if len(prices) > 0:
            logger.info(f"[H52]   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        return prices

    except Exception as e:
        logger.error(f"[H52] Failed to download data: {e}")
        raise


def download_spy_benchmark(start_date: str, end_date: str) -> pd.Series:
    """Download SPY for benchmark comparison."""
    lookback_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

    spy = yf.download('SPY', start=lookback_start, end=end_date, progress=False, auto_adjust=True)

    # Handle potential MultiIndex columns
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']

    # Ensure it's a Series with DatetimeIndex
    spy_close = pd.Series(spy_close.values, index=pd.DatetimeIndex(spy.index), name='SPY')
    return spy_close


# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def calculate_distance_to_high(
    prices: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    """
    Calculate distance from current price to 52-week high.

    distance = (high_52w - current) / high_52w

    Values:
    - 0.00 = at 52-week high
    - 0.05 = 5% below high
    - 0.10 = 10% below high
    """
    high_52w = prices.rolling(window=lookback, min_periods=lookback).max()
    distance = (high_52w - prices) / high_52w
    return distance


def select_top_n_stocks(
    distances: pd.Series,
    top_n: int = DEFAULT_TOP_N,
    max_distance: float = DEFAULT_MAX_DISTANCE_PCT
) -> List[str]:
    """
    Select top N stocks closest to their 52-week highs.

    Args:
        distances: Series of distance values for one date
        top_n: Number of stocks to select
        max_distance: Maximum allowed distance from high

    Returns:
        List of selected symbols
    """
    # Filter valid values
    valid = distances.dropna()

    # Optional: filter to stocks within max_distance
    if max_distance > 0:
        valid = valid[valid <= max_distance]

    # Sort ascending (closest to high first)
    sorted_dist = valid.sort_values(ascending=True)

    # Select top N
    return sorted_dist.head(top_n).index.tolist()


def is_first_trading_day_of_month(
    date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex
) -> bool:
    """Check if date is the first trading day of its month."""
    month_dates = trading_dates[
        (trading_dates.month == date.month) &
        (trading_dates.year == date.year)
    ]
    if len(month_dates) == 0:
        return False
    return date == month_dates[0]


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_n: int = DEFAULT_TOP_N,
    max_distance: float = DEFAULT_MAX_DISTANCE_PCT
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run monthly rebalancing backtest.

    Args:
        prices: DataFrame of adjusted close prices
        start_date: Backtest start date
        end_date: Backtest end date
        top_n: Number of positions
        max_distance: Maximum distance from 52-week high

    Returns:
        Tuple of (portfolio_value_series, trades_df)
    """
    logger.info(f"[H52] Running backtest: {start_date} to {end_date}")
    logger.info(f"[H52]   Top N: {top_n}, Max distance: {max_distance:.0%}")

    # Calculate distances
    distances = calculate_distance_to_high(prices)

    # Filter to backtest period
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    backtest_dates = prices.index[mask]

    # Initialize
    portfolio_value = 100000.0  # Starting capital
    cash = portfolio_value
    positions: Dict[str, float] = {}  # symbol -> shares

    portfolio_history = []
    trades_log = []

    # Monthly rebalance tracking
    last_rebalance_month = None

    for date in backtest_dates:
        current_month = (date.year, date.month)

        # Check if rebalance day (first trading day of month)
        should_rebalance = (
            is_first_trading_day_of_month(date, backtest_dates) and
            current_month != last_rebalance_month
        )

        if should_rebalance:
            # Get current distances
            date_distances = distances.loc[date].dropna()

            if len(date_distances) >= top_n:
                # Select new holdings
                new_holdings = select_top_n_stocks(
                    date_distances, top_n=top_n, max_distance=max_distance
                )

                # Calculate current portfolio value
                current_prices = prices.loc[date]
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

                # Buy new positions (equal weight)
                target_value_per_position = cash / top_n

                for sym in new_holdings:
                    if sym in current_prices.index:
                        price = current_prices[sym]
                        if pd.notna(price) and price > 0:
                            shares = int(target_value_per_position / price)
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
                                    'distance': date_distances.get(sym, 0)
                                })

                last_rebalance_month = current_month
                logger.debug(f"[H52] Rebalanced on {date.date()}: {len(positions)} positions")

        # Calculate daily portfolio value
        current_prices = prices.loc[date]
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

    # Create output DataFrames
    portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    return portfolio_df['portfolio_value'], trades_df


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> Dict[str, float]:
    """Calculate performance metrics."""
    # Total return
    total_return = (returns + 1).prod() - 1

    # Annualized return
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    annual_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate (daily)
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

    years = portfolio_values.index.year.unique()

    for year in years:
        # Filter to year
        year_mask = portfolio_values.index.year == year
        year_port = portfolio_values[year_mask]

        spy_mask = spy_values.index.year == year
        year_spy = spy_values[spy_mask]

        if len(year_port) < 20:  # Skip partial years
            continue

        # Calculate returns
        port_returns = year_port.pct_change().dropna()
        spy_returns = year_spy.pct_change().dropna()

        # Metrics
        port_metrics = calculate_metrics(port_returns)
        spy_metrics = calculate_metrics(spy_returns)

        results.append({
            'year': year,
            'strategy_return': port_metrics['total_return'],
            'spy_return': spy_metrics['total_return'],
            'alpha': port_metrics['total_return'] - spy_metrics['total_return'],
            'strategy_sharpe': port_metrics['sharpe_ratio'],
            'spy_sharpe': spy_metrics['sharpe_ratio'],
            'max_drawdown': port_metrics['max_drawdown'],
            'beat_spy': port_metrics['total_return'] > spy_metrics['total_return']
        })

    return pd.DataFrame(results)


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    start_year: int,
    end_year: int,
    top_n: int = DEFAULT_TOP_N,
    max_distance: float = DEFAULT_MAX_DISTANCE_PCT
) -> pd.DataFrame:
    """
    Run walk-forward validation with yearly periods.

    Each year is tested out-of-sample.
    """
    logger.info(f"[H52] Running walk-forward validation: {start_year}-{end_year}")

    # Run full period backtest first
    full_start = f"{start_year}-01-01"
    full_end = f"{end_year}-12-31"

    portfolio_values, trades = run_backtest(
        prices,
        start_date=full_start,
        end_date=full_end,
        top_n=top_n,
        max_distance=max_distance
    )

    if len(portfolio_values) == 0:
        logger.error("[H52] No portfolio values from backtest")
        return pd.DataFrame()

    # Remove timezone from index if present for comparison
    if portfolio_values.index.tz is not None:
        portfolio_values.index = portfolio_values.index.tz_localize(None)

    spy_copy = spy_prices.copy()
    if spy_copy.index.tz is not None:
        spy_copy.index = spy_copy.index.tz_localize(None)

    # Calculate yearly metrics from full backtest
    results = []

    for year in range(start_year, end_year + 1):
        # Filter portfolio values to year
        year_port = portfolio_values[portfolio_values.index.year == year]

        if len(year_port) < 20:
            logger.warning(f"[H52] Skipping {year}: insufficient data ({len(year_port)} days)")
            continue

        # Filter SPY to year
        year_spy = spy_copy[spy_copy.index.year == year]

        if len(year_spy) < 20:
            logger.warning(f"[H52] Skipping {year}: insufficient SPY data ({len(year_spy)} days)")
            continue

        # Calculate returns
        port_returns = year_port.pct_change().dropna()
        spy_returns = year_spy.pct_change().dropna()

        if len(port_returns) == 0 or len(spy_returns) == 0:
            logger.warning(f"[H52] Skipping {year}: no returns data")
            continue

        # Metrics
        port_metrics = calculate_metrics(port_returns)
        spy_metrics = calculate_metrics(spy_returns)

        results.append({
            'year': year,
            'strategy_return': port_metrics['total_return'],
            'spy_return': spy_metrics['total_return'],
            'alpha': port_metrics['total_return'] - spy_metrics['total_return'],
            'strategy_sharpe': port_metrics['sharpe_ratio'],
            'max_drawdown': port_metrics['max_drawdown'],
            'num_trades': len(trades[trades['date'].dt.year == year]) if len(trades) > 0 else 0,
            'beat_spy': port_metrics['total_return'] > spy_metrics['total_return']
        })

        logger.info(
            f"[H52] {year}: Strategy {port_metrics['total_return']:+.1%} | "
            f"SPY {spy_metrics['total_return']:+.1%} | "
            f"Alpha {port_metrics['total_return'] - spy_metrics['total_return']:+.1%}"
        )

    return pd.DataFrame(results)


# ============================================================================
# REPORTING
# ============================================================================

def print_results(
    yearly_results: pd.DataFrame,
    full_metrics: Dict[str, float]
):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("52-WEEK HIGH MONTHLY BREAKOUT BACKTEST RESULTS")
    print("=" * 70)

    print("\nYEARLY PERFORMANCE")
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

    # Summary stats
    avg_return = yearly_results['strategy_return'].mean()
    avg_spy = yearly_results['spy_return'].mean()
    avg_alpha = yearly_results['alpha'].mean()
    avg_sharpe = yearly_results['strategy_sharpe'].mean()
    win_years = yearly_results['beat_spy'].sum()
    total_years = len(yearly_results)

    print(f"\nSUMMARY STATISTICS")
    print("-" * 70)
    print(f"Average Annual Return:    {avg_return:+.1%}")
    print(f"Average SPY Return:       {avg_spy:+.1%}")
    print(f"Average Alpha:            {avg_alpha:+.1%}")
    print(f"Average Sharpe:           {avg_sharpe:.2f}")
    print(f"Beat SPY:                 {win_years}/{total_years} years ({100*win_years/total_years:.0f}%)")

    print(f"\nFULL PERIOD METRICS")
    print("-" * 70)
    print(f"Total Return:             {full_metrics['total_return']:+.1%}")
    print(f"Annual Return:            {full_metrics['annual_return']:+.1%}")
    print(f"Annual Volatility:        {full_metrics['annual_volatility']:.1%}")
    print(f"Sharpe Ratio:             {full_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:             {full_metrics['max_drawdown']:.1%}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="52-Week High Monthly Breakout Backtest"
    )
    parser.add_argument(
        '--start', type=int, default=2017,
        help='Start year (default: 2017)'
    )
    parser.add_argument(
        '--end', type=int, default=2024,
        help='End year (default: 2024)'
    )
    parser.add_argument(
        '--top-n', type=int, default=DEFAULT_TOP_N,
        help=f'Number of stocks to hold (default: {DEFAULT_TOP_N})'
    )
    parser.add_argument(
        '--max-distance', type=float, default=DEFAULT_MAX_DISTANCE_PCT,
        help=f'Max distance from 52-week high (default: {DEFAULT_MAX_DISTANCE_PCT})'
    )
    args = parser.parse_args()

    logger.info("[H52] Starting 52-Week High Monthly Breakout Backtest")
    logger.info(f"[H52]   Period: {args.start} - {args.end}")
    logger.info(f"[H52]   Top N: {args.top_n}")
    logger.info(f"[H52]   Max Distance: {args.max_distance:.0%}")

    # Load symbols
    symbols = load_sp500_symbols()

    # Define dates (need extra year for lookback)
    start_date = f"{args.start - 1}-01-01"
    end_date = f"{args.end}-12-31"

    # Download data
    prices = download_price_data(symbols, start_date, end_date)
    spy_prices = download_spy_benchmark(start_date, end_date)

    # Run full period backtest
    full_start = f"{args.start}-01-01"
    full_end = f"{args.end}-12-31"

    portfolio_values, trades = run_backtest(
        prices,
        start_date=full_start,
        end_date=full_end,
        top_n=args.top_n,
        max_distance=args.max_distance
    )

    # Calculate full period metrics
    full_returns = portfolio_values.pct_change().dropna()
    full_metrics = calculate_metrics(full_returns)

    # Remove timezone from indices for comparison
    if portfolio_values.index.tz is not None:
        portfolio_values.index = portfolio_values.index.tz_localize(None)

    spy_copy = spy_prices.copy()
    if spy_copy.index.tz is not None:
        spy_copy.index = spy_copy.index.tz_localize(None)

    # Calculate yearly metrics
    yearly_results = calculate_yearly_metrics(portfolio_values, spy_copy)

    # Print results
    print_results(yearly_results, full_metrics)

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"{timestamp}_HIGH52_BACKTEST_RESULTS.csv"
    yearly_results.to_csv(output_file, index=False)
    logger.info(f"[H52] Results saved to {output_file}")

    return yearly_results, full_metrics


if __name__ == "__main__":
    main()
