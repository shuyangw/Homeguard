"""
Combined OMR + MP Strategy Backtest.

Simulates running both strategies simultaneously with proper position sizing:
- OMR: 15% per position, max 3 positions (45% exposure), leveraged ETFs
- MP: 6.5% per position, max 10 positions (65% exposure), S&P 500

Combined max exposure: 110% (uses margin when both fully allocated)

Usage:
    python backtest_scripts/combined_omr_mp_backtest.py
    python backtest_scripts/combined_omr_mp_backtest.py --start 2020 --end 2024
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

import yfinance as yf

from src.utils.logger import logger


# ============================================================================
# Configuration
# ============================================================================

# OMR Configuration - Uses OPTIMAL_OMR from ETFUniverse (15 symbols ranked by performance)
OMR_SYMBOLS = [
    # Tier 1: Proven top performers (Sharpe > 0.4)
    'SDOW',   # Dow 3x Bear - Sharpe 0.88 (best)
    'SOXS',   # Semiconductor 3x Bear - Sharpe 0.45

    # Tier 2: High-liquidity core (broad market)
    'SQQQ',   # Nasdaq 3x Bear - Most liquid inverse
    'SPXU',   # S&P 3x Bear - Broad market
    'TQQQ',   # Nasdaq 3x Bull - Most liquid bull
    'UPRO',   # S&P 3x Bull - Broad market

    # Tier 3: Sector leaders (tech/semis)
    'TECS',   # Tech 3x Bear - Sharpe 0.04 (proven)
    'TECL',   # Tech 3x Bull
    'SOXL',   # Semiconductor 3x Bull

    # Tier 4: Financials & small cap
    'FAZ',    # Financial 3x Bear
    'FAS',    # Financial 3x Bull
    'TZA',    # Small Cap 3x Bear
    'TNA',    # Small Cap 3x Bull

    # Tier 5: Diversification
    'TMF',    # Treasury 3x Bull
    'ERX',    # Energy 3x Bull
]
OMR_POSITION_SIZE = 0.15  # 15% per position
OMR_MAX_POSITIONS = 3
OMR_MAX_EXPOSURE = 0.45  # 45% max

# MP Configuration
MP_POSITION_SIZE = 0.065  # 6.5% per position
MP_MAX_POSITIONS = 10
MP_MAX_EXPOSURE = 0.65  # 65% max
MP_TOP_N = 10

# Risk Management
VIX_THRESHOLD = 25.0  # Reduce MP exposure when VIX > 25
REDUCED_EXPOSURE = 0.5  # 50% exposure when risk high


def load_sp500_symbols() -> list:
    """Load S&P 500 symbols from CSV."""
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    try:
        df = pd.read_csv(csv_path)
        return df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Failed to load S&P 500 symbols: {e}")
        # Return a subset for testing
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'AVGO', 'JPM', 'V', 'UNH', 'MA', 'JNJ', 'XOM', 'PG']


def download_data(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Download price data for symbols."""
    logger.info(f"Downloading data for {len(symbols)} symbols from {start} to {end}...")

    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)

        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data

        logger.info(f"Downloaded {len(prices)} days of data")
        return prices
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return pd.DataFrame()


def calculate_omr_returns(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    position_size: float = 0.15,
    max_positions: int = 3
) -> pd.Series:
    """
    Calculate OMR strategy returns.

    Simplified overnight mean reversion:
    - Buy at close (when price drops from prior day)
    - Sell at next day's open
    - Use regime filter (SPY above 200 SMA, VIX < 35)
    """
    # Calculate overnight returns (close to next open, approximated by close-to-close)
    overnight_returns = prices.pct_change()

    # Calculate prior day returns (for mean reversion signal)
    prior_returns = prices.pct_change().shift(1)

    # Regime filter
    spy_sma_200 = spy_prices.rolling(200).mean()
    bullish_regime = spy_prices > spy_sma_200
    low_vix = vix_prices < 35

    # Align indexes
    common_idx = prices.index.intersection(spy_sma_200.dropna().index)
    common_idx = common_idx.intersection(vix_prices.dropna().index)

    strategy_returns = []

    for i, date in enumerate(common_idx[1:], 1):
        prev_date = common_idx[i-1]

        # Check regime
        if not (bullish_regime.get(prev_date, False) and low_vix.get(prev_date, True)):
            strategy_returns.append({'date': date, 'return': 0.0})
            continue

        # Get prior day returns for mean reversion signal
        prior_rets = prior_returns.loc[prev_date].dropna()

        # Select symbols that dropped (mean reversion)
        dropped = prior_rets[prior_rets < -0.01].nlargest(max_positions)

        if len(dropped) == 0:
            strategy_returns.append({'date': date, 'return': 0.0})
            continue

        # Calculate return (overnight return * position size)
        selected_symbols = dropped.index.tolist()
        day_returns = overnight_returns.loc[date, selected_symbols].dropna()

        if len(day_returns) > 0:
            # Equal weight across selected positions
            avg_return = day_returns.mean()
            # Scale by number of positions and position size
            portfolio_return = avg_return * position_size * len(day_returns)
            strategy_returns.append({'date': date, 'return': portfolio_return})
        else:
            strategy_returns.append({'date': date, 'return': 0.0})

    return pd.DataFrame(strategy_returns).set_index('date')['return']


def calculate_mp_returns(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    top_n: int = 10,
    position_size: float = 0.065,
    vix_threshold: float = 25.0,
    reduced_exposure: float = 0.5
) -> pd.Series:
    """
    Calculate Momentum Protection strategy returns.

    - Long top N stocks by 12-1 month momentum
    - Reduce exposure to 50% when VIX > 25
    """
    # Calculate 12-1 month momentum
    returns_12m = prices.pct_change(252)
    returns_1m = prices.pct_change(21)
    momentum = returns_12m - returns_1m

    daily_returns = prices.pct_change()

    # VIX-based exposure
    high_vix = vix_prices > vix_threshold

    strategy_returns = []

    for i in range(253, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i-1]

        # Get momentum scores
        scores = momentum.iloc[i-1].dropna()
        if len(scores) < top_n:
            strategy_returns.append({'date': date, 'return': 0.0})
            continue

        # Select top N
        top_stocks = scores.nlargest(top_n).index.tolist()

        # Calculate return
        day_returns = daily_returns.loc[date, top_stocks].dropna()

        if len(day_returns) > 0:
            avg_return = day_returns.mean()

            # Adjust for exposure
            exposure = reduced_exposure if high_vix.get(prev_date, False) else 1.0

            # Scale by position size and number of positions
            portfolio_return = avg_return * position_size * len(day_returns) * exposure
            strategy_returns.append({'date': date, 'return': portfolio_return})
        else:
            strategy_returns.append({'date': date, 'return': 0.0})

    return pd.DataFrame(strategy_returns).set_index('date')['return']


def calculate_metrics(returns: pd.Series, name: str) -> dict:
    """Calculate performance metrics."""
    if len(returns) == 0:
        return {}

    # Remove NaN
    returns = returns.dropna()

    # Cumulative return
    cum_return = (1 + returns).prod() - 1

    # Annualized return
    n_years = len(returns) / 252
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        'name': name,
        'cumulative_return': cum_return,
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_days': len(returns)
    }


def run_backtest(start_year: int = 2017, end_year: int = 2024):
    """Run combined backtest."""

    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    logger.info("=" * 70)
    logger.info("COMBINED OMR + MP STRATEGY BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Period: {start} to {end}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  OMR: {OMR_POSITION_SIZE:.0%} per position, max {OMR_MAX_POSITIONS} positions")
    logger.info(f"  MP:  {MP_POSITION_SIZE:.1%} per position, max {MP_MAX_POSITIONS} positions")
    logger.info(f"  Combined max exposure: {OMR_MAX_EXPOSURE + MP_MAX_EXPOSURE:.0%}")
    logger.info("")

    # Load S&P 500 symbols
    sp500_symbols = load_sp500_symbols()
    logger.info(f"MP Universe: {len(sp500_symbols)} S&P 500 stocks")
    logger.info(f"OMR Universe: {len(OMR_SYMBOLS)} leveraged ETFs")

    # Download all data
    all_symbols = list(set(sp500_symbols + OMR_SYMBOLS + ['SPY', '^VIX']))
    prices = download_data(all_symbols, start, end)

    if prices.empty:
        logger.error("No data downloaded")
        return

    # Extract SPY and VIX
    spy_prices = prices['SPY'] if 'SPY' in prices.columns else None
    vix_prices = prices['^VIX'] if '^VIX' in prices.columns else None

    if spy_prices is None or vix_prices is None:
        logger.error("Missing SPY or VIX data")
        return

    # Get OMR prices
    omr_symbols_available = [s for s in OMR_SYMBOLS if s in prices.columns]
    omr_prices = prices[omr_symbols_available]
    logger.info(f"OMR symbols available: {len(omr_symbols_available)}")

    # Get MP prices (exclude leveraged ETFs)
    mp_symbols = [s for s in sp500_symbols if s in prices.columns and s not in OMR_SYMBOLS]
    mp_prices = prices[mp_symbols]
    logger.info(f"MP symbols available: {len(mp_symbols)}")

    # Calculate returns
    logger.info("")
    logger.info("Calculating strategy returns...")

    omr_returns = calculate_omr_returns(
        omr_prices, spy_prices, vix_prices,
        position_size=OMR_POSITION_SIZE,
        max_positions=OMR_MAX_POSITIONS
    )

    mp_returns = calculate_mp_returns(
        mp_prices, spy_prices, vix_prices,
        top_n=MP_TOP_N,
        position_size=MP_POSITION_SIZE,
        vix_threshold=VIX_THRESHOLD,
        reduced_exposure=REDUCED_EXPOSURE
    )

    # Align returns
    common_dates = omr_returns.index.intersection(mp_returns.index)
    omr_returns = omr_returns.loc[common_dates]
    mp_returns = mp_returns.loc[common_dates]

    # Combined returns
    combined_returns = omr_returns + mp_returns

    # SPY benchmark
    spy_returns = spy_prices.pct_change().loc[common_dates]

    logger.info(f"Backtest period: {common_dates.min().date()} to {common_dates.max().date()}")
    logger.info(f"Trading days: {len(common_dates)}")
    logger.info("")

    # Calculate metrics
    omr_metrics = calculate_metrics(omr_returns, "OMR")
    mp_metrics = calculate_metrics(mp_returns, "MP")
    combined_metrics = calculate_metrics(combined_returns, "Combined")
    spy_metrics = calculate_metrics(spy_returns, "SPY (Benchmark)")

    # Print results
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info("")

    for metrics in [omr_metrics, mp_metrics, combined_metrics, spy_metrics]:
        logger.info(f"{metrics['name']}:")
        logger.info(f"  Cumulative Return: {metrics['cumulative_return']:+.1%}")
        logger.info(f"  Annual Return:     {metrics['annual_return']:+.1%}")
        logger.info(f"  Annual Volatility: {metrics['annual_volatility']:.1%}")
        logger.info(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown:      {metrics['max_drawdown']:.1%}")
        logger.info("")

    # Yearly breakdown
    logger.info("=" * 70)
    logger.info("YEARLY BREAKDOWN")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'Year':<6} {'OMR':>10} {'MP':>10} {'Combined':>10} {'SPY':>10}")
    logger.info("-" * 50)

    for year in range(start_year, end_year + 1):
        year_mask = combined_returns.index.year == year
        if year_mask.sum() == 0:
            continue

        omr_yr = (1 + omr_returns[year_mask]).prod() - 1
        mp_yr = (1 + mp_returns[year_mask]).prod() - 1
        comb_yr = (1 + combined_returns[year_mask]).prod() - 1
        spy_yr = (1 + spy_returns[year_mask]).prod() - 1

        logger.info(f"{year:<6} {omr_yr:>+10.1%} {mp_yr:>+10.1%} {comb_yr:>+10.1%} {spy_yr:>+10.1%}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 70)

    return {
        'omr_returns': omr_returns,
        'mp_returns': mp_returns,
        'combined_returns': combined_returns,
        'spy_returns': spy_returns,
        'metrics': {
            'omr': omr_metrics,
            'mp': mp_metrics,
            'combined': combined_metrics,
            'spy': spy_metrics
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined OMR + MP Backtest")
    parser.add_argument('--start', type=int, default=2017, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')

    args = parser.parse_args()

    run_backtest(start_year=args.start, end_year=args.end)
