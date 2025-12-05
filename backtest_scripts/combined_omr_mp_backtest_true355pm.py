"""
Combined OMR + MP Strategy Backtest - TRUE 3:55 PM Execution.

This version uses TODAY's momentum at 3:55 PM (when close is ~99% known).
- Select stocks based on momentum[T] at 3:55 PM
- Execute at close[T]
- Capture return: close[T+1] / close[T] - 1

This is NOT lookahead because at 3:55 PM, today's close is nearly finalized.

Usage:
    python backtest_scripts/combined_omr_mp_backtest_true355pm.py
    python backtest_scripts/combined_omr_mp_backtest_true355pm.py --start 2020 --end 2024
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

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel


# ============================================================================
# Configuration
# ============================================================================

OMR_CONFIG = {
    'position_size': 0.15,
    'stop_loss': -0.02,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'vix_threshold': 35,
    'max_positions': 3,
    'skip_regimes': ['BEAR'],
    'symbols': [
        'FAZ', 'UDOW', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
        'DFEN', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL'
    ]
}

MP_CONFIG = {
    'position_size': 0.065,
    'top_n': 10,
    'reduced_exposure': 0.50,
    'vix_threshold': 25.0,
    'vix_spike_threshold': 0.20,
    'spy_dd_threshold': -0.05,
    'mom_vol_percentile': 0.90
}

DATA_DIR = PROJECT_ROOT / 'data' / 'leveraged_etfs'


def load_sp500_symbols() -> list:
    csv_path = PROJECT_ROOT / 'backtest_lists' / 'sp500-2025.csv'
    try:
        df = pd.read_csv(csv_path)
        return df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Failed to load S&P 500 symbols: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'AVGO', 'JPM', 'V', 'UNH', 'MA', 'JNJ', 'XOM', 'PG']


def load_omr_data() -> dict:
    logger.info("Loading OMR data from parquet files...")
    data = {}

    spy_path = DATA_DIR / 'SPY_1d.parquet'
    vix_path = DATA_DIR / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("SPY or VIX data not found in data/leveraged_etfs/")
        return None

    spy_df = pd.read_parquet(spy_path)
    vix_df = pd.read_parquet(vix_path)

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    spy_df.columns = [col.lower() for col in spy_df.columns]
    vix_df.columns = [col.lower() for col in vix_df.columns]

    spy_df.index = pd.to_datetime(spy_df.index)
    vix_df.index = pd.to_datetime(vix_df.index)

    data['SPY'] = spy_df
    data['^VIX'] = vix_df

    logger.info(f"  SPY: {len(spy_df)} bars ({spy_df.index[0].date()} to {spy_df.index[-1].date()})")
    logger.info(f"  VIX: {len(vix_df)} bars")

    loaded = 0
    for symbol in OMR_CONFIG['symbols']:
        file_path = DATA_DIR / f'{symbol}_1d.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [col.lower() for col in df.columns]
            df.index = pd.to_datetime(df.index)
            data[symbol] = df
            loaded += 1

    logger.info(f"  Loaded {loaded}/{len(OMR_CONFIG['symbols'])} OMR symbols")
    return data


def download_mp_data(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Download S&P 500 price data for MP strategy."""
    logger.info(f"Downloading MP data for {len(symbols)} symbols...")

    try:
        data = yf.download(symbols + ['SPY', '^VIX'], start=start, end=end,
                          progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data

        logger.info(f"  Downloaded {len(prices)} trading days")
        return prices
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return pd.DataFrame()


def run_omr_backtest(data: dict, test_start: str, test_end: str) -> pd.Series:
    """Run OMR backtest (unchanged)."""
    logger.info("Running OMR backtest with production Bayesian model...")

    spy_data = data['SPY']
    vix_data = data['^VIX']

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel(data_frequency='daily')

    model_path = PROJECT_ROOT / 'models' / 'bayesian_reversion_model.pkl'
    if model_path.exists():
        bayesian_model.load_model()
        logger.info(f"  Loaded production Bayesian model")
    else:
        logger.error(f"  Production model not found at {model_path}")
        return pd.Series(dtype=float)

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    test_dates = spy_data[(spy_data.index >= test_start_ts) &
                          (spy_data.index <= test_end_ts)].index

    daily_returns = []
    config = OMR_CONFIG

    for date in test_dates:
        regime, confidence = regime_detector.classify_regime(spy_data, vix_data, date)

        if regime in config['skip_regimes']:
            daily_returns.append({'date': date, 'return': 0.0})
            continue

        vix_value = float(vix_data[vix_data.index <= date]['close'].iloc[-1])
        if vix_value > config['vix_threshold']:
            daily_returns.append({'date': date, 'return': 0.0})
            continue

        day_trades = []
        for symbol in config['symbols']:
            if symbol not in data:
                continue

            symbol_data = data[symbol]
            if date not in symbol_data.index:
                continue

            today = symbol_data.loc[date]

            if isinstance(today, pd.Series):
                today_open = float(today['open'])
                today_close = float(today['close'])
            else:
                today_open = float(today['open'].iloc[0])
                today_close = float(today['close'].iloc[0])

            intraday_return = (today_close - today_open) / today_open

            if abs(intraday_return) < 0.005:
                continue

            prob_data = bayesian_model.get_reversion_probability(
                symbol, regime, intraday_return
            )

            if prob_data is None:
                continue

            if (prob_data['probability'] < config['min_win_rate'] or
                prob_data['expected_return'] < config['min_expected_return'] or
                prob_data['sample_size'] < config['min_sample_size']):
                continue

            next_idx = symbol_data.index.get_loc(date) + 1
            if next_idx >= len(symbol_data):
                continue

            next_open = float(symbol_data.iloc[next_idx]['open'])
            overnight_return = (next_open - today_close) / today_close

            if overnight_return < config['stop_loss']:
                overnight_return = config['stop_loss']

            day_trades.append({
                'symbol': symbol,
                'overnight_return': overnight_return,
                'probability': prob_data['probability']
            })

        if day_trades:
            day_trades.sort(key=lambda x: x['probability'], reverse=True)
            selected = day_trades[:config['max_positions']]

            portfolio_return = sum(
                t['overnight_return'] * config['position_size']
                for t in selected
            )
            daily_returns.append({'date': date, 'return': portfolio_return})
        else:
            daily_returns.append({'date': date, 'return': 0.0})

    return pd.DataFrame(daily_returns).set_index('date')['return']


def run_mp_backtest_true_355pm(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    test_start: str,
    test_end: str
) -> pd.Series:
    """
    Run MP backtest with TRUE 3:55 PM execution.

    Timeline (NO lookahead - this is realistic):
      Day T at 3:55 PM: Use momentum[T] (close nearly final)
                        Buy at close[T]
      Day T+1 at 3:55 PM: Sell at close[T+1]
                          Return = close[T+1] / close[T] - 1

    This is NOT lookahead because at 3:55 PM, today's close is ~99% known.
    """
    logger.info("Running MP backtest with TRUE 3:55 PM execution...")
    logger.info("  Using TODAY's momentum (known at 3:55 PM)")
    logger.info("  Return: close[T+1] / close[T] - 1")

    symbols = [c for c in prices.columns if c not in ['SPY', '^VIX']]

    # Compute momentum using TODAY's close (known at 3:55 PM)
    logger.info("  Computing 1m-1w momentum (today's data)...")
    returns_1m = prices.pct_change(21, fill_method=None)
    returns_1w = prices.pct_change(5, fill_method=None)
    momentum_all = returns_1m - returns_1w

    # Risk signals using today's data (known at 3:55 PM)
    logger.info("  Computing risk signals (today's data)...")
    high_vix = vix_prices > MP_CONFIG['vix_threshold']
    vix_5d_change = vix_prices.pct_change(5)
    vix_spike = vix_5d_change > MP_CONFIG['vix_spike_threshold']

    spy_rolling_max = spy_prices.expanding().max()
    spy_drawdown = (spy_prices - spy_rolling_max) / spy_rolling_max
    spy_dd_trigger = spy_drawdown < MP_CONFIG['spy_dd_threshold']

    mom_mean = momentum_all.mean(axis=1)
    mom_vol = mom_mean.rolling(252).std()
    mom_vol_90pct = mom_vol.expanding().quantile(MP_CONFIG['mom_vol_percentile'])
    high_mom_vol = mom_vol > mom_vol_90pct

    # Daily returns (close to close)
    daily_returns = prices.pct_change()

    daily_returns_list = []

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    try:
        start_idx = prices.index.get_loc(test_start_ts)
    except KeyError:
        start_idx = prices.index.searchsorted(test_start_ts)
    start_idx = max(30, start_idx)

    logger.info(f"  Processing {len(prices) - start_idx} trading days...")

    for i in range(start_idx, len(prices) - 1):  # -1 because we need next day's return
        date = prices.index[i]
        next_date = prices.index[i + 1]

        if date < test_start_ts or date > test_end_ts:
            continue

        # Use TODAY's momentum (known at 3:55 PM)
        scores = momentum_all.loc[date].dropna()

        if len(scores) < MP_CONFIG['top_n']:
            daily_returns_list.append({'date': date, 'return': 0.0})
            continue

        # Check risk signals using TODAY's data (known at 3:55 PM)
        reduce_exposure = (
            high_vix.get(date, False) or
            vix_spike.get(date, False) or
            spy_dd_trigger.get(date, False) or
            high_mom_vol.get(date, False)
        )

        exposure = MP_CONFIG['reduced_exposure'] if reduce_exposure else 1.0

        # Select top N based on today's momentum
        top_stocks = scores.nlargest(MP_CONFIG['top_n']).index.tolist()

        # Return is NEXT day's close-to-close return
        # Buy at close[T], sell at close[T+1]
        day_rets = daily_returns.loc[next_date, top_stocks].dropna()

        if len(day_rets) > 0:
            avg_return = day_rets.mean()
            portfolio_return = avg_return * MP_CONFIG['position_size'] * len(day_rets) * exposure
            daily_returns_list.append({'date': date, 'return': portfolio_return})
        else:
            daily_returns_list.append({'date': date, 'return': 0.0})

    return pd.DataFrame(daily_returns_list).set_index('date')['return']


def calculate_metrics(returns: pd.Series, name: str) -> dict:
    if len(returns) == 0:
        return {}

    returns = returns.dropna()

    cum_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

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
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    logger.info("=" * 70)
    logger.info("COMBINED OMR + MP BACKTEST - TRUE 3:55 PM EXECUTION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("MP Strategy Logic:")
    logger.info("  At 3:55 PM on Day T:")
    logger.info("    - Use momentum[T] (today's close ~99% known)")
    logger.info("    - Select top 10 stocks")
    logger.info("    - Buy at close[T]")
    logger.info("  At 3:55 PM on Day T+1:")
    logger.info("    - Sell at close[T+1]")
    logger.info("    - Return = close[T+1] / close[T] - 1")
    logger.info("")
    logger.info(f"Period: {start} to {end}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  OMR: {OMR_CONFIG['position_size']:.0%} per position, max {OMR_CONFIG['max_positions']} positions")
    logger.info(f"  MP:  {MP_CONFIG['position_size']:.1%} per position, top {MP_CONFIG['top_n']} stocks")
    logger.info("")

    # Load OMR data
    omr_data = load_omr_data()
    if omr_data is None:
        logger.error("Failed to load OMR data")
        return None

    # Load MP data
    sp500_symbols = load_sp500_symbols()
    mp_prices = download_mp_data(sp500_symbols, start, end)

    if mp_prices.empty:
        logger.error("Failed to download MP data")
        return None

    spy_prices = mp_prices['SPY'] if 'SPY' in mp_prices.columns else None
    vix_prices = mp_prices['^VIX'] if '^VIX' in mp_prices.columns else None

    if spy_prices is None or vix_prices is None:
        logger.error("Missing SPY or VIX data")
        return None

    mp_symbols = [s for s in sp500_symbols if s in mp_prices.columns]
    mp_prices_filtered = mp_prices[mp_symbols]
    logger.info(f"MP universe: {len(mp_symbols)} S&P 500 stocks")

    # Run backtests
    logger.info("")
    logger.info("-" * 70)

    omr_returns = run_omr_backtest(omr_data, start, end)
    logger.info(f"OMR: {len(omr_returns)} days of returns")

    mp_returns = run_mp_backtest_true_355pm(
        mp_prices_filtered, spy_prices, vix_prices, start, end
    )
    logger.info(f"MP (True 3:55 PM): {len(mp_returns)} days of returns")

    # Align returns
    common_dates = omr_returns.index.intersection(mp_returns.index)
    omr_returns = omr_returns.loc[common_dates]
    mp_returns = mp_returns.loc[common_dates]

    combined_returns = omr_returns + mp_returns

    spy_returns = spy_prices.pct_change().loc[common_dates]

    logger.info("")
    logger.info(f"Backtest period: {common_dates.min().date()} to {common_dates.max().date()}")
    logger.info(f"Trading days: {len(common_dates)}")

    # Calculate metrics
    omr_metrics = calculate_metrics(omr_returns, "OMR (Bayesian)")
    mp_metrics = calculate_metrics(mp_returns, "MP (True 3:55 PM)")
    combined_metrics = calculate_metrics(combined_returns, "Combined")
    spy_metrics = calculate_metrics(spy_returns, "SPY (Benchmark)")

    # Print results
    logger.info("")
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
    parser = argparse.ArgumentParser(description="Combined OMR + MP Backtest (True 3:55 PM)")
    parser.add_argument('--start', type=int, default=2017, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')

    args = parser.parse_args()

    run_backtest(start_year=args.start, end_year=args.end)
