"""
Cross-Sectional Crypto Momentum (CSCM) Backtest Runner.

Runs multi-symbol backtest for the CSCM strategy using local crypto data.

Usage:
    python scripts/backtest/run_cscm_backtest.py
    python scripts/backtest/run_cscm_backtest.py --start 2022 --end 2024
    python scripts/backtest/run_cscm_backtest.py --top-n 5 --momentum-period 21
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.providers import create_crypto_data_provider
from src.strategies.advanced.cscm_indicators import CSCMIndicators
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='CSCM Backtest Runner')
    parser.add_argument('--start', type=int, default=2021, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')
    parser.add_argument('--top-n', type=int, default=3, help='Number of top coins to hold')
    parser.add_argument('--momentum-period', type=int, default=14, help='Momentum lookback days')
    parser.add_argument('--btc-sma-period', type=int, default=20, help='BTC SMA period for regime')
    parser.add_argument('--go-to-cash', action='store_true', default=True, help='Go to cash in bear regime')
    parser.add_argument('--no-go-to-cash', action='store_false', dest='go_to_cash', help='Stay invested in bear regime')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--fee-rate', type=float, default=0.001, help='Fee rate per trade')
    return parser.parse_args()


def run_backtest(
    start_year: int,
    end_year: int,
    top_n: int,
    momentum_period: int,
    btc_sma_period: int,
    go_to_cash_in_bear: bool,
    initial_capital: float,
    fee_rate: float
):
    """Run CSCM backtest."""
    logger.info("=" * 60)
    logger.info("CSCM BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Period: {start_year} - {end_year}")
    logger.info(f"Top N: {top_n}")
    logger.info(f"Momentum Period: {momentum_period} days")
    logger.info(f"BTC SMA Period: {btc_sma_period}")
    logger.info(f"Go to Cash in Bear: {go_to_cash_in_bear}")
    logger.info(f"Initial Capital: ${initial_capital:,.0f}")
    logger.info(f"Fee Rate: {fee_rate:.2%}")
    logger.info("")

    # Load data
    logger.info("Loading crypto data...")
    provider = create_crypto_data_provider()
    available = provider.get_available_symbols('1D')
    logger.info(f"Available symbols: {len(available)}")

    # Define universe (excluding stablecoins)
    universe = [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD',
        'DOGE/USD', 'DOT/USD', 'LTC/USD', 'BCH/USD', 'UNI/USD',
        'AAVE/USD', 'MKR/USD', 'XRP/USD', 'SUSHI/USD'
    ]
    # Filter to available
    universe = [s for s in universe if s in available]
    logger.info(f"Universe: {len(universe)} symbols")

    # Load data for all symbols
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    data_dict = provider.get_historical_bars_batch(universe, start_date, end_date, '1D')
    logger.info(f"Loaded data for {len(data_dict)} symbols")

    if 'BTC/USD' not in data_dict:
        logger.error("BTC data required for regime filter!")
        return None

    btc_data = data_dict['BTC/USD']

    # Generate signals
    logger.info("Generating CSCM signals...")
    selections, weights, regime = CSCMIndicators.generate_signals(
        data_dict=data_dict,
        btc_data=btc_data,
        momentum_period=momentum_period,
        btc_sma_period=btc_sma_period,
        top_n=top_n,
        rebalance_day='sunday',
        go_to_cash_in_bear=go_to_cash_in_bear
    )

    if selections.empty:
        logger.error("No signals generated!")
        return None

    logger.info(f"Generated signals for {len(selections)} days")

    # Calculate returns
    logger.info("Calculating returns...")
    returns_dict = {}
    for symbol, df in data_dict.items():
        if 'close' in df.columns:
            returns_dict[symbol] = df['close'].pct_change()
        elif 'Close' in df.columns:
            returns_dict[symbol] = df['Close'].pct_change()

    returns_df = pd.DataFrame(returns_dict)

    # Align all dataframes to same index
    common_index = selections.index.intersection(returns_df.index)
    selections = selections.loc[common_index]
    weights = weights.loc[common_index]
    returns_df = returns_df.loc[common_index]
    regime = regime.loc[common_index]

    # Calculate portfolio returns
    # For each day, sum of (weight * return) for selected symbols
    portfolio_returns = (weights * returns_df).sum(axis=1)

    # Apply fees on rebalance days
    # Detect when positions change
    position_changes = weights.diff().abs().sum(axis=1)
    rebalance_mask = position_changes > 0.01
    fee_costs = rebalance_mask.astype(float) * fee_rate * 2  # Round-trip

    # Net returns
    net_returns = portfolio_returns - fee_costs

    # Calculate cumulative returns
    cumulative = (1 + net_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # Calculate metrics
    days = len(net_returns)
    years = days / 365

    # Annualized return
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe ratio
    sharpe = CSCMIndicators.calculate_sharpe_ratio(net_returns, periods_per_year=365)

    # Max drawdown
    max_dd = CSCMIndicators.calculate_max_drawdown(cumulative)

    # Win rate
    wins = (net_returns > 0).sum()
    losses = (net_returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Regime statistics
    bullish_days = (regime == 'bullish').sum()
    bearish_days = (regime == 'bearish').sum()

    # Rebalance count
    rebalance_count = rebalance_mask.sum()

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Period: {common_index.min().date()} to {common_index.max().date()}")
    logger.info(f"Trading Days: {days}")
    logger.info(f"Years: {years:.2f}")
    logger.info("")
    logger.info("RETURNS")
    logger.info(f"  Total Return: {total_return:.2%}")
    logger.info(f"  Annualized Return: {ann_return:.2%}")
    logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
    logger.info(f"  Max Drawdown: {max_dd:.2%}")
    logger.info("")
    logger.info("TRADING")
    logger.info(f"  Rebalances: {rebalance_count}")
    logger.info(f"  Win Rate: {win_rate:.2%}")
    logger.info(f"  Avg Daily Return: {net_returns.mean():.4%}")
    logger.info(f"  Daily Volatility: {net_returns.std():.4%}")
    logger.info("")
    logger.info("REGIME")
    logger.info(f"  Bullish Days: {bullish_days} ({bullish_days/days:.1%})")
    logger.info(f"  Bearish Days: {bearish_days} ({bearish_days/days:.1%})")
    logger.info("")

    # Sharpe target check
    if sharpe >= 1.5:
        logger.info("[+] SUCCESS: Sharpe ratio >= 1.5 target achieved!")
    else:
        logger.info(f"[-] Sharpe ratio {sharpe:.3f} below 1.5 target")

    # Return results dict
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'rebalance_count': rebalance_count,
        'bullish_days': bullish_days,
        'bearish_days': bearish_days,
        'days': days,
        'years': years,
        'cumulative': cumulative,
        'returns': net_returns,
    }


def main():
    args = parse_args()

    results = run_backtest(
        start_year=args.start,
        end_year=args.end,
        top_n=args.top_n,
        momentum_period=args.momentum_period,
        btc_sma_period=args.btc_sma_period,
        go_to_cash_in_bear=args.go_to_cash,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate
    )

    if results is None:
        sys.exit(1)

    # Exit with error if Sharpe < 1.5
    if results['sharpe_ratio'] < 1.5:
        logger.warning(f"Sharpe ratio {results['sharpe_ratio']:.3f} below 1.5 target")

    return results


if __name__ == '__main__':
    main()
