"""
Debug short selling equity curve calculation.

Investigate why Sharpe improves but total returns worsen.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def analyze_equity_curves():
    """Analyze equity curves for long-only vs long/short."""
    logger.blank()
    logger.separator()
    logger.header("EQUITY CURVE ANALYSIS - SHORT SELLING")
    logger.separator()
    logger.blank()

    # Simple MA Crossover on AAPL 2022
    strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

    # Long-only
    logger.info("Running long-only backtest...")
    engine_long = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=False
    )
    engine_long.risk_config = RiskConfig.moderate()

    portfolio_long = engine_long.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    # Long/short
    logger.blank()
    logger.info("Running long/short backtest...")
    engine_short = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005,
        allow_shorts=True
    )
    engine_short.risk_config = RiskConfig.moderate()

    portfolio_short = engine_short.run(
        strategy=strategy,
        symbols='AAPL',
        start_date='2022-01-01',
        end_date='2022-12-31'
    )

    logger.blank()
    logger.separator()
    logger.header("EQUITY CURVE COMPARISON")
    logger.separator()
    logger.blank()

    # Get equity curves
    equity_long = portfolio_long.equity_curve
    equity_short = portfolio_short.equity_curve

    # Calculate returns
    returns_long = equity_long.pct_change().dropna()
    returns_short = equity_short.pct_change().dropna()

    logger.metric("LONG-ONLY:")
    logger.info(f"  Start value: ${equity_long.iloc[0]:.2f}")
    logger.info(f"  End value: ${equity_long.iloc[-1]:.2f}")
    logger.info(f"  Min value: ${equity_long.min():.2f}")
    logger.info(f"  Max value: ${equity_long.max():.2f}")
    logger.info(f"  Mean return: {returns_long.mean():.6f}")
    logger.info(f"  Std return: {returns_long.std():.6f}")
    logger.blank()

    logger.metric("LONG/SHORT:")
    logger.info(f"  Start value: ${equity_short.iloc[0]:.2f}")
    logger.info(f"  End value: ${equity_short.iloc[-1]:.2f}")
    logger.info(f"  Min value: ${equity_short.min():.2f}")
    logger.info(f"  Max value: ${equity_short.max():.2f}")
    logger.info(f"  Mean return: {returns_short.mean():.6f}")
    logger.info(f"  Std return: {returns_short.std():.6f}")
    logger.blank()

    # Analyze trades
    logger.separator()
    logger.header("TRADE ANALYSIS")
    logger.separator()
    logger.blank()

    trades_long = [t for t in portfolio_long.trades if t.get('type') in ['exit']]
    trades_short_all = portfolio_short.trades

    # Count trade types
    long_entries = len([t for t in trades_short_all if t.get('type') == 'entry'])
    long_exits = len([t for t in trades_short_all if t.get('type') == 'exit'])
    short_entries = len([t for t in trades_short_all if t.get('type') == 'short_entry'])
    short_covers = len([t for t in trades_short_all if t.get('type') == 'cover_short'])

    logger.metric("LONG-ONLY TRADES:")
    logger.info(f"  Total exits: {len(trades_long)}")
    if trades_long:
        avg_pnl = sum(t.get('pnl', 0) for t in trades_long) / len(trades_long)
        logger.info(f"  Avg P&L: ${avg_pnl:.2f}")
    logger.blank()

    logger.metric("LONG/SHORT TRADES:")
    logger.info(f"  Long entries: {long_entries}")
    logger.info(f"  Long exits: {long_exits}")
    logger.info(f"  Short entries: {short_entries}")
    logger.info(f"  Short covers: {short_covers}")
    logger.blank()

    # Analyze short P&L
    if short_covers > 0:
        short_cover_trades = [t for t in trades_short_all if t.get('type') == 'cover_short']
        total_short_pnl = sum(t.get('pnl', 0) for t in short_cover_trades)
        avg_short_pnl = total_short_pnl / len(short_cover_trades)

        logger.success(f"SHORT POSITION P&L:")
        logger.info(f"  Total short P&L: ${total_short_pnl:.2f}")
        logger.info(f"  Avg short P&L: ${avg_short_pnl:.2f}")
        logger.info(f"  # of short trades: {len(short_cover_trades)}")
        logger.blank()

        # Show sample short trades
        logger.info("Sample short trades:")
        for i, t in enumerate(short_cover_trades[:5]):
            logger.info(f"  Trade {i+1}: Entry ${t.get('price', 0):.2f}, "
                       f"P&L ${t.get('pnl', 0):.2f} ({t.get('pnl_pct', 0):.2f}%)")

    logger.blank()
    logger.separator()

    # Return analysis summary
    return {
        'long_final': equity_long.iloc[-1],
        'short_final': equity_short.iloc[-1],
        'long_trades': len(trades_long),
        'short_trades': short_covers
    }


if __name__ == '__main__':
    results = analyze_equity_curves()

    logger.blank()
    logger.separator()
    logger.header("CONCLUSION")
    logger.separator()
    logger.blank()

    logger.info("The equity curve analysis shows:")
    logger.info("1. How equity changes over time for each mode")
    logger.info("2. Trade P&L breakdown (long vs short)")
    logger.info("3. Why Sharpe might improve even if returns worsen")
    logger.blank()
    logger.separator()
