"""
OMR and RAMP Backtest with V2 Per-Minute Portfolio Tracking.

Uses the new backtesting_v2 framework to generate minute-level
portfolio state and visualize equity curves with matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader


def generate_omr_signals(data: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Generate OMR-style overnight mean reversion signals.

    Entry: Strong intraday down move (z-score < -1.5)
    Exit: Next bar (overnight hold)
    """
    close = data["close"]
    open_price = data["open"]

    # Calculate intraday return
    intraday_ret = (close - open_price) / open_price

    # Calculate z-score
    intraday_mean = intraday_ret.rolling(lookback).mean()
    intraday_std = intraday_ret.rolling(lookback).std()
    z_score = (intraday_ret - intraday_mean) / intraday_std.replace(0, np.nan)

    # Entry on strong down day
    entries = (z_score < -1.5).fillna(False)

    # Exit next bar
    exits = entries.shift(1).fillna(False)

    return entries, exits


def generate_ramp_signals(data: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Generate RAMP-style regime-aware momentum signals.

    Entry: Positive momentum in favorable regime
    Exit: Momentum turns negative
    """
    close = data["close"]

    # Calculate volatility regime
    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_avg = vol.rolling(60).mean()
    low_vol_regime = vol < vol_avg

    # Momentum
    momentum = close.pct_change(lookback)

    # Entry: momentum positive in low vol regime
    entry_condition = low_vol_regime & (momentum > 0.02)
    entries = entry_condition & ~entry_condition.shift(1).fillna(False)

    # Exit: momentum turns negative
    exits = (momentum < -0.01) & (momentum.shift(1) >= -0.01)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def run_backtest_v2(data: pd.DataFrame, entries: pd.Series, exits: pd.Series,
                    risk_config: RiskConfig, allow_shorts: bool = False) -> object:
    """Run backtest using V2 framework."""
    return from_signals_v2(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        freq="1min",
        market_hours_only=True,
        risk_config=risk_config,
        price_data=data,
        allow_shorts=allow_shorts,
        use_numba=True,
    )


def plot_portfolio_analysis(portfolio, symbol: str, strategy_name: str, ax_equity, ax_exposure):
    """Plot equity curve and exposure from portfolio state."""
    state = portfolio.portfolio_state

    if state.empty:
        print(f"  Warning: No portfolio state for {symbol}")
        return

    # Resample to hourly for cleaner visualization (minute data is too dense)
    state_df = state.set_index("timestamp")
    hourly_state = state_df.resample("1H").last().dropna()

    # Plot equity curve
    ax_equity.plot(hourly_state.index, hourly_state["total_equity"],
                   label=f"{symbol}", linewidth=1)

    # Plot exposure
    ax_exposure.fill_between(hourly_state.index, 0, hourly_state["exposure_pct"] * 100,
                             alpha=0.3, label=f"{symbol}")


def main():
    print("=" * 80)
    print("OMR & RAMP BACKTEST WITH V2 PER-MINUTE TRACKING")
    print("=" * 80)

    loader = StreamingDataLoader(timeframe="1min")

    # Date range for backtest
    start_date = "2023-01-01"
    end_date = "2023-06-30"

    # =========================================================================
    # OMR BACKTEST (Leveraged ETFs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("OMR STRATEGY BACKTEST")
    print("=" * 80)

    omr_symbols = ["TQQQ", "SQQQ", "UPRO", "SPXU"]
    omr_portfolios = {}
    omr_risk = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        stop_loss_type="percentage",
    )

    for symbol in omr_symbols:
        print(f"\nProcessing {symbol}...")
        symbol_data = None

        try:
            for sym, df in loader.iter_symbols([symbol], start_date, end_date):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Insufficient data for {symbol}")
            continue

        print(f"  Loaded {len(symbol_data):,} bars")

        # Generate signals
        entries, exits = generate_omr_signals(symbol_data)

        # Run backtest
        portfolio = run_backtest_v2(symbol_data, entries, exits, omr_risk)
        omr_portfolios[symbol] = portfolio

        # Get stats
        stats = portfolio.stats()
        state = portfolio.portfolio_state

        print(f"  Return: {stats['Total Return [%]']:.2f}%")
        print(f"  Sharpe: {stats['Sharpe Ratio']:.3f}")
        print(f"  Max DD: {stats['Max Drawdown [%]']:.2f}%")
        print(f"  Trades: {stats['Total Trades']}")
        print(f"  State bars: {len(state):,}")

    # =========================================================================
    # RAMP BACKTEST (Tech Stocks)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RAMP STRATEGY BACKTEST")
    print("=" * 80)

    ramp_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    ramp_portfolios = {}
    ramp_risk = RiskConfig(
        enabled=True,
        position_size_pct=0.20,  # Higher allocation for momentum
    )

    for symbol in ramp_symbols:
        print(f"\nProcessing {symbol}...")
        symbol_data = None

        try:
            for sym, df in loader.iter_symbols([symbol], start_date, end_date):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Insufficient data for {symbol}")
            continue

        print(f"  Loaded {len(symbol_data):,} bars")

        # Generate signals
        entries, exits = generate_ramp_signals(symbol_data)

        # Run backtest
        portfolio = run_backtest_v2(symbol_data, entries, exits, ramp_risk)
        ramp_portfolios[symbol] = portfolio

        # Get stats
        stats = portfolio.stats()
        state = portfolio.portfolio_state

        print(f"  Return: {stats['Total Return [%]']:.2f}%")
        print(f"  Sharpe: {stats['Sharpe Ratio']:.3f}")
        print(f"  Max DD: {stats['Max Drawdown [%]']:.2f}%")
        print(f"  Trades: {stats['Total Trades']}")
        print(f"  State bars: {len(state):,}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Set up dark theme
    plt.style.use('dark_background')

    # Figure 1: OMR Strategy
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig1.suptitle('OMR Strategy - Per-Minute Portfolio Tracking (V2)', fontsize=14, fontweight='bold')

    ax_omr_equity = axes1[0]
    ax_omr_exposure = axes1[1]
    ax_omr_drawdown = axes1[2]

    for symbol, portfolio in omr_portfolios.items():
        state = portfolio.portfolio_state
        if state.empty:
            continue

        state_df = state.set_index("timestamp")
        # Resample to hourly for visualization
        hourly = state_df.resample("1H").last().dropna()

        # Equity curve
        ax_omr_equity.plot(hourly.index, hourly["total_equity"], label=symbol, linewidth=1)

        # Exposure
        ax_omr_exposure.plot(hourly.index, hourly["exposure_pct"] * 100, label=symbol, linewidth=0.8, alpha=0.7)

        # Drawdown
        equity = hourly["total_equity"]
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        ax_omr_drawdown.fill_between(hourly.index, drawdown, 0, alpha=0.3, label=symbol)

    ax_omr_equity.set_ylabel('Equity ($)', fontsize=10)
    ax_omr_equity.legend(loc='upper left', fontsize=8)
    ax_omr_equity.axhline(y=100000, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax_omr_equity.grid(True, alpha=0.2)

    ax_omr_exposure.set_ylabel('Exposure (%)', fontsize=10)
    ax_omr_exposure.set_ylim(-5, 105)
    ax_omr_exposure.legend(loc='upper left', fontsize=8)
    ax_omr_exposure.grid(True, alpha=0.2)

    ax_omr_drawdown.set_ylabel('Drawdown (%)', fontsize=10)
    ax_omr_drawdown.set_xlabel('Date', fontsize=10)
    ax_omr_drawdown.legend(loc='lower left', fontsize=8)
    ax_omr_drawdown.grid(True, alpha=0.2)

    ax_omr_drawdown.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_omr_drawdown.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()

    # Save OMR figure
    omr_output = "logs/backtesting/results/omr_v2_portfolio_analysis.png"
    os.makedirs(os.path.dirname(omr_output), exist_ok=True)
    fig1.savefig(omr_output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\nOMR chart saved: {omr_output}")

    # Figure 2: RAMP Strategy
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle('RAMP Strategy - Per-Minute Portfolio Tracking (V2)', fontsize=14, fontweight='bold')

    ax_ramp_equity = axes2[0]
    ax_ramp_exposure = axes2[1]
    ax_ramp_drawdown = axes2[2]

    for symbol, portfolio in ramp_portfolios.items():
        state = portfolio.portfolio_state
        if state.empty:
            continue

        state_df = state.set_index("timestamp")
        hourly = state_df.resample("1H").last().dropna()

        # Equity curve
        ax_ramp_equity.plot(hourly.index, hourly["total_equity"], label=symbol, linewidth=1)

        # Exposure
        ax_ramp_exposure.plot(hourly.index, hourly["exposure_pct"] * 100, label=symbol, linewidth=0.8, alpha=0.7)

        # Drawdown
        equity = hourly["total_equity"]
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        ax_ramp_drawdown.fill_between(hourly.index, drawdown, 0, alpha=0.3, label=symbol)

    ax_ramp_equity.set_ylabel('Equity ($)', fontsize=10)
    ax_ramp_equity.legend(loc='upper left', fontsize=8)
    ax_ramp_equity.axhline(y=100000, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax_ramp_equity.grid(True, alpha=0.2)

    ax_ramp_exposure.set_ylabel('Exposure (%)', fontsize=10)
    ax_ramp_exposure.set_ylim(-5, 105)
    ax_ramp_exposure.legend(loc='upper left', fontsize=8)
    ax_ramp_exposure.grid(True, alpha=0.2)

    ax_ramp_drawdown.set_ylabel('Drawdown (%)', fontsize=10)
    ax_ramp_drawdown.set_xlabel('Date', fontsize=10)
    ax_ramp_drawdown.legend(loc='lower left', fontsize=8)
    ax_ramp_drawdown.grid(True, alpha=0.2)

    ax_ramp_drawdown.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_ramp_drawdown.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()

    # Save RAMP figure
    ramp_output = "logs/backtesting/results/ramp_v2_portfolio_analysis.png"
    fig2.savefig(ramp_output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"RAMP chart saved: {ramp_output}")

    # Figure 3: Combined Strategy Comparison
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('OMR vs RAMP Strategy Comparison (V2 Per-Minute Tracking)', fontsize=14, fontweight='bold')

    # OMR combined equity
    ax_omr_combined = axes3[0, 0]
    for symbol, portfolio in omr_portfolios.items():
        state = portfolio.portfolio_state
        if not state.empty:
            state_df = state.set_index("timestamp")
            daily = state_df.resample("1D").last().dropna()
            ax_omr_combined.plot(daily.index, daily["total_equity"], label=symbol, linewidth=1.5)
    ax_omr_combined.set_title('OMR - Daily Equity', fontsize=11)
    ax_omr_combined.set_ylabel('Equity ($)', fontsize=10)
    ax_omr_combined.legend(fontsize=8)
    ax_omr_combined.axhline(y=100000, color='white', linestyle='--', alpha=0.3)
    ax_omr_combined.grid(True, alpha=0.2)

    # RAMP combined equity
    ax_ramp_combined = axes3[0, 1]
    for symbol, portfolio in ramp_portfolios.items():
        state = portfolio.portfolio_state
        if not state.empty:
            state_df = state.set_index("timestamp")
            daily = state_df.resample("1D").last().dropna()
            ax_ramp_combined.plot(daily.index, daily["total_equity"], label=symbol, linewidth=1.5)
    ax_ramp_combined.set_title('RAMP - Daily Equity', fontsize=11)
    ax_ramp_combined.set_ylabel('Equity ($)', fontsize=10)
    ax_ramp_combined.legend(fontsize=8)
    ax_ramp_combined.axhline(y=100000, color='white', linestyle='--', alpha=0.3)
    ax_ramp_combined.grid(True, alpha=0.2)

    # OMR stats summary
    ax_omr_stats = axes3[1, 0]
    ax_omr_stats.axis('off')
    omr_stats_text = "OMR Strategy Statistics\n" + "-" * 40 + "\n"
    for symbol, portfolio in omr_portfolios.items():
        stats = portfolio.stats()
        omr_stats_text += f"\n{symbol}:\n"
        omr_stats_text += f"  Return: {stats['Total Return [%]']:+.2f}%\n"
        omr_stats_text += f"  Sharpe: {stats['Sharpe Ratio']:.3f}\n"
        omr_stats_text += f"  Max DD: {stats['Max Drawdown [%]']:.2f}%\n"
        omr_stats_text += f"  Trades: {stats['Total Trades']}\n"
    ax_omr_stats.text(0.1, 0.95, omr_stats_text, transform=ax_omr_stats.transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))

    # RAMP stats summary
    ax_ramp_stats = axes3[1, 1]
    ax_ramp_stats.axis('off')
    ramp_stats_text = "RAMP Strategy Statistics\n" + "-" * 40 + "\n"
    for symbol, portfolio in ramp_portfolios.items():
        stats = portfolio.stats()
        ramp_stats_text += f"\n{symbol}:\n"
        ramp_stats_text += f"  Return: {stats['Total Return [%]']:+.2f}%\n"
        ramp_stats_text += f"  Sharpe: {stats['Sharpe Ratio']:.3f}\n"
        ramp_stats_text += f"  Max DD: {stats['Max Drawdown [%]']:.2f}%\n"
        ramp_stats_text += f"  Trades: {stats['Total Trades']}\n"
    ax_ramp_stats.text(0.1, 0.95, ramp_stats_text, transform=ax_ramp_stats.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))

    plt.tight_layout()

    # Save comparison figure
    comparison_output = "logs/backtesting/results/omr_ramp_v2_comparison.png"
    fig3.savefig(comparison_output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Comparison chart saved: {comparison_output}")

    # Figure 4: Per-Minute Detail (zoom into one day)
    fig4, axes4 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig4.suptitle('Per-Minute Portfolio State Detail (Sample Day)', fontsize=14, fontweight='bold')

    # Pick first OMR portfolio with data
    sample_portfolio = None
    sample_symbol = None
    for symbol, portfolio in omr_portfolios.items():
        if not portfolio.portfolio_state.empty:
            sample_portfolio = portfolio
            sample_symbol = symbol
            break

    if sample_portfolio is not None:
        state = sample_portfolio.portfolio_state
        state_df = state.set_index("timestamp")

        # Get a sample day with trading activity
        daily_trades = state_df.groupby(state_df.index.date)["position_shares"].apply(
            lambda x: (x != x.shift()).sum()
        )
        active_days = daily_trades[daily_trades > 5].index

        if len(active_days) > 0:
            sample_day = pd.Timestamp(active_days[len(active_days) // 2])
            day_data = state_df[state_df.index.date == sample_day.date()]

            # Plot minute-level equity
            ax_minute_equity = axes4[0]
            ax_minute_equity.plot(day_data.index, day_data["total_equity"],
                                  color='#00ff88', linewidth=1)
            ax_minute_equity.fill_between(day_data.index, day_data["total_equity"].min(),
                                          day_data["total_equity"], alpha=0.2, color='#00ff88')
            ax_minute_equity.set_ylabel('Equity ($)', fontsize=10)
            ax_minute_equity.set_title(f'{sample_symbol} - Minute-Level Equity ({sample_day.strftime("%Y-%m-%d")})',
                                       fontsize=11)
            ax_minute_equity.grid(True, alpha=0.2)

            # Plot minute-level position and exposure
            ax_minute_pos = axes4[1]
            ax_minute_pos.fill_between(day_data.index, 0, day_data["exposure_pct"] * 100,
                                       alpha=0.5, color='#ff6b6b', label='Exposure %')
            ax_minute_pos.set_ylabel('Exposure (%)', fontsize=10)
            ax_minute_pos.set_xlabel('Time', fontsize=10)
            ax_minute_pos.set_ylim(-5, 105)
            ax_minute_pos.legend(fontsize=8)
            ax_minute_pos.grid(True, alpha=0.2)

            ax_minute_pos.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_minute_pos.xaxis.set_major_locator(mdates.HourLocator())

    plt.tight_layout()

    # Save minute detail figure
    minute_output = "logs/backtesting/results/v2_minute_detail.png"
    fig4.savefig(minute_output, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Minute detail chart saved: {minute_output}")

    # Show all figures
    plt.show()

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {omr_output}")
    print(f"  2. {ramp_output}")
    print(f"  3. {comparison_output}")
    print(f"  4. {minute_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
