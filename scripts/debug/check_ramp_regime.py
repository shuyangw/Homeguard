"""Check RAMP regime and configuration."""
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

load_dotenv()

print("=" * 80)
print("RAMP REGIME ANALYSIS")
print("=" * 80)

# Get VIX level
print("\nFetching VIX...")
vix = yf.Ticker("^VIX")
vix_hist = vix.history(period="5d")
if not vix_hist.empty:
    latest_vix = vix_hist['Close'].iloc[-1]
    print(f"Current VIX: {latest_vix:.2f}")
else:
    print("Failed to fetch VIX")
    latest_vix = None

# Get SPY for regime detection
print("\nFetching SPY...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period="1y")

if not spy_hist.empty:
    # Calculate regime indicators
    spy_close = spy_hist['Close']

    # 50-day and 200-day moving averages
    ma50 = spy_close.rolling(50).mean().iloc[-1]
    ma200 = spy_close.rolling(200).mean().iloc[-1]
    current_price = spy_close.iloc[-1]

    # 52-week high/low
    high_52w = spy_close.tail(252).max()
    low_52w = spy_close.tail(252).min()

    # Drawdown from 52-week high
    drawdown = (current_price - high_52w) / high_52w * 100

    print(f"\nSPY Current: ${current_price:.2f}")
    print(f"SPY 50-day MA: ${ma50:.2f}")
    print(f"SPY 200-day MA: ${ma200:.2f}")
    print(f"SPY 52-week High: ${high_52w:.2f}")
    print(f"SPY 52-week Low: ${low_52w:.2f}")
    print(f"Drawdown from High: {drawdown:.1f}%")

    # Regime estimation based on RAMP logic
    print("\n" + "-" * 40)
    print("REGIME ESTIMATION:")
    print("-" * 40)

    above_ma50 = current_price > ma50
    above_ma200 = current_price > ma200
    ma50_above_ma200 = ma50 > ma200

    print(f"Price > 50-day MA: {above_ma50}")
    print(f"Price > 200-day MA: {above_ma200}")
    print(f"50-day MA > 200-day MA: {ma50_above_ma200}")

    if latest_vix:
        print(f"VIX Level: {latest_vix:.2f}")
        high_vix = latest_vix > 25
        print(f"High VIX (>25): {high_vix}")

# RAMP regime parameters from strategy
print("\n" + "=" * 80)
print("RAMP REGIME PARAMETERS (from strategy code)")
print("=" * 80)

regimes = {
    'STRONG_BULL': {'top_n': 5, 'rebalance_threshold': 0.15, 'description': 'Concentrated, high conviction'},
    'WEAK_BULL': {'top_n': 10, 'rebalance_threshold': 0.10, 'description': 'Moderate diversification'},
    'SIDEWAYS': {'top_n': 15, 'rebalance_threshold': 0.08, 'description': 'Wider diversification'},
    'UNPREDICTABLE': {'top_n': 20, 'rebalance_threshold': 0.05, 'description': 'Maximum diversification'},
    'BEAR': {'top_n': 5, 'rebalance_threshold': 0.20, 'description': 'Defensive, few positions'},
}

for regime, params in regimes.items():
    print(f"\n{regime}:")
    print(f"  top_n: {params['top_n']} positions")
    print(f"  rebalance_threshold: {params['rebalance_threshold']:.0%}")
    print(f"  Description: {params['description']}")

# Estimate likely regime
print("\n" + "=" * 80)
print("LIKELY CURRENT REGIME")
print("=" * 80)

if latest_vix and not spy_hist.empty:
    # Simple regime estimation
    if drawdown < -20:
        likely_regime = "BEAR"
    elif latest_vix > 30:
        likely_regime = "UNPREDICTABLE"
    elif latest_vix > 25:
        likely_regime = "SIDEWAYS"
    elif above_ma50 and above_ma200 and ma50_above_ma200:
        likely_regime = "STRONG_BULL"
    elif above_ma200:
        likely_regime = "WEAK_BULL"
    else:
        likely_regime = "SIDEWAYS"

    params = regimes[likely_regime]
    print(f"\nEstimated Regime: {likely_regime}")
    print(f"Expected top_n: {params['top_n']} positions")
    print(f"\nWith {params['top_n']} positions and 100% allocation:")
    print(f"  Position size: {100/params['top_n']:.1f}% each")
    print(f"  With $100k portfolio: ${100000/params['top_n']:,.0f} per position")

# Check portfolio allocation
print("\n" + "=" * 80)
print("CURRENT PORTFOLIO ANALYSIS")
print("=" * 80)

from alpaca.trading.client import TradingClient

api_key = os.getenv('ALPACA_PAPER_KEY_ID')
secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')
client = TradingClient(api_key, secret_key, paper=True)

account = client.get_account()
positions = client.get_all_positions()

portfolio_value = float(account.portfolio_value)
cash = float(account.cash)
position_value = sum(float(p.market_value) for p in positions)

print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
print(f"Cash: ${cash:,.2f} ({cash/portfolio_value*100:.1f}%)")
print(f"Positions: ${position_value:,.2f} ({position_value/portfolio_value*100:.1f}%)")
print(f"Number of positions: {len(positions)}")

if positions:
    print(f"\nPosition sizes:")
    for p in sorted(positions, key=lambda x: float(x.market_value), reverse=True):
        value = float(p.market_value)
        pct = value / portfolio_value * 100
        print(f"  {p.symbol}: ${value:,.2f} ({pct:.1f}%)")
