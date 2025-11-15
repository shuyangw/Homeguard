#!/usr/bin/env python3
"""
Simple diagnostic script to check why OMR strategy generated 0 signals on Nov 14, 2025.
Checks VIX levels and market regime without importing strategy classes.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def fetch_spy_data(end_date='2025-11-14', days_back=252):
    """Fetch SPY data for regime detection."""
    end = pd.Timestamp(end_date)
    start = end - timedelta(days=days_back)

    print(f"Fetching SPY data from {start.date()} to {end.date()}...")
    spy = yf.download('SPY', start=start.strftime('%Y-%m-%d'),
                      end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
                      progress=False)
    print(f"[OK] Fetched {len(spy)} days of SPY data\n")
    return spy

def fetch_vix_data(end_date='2025-11-14', days_back=252):
    """Fetch VIX data."""
    end = pd.Timestamp(end_date)
    start = end - timedelta(days=days_back)

    print(f"Fetching VIX data from {start.date()} to {end.date()}...")
    vix = yf.download('^VIX', start=start.strftime('%Y-%m-%d'),
                      end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
                      progress=False)
    print(f"[OK] Fetched {len(vix)} days of VIX data\n")
    return vix

def calculate_vix_percentile(vix_data, lookback_days=252):
    """Calculate VIX percentile rank over lookback period."""
    if len(vix_data) == 0:
        return None

    # Extract Close column, handling both single and multi-index columns
    close_col = vix_data['Close']
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]  # Get first column if multi-index

    current_vix = float(close_col.iloc[-1])
    historical_vix = close_col.tail(lookback_days)

    percentile = (historical_vix < current_vix).sum() / len(historical_vix) * 100
    return percentile, current_vix

def calculate_moving_averages(spy_data):
    """Calculate 20, 50, 200-day moving averages."""
    spy_close = spy_data['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]  # Get first column if multi-index

    ma_20 = float(spy_close.rolling(20).mean().iloc[-1])
    ma_50 = float(spy_close.rolling(50).mean().iloc[-1])
    ma_200 = float(spy_close.rolling(200).mean().iloc[-1])
    current_price = float(spy_close.iloc[-1])

    return {
        'current': current_price,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'ma_200': ma_200,
        'above_20': current_price > ma_20,
        'above_50': current_price > ma_50,
        'above_200': current_price > ma_200
    }

def calculate_momentum_slope(spy_data, window=20):
    """Calculate momentum slope (rate of change)."""
    spy_close = spy_data['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]  # Get first column if multi-index

    if len(spy_close) < window:
        return None

    # Calculate rate of change over last 20 days
    current = float(spy_close.iloc[-1])
    past = float(spy_close.iloc[-window])
    pct_change = (current / past - 1)
    slope = pct_change / window  # Daily rate of change

    return slope

def classify_regime(vix_percentile, momentum_slope, ma_positions):
    """
    Classify market regime based on OMR strategy criteria.

    Regime criteria from market_regime_detector.py:
    - STRONG_BULL: VIX percentile < 30, momentum > 0.02, above all MAs
    - WEAK_BULL: VIX percentile < 50, momentum 0-0.02, above 20/50 MAs
    - BEAR: VIX percentile > 70 (blocks all trading)
    """

    # BEAR regime - VIX percentile > 70 blocks all trading
    if vix_percentile > 70:
        return 'BEAR'

    # STRONG_BULL - low volatility, strong momentum, above all MAs
    if (vix_percentile < 30 and
        momentum_slope > 0.02 and
        ma_positions['above_20'] and
        ma_positions['above_50'] and
        ma_positions['above_200']):
        return 'STRONG_BULL'

    # WEAK_BULL - moderate volatility, positive momentum
    if (vix_percentile < 50 and
        momentum_slope > 0.0 and
        momentum_slope <= 0.02 and
        ma_positions['above_20'] and
        ma_positions['above_50']):
        return 'WEAK_BULL'

    # Default to SIDEWAYS/UNPREDICTABLE
    if vix_percentile < 50:
        return 'SIDEWAYS'
    else:
        return 'UNPREDICTABLE'

def main():
    print("=" * 80)
    print("DIAGNOSING ZERO SIGNALS ON NOV 14, 2025")
    print("=" * 80)
    print()

    target_date = '2025-11-14'

    # Fetch data
    spy_data = fetch_spy_data(target_date)
    vix_data = fetch_vix_data(target_date)

    # Calculate metrics
    print("=" * 80)
    print("ANALYZING MARKET CONDITIONS")
    print("=" * 80)
    print()

    # VIX analysis
    vix_percentile, current_vix = calculate_vix_percentile(vix_data)
    print(f"VIX Analysis:")
    print(f"  Current VIX: {current_vix:.2f}")
    print(f"  VIX Percentile (252-day): {vix_percentile:.1f}%")
    print()

    # Moving averages
    ma_positions = calculate_moving_averages(spy_data)
    print(f"SPY Moving Averages:")
    print(f"  Current Price: ${ma_positions['current']:.2f}")
    print(f"  20-day MA: ${ma_positions['ma_20']:.2f} ({'ABOVE' if ma_positions['above_20'] else 'BELOW'})")
    print(f"  50-day MA: ${ma_positions['ma_50']:.2f} ({'ABOVE' if ma_positions['above_50'] else 'BELOW'})")
    print(f"  200-day MA: ${ma_positions['ma_200']:.2f} ({'ABOVE' if ma_positions['above_200'] else 'BELOW'})")
    print()

    # Momentum
    momentum_slope = calculate_momentum_slope(spy_data)
    print(f"Momentum:")
    print(f"  20-day Momentum Slope: {momentum_slope:.4f}")
    print(f"  Direction: {'POSITIVE' if momentum_slope > 0 else 'NEGATIVE'}")
    print()

    # Regime classification
    regime = classify_regime(vix_percentile, momentum_slope, ma_positions)

    print("=" * 80)
    print("REGIME CLASSIFICATION")
    print("=" * 80)
    print()
    print(f"  Detected Regime: {regime}")
    print()

    # Explain outcome
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if regime == 'BEAR':
        print("[SUCCESS] EXPLANATION FOUND: BEAR REGIME DETECTED")
        print()
        print(f"VIX percentile ({vix_percentile:.1f}%) exceeded the 70% threshold.")
        print("OMR strategy correctly disabled all trading during high volatility.")
        print()
        print("This is CORRECT BEHAVIOR - the strategy should not trade when:")
        print("  • VIX is in 70th percentile or higher")
        print("  • Market volatility is elevated")
        print("  • Risk of overnight gaps is high")
        print()
        print("Zero signals indicate proper risk management, not a bug.")
    else:
        print(f"[WARNING] NO BEAR REGIME DETECTED (Regime: {regime})")
        print()
        print("VIX levels were acceptable for trading.")
        print("Zero signals may be due to:")
        print("  1. No ETFs met Bayesian probability threshold (>60%)")
        print("  2. Intraday price movements didn't favor mean reversion")
        print("  3. Position sizing or other filters")
        print()
        print("This indicates market conditions simply didn't favor the strategy.")
        print("Not every day will generate signals - this is normal.")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
