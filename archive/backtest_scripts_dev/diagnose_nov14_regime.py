"""
Diagnose why Nov 14, 2025 was classified as BEAR regime.

Shows the exact evidence and calculations used by MarketRegimeDetector.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
import pandas as pd
import numpy as np

# Add src to path
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger

def load_data():
    """Load SPY and VIX data."""
    data_dir = Path('data/leveraged_etfs')

    spy_path = data_dir / 'SPY_1d.parquet'
    vix_path = data_dir / '^VIX_1d.parquet'

    if not spy_path.exists() or not vix_path.exists():
        logger.error("Data files not found")
        return None, None

    spy_df = pd.read_parquet(spy_path)
    vix_df = pd.read_parquet(vix_path)

    # Flatten multi-index columns
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    # Normalize to lowercase
    spy_df.columns = [col.lower() for col in spy_df.columns]
    vix_df.columns = [col.lower() for col in vix_df.columns]

    return spy_df, vix_df

def diagnose_regime(spy_data, vix_data, target_date):
    """Diagnose regime classification for specific date."""

    # Get data up to target date
    spy_data = spy_data[spy_data.index <= target_date]
    vix_data = vix_data[vix_data.index <= target_date]

    if len(spy_data) < 200:
        logger.error("Insufficient SPY data")
        return

    # Calculate indicators manually to show the evidence
    print("\n" + "="*80)
    print(f"REGIME DIAGNOSTIC FOR {target_date.strftime('%Y-%m-%d')}")
    print("="*80)

    # Current values
    current_spy = spy_data['close'].iloc[-1]
    current_vix = vix_data['close'].iloc[-1]

    print(f"\n📊 CURRENT MARKET DATA")
    print(f"   SPY Close: ${current_spy:.2f}")
    print(f"   VIX Close: {current_vix:.2f}")

    # Moving averages
    sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
    sma_50 = spy_data['close'].rolling(50).mean().iloc[-1]
    sma_200 = spy_data['close'].rolling(200).mean().iloc[-1]

    above_20 = current_spy > sma_20
    above_50 = current_spy > sma_50
    above_200 = current_spy > sma_200

    print(f"\n📈 MOVING AVERAGES")
    print(f"   20-day SMA:  ${sma_20:.2f}  {'✓ ABOVE' if above_20 else '✗ BELOW'}")
    print(f"   50-day SMA:  ${sma_50:.2f}  {'✓ ABOVE' if above_50 else '✗ BELOW'}")
    print(f"   200-day SMA: ${sma_200:.2f}  {'✓ ABOVE' if above_200 else '✗ BELOW'}")

    # Momentum slope
    sma_20_series = spy_data['close'].rolling(20).mean()
    momentum_slope = (sma_20_series.iloc[-1] - sma_20_series.iloc[-20]) / sma_20_series.iloc[-20]

    print(f"\n📉 MOMENTUM")
    print(f"   20-day Momentum Slope: {momentum_slope*100:.2f}%")
    print(f"   Interpretation: {'Positive (bullish)' if momentum_slope > 0 else 'Negative (bearish)'}")

    # VIX percentile
    lookback_vix = vix_data['close'].iloc[-252:]  # 1 year
    vix_percentile = (lookback_vix < current_vix).sum() / len(lookback_vix) * 100

    vix_20_avg = vix_data['close'].rolling(20).mean().iloc[-1]

    print(f"\n🔥 VOLATILITY")
    print(f"   VIX Absolute: {current_vix:.2f}")
    print(f"   VIX Percentile (252-day): {vix_percentile:.1f}%")
    print(f"   VIX 20-day Average: {vix_20_avg:.2f}")
    print(f"   Volatility Spike: {current_vix > (vix_20_avg * 1.5)}")

    # BEAR regime criteria
    print(f"\n🐻 BEAR REGIME CRITERIA CHECK")
    print("   Required for BEAR classification:")

    bear_criteria = {
        'momentum_slope_max': -0.02,     # Strong negative momentum
        'vix_percentile_min': 70,        # High volatility
        'below_mas': ['20', '50', '200'], # Below all moving averages
    }

    print(f"\n   1. Momentum Slope ≤ -2%")
    print(f"      Actual: {momentum_slope*100:.2f}%")
    print(f"      {'✓ PASS' if momentum_slope <= -0.02 else '✗ FAIL'}")

    print(f"\n   2. VIX Percentile ≥ 70%")
    print(f"      Actual: {vix_percentile:.1f}%")
    print(f"      {'✓ PASS' if vix_percentile >= 70 else '✗ FAIL'}")

    print(f"\n   3. Below ALL Moving Averages (20, 50, 200)")
    print(f"      Below 20-day MA: {'✓ YES' if not above_20 else '✗ NO'}")
    print(f"      Below 50-day MA: {'✓ YES' if not above_50 else '✗ NO'}")
    print(f"      Below 200-day MA: {'✓ YES' if not above_200 else '✗ NO'}")

    below_all = (not above_20) and (not above_50) and (not above_200)
    print(f"      Result: {'✓ PASS (below all)' if below_all else '✗ FAIL (not below all)'}")

    # Calculate BEAR regime score
    bear_score = 0.0
    criteria_count = 0

    # Momentum slope check
    if momentum_slope <= bear_criteria['momentum_slope_max']:
        bear_score += 1.0
    criteria_count += 1

    # VIX percentile check
    if vix_percentile >= bear_criteria['vix_percentile_min']:
        bear_score += 1.0
    criteria_count += 1

    # MA position check
    ma_score = 0
    for ma in ['20', '50', '200']:
        above_key = f'above_{ma}'
        if ma == '20' and not above_20:
            ma_score += 1
        elif ma == '50' and not above_50:
            ma_score += 1
        elif ma == '200' and not above_200:
            ma_score += 1
    bear_score += ma_score / 3
    criteria_count += 1

    bear_confidence = bear_score / criteria_count

    print(f"\n   BEAR Regime Score: {bear_score:.1f}/{criteria_count} = {bear_confidence*100:.1f}% confidence")

    # Use actual regime detector
    detector = MarketRegimeDetector()
    regime, confidence = detector.classify_regime(spy_data, vix_data, target_date)

    print(f"\n🎯 FINAL CLASSIFICATION")
    print(f"   Regime: {regime}")
    print(f"   Confidence: {confidence*100:.1f}%")

    # Explanation
    print(f"\n💡 EXPLANATION")
    if regime == 'BEAR':
        reasons = []
        if vix_percentile >= 70:
            reasons.append(f"VIX at {vix_percentile:.1f}% percentile (very high)")
        if momentum_slope <= -0.02:
            reasons.append(f"Strong negative momentum ({momentum_slope*100:.2f}%)")
        if not above_200:
            reasons.append("SPY below 200-day MA (long-term downtrend)")

        print("   Classified as BEAR because:")
        for i, reason in enumerate(reasons, 1):
            print(f"   {i}. {reason}")
    else:
        print(f"   Not BEAR regime - classified as {regime}")

    print("\n" + "="*80)

def main():
    logger.info("Loading market data...")
    spy_data, vix_data = load_data()

    if spy_data is None or vix_data is None:
        return

    # Diagnose Nov 14, 2025
    target_date = pd.Timestamp('2025-11-14')

    diagnose_regime(spy_data, vix_data, target_date)

    # Also show Nov 13 for comparison
    print("\n\n" + "="*80)
    print("COMPARISON: Nov 13, 2025 (day before)")
    print("="*80)
    target_date = pd.Timestamp('2025-11-13')
    diagnose_regime(spy_data, vix_data, target_date)

if __name__ == '__main__':
    main()
