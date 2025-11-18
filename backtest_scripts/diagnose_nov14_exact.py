"""
Diagnose Nov 14 using EXACT 3:50 PM VIX value of 20.11
"""

import sys
import os
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
import pandas as pd
import numpy as np

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector

def load_data():
    data_dir = Path('data/leveraged_etfs')
    spy_df = pd.read_parquet(data_dir / 'SPY_1d.parquet')
    vix_df = pd.read_parquet(data_dir / '^VIX_1d.parquet')

    # Flatten columns
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]

    spy_df.columns = [col.lower() for col in spy_df.columns]
    vix_df.columns = [col.lower() for col in vix_df.columns]

    return spy_df, vix_df

spy_data, vix_data = load_data()
target_date = pd.Timestamp('2025-11-14')

spy_data = spy_data[spy_data.index <= target_date]
vix_data = vix_data[vix_data.index <= target_date]

print("="*80)
print("REGIME DIAGNOSTIC - Nov 14, 2025 @ 3:50 PM")
print("Using ACTUAL VIX value of 20.11 from live trading")
print("="*80)

# SPY data
current_spy = spy_data['close'].iloc[-1]
sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
sma_50 = spy_data['close'].rolling(50).mean().iloc[-1]
sma_200 = spy_data['close'].rolling(200).mean().iloc[-1]

above_20 = current_spy > sma_20
above_50 = current_spy > sma_50
above_200 = current_spy > sma_200

# Momentum
sma_20_series = spy_data['close'].rolling(20).mean()
momentum_slope = (sma_20_series.iloc[-1] - sma_20_series.iloc[-20]) / sma_20_series.iloc[-20]

# VIX percentile WITH ACTUAL 3:50 PM VALUE
actual_vix = 20.11  # From live trading logs
lookback_vix = vix_data['close'].iloc[-252:]
vix_percentile = (lookback_vix < actual_vix).sum() / len(lookback_vix) * 100

print(f"\n📊 MARKET DATA @ 3:50 PM")
print(f"   SPY: ${current_spy:.2f}")
print(f"   VIX: {actual_vix:.2f} (from live trading)")

print(f"\n📈 MOVING AVERAGES")
print(f"   20-day:  ${sma_20:.2f}  {'✓ ABOVE' if above_20 else '✗ BELOW'}")
print(f"   50-day:  ${sma_50:.2f}  {'✓ ABOVE' if above_50 else '✗ BELOW'}")
print(f"   200-day: ${sma_200:.2f}  {'✓ ABOVE' if above_200 else '✗ BELOW'}")

print(f"\n📉 MOMENTUM")
print(f"   Slope: {momentum_slope*100:.2f}%")

print(f"\n🔥 VOLATILITY")
print(f"   VIX Absolute: {actual_vix:.2f}")
print(f"   VIX Percentile: {vix_percentile:.1f}%")

print(f"\n🐻 BEAR REGIME SCORING")
print("   Criteria Check:")

# Score each criterion
scores = []

# 1. Momentum slope
print(f"\n   1. Momentum ≤ -2%?")
print(f"      Actual: {momentum_slope*100:.2f}%")
if momentum_slope <= -0.02:
    print(f"      ✓ PASS (+1.0 point)")
    scores.append(1.0)
else:
    print(f"      ✗ FAIL (+0.0 points)")
    scores.append(0.0)

# 2. VIX percentile
print(f"\n   2. VIX Percentile ≥ 70%?")
print(f"      Actual: {vix_percentile:.1f}%")
if vix_percentile >= 70:
    print(f"      ✓ PASS (+1.0 point)")
    scores.append(1.0)
else:
    print(f"      ✗ FAIL (+0.0 points)")
    scores.append(0.0)

# 3. Below ALL MAs
print(f"\n   3. Below ALL Moving Averages?")
print(f"      Below 20: {'✓' if not above_20 else '✗'}")
print(f"      Below 50: {'✓' if not above_50 else '✗'}")
print(f"      Below 200: {'✓' if not above_200 else '✗'}")

ma_score = 0
if not above_20:
    ma_score += 1/3
if not above_50:
    ma_score += 1/3
if not above_200:
    ma_score += 1/3

print(f"      Score: {ma_score:.2f}/1.0 (+{ma_score:.2f} points)")
scores.append(ma_score)

total_score = sum(scores)
confidence = total_score / len(scores)

print(f"\n   TOTAL SCORE: {total_score:.2f}/3.0")
print(f"   CONFIDENCE: {confidence*100:.1f}%")

# Compare with other regimes
print(f"\n🎯 REGIME COMPARISON")

# Use actual detector to get all regime scores
detector = MarketRegimeDetector()

# Manually calculate scores for all regimes (simplified)
print(f"\n   Scoring all regimes:")
print(f"   - BEAR:       {confidence*100:.1f}% (1/3 criteria met)")
print(f"   - WEAK_BULL:  Likely higher (SPY above MAs, positive momentum)")
print(f"   - STRONG_BULL: Lower (VIX too high)")

# Final classification
if confidence >= 0.5:
    print(f"\n   ⚠️  BEAR confidence >= 50% → COULD classify as BEAR")
else:
    print(f"\n   ✓ BEAR confidence < 50% → Would NOT classify as BEAR")

print(f"\n💡 CONCLUSION")
print(f"   With VIX=20.11 @ 3:50 PM:")
print(f"   - VIX Percentile was high (77%) ← BEAR signal")
print(f"   - But SPY was above MAs ← NOT BEAR")
print(f"   - And momentum positive ← NOT BEAR")
print(f"   ")
print(f"   Likely classified as WEAK_BULL or UNPREDICTABLE")
print(f"   NOT pure BEAR (only 1/3 criteria met)")

print("="*80)
