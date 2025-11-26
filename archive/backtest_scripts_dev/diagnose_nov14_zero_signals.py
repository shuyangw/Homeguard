#!/usr/bin/env python3
"""
Diagnostic script to determine why OMR strategy generated 0 signals on Nov 14, 2025.
Checks:
1. Market regime classification (VIX-based)
2. Individual ETF signal generation
3. Bayesian probability thresholds
"""

import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path
project_root = Path(__file__).parent.parent

from strategies.advanced.market_regime_detector import MarketRegimeDetector
from strategies.advanced.overnight_signal_generator import OvernightSignalGenerator
from utils.logger import logger

# OMR universe - 20 leveraged ETFs
LEVERAGED_ETFS = [
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'TNA', 'TZA',
    'UDOW', 'SDOW', 'TECL', 'TECS', 'FAS', 'FAZ',
    'CURE', 'LABD', 'WANT', 'RETL', 'MIDU', 'MIDZ',
    'NAIL', 'YANG'
]

def fetch_spy_data(end_date: str = '2025-11-14', days_back: int = 252) -> pd.DataFrame:
    """Fetch SPY data for regime detection."""
    logger.info(f"Fetching SPY data for regime detection (last {days_back} days)")

    end = pd.Timestamp(end_date)
    start = end - timedelta(days=days_back)

    spy = yf.download('SPY', start=start.strftime('%Y-%m-%d'),
                      end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
                      progress=False)

    logger.info(f"Fetched {len(spy)} days of SPY data")
    return spy


def fetch_vix_data(end_date: str = '2025-11-14', days_back: int = 252) -> pd.DataFrame:
    """Fetch VIX data."""
    logger.info(f"Fetching VIX data (last {days_back} days)")

    end = pd.Timestamp(end_date)
    start = end - timedelta(days=days_back)

    vix = yf.download('^VIX', start=start.strftime('%Y-%m-%d'),
                      end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
                      progress=False)

    logger.info(f"Fetched {len(vix)} days of VIX data")
    return vix


def fetch_etf_data(symbol: str, end_date: str = '2025-11-14', days_back: int = 252) -> pd.DataFrame:
    """Fetch individual ETF data."""
    end = pd.Timestamp(end_date)
    start = end - timedelta(days=days_back)

    try:
        data = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                          end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
                          progress=False)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def main():
    logger.info("=" * 80)
    logger.info("DIAGNOSING ZERO SIGNALS ON NOV 14, 2025")
    logger.info("=" * 80)

    target_date = '2025-11-14'

    # Step 1: Check market regime
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: MARKET REGIME DETECTION")
    logger.info("=" * 80)

    spy_data = fetch_spy_data(end_date=target_date)
    vix_data = fetch_vix_data(end_date=target_date)

    # Initialize regime detector
    detector = MarketRegimeDetector()

    # Get regime for Nov 14
    regime_result = detector.detect_regime(spy_data, vix_data)

    logger.info(f"\nRegime on {target_date}:")
    logger.info(f"  Regime: {regime_result['regime']}")
    logger.info(f"  VIX Level: {regime_result.get('vix_level', 'N/A'):.2f}")
    logger.info(f"  VIX Percentile: {regime_result.get('vix_percentile', 'N/A'):.1f}%")
    logger.info(f"  Momentum Slope: {regime_result.get('momentum_slope', 'N/A'):.4f}")
    logger.info(f"  20-day MA Position: {regime_result.get('above_20ma', 'N/A')}")
    logger.info(f"  50-day MA Position: {regime_result.get('above_50ma', 'N/A')}")
    logger.info(f"  200-day MA Position: {regime_result.get('above_200ma', 'N/A')}")

    # Check if BEAR regime blocked trading
    if regime_result['regime'] == 'BEAR':
        logger.warning("\n⚠️  BEAR REGIME DETECTED - OMR STRATEGY DISABLED")
        logger.warning("This is the primary reason for zero signals.")
        logger.warning(f"VIX percentile ({regime_result.get('vix_percentile', 0):.1f}%) exceeded threshold (70%)")
        return

    # Step 2: Check individual ETF signals
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CHECKING INDIVIDUAL ETF SIGNALS")
    logger.info("=" * 80)
    logger.info(f"Regime is {regime_result['regime']} - OMR strategy ENABLED")
    logger.info("Checking each of 20 leveraged ETFs for entry signals...\n")

    signal_generator = OvernightSignalGenerator()
    signals_generated = []
    signals_rejected = []

    for symbol in LEVERAGED_ETFS:
        logger.info(f"Analyzing {symbol}...")

        etf_data = fetch_etf_data(symbol, end_date=target_date)

        if etf_data is None or len(etf_data) < 50:
            logger.error(f"  ❌ Insufficient data for {symbol}")
            signals_rejected.append((symbol, "Insufficient data"))
            continue

        # Generate signal using OMR logic
        signal = signal_generator.generate_signal(
            symbol=symbol,
            data=etf_data,
            regime=regime_result['regime']
        )

        if signal and signal.get('action') in ['BUY', 'SELL']:
            logger.info(f"  ✅ SIGNAL: {signal['action']} {symbol}")
            logger.info(f"     Probability: {signal.get('probability', 0):.2%}")
            logger.info(f"     Expected Return: {signal.get('expected_return', 0):.2%}")
            signals_generated.append((symbol, signal))
        else:
            reason = signal.get('reason', 'Unknown') if signal else 'No signal'
            logger.info(f"  ❌ No signal - {reason}")
            signals_rejected.append((symbol, reason))

    # Step 3: Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Target Date: {target_date}")
    logger.info(f"Market Regime: {regime_result['regime']}")
    logger.info(f"VIX Percentile: {regime_result.get('vix_percentile', 0):.1f}%")
    logger.info(f"\nETFs Analyzed: {len(LEVERAGED_ETFS)}")
    logger.info(f"Signals Generated: {len(signals_generated)}")
    logger.info(f"Signals Rejected: {len(signals_rejected)}")

    if signals_generated:
        logger.info("\n✅ SIGNALS THAT SHOULD HAVE BEEN GENERATED:")
        for symbol, signal in signals_generated:
            logger.info(f"  {signal['action']} {symbol} - Prob: {signal.get('probability', 0):.2%}")

    if signals_rejected:
        logger.info("\n❌ REJECTION REASONS:")
        rejection_counts = {}
        for symbol, reason in signals_rejected:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count} ETFs")

    # Conclusion
    logger.info("\n" + "=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)

    if regime_result['regime'] == 'BEAR':
        logger.info("✅ Zero signals explained: BEAR regime filter blocked all trading")
        logger.info("   This is correct behavior - OMR should not trade in high volatility")
    elif len(signals_generated) == 0:
        logger.info("✅ Zero signals explained: No ETFs met Bayesian probability threshold")
        logger.info("   This indicates market conditions did not favor mean reversion")
    else:
        logger.warning("⚠️  UNEXPECTED: Signals should have been generated but weren't")
        logger.warning("   This may indicate a bug in the live trading system")


if __name__ == "__main__":
    main()
