"""
Retrain Bayesian Model with Full LEVERAGED_3X Universe.

Downloads historical data and trains the Bayesian reversion model
on all 20 leveraged 3x ETF symbols.

Usage:
    python scripts/trading/retrain_bayesian_model.py
    python scripts/trading/retrain_bayesian_model.py --years 5
    python scripts/trading/retrain_bayesian_model.py --dry-run
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from src.utils.logger import logger
from src.strategies.universe import ETFUniverse
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.config import get_models_dir


def download_historical_data(symbols: list, years: int = 10) -> dict:
    """
    Download historical daily data for all symbols.

    Args:
        symbols: List of symbols to download
        years: Number of years of history

    Returns:
        Dict of symbol -> DataFrame
    """
    logger.info(f"Downloading {years} years of historical data for {len(symbols)} symbols...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    historical_data = {}
    failed = []

    for symbol in symbols:
        try:
            # Use yfinance for all symbols
            ticker = symbol if not symbol.startswith('^') else symbol

            logger.info(f"  Downloading {symbol}...")
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if df is None or df.empty:
                logger.warning(f"  {symbol}: No data returned")
                failed.append(symbol)
                continue

            # Normalize column names to lowercase
            if hasattr(df.columns, 'levels'):
                # Multi-level columns from yfinance
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Ensure timezone-aware index
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')

            historical_data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} days")

        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")
            failed.append(symbol)
            continue

    logger.info(f"Downloaded {len(historical_data)}/{len(symbols)} symbols")
    if failed:
        logger.warning(f"Failed symbols: {failed}")

    return historical_data


def main():
    parser = argparse.ArgumentParser(description='Retrain Bayesian model with full universe')
    parser.add_argument('--years', type=int, default=10, help='Years of historical data (default: 10)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without training')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("BAYESIAN MODEL RETRAINING")
    logger.info("=" * 70)

    # Get symbols to train
    symbols = ETFUniverse.LEVERAGED_3X.copy()
    market_symbols = ['SPY', '^VIX']
    all_symbols = symbols + market_symbols

    logger.info(f"Target symbols: {len(symbols)} leveraged 3x ETFs")
    logger.info(f"  {symbols}")
    logger.info(f"Plus market indicators: {market_symbols}")
    logger.info(f"Training period: {args.years} years")
    logger.info(f"Model will be saved to: {get_models_dir() / 'bayesian_reversion_model.pkl'}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would download and train on these symbols:")
        for s in all_symbols:
            logger.info(f"  - {s}")
        logger.info("\nRun without --dry-run to actually train")
        return

    # Download historical data
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOADING HISTORICAL DATA")
    logger.info("=" * 70)

    historical_data = download_historical_data(all_symbols, years=args.years)

    # Validate required data
    if 'SPY' not in historical_data:
        logger.error("FATAL: Failed to download SPY data - cannot train")
        return 1

    if '^VIX' not in historical_data:
        logger.error("FATAL: Failed to download VIX data - cannot train")
        return 1

    # Rename ^VIX to VIX for consistency
    if '^VIX' in historical_data:
        historical_data['VIX'] = historical_data.pop('^VIX')

    # Count successful ETF downloads
    etf_count = len([s for s in symbols if s in historical_data])
    logger.info(f"\nSuccessfully downloaded {etf_count}/{len(symbols)} leveraged ETFs")

    if etf_count < 10:
        logger.error(f"Too few symbols downloaded ({etf_count}/20). Aborting.")
        return 1

    # Initialize components
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING BAYESIAN MODEL")
    logger.info("=" * 70)

    regime_detector = MarketRegimeDetector()
    bayesian_model = BayesianReversionModel(lookback_years=args.years, data_frequency='daily')

    # Prepare data for training
    spy_data = historical_data['SPY']
    vix_data = historical_data['VIX']

    # Train the model
    try:
        bayesian_model.train(
            historical_data=historical_data,
            regime_detector=regime_detector,
            spy_data=spy_data,
            vix_data=vix_data
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify the model
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    model_symbols = list(bayesian_model.regime_probabilities.keys())
    logger.info(f"Model trained on {len(model_symbols)} symbols:")

    # Check coverage of LEVERAGED_3X
    covered = [s for s in symbols if s in model_symbols]
    missing = [s for s in symbols if s not in model_symbols]

    logger.info(f"\nLEVERAGED_3X coverage: {len(covered)}/{len(symbols)}")
    logger.info(f"  Covered: {covered}")
    if missing:
        logger.warning(f"  Missing: {missing}")

    # Show sample probabilities
    logger.info("\nSample probability data (SIDEWAYS regime, flat bucket):")
    for symbol in covered[:5]:
        prob_data = bayesian_model.regime_probabilities.get(symbol, {}).get('SIDEWAYS', {}).get('flat')
        if prob_data:
            logger.info(f"  {symbol}: win_rate={prob_data['probability']:.1%}, "
                       f"exp_return={prob_data['expected_return']:.3%}, "
                       f"samples={prob_data['sample_size']}")

    logger.info("\n" + "=" * 70)
    logger.success(f"MODEL RETRAINED SUCCESSFULLY")
    logger.info(f"Saved to: {bayesian_model.model_path}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
