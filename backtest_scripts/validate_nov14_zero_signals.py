"""
Comprehensive validation of why Nov 14, 2025 had zero signals.

Downloads actual intraday data and validates:
1. Intraday price movements
2. Bayesian probability filtering

Author: Homeguard Quantitative Research
Date: 2025-11-17
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel

# OMR symbol universe
OMR_SYMBOLS = [
    'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
    'DFEN', 'WEBL', 'UCO', 'NAIL', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL', 'CUT'
]

# OMR filters
FILTERS = {
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'min_intraday_move': 0.005  # 0.5% minimum
}

def download_intraday_data(symbol, date):
    """Download 1-minute intraday data for a specific date."""
    try:
        # Download data for the specific date
        start_date = pd.Timestamp(date)
        end_date = start_date + timedelta(days=1)

        data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1m',
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        return data

    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return None

def calculate_intraday_move(intraday_data, entry_time='15:50'):
    """
    Calculate intraday price movement from open to 3:50 PM.

    Returns:
        dict with open_price, close_price (at 3:50), intraday_return
    """
    if intraday_data is None or intraday_data.empty:
        return None

    try:
        # Get market open price (9:30 AM)
        # Data is in UTC, so 9:30 AM EST = 14:30 UTC
        morning_data = intraday_data.between_time('14:30', '14:35')
        if morning_data.empty:
            # Try first available price
            open_price = intraday_data['Open'].iloc[0]
        else:
            open_price = morning_data['Open'].iloc[0]

        # Get 3:50 PM price
        # 3:50 PM EST = 20:50 UTC
        entry_time_data = intraday_data.between_time('20:48', '20:52')
        if entry_time_data.empty:
            # No data at 3:50 PM
            return None

        close_price = entry_time_data['Close'].iloc[-1]

        intraday_return = (close_price - open_price) / open_price

        return {
            'open_price': float(open_price),
            'close_price': float(close_price),
            'intraday_return': float(intraday_return)
        }

    except Exception as e:
        logger.error(f"Error calculating intraday move: {e}")
        return None

def load_trained_model():
    """Load the trained Bayesian reversion model."""
    try:
        model_path = Path('/Users/shuyangw/Library/CloudStorage/Dropbox/cs/stonk/output/models/bayesian_reversion_model.pkl')

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            return None

        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Create model instance and load data
        model = BayesianReversionModel(data_frequency='daily')
        model.regime_probabilities = model_data.get('regime_probabilities', {})
        model.trained = model_data.get('trained', False)
        model.training_stats = model_data.get('training_stats', {})

        logger.info(f"Loaded Bayesian model with {len(model.regime_probabilities)} symbols, trained={model.trained}")
        if model.trained:
            logger.info(f"   Training stats: {model.training_stats.get('total_setups', 0)} setups, {model.training_stats.get('symbols_trained', 0)} symbols")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def check_bayesian_probability(model, symbol, regime, intraday_return):
    """Check if symbol passes Bayesian probability filters."""
    if model is None:
        return None

    try:
        prob_data = model.get_reversion_probability(symbol, regime, intraday_return)

        if prob_data is None:
            return {
                'symbol': symbol,
                'passes': False,
                'reason': 'No probability data available',
                'probability': None,
                'expected_return': None,
                'sample_size': None
            }

        # Check filters
        passes_win_rate = prob_data['probability'] >= FILTERS['min_win_rate']
        passes_return = prob_data['expected_return'] >= FILTERS['min_expected_return']
        passes_sample = prob_data['sample_size'] >= FILTERS['min_sample_size']

        passes_all = passes_win_rate and passes_return and passes_sample

        reasons = []
        if not passes_win_rate:
            reasons.append(f"Win rate {prob_data['probability']*100:.1f}% < {FILTERS['min_win_rate']*100:.1f}%")
        if not passes_return:
            reasons.append(f"Expected return {prob_data['expected_return']*100:.2f}% < {FILTERS['min_expected_return']*100:.2f}%")
        if not passes_sample:
            reasons.append(f"Sample size {prob_data['sample_size']} < {FILTERS['min_sample_size']}")

        return {
            'symbol': symbol,
            'passes': passes_all,
            'reason': '; '.join(reasons) if reasons else 'PASS',
            'probability': prob_data['probability'],
            'expected_return': prob_data['expected_return'],
            'sample_size': prob_data['sample_size']
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'passes': False,
            'reason': f'Error: {str(e)}',
            'probability': None,
            'expected_return': None,
            'sample_size': None
        }

def main():
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE VALIDATION: Nov 14, 2025 Zero Signals")
    logger.info("="*80)

    target_date = '2025-11-14'

    # Load regime detector and classify regime
    logger.info("\n[1/4] Classifying Market Regime...")

    # Load SPY and VIX for regime detection
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

    spy_data = spy_df[spy_df.index <= target_date]
    vix_data = vix_df[vix_df.index <= target_date]

    detector = MarketRegimeDetector()
    regime, confidence = detector.classify_regime(spy_data, vix_data, pd.Timestamp(target_date))

    logger.info(f"   Regime: {regime} ({confidence*100:.1f}% confidence)")

    # Download intraday data
    logger.info("\n[2/4] Downloading Intraday Data for Nov 14...")

    intraday_data = {}
    for symbol in OMR_SYMBOLS:
        logger.info(f"   Downloading {symbol}...")
        data = download_intraday_data(symbol, target_date)
        if data is not None and not data.empty:
            intraday_data[symbol] = data

    logger.info(f"\n   Downloaded {len(intraday_data)}/{len(OMR_SYMBOLS)} symbols")

    # Calculate intraday moves
    logger.info("\n[3/4] Calculating Intraday Price Movements...")

    movements = []
    for symbol, data in intraday_data.items():
        move = calculate_intraday_move(data)
        if move is not None:
            movements.append({
                'symbol': symbol,
                **move,
                'abs_move': abs(move['intraday_return']),
                'passes_min_move': abs(move['intraday_return']) >= FILTERS['min_intraday_move']
            })

    movements_df = pd.DataFrame(movements)

    if not movements_df.empty:
        movements_df = movements_df.sort_values('abs_move', ascending=False)

        logger.info(f"\n   Intraday Movement Analysis:")
        logger.info(f"   Total symbols with data: {len(movements_df)}")
        logger.info(f"   Symbols with ≥0.5% move: {movements_df['passes_min_move'].sum()}")
        logger.info(f"   Symbols with <0.5% move: {(~movements_df['passes_min_move']).sum()}")

    # Load Bayesian model and check probabilities
    logger.info("\n[4/4] Checking Bayesian Probability Filters...")

    model = load_trained_model()

    if model is not None and not movements_df.empty:
        # Check probabilities for symbols that passed intraday move filter
        qualifying_symbols = movements_df[movements_df['passes_min_move']]

        logger.info(f"\n   Checking {len(qualifying_symbols)} symbols with sufficient intraday moves...")

        bayesian_results = []
        for _, row in qualifying_symbols.iterrows():
            result = check_bayesian_probability(
                model,
                row['symbol'],
                regime,
                row['intraday_return']
            )
            if result:
                bayesian_results.append(result)

        bayesian_df = pd.DataFrame(bayesian_results)

        # Generate report
        logger.info("\n" + "="*80)
        logger.info("VALIDATION RESULTS")
        logger.info("="*80)

        logger.info(f"\n📊 HYPOTHESIS 1: Intraday Moves Too Small")
        logger.info(f"   Total symbols analyzed: {len(movements_df)}")
        logger.info(f"   Symbols with ≥0.5% move: {movements_df['passes_min_move'].sum()}")
        logger.info(f"   Symbols with <0.5% move: {(~movements_df['passes_min_move']).sum()}")

        if movements_df['passes_min_move'].sum() == 0:
            logger.info(f"   ✓ HYPOTHESIS CONFIRMED: All symbols had <0.5% intraday moves")
        else:
            logger.info(f"   ✗ HYPOTHESIS REJECTED: {movements_df['passes_min_move'].sum()} symbols had sufficient moves")

        logger.info(f"\n🎯 HYPOTHESIS 2: Failed Bayesian Probability Filters")
        logger.info(f"   Symbols checked: {len(bayesian_df)}")
        if not bayesian_df.empty:
            passing = bayesian_df['passes'].sum()
            failing = len(bayesian_df) - passing

            logger.info(f"   Symbols PASSING all filters: {passing}")
            logger.info(f"   Symbols FAILING filters: {failing}")

            if passing == 0:
                logger.info(f"   ✓ HYPOTHESIS CONFIRMED: All symbols failed Bayesian filters")
            else:
                logger.info(f"   ✗ HYPOTHESIS REJECTED: {passing} symbols passed filters")

        # Detailed breakdown
        logger.info(f"\n" + "="*80)
        logger.info("DETAILED SYMBOL ANALYSIS")
        logger.info("="*80)

        # Show top movers
        logger.info(f"\n🔝 TOP 10 INTRADAY MOVERS:")
        for i, row in movements_df.head(10).iterrows():
            status = "✓ PASS" if row['passes_min_move'] else "✗ FAIL"
            logger.info(f"   {row['symbol']:6s}: {row['intraday_return']*100:+6.2f}% {status}")

        # Show Bayesian filter results for qualifying symbols
        if not bayesian_df.empty:
            logger.info(f"\n🎯 BAYESIAN FILTER RESULTS (symbols with ≥0.5% move):")
            for _, row in bayesian_df.iterrows():
                status = "✓ PASS" if row['passes'] else "✗ FAIL"
                logger.info(f"\n   {row['symbol']:6s}: {status}")
                if row['probability'] is not None:
                    logger.info(f"      Win Rate: {row['probability']*100:.1f}% (need ≥58%)")
                    logger.info(f"      Exp Return: {row['expected_return']*100:.2f}% (need ≥0.2%)")
                    logger.info(f"      Sample Size: {row['sample_size']} (need ≥15)")
                if not row['passes']:
                    logger.info(f"      Reason: {row['reason']}")

        # Final conclusion
        logger.info(f"\n" + "="*80)
        logger.info("FINAL CONCLUSION")
        logger.info("="*80)

        total_qualifying = movements_df['passes_min_move'].sum()

        if total_qualifying == 0:
            logger.info("\n✓ PRIMARY REASON: ALL symbols had insufficient intraday moves (<0.5%)")
            logger.info("  This was a low-volatility day with minimal price action.")
            logger.info("  Strategy correctly skipped trading - no setups available.")
        elif not bayesian_df.empty and bayesian_df['passes'].sum() == 0:
            logger.info(f"\n✓ PRIMARY REASON: {total_qualifying} symbols had sufficient moves,")
            logger.info("  but ALL failed Bayesian probability filters (win rate, expected return, or sample size).")
            logger.info("  Strategy correctly filtered out low-quality setups.")
        elif not bayesian_df.empty and bayesian_df['passes'].sum() > 0:
            logger.info(f"\n⚠️  UNEXPECTED: {bayesian_df['passes'].sum()} symbols SHOULD have generated signals!")
            logger.info("  This suggests another filter blocked trading:")
            logger.info("  - Regime filter (BEAR or UNPREDICTABLE)")
            logger.info("  - Position limit already reached")
            logger.info("  - System error")
        else:
            logger.info("\n⚠️  Unable to fully validate - missing Bayesian model data")

        # Save detailed results
        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)

        movements_df.to_csv(output_dir / '20251114_intraday_movements.csv', index=False)
        if not bayesian_df.empty:
            bayesian_df.to_csv(output_dir / '20251114_bayesian_results.csv', index=False)

        logger.info(f"\n📁 Detailed results saved:")
        logger.info(f"   reports/20251114_intraday_movements.csv")
        if not bayesian_df.empty:
            logger.info(f"   reports/20251114_bayesian_results.csv")

    else:
        logger.error("Failed to load Bayesian model or no movement data available")

    logger.info("\n" + "="*80)

if __name__ == '__main__':
    main()
