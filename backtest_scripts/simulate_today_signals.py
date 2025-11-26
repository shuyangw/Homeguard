"""
Simulate today's OMR signal generation to debug why 0 signals were generated.

Downloads today's market data and runs the signal generator with verbose logging.
"""

import sys
from pathlib import Path
from datetime import datetime, time, timedelta
import pandas as pd

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

from src.utils.logger import logger
from src.strategies.universe.etf_universe import ETFUniverse
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals


def download_vix_data(date: datetime) -> pd.DataFrame:
    """
    Download VIX data from yfinance.

    Args:
        date: Target date

    Returns:
        DataFrame with VIX OHLCV data
    """
    import yfinance as yf
    import pytz

    eastern = pytz.timezone('US/Eastern')

    # Download last 252 days for regime detection
    end_date = date + timedelta(days=1)
    start_date = date - timedelta(days=365)

    logger.info(f"Downloading VIX data from yfinance...")

    vix = yf.download('^VIX', start=start_date.strftime('%Y-%m-%d'),
                     end=end_date.strftime('%Y-%m-%d'), progress=False)

    if vix.empty:
        raise ValueError("Failed to download VIX data from yfinance")

    # Standardize column names
    vix.columns = vix.columns.get_level_values(0).str.lower()
    vix.index = pd.to_datetime(vix.index)

    # Localize to Eastern time
    if vix.index.tz is None:
        vix.index = vix.index.tz_localize(eastern)
    else:
        vix.index = vix.index.tz_convert(eastern)

    logger.info(f"  VIX: {len(vix)} daily bars")

    return vix


def download_today_data(symbols: list, date: datetime = None) -> dict:
    """
    Download intraday data for today using Alpaca.

    Args:
        symbols: List of symbols to download
        date: Date to download (defaults to today)

    Returns:
        Dict of symbol -> DataFrame with OHLCV data
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import os

    if date is None:
        date = datetime.now()

    # Get Alpaca credentials (try multiple env var names)
    api_key = (
        os.environ.get('ALPACA_PAPER_KEY_ID') or
        os.environ.get('ALPACA_API_KEY') or
        os.environ.get('APCA_API_KEY_ID')
    )
    api_secret = (
        os.environ.get('ALPACA_PAPER_SECRET_KEY') or
        os.environ.get('ALPACA_API_SECRET') or
        os.environ.get('APCA_API_SECRET_KEY')
    )

    if not api_key or not api_secret:
        # Try loading from .env or config
        try:
            from dotenv import load_dotenv
            env_path = ROOT_DIR / '.env'
            load_dotenv(env_path)
            api_key = (
                os.environ.get('ALPACA_PAPER_KEY_ID') or
                os.environ.get('ALPACA_API_KEY')
            )
            api_secret = (
                os.environ.get('ALPACA_PAPER_SECRET_KEY') or
                os.environ.get('ALPACA_API_SECRET')
            )
        except ImportError:
            pass

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API credentials not found. "
            "Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY environment variables"
        )

    client = StockHistoricalDataClient(api_key, api_secret)

    # For intraday, we need to handle timezone properly
    import pytz
    eastern = pytz.timezone('US/Eastern')

    # Request 1-minute bars for today
    start = datetime.combine(date.date(), time(9, 30))
    end = datetime.combine(date.date(), time(16, 0))
    start = eastern.localize(start)
    end = eastern.localize(end)

    # Filter out VIX (not available from Alpaca)
    alpaca_symbols = [s for s in symbols if s != 'VIX']

    logger.info(f"Downloading data for {len(alpaca_symbols)} symbols from Alpaca...")

    request = StockBarsRequest(
        symbol_or_symbols=alpaca_symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )

    bars = client.get_stock_bars(request)

    # Convert to dict of DataFrames
    data = {}
    for symbol in alpaca_symbols:
        if symbol in bars.data:
            df = bars.data[symbol]
            if hasattr(df, '__iter__') and len(df) > 0:
                records = []
                for bar in df:
                    records.append({
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    })
                df = pd.DataFrame(records)
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
                # Convert to Eastern time
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(eastern)
                data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} bars")
            else:
                logger.warning(f"  {symbol}: No data")
        else:
            logger.warning(f"  {symbol}: Not in response")

    # Download VIX separately from yfinance
    if 'VIX' in symbols:
        try:
            vix_data = download_vix_data(date)
            data['VIX'] = vix_data
        except Exception as e:
            logger.error(f"Failed to download VIX: {e}")

    return data


def load_models():
    """Load trained regime detector and Bayesian model."""
    # Create and load Bayesian model
    bayesian_model = BayesianReversionModel()
    bayesian_model.load_model()

    logger.success(f"Loaded Bayesian model with {len(bayesian_model.regime_probabilities)} symbols")

    # Create regime detector (uses default parameters)
    regime_detector = MarketRegimeDetector()

    return regime_detector, bayesian_model


def simulate_signals(date: datetime = None):
    """
    Simulate signal generation for a given date.

    Args:
        date: Date to simulate (defaults to today)
    """
    if date is None:
        date = datetime.now()

    logger.blank()
    logger.separator()
    logger.header(f"SIMULATING OMR SIGNALS FOR {date.strftime('%Y-%m-%d')}")
    logger.separator()
    logger.blank()

    # Load models
    logger.info("Loading trained models...")
    regime_detector, bayesian_model = load_models()

    # Check what symbols the Bayesian model covers
    model_symbols = list(bayesian_model.regime_probabilities.keys())
    logger.info(f"Bayesian model covers {len(model_symbols)} symbols: {model_symbols}")
    logger.blank()

    # Get trading symbols (use the same as production)
    trading_symbols = ETFUniverse.LEVERAGED_3X
    logger.info(f"Trading universe: {len(trading_symbols)} symbols")

    # All symbols we need (trading + SPY/VIX for regime)
    all_symbols = list(set(trading_symbols + ['SPY', 'VIX']))

    # Download data
    logger.separator()
    logger.header("DOWNLOADING TODAY'S DATA")
    logger.separator()

    try:
        market_data = download_today_data(all_symbols, date)
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.success(f"Downloaded data for {len(market_data)} symbols")
    logger.blank()

    # Check data availability
    logger.separator()
    logger.header("DATA AVAILABILITY CHECK")
    logger.separator()

    for symbol in all_symbols:
        if symbol in market_data:
            df = market_data[symbol]
            has_open = not df.between_time(time(9, 30), time(9, 31)).empty
            has_close = not df.between_time(time(15, 50), time(15, 50)).empty

            status = "OK" if has_open and has_close else "MISSING"
            logger.info(f"  {symbol}: {len(df)} bars, open={has_open}, 3:50pm={has_close} [{status}]")
        else:
            logger.warning(f"  {symbol}: NO DATA")

    logger.blank()

    # Create signal generator with production settings
    logger.separator()
    logger.header("INITIALIZING SIGNAL GENERATOR")
    logger.separator()

    signal_generator = OvernightReversionSignals(
        regime_detector=regime_detector,
        bayesian_model=bayesian_model,
        symbols=trading_symbols,
        min_probability=0.58,  # Production threshold
        min_expected_return=0.002,  # 0.2% minimum
        max_positions=5,
        skip_bear_regime=True
    )

    logger.blank()

    # Simulate signal generation at 3:50 PM
    logger.separator()
    logger.header("GENERATING SIGNALS AT 3:50 PM")
    logger.separator()

    # Create timestamp at 3:50 PM
    import pytz
    eastern = pytz.timezone('US/Eastern')
    signal_time = eastern.localize(datetime.combine(date.date(), time(15, 50)))

    signals = signal_generator.generate_signals(market_data, signal_time)

    logger.blank()
    logger.separator()
    logger.header("SIMULATION RESULTS")
    logger.separator()
    logger.blank()

    if signals:
        logger.success(f"Generated {len(signals)} signals:")
        for sig in signals:
            logger.profit(
                f"  {sig['symbol']}: {sig['direction']} | "
                f"prob={sig['probability']:.1%} | "
                f"exp_ret={sig['expected_return']:.2%} | "
                f"strength={sig['signal_strength']:.2f}"
            )
    else:
        logger.warning("No signals generated!")
        logger.info("Check individual symbol evaluations above for rejection reasons")

    logger.blank()
    logger.separator()

    return signals


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simulate OMR signal generation")
    parser.add_argument('--date', type=str, default=None,
                       help="Date to simulate (YYYY-MM-DD), defaults to today")

    args = parser.parse_args()

    if args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        date = datetime.now()

    try:
        signals = simulate_signals(date)

        if signals:
            logger.success(f"[SUCCESS] Generated {len(signals)} signals")
        else:
            logger.warning("[RESULT] No signals generated for today")

        sys.exit(0)

    except Exception as e:
        logger.error(f"[FAILED] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
