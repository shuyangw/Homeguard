"""
OMR Strategy Integration Test

Tests the complete integration of:
- MarketRegimeDetector
- BayesianReversionModel
- OvernightReversionSignals
- OMRLiveStrategy
- PaperTradingBot

Validates all components work together correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.trading.strategies.omr_live_strategy import OMRLiveStrategy
from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_sample_data():
    """
    Load sample historical data for testing.

    Returns:
        Dict of symbol -> DataFrame with historical data
    """
    logger.info("Loading sample historical data...")

    data_dir = project_root / "data" / "leveraged_etfs"

    # Try to load existing parquet files
    historical_data = {}

    symbols_to_load = ['SPY', 'TQQQ', 'SQQQ', 'UPRO']

    for symbol in symbols_to_load:
        parquet_file = data_dir / f"{symbol}_1d.parquet"

        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                # Ensure lowercase column names
                df.columns = [col.lower() for col in df.columns]

                # Take last 2 years of data
                end_date = df.index.max()
                start_date = end_date - timedelta(days=730)
                df = df[df.index >= start_date]

                historical_data[symbol] = df
                logger.info(f"  Loaded {symbol}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

            except Exception as e:
                logger.warning(f"  Failed to load {symbol}: {e}")
        else:
            logger.warning(f"  {symbol} data file not found: {parquet_file}")

    # Try to load VIX, or create mock from SPY
    vix_file = data_dir / "VIX_1d.parquet"
    if vix_file.exists():
        try:
            vix_df = pd.read_parquet(vix_file)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [col[0] for col in vix_df.columns]
            vix_df.columns = [col.lower() for col in vix_df.columns]

            # Take last 2 years
            end_date = vix_df.index.max()
            start_date = end_date - timedelta(days=730)
            vix_df = vix_df[vix_df.index >= start_date]

            historical_data['VIX'] = vix_df
            logger.info(f"  Loaded VIX: {len(vix_df)} days")
        except Exception as e:
            logger.warning(f"  Failed to load VIX: {e}")

    # If VIX not loaded and SPY exists, create mock VIX from SPY volatility
    if 'VIX' not in historical_data and 'SPY' in historical_data:
        logger.warning("  VIX data not found - creating mock VIX from SPY volatility")
        spy_df = historical_data['SPY']

        # Calculate rolling volatility as proxy for VIX
        returns = spy_df['close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized vol as %

        # Create VIX-like dataframe
        vix_df = pd.DataFrame({
            'open': rolling_vol,
            'high': rolling_vol * 1.1,
            'low': rolling_vol * 0.9,
            'close': rolling_vol,
            'volume': 0
        }, index=spy_df.index)

        # Fill NaN with default value
        vix_df = vix_df.fillna(15.0)

        historical_data['VIX'] = vix_df
        logger.info(f"  Created mock VIX: {len(vix_df)} days")

    if not historical_data:
        logger.error("No historical data could be loaded!")
        logger.info("\nTo run this test, you need historical data files:")
        logger.info("  Run: python backtest_scripts/download_etf_universe.py")
        return None

    if 'SPY' not in historical_data or 'VIX' not in historical_data:
        logger.error("Missing required SPY or VIX data!")
        return None

    return historical_data


def test_strategy_initialization():
    """Test 1: Strategy initialization."""
    logger.header("\nTest 1: Strategy Initialization")

    try:
        config = {
            'min_win_rate': 0.55,
            'min_expected_return': 0.002,
            'min_sample_size': 15,
            'skip_regimes': ['BEAR'],
            'symbols': ['TQQQ', 'SQQQ', 'UPRO'],
            'vix_threshold': 35
        }

        strategy = OMRLiveStrategy(config)

        logger.success("[PASS] Strategy initialized successfully")
        logger.info(f"  Min win rate: {strategy.min_win_rate:.1%}")
        logger.info(f"  Min expected return: {strategy.min_expected_return:.2%}")
        logger.info(f"  Symbols: {strategy.symbols}")

        return True, strategy

    except Exception as e:
        logger.error(f"[FAIL] Strategy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_strategy_training(strategy, historical_data):
    """Test 2: Strategy training."""
    logger.header("\nTest 2: Strategy Training")

    try:
        logger.info("Training strategy with historical data...")
        logger.info(f"  Data symbols: {list(historical_data.keys())}")
        logger.info(f"  SPY data: {len(historical_data['SPY'])} days")
        logger.info(f"  VIX data: {len(historical_data['VIX'])} days")

        # This will train the Bayesian model and regime detector
        strategy.train(historical_data)

        logger.success("[PASS] Strategy training completed")
        logger.info(f"  Strategy trained: {strategy.is_trained}")
        logger.info(f"  Regime detector initialized: {strategy.regime_detector is not None}")
        logger.info(f"  Bayesian model initialized: {strategy.bayesian_model is not None}")
        logger.info(f"  Signal generator initialized: {strategy.signal_generator is not None}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Strategy training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regime_detection(strategy, historical_data):
    """Test 3: Regime detection."""
    logger.header("\nTest 3: Regime Detection")

    try:
        # Test regime detection with current data
        regime, confidence = strategy._detect_current_regime(historical_data)

        logger.success("[PASS] Regime detection successful")
        logger.info(f"  Current regime: {regime}")
        logger.info(f"  Confidence: {confidence:.1%}")

        # Get regime parameters
        regime_params = strategy.regime_detector.get_regime_parameters(regime)
        logger.info(f"  Regime parameters:")
        for key, value in regime_params.items():
            logger.info(f"    {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Regime detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_broker_connection():
    """Test 4: Broker connection."""
    logger.header("\nTest 4: Broker Connection")

    try:
        config_path = project_root / "config" / "trading" / "broker_alpaca.yaml"
        broker = BrokerFactory.create_from_yaml(str(config_path))

        if broker.test_connection():
            logger.success("[PASS] Broker connection successful")

            account = broker.get_account()
            logger.info(f"  Account ID: {account['account_id']}")
            logger.info(f"  Portfolio value: ${account['portfolio_value']:,.2f}")

            return True, broker
        else:
            logger.error("[FAIL] Broker connection failed")
            return False, None

    except Exception as e:
        logger.error(f"[FAIL] Broker connection error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_signal_generation(strategy, broker, historical_data):
    """Test 5: Signal generation."""
    logger.header("\nTest 5: Signal Generation")

    try:
        logger.info("Generating entry signals...")

        # Generate signals (note: may not work perfectly with daily data)
        signals = strategy.generate_entry_signals(historical_data, broker)

        logger.success(f"[PASS] Signal generation completed")
        logger.info(f"  Signals generated: {len(signals)}")

        if signals:
            logger.info("\n  Top signals:")
            for i, signal in enumerate(signals[:3], 1):
                logger.info(f"    {i}. {signal['symbol']}: {signal.get('direction', 'BUY')}")
                logger.info(f"       Probability: {signal.get('probability', 0):.1%}")
                logger.info(f"       Expected return: {signal.get('expected_return', 0):.2%}")
                logger.info(f"       Entry price: ${signal.get('entry_price', 0):.2f}")
        else:
            logger.warning("  No signals generated (expected with daily data)")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.header("="*70)
    logger.header("OMR Strategy Integration Test")
    logger.header("="*70)
    logger.blank()

    results = []

    # Load sample data
    historical_data = load_sample_data()
    if historical_data is None:
        logger.error("\nCannot proceed without historical data")
        return False

    # Test 1: Initialize strategy
    passed, strategy = test_strategy_initialization()
    results.append(("Strategy Initialization", passed))
    if not passed:
        logger.error("\nCritical test failed. Stopping.")
        return False

    # Test 2: Train strategy
    passed = test_strategy_training(strategy, historical_data)
    results.append(("Strategy Training", passed))
    if not passed:
        logger.error("\nCritical test failed. Stopping.")
        return False

    # Test 3: Regime detection
    passed = test_regime_detection(strategy, historical_data)
    results.append(("Regime Detection", passed))

    # Test 4: Broker connection
    passed, broker = test_broker_connection()
    results.append(("Broker Connection", passed))
    if not passed:
        logger.warning("\nBroker tests skipped (no connection)")
        broker = None

    # Test 5: Signal generation (if broker available)
    if broker:
        passed = test_signal_generation(strategy, broker, historical_data)
        results.append(("Signal Generation", passed))

    # Summary
    logger.blank()
    logger.header("="*70)
    logger.header("Test Results Summary")
    logger.header("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"  {status} {test_name}")

    logger.blank()
    logger.header(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.success("\nALL TESTS PASSED!")
        logger.success("OMR strategy integration validated successfully")
        return True
    else:
        logger.warning(f"\n{total_count - passed_count} test(s) failed")
        logger.info("Note: Some failures expected with daily data (needs minute data)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
