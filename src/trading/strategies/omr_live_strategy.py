"""
OMR (Overnight Mean Reversion) Live Trading Strategy

Adapts the backtested OMR strategy for live trading with the broker abstraction layer.
Generates trading signals at 3:50 PM EST and exits positions at 9:31 AM EST next day.

This is a broker-agnostic implementation that works with any BrokerInterface.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import pandas as pd

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.trading.brokers.broker_interface import BrokerInterface, OrderSide, OrderType
from src.utils.logger import get_logger

logger = get_logger()  # Use global logger (no file creation)


class OMRLiveStrategy:
    """
    Live trading adapter for Overnight Mean Reversion strategy.

    Uses trained models to generate signals and integrates with broker for execution.
    Completely broker-agnostic via BrokerInterface.
    """

    def __init__(self, config: Dict):
        """
        Initialize OMR live trading strategy.

        Args:
            config: Strategy configuration dict with:
                - min_win_rate: Minimum win rate for trades (default 0.58)
                - min_expected_return: Minimum expected return (default 0.002)
                - min_sample_size: Minimum historical samples (default 15)
                - skip_regimes: List of regimes to skip (e.g., ['BEAR'])
                - symbols: List of symbols to trade
                - vix_threshold: Max VIX for trading (default 35)
        """
        self.config = config

        # Extract parameters
        self.min_win_rate = config.get('min_win_rate', 0.58)
        self.min_expected_return = config.get('min_expected_return', 0.002)
        self.min_sample_size = config.get('min_sample_size', 15)
        self.skip_regimes = config.get('skip_regimes', ['BEAR'])
        self.symbols = config.get('symbols', [])
        self.vix_threshold = config.get('vix_threshold', 35)

        # Initialize components (will be trained before trading)
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.bayesian_model: Optional[BayesianReversionModel] = None
        self.signal_generator: Optional[OvernightReversionSignals] = None

        self.is_trained = False

        logger.info("Initialized OMR Live Strategy")
        logger.info(f"  Symbols: {len(self.symbols)}")
        logger.info(f"  Min Win Rate: {self.min_win_rate:.1%}")
        logger.info(f"  Min Expected Return: {self.min_expected_return:.2%}")
        logger.info(f"  VIX Threshold: {self.vix_threshold}")

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train the strategy models with historical data.

        Args:
            historical_data: Dict of symbol -> DataFrame with historical data
                Must include 'SPY' and 'VIX' for regime detection
                Should include leveraged ETFs for Bayesian model training
        """
        logger.info("Training OMR strategy models...")

        # Validate required data
        if 'SPY' not in historical_data or 'VIX' not in historical_data:
            raise ValueError("Historical data must include SPY and VIX for regime detection")

        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.bayesian_model = BayesianReversionModel(data_frequency='daily')

        # Prepare daily data for regime detection
        spy_daily = self._resample_to_daily(historical_data['SPY'])
        vix_daily = self._resample_to_daily(historical_data['VIX'])

        # Train Bayesian model (requires regime detector for labeling)
        logger.info("Training Bayesian probability model...")
        self.bayesian_model.train(
            historical_data=historical_data,
            regime_detector=self.regime_detector,
            spy_data=spy_daily,
            vix_data=vix_daily
        )

        # Initialize signal generator
        logger.info("Initializing signal generator...")
        self.signal_generator = OvernightReversionSignals(
            regime_detector=self.regime_detector,
            bayesian_model=self.bayesian_model,
            min_probability=self.min_win_rate,
            min_expected_return=self.min_expected_return,
            max_positions=5
        )

        self.is_trained = True
        logger.success("OMR strategy models trained successfully")

    def generate_entry_signals(
        self,
        current_data: Dict[str, pd.DataFrame],
        broker: BrokerInterface
    ) -> List[Dict]:
        """
        Generate entry signals at 3:50 PM EST.

        Args:
            current_data: Dict of symbol -> recent DataFrame with price data
            broker: Broker interface for fetching latest quotes

        Returns:
            List of signal dicts with:
                - symbol: Stock symbol
                - side: BUY or SHORT based on signal direction
                - probability: Predicted win rate
                - expected_return: Expected return
                - regime: Current market regime
                - entry_price: Suggested entry price
        """
        if not self.is_trained:
            raise ValueError("Strategy not trained. Call train() first.")

        logger.info("Generating entry signals...")

        # Check VIX threshold (3 fallback sources for resilience)
        current_vix = None
        vix_source = None

        # Source 1: Pre-fetched VIX from current_data
        try:
            if 'VIX' in current_data and not current_data['VIX'].empty:
                vix_df = current_data['VIX']
                current_vix = float(vix_df['close'].iloc[-1])
                vix_source = "current_data cache"
        except Exception as e:
            logger.warning(f"VIX source 1 (cache) failed: {e}")

        # Source 2: yfinance Ticker.info API
        if current_vix is None:
            try:
                import yfinance as yf
                vix_ticker = yf.Ticker('^VIX')
                vix_info = vix_ticker.info
                current_vix = vix_info.get('regularMarketPrice') or vix_info.get('previousClose')
                if current_vix:
                    current_vix = float(current_vix)
                    vix_source = "yfinance Ticker.info"
            except Exception as e:
                logger.warning(f"VIX source 2 (Ticker.info) failed: {e}")

        # Source 3: yfinance download API (different endpoint)
        if current_vix is None:
            try:
                import yfinance as yf
                vix_df = yf.download('^VIX', period='1d', progress=False)
                if not vix_df.empty:
                    # Handle multi-level columns from yfinance
                    if hasattr(vix_df.columns, 'levels'):
                        close_col = [c for c in vix_df.columns if c[0].lower() == 'close'][0]
                        current_vix = float(vix_df[close_col].iloc[-1])
                    else:
                        current_vix = float(vix_df['Close'].iloc[-1])
                    vix_source = "yfinance download"
            except Exception as e:
                logger.warning(f"VIX source 3 (download) failed: {e}")

        # Evaluate VIX result
        if current_vix is None:
            logger.error("All 3 VIX sources failed - blocking trading for safety")
            return []

        logger.info(f"VIX = {current_vix:.2f} (source: {vix_source})")

        if current_vix > self.vix_threshold:
            logger.warning(f"VIX ({current_vix:.2f}) exceeds threshold ({self.vix_threshold}). No trading.")
            return []

        # Generate signals using signal generator
        # This returns all signals already filtered and ranked
        timestamp = datetime.now()
        signals = self.signal_generator.generate_signals(current_data, timestamp)

        if not signals:
            logger.info("No signals generated")
            return []

        # Filter by regime if needed
        filtered_signals = []
        for signal in signals:
            if signal['regime'] in self.skip_regimes:
                logger.info(f"Skipping {signal['symbol']} due to {signal['regime']} regime")
                continue

            # Convert to broker interface format
            try:
                quote = broker.get_latest_quote(signal['symbol'])

                # Determine side and entry price based on signal direction
                if signal['direction'] == 'BUY':
                    signal['side'] = OrderSide.BUY
                    signal['entry_price'] = quote['ask']
                else:  # SHORT
                    signal['side'] = OrderSide.SELL
                    signal['entry_price'] = quote['bid']

                filtered_signals.append(signal)

                logger.info(
                    f"{signal['symbol']}: {signal['direction']} | "
                    f"Win rate={signal['probability']:.1%}, "
                    f"Expected return={signal['expected_return']:.2%}, "
                    f"Entry price=${signal['entry_price']:.2f}"
                )

            except Exception as e:
                logger.error(f"Failed to get quote for {signal['symbol']}: {e}")
                continue

        logger.success(f"Generated {len(filtered_signals)} entry signals")
        return filtered_signals

    def generate_exit_signals(
        self,
        broker: BrokerInterface
    ) -> List[Dict]:
        """
        Generate exit signals at 9:31 AM EST.

        For OMR strategy, we simply exit all open positions at market open.

        Args:
            broker: Broker interface

        Returns:
            List of exit signal dicts with:
                - symbol: Stock symbol
                - side: Always SELL for OMR
                - quantity: Number of shares to sell
                - exit_price: Current ask price
        """
        logger.info("Generating exit signals...")

        positions = broker.get_positions()
        if not positions:
            logger.info("No positions to exit")
            return []

        exit_signals = []
        for position in positions:
            try:
                symbol = position['symbol']
                quantity = abs(position['quantity'])

                # Get current quote
                quote = broker.get_latest_quote(symbol)
                exit_price = quote['bid']  # Use bid price for selling

                exit_signals.append({
                    'symbol': symbol,
                    'side': OrderSide.SELL,
                    'quantity': quantity,
                    'exit_price': exit_price,
                    'entry_price': position.get('avg_entry_price'),
                })

                logger.info(f"{symbol}: Exiting {quantity} shares @ ${exit_price:.2f}")

            except Exception as e:
                logger.error(f"Failed to generate exit signal for {position['symbol']}: {e}")
                continue

        logger.success(f"Generated {len(exit_signals)} exit signals")
        return exit_signals

    def _detect_current_regime(
        self,
        current_data: Dict[str, pd.DataFrame]
    ) -> Tuple[str, float]:
        """
        Detect current market regime.

        Note: This method is kept for debugging/manual checks.
        Signal generation uses regime detection internally.

        Args:
            current_data: Dict of symbol -> recent price data

        Returns:
            Tuple of (regime_name, confidence)
        """
        if not self.regime_detector:
            return 'SIDEWAYS', 0.5

        if 'SPY' not in current_data or 'VIX' not in current_data:
            logger.warning("Missing SPY or VIX data for regime detection")
            return 'SIDEWAYS', 0.5

        try:
            spy_daily = self._resample_to_daily(current_data['SPY'])
            vix_daily = self._resample_to_daily(current_data['VIX'])

            if len(spy_daily) < 200 or len(vix_daily) < 252:
                logger.warning("Insufficient data for regime detection")
                return 'SIDEWAYS', 0.5

            # Classify regime using regime detector
            regime, confidence = self.regime_detector.classify_regime(
                spy_data=spy_daily,
                vix_data=vix_daily,
                timestamp=datetime.now()
            )

            return regime, confidence

        except Exception as e:
            logger.error(f"Failed to detect regime: {e}")
            return 'SIDEWAYS', 0.5

    @staticmethod
    def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Resample minute data to daily."""
        if 'Close' in df.columns:
            daily = df.resample('1D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            # Already daily data
            daily = df.copy()

        return daily
