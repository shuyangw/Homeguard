"""
Overnight Mean Reversion (OMR) Live Trading Adapter.

Connects OMR strategy to live trading infrastructure.
Runs at 3:50 PM EST to generate overnight signals.
"""

from typing import List, Dict, Optional
from datetime import datetime, time
import pandas as pd

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.universe import ETFUniverse
from src.trading.brokers.broker_interface import BrokerInterface
from src.utils.logger import logger


class OMRLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for Overnight Mean Reversion strategy.

    Generates signals at 3:50 PM EST based on:
    - Market regime
    - Intraday price movements
    - Bayesian reversion probabilities

    Positions are entered at 3:50 PM and exited next day at 9:31 AM.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: Optional[List[str]] = None,
        min_probability: float = 0.55,
        min_expected_return: float = 0.002,
        max_positions: int = 5,
        position_size: float = 0.1,
        regime_detector: Optional[MarketRegimeDetector] = None,
        bayesian_model: Optional[BayesianReversionModel] = None
    ):
        """
        Initialize OMR live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade (default: leveraged 3x ETFs)
            min_probability: Min win rate threshold (default: 0.55)
            min_expected_return: Min expected return threshold (default: 0.002)
            max_positions: Max concurrent positions (default: 5)
            position_size: Position size as fraction (default: 0.1)
            regime_detector: Trained regime detector (optional)
            bayesian_model: Trained Bayesian model (optional)
        """
        # Use default symbols if not specified
        if symbols is None:
            symbols = ETFUniverse.LEVERAGED_3X
            logger.info(f"Using default OMR universe: {len(symbols)} leveraged 3x ETFs")

        # Initialize regime detector if not provided
        if regime_detector is None:
            regime_detector = MarketRegimeDetector()
            logger.info("Created new MarketRegimeDetector (untrained)")

        # Initialize Bayesian model if not provided
        if bayesian_model is None:
            bayesian_model = BayesianReversionModel()
            logger.info("Created new BayesianReversionModel (untrained)")

        # Create pure OMR strategy with injected symbols
        strategy = OvernightReversionSignals(
            regime_detector=regime_detector,
            bayesian_model=bayesian_model,
            symbols=symbols,  # ✅ Inject symbols instead of using hardcoded list
            min_probability=min_probability,
            min_expected_return=min_expected_return,
            max_positions=max_positions
        )

        # OMR needs intraday data, so need more lookback
        data_lookback_days = 365

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=position_size,
            max_positions=max_positions,
            data_lookback_days=data_lookback_days
        )

        self.min_probability = min_probability
        self.min_expected_return = min_expected_return

        logger.info("OMR Strategy Configuration:")
        logger.info(f"  Min probability: {min_probability:.1%}")
        logger.info(f"  Min expected return: {min_expected_return:.2%}")
        logger.info(f"  Signal time: 3:50 PM EST")
        logger.info(f"  Entry: 3:50 PM | Exit: Next day 9:31 AM")

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday market data for OMR strategy.

        OMR needs intraday bars to calculate intraday moves.
        """
        try:
            import pandas as pd
            from datetime import timedelta

            market_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.data_lookback_days)

            # For OMR, we need intraday data (1-minute or 5-minute bars)
            # Fetch daily data for now (can be enhanced to intraday later)
            for symbol in self.symbols:
                try:
                    # Fetch daily data
                    df = self.broker.get_historical_bars(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe='1D'
                    )

                    if df is not None and not df.empty:
                        market_data[symbol] = df
                    else:
                        logger.warning(f"No data returned for {symbol}")

                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue

            # Also need SPY and VIX for regime detection
            for market_symbol in ['SPY', 'VIX']:
                if market_symbol not in market_data:
                    try:
                        df = self.broker.get_historical_bars(
                            symbol=market_symbol,
                            start=start_date,
                            end=end_date,
                            timeframe='1D'
                        )
                        if df is not None and not df.empty:
                            market_data[market_symbol] = df
                    except Exception as e:
                        logger.error(f"Error fetching {market_symbol}: {e}")

            logger.info(
                f"Fetched data for {len(market_data)} symbols "
                f"(including SPY/VIX for regime)"
            )
            return market_data

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}")
            return {}

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        Returns:
            Schedule dict (run at 3:50 PM EST specifically)
        """
        return {
            'specific_time': '15:50',  # 3:50 PM EST
            'market_hours_only': True
        }

    def close_overnight_positions(self) -> None:
        """
        Close overnight positions at market open (9:31 AM).

        Should be called at 9:31 AM to exit positions entered at 3:50 PM.
        """
        try:
            now = datetime.now()
            if now.time() < time(9, 30) or now.time() > time(9, 35):
                logger.warning(
                    f"close_overnight_positions called at {now.time()}, "
                    "expected 9:31 AM"
                )

            # Get all open positions
            positions = self.broker.get_positions()

            if not positions:
                logger.info("No overnight positions to close")
                return

            logger.info(f"Closing {len(positions)} overnight positions at market open")

            for position in positions:
                try:
                    # Calculate P&L
                    entry_price = float(position.avg_entry_price)
                    current_price = float(position.current_price)
                    qty = int(position.qty)

                    pnl = (current_price - entry_price) * qty
                    pnl_pct = (current_price - entry_price) / entry_price * 100

                    logger.info(
                        f"Closing {position.symbol}: {qty} shares "
                        f"@ ${entry_price:.2f} → ${current_price:.2f} "
                        f"(P&L: ${pnl:+.2f}, {pnl_pct:+.2f}%)"
                    )

                    # Place market order to close
                    side = 'sell' if qty > 0 else 'buy'
                    order = self.execution_engine.place_market_order(
                        symbol=position.symbol,
                        qty=abs(qty),
                        side=side
                    )

                    if order:
                        logger.success(f"Close order placed: {order.id}")
                    else:
                        logger.error(f"Failed to close {position.symbol}")

                except Exception as e:
                    logger.error(f"Error closing {position.symbol}: {e}")
                    continue

            logger.info("Overnight position closing complete")

        except Exception as e:
            logger.error(f"Error in close_overnight_positions: {e}")


if __name__ == "__main__":
    logger.info("OMR (Overnight Mean Reversion) Live Trading Adapter")
    logger.info("=" * 60)
    logger.info("Generates signals at 3:50 PM EST based on:")
    logger.info("  - Market regime (bull/bear/choppy)")
    logger.info("  - Intraday price movements")
    logger.info("  - Bayesian reversion probabilities")
    logger.info("")
    logger.info("Entry: 3:50 PM EST")
    logger.info("Exit: Next day 9:31 AM EST")
    logger.info("")
    logger.info("Default universe: Leveraged 3x ETFs")
    logger.info("  (TQQQ, SQQQ, UPRO, SPXU, TMF, TMV, etc.)")
