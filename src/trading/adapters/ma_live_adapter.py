"""
MA Crossover Live Trading Adapter.

Connects MA crossover pure strategies to live trading infrastructure.
"""

from typing import List, Dict
from datetime import datetime

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.strategies.implementations import MACrossoverSignals, TripleMACrossoverSignals
from src.trading.brokers.broker_interface import BrokerInterface
from src.utils.logger import logger


class MACrossoverLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for MA Crossover strategy.

    Runs every 5 minutes during market hours to check for MA crossovers.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: List[str],
        fast_period: int = 50,
        slow_period: int = 200,
        ma_type: str = 'sma',
        min_confidence: float = 0.7,
        position_size: float = 0.1,
        max_positions: int = 5
    ):
        """
        Initialize MA Crossover live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade
            fast_period: Fast MA period (default: 50)
            slow_period: Slow MA period (default: 200)
            ma_type: MA type ('sma' or 'ema', default: 'sma')
            min_confidence: Minimum confidence threshold (default: 0.7)
            position_size: Position size as fraction (default: 0.1)
            max_positions: Max concurrent positions (default: 5)
        """
        # Create pure strategy
        strategy = MACrossoverSignals(
            fast_period=fast_period,
            slow_period=slow_period,
            ma_type=ma_type,
            min_confidence=min_confidence
        )

        # Calculate data lookback needed
        data_lookback_days = slow_period * 2  # 2x for safety

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=position_size,
            max_positions=max_positions,
            data_lookback_days=data_lookback_days
        )

        logger.info(f"MA Crossover: {fast_period}/{slow_period} {ma_type.upper()}")

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        Returns:
            Schedule dict (run every 5 minutes during market hours)
        """
        return {
            'interval': '5min',
            'market_hours_only': True
        }


class TripleMACrossoverLiveAdapter(StrategyAdapter):
    """
    Live trading adapter for Triple MA Crossover strategy.

    Runs every 5 minutes during market hours to check for triple MA alignment.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: List[str],
        fast_period: int = 10,
        medium_period: int = 20,
        slow_period: int = 50,
        ma_type: str = 'ema',
        min_confidence: float = 0.75,
        position_size: float = 0.1,
        max_positions: int = 5
    ):
        """
        Initialize Triple MA Crossover live adapter.

        Args:
            broker: Broker interface
            symbols: List of symbols to trade
            fast_period: Fast MA period (default: 10)
            medium_period: Medium MA period (default: 20)
            slow_period: Slow MA period (default: 50)
            ma_type: MA type (default: 'ema')
            min_confidence: Minimum confidence threshold (default: 0.75)
            position_size: Position size as fraction (default: 0.1)
            max_positions: Max concurrent positions (default: 5)
        """
        # Create pure strategy
        strategy = TripleMACrossoverSignals(
            fast_period=fast_period,
            medium_period=medium_period,
            slow_period=slow_period,
            ma_type=ma_type,
            min_confidence=min_confidence
        )

        # Calculate data lookback needed
        data_lookback_days = slow_period * 2

        # Initialize base adapter
        super().__init__(
            strategy=strategy,
            broker=broker,
            symbols=symbols,
            position_size=position_size,
            max_positions=max_positions,
            data_lookback_days=data_lookback_days
        )

        logger.info(
            f"Triple MA: {fast_period}/{medium_period}/{slow_period} {ma_type.upper()}"
        )

    def get_schedule(self) -> Dict[str, any]:
        """
        Get scheduling configuration.

        Returns:
            Schedule dict (run every 5 minutes during market hours)
        """
        return {
            'interval': '5min',
            'market_hours_only': True
        }


if __name__ == "__main__":
    logger.info("MA Crossover Live Trading Adapters")
    logger.info("=" * 60)
    logger.info("MACrossoverLiveAdapter:")
    logger.info("  - 50/200 SMA golden cross by default")
    logger.info("  - Runs every 5 minutes during market hours")
    logger.info("")
    logger.info("TripleMACrossoverLiveAdapter:")
    logger.info("  - 10/20/50 EMA alignment by default")
    logger.info("  - Runs every 5 minutes during market hours")
