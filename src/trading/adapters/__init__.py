"""
Live Trading Adapters.

Adapters that connect pure strategy implementations to the live trading infrastructure.

Each adapter:
1. Extends StrategyAdapter base class
2. Wraps a pure StrategySignals implementation
3. Handles data fetching, signal generation, and order execution
4. Manages scheduling and position lifecycle

Usage:
    ```python
    from src.trading.adapters import MACrossoverLiveAdapter
    from src.trading.brokers import AlpacaBroker

    # Create broker
    broker = AlpacaBroker('paper')

    # Create live trading adapter
    adapter = MACrossoverLiveAdapter(
        broker=broker,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        fast_period=50,
        slow_period=200,
        position_size=0.1,
        max_positions=5
    )

    # Run strategy once
    adapter.run_once()

    # Or schedule to run automatically
    while True:
        if adapter.should_run_now():
            adapter.run_once()
        time.sleep(60)  # Check every minute
    ```
"""

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.trading.adapters.ma_live_adapter import (
    MACrossoverLiveAdapter,
    TripleMACrossoverLiveAdapter
)
from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
from src.trading.adapters.momentum_live_adapter import MomentumLiveAdapter

__all__ = [
    'StrategyAdapter',
    'MACrossoverLiveAdapter',
    'TripleMACrossoverLiveAdapter',
    'OMRLiveAdapter',
    'MomentumLiveAdapter'
]
