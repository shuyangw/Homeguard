"""
Demo Trading Module - Custom paper trading with Binance streaming.

Provides a complete demo/paper trading system using real-time Binance
data with simulated execution, replacing Alpaca's limited paper trading
(which requires integer quantities).

Key Components:
    DemoBroker - Main broker implementing CryptoTradingInterface
    DemoPortfolioState - Persistent portfolio state (JSON)
    ExecutionSimulator - Slippage and fee simulation
    DemoTradeLogger - Trade logging with portfolio snapshots

Example:
    from src.trading.demo import DemoBroker

    broker = DemoBroker(initial_cash=10000.0)
    broker.start_streaming(['BTC/USD', 'ETH/USD', 'MKR/USD'])

    # Place orders with fractional quantities
    order = broker.place_crypto_order(
        symbol='BTC/USD',
        quantity=Decimal('0.001'),
        side=OrderSide.BUY
    )

    # Portfolio persists across restarts
    broker.stop_streaming()
"""

from src.trading.demo.portfolio_state import DemoPortfolioState, DemoPosition
from src.trading.demo.execution_simulator import ExecutionSimulator, ExecutionResult
from src.trading.demo.trade_logger import DemoTradeLogger
from src.trading.demo.demo_broker import DemoBroker

__all__ = [
    "DemoBroker",
    "DemoPortfolioState",
    "DemoPosition",
    "ExecutionSimulator",
    "ExecutionResult",
    "DemoTradeLogger",
]
