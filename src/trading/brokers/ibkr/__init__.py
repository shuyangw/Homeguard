"""
IBKR Integration Module for Homeguard.

Provides data download, live streaming, and order execution via Interactive
Brokers through the ib_async library.

Public API:
    IBKRBroker             - BrokerInterface + OptionsTradingInterface
    IBKRDataProvider       - DataProviderInterface for historical data
    IBKRStreamingProvider  - StreamingProviderInterface for real-time data
    IBKRConnectionManager  - Managed connection lifecycle
    IBKRConfig             - Configuration model

Usage:
    from src.trading.brokers.ibkr import IBKRBroker, IBKRConfig

    config = IBKRConfig(port=4002)  # Paper trading gateway
    broker = IBKRBroker(config)
    broker.start()

    account = broker.get_account()
    positions = broker.get_stock_positions()
    chain = broker.get_options_chain('AAPL')

    broker.stop()

Dependencies:
    pip install ib_async  (no ibapi needed)
"""

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
from src.trading.brokers.ibkr.ibkr_futures_broker import IBKRFuturesBroker
from src.trading.brokers.ibkr.data_download import IBKRDataProvider
from src.trading.brokers.ibkr.streaming import IBKRStreamingProvider

__all__ = [
    "IBKRConfig",
    "IBKRConnectionManager",
    "IBKRBroker",
    "IBKRFuturesBroker",
    "IBKRDataProvider",
    "IBKRStreamingProvider",
]
