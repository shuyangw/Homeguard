"""
IBKR Integration Module for Homeguard.

Provides data download, live streaming, and order execution via Interactive
Brokers through the ib_async library.

Public API:
    IBKRConfig             - Configuration model

Usage:
    from src.trading.brokers.ibkr import IBKRConfig

    config = IBKRConfig(port=4002)  # Paper trading gateway

Dependencies:
    pip install ib_async  (no ibapi needed)
"""

from src.trading.brokers.ibkr.config import IBKRConfig

__all__ = [
    "IBKRConfig",
]
