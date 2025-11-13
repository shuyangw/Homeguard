"""
Homeguard Trading Module

Broker-agnostic paper trading system for executing trading strategies.

Architecture:
- brokers/: Broker abstraction layer (BrokerInterface, implementations)
- core/: Core trading logic (ExecutionEngine, PaperTradingBot, PositionManager)
- schedulers/: Market and strategy timing
- monitoring/: Logging, performance tracking, alerts
- utils/: Trading utilities

Design Principles:
- Dependency Inversion: Core depends on BrokerInterface, not concrete brokers
- Dependency Injection: Components receive broker via constructor
- Factory Pattern: BrokerFactory creates broker instances from config
"""

__version__ = "2.0.0"
__author__ = "Homeguard Trading Team"

from .brokers.broker_factory import BrokerFactory
from .brokers.broker_interface import BrokerInterface

__all__ = [
    "BrokerFactory",
    "BrokerInterface",
]
