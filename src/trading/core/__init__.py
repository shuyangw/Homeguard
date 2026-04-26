"""
Core Trading Components (Broker-Agnostic)

All components in this module depend on BrokerInterface, not concrete brokers.
This ensures the core trading logic works with any broker implementation.

Components:
- PositionManager: Track positions, calculate P&L, enforce risk limits
- ExecutionEngine: Order execution and management
"""

from .position_manager import PositionManager
from .execution_engine import ExecutionEngine, ExecutionStatus

__all__ = [
    "PositionManager",
    "ExecutionEngine",
    "ExecutionStatus",
]
