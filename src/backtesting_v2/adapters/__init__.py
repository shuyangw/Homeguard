"""Strategy adapters for v2 backtesting framework."""

from src.backtesting_v2.adapters.omr_adapter import OMRStrategyAdapter, SimpleOMRAdapter
from src.backtesting_v2.adapters.ramp_adapter import RAMPStrategyAdapter

__all__ = ["OMRStrategyAdapter", "SimpleOMRAdapter", "RAMPStrategyAdapter"]
