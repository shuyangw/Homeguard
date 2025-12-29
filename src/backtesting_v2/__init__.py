"""
Backtesting Framework v2 - Standalone enhanced backtesting with trade enrichment.

This module provides an enhanced backtesting framework that:
- Maintains complete parity with the original backtesting engine
- Adds per-trade enrichment via adapter pattern
- Tracks bar-by-bar portfolio state
- Supports minute-level stop simulation
- Works alongside original framework as fallback

Usage:
    from src.backtesting_v2.engine import BacktestEngineV2, PortfolioV2
    from src.backtesting_v2.base import BaseStrategyV2

    engine = BacktestEngineV2(initial_capital=100000)
    portfolio = engine.run(strategy, symbols, start, end)

    # New capabilities
    portfolio.trades_df        # Enriched trades with custom columns
    portfolio.portfolio_state  # Bar-by-bar state tracking

    # Strategy adapters
    from src.backtesting_v2.adapters import OMRStrategyAdapter, RAMPStrategyAdapter
"""

from src.backtesting_v2.base.strategy import BaseStrategyV2
from src.backtesting_v2.engine.trade_schema import (
    BASE_TRADE_COLUMNS,
    MULTI_ASSET_COLUMNS,
    STATE_COLUMNS,
    TradeType,
    ExitReason,
)
from src.backtesting_v2.engine import (
    BacktestEngineV2,
    PortfolioV2,
    from_signals_v2,
    TradeLoggerV2,
)

__all__ = [
    # Base
    "BaseStrategyV2",
    # Schema
    "BASE_TRADE_COLUMNS",
    "MULTI_ASSET_COLUMNS",
    "STATE_COLUMNS",
    "TradeType",
    "ExitReason",
    # Engine
    "BacktestEngineV2",
    "PortfolioV2",
    "from_signals_v2",
    "TradeLoggerV2",
]
