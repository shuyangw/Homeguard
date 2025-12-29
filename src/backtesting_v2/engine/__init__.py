"""Engine components for backtesting v2 framework."""

from src.backtesting_v2.engine.trade_schema import (
    BASE_TRADE_COLUMNS,
    MULTI_ASSET_COLUMNS,
    STATE_COLUMNS,
    TradeType,
    ExitReason,
)
from src.backtesting_v2.engine.portfolio_simulator import (
    PortfolioV2,
    from_signals_v2,
)
from src.backtesting_v2.engine.backtest_engine import (
    BacktestEngineV2,
    run_backtest_v2,
)
from src.backtesting_v2.engine.trade_logger import (
    TradeLoggerV2,
    export_trades_v2,
    export_analysis_v2,
)
from src.backtesting_v2.engine.intraday_stop_simulator import (
    IntradayStopSimulator,
    simulate_intraday_stops,
)

__all__ = [
    # Schema
    "BASE_TRADE_COLUMNS",
    "MULTI_ASSET_COLUMNS",
    "STATE_COLUMNS",
    "TradeType",
    "ExitReason",
    # Portfolio
    "PortfolioV2",
    "from_signals_v2",
    # Engine
    "BacktestEngineV2",
    "run_backtest_v2",
    # Logger
    "TradeLoggerV2",
    "export_trades_v2",
    "export_analysis_v2",
    # Stop simulation
    "IntradayStopSimulator",
    "simulate_intraday_stops",
]
