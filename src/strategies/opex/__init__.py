"""
OpEx (Options Expiration) Strategy Module.

Implements the OpEx Pinning & Release strategy based on gamma hedging dynamics.

Components:
- OpExCalendar: Monthly expiration dates and phase detection
- GEXCalculator: Gamma Exposure calculations and pin strike identification
- OpExSignalGenerator: Trading signal generation
- OpExBacktestRunner: Specialized backtest with options data integration
- OpExValidator: Statistical validation of strategy hypotheses

Key Concepts:
- GEX (Gamma Exposure): Measures dealer hedging pressure on stocks
- Pin Strike: The strike price that stock is "pulled" toward near expiration
- OpEx Phases: NORMAL -> PRE_OPEX -> PINNING -> OPEX_DAY -> POST_OPEX
"""

from src.strategies.opex.calendar import OpExCalendar, OpExPhase, OpExCycle
from src.strategies.opex.gex_calculator import GEXCalculator, GEXProfile
from src.strategies.opex.signal_generator import (
    OpExSignalGenerator,
    OpExSignal,
    SignalConfig,
    SignalDirection,
)
from src.strategies.opex.backtest import (
    OpExBacktestRunner,
    OpExBacktestResults,
    OpExTradeRecord,
)
from src.strategies.opex.validation import (
    OpExValidator,
    ValidationReport,
    PinningFrequencyReport,
    PhaseReturnsReport,
    GEXPredictiveReport,
)

__all__ = [
    # Calendar
    "OpExCalendar",
    "OpExPhase",
    "OpExCycle",
    # GEX
    "GEXCalculator",
    "GEXProfile",
    # Signals
    "OpExSignalGenerator",
    "OpExSignal",
    "SignalConfig",
    "SignalDirection",
    # Backtest
    "OpExBacktestRunner",
    "OpExBacktestResults",
    "OpExTradeRecord",
    # Validation
    "OpExValidator",
    "ValidationReport",
    "PinningFrequencyReport",
    "PhaseReturnsReport",
    "GEXPredictiveReport",
]
