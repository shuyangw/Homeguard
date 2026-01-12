"""
OpEx Pinning & Release Strategy.

Exploits gamma hedging dynamics near monthly options expiration:
1. PINNING phase (T-2 to T-1): Fade moves away from pin strike (mean reversion)
2. POST_OPEX phase (T+1 to T+2): Follow momentum after gamma release

Requirements:
- Positive GEX environment (stabilizing)
- Price distance from pin within acceptable range (0.5% to 3%)
- Appropriate OpEx phase (not OPEX_DAY or NORMAL)

Key Concepts:
- Dealers hedge short gamma by buying on rallies, selling on dips
- This creates "pinning" - prices gravitate toward high-GEX strikes
- After OpEx, gamma unwinding creates momentum opportunities
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime
import pandas as pd
import numpy as np

from src.backtesting.base.strategy import BaseStrategy
from src.strategies.opex.calendar import OpExCalendar, OpExPhase
from src.strategies.opex.signal_generator import (
    OpExSignalGenerator,
    SignalConfig,
    SignalDirection,
    OpExSignal,
)
from src.data.options.options_schema import OptionsChain, DailyGEXSummary
from src.utils.logger import logger


class OpExPinningStrategy(BaseStrategy):
    """
    OpEx Pinning & Release Strategy for backtesting and live trading.

    Strategy Logic:
    - PINNING Phase: Fade moves away from pin strike (expect reversion)
    - POST_OPEX Phase: Follow momentum (gamma released, breakout opportunity)
    - Skip: OPEX_DAY (high uncertainty), NORMAL (no edge), negative GEX

    Parameters:
        min_distance_pct: Minimum distance from pin (default 0.005 = 0.5%)
        max_distance_pct: Maximum distance from pin (default 0.03 = 3%)
        active_phases: Phases to trade (default: PINNING, POST_OPEX)
        holding_period: 'overnight' or 'eod' (default: overnight)
        position_size_pct: Position size as fraction of capital (default: 0.10)

    Usage:
        # For backtesting with pre-computed GEX data
        strategy = OpExPinningStrategy(params={
            'min_distance_pct': 0.005,
            'max_distance_pct': 0.03,
        })

        # Set GEX data for the backtest period
        strategy.set_gex_data(gex_summaries)

        # Generate signals
        entries, exits = strategy.generate_signals(price_data)
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the OpEx Pinning strategy.

        Args:
            params: Strategy parameters
        """
        params = params or {}

        # Extract parameters with defaults BEFORE super().__init__
        # because validate_parameters() is called by parent __init__
        self.min_distance_pct = params.get('min_distance_pct', 0.005)
        self.max_distance_pct = params.get('max_distance_pct', 0.03)
        self.holding_period = params.get('holding_period', 'overnight')
        self.position_size_pct = params.get('position_size_pct', 0.10)

        # Now call parent init (which calls validate_parameters)
        super().__init__(**params)

        # Parse active phases
        active_phase_names = params.get('active_phases', ['PINNING', 'POST_OPEX'])
        self.active_phases = tuple(
            OpExPhase(name) if isinstance(name, str) else name
            for name in active_phase_names
        )

        # Initialize components
        self.calendar = OpExCalendar()
        signal_config = SignalConfig(
            min_distance_pct=self.min_distance_pct,
            max_distance_pct=self.max_distance_pct,
            active_phases=self.active_phases,
        )
        self.signal_generator = OpExSignalGenerator(
            config=signal_config,
            calendar=self.calendar,
        )

        # GEX data storage (set externally for backtesting)
        self._gex_data: Dict[date, DailyGEXSummary] = {}

        # Track signals for debugging
        self._signal_history: List[OpExSignal] = []

        logger.info("Initialized OpEx Pinning Strategy")
        logger.info(f"  Min Distance: {self.min_distance_pct:.2%}")
        logger.info(f"  Max Distance: {self.max_distance_pct:.2%}")
        logger.info(f"  Active Phases: {[p.value for p in self.active_phases]}")

    def set_gex_data(self, gex_summaries: List[DailyGEXSummary]) -> None:
        """
        Set pre-computed GEX data for backtesting.

        Args:
            gex_summaries: List of daily GEX summaries
        """
        self._gex_data = {s.date: s for s in gex_summaries}
        logger.info(f"Loaded {len(self._gex_data)} GEX summaries")

    def generate_signals(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals from OHLCV data.

        Requires GEX data to be set via set_gex_data() or will use
        estimated pin strikes from price levels.

        Args:
            data: DataFrame with OHLCV columns and datetime index

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        # Initialize output Series
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Get unique dates
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index.normalize().unique()
        else:
            dates = pd.to_datetime(data.index).normalize().unique()

        # Track open positions for exit logic
        in_position = False
        entry_date = None

        for dt in dates:
            current_date = dt.date() if hasattr(dt, 'date') else dt

            # Get phase
            phase = self.calendar.get_phase(current_date)

            # Get GEX data or estimate
            gex_summary = self._gex_data.get(current_date)

            if gex_summary:
                pin_strike = gex_summary.pin_strike
                net_gex = gex_summary.total_gex
            else:
                # Estimate pin from round number near close
                day_data = data[data.index.date == current_date] if hasattr(data.index, 'date') else data.loc[current_date:current_date]
                if len(day_data) == 0:
                    continue
                close = day_data['close'].iloc[-1] if 'close' in day_data else day_data['Close'].iloc[-1]
                pin_strike = round(close / 5) * 5  # Round to nearest $5
                net_gex = 1_000_000  # Assume positive GEX

            # Get current price
            day_data = data[data.index.date == current_date] if hasattr(data.index, 'date') else data.loc[current_date:current_date]
            if len(day_data) == 0:
                continue
            close = day_data['close'].iloc[-1] if 'close' in day_data else day_data['Close'].iloc[-1]

            # Calculate distance to pin
            distance_to_pin = (close - pin_strike) / pin_strike if pin_strike > 0 else 0

            # Exit logic: exit after overnight hold or at EOD
            if in_position:
                if self.holding_period == 'overnight':
                    # Exit on next day's open
                    days_held = (current_date - entry_date).days if entry_date else 0
                    if days_held >= 1:
                        # Mark exit on first bar of the day
                        day_idx = day_data.index[0]
                        exits.loc[day_idx] = True
                        in_position = False
                        entry_date = None
                else:  # EOD exit
                    # Exit at end of day
                    day_idx = day_data.index[-1]
                    exits.loc[day_idx] = True
                    in_position = False
                    entry_date = None

            # Entry logic
            if not in_position:
                # Check skip conditions
                if phase not in self.active_phases:
                    continue

                if net_gex <= 0:
                    continue

                abs_distance = abs(distance_to_pin)
                if abs_distance < self.min_distance_pct or abs_distance > self.max_distance_pct:
                    continue

                # Generate signal based on phase
                direction = self._get_signal_direction(phase, distance_to_pin)

                if direction != SignalDirection.NEUTRAL:
                    # For long-only backtest, only take LONG signals
                    if direction == SignalDirection.LONG:
                        day_idx = day_data.index[-1]  # Entry at close
                        entries.loc[day_idx] = True
                        in_position = True
                        entry_date = current_date

        return entries, exits

    def _get_signal_direction(
        self,
        phase: OpExPhase,
        distance_to_pin: float,
    ) -> SignalDirection:
        """
        Determine signal direction based on phase and distance.

        Args:
            phase: Current OpEx phase
            distance_to_pin: Signed distance (positive = above pin)

        Returns:
            Signal direction
        """
        if phase == OpExPhase.PINNING:
            # Fade moves - go opposite direction
            if distance_to_pin > 0:
                return SignalDirection.SHORT  # Price above pin, fade by shorting
            else:
                return SignalDirection.LONG   # Price below pin, fade by going long

        elif phase == OpExPhase.POST_OPEX:
            # Follow momentum
            if distance_to_pin > 0:
                return SignalDirection.LONG   # Price above pin, momentum long
            else:
                return SignalDirection.SHORT  # Price below pin, momentum short

        return SignalDirection.NEUTRAL

    def generate_signals_with_options(
        self,
        price_data: pd.DataFrame,
        options_chains: Dict[date, OptionsChain],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals using actual options chain data.

        This is the preferred method when options data is available.

        Args:
            price_data: OHLCV DataFrame
            options_chains: Dict of date -> OptionsChain

        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        # Compute GEX summaries from options chains
        from src.strategies.opex.gex_calculator import GEXCalculator
        gex_calc = GEXCalculator()

        gex_summaries = []
        for chain_date, chain in options_chains.items():
            summary = gex_calc.calculate_daily_summary(chain)
            gex_summaries.append(summary)

        # Set GEX data and generate signals
        self.set_gex_data(gex_summaries)
        return self.generate_signals(price_data)

    def get_signal_history(self) -> List[OpExSignal]:
        """Get recorded signal history for analysis."""
        return self._signal_history.copy()

    def validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.min_distance_pct < 0 or self.min_distance_pct > 1:
            raise ValueError(f"min_distance_pct must be 0-1, got {self.min_distance_pct}")

        if self.max_distance_pct < 0 or self.max_distance_pct > 1:
            raise ValueError(f"max_distance_pct must be 0-1, got {self.max_distance_pct}")

        if self.min_distance_pct >= self.max_distance_pct:
            raise ValueError("min_distance_pct must be less than max_distance_pct")

        if self.holding_period not in ('overnight', 'eod'):
            raise ValueError(f"holding_period must be 'overnight' or 'eod', got {self.holding_period}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'min_distance_pct': self.min_distance_pct,
            'max_distance_pct': self.max_distance_pct,
            'active_phases': [p.value for p in self.active_phases],
            'holding_period': self.holding_period,
            'position_size_pct': self.position_size_pct,
        }
