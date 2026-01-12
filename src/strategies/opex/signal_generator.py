"""
OpEx Signal Generator.

Generates trading signals based on OpEx phase and GEX dynamics.

Signal Logic:
- PINNING phase: Fade moves away from pin strike (mean reversion)
- POST_OPEX phase: Momentum in direction of prior gamma pressure release
- Skip on OPEX_DAY (high uncertainty) and NORMAL phase (no edge)

Requirements for signal:
1. Positive GEX (stabilizing environment)
2. Price distance from pin within acceptable range
3. Appropriate OpEx phase
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Optional, Tuple

from src.strategies.opex.calendar import OpExCalendar, OpExPhase
from src.strategies.opex.gex_calculator import GEXCalculator
from src.data.options.options_schema import OptionsChain, DailyGEXSummary


class SignalDirection(int, Enum):
    """Trading signal direction."""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1


@dataclass
class OpExSignal:
    """
    OpEx trading signal.

    Attributes:
        symbol: Trading symbol
        date: Signal date
        direction: Trading direction (1=long, -1=short, 0=neutral)
        strength: Signal strength (0.0 to 1.0)
        phase: Current OpEx phase
        pin_strike: Identified pin strike
        distance_to_pin: Percentage distance from spot to pin
        net_gex: Net gamma exposure
        rationale: Human-readable signal explanation
    """
    symbol: str
    date: date
    direction: SignalDirection
    strength: float
    phase: OpExPhase
    pin_strike: float
    distance_to_pin: float
    net_gex: float
    rationale: str

    @property
    def is_actionable(self) -> bool:
        """Whether signal is actionable (non-neutral)."""
        return self.direction != SignalDirection.NEUTRAL

    @property
    def direction_str(self) -> str:
        """Human-readable direction."""
        return {
            SignalDirection.LONG: "LONG",
            SignalDirection.SHORT: "SHORT",
            SignalDirection.NEUTRAL: "NEUTRAL",
        }[self.direction]


@dataclass
class SignalConfig:
    """
    Configuration for signal generation.

    Attributes:
        min_distance_pct: Minimum distance from pin to trade (default 0.5%)
        max_distance_pct: Maximum distance from pin to trade (default 3%)
        min_gex_threshold: Minimum net GEX for signal (0 = any positive)
        active_phases: Phases to generate signals in
        fade_on_pinning: Whether to fade moves in PINNING phase
        momentum_on_release: Whether to follow momentum in POST_OPEX
    """
    min_distance_pct: float = 0.005  # 0.5%
    max_distance_pct: float = 0.03   # 3%
    min_gex_threshold: float = 0.0   # Any positive GEX
    active_phases: Tuple[OpExPhase, ...] = (OpExPhase.PINNING, OpExPhase.POST_OPEX)
    fade_on_pinning: bool = True     # Fade moves away from pin
    momentum_on_release: bool = True  # Follow momentum after release


class OpExSignalGenerator:
    """
    Generates trading signals based on OpEx dynamics.

    Strategy Logic:
    1. PINNING Phase (T-2 to T-1):
       - Positive GEX = dealers absorb moves = mean reversion
       - If price > pin: SHORT (fade the move, expect reversion to pin)
       - If price < pin: LONG (fade the move, expect reversion to pin)

    2. POST_OPEX Phase (T+1 to T+2):
       - Gamma pressure released = momentum opportunity
       - Follow the direction of the move (breakout from pin)

    3. Skip Conditions:
       - OPEX_DAY: High uncertainty, avoid new positions
       - NORMAL/PRE_OPEX: No significant gamma effect yet
       - Negative GEX: Destabilizing, dealers amplify moves
       - Distance outside acceptable range

    Usage:
        generator = OpExSignalGenerator()
        signal = generator.generate_signal(chain, current_date)

        if signal.is_actionable:
            print(f"Signal: {signal.direction_str} {signal.symbol}")
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        calendar: Optional[OpExCalendar] = None,
        gex_calculator: Optional[GEXCalculator] = None,
    ):
        """
        Initialize signal generator.

        Args:
            config: Signal generation configuration
            calendar: OpEx calendar instance
            gex_calculator: GEX calculator instance
        """
        self.config = config or SignalConfig()
        self.calendar = calendar or OpExCalendar()
        self.gex_calc = gex_calculator or GEXCalculator()

    def generate_signal(
        self,
        chain: OptionsChain,
        current_date: Optional[date] = None,
        pre_computed_gex: Optional[DailyGEXSummary] = None,
    ) -> OpExSignal:
        """
        Generate trading signal for current conditions.

        Args:
            chain: Options chain with current data
            current_date: Date to generate signal for (defaults to chain date)
            pre_computed_gex: Pre-computed GEX summary (optional, saves recalc)

        Returns:
            OpExSignal with direction and metadata
        """
        if current_date is None:
            current_date = chain.date

        symbol = chain.symbol
        spot = chain.underlying_price

        # Get OpEx phase
        phase, opex_date, days = self.calendar.get_phase_info(current_date)

        # Calculate GEX if not provided
        if pre_computed_gex:
            net_gex = pre_computed_gex.total_gex
            pin_strike = pre_computed_gex.pin_strike
        else:
            net_gex = self.gex_calc.calculate_total_gex(chain)
            pin_strike = self.gex_calc.find_pin_strike(chain) or spot

        # Calculate distance to pin
        distance_to_pin = (spot - pin_strike) / pin_strike if pin_strike else 0.0

        # Default neutral signal
        direction = SignalDirection.NEUTRAL
        strength = 0.0
        rationale = ""

        # Check skip conditions
        skip_reason = self._check_skip_conditions(
            phase=phase,
            net_gex=net_gex,
            distance_to_pin=distance_to_pin,
        )

        if skip_reason:
            return OpExSignal(
                symbol=symbol,
                date=current_date,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                phase=phase,
                pin_strike=pin_strike,
                distance_to_pin=distance_to_pin,
                net_gex=net_gex,
                rationale=skip_reason,
            )

        # Generate signal based on phase
        if phase == OpExPhase.PINNING and self.config.fade_on_pinning:
            direction, strength, rationale = self._generate_pinning_signal(
                spot=spot,
                pin_strike=pin_strike,
                distance_to_pin=distance_to_pin,
                net_gex=net_gex,
            )

        elif phase == OpExPhase.POST_OPEX and self.config.momentum_on_release:
            direction, strength, rationale = self._generate_release_signal(
                spot=spot,
                pin_strike=pin_strike,
                distance_to_pin=distance_to_pin,
                net_gex=net_gex,
            )

        return OpExSignal(
            symbol=symbol,
            date=current_date,
            direction=direction,
            strength=strength,
            phase=phase,
            pin_strike=pin_strike,
            distance_to_pin=distance_to_pin,
            net_gex=net_gex,
            rationale=rationale,
        )

    def _check_skip_conditions(
        self,
        phase: OpExPhase,
        net_gex: float,
        distance_to_pin: float,
    ) -> Optional[str]:
        """
        Check if signal should be skipped.

        Returns:
            Skip reason string, or None if should proceed
        """
        # Check phase
        if phase not in self.config.active_phases:
            return f"Phase {phase.value} not in active phases"

        # Check GEX (require positive/stabilizing)
        if net_gex <= self.config.min_gex_threshold:
            return f"GEX {net_gex:.0f} below threshold (destabilizing environment)"

        # Check distance
        abs_distance = abs(distance_to_pin)
        if abs_distance < self.config.min_distance_pct:
            return f"Distance {abs_distance:.2%} below minimum {self.config.min_distance_pct:.2%}"

        if abs_distance > self.config.max_distance_pct:
            return f"Distance {abs_distance:.2%} above maximum {self.config.max_distance_pct:.2%}"

        return None

    def _generate_pinning_signal(
        self,
        spot: float,
        pin_strike: float,
        distance_to_pin: float,
        net_gex: float,
    ) -> Tuple[SignalDirection, float, str]:
        """
        Generate signal for PINNING phase.

        Logic: Fade moves away from pin (mean reversion)
        - Price above pin -> SHORT (expect reversion down)
        - Price below pin -> LONG (expect reversion up)
        """
        abs_distance = abs(distance_to_pin)

        # Calculate strength based on distance (closer to max = stronger)
        strength = min(
            abs_distance / self.config.max_distance_pct,
            1.0
        )

        if distance_to_pin > 0:
            # Price above pin -> fade by going short
            direction = SignalDirection.SHORT
            rationale = (
                f"PINNING: Price {abs_distance:.2%} above pin ${pin_strike:.2f}. "
                f"Positive GEX ${net_gex/1e6:.1f}M = stabilizing. "
                f"Fade the move -> SHORT"
            )
        else:
            # Price below pin -> fade by going long
            direction = SignalDirection.LONG
            rationale = (
                f"PINNING: Price {abs_distance:.2%} below pin ${pin_strike:.2f}. "
                f"Positive GEX ${net_gex/1e6:.1f}M = stabilizing. "
                f"Fade the move -> LONG"
            )

        return direction, strength, rationale

    def _generate_release_signal(
        self,
        spot: float,
        pin_strike: float,
        distance_to_pin: float,
        net_gex: float,
    ) -> Tuple[SignalDirection, float, str]:
        """
        Generate signal for POST_OPEX phase.

        Logic: Follow momentum after gamma release
        - Price above pin -> LONG (momentum continuation)
        - Price below pin -> SHORT (momentum continuation)
        """
        abs_distance = abs(distance_to_pin)

        # Calculate strength based on distance (farther = stronger momentum)
        strength = min(
            abs_distance / self.config.max_distance_pct,
            1.0
        )

        if distance_to_pin > 0:
            # Price above pin -> follow momentum long
            direction = SignalDirection.LONG
            rationale = (
                f"POST_OPEX: Price {abs_distance:.2%} above pin ${pin_strike:.2f}. "
                f"Gamma released, momentum continuation -> LONG"
            )
        else:
            # Price below pin -> follow momentum short
            direction = SignalDirection.SHORT
            rationale = (
                f"POST_OPEX: Price {abs_distance:.2%} below pin ${pin_strike:.2f}. "
                f"Gamma released, momentum continuation -> SHORT"
            )

        return direction, strength, rationale

    def generate_signals_batch(
        self,
        chains: List[OptionsChain],
        current_date: Optional[date] = None,
    ) -> List[OpExSignal]:
        """
        Generate signals for multiple symbols.

        Args:
            chains: List of options chains
            current_date: Date for signals

        Returns:
            List of OpExSignal, one per chain
        """
        return [
            self.generate_signal(chain, current_date)
            for chain in chains
        ]

    def get_actionable_signals(
        self,
        chains: List[OptionsChain],
        current_date: Optional[date] = None,
    ) -> List[OpExSignal]:
        """
        Get only actionable (non-neutral) signals.

        Args:
            chains: List of options chains
            current_date: Date for signals

        Returns:
            List of actionable OpExSignal
        """
        signals = self.generate_signals_batch(chains, current_date)
        return [s for s in signals if s.is_actionable]
