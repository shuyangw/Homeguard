"""
Options data schema definitions.

Dataclasses for options chain data, Greeks, and GEX calculations.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
from enum import Enum


class OptionRight(str, Enum):
    """Option type (call or put)."""
    CALL = "C"
    PUT = "P"


@dataclass
class OptionSnapshot:
    """
    Single option contract snapshot with Greeks.

    Represents EOD data for one option contract on one date.
    """
    # Contract identifiers
    symbol: str                  # Underlying symbol (e.g., "SPY")
    date: date                   # Snapshot date
    expiry: date                 # Option expiration date
    strike: float                # Strike price in dollars
    option_type: OptionRight     # 'C' or 'P'

    # Market data
    open_interest: int           # Open interest (contracts)
    volume: int = 0              # Daily volume
    bid: float = 0.0             # Best bid
    ask: float = 0.0             # Best ask
    underlying_price: float = 0.0  # Spot price at snapshot

    # Greeks (first-order)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Volatility
    implied_vol: float = 0.0

    # Derived
    days_to_expiry: int = 0

    def __post_init__(self):
        """Calculate derived fields."""
        if isinstance(self.date, datetime):
            self.date = self.date.date()
        if isinstance(self.expiry, datetime):
            self.expiry = self.expiry.date()
        if isinstance(self.option_type, str):
            self.option_type = OptionRight(self.option_type)
        if self.days_to_expiry == 0:
            self.days_to_expiry = (self.expiry - self.date).days

    @property
    def is_call(self) -> bool:
        """True if this is a call option."""
        return self.option_type == OptionRight.CALL

    @property
    def is_put(self) -> bool:
        """True if this is a put option."""
        return self.option_type == OptionRight.PUT

    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ask if self.ask > 0 else self.bid

    @property
    def moneyness(self) -> float:
        """
        Moneyness ratio: spot / strike.

        > 1.0: In-the-money for calls, OTM for puts
        < 1.0: Out-of-the-money for calls, ITM for puts
        """
        if self.strike > 0:
            return self.underlying_price / self.strike
        return 0.0


@dataclass
class StrikeGEX:
    """
    Gamma Exposure (GEX) aggregated at a single strike price.

    Combines call and put GEX for dealer positioning analysis.
    """
    strike: float
    call_gex: float              # GEX from calls at this strike
    put_gex: float               # GEX from puts at this strike
    net_gex: float               # call_gex + put_gex
    call_oi: int = 0             # Call open interest
    put_oi: int = 0              # Put open interest
    total_oi: int = 0            # Combined open interest

    def __post_init__(self):
        """Calculate total OI if not provided."""
        if self.total_oi == 0:
            self.total_oi = self.call_oi + self.put_oi


@dataclass
class OptionsChain:
    """
    Complete options chain for a symbol/date/expiry combination.

    Contains all option contracts for analysis.
    """
    symbol: str
    date: date
    expiry: date
    underlying_price: float
    contracts: List[OptionSnapshot] = field(default_factory=list)

    # Pre-computed aggregates
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 0.0

    def __post_init__(self):
        """Compute aggregates from contracts."""
        if self.contracts:
            self._compute_aggregates()

    def _compute_aggregates(self):
        """Calculate aggregate statistics from contracts."""
        self.total_call_oi = sum(
            c.open_interest for c in self.contracts if c.is_call
        )
        self.total_put_oi = sum(
            c.open_interest for c in self.contracts if c.is_put
        )
        self.total_call_volume = sum(
            c.volume for c in self.contracts if c.is_call
        )
        self.total_put_volume = sum(
            c.volume for c in self.contracts if c.is_put
        )

        if self.total_call_oi > 0:
            self.put_call_ratio = self.total_put_oi / self.total_call_oi

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return (self.expiry - self.date).days

    @property
    def total_oi(self) -> int:
        """Total open interest across all contracts."""
        return self.total_call_oi + self.total_put_oi

    def get_calls(self) -> List[OptionSnapshot]:
        """Get all call contracts."""
        return [c for c in self.contracts if c.is_call]

    def get_puts(self) -> List[OptionSnapshot]:
        """Get all put contracts."""
        return [c for c in self.contracts if c.is_put]

    def get_strikes(self) -> List[float]:
        """Get unique strike prices, sorted."""
        return sorted(set(c.strike for c in self.contracts))

    def get_atm_strike(self) -> float:
        """
        Get the at-the-money strike (closest to spot).
        """
        strikes = self.get_strikes()
        if not strikes:
            return 0.0
        return min(strikes, key=lambda s: abs(s - self.underlying_price))

    def filter_by_strike_range(
        self,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[OptionSnapshot]:
        """Filter contracts by strike range."""
        result = self.contracts
        if min_strike is not None:
            result = [c for c in result if c.strike >= min_strike]
        if max_strike is not None:
            result = [c for c in result if c.strike <= max_strike]
        return result

    def filter_near_money(self, pct_range: float = 0.10) -> List[OptionSnapshot]:
        """
        Filter to contracts within pct_range of spot.

        Args:
            pct_range: Percentage range (0.10 = +/- 10% from spot)

        Returns:
            Contracts with strikes within range
        """
        min_strike = self.underlying_price * (1 - pct_range)
        max_strike = self.underlying_price * (1 + pct_range)
        return self.filter_by_strike_range(min_strike, max_strike)


@dataclass
class DailyGEXSummary:
    """
    Daily GEX summary for a symbol.

    Pre-computed aggregate for quick lookups.
    """
    symbol: str
    date: date
    expiry: date
    underlying_price: float

    # GEX metrics
    total_gex: float             # Net GEX across all strikes
    call_gex: float              # Total call GEX
    put_gex: float               # Total put GEX

    # Pin strike identification
    pin_strike: float            # Most likely pin strike
    pin_strike_gex: float        # GEX at pin strike
    distance_to_pin: float       # Spot distance to pin as %

    # Volume and OI
    total_oi: int
    put_call_ratio: float

    # GEX profile summary
    gex_percentile: float = 0.0  # Where today's GEX ranks historically

    @property
    def is_positive_gex(self) -> bool:
        """True if net GEX is positive (stabilizing regime)."""
        return self.total_gex > 0

    @property
    def is_near_pin(self) -> bool:
        """True if price is within 2% of pin strike."""
        return abs(self.distance_to_pin) < 0.02
