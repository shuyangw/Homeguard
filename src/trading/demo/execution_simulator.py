"""
Execution simulation for demo trading.

Provides realistic execution simulation including:
- Price slippage (market impact)
- Trading fees
- Execution latency

Based on typical crypto exchange characteristics.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from src.trading.brokers.interfaces.base import OrderSide
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """
    Result of a simulated execution.

    Attributes:
        fill_price: Actual fill price after slippage
        fill_qty: Quantity filled
        fees: Trading fees charged
        slippage_cost: Cost of slippage (difference from market price)
        latency_ms: Simulated execution latency
        timestamp: Execution timestamp
        market_price: Original market price before slippage
    """

    fill_price: float
    fill_qty: Decimal
    fees: float
    slippage_cost: float
    latency_ms: int
    timestamp: datetime
    market_price: float

    @property
    def total_cost(self) -> float:
        """Total cost including slippage and fees."""
        return self.slippage_cost + self.fees

    @property
    def effective_price(self) -> float:
        """Effective price including all costs (for buys) or net proceeds (for sells)."""
        return self.fill_price + (self.fees / float(self.fill_qty))


class ExecutionSimulator:
    """
    Simulates realistic trade execution.

    Models slippage, fees, and latency based on typical crypto exchange
    characteristics.

    Slippage Model:
        - BUY: fill_price = market_price * (1 + slippage_bps / 10000)
        - SELL: fill_price = market_price * (1 - slippage_bps / 10000)

    Fee Model:
        - fees = notional_value * (fee_bps / 10000)

    Latency Model:
        - Random uniform distribution between min and max latency

    Example:
        simulator = ExecutionSimulator(slippage_bps=5, fee_bps=10)
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal('0.01'),
            market_price=50000.0
        )
        print(f"Fill @ ${result.fill_price:.2f}, fees: ${result.fees:.2f}")
    """

    # Default parameters (typical crypto exchange)
    DEFAULT_SLIPPAGE_BPS = 5  # 5 basis points = 0.05%
    DEFAULT_FEE_BPS = 10  # 10 basis points = 0.1% (typical maker/taker fee)
    DEFAULT_LATENCY_MIN_MS = 50  # 50ms minimum
    DEFAULT_LATENCY_MAX_MS = 200  # 200ms maximum

    def __init__(
        self,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        fee_bps: float = DEFAULT_FEE_BPS,
        latency_min_ms: int = DEFAULT_LATENCY_MIN_MS,
        latency_max_ms: int = DEFAULT_LATENCY_MAX_MS,
        randomize_slippage: bool = True,
    ):
        """
        Initialize execution simulator.

        Args:
            slippage_bps: Slippage in basis points (default: 5)
            fee_bps: Trading fee in basis points (default: 10)
            latency_min_ms: Minimum latency in milliseconds (default: 50)
            latency_max_ms: Maximum latency in milliseconds (default: 200)
            randomize_slippage: Whether to randomize slippage (0 to max) or use fixed
        """
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.latency_min_ms = latency_min_ms
        self.latency_max_ms = latency_max_ms
        self.randomize_slippage = randomize_slippage

        logger.info(
            f"[ExecutionSim] Initialized with slippage={slippage_bps}bps, "
            f"fees={fee_bps}bps, latency={latency_min_ms}-{latency_max_ms}ms"
        )

    def simulate_execution(
        self,
        side: OrderSide,
        quantity: Decimal,
        market_price: float,
        symbol: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Simulate trade execution with slippage, fees, and latency.

        Args:
            side: OrderSide.BUY or OrderSide.SELL
            quantity: Order quantity (Decimal)
            market_price: Current market price
            symbol: Optional symbol for logging

        Returns:
            ExecutionResult with all execution details
        """
        quantity = Decimal(str(quantity))
        notional = float(quantity) * market_price

        # Calculate slippage
        if self.randomize_slippage:
            # Random slippage between 0 and max
            actual_slippage_bps = random.uniform(0, self.slippage_bps)
        else:
            actual_slippage_bps = self.slippage_bps

        slippage_factor = actual_slippage_bps / 10000

        if side == OrderSide.BUY:
            # Buys get filled at slightly higher price (adverse slippage)
            fill_price = market_price * (1 + slippage_factor)
        else:
            # Sells get filled at slightly lower price (adverse slippage)
            fill_price = market_price * (1 - slippage_factor)

        # Calculate slippage cost
        slippage_cost = abs(fill_price - market_price) * float(quantity)

        # Calculate fees (on notional value at fill price)
        fill_notional = float(quantity) * fill_price
        fees = fill_notional * (self.fee_bps / 10000)

        # Simulate latency
        latency_ms = random.randint(self.latency_min_ms, self.latency_max_ms)

        result = ExecutionResult(
            fill_price=fill_price,
            fill_qty=quantity,
            fees=fees,
            slippage_cost=slippage_cost,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            market_price=market_price,
        )

        if symbol:
            logger.debug(
                f"[ExecutionSim] {symbol} {side.value.upper()} {quantity} @ ${fill_price:.2f} "
                f"(market: ${market_price:.2f}, slippage: ${slippage_cost:.4f}, "
                f"fees: ${fees:.4f}, latency: {latency_ms}ms)"
            )

        return result

    def estimate_execution(
        self,
        side: OrderSide,
        quantity: Decimal,
        market_price: float,
    ) -> dict:
        """
        Estimate execution costs without randomization.

        Useful for pre-trade analysis.

        Args:
            side: OrderSide.BUY or OrderSide.SELL
            quantity: Order quantity (Decimal)
            market_price: Current market price

        Returns:
            Dict with estimated costs
        """
        quantity = Decimal(str(quantity))
        notional = float(quantity) * market_price

        slippage_factor = self.slippage_bps / 10000

        if side == OrderSide.BUY:
            estimated_fill_price = market_price * (1 + slippage_factor)
        else:
            estimated_fill_price = market_price * (1 - slippage_factor)

        slippage_cost = abs(estimated_fill_price - market_price) * float(quantity)
        fees = notional * (self.fee_bps / 10000)

        return {
            "estimated_fill_price": estimated_fill_price,
            "estimated_slippage_cost": slippage_cost,
            "estimated_fees": fees,
            "estimated_total_cost": slippage_cost + fees,
            "notional_value": notional,
        }

    def get_config(self) -> dict:
        """Get current simulator configuration."""
        return {
            "slippage_bps": self.slippage_bps,
            "fee_bps": self.fee_bps,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "randomize_slippage": self.randomize_slippage,
        }
