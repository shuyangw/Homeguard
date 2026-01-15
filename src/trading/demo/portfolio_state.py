"""
Persistent portfolio state for demo trading.

Stores portfolio state (cash, positions) in JSON format for persistence
across restarts. Thread-safe for concurrent access.
"""

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _decimal_encoder(obj):
    """JSON encoder for Decimal and datetime objects."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _decimal_decoder(dct):
    """JSON decoder that converts string decimals back to Decimal."""
    for key in ["quantity", "avg_entry_price", "cost_basis"]:
        if key in dct and isinstance(dct[key], str):
            try:
                dct[key] = Decimal(dct[key])
            except (ValueError, TypeError):
                pass
    return dct


@dataclass
class DemoPosition:
    """
    Represents a single position in the demo portfolio.

    Attributes:
        symbol: Crypto symbol (e.g., 'BTC/USD')
        quantity: Amount held (Decimal for precision)
        avg_entry_price: Average entry price
        cost_basis: Total cost basis
        opened_at: When position was first opened
        last_updated: When position was last modified
    """

    symbol: str
    quantity: Decimal
    avg_entry_price: float
    cost_basis: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Ensure quantity is Decimal."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if self.cost_basis == 0.0:
            self.cost_basis = float(self.quantity) * self.avg_entry_price

    def market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        return float(self.quantity) * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return self.market_value(current_price) - self.cost_basis

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.cost_basis) * 100

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "avg_entry_price": self.avg_entry_price,
            "cost_basis": self.cost_basis,
            "opened_at": self.opened_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DemoPosition":
        """Create from dict (JSON deserialization)."""
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(data["quantity"]),
            avg_entry_price=data["avg_entry_price"],
            cost_basis=data.get("cost_basis", 0.0),
            opened_at=datetime.fromisoformat(data["opened_at"]),
            last_updated=datetime.fromisoformat(data.get("last_updated", data["opened_at"])),
        )


class DemoPortfolioState:
    """
    Persistent portfolio state manager.

    Stores portfolio state (cash, positions, realized P&L) in JSON format.
    Thread-safe for concurrent access from multiple threads.

    State file location: ~/.homeguard/demo/portfolio_state.json

    Example:
        state = DemoPortfolioState(initial_cash=10000.0)

        # Add a position
        state.add_or_update_position('BTC/USD', Decimal('0.01'), 50000.0)

        # Get total value
        prices = {'BTC/USD': 55000.0}
        total = state.get_total_value(prices)

        # State auto-saves on changes
    """

    DEFAULT_STATE_DIR = Path.home() / ".homeguard" / "demo"
    STATE_FILENAME = "portfolio_state.json"

    def __init__(
        self,
        initial_cash: float = 10000.0,
        state_dir: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        Initialize portfolio state.

        Args:
            initial_cash: Starting cash balance (used if no saved state)
            state_dir: Directory for state file (default: ~/.homeguard/demo/)
            auto_save: Whether to auto-save on changes
        """
        self._state_dir = state_dir or self.DEFAULT_STATE_DIR
        self._state_file = self._state_dir / self.STATE_FILENAME
        self._auto_save = auto_save
        self._lock = threading.RLock()

        # Portfolio state
        self._cash: float = initial_cash
        self._positions: Dict[str, DemoPosition] = {}
        self._realized_pnl: float = 0.0
        self._created_at: datetime = datetime.now()
        self._last_modified: datetime = datetime.now()

        # Try to load existing state
        self._load()

    def _ensure_state_dir(self) -> None:
        """Ensure state directory exists."""
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load state from file if exists."""
        if not self._state_file.exists():
            logger.info(
                f"[DemoPortfolio] No saved state found, starting fresh with ${self._cash:.2f}"
            )
            return

        try:
            with open(self._state_file, "r") as f:
                data = json.load(f, object_hook=_decimal_decoder)

            self._cash = data.get("cash", self._cash)
            self._realized_pnl = data.get("realized_pnl", 0.0)
            self._created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            self._last_modified = datetime.fromisoformat(data.get("last_modified", datetime.now().isoformat()))

            # Load positions
            self._positions = {}
            for pos_data in data.get("positions", []):
                pos = DemoPosition.from_dict(pos_data)
                self._positions[pos.symbol] = pos

            logger.info(
                f"[DemoPortfolio] Loaded state: ${self._cash:.2f} cash, "
                f"{len(self._positions)} positions, ${self._realized_pnl:.2f} realized P&L"
            )

        except Exception as e:
            logger.error(f"[DemoPortfolio] Error loading state: {e}")
            logger.warning("[DemoPortfolio] Starting with fresh state")

    def save(self) -> None:
        """Save current state to file."""
        with self._lock:
            try:
                self._ensure_state_dir()
                self._last_modified = datetime.now()

                data = {
                    "cash": self._cash,
                    "realized_pnl": self._realized_pnl,
                    "created_at": self._created_at.isoformat(),
                    "last_modified": self._last_modified.isoformat(),
                    "positions": [pos.to_dict() for pos in self._positions.values()],
                }

                # Write atomically
                temp_file = self._state_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2, default=_decimal_encoder)
                temp_file.replace(self._state_file)

                logger.debug(f"[DemoPortfolio] State saved to {self._state_file}")

            except Exception as e:
                logger.error(f"[DemoPortfolio] Error saving state: {e}")

    def _maybe_save(self) -> None:
        """Save if auto_save is enabled."""
        if self._auto_save:
            self.save()

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        with self._lock:
            return self._cash

    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        with self._lock:
            return self._realized_pnl

    def get_position(self, symbol: str) -> Optional[DemoPosition]:
        """Get position by symbol."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> List[DemoPosition]:
        """Get all positions."""
        with self._lock:
            return list(self._positions.values())

    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        with self._lock:
            return symbol in self._positions

    def add_or_update_position(
        self,
        symbol: str,
        quantity_delta: Decimal,
        price: float,
    ) -> DemoPosition:
        """
        Add to or update a position.

        For buys: Increases quantity, averages entry price.
        For sells: Decreases quantity, realizes P&L.

        Args:
            symbol: Crypto symbol
            quantity_delta: Change in quantity (positive=buy, negative=sell)
            price: Execution price

        Returns:
            Updated position
        """
        with self._lock:
            quantity_delta = Decimal(str(quantity_delta))

            if symbol in self._positions:
                pos = self._positions[symbol]

                if quantity_delta > 0:
                    # Buy: Average in
                    old_value = float(pos.quantity) * pos.avg_entry_price
                    new_value = float(quantity_delta) * price
                    new_quantity = pos.quantity + quantity_delta
                    new_avg = (old_value + new_value) / float(new_quantity)

                    pos.quantity = new_quantity
                    pos.avg_entry_price = new_avg
                    pos.cost_basis = float(new_quantity) * new_avg
                    pos.last_updated = datetime.now()

                else:
                    # Sell: Realize P&L
                    sell_quantity = abs(quantity_delta)
                    if sell_quantity > pos.quantity:
                        raise ValueError(
                            f"Cannot sell {sell_quantity} {symbol}, only have {pos.quantity}"
                        )

                    # Calculate realized P&L
                    sell_value = float(sell_quantity) * price
                    cost = float(sell_quantity) * pos.avg_entry_price
                    realized = sell_value - cost
                    self._realized_pnl += realized

                    # Update position
                    pos.quantity -= sell_quantity
                    pos.cost_basis = float(pos.quantity) * pos.avg_entry_price
                    pos.last_updated = datetime.now()

                    # Remove if fully closed
                    if pos.quantity <= 0:
                        del self._positions[symbol]
                        logger.info(
                            f"[DemoPortfolio] Closed {symbol} position, "
                            f"realized P&L: ${realized:.2f}"
                        )
                        self._maybe_save()
                        return pos

            else:
                if quantity_delta <= 0:
                    raise ValueError(f"Cannot sell {symbol}, no position exists")

                # New position
                pos = DemoPosition(
                    symbol=symbol,
                    quantity=quantity_delta,
                    avg_entry_price=price,
                    cost_basis=float(quantity_delta) * price,
                )
                self._positions[symbol] = pos
                logger.info(
                    f"[DemoPortfolio] Opened {symbol} position: "
                    f"{quantity_delta} @ ${price:.2f}"
                )

            self._maybe_save()
            return pos

    def adjust_cash(self, amount: float) -> float:
        """
        Adjust cash balance.

        Args:
            amount: Amount to add (positive) or subtract (negative)

        Returns:
            New cash balance
        """
        with self._lock:
            self._cash += amount
            self._maybe_save()
            return self._cash

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            prices: Dict mapping symbol to current price

        Returns:
            Total value (cash + positions market value)
        """
        with self._lock:
            total = self._cash
            for symbol, pos in self._positions.items():
                price = prices.get(symbol, pos.avg_entry_price)
                total += pos.market_value(price)
            return total

    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized P&L.

        Args:
            prices: Dict mapping symbol to current price

        Returns:
            Total unrealized P&L across all positions
        """
        with self._lock:
            total = 0.0
            for symbol, pos in self._positions.items():
                price = prices.get(symbol, pos.avg_entry_price)
                total += pos.unrealized_pnl(price)
            return total

    def get_portfolio_snapshot(self, prices: Dict[str, float]) -> Dict:
        """
        Get complete portfolio snapshot.

        Args:
            prices: Dict mapping symbol to current price

        Returns:
            Dict with complete portfolio state
        """
        with self._lock:
            positions_data = {}
            for symbol, pos in self._positions.items():
                price = prices.get(symbol, pos.avg_entry_price)
                positions_data[symbol] = {
                    "quantity": float(pos.quantity),
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": price,
                    "market_value": pos.market_value(price),
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct(price),
                }

            return {
                "timestamp": datetime.now().isoformat(),
                "cash": self._cash,
                "positions": positions_data,
                "total_value": self.get_total_value(prices),
                "unrealized_pnl": self.get_unrealized_pnl(prices),
                "realized_pnl": self._realized_pnl,
            }

    def reset(self, initial_cash: float = 10000.0) -> None:
        """
        Reset portfolio to initial state.

        Args:
            initial_cash: New starting cash balance
        """
        with self._lock:
            self._cash = initial_cash
            self._positions.clear()
            self._realized_pnl = 0.0
            self._created_at = datetime.now()
            self._last_modified = datetime.now()
            self._maybe_save()
            logger.info(f"[DemoPortfolio] Reset with ${initial_cash:.2f} cash")
