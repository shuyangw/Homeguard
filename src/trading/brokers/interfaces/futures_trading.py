"""FuturesTradingInterface ABC.

Parallel to StockTradingInterface and OptionsTradingInterface. Concrete
implementations (IBKRFuturesBroker) implement this contract.

Futures semantics differ from stocks: contracts have expiration months,
margin is SPAN-based (not Reg-T), positions identified by (symbol_root,
contract_month). See docs/superpowers/specs/2026-05-11-futures-broker-
safeguards-design.md for the broader rationale.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.brokers.interfaces.order_management import OrderManagementInterface


class FuturesTradingInterface(OrderManagementInterface):
    """Trading methods for futures contracts (separate from stocks/options)."""

    @abstractmethod
    def place_futures_order(
        self,
        symbol_root: str,
        contract_month: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict[str, Any]:
        """Submit a single-leg futures order.

        symbol_root: e.g. "MES"
        contract_month: "YYYYMM", e.g. "202603"
        side: BUY or SELL
        quantity: absolute number of contracts
        order_type: MARKET / LIMIT / STOP / etc.

        Returns broker order response dict (broker-specific shape).
        """
        ...

    @abstractmethod
    def place_futures_combo_order(
        self,
        legs: list[dict[str, Any]],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict[str, Any]:
        """Submit a multi-leg futures combo (calendar spread, inter-commodity spread).

        legs: list of dicts each with {symbol_root, contract_month, side, ratio}.
        Combo executes atomically as a single CME BAG instrument.
        """
        ...

    @abstractmethod
    def get_futures_positions(self) -> list[dict[str, Any]]:
        """All open futures positions across all (symbol_root, contract_month) keys."""
        ...

    @abstractmethod
    def get_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any] | None:
        """Single position by (symbol_root, contract_month). None if not held."""
        ...

    @abstractmethod
    def close_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any]:
        """Close a specific (symbol_root, contract_month) position."""
        ...

    @abstractmethod
    def close_all_futures_positions(self) -> list[dict[str, Any]]:
        """Close every open futures position. Returns list of close results."""
        ...

    @abstractmethod
    def what_if_order(
        self,
        symbol_root: str,
        contract_month: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Projected post-trade margin from IBKR without actually placing.

        Returns dict with at least:
            initial_margin_after: float
            maintenance_margin_after: float
            buying_power_after: float

        Used by MarginGuard (safeguards spec section 2.5) for the 30% cash
        buffer pre-check.
        """
        ...

    @abstractmethod
    def get_margin_status(self) -> dict[str, Any]:
        """Current SPAN margin breakdown for the futures account.

        Returns dict with at least:
            net_liquidation: float
            initial_margin: float
            maintenance_margin: float
            free_cash: float
            buying_power: float
        """
        ...
