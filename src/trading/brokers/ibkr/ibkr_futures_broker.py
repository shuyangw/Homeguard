"""IBKRFuturesBroker skeleton.

Implements FuturesTradingInterface for IBKR. This is the skeleton --
methods raise NotImplementedError pending sub-chunks 6c (audit log),
6d (symbol resolver), 6e (expiration guard), 6f (margin guard),
6g (combo atomicity), 6h (per-cycle reconciler), 6i (real upcoming
rolls), and 6j (integration wiring).

The skeleton exists so the broker contract test catches missing-method
regressions and subsequent sub-chunks have a class to fill in.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
for the full safeguard specification.
"""
from __future__ import annotations

from typing import Any

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.brokers.interfaces.futures_trading import FuturesTradingInterface


class IBKRFuturesBroker(FuturesTradingInterface):
    """IBKR adapter for futures trading.

    Skeleton: methods raise NotImplementedError. Each safeguard sub-chunk
    fills in part of the implementation. See progress docs
    20260511_CHUNK6*.md for the integration history.
    """

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        # Connection management deferred until first use; sub-chunk 6j wires
        # this up.
        self._ib = None

    # --- OrderManagementInterface methods (inherited) ---

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def get_order(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def get_orders(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def get_open_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    # --- FuturesTradingInterface methods ---

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
        raise NotImplementedError(
            "wired progressively through sub-chunks 6d-6j"
        )

    def place_futures_combo_order(
        self,
        legs: list[dict[str, Any]],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict[str, Any]:
        raise NotImplementedError("wired in sub-chunk 6g (combo atomicity)")

    def get_futures_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def get_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def close_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def close_all_futures_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError("wired in sub-chunk 6j integration")

    def what_if_order(
        self,
        symbol_root: str,
        contract_month: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("wired in sub-chunk 6f (margin guard)")

    def get_margin_status(self) -> dict[str, Any]:
        raise NotImplementedError("wired in sub-chunk 6f (margin guard)")
