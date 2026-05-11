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

from datetime import date, datetime, timezone
from typing import Any

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.brokers.interfaces.futures_trading import FuturesTradingInterface
from src.trading.futures.expiration_guard import ExpirationGuard, ExpirationVerdict
from src.trading.futures.margin_guard import MarginGuard, MarginVerdict
from src.trading.futures.symbol_resolver import ResolvedOrder


class OrderRejectedError(Exception):
    """Order was rejected by a safeguard (expiration, margin, or other guard)."""


class IBKRFuturesBroker(FuturesTradingInterface):
    """IBKR adapter for futures trading.

    Skeleton: methods raise NotImplementedError. Each safeguard sub-chunk
    fills in part of the implementation. See progress docs
    20260511_CHUNK6*.md for the integration history.
    """

    def __init__(
        self,
        config: IBKRConfig,
        audit_log: Any = None,
        expiration_guard: Any = None,
        margin_guard: Any = None,
    ) -> None:
        self._config = config
        # Connection management deferred until first use; sub-chunk 6j wires
        # this up.
        self._ib = None
        # Safeguards: dependency-injected for tests; defaults constructed lazily
        self._audit_log = audit_log
        self._expiration_guard = expiration_guard
        self._margin_guard = margin_guard

    def _ensure_safeguards(self) -> None:
        """Lazy-init the safeguards if they weren't injected."""
        if self._audit_log is None:
            from src.trading.futures.audit_log import AuditLog
            self._audit_log = AuditLog()
        if self._expiration_guard is None:
            self._expiration_guard = ExpirationGuard()
        if self._margin_guard is None:
            self._margin_guard = MarginGuard(broker=self)

    def submit_resolved_order(
        self,
        resolved_order: ResolvedOrder,
        expiration_date: date | None = None,
        hold_overnight: bool = False,
    ) -> dict[str, Any]:
        """Submit a resolved order through the safeguard chain.

        Order of checks:
          1. ExpirationGuard.check_new_entry_with_expiration
          2. MarginGuard.pre_trade_check
        If any fails, AuditLog.log_reject is called and OrderRejectedError
        is raised. If all pass, the order is forwarded to IBKR (currently
        a stub; real IBKR API call wired in 6l) and AuditLog.log_submission
        records success.

        Expiration date resolution:
          - If `expiration_date` argument is given, use it (explicit override).
          - Otherwise read `resolved_order.expiration_date` (populated when
            the resolver was constructed with a FuturesDefinitionsLoader).
          - If neither is available, raise ValueError -- the ExpirationGuard
            cannot operate without a real expiration date.

        This is the main entry point for strategy adapters using the
        safeguard chain. Strategy code (preferred form):
            resolver = FuturesSymbolResolver(definitions_loader=loader)
            resolved = resolver.resolve_for_order(...)
            broker.submit_resolved_order(resolved, hold_overnight=True)
        """
        self._ensure_safeguards()

        if expiration_date is None:
            expiration_date = resolved_order.expiration_date
        if expiration_date is None:
            raise ValueError(
                "no expiration_date available: pass it explicitly or "
                "construct the resolver with a FuturesDefinitionsLoader"
            )

        # 1. Expiration check
        exp_verdict = self._expiration_guard.check_new_entry_with_expiration(
            resolved_order.symbol_root, expiration_date,
        )
        if exp_verdict not in (ExpirationVerdict.OK,):
            self._audit_log.log_reject(
                timestamp=datetime.now(timezone.utc),
                strategy=resolved_order.strategy,
                raw_symbol=resolved_order.raw_symbol,
                contract_month=resolved_order.contract_month,
                ibkr_order_id=None,
                error_message=f"expiration verdict: {exp_verdict.value}",
            )
            raise OrderRejectedError(
                f"expiration guard rejected order: {exp_verdict.value}"
            )

        # 2. Margin check
        margin_result = self._margin_guard.pre_trade_check(
            resolved_order, hold_overnight=hold_overnight,
        )
        if margin_result.verdict != MarginVerdict.OK:
            self._audit_log.log_reject(
                timestamp=datetime.now(timezone.utc),
                strategy=resolved_order.strategy,
                raw_symbol=resolved_order.raw_symbol,
                contract_month=resolved_order.contract_month,
                ibkr_order_id=None,
                error_message=f"margin verdict: {margin_result.verdict.value} -- {margin_result.reason}",
            )
            raise OrderRejectedError(
                f"margin guard rejected order: {margin_result.reason}"
            )

        # 3. All guards passed -- forward to IBKR (stub for now)
        ibkr_response = self._ibkr_submit_stub(resolved_order)
        self._audit_log.log_submission(resolved_order, ibkr_response)
        return ibkr_response

    def _ibkr_submit_stub(self, resolved_order: ResolvedOrder) -> dict[str, Any]:
        """Stub IBKR submission. Real IBKR API call wired in sub-chunk 6k.

        Returns a synthetic order ID so the AuditLog has something to record.
        """
        import time
        synthetic_id = int(time.time() * 1000) % 100_000_000
        return {
            "orderId": synthetic_id,
            "permId": synthetic_id + 1,
            "status": "stub_accepted",
        }

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
