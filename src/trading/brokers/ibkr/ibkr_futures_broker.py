"""IBKRFuturesBroker -- real ib_async integration.

Implements FuturesTradingInterface against Interactive Brokers via the
ib_async library, mirroring the patterns established in IBKRBroker (the
stocks/options broker):

  - IBKRConnectionManager owns the ib_async.IB instance on a background
    event loop; this class never touches asyncio directly.
  - `_build_order` translates the broker-agnostic OrderType/TimeInForce
    enums to ib_async order classes.
  - `_build_future_contract` translates (symbol_root, contract_month) to
    an `ib_async.Future` with the right exchange routing per product family.
  - `submit_resolved_order` is the safeguard chain entry point used by
    strategies; it runs ExpirationGuard -> MarginGuard -> AuditLog ->
    real IBKR submission.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
for the safeguard contract, and docs/progress/20260511_CHUNK6_FUTURES_BROKER_SAFEGUARDS.md
for the integration history.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce
from src.trading.brokers.interfaces.futures_trading import FuturesTradingInterface
from src.trading.futures.expiration_guard import ExpirationGuard, ExpirationVerdict
from src.trading.futures.margin_guard import MarginGuard, MarginVerdict
from src.trading.futures.symbol_resolver import ResolvedOrder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Exchange routing per symbol-root family. IB's exchange names are legacy
# (GLOBEX is CME futures, ECBOT is CBOT, etc.). Extend as new products are added.
_EXCHANGE_BY_ROOT: dict[str, str] = {
    # CME (S&P, Nasdaq, Russell, Dow, FX, livestock)
    "ES": "CME", "MES": "CME", "NQ": "CME", "MNQ": "CME",
    "RTY": "CME", "M2K": "CME", "YM": "CBOT", "MYM": "CBOT",
    "6E": "CME", "6J": "CME", "6B": "CME", "6A": "CME", "6C": "CME",
    "LE": "CME", "HE": "CME", "GF": "CME",
    # NYMEX (energy)
    "CL": "NYMEX", "MCL": "NYMEX", "NG": "NYMEX", "HO": "NYMEX",
    "RB": "NYMEX",
    # COMEX (metals)
    "GC": "COMEX", "MGC": "COMEX", "SI": "COMEX", "SIL": "COMEX",
    "HG": "COMEX", "PL": "NYMEX", "PA": "NYMEX",
    # CBOT (rates, ags)
    "ZB": "CBOT", "ZN": "CBOT", "ZF": "CBOT", "ZT": "CBOT",
    "TN": "CBOT", "UB": "CBOT",
    "2YY": "CBOT", "5YY": "CBOT", "10Y": "CBOT", "30Y": "CBOT",
    "ZC": "CBOT", "ZS": "CBOT", "ZW": "CBOT", "ZL": "CBOT",
    "ZM": "CBOT", "ZO": "CBOT", "ZR": "CBOT",
}
_DEFAULT_EXCHANGE = "CME"


class OrderRejectedError(Exception):
    """Order was rejected by a safeguard (expiration, margin, or other guard)."""


class IBKRFuturesBroker(FuturesTradingInterface):
    """IBKR adapter for futures trading via ib_async."""

    def __init__(
        self,
        config: IBKRConfig,
        audit_log: Any = None,
        expiration_guard: Any = None,
        margin_guard: Any = None,
    ) -> None:
        self._config = config
        self._conn: IBKRConnectionManager | None = None
        self._audit_log = audit_log
        self._expiration_guard = expiration_guard
        self._margin_guard = margin_guard

    # ==================== Lifecycle ====================

    def start(self) -> None:
        """Initialize ib_async connection. Must be called before any IBKR ops."""
        if self._conn is None:
            self._conn = IBKRConnectionManager(self._config)
            self._conn.start()
            logger.info(f"[IBKR-FUT] Broker started ({self._config.gateway_type})")

    def stop(self) -> None:
        """Graceful shutdown."""
        if self._conn is not None:
            self._conn.stop()
            self._conn = None
            logger.info("[IBKR-FUT] Broker stopped")

    def _ensure_connection(self) -> IBKRConnectionManager:
        if self._conn is None:
            self.start()
        assert self._conn is not None
        return self._conn

    def _ensure_safeguards(self) -> None:
        """Lazy-init the safeguards if they weren't injected."""
        if self._audit_log is None:
            from src.trading.futures.audit_log import AuditLog
            self._audit_log = AuditLog()
        if self._expiration_guard is None:
            self._expiration_guard = ExpirationGuard()
        if self._margin_guard is None:
            self._margin_guard = MarginGuard(broker=self)

    # ==================== Safeguard entry point ====================

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
        is raised. If all pass, the order is forwarded to IBKR and
        AuditLog.log_submission records success.

        Expiration date resolution:
          - If `expiration_date` argument is given, use it (explicit override).
          - Otherwise read `resolved_order.expiration_date` (populated when
            the resolver was constructed with a FuturesDefinitionsLoader).
          - If neither is available, raise ValueError.
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
        if exp_verdict != ExpirationVerdict.OK:
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

        # 3. All guards passed -- forward to IBKR
        ibkr_response = self._ibkr_submit(resolved_order)
        self._audit_log.log_submission(resolved_order, ibkr_response)
        return ibkr_response

    # ==================== ib_async helpers ====================

    @staticmethod
    def _side_to_ibkr(side: OrderSide) -> str:
        return "BUY" if side == OrderSide.BUY else "SELL"

    @staticmethod
    def _tif_to_ibkr(tif: TimeInForce) -> str:
        return {
            TimeInForce.DAY: "DAY", TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC", TimeInForce.FOK: "FOK",
        }.get(tif, "DAY")

    @staticmethod
    def _exchange_for(symbol_root: str) -> str:
        return _EXCHANGE_BY_ROOT.get(symbol_root, _DEFAULT_EXCHANGE)

    def _build_future_contract(self, symbol_root: str, contract_month: str) -> Any:
        """Build an ib_async Future contract from (root, YYYYMM)."""
        from ib_async import Future
        contract = Future(
            symbol=symbol_root,
            lastTradeDateOrContractMonth=contract_month,
            exchange=self._exchange_for(symbol_root),
        )
        # Bind to the actual instrument; raises if ambiguous/missing
        qualified = self._ensure_connection().ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(
                f"could not qualify futures contract: {symbol_root} {contract_month}"
            )
        return qualified[0]

    def _build_order(
        self,
        action: str, qty: int, order_type: OrderType,
        limit_price: float | None, stop_price: float | None,
        tif: TimeInForce, what_if: bool = False,
    ) -> Any:
        from ib_async import LimitOrder, MarketOrder, StopOrder, StopLimitOrder
        if order_type == OrderType.MARKET:
            order = MarketOrder(action, qty)
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("Limit price required for LIMIT orders")
            order = LimitOrder(action, qty, limit_price)
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("Stop price required for STOP orders")
            order = StopOrder(action, qty, stop_price)
        elif order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                raise ValueError("Both prices required for STOP_LIMIT orders")
            order = StopLimitOrder(action, qty, stop_price, limit_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        order.tif = self._tif_to_ibkr(tif)
        if what_if:
            order.whatIf = True
        return order

    async def _place_async(self, contract: Any, order: Any) -> Any:
        trade = self._ensure_connection().ib.placeOrder(contract, order)
        await asyncio.sleep(0.5)  # let IBKR acknowledge
        return trade

    def _translate_trade(self, trade: Any) -> dict[str, Any]:
        """ib_async Trade -> standardized order dict."""
        order = trade.order
        status = trade.orderStatus
        contract = trade.contract
        status_map = {
            "PendingSubmit": "pending", "PreSubmitted": "pending",
            "Submitted": "pending", "Filled": "filled",
            "Cancelled": "cancelled", "ApiCancelled": "cancelled",
            "Inactive": "rejected",
        }
        return {
            "orderId": int(order.orderId),
            "permId": int(getattr(order, "permId", 0) or 0),
            "status": status_map.get(status.status, status.status.lower()),
            "raw_status": status.status,
            "symbol": getattr(contract, "localSymbol", None) or contract.symbol,
            "contract_month": getattr(contract, "lastTradeDateOrContractMonth", None),
            "quantity": int(order.totalQuantity),
            "side": "buy" if order.action == "BUY" else "sell",
            "order_type": (order.orderType or "MKT").lower(),
            "limit_price": float(order.lmtPrice) if order.lmtPrice else None,
            "stop_price": float(order.auxPrice) if order.auxPrice else None,
            "filled_qty": int(status.filled) if status.filled else 0,
            "filled_avg_price": float(status.avgFillPrice) if status.avgFillPrice else None,
        }

    def _ibkr_submit(self, resolved: ResolvedOrder) -> dict[str, Any]:
        """Forward a resolved order to IBKR. Returns the standardized order dict."""
        contract = self._build_future_contract(
            resolved.symbol_root, resolved.contract_month,
        )
        action = self._side_to_ibkr(resolved.side)
        order = self._build_order(
            action, resolved.quantity, resolved.order_type,
            resolved.limit_price, resolved.stop_price, resolved.time_in_force,
        )
        trade = self._ensure_connection().run_sync(self._place_async(contract, order))
        result = self._translate_trade(trade)
        logger.info(
            f"[IBKR-FUT] Submitted {resolved.raw_symbol} {action} "
            f"{resolved.quantity} -> orderId={result['orderId']} "
            f"status={result['raw_status']}"
        )
        return result

    # ==================== OrderManagementInterface ====================

    def cancel_order(self, order_id: str) -> bool:
        try:
            ib = self._ensure_connection().ib
            for trade in ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    ib.cancelOrder(trade.order)
                    logger.info(f"[IBKR-FUT] Cancelled order {order_id}")
                    return True
            logger.warning(f"[IBKR-FUT] Order {order_id} not found in open trades")
            return False
        except Exception as e:
            logger.error(f"[IBKR-FUT] Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> dict[str, Any]:
        ib = self._ensure_connection().ib
        for trade in ib.trades():
            if str(trade.order.orderId) == str(order_id):
                return self._translate_trade(trade)
        raise LookupError(f"Order {order_id} not found")

    def get_orders(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        ib = self._ensure_connection().ib
        out: list[dict[str, Any]] = []
        for trade in ib.trades():
            # Filter to futures only (secType FUT)
            if getattr(trade.contract, "secType", None) != "FUT":
                continue
            d = self._translate_trade(trade)
            if status_filter is None or d["status"] == status_filter:
                out.append(d)
        return out

    def get_open_orders(self) -> list[dict[str, Any]]:
        ib = self._ensure_connection().ib
        return [
            self._translate_trade(t) for t in ib.openTrades()
            if getattr(t.contract, "secType", None) == "FUT"
        ]

    # ==================== FuturesTradingInterface ====================

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
        """Low-level futures order placement (no safeguards).

        Use submit_resolved_order for the standard safeguard-gated entry point.
        This method exists for tooling/admin use where the safeguard chain
        is intentionally bypassed (e.g., force-close in operator emergency).
        """
        contract = self._build_future_contract(symbol_root, contract_month)
        order = self._build_order(
            self._side_to_ibkr(side), quantity, order_type,
            limit_price, stop_price, time_in_force,
        )
        trade = self._ensure_connection().run_sync(self._place_async(contract, order))
        return self._translate_trade(trade)

    def place_futures_combo_order(
        self,
        legs: list[dict[str, Any]],
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict[str, Any]:
        """Combo (BAG) order for calendar rolls / spreads.

        Each leg is a dict with keys: symbol_root, contract_month, side
        (OrderSide), ratio (int, sign-bearing for buy/sell legs).
        Submitted as a single atomic order; on rejection, ComboOrderRejected
        is raised -- do NOT fall back to separate-leg orders.
        """
        from ib_async import Bag, ComboLeg
        from src.trading.futures.combo_orders import ComboOrderRejected

        if not legs:
            raise ValueError("place_futures_combo_order requires >=1 leg")

        ib = self._ensure_connection().ib
        combo_legs: list[Any] = []
        for leg in legs:
            contract = self._build_future_contract(
                leg["symbol_root"], leg["contract_month"],
            )
            cl = ComboLeg(
                conId=contract.conId,
                ratio=int(abs(leg["ratio"])),
                action=self._side_to_ibkr(leg["side"]),
                exchange=contract.exchange,
            )
            combo_legs.append(cl)

        # Use first leg's symbol/exchange as the bag wrapper
        first = legs[0]
        bag = Bag(
            symbol=first["symbol_root"],
            exchange=self._exchange_for(first["symbol_root"]),
            currency="USD",
            comboLegs=combo_legs,
        )

        order = self._build_order(
            "BUY", 1, order_type, limit_price, None, time_in_force,
        )
        trade = self._ensure_connection().run_sync(self._place_async(bag, order))
        result = self._translate_trade(trade)
        if result["status"] == "rejected":
            raise ComboOrderRejected(
                f"IBKR rejected combo order: {result.get('raw_status')}"
            )
        return result

    def _translate_position(self, pos: Any) -> dict[str, Any]:
        """ib_async portfolio item -> standardized futures position dict."""
        c = pos.contract
        avg_cost = float(getattr(pos, "averageCost", None) or getattr(pos, "avgCost", 0))
        return {
            "symbol_root": c.symbol,
            "contract_month": getattr(c, "lastTradeDateOrContractMonth", "") or "",
            "raw_symbol": getattr(c, "localSymbol", None) or c.symbol,
            "quantity": int(pos.position),
            "avg_entry_price": avg_cost,
            "multiplier": float(getattr(c, "multiplier", 0) or 0),
            "market_price": float(getattr(pos, "marketPrice", 0) or 0),
            "market_value": float(getattr(pos, "marketValue", 0) or 0),
            "unrealized_pnl": float(getattr(pos, "unrealizedPNL", 0) or 0),
        }

    def get_futures_positions(self) -> list[dict[str, Any]]:
        ib = self._ensure_connection().ib
        return [
            self._translate_position(p) for p in ib.portfolio()
            if getattr(p.contract, "secType", None) == "FUT"
        ]

    def get_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any] | None:
        for pos in self.get_futures_positions():
            if (pos["symbol_root"] == symbol_root
                    and pos["contract_month"] == contract_month):
                return pos
        return None

    def close_futures_position(
        self, symbol_root: str, contract_month: str,
    ) -> dict[str, Any]:
        pos = self.get_futures_position(symbol_root, contract_month)
        if pos is None:
            raise LookupError(f"no position for {symbol_root} {contract_month}")
        qty = abs(pos["quantity"])
        close_side = OrderSide.SELL if pos["quantity"] > 0 else OrderSide.BUY
        return self.place_futures_order(
            symbol_root, contract_month, close_side, qty,
            OrderType.MARKET, time_in_force=TimeInForce.DAY,
        )

    def close_all_futures_positions(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pos in self.get_futures_positions():
            try:
                results.append(self.close_futures_position(
                    pos["symbol_root"], pos["contract_month"],
                ))
            except Exception as e:
                logger.error(
                    f"[IBKR-FUT] Failed to close "
                    f"{pos['symbol_root']} {pos['contract_month']}: {e}"
                )
        return results

    def what_if_order(
        self,
        symbol_root: str,
        contract_month: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Pre-trade margin/cost estimate without submitting the order.

        Returns commission, initial_margin, maintenance_margin, equity_with_loan.
        Used by MarginGuard.pre_trade_check.
        """
        contract = self._build_future_contract(symbol_root, contract_month)
        order = self._build_order(
            self._side_to_ibkr(side), quantity, order_type,
            limit_price, None, TimeInForce.DAY, what_if=True,
        )
        ib = self._ensure_connection().ib
        # whatIfOrder is sync; returns OrderState directly
        order_state = ib.whatIfOrder(contract, order)
        return {
            "commission": float(getattr(order_state, "commission", 0) or 0),
            "initial_margin": float(
                getattr(order_state, "initMarginChange", None)
                or getattr(order_state, "initMargin", 0) or 0
            ),
            "maintenance_margin": float(
                getattr(order_state, "maintMarginChange", None)
                or getattr(order_state, "maintMargin", 0) or 0
            ),
            "equity_with_loan": float(
                getattr(order_state, "equityWithLoanChange", None)
                or getattr(order_state, "equityWithLoan", 0) or 0
            ),
        }

    def get_margin_status(self) -> dict[str, Any]:
        """Account margin snapshot for MarginGuard.

        Reads from ib_async accountValues. Returns net_liquidation, free_cash,
        initial_margin, maintenance_margin, available_funds.
        """
        ib = self._ensure_connection().ib
        # accountValues returns one row per (tag, currency, account)
        out = {
            "net_liquidation": 0.0, "free_cash": 0.0,
            "initial_margin": 0.0, "maintenance_margin": 0.0,
            "available_funds": 0.0,
        }
        tag_map = {
            "NetLiquidation": "net_liquidation",
            "AvailableFunds": "available_funds",
            "TotalCashValue": "free_cash",
            "InitMarginReq": "initial_margin",
            "MaintMarginReq": "maintenance_margin",
        }
        for av in ib.accountValues():
            if av.currency not in ("USD", "BASE", ""):
                continue
            if av.tag in tag_map:
                try:
                    out[tag_map[av.tag]] = float(av.value)
                except (TypeError, ValueError):
                    pass
        return out
