"""Append-only audit log for futures broker activity.

JSONL format, one entry per line. File rotates daily by entry's UTC date.
Used for post-incident forensics: 'what did we send to IBKR and when?'
must be answerable from disk.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.7 for the broader rationale.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """One row of the audit log."""
    timestamp: datetime
    event_type: str  # "submit" | "fill" | "partial_fill" | "cancel" | "reject" | "error"
    strategy: str
    broker: str
    raw_symbol: str
    contract_month: str | None
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    fill_price: float | None
    fill_quantity: int | None
    ibkr_order_id: int | None
    ibkr_perm_id: int | None
    ibkr_exec_id: str | None
    error_message: str | None
    extras: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Append-only JSONL log with daily file rotation by UTC date."""

    def __init__(self, log_dir: Path | None = None) -> None:
        if log_dir is None:
            log_dir = Path.home() / ".homeguard" / "audit"
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ts: datetime) -> Path:
        return self._dir / f"audit_{ts.strftime('%Y%m%d')}.jsonl"

    def log(self, entry: AuditEntry) -> None:
        """Append a single entry to the daily file."""
        path = self._path_for(entry.timestamp)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    def log_submission(self, order: Any, ibkr_response: dict[str, Any]) -> None:
        """Log an order submission. `order` is a ResolvedOrder-like object."""
        self.log(AuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="submit",
            strategy=order.strategy,
            broker="ibkr_futures",
            raw_symbol=order.raw_symbol,
            contract_month=order.contract_month,
            side=order.side.value if hasattr(order.side, "value") else str(order.side),
            quantity=order.quantity,
            order_type=order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            limit_price=getattr(order, "limit_price", None),
            fill_price=None,
            fill_quantity=None,
            ibkr_order_id=ibkr_response.get("orderId"),
            ibkr_perm_id=ibkr_response.get("permId"),
            ibkr_exec_id=None,
            error_message=None,
            extras={"resolved_from": getattr(order, "strategy_intent", "")},
        ))

    def log_fill(self, fill_event: Any) -> None:
        """Log a fill event from IBKR's event stream."""
        self.log(AuditEntry(
            timestamp=fill_event.timestamp,
            event_type="fill",
            strategy=fill_event.strategy_tag,
            broker="ibkr_futures",
            raw_symbol=fill_event.symbol,
            contract_month=fill_event.contract_month,
            side=fill_event.side,
            quantity=fill_event.quantity,
            order_type="",
            limit_price=None,
            fill_price=fill_event.fill_price,
            fill_quantity=fill_event.fill_quantity,
            ibkr_order_id=fill_event.order_id,
            ibkr_perm_id=None,
            ibkr_exec_id=fill_event.exec_id,
            error_message=None,
        ))

    def log_cancel(
        self,
        timestamp: datetime,
        strategy: str,
        raw_symbol: str,
        contract_month: str | None,
        ibkr_order_id: int | None,
    ) -> None:
        """Log an order cancel."""
        self.log(AuditEntry(
            timestamp=timestamp,
            event_type="cancel",
            strategy=strategy,
            broker="ibkr_futures",
            raw_symbol=raw_symbol,
            contract_month=contract_month,
            side="",
            quantity=0,
            order_type="",
            limit_price=None,
            fill_price=None,
            fill_quantity=None,
            ibkr_order_id=ibkr_order_id,
            ibkr_perm_id=None,
            ibkr_exec_id=None,
            error_message=None,
        ))

    def log_reject(
        self,
        timestamp: datetime,
        strategy: str,
        raw_symbol: str,
        contract_month: str | None,
        ibkr_order_id: int | None,
        error_message: str,
    ) -> None:
        """Log an order rejection."""
        self.log(AuditEntry(
            timestamp=timestamp,
            event_type="reject",
            strategy=strategy,
            broker="ibkr_futures",
            raw_symbol=raw_symbol,
            contract_month=contract_month,
            side="",
            quantity=0,
            order_type="",
            limit_price=None,
            fill_price=None,
            fill_quantity=None,
            ibkr_order_id=ibkr_order_id,
            ibkr_perm_id=None,
            ibkr_exec_id=None,
            error_message=error_message,
        ))
