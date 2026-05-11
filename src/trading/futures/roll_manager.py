"""Futures-broker roll manager.

Replaces the v1 stub from src/data/roll_detector.py (which returned an
empty list from get_upcoming_rolls). This version reads positions from
the futures broker, checks expiration vs lookahead, and suggests new
contract months via the volume-active resolver.

The data-layer `src/data/roll_detector.py::FuturesRollManager` still
exists for compatibility -- it provides get_active_contract for strategy
data lookups. This trading-layer manager is what live execution uses.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from src.trading.futures.expiration_guard import EXPIRATION_THRESHOLDS
from src.trading.futures.position import FuturesPosition


@dataclass(frozen=True)
class RollEvent:
    """A position that needs to roll within the lookahead window."""
    position: FuturesPosition
    days_until_required: int           # max(0, days_to_expiry - threshold)
    suggested_new_month: str | None    # YYYYMM; None if resolver couldn't determine
    suggested_action: Literal["roll", "close", "settle"]


class FuturesRollManager:
    """Identifies positions that need rolling and suggests the target contract."""

    def __init__(self, broker: Any, resolver: Any) -> None:
        self._broker = broker
        self._resolver = resolver

    def _threshold(self, symbol_root: str) -> int:
        return EXPIRATION_THRESHOLDS.get(
            symbol_root, EXPIRATION_THRESHOLDS["_DEFAULT"]
        )

    def _suggest_action(self, position: FuturesPosition) -> Literal["roll", "close", "settle"]:
        """v1 heuristic: roll for continuous-exposure strategies, close otherwise.

        Strategies that explicitly need rolling (adaptation_a, adaptation_b)
        get roll. Others (adaptation_d) get close. Operator can override per
        position via roll calendar.
        """
        if position.strategy in ("adaptation_a", "adaptation_b"):
            return "roll"
        return "close"

    def get_upcoming_rolls(
        self,
        lookahead_days: int = 14,
        today: date | None = None,
    ) -> list[RollEvent]:
        """Return positions whose expiration is within lookahead_days.

        For each, suggest a new contract month via the resolver and a
        suggested action (roll/close/settle) based on strategy.
        """
        if today is None:
            today = date.today()
        positions = self._broker.get_futures_positions()
        events: list[RollEvent] = []
        for pos in positions:
            days = (pos.expiration_date - today).days
            if days > lookahead_days:
                continue
            threshold = self._threshold(pos.symbol_root)
            days_until_required = max(0, days - threshold)

            # Suggest the new contract month via volume-active resolver
            try:
                resolved = self._resolver.resolve_active_contract(pos.symbol_root, today)
                # If the volume-active contract is the SAME as the current position
                # (no roll has happened yet), fall back to None to signal "look later".
                if resolved.contract_month == pos.contract_month:
                    new_month: str | None = None
                else:
                    new_month = resolved.contract_month
            except Exception:
                new_month = None

            events.append(RollEvent(
                position=pos,
                days_until_required=days_until_required,
                suggested_new_month=new_month,
                suggested_action=self._suggest_action(pos),
            ))
        return events
