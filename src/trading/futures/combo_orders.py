"""Combo order builders for futures spreads and rolls.

All multi-leg futures orders MUST execute as a single combo (BAG)
instrument. This module builds the spec; the broker submits it. The
critical safety property: NO sequential separate-leg fallback if the
combo fails -- caller raises ComboOrderRejected and operator
investigates.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.4.
"""
from __future__ import annotations

from dataclasses import dataclass


class ComboOrderRejected(Exception):
    """Combo failed; do NOT fall back to separate-leg orders.

    Operator must investigate. Common causes:
    - SPAN credit not granted for the spread (margin model mismatch)
    - One leg's contract not currently active
    - Exchange routing rejection (after-hours, halted product)
    """


@dataclass(frozen=True)
class ComboLegSpec:
    """One leg of a futures combo order.

    action: "BUY" or "SELL"
    ratio: absolute number of contracts on this leg
    """
    symbol_root: str
    contract_month: str
    action: str
    ratio: int


@dataclass(frozen=True)
class ComboOrderSpec:
    """A complete combo order ready for broker.submit_combo_order()."""
    legs: tuple[ComboLegSpec, ...]
    exchange: str = "GLOBEX"


class FuturesComboOrderBuilder:
    """Build BAG combo specs for calendar rolls and inter-commodity spreads."""

    def build_calendar_roll(
        self,
        symbol_root: str,
        from_month: str,
        to_month: str,
        quantity: int,
    ) -> ComboOrderSpec:
        """Calendar spread: close from_month, open to_month.

        Positive quantity = current position is long; SELL from, BUY to.
        Negative quantity = current position is short; BUY from (cover), SELL to.
        """
        if quantity == 0:
            raise ValueError("calendar_roll requires nonzero quantity")
        abs_qty = abs(quantity)
        if quantity > 0:
            close_action = "SELL"
            open_action = "BUY"
        else:
            close_action = "BUY"
            open_action = "SELL"
        legs = (
            ComboLegSpec(
                symbol_root=symbol_root, contract_month=from_month,
                action=close_action, ratio=abs_qty,
            ),
            ComboLegSpec(
                symbol_root=symbol_root, contract_month=to_month,
                action=open_action, ratio=abs_qty,
            ),
        )
        return ComboOrderSpec(legs=legs)

    def build_inter_commodity_spread(
        self,
        leg_a_symbol: str, leg_a_month: str, leg_a_qty: int,
        leg_b_symbol: str, leg_b_month: str, leg_b_qty: int,
    ) -> ComboOrderSpec:
        """Inter-commodity spread (e.g., ES vs NQ ratio).

        Signed quantities encode side per leg.
        """
        def _action(q: int) -> str:
            return "BUY" if q > 0 else "SELL"
        legs = (
            ComboLegSpec(
                symbol_root=leg_a_symbol, contract_month=leg_a_month,
                action=_action(leg_a_qty), ratio=abs(leg_a_qty),
            ),
            ComboLegSpec(
                symbol_root=leg_b_symbol, contract_month=leg_b_month,
                action=_action(leg_b_qty), ratio=abs(leg_b_qty),
            ),
        )
        return ComboOrderSpec(legs=legs)
