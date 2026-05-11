"""MarginGuard: SPAN margin pre-check with 30% cash buffer.

Before every order, the guard queries broker.what_if_order() for the
projected post-trade margin. If the resulting margin would leave less
than 30% of net liquidation as free cash, the order is rejected.

The 30% rule is from Homeguard CLAUDE.md Section 9.4 (Margin management).
A separate overnight check applies to orders that will be held past
3:55pm ET when SPAN initial margin doubles.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.trading.futures.symbol_resolver import ResolvedOrder


class MarginVerdict(Enum):
    OK = "ok"
    REJECT = "reject"
    REJECT_OVERNIGHT = "reject_overnight"


@dataclass(frozen=True)
class MarginCheckResult:
    verdict: MarginVerdict
    reason: str = ""


class MarginGuard:
    """Pre-trade margin gate enforcing 30% cash buffer."""

    CASH_BUFFER_PCT = 0.30  # Homeguard CLAUDE.md Section 9.4
    OVERNIGHT_INITIAL_MULTIPLIER = 2.0  # IBKR typical; verify against actual account

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def pre_trade_check(
        self,
        proposed_order: ResolvedOrder,
        hold_overnight: bool = False,
    ) -> MarginCheckResult:
        """Project post-trade margin via what_if_order; reject if free cash
        drops below 30% of net liquidation.

        If hold_overnight=True, also check the overnight margin (2x initial)
        against the same buffer rule.
        """
        account = self._broker.get_margin_status()
        equity = float(account["net_liquidation"])
        required_free_cash = equity * self.CASH_BUFFER_PCT

        # Project post-trade margin
        whatif = self._broker.what_if_order(
            symbol_root=proposed_order.symbol_root,
            contract_month=proposed_order.contract_month,
            side=proposed_order.side,
            quantity=proposed_order.quantity,
            order_type=proposed_order.order_type,
            limit_price=proposed_order.limit_price,
        )
        post_trade_maint = float(whatif["maintenance_margin_after"])
        post_trade_init = float(whatif["initial_margin_after"])

        # Intraday check: maintenance margin must leave 30% buffer
        free_cash_after = equity - post_trade_maint
        if free_cash_after < required_free_cash:
            return MarginCheckResult(
                verdict=MarginVerdict.REJECT,
                reason=(
                    f"Post-trade maintenance margin ${post_trade_maint:,.0f} "
                    f"leaves only ${free_cash_after:,.0f} free cash; "
                    f"need ${required_free_cash:,.0f} (30% buffer)"
                ),
            )

        if hold_overnight:
            overnight_margin = post_trade_init * self.OVERNIGHT_INITIAL_MULTIPLIER
            free_cash_overnight = equity - overnight_margin
            if free_cash_overnight < required_free_cash:
                return MarginCheckResult(
                    verdict=MarginVerdict.REJECT_OVERNIGHT,
                    reason=(
                        f"Overnight margin ${overnight_margin:,.0f} (2x initial) "
                        f"would leave only ${free_cash_overnight:,.0f} free cash; "
                        f"need ${required_free_cash:,.0f} (30% buffer)"
                    ),
                )

        return MarginCheckResult(verdict=MarginVerdict.OK)
