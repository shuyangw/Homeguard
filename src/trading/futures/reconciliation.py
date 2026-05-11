"""PerCycleReconciler: state vs broker reconciliation every cycle.

Existing reconciliation runs only at strategy startup. This adds per-cycle
reconciliation so we catch IBKR auto-liquidations, silent expirations,
manual operator interventions, async fill confirmations, and out-of-band
rolls -- all of which mutate broker state without the strategy's
knowledge.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.trading.futures.position import FuturesPosition


class ReconciliationVerdict(Enum):
    MATCH = "match"
    DRIFT_QUANTITY = "drift_quantity"
    DRIFT_PRICE = "drift_price"
    MISSING_ON_BROKER = "missing_on_broker"
    MISSING_IN_STATE = "missing_in_state"
    EXPIRATION_DISAPPEARED = "expired_silently"


@dataclass(frozen=True)
class PositionDiff:
    """One position-level disagreement between state and broker."""
    key: tuple[str, str]   # (symbol_root, contract_month)
    state_quantity: int | None
    broker_quantity: int | None
    detail: str = ""


@dataclass(frozen=True)
class ReconciliationResult:
    verdict: ReconciliationVerdict
    diffs: list[PositionDiff] = field(default_factory=list)

    def summary(self) -> str:
        if self.verdict == ReconciliationVerdict.MATCH:
            return "all positions match"
        diff_summaries = [
            f"{d.key[0]}/{d.key[1]}: state={d.state_quantity} broker={d.broker_quantity}"
            for d in self.diffs
        ]
        return f"{self.verdict.value}: {'; '.join(diff_summaries)}"


class PerCycleReconciler:
    """Compare state vs broker for a strategy before each cycle.

    Reconciliation key for futures: (symbol_root, contract_month).
    """

    def __init__(
        self,
        state_loader: Callable[[str], list[FuturesPosition]],
        broker_positions: Callable[[], list[FuturesPosition]],
        notifier: Callable[..., None] | None = None,
    ) -> None:
        self._state_loader = state_loader
        self._broker_positions = broker_positions
        self._notifier = notifier

    def reconcile(self, strategy: str) -> ReconciliationResult:
        state_list = self._state_loader(strategy)
        broker_list = self._broker_positions()

        state_by_key = {p.position_key: p for p in state_list}
        broker_by_key = {p.position_key: p for p in broker_list}

        diffs: list[PositionDiff] = []
        # Missing on broker
        for key, sp in state_by_key.items():
            if key not in broker_by_key:
                diffs.append(PositionDiff(
                    key=key,
                    state_quantity=sp.quantity,
                    broker_quantity=None,
                    detail="state has it; broker reports zero",
                ))
        # Missing in state
        for key, bp in broker_by_key.items():
            if key not in state_by_key:
                diffs.append(PositionDiff(
                    key=key,
                    state_quantity=None,
                    broker_quantity=bp.quantity,
                    detail="broker has it; state has no record",
                ))
        # Drift quantity
        for key in set(state_by_key.keys()) & set(broker_by_key.keys()):
            sp = state_by_key[key]
            bp = broker_by_key[key]
            if sp.quantity != bp.quantity:
                diffs.append(PositionDiff(
                    key=key,
                    state_quantity=sp.quantity,
                    broker_quantity=bp.quantity,
                    detail="quantity mismatch",
                ))

        if not diffs:
            return ReconciliationResult(verdict=ReconciliationVerdict.MATCH, diffs=[])

        # Classify the verdict from the highest-severity diff type
        if any(d.broker_quantity is None for d in diffs):
            verdict = ReconciliationVerdict.MISSING_ON_BROKER
        elif any(d.state_quantity is None for d in diffs):
            verdict = ReconciliationVerdict.MISSING_IN_STATE
        else:
            verdict = ReconciliationVerdict.DRIFT_QUANTITY

        return ReconciliationResult(verdict=verdict, diffs=diffs)

    def reconcile_and_gate(self, strategy: str) -> bool:
        """Returns True if cycle can proceed, False otherwise."""
        result = self.reconcile(strategy)
        if result.verdict == ReconciliationVerdict.MATCH:
            return True
        # Drift: notify operator and refuse cycle
        if self._notifier is not None:
            self._notifier(
                channel="reconciliation",
                severity="CRITICAL",
                strategy=strategy,
                message=f"Reconciliation drift: {result.summary()}",
                details=[
                    {"key": list(d.key), "state": d.state_quantity, "broker": d.broker_quantity}
                    for d in result.diffs
                ],
            )
        return False
