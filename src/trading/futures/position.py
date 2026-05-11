"""FuturesPosition dataclass: contract-month-aware position identity.

For stocks, 'long 100 TQQQ' is unambiguous. For futures, 'long 2 MES' is
incomplete -- March or June contract? Without contract-month identity,
state and broker can disagree about what positions exist with no
reconciliation logic able to resolve the ambiguity. This is the
foundation that every other futures safeguard depends on.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.1 for the broader rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


def _today() -> date:
    """Return today's date. Indirection for test monkeypatching."""
    return date.today()


@dataclass
class FuturesPosition:
    """Identity + state of a single futures position.

    Reconciliation key: (symbol_root, contract_month). Two positions on
    the same root but different contract months are distinct positions.
    """

    symbol_root: str          # e.g. "MES"
    contract_month: str       # e.g. "202603" (YYYYMM)
    raw_symbol: str           # IBKR/CME format, e.g. "MESH6"
    quantity: int             # signed; positive = long, negative = short
    avg_entry_price: float
    multiplier: float         # USD per index point; e.g. 5 for MES
    tick_size: float          # e.g. 0.25 for MES
    tick_value: float         # multiplier * tick_size; e.g. 1.25 for MES
    expiration_date: date     # last trade date
    broker: str               # which broker holds this position
    strategy: str             # which strategy opened it
    opened_at: datetime

    @property
    def position_key(self) -> tuple[str, str]:
        """Reconciliation key. Two positions match iff their keys are equal."""
        return (self.symbol_root, self.contract_month)

    @property
    def days_to_expiration(self) -> int:
        """Days from today until expiration. Negative if expired."""
        return (self.expiration_date - _today()).days
