"""Roll detection automation for futures live trading.

Thin wrapper over ContinuousContractDataLoader.detect_roll_dates. Used by
live strategies to identify the active contract for any date and upcoming
rolls within a lookahead window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.data.continuous_contract_loader import ContinuousContractDataLoader


@dataclass(frozen=True)
class RollEvent:
    """A predicted or detected contract roll."""
    root: str
    roll_date: date
    from_contract: str
    to_contract: str


class FuturesRollManager:
    """Manages roll dates and active contract identification."""

    def __init__(self) -> None:
        self._loader = ContinuousContractDataLoader()

    def get_active_contract(self, root: str, d: date) -> str:
        """Return the active contract symbol for `root` on date `d`."""
        active_df = self._loader._active_contract_per_day(root, d, d)
        if active_df.is_empty():
            raise ValueError(f"no active contract data for {root} on {d}")
        return active_df.row(0, named=True)["active"]

    def get_upcoming_rolls(
        self,
        roots: list[str],
        today: date | None = None,
        lookahead_days: int = 14,
    ) -> list[RollEvent]:
        """Predict rolls within lookahead_days.

        v1: returns empty list. True upcoming-roll prediction requires expiration
        date lookup from futures_definitions and rule-based timing (volume
        crossover heuristics or fixed-day-before-expiration). Out of scope here;
        caller should consult the roll calendar manually for now.
        """
        if today is None:
            today = date.today()
        return []
