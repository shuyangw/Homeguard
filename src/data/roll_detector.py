"""Roll detection automation for futures live trading.

Thin wrapper over ContinuousContractDataLoader.detect_roll_dates. Used by
live strategies to identify the active contract for any date and upcoming
rolls within a lookahead window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures.roll_calendar import RollCalendar, NoActiveContractError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RollEvent:
    """A predicted or detected contract roll."""
    root: str
    roll_date: date
    from_contract: str
    to_contract: str


class FuturesRollManager:
    """Manages roll dates and active contract identification."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._loader = ContinuousContractDataLoader()
        self._calendar = RollCalendar(cache_dir=cache_dir)

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
        """Return real roll events within [today, today + lookahead_days] for each root.

        Reads the built RollCalendar for each root and converts calendar
        RollEvents (from_symbol/to_symbol) into roll_detector RollEvents
        (root/from_contract/to_contract).
        """
        if today is None:
            today = date.today()
        horizon = today + timedelta(days=lookahead_days)
        out: list[RollEvent] = []
        for root in roots:
            try:
                events = self._calendar.roll_events(root)
            except NoActiveContractError as e:
                logger.debug(f"no roll calendar for {root}, skipping upcoming-rolls: {e}")
                continue
            for ev in events:
                if today <= ev.roll_date <= horizon:
                    out.append(RollEvent(
                        root=root,
                        roll_date=ev.roll_date,
                        from_contract=ev.from_symbol,
                        to_contract=ev.to_symbol,
                    ))
        return out
