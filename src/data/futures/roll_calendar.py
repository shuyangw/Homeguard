"""OI-primary futures roll calendar.

detect_rolls() is a pure function over a per-day per-contract OI series; it has
no I/O so it is fully deterministic and unit-testable. The RollCalendar class
(Task 9) wraps it with real data, the FND clamp, and a cached artifact.

Roll rule: the front contract rolls to the back contract when the back
contract's OI exceeds the front's for `hysteresis` consecutive days (anti-blip).
The roll date is the day the streak completes. Trigger is recorded so callers
(and tests) can see WHY each roll fired.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.data.futures.contract_specs import get_spec
from src.data.futures.paths import roll_calendar_dir


@dataclass(frozen=True)
class ContractRef:
    raw_symbol: str
    expiration: date
    activation: date


@dataclass(frozen=True)
class RollEvent:
    roll_date: date
    from_symbol: str
    to_symbol: str
    trigger: str   # "oi_crossover" | "fnd_clamp" | "calendar_fallback"


def _front_by_oi(day_oi: dict[str, int]) -> str | None:
    """Contract with the highest OI on a day, or None if empty.

    Ties break deterministically by symbol name (lexicographically smallest)
    since `oi_by_day` may be built from an unordered groupby (Task 10).
    """
    if not day_oi:
        return None
    return min(day_oi, key=lambda s: (-day_oi[s], s))


def detect_rolls(
    root: str,
    oi_by_day: dict[date, dict[str, int]],
    hysteresis: int = 2,
) -> list[RollEvent]:
    """Detect OI-crossover rolls with a consecutive-day hysteresis.

    Args:
        root: symbol root (for context only; symbols come from the OI dict).
        oi_by_day: {date: {contract_symbol: open_interest}}.
        hysteresis: consecutive days the new front must dominate before rolling.

    Returns:
        Chronological list of RollEvent with trigger="oi_crossover".
    """
    days = sorted(oi_by_day)
    rolls: list[RollEvent] = []
    current_front: str | None = None
    candidate: str | None = None
    streak = 0

    for d in days:
        day_oi = oi_by_day[d]
        top = _front_by_oi(day_oi)
        if top is None:
            continue
        if current_front is None:
            current_front = top
            continue
        if top == current_front:
            candidate = None
            streak = 0
            continue
        # a different contract leads today
        if top == candidate:
            streak += 1
        else:
            candidate = top
            streak = 1
        if streak >= hysteresis:
            rolls.append(RollEvent(
                roll_date=d,
                from_symbol=current_front,
                to_symbol=top,
                trigger="oi_crossover",
            ))
            current_front = top
            candidate = None
            streak = 0
    return rolls


class NoActiveContractError(LookupError):
    """No active contract for the requested (root, date), or root not built."""


def _minus_business_days(d: date, n: int) -> date:
    """Return the date n business days before d (Mon-Fri only)."""
    cur = d
    remaining = n
    while remaining > 0:
        cur -= timedelta(days=1)
        if cur.weekday() < 5:
            remaining -= 1
    return cur


def apply_fnd_clamp(
    root: str,
    rolls: list[RollEvent],
    expirations: dict[str, date],
) -> list[RollEvent]:
    """Pull a physical root's roll earlier if it sits past its FND cutoff.

    Financial roots (fnd_offset_days == 0) are returned unchanged. The clamp
    only ever moves a roll EARLIER (never later); trigger becomes "fnd_clamp"
    when it fires.
    """
    spec = get_spec(root)
    if spec.settlement_type == "financial" or spec.fnd_offset_days == 0:
        return rolls
    out: list[RollEvent] = []
    for ev in rolls:
        exp = expirations.get(ev.from_symbol)
        if exp is None:
            out.append(ev)
            continue
        cutoff = _minus_business_days(exp, spec.fnd_offset_days)
        if ev.roll_date > cutoff:
            out.append(RollEvent(cutoff, ev.from_symbol, ev.to_symbol, "fnd_clamp"))
        else:
            out.append(ev)
    return out


class RollCalendar:
    """Lookup API over cached per-root roll calendars.

    Cache schema (futures/roll_calendar/{root}.parquet): one row per date with
    [date, front_symbol, next_cycle_symbol, next_oi_symbol, dte_front].
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir if cache_dir is not None else roll_calendar_dir()
        self._cache: dict[str, pl.DataFrame] = {}

    def _load(self, root: str) -> pl.DataFrame:
        if root in self._cache:
            return self._cache[root]
        path = self._dir / f"{root}.parquet"
        if not path.exists():
            raise NoActiveContractError(f"no roll calendar built for {root}: {path}")
        df = pl.read_parquet(path)
        self._cache[root] = df
        return df

    def _row(self, root: str, on: date) -> dict:
        df = self._load(root)
        matched = df.filter(pl.col("date") == on)
        if matched.is_empty():
            raise NoActiveContractError(f"no active contract for {root} on {on}")
        return matched.row(0, named=True)

    def get_front(self, root: str, on: date) -> ContractRef:
        r = self._row(root, on)
        return ContractRef(r["front_symbol"], r["front_expiration"], r["front_activation"])

    def get_nth_by_cycle(self, root: str, on: date, n: int) -> ContractRef:
        if n not in (0, 1):
            raise ValueError(f"get_nth_by_cycle supports n in {{0,1}} in v1, got {n}")
        r = self._row(root, on)
        sym = r["front_symbol"] if n == 0 else r["next_cycle_symbol"]
        return ContractRef(sym, r["front_expiration"], r["front_activation"])

    def get_nth_by_oi(self, root: str, on: date, n: int) -> ContractRef:
        if n not in (0, 1):
            raise ValueError(f"get_nth_by_oi supports n in {{0,1}} in v1, got {n}")
        r = self._row(root, on)
        sym = r["front_symbol"] if n == 0 else r["next_oi_symbol"]
        return ContractRef(sym, r["front_expiration"], r["front_activation"])

    def days_to_expiry(self, root: str, on: date) -> int:
        return int(self._row(root, on)["dte_front"])

    def settlement_type(self, root: str) -> str:
        return get_spec(root).settlement_type

    def roll_events(self, root: str) -> list[RollEvent]:
        df = self._load(root).sort("date")
        events: list[RollEvent] = []
        prev = None
        for r in df.iter_rows(named=True):
            if prev is not None and r["front_symbol"] != prev["front_symbol"]:
                events.append(RollEvent(
                    r["date"], prev["front_symbol"], r["front_symbol"],
                    r.get("roll_trigger", "oi_crossover"),
                ))
            prev = r
        return events
