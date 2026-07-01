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
from datetime import date


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
    """Contract with the highest OI on a day, or None if empty."""
    if not day_oi:
        return None
    return max(day_oi, key=day_oi.get)


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
