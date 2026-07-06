"""Curated central-bank decision calendar for FX macro-event blackouts.

`load_cb_decisions` is the single source of truth for CB decision dates. It
currently reads a curated, version-controlled yaml (`config/macro_calendar/
cb_decisions.yaml`) with no external API dependency (keyless). This is a
documented seam: a future API-backed feed (e.g. a central-bank calendar
provider) can replace this function's body while keeping the same
`dict[str, list[date]]` return type, so callers of `load_cb_decisions()` and
`blackout()` never need to change.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

_CB_FOR_CCY = {
    "EUR": "ECB", "GBP": "BOE", "JPY": "BOJ", "CHF": "SNB",
    "AUD": "RBA", "NZD": "RBNZ", "NOK": "NORGES", "SEK": "RIKSBANK",
    "MXN": "BANXICO", "USD": "FOMC",
}
_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_calendar" / "cb_decisions.yaml"


def load_cb_decisions() -> dict[str, list[date]]:
    with open(_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, list[date]] = {}
    for bank, dates in raw.items():
        out[bank] = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in dates]
    return out


def blackout(currency: str, day: date, days: int = 1) -> bool:
    bank = _CB_FOR_CCY.get(currency)
    if bank is None:
        return False
    decisions = load_cb_decisions().get(bank, [])
    return any(abs((day - dd).days) <= days for dd in decisions)
