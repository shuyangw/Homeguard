"""ExpirationGuard: refuse trades on contracts too close to expiration.

Per-family thresholds based on asset class characteristics:
- Rates have short rolls (high liquidity into next contract)
- Equity index gets a wider window (higher vol; safer to roll early)
- Livestock (physical delivery) gets the widest window to avoid forced delivery

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.3.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from src.trading.futures.position import FuturesPosition


# Days before expiration where new positions are disallowed.
EXPIRATION_THRESHOLDS: dict[str, int] = {
    # Equity index -- high vol, more roll lead time
    "ES": 5, "NQ": 5, "YM": 5, "RTY": 5,
    "MES": 5, "MNQ": 5, "M2K": 5, "MYM": 5,
    # Rates -- lower vol, shorter roll window
    "ZT": 2, "ZF": 2, "ZN": 2, "TN": 2, "ZB": 2, "UB": 2,
    "SR3": 2, "SR1": 2,
    "10Y": 2, "30Y": 2, "5YY": 2, "2YY": 2,
    # Energy
    "CL": 3, "NG": 3, "HO": 3, "RB": 3, "BZ": 3,
    "MCL": 3, "MNG": 3,
    # Metals
    "GC": 3, "SI": 3, "HG": 3, "PL": 3,
    "MGC": 3, "SIL": 3,
    # FX
    "6E": 3, "6J": 3, "6B": 3, "6A": 3, "6C": 3, "6S": 3, "6N": 3, "6M": 3,
    # Agriculture
    "ZC": 3, "ZS": 3, "ZW": 3, "KE": 3, "ZL": 3, "ZM": 3,
    # Livestock -- physical delivery, wider buffer
    "LE": 5, "HE": 5,
    # Crypto
    "BTC": 3, "MBT": 3, "ETH": 3, "MET": 3,
    # Default for unknown roots
    "_DEFAULT": 5,
}


class ExpirationVerdict(Enum):
    """Verdict on whether trading on a contract is allowed."""

    OK = "ok"                          # > threshold days; safe to trade
    WARN = "warn"                      # <= threshold but > 1 day; no new entries
    MUST_ROLL_OR_CLOSE = "must_act"    # <= 1 day with position; force action
    EXPIRED = "expired"                # past expiration; emergency reconciliation


def _today() -> date:
    """Return today's date. Indirection for test monkeypatching."""
    return date.today()


class ExpirationGuard:
    """Pre-trade gate based on contract expiration proximity."""

    def _threshold(self, symbol_root: str) -> int:
        """Get the threshold in days for a given symbol root."""
        return EXPIRATION_THRESHOLDS.get(symbol_root, EXPIRATION_THRESHOLDS["_DEFAULT"])

    def check_new_entry_with_expiration(
        self, symbol_root: str, expiration_date: date,
    ) -> ExpirationVerdict:
        """Called before opening a new position.

        Returns OK if safe; WARN/EXPIRED otherwise. WARN means existing
        positions still permitted but no NEW entries.
        """
        days = (expiration_date - _today()).days
        if days <= 0:
            return ExpirationVerdict.EXPIRED
        threshold = self._threshold(symbol_root)
        if days <= threshold:
            return ExpirationVerdict.WARN
        return ExpirationVerdict.OK

    def check_existing_position(
        self, position: FuturesPosition,
    ) -> ExpirationVerdict:
        """Called by pre-flight reconciliation for every open position."""
        days = (position.expiration_date - _today()).days
        if days <= 0:
            return ExpirationVerdict.EXPIRED
        if days <= 1:
            return ExpirationVerdict.MUST_ROLL_OR_CLOSE
        threshold = self._threshold(position.symbol_root)
        if days <= threshold:
            return ExpirationVerdict.WARN
        return ExpirationVerdict.OK
