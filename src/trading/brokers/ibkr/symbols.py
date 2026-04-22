"""
Symbol format normalization at the IBKR boundary.

Our strategies and universe files use dot format for multi-class shares
(BRK.B, BF.B) -- matching Alpaca, yfinance, and our data on disk. IBKR
expects space-separated class suffixes (BRK B, BF B) and will fail to
resolve the dot form.

These helpers translate only at the IBKR interface boundary:
    - Outbound (domain -> IB): `to_ibkr_symbol` before constructing a
      Contract or calling a method that takes a raw symbol string.
    - Inbound (IB -> domain): `from_ibkr_symbol` when reading
      `contract.symbol` off an IB Position or Trade before returning
      the symbol to strategy-level code.

Scope:
    - Multi-class US equity tickers only (pattern: 1-5 letters, dot or
      space, one letter suffix). `AAPL`, `SPY`, `V` pass through
      unchanged. Unknown patterns pass through unchanged -- we don't
      case-fold, strip whitespace, or guess.
"""

from __future__ import annotations

import re

# Matches domain format: 'BRK.B', 'BF.B', 'BRK.A'. Not 'A.B.C' (two dots)
# and not 'BRK.BB' (suffix must be exactly one letter).
_DOMAIN_MULTI_CLASS = re.compile(r"^([A-Z]{1,5})\.([A-Z])$")

# Matches IBKR format: 'BRK B', 'BF B'. Single space, single letter suffix.
_IBKR_MULTI_CLASS = re.compile(r"^([A-Z]{1,5}) ([A-Z])$")


def to_ibkr_symbol(symbol: str) -> str:
    """Convert a domain-format symbol to IBKR format.

    'BRK.B' -> 'BRK B'. Non-multi-class symbols pass through unchanged.
    """
    m = _DOMAIN_MULTI_CLASS.match(symbol)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return symbol


def from_ibkr_symbol(symbol: str) -> str:
    """Convert an IBKR-format symbol to domain format.

    'BRK B' -> 'BRK.B'. Non-multi-class symbols pass through unchanged.
    """
    m = _IBKR_MULTI_CLASS.match(symbol)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return symbol
