"""Parse databento futures-option symbols like 'ESH2 C4725'.

Format: <ROOT><MONTHCODE><YEARDIGIT> <C|P><STRIKE>. The year digit is the
last digit of the expiry year (CME convention) and is ambiguous across
decades on its own -- the options data spans 2010-2029. We resolve it using
the ref_year (the trading date the symbol was observed on): the expiry year
is the smallest year >= ref_year whose last digit matches, since an option
symbol observed on a given trading day always expires in the current or a
near-future year, never the past. When ref_year is not supplied we fall back
to the 2020s decade for hermetic unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data.carry_calculator import _MONTH_CODES

_ROOTS = ("ES", "NQ")


@dataclass(frozen=True)
class OptionSymbol:
    root: str
    expiry_year: int
    expiry_month: int
    right: str
    strike: float


def _resolve_year(digit: int, ref_year: int | None) -> int:
    if ref_year is None:
        return 2020 + digit
    year = ref_year - (ref_year % 10) + digit
    if year < ref_year:
        year += 10
    return year


def parse_option_symbol(sym: str, ref_year: int | None = None) -> OptionSymbol | None:
    if not sym or " " not in sym:
        return None
    left, right_part = sym.split(" ", 1)
    root = next((r for r in _ROOTS if left.startswith(r)), None)
    if root is None:
        return None
    suffix = left[len(root):]
    if len(suffix) != 2 or suffix[0] not in _MONTH_CODES or not suffix[1].isdigit():
        return None
    month = _MONTH_CODES.index(suffix[0]) + 1
    year = _resolve_year(int(suffix[1]), ref_year)
    if not right_part or right_part[0] not in ("C", "P"):
        return None
    try:
        strike = float(right_part[1:])
    except ValueError:
        return None
    return OptionSymbol(root, year, month, right_part[0], strike)
