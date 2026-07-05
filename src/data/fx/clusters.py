"""Coarse FX clusters for IDM diversification weighting.

Groups pairs by dominant risk driver so the Instrument Diversification
Multiplier gives each cluster an equal risk budget. Deterministic, data-free.
"""
from __future__ import annotations

_METALS = {"XAU", "XAG"}
_EM = {"BRL", "CNH", "CLP", "CZK", "HKD", "HUF", "ILS", "INR", "KRW",
       "MXN", "PLN", "RUB", "TRY", "ZAR", "SGD"}


def fx_cluster_for(pair: str) -> str:
    base, quote = pair[:3], pair[3:]
    if base in _METALS or quote in _METALS:
        return "metal"
    if base in _EM or quote in _EM:
        return "em"
    if "USD" in (base, quote):
        return "usd_major"
    if "EUR" in (base, quote):
        return "eur_cross"
    if "JPY" in (base, quote):
        return "jpy_cross"
    return "other_cross"
