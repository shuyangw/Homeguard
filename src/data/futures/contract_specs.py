"""Static, hand-verified futures contract specifications.

Source of truth for multiplier / tick / settlement type -- the definitions
dataset's contract_multiplier is an unreliable i32 sentinel, so multipliers
MUST come from here. settlement_type + fnd_offset_days drive the FND clamp in
the roll calendar: physical roots roll before first notice; financial
(cash-settled) roots have no delivery risk (fnd_offset_days == 0).

fnd_offset_days is an APPROXIMATE, conservative business-day cushion before
`expiration` past which a physical contract must not remain front. It only
ever moves a roll EARLIER. Refine per family if golden-date tests disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Settlement = Literal["physical", "financial"]


@dataclass(frozen=True)
class ContractSpec:
    root: str
    multiplier: float
    tick_size: float
    tick_value: float
    currency: str
    cycle_months: str        # subset of FGHJKMNQUVXZ that lists liquid contracts
    settlement_type: Settlement
    fnd_offset_days: int     # business-day cushion before expiration; 0 for financial


def _s(root, mult, tick, tick_val, ccy, cycle, settle, fnd):
    return ContractSpec(root, mult, tick, tick_val, ccy, cycle, settle, fnd)


ALL = "FGHJKMNQUVXZ"
QTR = "HMUZ"   # quarterly cycle (equity index, rates, FX)

SPECS: dict[str, ContractSpec] = {
    # Equity index -- cash settled (financial)
    "ES": _s("ES", 50.0, 0.25, 12.5, "USD", QTR, "financial", 0),
    "NQ": _s("NQ", 20.0, 0.25, 5.0, "USD", QTR, "financial", 0),
    "YM": _s("YM", 5.0, 1.0, 5.0, "USD", QTR, "financial", 0),
    "RTY": _s("RTY", 50.0, 0.10, 5.0, "USD", QTR, "financial", 0),
    "MES": _s("MES", 5.0, 0.25, 1.25, "USD", QTR, "financial", 0),
    "MNQ": _s("MNQ", 2.0, 0.25, 0.5, "USD", QTR, "financial", 0),
    "M2K": _s("M2K", 5.0, 0.10, 0.5, "USD", QTR, "financial", 0),
    "MYM": _s("MYM", 0.5, 1.0, 0.5, "USD", QTR, "financial", 0),
    # Energy -- physical
    "CL": _s("CL", 1000.0, 0.01, 10.0, "USD", ALL, "physical", 4),
    "NG": _s("NG", 10000.0, 0.001, 10.0, "USD", ALL, "physical", 4),
    "HO": _s("HO", 42000.0, 0.0001, 4.2, "USD", ALL, "physical", 4),
    "RB": _s("RB", 42000.0, 0.0001, 4.2, "USD", ALL, "physical", 4),
    "BZ": _s("BZ", 1000.0, 0.01, 10.0, "USD", ALL, "physical", 4),
    "MCL": _s("MCL", 100.0, 0.01, 1.0, "USD", ALL, "physical", 4),
    "MNG": _s("MNG", 2500.0, 0.001, 2.5, "USD", ALL, "physical", 4),
    # Metals -- physical
    "GC": _s("GC", 100.0, 0.1, 10.0, "USD", "GJMQVZ", "physical", 3),
    "SI": _s("SI", 5000.0, 0.005, 25.0, "USD", "HKNUZ", "physical", 3),
    "HG": _s("HG", 25000.0, 0.0005, 12.5, "USD", "HKNUZ", "physical", 3),
    "PL": _s("PL", 50.0, 0.1, 5.0, "USD", "FJNV", "physical", 3),
    "MGC": _s("MGC", 10.0, 0.1, 1.0, "USD", "GJMQVZ", "physical", 3),
    "SIL": _s("SIL", 1000.0, 0.005, 5.0, "USD", "HKNUZ", "physical", 3),
    # Rates -- physical delivery (bonds) / financial (SOFR, micro yield)
    "ZT": _s("ZT", 2000.0, 0.0078125, 15.625, "USD", QTR, "physical", 2),
    "ZF": _s("ZF", 1000.0, 0.0078125, 7.8125, "USD", QTR, "physical", 2),
    "ZN": _s("ZN", 1000.0, 0.015625, 15.625, "USD", QTR, "physical", 2),
    "TN": _s("TN", 1000.0, 0.015625, 15.625, "USD", QTR, "physical", 2),
    "ZB": _s("ZB", 1000.0, 0.03125, 31.25, "USD", QTR, "physical", 2),
    "UB": _s("UB", 1000.0, 0.03125, 31.25, "USD", QTR, "physical", 2),
    "SR3": _s("SR3", 2500.0, 0.005, 12.5, "USD", QTR, "financial", 0),
    "SR1": _s("SR1", 4167.0, 0.005, 20.835, "USD", ALL, "financial", 0),
    "10Y": _s("10Y", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "30Y": _s("30Y", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "5YY": _s("5YY", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "2YY": _s("2YY", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    # FX -- physically deliverable currency
    "6E": _s("6E", 125000.0, 0.00005, 6.25, "USD", QTR, "physical", 2),
    "6J": _s("6J", 12500000.0, 0.0000005, 6.25, "USD", QTR, "physical", 2),
    "6B": _s("6B", 62500.0, 0.0001, 6.25, "USD", QTR, "physical", 2),
    "6A": _s("6A", 100000.0, 0.0001, 10.0, "USD", QTR, "physical", 2),
    "6C": _s("6C", 100000.0, 0.00005, 5.0, "USD", QTR, "physical", 2),
    "6S": _s("6S", 125000.0, 0.0001, 12.5, "USD", QTR, "physical", 2),
    "6N": _s("6N", 100000.0, 0.0001, 10.0, "USD", QTR, "physical", 2),
    "6M": _s("6M", 500000.0, 0.00001, 5.0, "USD", QTR, "physical", 2),
    # Ag -- physical
    "ZC": _s("ZC", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "ZS": _s("ZS", 50.0, 0.25, 12.5, "USD", "FHKNQUX", "physical", 2),
    "ZW": _s("ZW", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "KE": _s("KE", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "ZL": _s("ZL", 600.0, 0.01, 6.0, "USD", "FHKNQUVZ", "physical", 2),
    "ZM": _s("ZM", 100.0, 0.1, 10.0, "USD", "FHKNQUVZ", "physical", 2),
    "LE": _s("LE", 40000.0, 0.00025, 10.0, "USD", "GJMQVZ", "physical", 2),
    "HE": _s("HE", 40000.0, 0.00025, 10.0, "USD", "GJKMNQVZ", "physical", 2),
    # Crypto -- cash settled (financial)
    "BTC": _s("BTC", 5.0, 5.0, 25.0, "USD", ALL, "financial", 0),
    "MBT": _s("MBT", 0.1, 5.0, 0.5, "USD", ALL, "financial", 0),
    "ETH": _s("ETH", 50.0, 0.5, 25.0, "USD", ALL, "financial", 0),
    "MET": _s("MET", 0.1, 0.5, 0.05, "USD", ALL, "financial", 0),
}


def get_spec(root: str) -> ContractSpec:
    """Return the ContractSpec for `root`, or raise KeyError."""
    return SPECS[root]
