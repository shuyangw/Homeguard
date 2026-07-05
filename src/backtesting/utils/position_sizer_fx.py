"""Notional (base-currency-unit) FX position sizer.

Carver forecast -> vol-targeted notional, in units of the base currency. The
risk of holding one base unit is base_to_usd * annualized_return_vol (its USD
standard deviation), so dividing the USD risk budget by that term targets equal
USD risk per instrument. Unlike the futures sizer there is no contract
multiplier or integer/contract cap -- FX trades in continuous notional and the
leverage cap is enforced at the portfolio level.
"""
from __future__ import annotations


def size_from_forecast_fx(forecast: float, capital: float, vol_target: float,
                          base_to_usd: float, daily_vol: float,
                          div_mult: float = 1.0) -> float:
    if daily_vol <= 0 or base_to_usd <= 0 or vol_target <= 0:
        return 0.0
    ann_vol = daily_vol * (252 ** 0.5)
    denom = base_to_usd * ann_vol
    if denom <= 0:
        return 0.0
    return (forecast / 10.0) * capital * vol_target * div_mult / denom
