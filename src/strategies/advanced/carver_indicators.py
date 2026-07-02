"""Carver EWMAC forecasts (parameter-free by design).

Speeds (4,16),(16,64),(64,256) and cap 20 are DOCTRINE (Carver, Systematic
Trading). NEVER expose to optimization. Forecast scalars are Table 19 constants.
"""
from __future__ import annotations

import pandas as pd

FORECAST_SCALARS: dict[tuple[int, int], float] = {
    (4, 16): 10.6,
    (16, 64): 6.49,
    (64, 256): 3.75,
}


def ewmac_forecast(prices: pd.Series, n_fast: int, n_slow: int,
                   daily_price_vol: pd.Series, cap: float = 20.0) -> pd.Series:
    raw = prices.ewm(span=n_fast).mean() - prices.ewm(span=n_slow).mean()
    normalized = raw / daily_price_vol.replace(0, pd.NA)
    scalar = FORECAST_SCALARS[(n_fast, n_slow)]
    return (normalized * scalar).clip(-cap, cap)


def combined_forecast(prices: pd.Series, daily_price_vol: pd.Series,
                      speeds: list[tuple[int, int]], cap: float = 20.0) -> pd.Series:
    forecasts = [ewmac_forecast(prices, f, s, daily_price_vol, cap) for f, s in speeds]
    combined = sum(forecasts) / len(forecasts)
    return combined.clip(-cap, cap)
