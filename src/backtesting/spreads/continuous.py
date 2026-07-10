"""Continuous-forecast spread engine ([C] strategies #35, #36).

Holds a forecast-weighted position in a spread, vol-scaled to a target, and
unwinds as the forecast decays. Positions are causal (forecast.shift(1)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.spreads.construction import SpreadSeries


def zscore_mr_forecast(signal: pd.Series, window: int, cap: float = 2.0) -> pd.Series:
    mean = signal.rolling(window).mean().shift(1)
    std = signal.rolling(window).std().shift(1)
    z = (signal - mean) / std.replace(0.0, np.nan)
    return (-z).clip(-cap, cap)


def momentum_forecast(signal: pd.Series, lookback: int = 252, skip: int = 21,
                      cap: float = 2.0) -> pd.Series:
    raw = signal.shift(skip) - signal.shift(lookback)
    std = raw.rolling(lookback).std().shift(1)
    z = raw / std.replace(0.0, np.nan)
    return z.clip(-cap, cap)


def continuous_return_stream(spread: SpreadSeries, forecast: pd.Series,
                             cost_usd: float, target_vol: float = 0.15,
                             notional: float = 100_000.0) -> pd.Series:
    unit = spread.unit_return.reindex(forecast.index)
    unit_vol = unit.std()
    if not np.isfinite(unit_vol) or unit_vol <= 0:
        return pd.Series(dtype=float)
    vol_scalar = (target_vol / np.sqrt(252)) / unit_vol
    position = forecast.shift(1) * vol_scalar
    gross = position * unit
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_usd / notional)
    return (gross - cost).rename("return").dropna()
