"""Spot-FX reference strategies.

Both are price-only forecast_panel strategies, so they reuse the futures
forecast logic unchanged: FX trend = Carver multi-speed EWMAC; FX value =
Asness nominal 5yr-to-1yr reversal. Thin subclasses keep the FX names distinct
in the registry and leave room to diverge (e.g. a future PPP value signal).
"""
from __future__ import annotations

from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
from src.strategies.advanced.futures_value_strategy import FuturesValueStrategy


class FxTrendStrategy(CarverMomentumStrategy):
    pass


class FxValueStrategy(FuturesValueStrategy):
    pass
