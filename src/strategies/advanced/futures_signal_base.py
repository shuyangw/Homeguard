"""Reusable daily-forecast signal templates for futures strategies.

Each base class implements the forecast_panel(close_panel) -> forecast_panel
protocol the futures engine calls. Subclasses supply only the signal-specific
hook; the base handles the shared vol-scaling / cross-sectional / masking /
clipping machinery. All statistics are causal (same-day cross-sectional or
strictly-prior time-series)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.data.futures.asset_class import asset_class_for

_XS_SCALE = 10.0  # doctrine: maps a same-day cross-sectional z-score to forecast units
_CAP = 20.0       # doctrine: Carver forecast cap


class CrossSectionalRankStrategy:
    """Rank a per-root raw signal cross-sectionally within groups.

    forecast = clip(z(raw - within_group_mean) * xs_scale, -cap, cap), where the
    mean and dispersion are same-day within-group stats (causal). Singleton or
    zero-dispersion groups contribute 0.0 (no relative-value bet)."""

    def __init__(self, universe, group_fn: Optional[Callable[[str], str]] = None,
                 xs_scale: float = _XS_SCALE, cap: float = _CAP, **params):
        self.universe = list(universe)
        self.group_fn = group_fn or asset_class_for
        self.xs_scale = float(xs_scale)
        self.cap = float(cap)

    def _raw_signal_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("subclass must supply the raw per-root signal panel")

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        raw = self._raw_signal_panel(close_panel)
        groups: dict[str, list[str]] = {}
        for r in self.universe:
            groups.setdefault(self.group_fn(r), []).append(r)
        out = pd.DataFrame(0.0, index=raw.index, columns=self.universe)
        for _, roots in groups.items():
            present = [r for r in roots if r in raw.columns]
            if len(present) < 2:
                continue
            block = raw[present]
            mean = block.mean(axis=1)
            std = block.std(axis=1)
            valid = block.notna().all(axis=1)
            z = block.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
            zero_dispersion = valid & std.eq(0.0)
            z = z.where(~zero_dispersion, 0.0)
            scaled = (z * self.xs_scale).clip(-self.cap, self.cap)
            all_nan = block.isna().all(axis=1)
            out[present] = scaled.where(~all_nan, 0.0)
        return out.reindex(columns=self.universe)


class CalendarMaskStrategy:
    """Daily on/off hold by a trading-calendar rule.

    forecast = sign(date) * cap on active days, 0 otherwise, broadcast across all
    universe roots. Subclass supplies _active_and_sign(index) -> Series in
    {-1, 0, +1}. The calendar is derived from close_panel.index (the traded days)
    so it is holiday-aware by construction."""

    def __init__(self, universe, cap: float = _CAP, **params):
        self.universe = list(universe)
        self.cap = float(cap)

    def _active_and_sign(self, index: pd.DatetimeIndex) -> pd.Series:
        raise NotImplementedError("subclass must supply per-date sign in {-1, 0, +1}")

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        index = close_panel.index
        sign = self._active_and_sign(index).reindex(index).fillna(0.0)
        col = (sign * self.cap).astype(float)
        return pd.DataFrame({r: col.values for r in self.universe}, index=index)


class ConditioningOverlayStrategy:
    """Combine a base forecast with a conditioning forecast.

    Instantiates two registered strategies (base + conditioning) over the same
    universe and combines their forecast panels via _combine (subclass hook).
    Used for gate/tilt interactions (carry-trend, vol-filter, OI). The combined
    signal must be validated by the marginal-Sharpe-vs-the-pair test, not a naive
    correlation screen."""

    def __init__(self, universe, base_name: str, cond_name: str,
                 base_params: Optional[dict] = None, cond_params: Optional[dict] = None,
                 cap: float = _CAP, **params):
        from src.strategies.registry import get_strategy_class
        self.universe = list(universe)
        self.cap = float(cap)
        self._base = get_strategy_class(base_name)(universe, **(base_params or {}))
        self._cond = get_strategy_class(cond_name)(universe, **(cond_params or {}))

    def _base_forecast(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        return self._base.forecast_panel(close_panel)

    def _cond_forecast(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        return self._cond.forecast_panel(close_panel)

    def _combine(self, base_fc: pd.DataFrame, cond_fc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("subclass must combine base and conditioning forecasts")

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        base_fc = self._base_forecast(close_panel)
        cond_fc = self._cond_forecast(close_panel).reindex(
            index=base_fc.index, columns=base_fc.columns)
        return self._combine(base_fc, cond_fc).clip(-self.cap, self.cap)
