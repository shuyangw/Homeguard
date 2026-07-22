"""Spot-FX reference strategies.

Both are price-only forecast_panel strategies, so they reuse the futures
forecast logic unchanged: FX trend = Carver multi-speed EWMAC; FX value =
Asness nominal 5yr-to-1yr reversal. Thin subclasses keep the FX names distinct
in the registry and leave room to diverge (e.g. a future PPP value signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
from src.strategies.advanced.futures_value_strategy import FuturesValueStrategy


class FxTrendStrategy(CarverMomentumStrategy):
    pass


class FxValueStrategy(FuturesValueStrategy):
    pass


class FxTSMOMStrategy:
    """Time-series momentum (#3, Moskowitz-Ooi-Pedersen).

    Forecast = scale * mean(sign(ret_short), sign(ret_long)): long when both the
    short and long trailing returns are positive, short when both negative, flat
    when they disagree. Vol-scaling comes from the engine's per-instrument
    vol-target sizing. Forecast is on the Carver scale (10 = full 1x position).
    """

    def __init__(self, universe, lookback_short: int = 63, lookback_long: int = 252,
                 scale: float = 10.0, **params):
        self.universe = list(universe)
        self.lookback_short = int(lookback_short)
        self.lookback_long = int(lookback_long)
        self.scale = float(scale)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for root in self.universe:
            if root not in close_panel.columns:
                continue
            c = close_panel[root].astype(float)
            s = np.sign(c.pct_change(self.lookback_short, fill_method=None))
            l = np.sign(c.pct_change(self.lookback_long, fill_method=None))
            out[root] = (self.scale * (s + l) / 2.0).fillna(0.0)
        cols = [r for r in self.universe if r in out]
        return pd.DataFrame(out)[cols]


class FxCarryStrategy:
    """Carry factor (#15). Forecast proportional to the annualized interest-rate
    differential (base minus quote); long high-carry pairs, short negative-carry.
    Loads its own FRED rate panel (forecast_panel only receives close). Vol-target
    sizing gives the vol-scaled carry basket. Forecast on the Carver scale.
    """

    def __init__(self, universe, scale: float = 500.0, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.scale = float(scale)
        self.cap = float(cap)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        from src.data.fx_rates import (load_fx_rate_panel, build_rate_diff_panel,
                                        currencies_for_pairs)
        present = [p for p in self.universe if p in close_panel.columns]
        rate_panel = load_fx_rate_panel(currencies_for_pairs(present), close_panel.index)
        rd = build_rate_diff_panel(present, rate_panel)  # date x pair, annualized decimal
        fc = (rd * self.scale).clip(-self.cap, self.cap)
        return fc[present].fillna(0.0)


class FxGoldSilverStrategy:
    """Gold/silver ratio reversion (#43). z-score the XAU/XAG ratio vs a rolling
    window; when the ratio is rich (gold expensive vs silver) short gold / long
    silver, and vice versa. Two-instrument RV.
    """

    def __init__(self, universe=("XAUUSD", "XAGUSD"), lookback: int = 756,
                 scale: float = 5.0, **params):
        self.universe = list(universe)
        self.lookback = int(lookback)
        self.scale = float(scale)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        if "XAUUSD" not in close_panel or "XAGUSD" not in close_panel:
            return pd.DataFrame(index=close_panel.index)
        ratio = close_panel["XAUUSD"].astype(float) / close_panel["XAGUSD"].astype(float)
        z = (ratio - ratio.rolling(self.lookback).mean()) / ratio.rolling(self.lookback).std()
        out = {"XAUUSD": (-z * self.scale).clip(-20, 20),
               "XAGUSD": (z * self.scale).clip(-20, 20)}
        return pd.DataFrame(out).fillna(0.0)


class FxCarryMomStrategy:
    """Carry + time-series-momentum blend (#5). Equal-weight sum of the FxCarry
    and FxTSMOM Carver-scale forecasts -- the classic carry+momentum
    diversification (the two factors are near-uncorrelated). Sizing is the
    engine's vol-target on the blended forecast.
    """

    def __init__(self, universe, carry_scale: float = 500.0, carry_cap: float = 20.0,
                 lookback_short: int = 63, lookback_long: int = 252, **params):
        self.universe = list(universe)
        self._carry = FxCarryStrategy(universe, scale=carry_scale, cap=carry_cap)
        self._mom = FxTSMOMStrategy(universe, lookback_short=lookback_short,
                                    lookback_long=lookback_long)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        fc_c = self._carry.forecast_panel(close_panel)
        fc_m = self._mom.forecast_panel(close_panel)
        cols = [c for c in fc_c.columns if c in fc_m.columns]
        blend = 0.5 * fc_c[cols] + 0.5 * fc_m[cols]
        return blend.fillna(0.0)


class FxMeanRevStrategy:
    """Single-instrument close-only mean reversion (#8/#12/#29 daily form).
    Fade the z-score of price vs its rolling mean: forecast = -z, so the position
    is short when price is stretched high and long when stretched low. Continuous
    Carver-scale forecast (the standard stateless form); vol-target sizing applies.
    """

    def __init__(self, universe, lookback: int = 60, scale: float = 4.0,
                 cap: float = 20.0, **params):
        self.universe = list(universe)
        self.lookback = int(lookback)
        self.scale = float(scale)
        self.cap = float(cap)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for root in self.universe:
            if root not in close_panel.columns:
                continue
            c = close_panel[root].astype(float)
            m = c.rolling(self.lookback).mean()
            sd = c.rolling(self.lookback).std()
            z = (c - m) / sd.replace(0, np.nan)
            out[root] = (-z * self.scale).clip(-self.cap, self.cap).fillna(0.0)
        cols = [r for r in self.universe if r in out]
        return pd.DataFrame(out)[cols]


class FxXSectMomStrategy:
    """Cross-sectional momentum (#4). Rank pairs by trailing risk-adjusted return
    each day; long the strongest, short the weakest (cross-sectional z of
    return/vol). Nets out common direction.
    """

    def __init__(self, universe, lookback: int = 63, vol_window: int = 63,
                 scale: float = 7.0, **params):
        self.universe = list(universe)
        self.lookback = int(lookback)
        self.vol_window = int(vol_window)
        self.scale = float(scale)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        present = [p for p in self.universe if p in close_panel.columns]
        c = close_panel[present].astype(float)
        mom = c.pct_change(self.lookback, fill_method=None)
        vol = c.pct_change(fill_method=None).rolling(self.vol_window).std()
        radj = mom / vol.replace(0, np.nan)
        z = radj.sub(radj.mean(axis=1), axis=0).div(radj.std(axis=1), axis=0)
        return (z * self.scale).clip(-20, 20).fillna(0.0)


class FxCotPositioningStrategy:
    """CFTC speculative-positioning (COT) signal. Forecast from weekly net%OI (signed
    bullish-the-pair, publication-lagged), computed on the weekly panel and forward-
    filled to daily. FORM selects the pre-registered mechanism (signs fixed a priori):
    contrarian_ts (fade crowded level), momentum_ts (follow positioning flow),
    contrarian_xs (cross-sectional fade the most-crowded). Vol-target sizing applies.
    """

    FORM = "contrarian_ts"

    def __init__(self, universe, z_window: int = 156, mom_horizon: int = 4,
                 scale: float = 5.0, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.z_window = int(z_window)
        self.mom_horizon = int(mom_horizon)
        self.scale = float(scale)
        self.cap = float(cap)

    def _weekly_forecast(self, w: pd.DataFrame) -> pd.DataFrame:
        if self.FORM == "contrarian_ts":
            z = (w - w.rolling(self.z_window).mean()) / w.rolling(self.z_window).std()
            return -z.clip(-2, 2) * self.scale
        if self.FORM == "momentum_ts":
            chg = w - w.shift(self.mom_horizon)
            z = (chg - chg.rolling(self.z_window).mean()) / chg.rolling(self.z_window).std()
            return z.clip(-2, 2) * self.scale
        if self.FORM == "contrarian_xs":
            z = w.sub(w.mean(axis=1), axis=0).div(w.std(axis=1).replace(0, np.nan), axis=0)
            return -z.clip(-2, 2) * self.scale
        raise ValueError(f"unknown COT form {self.FORM!r}")

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        from src.data.cot import load_cot_weekly_panel, to_daily
        present = [p for p in self.universe if p in close_panel.columns]
        weekly = load_cot_weekly_panel(present)
        if weekly.empty:
            return pd.DataFrame(0.0, index=close_panel.index, columns=present)
        cols = [c for c in present if c in weekly.columns]
        fc = to_daily(self._weekly_forecast(weekly[cols]), close_panel.index)
        return fc[cols].clip(-self.cap, self.cap).fillna(0.0)


class FxCotContrarianTS(FxCotPositioningStrategy):
    FORM = "contrarian_ts"


class FxCotMomentumTS(FxCotPositioningStrategy):
    FORM = "momentum_ts"


class FxCotContrarianXS(FxCotPositioningStrategy):
    FORM = "contrarian_xs"
