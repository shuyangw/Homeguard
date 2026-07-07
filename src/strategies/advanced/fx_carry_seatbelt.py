"""FxCarrySeatbelt: filtered long-carry book + carry-unwind veto/short (#16 + #19).

Replaces the naive FxCarry (held every pair through every crash, failed the
gate). Long a pair only when its rate-differential carry proxy exceeds +2%
annualized AND price momentum agrees (close > EMA(50) with positive 10-day EMA
slope); flat otherwise (never short for carry). A reusable carry-unwind composite
score zeroes all longs on risk-off days (defensive veto) and adds half-size
shorts on AUDJPY/NZDJPY during a detected cascade (offensive leg). All signals
are causal. Forecasts are Carver-scaled (10 = 1x vol-target).
"""
from __future__ import annotations

import pandas as pd

from src.backtesting.signals.carry_unwind import compute_unwind_score


class FxCarrySeatbelt:
    def __init__(self, universe, carry_gate: float = 0.02, ema_span: int = 50,
                 slope_lookback: int = 10, veto_threshold: float = 1.0,
                 veto_clear_days: int = 3, short_threshold: float = 2.5,
                 short_low_lookback: int = 20, full_forecast: float = 10.0,
                 short_forecast: float = 5.0, z_window: int = 252, **params):
        self.universe = list(universe)
        self.carry_gate = float(carry_gate)
        self.ema_span = int(ema_span)
        self.slope_lookback = int(slope_lookback)
        self.veto_threshold = float(veto_threshold)
        self.veto_clear_days = int(veto_clear_days)
        self.short_threshold = float(short_threshold)
        self.short_low_lookback = int(short_low_lookback)
        self.full_forecast = float(full_forecast)
        self.short_forecast = float(short_forecast)
        self.z_window = int(z_window)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        from src.data.fx_rates import (load_fx_rate_panel, build_rate_diff_panel,
                                        currencies_for_pairs)
        present = [p for p in self.universe if p in close_panel.columns]
        rate_panel = load_fx_rate_panel(currencies_for_pairs(present), close_panel.index)
        rate_diff = build_rate_diff_panel(present, rate_panel)

        close = close_panel[present].astype(float)
        ema = close.ewm(span=self.ema_span, adjust=False).mean()
        momentum_ok = (close > ema) & (ema > ema.shift(self.slope_lookback))
        carry_ok = rate_diff[present] > self.carry_gate
        longs = (carry_ok & momentum_ok).astype(float) * self.full_forecast

        score = compute_unwind_score(close_panel, z_window=self.z_window)
        veto = self._veto_mask(score)
        longs.loc[veto.values, :] = 0.0

        out = longs
        cascade = score > self.short_threshold
        for pair in ("AUDJPY", "NZDJPY"):
            if pair in out.columns:
                prior_low = close_panel[pair].rolling(self.short_low_lookback).min().shift(1)
                fire = cascade & (close_panel[pair] < prior_low)
                out.loc[fire.values, pair] = -self.short_forecast
        return out.fillna(0.0)

    def _veto_mask(self, score: pd.Series) -> pd.Series:
        engaged, run_below, mask = False, 0, []
        for v in score.values:
            if v >= self.veto_threshold:
                engaged, run_below = True, 0
            else:
                run_below += 1
                if run_below >= self.veto_clear_days:
                    engaged = False
            mask.append(engaged)
        return pd.Series(mask, index=score.index)
