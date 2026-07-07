"""Carry-unwind composite risk-off score (research #19).

A single daily score, higher = more cascade-like. Built from four causal,
trailing-z-scored terms: JPY strength change, CHF strength change (both funding
currencies that appreciate in an unwind), AUDJPY short-horizon realized vol, and
XAUUSD 3-day return (gold bid). Designed as a shared risk-off brain reusable by
#15/#16/#18/#42; kept dependency-free (pure functions on a close panel) so any
strategy can call it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = s.rolling(window, min_periods=max(window // 2, 2)).std()
    # Floor std so a near-constant (float-noise-only) window does not produce
    # exploded z-scores; NaN std (insufficient window) falls through to fillna.
    std = std.clip(lower=1e-10)
    z = (s - mean) / std
    return z.fillna(0.0)


def currency_strength(close_panel: pd.DataFrame) -> pd.DataFrame:
    rets = close_panel.pct_change(fill_method=None).fillna(0.0)
    contrib: dict[str, list[pd.Series]] = {}
    for pair in rets.columns:
        base, quote = pair[:3], pair[3:]
        contrib.setdefault(base, []).append(rets[pair])
        contrib.setdefault(quote, []).append(-rets[pair])
    strength = {ccy: pd.concat(series, axis=1).mean(axis=1).cumsum()
                for ccy, series in contrib.items()}
    return pd.DataFrame(strength)


def compute_unwind_score(close_panel: pd.DataFrame, z_window: int = 252) -> pd.Series:
    idx = close_panel.index
    strength = currency_strength(close_panel)

    def delta_strength(ccy: str) -> pd.Series:
        if ccy not in strength.columns:
            return pd.Series(0.0, index=idx)
        return strength[ccy].diff(3).fillna(0.0)

    # Our strength convention: appreciation -> strength rises. JPY/CHF appreciate
    # in an unwind, so their positive strength-delta enters with a POSITIVE sign.
    jpy_term = _trailing_zscore(delta_strength("JPY"), z_window)
    chf_term = _trailing_zscore(delta_strength("CHF"), z_window)

    if "AUDJPY" in close_panel.columns:
        audjpy_vol = (close_panel["AUDJPY"].pct_change(fill_method=None)
                      .rolling(5, min_periods=3).std().fillna(0.0))
    else:
        audjpy_vol = pd.Series(0.0, index=idx)
    vol_term = _trailing_zscore(audjpy_vol, z_window)

    if "XAUUSD" in close_panel.columns:
        gold_ret = close_panel["XAUUSD"].pct_change(3, fill_method=None).fillna(0.0)
    else:
        gold_ret = pd.Series(0.0, index=idx)
    gold_term = _trailing_zscore(gold_ret, z_window)

    score = jpy_term + chf_term + vol_term + gold_term
    return score.fillna(0.0)
