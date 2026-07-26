"""Range-based indicators (ATR, ADX, Parkinson RV) -- all strictly causal.

These require the intraday HIGH/LOW and therefore could not be expressed until
the FX engine stopped discarding open/high/low before calling a strategy
(`wants_ohlc`, 2026-07-25). Every function here uses trailing windows only: the
value at bar t depends on bars <= t and is never revised by later data.

Wilder's smoothing (ewm alpha=1/n, adjust=False) is used for ATR and ADX, which
is the textbook definition -- not a tuned choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(H-L, |H - prev_close|, |L - prev_close|). Causal (uses prev close)."""
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    return true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index (trend STRENGTH, 0-100, direction-agnostic).

    +DM = up_move when it exceeds down_move and is positive, else 0 (mirrored for
    -DM); both Wilder-smoothed and normalised by ATR to give +DI/-DI; DX is their
    normalised spread, and ADX is the Wilder-smoothed DX.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr_n = atr(high, low, close, n)
    sm = lambda s: s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * sm(plus_dm) / atr_n.replace(0, np.nan)
    minus_di = 100.0 * sm(minus_dm) / atr_n.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return sm(dx)


def parkinson_rv(high: pd.Series, low: pd.Series, n: int = 10,
                 annualization_factor: float = 252.0) -> pd.Series:
    """Parkinson high-low range volatility.

    sqrt( mean(ln(H/L)^2) / (4 ln 2) ), annualized. Roughly 5x more efficient
    than a close-to-close estimator on the same sample, which is the point of
    using it rather than rolling std of returns.
    """
    ratio = np.log(high.astype(float) / low.astype(float)) ** 2
    var = ratio.rolling(n, min_periods=n).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * annualization_factor)


def pair_ohlc(panel: pd.DataFrame, pair: str):
    """(high, low, close) for `pair` from a (pair, field) MultiIndex panel."""
    return (panel[(pair, "high")].astype(float),
            panel[(pair, "low")].astype(float),
            panel[(pair, "close")].astype(float))
