"""HAR (Corsi 2009) realized-volatility forecast from 1-min bars.

daily RV = sum of squared 1-min log returns. HAR regresses next-day RV on daily,
weekly (5d mean), monthly (22d mean) lagged RV, fit causally (coefficients use
only strictly-prior data). Forecast is next-day E[RV] in daily-variance units."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.continuous_contract_loader import ContinuousContractDataLoader


def daily_realized_variance(root: str, start: date, end: date) -> pd.Series:
    df = ContinuousContractDataLoader().load(root, method="ratio_adjusted", start=start, end=end)
    pdf = df.select(["timestamp", "close"]).to_pandas()
    pdf["logret"] = np.log(pdf["close"]).diff()
    pdf["d"] = pd.to_datetime(pdf["timestamp"]).dt.date
    # first bar of each trading day is an overnight/weekend gap, not a within-session
    # return -- exclude it so RV isn't inflated by the close-to-open jump
    is_first_of_day = pdf["d"] != pdf["d"].shift(1)
    pdf.loc[is_first_of_day, "logret"] = np.nan
    rv = pdf.groupby("d")["logret"].apply(lambda s: float(np.nansum(s.values**2)))
    rv.index = pd.to_datetime(rv.index)
    return rv.rename(f"{root}_rv").sort_index()


def _har_design(rv: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    d = rv
    w = rv.rolling(5).mean()
    m = rv.rolling(22).mean()
    X = pd.DataFrame({"const": 1.0, "d": d, "w": w, "m": m})
    y = rv.shift(-1)  # next-day RV is the target
    return X, y


def har_forecast(rv_daily: pd.Series, min_train: int = 252) -> pd.Series:
    rv_daily = rv_daily.dropna()
    X, y = _har_design(rv_daily)
    out = pd.Series(np.nan, index=rv_daily.index)
    Xv = X.values
    yv = y.values
    for i in range(min_train, len(rv_daily)):
        # fit on rows [22 .. i-1] whose target y (=rv[t+1]) is known and features non-NaN
        lo = 22
        Xtr = Xv[lo:i]
        ytr = yv[lo:i]
        good = ~np.isnan(Xtr).any(axis=1) & ~np.isnan(ytr)
        if good.sum() < 50:
            continue
        beta, *_ = np.linalg.lstsq(Xtr[good], ytr[good], rcond=None)
        xi = Xv[i]
        if np.isnan(xi).any():
            continue
        out.iloc[i] = max(float(xi @ beta), 0.0)
    return out.rename("har_forecast")


def har_forecast_vol_annualized(root: str, start: date, end: date) -> pd.Series:
    rv = daily_realized_variance(root, start, end)
    fc = har_forecast(rv)
    return np.sqrt(fc * 252.0).rename(f"{root}_har_vol")
