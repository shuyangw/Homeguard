import numpy as np
import pandas as pd
from src.strategies.advanced.futures_momentum_strategy import FuturesXSMomentumStrategy


def test_momentum_ranks_higher_trailing_return_long():
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    # CL trends up, NG flat -> CL has higher 12-1 return -> long CL, short NG
    close = pd.DataFrame({
        "CL": np.linspace(50, 100, 300),
        "NG": np.full(300, 3.0),
    }, index=idx)
    strat = FuturesXSMomentumStrategy(["CL", "NG"])
    fc = strat.forecast_panel(close)
    last = fc.iloc[-1]
    assert last["CL"] > 0 > last["NG"]
    assert fc.abs().max().max() <= 20.0


def test_skip_month_excludes_last_21_days():
    # CL is the clear 12-1 momentum winner; a crash only in the final 21 days must
    # NOT flip its long ranking (the 12-1 window ends at t-21 and skips the crash).
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    cl = np.linspace(50, 100, 300).copy()  # strong uptrend
    cl[-21:] = 5.0                          # recent crash, but skip-month ignores it
    close = pd.DataFrame({"CL": cl, "NG": np.linspace(3, 3.5, 300)}, index=idx)  # NG near-flat
    strat = FuturesXSMomentumStrategy(["CL", "NG"])
    fc = strat.forecast_panel(close)
    # 12-1 return of CL (t-252..t-21) is still strongly positive vs near-flat NG
    assert fc.iloc[-1]["CL"] > 0 > fc.iloc[-1]["NG"]


def test_warmup_rows_are_nan_free_forecast_zero():
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    close = pd.DataFrame({"CL": np.linspace(50, 60, 300), "NG": np.linspace(3, 9, 300)}, index=idx)
    strat = FuturesXSMomentumStrategy(["CL", "NG"])
    fc = strat.forecast_panel(close)
    assert fc.iloc[0].eq(0.0).all()  # insufficient lookback -> no bet
