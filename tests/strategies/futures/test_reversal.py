import numpy as np
import pandas as pd
from src.strategies.advanced.futures_momentum_strategy import FuturesReversalStrategy


def test_recent_up_move_gives_short_forecast():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    px = np.linspace(100, 101, 120).copy()
    px[-1] = 130.0  # sharp recent up move -> reversal wants short (negative forecast)
    close = pd.DataFrame({"ES": px}, index=idx)
    strat = FuturesReversalStrategy(["ES"])
    fc = strat.forecast_panel(close)
    assert fc.iloc[-1]["ES"] < 0
    assert fc.abs().max().max() <= 20.0


def test_flat_series_forecast_near_zero():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    close = pd.DataFrame({"ES": np.full(120, 100.0)}, index=idx)
    strat = FuturesReversalStrategy(["ES"])
    fc = strat.forecast_panel(close)
    assert abs(fc.iloc[-1]["ES"]) < 1e-9
