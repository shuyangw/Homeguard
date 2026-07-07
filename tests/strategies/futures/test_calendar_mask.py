import numpy as np
import pandas as pd
from src.strategies.advanced.futures_seasonal_strategy import FuturesTurnOfMonthStrategy


def test_long_only_on_turn_of_month_window():
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="B")
    close = pd.DataFrame({"ES": np.arange(len(idx), dtype=float) + 100}, index=idx)
    strat = FuturesTurnOfMonthStrategy(["ES"], cap=20.0)
    fc = strat.forecast_panel(close)
    # last business day of Jan 2021 is 2021-01-29; it must be active (+cap)
    assert fc.loc["2021-01-29", "ES"] == 20.0
    # a mid-month day (2021-02-16) must be flat
    assert fc.loc["2021-02-16", "ES"] == 0.0
    # forecast is only 0 or +cap (long-only)
    assert set(np.unique(fc["ES"].values)) <= {0.0, 20.0}


def test_first_three_days_of_month_active():
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="B")
    close = pd.DataFrame({"ES": np.ones(len(idx))}, index=idx)
    strat = FuturesTurnOfMonthStrategy(["ES"], cap=20.0)
    fc = strat.forecast_panel(close)
    # first 3 business days of Feb 2021: 1,2,3 Feb -> active
    for d in ["2021-02-01", "2021-02-02", "2021-02-03"]:
        assert fc.loc[d, "ES"] == 20.0
    # fourth business day 2021-02-04 -> flat
    assert fc.loc["2021-02-04", "ES"] == 0.0
