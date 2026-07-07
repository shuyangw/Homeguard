import numpy as np
import pandas as pd
from src.strategies.advanced.futures_seasonal_strategy import FuturesSameMonthSeasonalityStrategy


def _seasonal_close(years, strong_month, strong_root, roots):
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="BME")
    data = {}
    for r in roots:
        base = np.ones(len(idx))
        rets = np.zeros(len(idx))
        if r == strong_root:
            rets[idx.month == strong_month] = 0.10  # +10% every strong_month
        data[r] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=idx)


def test_seasonally_strong_root_is_long_in_its_month():
    roots = ["NG", "CL"]
    close = _seasonal_close(range(2011, 2021), strong_month=10, strong_root="NG", roots=roots)
    strat = FuturesSameMonthSeasonalityStrategy(roots)
    fc = strat.forecast_panel(close)
    october_rows = fc[fc.index.month == 10]
    # in later Octobers (enough prior history) NG ranks long, CL short
    assert october_rows["NG"].iloc[-1] > 0 > october_rows["CL"].iloc[-1]
    assert fc.abs().max().max() <= 20.0


def test_uses_only_prior_years_causal():
    # first occurrence of the strong month has no prior history -> no bet (0)
    roots = ["NG", "CL"]
    close = _seasonal_close(range(2011, 2021), strong_month=10, strong_root="NG", roots=roots)
    strat = FuturesSameMonthSeasonalityStrategy(roots)
    fc = strat.forecast_panel(close)
    first_october = fc[fc.index.month == 10].iloc[0]
    assert first_october["NG"] == 0.0 and first_october["CL"] == 0.0
