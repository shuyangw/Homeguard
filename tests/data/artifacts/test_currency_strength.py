import numpy as np
import pandas as pd
from src.data.artifacts.currency_strength import aggregate_currency_returns, currency_returns


def test_currency_returns_averages_pairs():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    close = pd.DataFrame({"EURUSD": [1.10, 1.21], "GBPUSD": [1.30, 1.30]}, index=idx)
    cr = currency_returns(close)
    # EUR appreciates ~10% vs USD on day 2; USD is the average of the inverses.
    assert cr.loc[idx[1], "EUR"] > 0
    assert "USD" in cr.columns


def test_aggregate_currency_returns_preserves_ret_after_gap():
    # NOKSEK has a calendar gap on day 2 (NaN ret from its own native index);
    # EURUSD has a valid return every day. Using native-index ret means
    # EURUSD's day-3 return must NOT be NaN-contaminated by NOKSEK's gap.
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    rets = pd.DataFrame(
        {"NOKSEK": [0.001, np.nan, 0.002], "EURUSD": [0.01, 0.02, 0.03]}, index=idx
    )
    cr = aggregate_currency_returns(rets)
    assert not np.isnan(cr.loc[idx[2], "EUR"])
    assert cr.loc[idx[2], "EUR"] == 0.03
    assert not np.isnan(cr.loc[idx[1], "USD"])
