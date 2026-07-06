import numpy as np
import pandas as pd
from src.data.artifacts.currency_strength import currency_returns


def test_currency_returns_averages_pairs():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    close = pd.DataFrame({"EURUSD": [1.10, 1.21], "GBPUSD": [1.30, 1.30]}, index=idx)
    cr = currency_returns(close)
    # EUR appreciates ~10% vs USD on day 2; USD is the average of the inverses.
    assert cr.loc[idx[1], "EUR"] > 0
    assert "USD" in cr.columns
