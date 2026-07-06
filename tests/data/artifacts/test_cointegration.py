import numpy as np
import pandas as pd
from src.data.artifacts.cointegration import ou_half_life, test_pair


def test_half_life_of_ar1():
    rng = np.random.default_rng(1)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
    hl = ou_half_life(pd.Series(x))
    # AR(1) phi=0.9 -> half life = ln(2)/-ln(0.9) ~ 6.58
    assert 4 < hl < 10


def test_cointegrated_pair_low_pvalue():
    rng = np.random.default_rng(2)
    n = 1000
    a = np.cumsum(rng.normal(0, 1, n)) + 100
    b = a + rng.normal(0, 0.5, n)
    idx = pd.date_range("2018-01-01", periods=n)
    res = test_pair(pd.Series(a, index=idx), pd.Series(b, index=idx))
    assert res["adf_pvalue"] < 0.05
