import numpy as np
import pandas as pd
from src.backtesting.vol.vrp_strategy import percentile_rank_causal, reexpression_stats


def test_percentile_rank_causal():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    s = pd.Series(np.arange(300.0), index=idx)
    pr = percentile_rank_causal(s, window=100)
    # strictly-prior: first `window` entries NaN
    assert pr.iloc[:100].isna().all()
    # a monotonically rising series sits at the top of its trailing window -> ~1.0
    assert pr.dropna().iloc[-1] > 0.9


def test_reexpression_stats():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    b = a * 0.9 + pd.Series(rng.normal(0, 0.002, 300), index=idx)  # highly correlated
    out = reexpression_stats(a, b)
    assert out["corr"] > 0.9
    assert "marginal_sharpe" in out
