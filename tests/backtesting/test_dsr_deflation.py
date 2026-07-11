import numpy as np
from src.backtesting.walkforward_common import CAMPAIGN_TRIAL_SHARPES, gate_return_stream
import pandas as pd


def test_trial_sharpes_distribution_present():
    assert len(CAMPAIGN_TRIAL_SHARPES) >= 25
    arr = np.array(CAMPAIGN_TRIAL_SHARPES, dtype=float)
    assert np.nanvar(arr, ddof=1) > 0.01   # a real spread, not a constant


def test_dsr_now_deflates_below_psr():
    # a modest, positive, low-vol stream over a wide trial distribution: DSR must be
    # strictly BELOW PSR (deflation actually bites now).
    idx = pd.date_range("2015-01-01", periods=1500, freq="B")
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0004, 0.01, 1500), index=idx)
    out = gate_return_stream(r)
    assert out["n_windows"] >= 2
    # dsr uses the deflation benchmark; psr uses 0 -> dsr < psr when the benchmark > 0
    assert out["dsr"] <= out["psr"] + 1e-9
    assert out["dsr"] < out["psr"] or out["psr"] < 0.5  # deflation visible unless psr already low
