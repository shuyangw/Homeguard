import numpy as np
import pandas as pd
from src.backtesting.walkforward_common import gate_return_stream as canonical
from src.backtesting.vix.vix_rolldown_eval import gate_return_stream as via_vix


def _series(n=1500):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_vix_reexports_canonical_gate():
    assert via_vix is canonical


def test_gate_returns_expected_keys():
    out = canonical(_series())
    assert set(out) >= {"n_windows", "oos_sharpe", "psr", "dsr", "pbo"}
    assert out["n_windows"] >= 1
