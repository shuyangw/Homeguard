import numpy as np
import pandas as pd
from src.backtesting.session.session_walkforward import aggregate_returns, gate_session_stream


def test_aggregate_vol_normalizes_and_fills_zero():
    idx = pd.date_range("2015-01-01", periods=5, freq="B")
    a = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01], index=idx)
    b = pd.Series([0.02, 0.0, -0.02], index=idx[:3])  # shorter -> missing dates contribute 0
    agg = aggregate_returns({"ES": a, "NQ": b})
    assert list(agg.index) == list(idx)               # union of dates
    assert agg.notna().all()


def test_gate_returns_metric_keys():
    idx = pd.date_range("2015-01-01", periods=1500, freq="B")
    # deterministic, non-degenerate (varying) small positive-drift series
    r = pd.Series(0.0004 + 0.003 * np.sin(np.arange(1500) / 7.0), index=idx)
    g = gate_session_stream(r)
    for k in ("oos_sharpe", "n_oos", "n_windows", "psr", "dsr", "pbo"):
        assert k in g
    assert g["n_windows"] >= 1
    assert np.isfinite(g["oos_sharpe"])
