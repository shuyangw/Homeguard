import numpy as np
import pandas as pd
from src.backtesting.vix.vix_rolldown_eval import subperiod_audit


def test_subperiod_audit_shape_and_drawdown():
    idx = pd.date_range("2017-01-01", periods=800, freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 800), index=idx)
    # inject a crash day
    r.iloc[400] = -0.20
    out = subperiod_audit(r)
    assert set(out) >= {"by_year", "skew", "kurtosis", "worst_day", "max_drawdown"}
    assert out["worst_day"] <= -0.20 + 1e-9
    assert out["max_drawdown"] < 0
    assert 2017 in out["by_year"] and isinstance(out["by_year"][2017], float)


def test_worst_day_and_skew_negative_with_crash():
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    r = pd.Series(0.001, index=idx)
    r.iloc[25] = -0.5  # Volmageddon-like
    out = subperiod_audit(r)
    assert out["skew"] < 0
    assert out["worst_day"] == -0.5
