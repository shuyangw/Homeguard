import numpy as np
import pytest
from datetime import date

from src.data.continuous_contract_loader import ContinuousContractDataLoader, continuous_1min_dir


@pytest.mark.skipif(not (continuous_1min_dir() / "symbol=ES").exists(), reason="futures store not present")
@pytest.mark.parametrize("root", ["ES", "CL"])
@pytest.mark.parametrize("window", [(date(2010, 6, 7), date(2026, 2, 20)), (date(2018, 1, 1), date(2021, 1, 1))])
def test_daily_ratio_adjust_matches_onthefly(root, window):
    ld = ContinuousContractDataLoader()
    start, end = window
    current = ld.aggregate_to_daily(root, method="ratio_adjusted", start=start, end=end).sort("timestamp")
    raw = ld.aggregate_to_daily(root, method="raw", start=start, end=end)
    viacache = ld.ratio_adjust_daily(raw, root, start=start, end=end).sort("timestamp")
    a = current.select(["timestamp", "close"]).to_pandas().set_index("timestamp")["close"]
    b = viacache.select(["timestamp", "close"]).to_pandas().set_index("timestamp")["close"]
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), rtol=0, atol=1e-9)
