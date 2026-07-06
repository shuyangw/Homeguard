import numpy as np
import pandas as pd
from src.data.artifacts.event_registries import label_vol_spikes


def test_flags_large_move():
    idx = pd.date_range("2020-01-01", periods=100)
    r = pd.Series(np.r_[np.random.default_rng(0).normal(0, 0.001, 99), 0.05], index=idx)
    spikes = label_vol_spikes(r.to_frame("EURUSD"), z=3.0)
    assert (spikes["date"] == idx[-1]).any()
