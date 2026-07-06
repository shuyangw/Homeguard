import numpy as np
import pandas as pd

from src.data.artifacts.spike_clean import scrub_spike_reverts


def _series(vals):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx, dtype=float)


def test_flags_and_nulls_spike_revert():
    close = _series([100, 100, 75, 100, 100])  # -28% then +28% revert
    cleaned, flagged = scrub_spike_reverts(close)
    assert len(flagged) == 1
    assert flagged[0] == close.index[2]
    assert np.isnan(cleaned.iloc[2])
    # untouched days unchanged
    assert cleaned.iloc[0] == 100 and cleaned.iloc[4] == 100


def test_spares_persistent_move():
    # SNB-like: -20% that does NOT revert (stays down)
    close = _series([120, 120, 98, 99, 99])
    cleaned, flagged = scrub_spike_reverts(close)
    assert flagged == []
    assert cleaned.equals(close)


def test_ignores_normal_moves():
    close = _series([100, 101, 102, 101, 103])
    cleaned, flagged = scrub_spike_reverts(close)
    assert flagged == []
    assert cleaned.equals(close)


def test_real_usdcad_artifact_is_flagged():
    # USDCAD 2024-12-20 bad-close print (verified against minute source)
    close = _series([1.43082, 1.44436, 1.43960, 1.09843, 1.43692, 1.43560])
    cleaned, flagged = scrub_spike_reverts(close)
    assert close.index[3] in flagged
    assert np.isnan(cleaned.iloc[3])
