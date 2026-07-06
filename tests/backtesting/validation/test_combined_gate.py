import numpy as np
from src.backtesting.validation.combined_gate import combined_gate


def test_strong_signal_passes():
    rng = np.random.default_rng(0)
    # positive-drift returns across splits -> should pass
    splits = [rng.normal(0.001, 0.005, 200) for _ in range(10)]
    res = combined_gate(splits, n_trials=1)
    assert set(res) >= {"dsr", "pbo", "mean_oos_sharpe", "pass"}
    assert isinstance(res["pass"], bool)
    assert res["pass"] is True


def test_noise_does_not_pass():
    rng = np.random.default_rng(1)
    splits = [rng.normal(0.0, 0.01, 200) for _ in range(10)]
    res = combined_gate(splits, n_trials=50)
    assert res["pass"] is False
