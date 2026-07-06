from unittest.mock import patch

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


def test_real_pbo_path_does_not_warn():
    rng = np.random.default_rng(0)
    # Long splits -> a real CSCV fold is achievable -> no fallback warning.
    splits = [rng.normal(0.001, 0.005, 200) for _ in range(10)]
    with patch("src.backtesting.validation.combined_gate.logger") as mock_logger:
        combined_gate(splits, n_trials=1)
        mock_logger.warning.assert_not_called()


def test_pbo_falls_back_to_proxy_when_splits_too_short():
    rng = np.random.default_rng(2)
    # Splits too short for any s in (16, 12, 8, 6, 4) to yield a real CSCV
    # fold -- must hit the median-split proxy fallback.
    splits = [rng.normal(0.001, 0.005, 6) for _ in range(5)]
    # Homeguard's Logger is a module-level Rich console singleton bound at
    # import time, so it can't be observed via caplog/capsys/capfd -- patch
    # the module's logger directly to confirm the fallback warns.
    with patch("src.backtesting.validation.combined_gate.logger") as mock_logger:
        res = combined_gate(splits, n_trials=1)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "median-split proxy" in warning_msg
        assert "n_splits=5" in warning_msg
        assert "min_split_len=6" in warning_msg

    assert set(res) >= {"dsr", "pbo", "mean_oos_sharpe", "pass"}
    assert np.isfinite(res["pbo"])
    assert 0.0 <= res["pbo"] <= 1.0
