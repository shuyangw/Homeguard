import pytest

from src.data.futures.paths import continuous_1min_dir
from scripts.backtest_scripts.run_carver_walkforward import walk_forward_carver


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures store not present")


def test_walkforward_returns_oos_and_gate():
    out = walk_forward_carver(train_months=36, test_months=12, step_months=12,
                              start="2014-01-01", end="2020-12-31")  # short range for the test
    assert "oos_sharpe" in out
    assert "psr" in out and "dsr" in out and "pbo" in out
    assert "oos_sharpe_1_5x_cost" in out
    assert out["n_windows"] >= 2
