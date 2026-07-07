import yaml
from pathlib import Path

import pytest

from src.backtesting.engine.fx_backtest import run_fx_backtest


@pytest.mark.parametrize("cadence", ["daily", "weekly"])
def test_config_runs_end_to_end(cadence):
    cfg = yaml.safe_load(
        Path(f"config/backtesting/fx_carry_seatbelt_{cadence}.yaml").read_text())
    # short window keeps the smoke test fast; full run happens in the harness
    cfg["dates"] = {"start": "2019-01-01", "end": "2021-01-01"}
    res = run_fx_backtest(cfg, register=False, log_trades=False)
    assert res["n_days"] > 100
    assert len(res["equity_curve"]) == res["n_days"]
