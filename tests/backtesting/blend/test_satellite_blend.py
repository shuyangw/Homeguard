import numpy as np
import pandas as pd
from datetime import date

from src.backtesting.blend.satellite_blend import blend_books


def _win(vals, start_day):
    idx = [date(2020, 1, d) for d in range(start_day, start_day + len(vals))]
    return pd.Series(vals, index=idx)


def test_zero_weight_reduces_to_core():
    core = [_win([0.01, -0.02, 0.03, 0.00, 0.01], 2), _win([0.02, -0.01, 0.00, 0.01, -0.02], 10)]
    sat = [_win([0.05, -0.06, 0.04, 0.02, -0.03], 2), _win([-0.04, 0.05, -0.02, 0.03, 0.01], 10)]
    b0 = blend_books(core, sat, sat_weight=0.0)
    # at sat_weight 0, blended = core/core_vol -> same Sharpe as core-only normalized (scale-invariant)
    from scripts.backtest_scripts.run_carver_walkforward import _annualized_sharpe
    core_stitch = np.concatenate([w.to_numpy() for w in core])
    assert abs(b0["oos_sharpe"] - _annualized_sharpe(core_stitch)) < 1e-9


def test_satellite_missing_dates_zero():
    core = [_win([0.01, 0.02, 0.03, 0.04], 2)]
    sat = [_win([0.10, 0.20], 2)]  # covers only first 2 of 4 core dates
    b = blend_books(core, sat, sat_weight=0.15)
    assert np.isfinite(b["oos_sharpe"]) and b["n_oos_days"] == 4


def test_blend_is_weighted_sum():
    core = [_win([0.01, -0.01, 0.02], 2)]
    sat = [_win([0.03, -0.02, 0.01], 2)]
    cv = float(np.std([0.01, -0.01, 0.02], ddof=1))
    sv = float(np.std([0.03, -0.02, 0.01], ddof=1))
    b = blend_books(core, sat, sat_weight=0.15, core_vol=cv, sat_vol=sv)
    expected = 0.85 * np.array([0.01, -0.01, 0.02]) / cv + 0.15 * np.array([0.03, -0.02, 0.01]) / sv
    from scripts.backtest_scripts.run_carver_walkforward import _annualized_sharpe
    assert abs(b["oos_sharpe"] - _annualized_sharpe(expected)) < 1e-9
