import numpy as np

from scripts.backtest_scripts import run_fx_walkforward as wf


def test_build_windows_non_overlapping():
    import datetime as dt
    windows = wf._build_windows(36, 12, 12, dt.date(2011, 1, 1), dt.date(2020, 1, 1))
    assert len(windows) >= 2
    # OOS windows are non-overlapping and ordered
    for (ts1, tst1, te1), (ts2, tst2, te2) in zip(windows, windows[1:]):
        assert tst2 >= te1


def test_verdict_reject_on_nonpositive_sharpe():
    result = {"psr": 0.5, "dsr": 0.5, "pbo": 0.3, "oos_sharpe": -0.1,
              "oos_sharpe_1_5x_cost": -0.2}
    assert wf._verdict_fx(result).startswith("REJECT")
