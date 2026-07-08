import numpy as np
import pandas as pd

from src.backtesting.vix.vix_rolldown_eval import rolldown_returns


def _curve(dates, vx1, vx2):
    return pd.DataFrame({"date": dates, "vx1_settle": vx1, "vx2_settle": vx2})


def test_contango_short_vx1_profits_when_vx1_falls():
    d = pd.date_range("2015-01-05", periods=4, freq="B")
    # contango (vx2 > vx1); vx1 falls 18 -> 17 -> short profits (positive return)
    curve = _curve(d, [18.0, 17.0, 16.0, 15.0], [20.0, 20.0, 20.0, 20.0])
    r = rolldown_returns(curve)
    assert (r.dropna() > 0).all()  # short a falling VX1 in contango -> gains


def test_backwardation_kill_switch_flat():
    d = pd.date_range("2015-01-05", periods=3, freq="B")
    # backwardation (vx1 > vx2) -> kill switch -> flat -> zero return regardless of moves
    curve = _curve(d, [25.0, 30.0, 28.0], [20.0, 20.0, 20.0])
    r = rolldown_returns(curve)
    assert (r.fillna(0.0) == 0.0).all()
