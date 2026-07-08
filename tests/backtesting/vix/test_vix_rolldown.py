import numpy as np
import pandas as pd

from src.backtesting.vix.vix_rolldown_eval import rolldown_returns


def _curve(dates, vx1, vx2, vx1_dte=None):
    data = {"date": dates, "vx1_settle": vx1, "vx2_settle": vx2}
    if vx1_dte is not None:
        data["vx1_dte"] = vx1_dte
    return pd.DataFrame(data)


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


def test_roll_day_jump_excluded():
    # vx1_settle is a continuous nearest-unexpired front: at expiry it switches to a
    # further-out contract and JUMPS. A rolled position never realizes that jump.
    # Detect the roll via vx1_dte snapping UP (front-contract switch) and zero the return.
    d = pd.date_range("2015-01-05", periods=4, freq="B")
    # Day 3 is a roll: vx1_dte jumps 1 -> 28 and vx1_settle jumps 16 -> 22 (spurious).
    # Contango throughout (vx2 > vx1), prior-day position is short on the roll day.
    curve = _curve(
        d,
        [18.0, 17.0, 16.0, 22.0],
        [25.0, 25.0, 25.0, 25.0],
        vx1_dte=[3, 2, 1, 28],
    )
    r = rolldown_returns(curve)
    # The roll day (index 3) must contribute exactly 0.0, NOT the raw +37.5% jump.
    assert r.iloc[3] == 0.0
    # Sanity: the non-roll days before it still reflect the real short-in-contango P&L.
    assert r.iloc[2] > 0.0  # vx1 fell 17 -> 16, short profits
