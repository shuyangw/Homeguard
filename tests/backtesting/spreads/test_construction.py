import numpy as np
import pandas as pd
from src.backtesting.spreads.construction import SpreadLeg, build_spread, round_trip_cost_usd


def _closes(cols):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame({c: np.arange(5) + base for c, base in cols.items()}, index=idx)


def test_additive_dv01_steepener_signal_and_return():
    # 2s10s: long 10Y (+1), short 2YY (-1); close IS yield
    closes = _closes({"10Y": 3.0, "2YY": 1.0})  # 10Y: 3..7, 2YY: 1..5
    legs = [SpreadLeg("10Y", 1.0), SpreadLeg("2YY", -1.0)]
    s = build_spread(legs, closes, mode="additive")
    # signal = 10Y - 2YY = constant 2.0 (both rise in lockstep here)
    assert np.allclose(s.signal.values, 2.0)
    # level is flat -> unit_return all ~0
    assert np.allclose(s.unit_return.dropna().values, 0.0)


def test_additive_return_tracks_slope_change():
    closes = pd.DataFrame({
        "10Y": [3.0, 3.5, 3.5],   # +0.5 then flat
        "2YY": [1.0, 1.0, 1.2],   # flat then +0.2
    }, index=pd.date_range("2020-01-01", periods=3, freq="B"))
    legs = [SpreadLeg("10Y", 1.0), SpreadLeg("2YY", -1.0)]
    s = build_spread(legs, closes, mode="additive")
    # level = 10Y-2YY = [2.0, 2.5, 2.3]; diffs = [+0.5, -0.2]
    diffs = s.signal.diff().dropna().values
    assert np.allclose(diffs, [0.5, -0.2])
    # unit_return has same sign pattern as level diffs
    assert np.sign(s.unit_return.dropna().values).tolist() == [1.0, -1.0]


def test_multiplicative_ratio_return():
    closes = pd.DataFrame({
        "GC": [100.0, 110.0],
        "SI": [100.0, 100.0],
    }, index=pd.date_range("2020-01-01", periods=2, freq="B"))
    legs = [SpreadLeg("GC", 1.0), SpreadLeg("SI", -1.0)]
    s = build_spread(legs, closes, mode="multiplicative")
    # signal = log(GC/SI); long GC short SI return = r_GC - r_SI = 0.10 - 0 = 0.10
    assert np.isclose(s.unit_return.dropna().iloc[0], 0.10)
    assert np.isclose(s.signal.iloc[0], 0.0)


def test_round_trip_cost_sums_legs():
    legs = [SpreadLeg("10Y", 1.0), SpreadLeg("2YY", -1.0)]
    c = round_trip_cost_usd(legs)
    assert c > 0
