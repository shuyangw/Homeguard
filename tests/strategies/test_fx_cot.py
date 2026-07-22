"""COT positioning strategies: sign logic (contrarian vs momentum), lag, registry."""
import numpy as np
import pandas as pd

from src.data.cot import to_daily
from src.strategies.registry import get_strategy_class
from src.strategies.advanced.fx_strategies import (
    FxCotContrarianTS, FxCotMomentumTS, FxCotContrarianXS,
)


def _weekly(series_by_pair, n=200):
    idx = pd.date_range("2019-01-01", periods=n, freq="W")
    return pd.DataFrame({p: v for p, v in series_by_pair.items()}, index=idx)


def test_to_daily_no_lookahead():
    wk = pd.DataFrame({"EURUSD": [1.0, 2.0]},
                      index=pd.to_datetime(["2020-01-08", "2020-01-15"]))
    daily = pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-14", "2020-01-20"])
    out = to_daily(wk, daily)
    assert pd.isna(out["EURUSD"].iloc[0])   # before first active date -> no value
    assert out["EURUSD"].iloc[1] == 1.0
    assert out["EURUSD"].iloc[2] == 1.0     # second reading not active until 01-15
    assert out["EURUSD"].iloc[3] == 2.0


def test_contrarian_ts_fades_crowded_long():
    # positioning spikes to an extreme high on the last week -> fade (negative)
    w = _weekly({"EURUSD": np.concatenate([np.zeros(199), [5.0]])})
    fc = FxCotContrarianTS(["EURUSD"], z_window=50)._weekly_forecast(w)
    assert fc["EURUSD"].iloc[-1] < 0


def test_momentum_ts_follows_rising_positioning():
    # positioning flat, then accelerates up at the end: recent CHANGE is high vs its
    # own history -> follow (positive). (A perfectly linear ramp has constant change
    # and thus a zero-variance change-z -- no signal -- so the accel is what matters.)
    w = _weekly({"EURUSD": np.concatenate([np.zeros(180), np.linspace(0.0, 5.0, 20)])})
    fc = FxCotMomentumTS(["EURUSD"], z_window=50, mom_horizon=4)._weekly_forecast(w)
    assert fc["EURUSD"].iloc[-1] > 0


def test_contrarian_xs_shorts_most_crowded():
    # cross-section: EURUSD most crowded long, USDJPY most crowded short (last row)
    w = _weekly({"EURUSD": np.concatenate([np.zeros(199), [3.0]]),
                 "USDJPY": np.concatenate([np.zeros(199), [-3.0]]),
                 "GBPUSD": np.zeros(200)})
    fc = FxCotContrarianXS(["EURUSD", "USDJPY", "GBPUSD"])._weekly_forecast(w)
    assert fc["EURUSD"].iloc[-1] < 0   # most crowded long -> short
    assert fc["USDJPY"].iloc[-1] > 0   # most crowded short -> long


def test_registry_resolves_cot_names():
    assert get_strategy_class("FxCotContrarianTS") is FxCotContrarianTS
    assert get_strategy_class("FxCotMomentumTS") is FxCotMomentumTS
    assert get_strategy_class("FxCotContrarianXS") is FxCotContrarianXS
