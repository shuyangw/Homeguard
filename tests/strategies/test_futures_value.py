import numpy as np
import pandas as pd

from src.strategies.advanced.futures_value_strategy import FuturesValueStrategy
from src.strategies.registry import get_strategy_class


def _trending_panel(n=1500):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    # UP: steadily rising -> "expensive" over the 5yr-to-1yr window -> value should short it
    up = 100.0 * np.exp(0.0004 * np.arange(n))
    # DOWN: steadily falling
    down = 100.0 * np.exp(-0.0004 * np.arange(n))
    # FLAT: noisy but flat, small random walk
    rng = np.random.RandomState(0)
    flat = 100.0 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({"UP": up, "DOWN": down, "FLAT": flat}, index=idx)


def test_registered():
    assert get_strategy_class("FuturesValue") is FuturesValueStrategy
    assert get_strategy_class("Value") is FuturesValueStrategy
    assert get_strategy_class("Futures Value") is FuturesValueStrategy


def test_causal_and_capped():
    close = _trending_panel()
    strat = FuturesValueStrategy(["UP", "DOWN", "FLAT"])
    fc = strat.forecast_panel(close)

    assert list(fc.columns) == ["UP", "DOWN", "FLAT"]
    assert fc.index.equals(close.index)

    valid = fc.dropna()
    assert not valid.empty
    assert ((valid >= -strat.cap) & (valid <= strat.cap)).all().all()

    # warmup: need 1260 bars of history (shift(1260)) before any forecast is valid
    assert fc.iloc[:1259].isna().all().all()

    # UP rose strongly over the skip window (t-5yr .. t-1yr) -> "expensive" -> short (negative)
    last_up = fc["UP"].dropna().iloc[-1]
    assert last_up < 0

    # DOWN fell strongly over the skip window -> "cheap" -> long (positive)
    last_down = fc["DOWN"].dropna().iloc[-1]
    assert last_down > 0


def test_no_lookahead():
    close = _trending_panel(1500)
    strat = FuturesValueStrategy(["UP", "DOWN", "FLAT"])
    fc_orig = strat.forecast_panel(close)

    extra_idx = pd.date_range(close.index[-1] + pd.tseries.offsets.BDay(1), periods=100, freq="B")
    extra = pd.DataFrame({
        "UP": close["UP"].iloc[-1] * np.exp(0.0004 * np.arange(1, 101)),
        "DOWN": close["DOWN"].iloc[-1] * np.exp(-0.0004 * np.arange(1, 101)),
        "FLAT": close["FLAT"].iloc[-1] + np.cumsum(np.random.RandomState(1).normal(0, 0.2, 100)),
    }, index=extra_idx)
    close_extended = pd.concat([close, extra])

    fc_extended = strat.forecast_panel(close_extended)

    aligned = fc_extended.reindex(fc_orig.index)
    pd.testing.assert_frame_equal(aligned, fc_orig)
