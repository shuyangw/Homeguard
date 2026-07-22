"""FxMeanRev and FxCarryMom strategies + registry resolution."""
import numpy as np
import pandas as pd

from src.strategies.registry import get_strategy_class
from src.strategies.advanced.fx_strategies import (
    FxMeanRevStrategy, FxCarryMomStrategy, FxTSMOMStrategy,
)


def _ramp_then_spike(n=120):
    idx = pd.bdate_range("2020-01-01", periods=n)
    base = pd.Series(np.linspace(10.0, 10.0, n), index=idx)  # flat
    base.iloc[-1] = 11.5  # last bar stretched high
    return pd.DataFrame({"USDMXN": base})


def test_meanrev_fades_high_deviation():
    close = _ramp_then_spike()
    fc = FxMeanRevStrategy(["USDMXN"], lookback=60).forecast_panel(close)
    # price stretched HIGH on the last bar -> negative (short) forecast
    assert fc["USDMXN"].iloc[-1] < 0


def test_meanrev_registry_and_shape():
    cls = get_strategy_class("FxMeanRev")
    assert cls is FxMeanRevStrategy
    idx = pd.bdate_range("2020-01-01", periods=100)
    close = pd.DataFrame({"USDMXN": np.linspace(10, 12, 100),
                          "USDZAR": np.linspace(15, 14, 100)}, index=idx)
    fc = cls(["USDMXN", "USDZAR"]).forecast_panel(close)
    assert list(fc.columns) == ["USDMXN", "USDZAR"]
    assert fc.abs().max().max() <= 20.0  # capped


def test_carrymom_is_half_carry_plus_half_mom():
    cls = get_strategy_class("FxCarryMom")
    assert cls is FxCarryMomStrategy
    idx = pd.bdate_range("2018-01-01", periods=400)
    close = pd.DataFrame({"USDMXN": np.linspace(15, 20, 400),
                          "USDZAR": np.linspace(14, 18, 400)}, index=idx)
    cm = FxCarryMomStrategy(["USDMXN", "USDZAR"])
    blended = cm.forecast_panel(close)
    # the momentum leg alone, for cross-check of the 0.5 weight
    mom = FxTSMOMStrategy(["USDMXN", "USDZAR"]).forecast_panel(close)
    carry = cm._carry.forecast_panel(close)
    cols = [c for c in carry.columns if c in mom.columns]
    expected = (0.5 * carry[cols] + 0.5 * mom[cols]).fillna(0.0)
    pd.testing.assert_frame_equal(blended[cols], expected[cols])
