import numpy as np
import pandas as pd
from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy


def test_forecast_panel_shape_and_cap():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.DataFrame({
        "MES": np.linspace(3000, 4000, 400),
        "MGC": np.linspace(1800, 1700, 400),  # downtrend
    }, index=dates)
    strat = CarverMomentumStrategy(universe=["MES", "MGC"])
    fc = strat.forecast_panel(close)
    assert list(fc.columns) == ["MES", "MGC"]
    assert fc.abs().max().max() <= 20.0 + 1e-9
    assert fc["MES"].iloc[-1] > 0   # uptrend
    assert fc["MGC"].iloc[-1] < 0   # downtrend


def test_forecast_panel_no_future_warning():
    import warnings
    import numpy as np
    import pandas as pd
    from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.DataFrame({
        "MES": np.linspace(3000, 4000, 400),
        "MGC": np.linspace(1800, 1700, 400),
    }, index=dates)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # any FutureWarning becomes an error
        CarverMomentumStrategy(universe=["MES", "MGC"]).forecast_panel(close)
