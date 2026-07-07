import numpy as np
import pandas as pd
from src.strategies.advanced.futures_signal_base import CrossSectionalRankStrategy


class _RawPassThrough(CrossSectionalRankStrategy):
    """Raw signal = the close panel itself (for testing the XS transform)."""
    def _raw_signal_panel(self, close_panel):
        return close_panel


def _panel(cols, rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx, columns=cols)


def test_within_group_demean_sums_to_zero():
    # two roots in the same asset_class -> demeaned forecasts are opposite in sign
    strat = _RawPassThrough(["ES", "NQ"], group_fn=lambda r: "equity_index")
    fc = strat.forecast_panel(_panel(["ES", "NQ"], [[1.0, 3.0], [2.0, 2.0]]))
    # row 0: mean 2, dispersion>0 -> ES negative, NQ positive, sum ~ 0
    assert fc.iloc[0].sum() == 0.0 or abs(fc.iloc[0].sum()) < 1e-9
    assert fc.loc[fc.index[0], "ES"] < 0 < fc.loc[fc.index[0], "NQ"]


def test_zero_dispersion_gives_zero_not_nan():
    strat = _RawPassThrough(["ES", "NQ"], group_fn=lambda r: "equity_index")
    fc = strat.forecast_panel(_panel(["ES", "NQ"], [[2.0, 2.0]]))
    assert (fc.iloc[0] == 0.0).all()


def test_forecast_bounded_by_cap():
    strat = _RawPassThrough(["ES", "NQ"], group_fn=lambda r: "equity_index", cap=20.0)
    fc = strat.forecast_panel(_panel(["ES", "NQ"], [[-1e6, 1e6]]))
    assert fc.abs().max().max() <= 20.0


def test_singleton_group_contributes_zero():
    strat = _RawPassThrough(["ES"], group_fn=lambda r: "equity_index")
    fc = strat.forecast_panel(_panel(["ES"], [[5.0]]))
    assert (fc.iloc[0] == 0.0).all()
