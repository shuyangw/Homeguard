import numpy as np
import pandas as pd

from src.backtesting.engine.spread_sizing import Spread
from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs


def _coint_panel(n=400, div_start=350):
    # AUDUSD and NZDUSD co-move; inject a residual divergence late so |z|>2.
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.004, n))
    aud = 0.70 * np.exp(common + rng.normal(0, 0.0005, n))
    nzd = 0.65 * np.exp(common + rng.normal(0, 0.0005, n))
    aud[div_start:] *= 1.03  # AUD richens vs NZD -> residual z spikes
    return pd.DataFrame({"AUDUSD": aud, "NZDUSD": nzd}, index=pd.Index(idx))


def test_emits_spread_when_residual_z_exceeds_entry():
    close = _coint_panel()
    book, sigma = AudNzdPairs().spread_book(close)
    # some late date has an active AUDUSD/NZDUSD spread
    active = [d for d, sps in book.items() if sps]
    assert active, "expected at least one active spread after the divergence"
    sp = book[active[-1]][0]
    assert {sp.leg_a, sp.leg_b} == {"AUDUSD", "NZDUSD"}
    assert (sp.leg_a, sp.leg_b) in sigma[active[-1]]  # spread vol provided


def test_hedge_ratio_is_from_regression_not_one():
    close = _coint_panel()
    book, _ = AudNzdPairs().spread_book(close)
    active = [book[d][0] for d in book if book[d]]
    assert active
    # beta from ln-regression should differ from a naive 1.0
    assert abs(active[-1].hedge_ratio - 1.0) > 1e-6
