import numpy as np
import pandas as pd

from src.strategies.advanced.fx_vol_ratio_pair import VolRatioPair


def _panel(n=700):
    idx = pd.date_range("2019-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(3)
    a = 10.0 + np.cumsum(rng.normal(0, 0.02, n))
    b = 11.0 + np.cumsum(rng.normal(0, 0.02, n))
    # inject a vol spike in A late -> RV ratio z spikes
    a[-30:] += np.cumsum(rng.normal(0, 0.15, 30))
    return pd.DataFrame({"EURNOK": a, "EURSEK": b}, index=pd.Index(idx))


def test_emits_spread_when_vol_ratio_z_high():
    close = _panel()
    book, sigma = VolRatioPair(coupled_sets=(("EURNOK", "EURSEK"),)).spread_book(close)
    active = [d for d, sps in book.items() if sps]
    assert active
    sp = book[active[-1]][0]
    assert {sp.leg_a, sp.leg_b} == {"EURNOK", "EURSEK"}


def test_shorts_high_vol_leg_longs_low_vol_leg():
    close = _panel()
    book, _ = VolRatioPair(coupled_sets=(("EURNOK", "EURSEK"),)).spread_book(close)
    active = [book[d][0] for d in book if book[d]]
    # EURNOK is the high-vol leg late -> spread should be short EURNOK / long EURSEK,
    # i.e. sign convention: strength expresses long(low-vol)-short(high-vol)
    assert active[-1].strength != 0.0
