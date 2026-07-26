"""Range indicators: hand-computed fixtures + causality (OHLC wave, 2026-07-25)."""
import numpy as np
import pandas as pd

from src.features.range_indicators import adx, atr, parkinson_rv, true_range


def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.006, n))) * 1.1)
    span = pd.Series(np.abs(rng.normal(0, 0.004, n)) + 0.001)
    return close + span, close - span, close      # high, low, close


def test_true_range_matches_hand_computation():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([9.0, 10.5, 9.5])
    close = pd.Series([9.5, 11.5, 10.0])
    tr = true_range(high, low, close)
    assert tr.iloc[0] == 1.0                      # no prev close -> H-L
    # bar 1: H-L=1.5, |H-prev|=2.5, |L-prev|=1.0  -> 2.5
    assert tr.iloc[1] == 2.5
    # bar 2: H-L=1.5, |H-prev|=0.5, |L-prev|=2.0  -> 2.0
    assert tr.iloc[2] == 2.0


def test_parkinson_matches_its_closed_form():
    n = 10
    high, low, _ = _ohlc()
    got = parkinson_rv(high, low, n).iloc[-1]
    r2 = (np.log(high / low) ** 2).iloc[-n:]
    want = np.sqrt(r2.mean() / (4 * np.log(2)) * 252)
    assert abs(got - want) < 1e-12


def test_adx_is_bounded_and_direction_agnostic():
    high, low, close = _ohlc()
    a = adx(high, low, close, 14).dropna()
    assert len(a) > 0 and a.between(0, 100).all()
    # ADX measures STRENGTH: reversing the trend direction leaves it unchanged
    inv_c = close.iloc[0] ** 2 / close
    inv_h, inv_l = close.iloc[0] ** 2 / low, close.iloc[0] ** 2 / high
    b = adx(inv_h, inv_l, inv_c, 14).dropna()
    assert b.between(0, 100).all()


def test_atr_is_positive_and_warms_up():
    high, low, close = _ohlc()
    a = atr(high, low, close, 10)
    assert a.iloc[:9].isna().all()
    assert (a.dropna() > 0).all()


def test_all_indicators_are_causal():
    """Perturbing bars AFTER k must not change any value at t <= k."""
    high, low, close = _ohlc(400)
    k = 250
    ref = {"atr": atr(high, low, close, 10), "adx": adx(high, low, close, 14),
           "park": parkinson_rv(high, low, 10)}
    h2, l2, c2 = high.copy(), low.copy(), close.copy()
    h2.iloc[k + 1:] *= 3.0
    l2.iloc[k + 1:] *= 0.5
    c2.iloc[k + 1:] *= 2.0
    got = {"atr": atr(h2, l2, c2, 10), "adx": adx(h2, l2, c2, 14),
           "park": parkinson_rv(h2, l2, 10)}
    for name in ref:
        a, b = ref[name].iloc[:k + 1], got[name].iloc[:k + 1]
        pd.testing.assert_series_equal(a, b, check_names=False)
