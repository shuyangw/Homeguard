import numpy as np

from src.backtesting.vol.atm_iv import black76_iv


def _black76_price(F, K, T, r, sigma, right):
    from scipy.stats import norm
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if right == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def test_black76_iv_round_trip():
    F, K, T, r, sigma = 4000.0, 4000.0, 30 / 365, 0.03, 0.20
    for right in ("C", "P"):
        price = _black76_price(F, K, T, r, sigma, right)
        iv = black76_iv(price, F, K, T, r, right)
        assert abs(iv - 0.20) < 1e-3


def test_black76_below_intrinsic_is_nan():
    # A price below intrinsic value has no IV solution.
    assert np.isnan(black76_iv(0.01, 4200.0, 4000.0, 30 / 365, 0.03, "C"))
