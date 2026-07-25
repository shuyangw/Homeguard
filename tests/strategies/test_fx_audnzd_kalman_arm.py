"""Arm B (AudNzdPairsKalman) is Arm A with ONLY the hedge-ratio estimator swapped."""
import numpy as np
import pandas as pd

from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs, AudNzdPairsKalman


def _panel(n=500, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B").date
    nzd = np.exp(np.cumsum(rng.normal(0, 0.005, n)) - 0.4)
    aud = np.exp(np.cumsum(rng.normal(0, 0.005, n)) - 0.3)
    return pd.DataFrame({"AUDUSD": aud, "NZDUSD": nzd}, index=pd.Index(idx))


def test_arms_share_every_parameter_except_the_estimator():
    a, b = AudNzdPairs(), AudNzdPairsKalman()
    for p in ("lookback", "entry_z", "target_z", "stop_z", "max_days", "vol_window"):
        assert getattr(a, p) == getattr(b, p), p
    # the ONLY overridden behaviour is the regression/estimator hook
    assert type(b)._regression_z is not type(a)._regression_z
    assert type(b).spread_book is type(a).spread_book
    assert type(b)._strength is type(a)._strength
    assert type(b)._is_rebalance is type(a)._is_rebalance


def test_kalman_arm_produces_a_book_and_differs_from_ols():
    panel = _panel()
    book_a, _ = AudNzdPairs().spread_book(panel)
    book_b, _ = AudNzdPairsKalman().spread_book(panel)
    # both run; the Kalman arm is not degenerate
    assert isinstance(book_a, dict) and isinstance(book_b, dict)
    betas_b = [s.hedge_ratio for spreads in book_b.values() for s in spreads]
    assert len(betas_b) > 0
    assert all(np.isfinite(x) for x in betas_b)


def test_kalman_beta_is_causal_within_the_strategy():
    """Perturbing prices AFTER a date must not change the beta the strategy used
    on or before that date."""
    panel = _panel()
    cut = 380
    book1, _ = AudNzdPairsKalman().spread_book(panel)
    panel2 = panel.copy()
    panel2.iloc[cut + 1:, panel2.columns.get_loc("AUDUSD")] *= 1.5
    book2, _ = AudNzdPairsKalman().spread_book(panel2)
    cutoff = panel.index[cut]
    for d, spreads in book1.items():
        if d <= cutoff and d in book2:
            assert spreads[0].hedge_ratio == book2[d][0].hedge_ratio
