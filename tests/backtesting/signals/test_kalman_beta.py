"""Causal Kalman beta: no-lookahead proof, recovery, drift tracking, and
robustness of the warmup/R seed to non-finite leading observations (the
windows-1/13 defect documented in
docs/strategies/research/20260722_fx_kalman_hedge_ratio_results.md)."""
import numpy as np

from src.backtesting.signals.kalman_beta import causal_dynamic_beta, select_warmup_indices


def _series(n=400, beta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    y = 1.0 + beta * x + rng.normal(0, 0.001, n)
    return y, x


def test_future_data_cannot_change_past_beta():
    """THE no-lookahead proof: mutating observations after index k must leave the
    filtered path at every t <= k bit-identical. A smoother would fail this."""
    y, x = _series()
    k = 250
    a1, b1 = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    y2 = y.copy()
    y2[k + 1:] += 5.0          # violently perturb the FUTURE only
    a2, b2 = causal_dynamic_beta(y2, x, delta=1e-4, r_var=1e-6, warmup=120)
    np.testing.assert_array_equal(b1[:k + 1], b2[:k + 1])
    np.testing.assert_array_equal(a1[:k + 1], a2[:k + 1])


def test_recovers_constant_beta():
    y, x = _series(beta=2.0)
    _, beta = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    assert abs(beta[-1] - 2.0) < 0.15


def test_tracks_a_drifting_beta():
    """Beta steps from 1.5 to 2.5 midway; the filter must move toward the new level
    (this is the whole premise of preferring it to a static window)."""
    n, split = 600, 300
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    betas = np.where(np.arange(n) < split, 1.5, 2.5)
    y = 1.0 + betas * x + rng.normal(0, 0.001, n)
    _, beta = causal_dynamic_beta(y, x, delta=1e-3, r_var=1e-6, warmup=120)
    assert beta[split - 10] < beta[-1]          # moved upward after the break
    assert abs(beta[-1] - 2.5) < abs(beta[split - 10] - 2.5)


def test_warmup_is_nan_and_degenerate_inputs_are_safe():
    y, x = _series(n=200)
    _, beta = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    assert np.all(np.isnan(beta[:119]))
    assert np.isfinite(beta[-1])
    # non-positive R and too-short input return all-NaN rather than raising
    _, b_bad = causal_dynamic_beta(y, x, delta=1e-4, r_var=0.0, warmup=120)
    assert np.all(np.isnan(b_bad))
    _, b_short = causal_dynamic_beta(y[:50], x[:50], delta=1e-4, r_var=1e-6, warmup=120)
    assert np.all(np.isnan(b_short))


def test_select_warmup_indices_skips_nan_rows():
    y, x = _series(n=200)
    y = y.copy()
    y[3] = np.nan
    x[10] = np.nan
    idx = select_warmup_indices(y, x, 120)
    assert idx is not None and len(idx) == 120
    assert 3 not in idx and 10 not in idx
    assert np.all(np.isfinite(y[idx])) and np.all(np.isfinite(x[idx]))


def test_select_warmup_indices_returns_none_when_insufficient_finite():
    y, x = _series(n=100)
    assert select_warmup_indices(y, x, 120) is None


def test_seed_robust_to_leading_nan_block():
    """The window-1 bug: a panel that opens before real price history begins
    (leading NaN block) must not poison the whole recursion to all-NaN --
    the seed should be found from the first `warmup` FINITE rows, wherever
    they occur, and the filter should still warm up and recover the true beta."""
    y, x = _series(n=400, beta=2.0)
    y_leading_nan, x_leading_nan = y.copy(), x.copy()
    y_leading_nan[:107] = np.nan
    x_leading_nan[:107] = np.nan
    _, beta = causal_dynamic_beta(y_leading_nan, x_leading_nan, delta=1e-4, r_var=1e-6, warmup=120)
    assert np.all(np.isnan(beta[:226]))            # NaN until the 120th finite row (index 226)
    assert np.isfinite(beta[226])                  # warmup completes exactly there
    assert np.isfinite(beta[-1])
    assert abs(beta[-1] - 2.0) < 0.2                # still recovers the true beta


def test_seed_robust_to_interior_nan_spike_revert():
    """The window-13 bug: a small number of interior nulled bars (data-quality
    spike-revert artifacts) landing inside the naive [:warmup] slice must not
    poison the seed -- select_warmup_indices should simply skip them."""
    y, x = _series(n=400, beta=2.0)
    y_spiked, x_spiked = y.copy(), x.copy()
    for i in (5, 40, 90):
        x_spiked[i] = np.nan
    _, beta_spiked = causal_dynamic_beta(y_spiked, x_spiked, delta=1e-4, r_var=1e-6, warmup=120)
    _, beta_clean = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    assert np.isfinite(beta_spiked[-1])
    assert abs(beta_spiked[-1] - 2.0) < 0.2
    # warmup now completes 3 rows later than the clean case (3 rows skipped)
    first_finite_spiked = np.flatnonzero(np.isfinite(beta_spiked))[0]
    first_finite_clean = np.flatnonzero(np.isfinite(beta_clean))[0]
    assert first_finite_spiked == first_finite_clean + 3


def test_seed_still_all_nan_when_insufficient_finite_obs_anywhere():
    """If fewer than `warmup` finite, aligned observations exist anywhere in
    the input, the fix must still fail closed (all-NaN), not raise or use a
    partial/degenerate seed."""
    y, x = _series(n=150)
    y[:40] = np.nan
    _, beta = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    assert np.all(np.isnan(beta))                   # only 110 finite rows available, need 120


def test_leading_nan_fix_does_not_use_future_data_either():
    """Behavior-neutrality of the causality guarantee under the fix: perturbing
    observations strictly after the (now-later) warmup point must still leave
    every earlier filtered value bit-identical."""
    y, x = _series(n=400, beta=2.0)
    y[:107] = np.nan
    x[:107] = np.nan
    k = 300
    a1, b1 = causal_dynamic_beta(y, x, delta=1e-4, r_var=1e-6, warmup=120)
    y2 = y.copy()
    y2[k + 1:] += 5.0
    a2, b2 = causal_dynamic_beta(y2, x, delta=1e-4, r_var=1e-6, warmup=120)
    np.testing.assert_array_equal(b1[:k + 1], b2[:k + 1])
    np.testing.assert_array_equal(a1[:k + 1], a2[:k + 1])
