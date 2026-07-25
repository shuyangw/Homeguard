"""Causal Kalman beta: no-lookahead proof, recovery, and drift tracking."""
import numpy as np

from src.backtesting.signals.kalman_beta import causal_dynamic_beta


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
