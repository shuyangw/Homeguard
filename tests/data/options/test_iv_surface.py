"""Tests for M7 -- the smoothed implied-volatility surface.

Written before the implementation. The no-arbitrage tests carry explicit
NEGATIVE CONTROLS: a deliberately arbitrageable surface must FAIL the check.
A no-arb test that only ever sees arb-free input demonstrates nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.options.iv_surface import (
    REASON_ARB_VIOLATION,
    REASON_NO_FORWARD,
    REASON_OK,
    REASON_ONE_SIDED,
    REASON_TOO_FEW_STRIKES,
    REASON_ZERO_DTE,
    SVIParams,
    black76_price,
    butterfly_violation_rate,
    calendar_violation_rate,
    fit_expiry,
    implied_forward,
    smooth_delta,
    smooth_iv,
    svi_is_arbitrage_free,
    svi_total_variance,
)

# A benign, arbitrage-free SVI slice: modest skew, well-behaved wings.
BENIGN = SVIParams(a=0.006, b=0.08, rho=-0.55, m=0.01, sigma=0.12)
T_BENIGN = 0.25


# ---------------------------------------------------------------------------
# The parameterization itself
# ---------------------------------------------------------------------------


def test_svi_total_variance_matches_closed_form():
    p = BENIGN
    k = np.array([-0.2, 0.0, 0.15])
    expected = p.a + p.b * (
        p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2)
    )
    np.testing.assert_allclose(svi_total_variance(k, p), expected, rtol=1e-12)


def test_svi_total_variance_is_positive_and_convex_for_benign_params():
    k = np.linspace(-0.6, 0.6, 201)
    w = svi_total_variance(k, BENIGN)
    assert np.all(w > 0)
    assert np.all(np.diff(w, 2) > -1e-12), "total variance must be convex in k"


def test_analytic_jacobian_matches_finite_differences():
    """The fit supplies a closed-form Jacobian; a wrong one would bias every fit."""
    k = np.linspace(-0.4, 0.25, 30)
    theta = np.array([0.006, 0.08, -0.55, 0.01, 0.12])

    def model(t):
        a, b, rho, m, s = t
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))

    x = k - theta[3]
    r = np.sqrt(x**2 + theta[4] ** 2)
    analytic = np.column_stack([
        np.ones_like(x),
        theta[2] * x + r,
        theta[1] * x,
        -theta[1] * (theta[2] + x / r),
        theta[1] * theta[4] / r,
    ])

    eps = 1e-7
    numeric = np.column_stack([
        (model(theta + eps * np.eye(5)[i]) - model(theta - eps * np.eye(5)[i]))
        / (2 * eps)
        for i in range(5)
    ])
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_svi_wings_are_asymptotically_linear():
    """The reason SVI was chosen over a spline: bounded linear wings."""
    k = np.array([8.0, 9.0, 10.0, 11.0])
    w = svi_total_variance(k, BENIGN)
    second_diff = np.diff(w, 2)
    assert np.all(np.abs(second_diff) < 1e-4), "far wing must be effectively linear"


# ---------------------------------------------------------------------------
# Black-76 pricing
# ---------------------------------------------------------------------------


def test_black76_put_call_parity_holds():
    F, K, T, sigma, D = 100.0, 95.0, 0.5, 0.22, 0.985
    c = black76_price(F, K, T, sigma, "C", D)
    p = black76_price(F, K, T, sigma, "P", D)
    assert c - p == pytest.approx(D * (F - K), abs=1e-10)


def test_vectorized_iv_agrees_with_the_repo_scalar_inverter():
    """Pin the fast bisection inverter to the repo's existing scalar brentq.

    `black76_iv_vec` exists only because `atm_iv.black76_iv` is ~2 orders of
    magnitude too slow at this scale. Two inverters that could disagree is a
    correctness risk, so this test forbids divergence.
    """
    from src.backtesting.vol.atm_iv import black76_iv
    from src.data.options.iv_surface import black76_iv_vec

    F, T, D = 470.0, 0.25, 0.9868
    r = -np.log(D) / T
    strikes = np.array([380.0, 420.0, 455.0, 470.0, 485.0, 510.0, 550.0])
    is_call = strikes >= F
    truth_iv = np.array([0.31, 0.24, 0.19, 0.17, 0.16, 0.18, 0.22])
    prices = np.array([
        black76_price(F, k, T, s, "C" if c else "P", D)
        for k, s, c in zip(strikes, truth_iv, is_call)
    ])

    fast = black76_iv_vec(prices, F, strikes, T, D, is_call)
    slow = np.array([
        black76_iv(p, F, k, T, r, "C" if c else "P")
        for p, k, c in zip(prices, strikes, is_call)
    ])

    np.testing.assert_allclose(fast, truth_iv, atol=1e-8)
    np.testing.assert_allclose(fast, slow, atol=1e-6)


def test_vectorized_iv_returns_nan_below_intrinsic():
    from src.data.options.iv_surface import black76_iv_vec

    F, T, D = 470.0, 0.25, 0.9868
    strikes = np.array([400.0, 400.0, 400.0])
    # below intrinsic; above the forward itself (unattainable); and a genuine
    # deep-ITM price that IS invertible and must NOT be discarded.
    prices = np.array([1.0, 600.0, 200.0])
    out = black76_iv_vec(prices, F, strikes, T, D, np.array([True, True, True]))
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert np.isfinite(out[2])


def test_black76_price_is_above_intrinsic():
    F, K, T, D = 100.0, 90.0, 0.5, 0.99
    c = black76_price(F, K, T, 0.20, "C", D)
    assert c > D * (F - K)


# ---------------------------------------------------------------------------
# The parity-implied forward (this is how "dividend-aware" is delivered)
# ---------------------------------------------------------------------------


def test_implied_forward_recovers_forward_from_parity_consistent_quotes():
    F_true, T, D, sigma = 471.30, 0.12, 0.9945, 0.14
    strikes = np.arange(440.0, 505.0, 5.0)
    calls = np.array([black76_price(F_true, k, T, sigma, "C", D) for k in strikes])
    puts = np.array([black76_price(F_true, k, T, sigma, "P", D) for k in strikes])

    fit = implied_forward(strikes, calls, strikes, puts, D=D, spot=470.0)

    assert fit.reason == REASON_OK
    assert fit.forward == pytest.approx(F_true, abs=1e-6)
    assert fit.n_pairs == len(strikes)


def test_implied_forward_is_dividend_aware_without_a_dividend_series():
    """A dividend-bearing underlying prices F below S*exp(rT); parity finds it."""
    spot, r, q, T = 100.0, 0.05, 0.02, 1.0
    D = float(np.exp(-r * T))
    F_true = spot * np.exp((r - q) * T)
    strikes = np.arange(85.0, 116.0, 2.5)
    calls = np.array([black76_price(F_true, k, T, 0.2, "C", D) for k in strikes])
    puts = np.array([black76_price(F_true, k, T, 0.2, "P", D) for k in strikes])

    fit = implied_forward(strikes, calls, strikes, puts, D=D, spot=spot)

    assert fit.forward == pytest.approx(F_true, abs=1e-6)
    assert fit.forward < spot * np.exp(r * T)  # the dividend is visible
    assert fit.implied_div_yield == pytest.approx(q, abs=1e-6)


def test_implied_forward_refuses_when_too_few_paired_strikes():
    strikes_c = np.array([100.0, 105.0])
    strikes_p = np.array([100.0, 105.0])
    fit = implied_forward(
        strikes_c, np.array([5.0, 2.0]), strikes_p, np.array([2.0, 4.0]),
        D=0.99, spot=100.0,
    )
    assert fit.reason == REASON_NO_FORWARD
    assert not np.isfinite(fit.forward)


# ---------------------------------------------------------------------------
# Fitting -- synthetic round trip
# ---------------------------------------------------------------------------


def _synthetic_slice(params: SVIParams, T: float, F: float, D: float,
                     k_lo: float = -0.35, k_hi: float = 0.22, n: int = 45):
    """Quotes generated FROM a known SVI slice, so the truth is known exactly."""
    k = np.linspace(k_lo, k_hi, n)
    strikes = F * np.exp(k)
    iv = np.sqrt(np.maximum(svi_total_variance(k, params), 1e-12) / T)
    rights = np.where(strikes >= F, "C", "P")
    mids = np.array([
        black76_price(F, kk, T, ss, rr, D)
        for kk, ss, rr in zip(strikes, iv, rights)
    ])
    # a tight, symmetric two-sided market around the true mid
    half = np.maximum(0.01, 0.004 * mids)
    return strikes, rights, mids - half, mids + half


def test_fit_recovers_a_known_svi_slice():
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(BENIGN, T, F, D)

    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=91, spot=468.0)

    assert fit.reason == REASON_OK
    k = np.array([-0.25, -0.1, 0.0, 0.1])
    truth = np.sqrt(svi_total_variance(k, BENIGN) / T)
    np.testing.assert_allclose(smooth_iv(fit.params, k, T), truth, atol=2e-3)


def test_fit_reports_residuals_in_vol_points():
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(BENIGN, T, F, D)
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=91, spot=468.0)
    assert fit.rmse_vol_points < 2e-3
    assert fit.n_points > 20


def test_fit_refuses_zero_dte():
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(BENIGN, T, F, D)
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=0.0, dte=0, spot=468.0)
    assert fit.reason == REASON_ZERO_DTE
    assert fit.params is None


def test_fit_refuses_too_few_strikes():
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(BENIGN, T, F, D, n=6)
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=91, spot=468.0)
    assert fit.reason == REASON_TOO_FEW_STRIKES
    assert fit.params is None


def test_fit_refuses_one_sided_chain():
    """Only puts, all well below the forward -- a fit here would be pure fantasy."""
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(
        BENIGN, T, F, D, k_lo=-0.40, k_hi=-0.05, n=30
    )
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=91, spot=468.0)
    assert fit.reason == REASON_ONE_SIDED
    assert fit.params is None


def test_long_dated_slice_does_not_spuriously_refuse_as_arb_violation():
    """Regression: the `b` box bound must be consistent with the Lee check.

    Bounding b at 4/T while checking b*(1+|rho|) <= 4/T let the optimizer settle
    where the box allowed but the check rejected, refusing perfectly good
    long-dated slices (observed on real SPY 2024-01-16 at dte 339 and 350).
    """
    F, D, T, dte = 490.0, 0.952, 339 / 365.0, 339
    params = SVIParams(a=0.02, b=0.22, rho=-0.62, m=0.05, sigma=0.30)
    strikes, rights, bids, asks = _synthetic_slice(
        params, T, F, D, k_lo=-0.37, k_hi=0.21, n=44
    )
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=dte, spot=486.0)
    assert fit.reason == REASON_OK, f"spurious refusal: {fit.reason}"
    assert fit.params.b * (1 + abs(fit.params.rho)) <= 4.0 / T


def test_refused_fit_yields_null_not_a_number():
    """P1 must be able to tell 'no surface' from 'surface says 0.05'."""
    F, D, T = 470.0, 0.996, 0.25
    strikes, rights, bids, asks = _synthetic_slice(BENIGN, T, F, D, n=6)
    fit = fit_expiry(strikes=strikes, rights=rights, bids=bids, asks=asks,
                     forward=F, discount=D, T=T, dte=91, spot=468.0)
    assert fit.params is None
    assert fit.reason != REASON_OK
    assert not np.isfinite(fit.rmse_vol_points)


# ---------------------------------------------------------------------------
# Static no-arbitrage -- WITH NEGATIVE CONTROLS
# ---------------------------------------------------------------------------


def test_arb_free_params_pass_the_constraint_check():
    assert svi_is_arbitrage_free(BENIGN, T_BENIGN)


def test_negative_control_negative_b_is_rejected():
    """b < 0 inverts the wings -- must be rejected, not tolerated."""
    bad = SVIParams(a=0.006, b=-0.08, rho=-0.55, m=0.01, sigma=0.12)
    assert not svi_is_arbitrage_free(bad, T_BENIGN)


def test_negative_control_rho_out_of_bounds_is_rejected():
    bad = SVIParams(a=0.006, b=0.08, rho=-1.4, m=0.01, sigma=0.12)
    assert not svi_is_arbitrage_free(bad, T_BENIGN)


def test_negative_control_negative_total_variance_is_rejected():
    """a chosen so w(k) dips below zero -- an impossible variance."""
    bad = SVIParams(a=-0.20, b=0.08, rho=-0.55, m=0.01, sigma=0.12)
    assert not svi_is_arbitrage_free(bad, T_BENIGN)


def test_negative_control_excessive_wing_slope_is_rejected():
    """b*(1+|rho|) > 4/T breaches Lee's bound -> butterfly explosion.

    At T=0.25 the bound is 4/0.25 = 16.0, so b*(1+0.55) must exceed it: b > 10.3.
    """
    bad = SVIParams(a=0.006, b=15.0, rho=-0.55, m=0.01, sigma=0.12)
    assert bad.b * (1 + abs(bad.rho)) > 4.0 / T_BENIGN  # the control really violates
    assert not svi_is_arbitrage_free(bad, T_BENIGN)

    # and a slope just inside the bound must still be accepted
    ok = SVIParams(a=0.006, b=9.0, rho=-0.55, m=0.01, sigma=0.12)
    assert svi_is_arbitrage_free(ok, T_BENIGN)


def test_butterfly_check_passes_on_a_benign_slice():
    k = np.linspace(-0.5, 0.5, 101)
    assert butterfly_violation_rate(k, BENIGN, T_BENIGN) == 0.0


def test_negative_control_butterfly_check_catches_an_arbitrageable_slice():
    """A deliberately arbitrageable slice MUST be flagged.

    Large b with a tiny sigma produces a kink sharp enough to drive the
    risk-neutral density negative -- a butterfly arbitrage.
    """
    arbitrageable = SVIParams(a=0.001, b=0.95, rho=-0.92, m=0.0, sigma=0.008)
    k = np.linspace(-0.5, 0.5, 401)
    rate = butterfly_violation_rate(k, arbitrageable, T_BENIGN)
    assert rate > 0.0, "negative control failed: an arbitrageable slice passed"


def test_calendar_check_passes_when_total_variance_increases_in_T():
    k = np.linspace(-0.3, 0.3, 61)
    near = SVIParams(a=0.004, b=0.06, rho=-0.5, m=0.0, sigma=0.12)
    far = SVIParams(a=0.010, b=0.10, rho=-0.5, m=0.0, sigma=0.12)
    assert calendar_violation_rate(k, near, far) == 0.0


def test_negative_control_calendar_check_catches_decreasing_total_variance():
    """Total variance falling with maturity is a calendar arbitrage."""
    k = np.linspace(-0.3, 0.3, 61)
    near = SVIParams(a=0.030, b=0.10, rho=-0.5, m=0.0, sigma=0.12)
    far = SVIParams(a=0.004, b=0.06, rho=-0.5, m=0.0, sigma=0.12)
    rate = calendar_violation_rate(k, near, far)
    assert rate > 0.5, "negative control failed: a calendar arb passed"


# ---------------------------------------------------------------------------
# Smoothed delta -- what P1 actually consumes
# ---------------------------------------------------------------------------


def test_smooth_delta_is_bounded_and_correctly_signed():
    F, D, T, spot = 470.0, 0.996, 0.25, 468.0
    for k, right in [(-0.3, "P"), (0.0, "C"), (0.2, "C"), (-0.1, "P")]:
        d = smooth_delta(BENIGN, k, T, F, D, spot, right)
        assert abs(d) <= 1.0
        assert (d > 0) if right == "C" else (d < 0)


def test_smooth_delta_decreases_monotonically_across_call_strikes():
    F, D, T, spot = 470.0, 0.996, 0.25, 468.0
    ks = np.linspace(-0.1, 0.5, 40)
    deltas = [smooth_delta(BENIGN, k, T, F, D, spot, "C") for k in ks]
    assert np.all(np.diff(deltas) < 0)


def test_smooth_delta_reaches_the_far_wing_that_opt_047_needs():
    """OPT-047 selects at 0.05 delta -- the surface must produce that value."""
    F, D, T, spot = 470.0, 0.996, 0.25, 468.0
    ks = np.linspace(-1.0, 0.0, 400)
    deltas = np.array([abs(smooth_delta(BENIGN, k, T, F, D, spot, "P")) for k in ks])
    assert deltas.min() < 0.05 < deltas.max()


def test_smooth_delta_respects_put_call_parity_in_delta():
    """delta_call - delta_put = exp(-qT), the carry factor."""
    F, D, T, spot = 470.0, 0.996, 0.25, 468.0
    k = -0.05
    dc = smooth_delta(BENIGN, k, T, F, D, spot, "C")
    dp = smooth_delta(BENIGN, k, T, F, D, spot, "P")
    assert dc - dp == pytest.approx(D * F / spot, abs=1e-9)
