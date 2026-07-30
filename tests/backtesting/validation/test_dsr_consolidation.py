"""Regression tests for the 2026-07-30 DSR consolidation.

`src/backtesting/validation/deflated_sharpe.py` used to carry its own DSR
formula that subtracted a STANDARD-NORMAL quantile from an ANNUALIZED Sharpe.
It now delegates to `src/backtesting/statistics/dsr.py`, the declared single
source of truth.

Several tests here are NEGATIVE CONTROLS: they are written to fail against the
superseded formula, so the suite demonstrably has power against the bug. Each
is marked and states the value the old code produced.
"""
from pathlib import Path

import numpy as np
import pytest

from src.backtesting.statistics.dsr import expected_max_sharpe
from src.backtesting.statistics.psr import psr
from src.backtesting.validation.deflated_sharpe import (
    DSRResult,
    compute_deflated_sharpe,
)

_EULER_MASCHERONI = 0.5772156649


def _sigma_injection(sigma: float) -> list:
    """Two-point sample whose ddof=1 variance is exactly sigma^2."""
    return [0.0, sigma * np.sqrt(2.0)]


def _returns_with_sharpe(target_annual_sharpe: float, n: int, seed: int = 7) -> np.ndarray:
    """A daily return series whose realized annualized Sharpe is exactly the target."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=n)
    r = (r - r.mean()) / r.std()
    return r * 0.01 + target_annual_sharpe / np.sqrt(252.0) * 0.01


# ---------------------------------------------------------------------------
# Known-good analytic case
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_reproduces_the_options_prereg_table() -> None:
    """Independently-verified fixture from the options pre-registration.

    sigma_SR = 1/sqrt(13.7 years) = 0.2702. These four values are quoted in
    docs/strategies/research/options-slate/20260728_lifetime_trial_count.md S1.1
    and were reproduced there against the pre-registration's own table.
    """
    sigma = 1.0 / np.sqrt(13.7)
    expected = {9: 0.4109, 19: 0.5074, 31: 0.5638, 38: 0.5860}
    for n_trials, want in expected.items():
        got = expected_max_sharpe(_sigma_injection(sigma), n_trials)
        assert got == pytest.approx(want, abs=5e-4), f"N={n_trials}"


def test_benchmark_scales_linearly_with_sigma_sr() -> None:
    """SR_0 = sigma_SR * E[max Z]: doubling the dispersion doubles the hurdle.

    The superseded formula had NO sigma_SR term at all, so it could not vary
    with dispersion. This is the defining property the old code lacked.
    """
    a = expected_max_sharpe(_sigma_injection(0.25), 100)
    b = expected_max_sharpe(_sigma_injection(0.50), 100)
    assert b == pytest.approx(2.0 * a, rel=1e-9)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- these fail against the superseded formula
# ---------------------------------------------------------------------------

def test_benchmark_is_a_sharpe_not_a_standard_normal_quantile() -> None:
    """NEGATIVE CONTROL. Old code returned 1.9733 here; correct value is ~0.75.

    Reproduces docs/reports/ramp-long-calls/20260402_statistical_validation.json
    exactly: n_trials=18 over 1635 daily observations (~6.5 years).
    """
    n_obs, n_trials = 1635, 18
    result = compute_deflated_sharpe(_returns_with_sharpe(0.5, n_obs), n_trials=n_trials)

    old_buggy_value = (
        np.sqrt(2 * np.log(n_trials))
        - (_EULER_MASCHERONI + np.log(np.log(n_trials) + np.pi / 2))
        / (2 * np.sqrt(2 * np.log(n_trials)))
    )
    assert old_buggy_value == pytest.approx(1.9733, abs=1e-3)   # the artifact's number

    sigma = 1.0 / np.sqrt(n_obs / 252.0)
    assert result.expected_max_sharpe == pytest.approx(
        expected_max_sharpe(_sigma_injection(sigma), n_trials), rel=1e-9)
    assert result.expected_max_sharpe == pytest.approx(0.75, abs=0.05)
    assert result.expected_max_sharpe < 1.0 < old_buggy_value


def test_the_old_formula_would_have_killed_a_viable_candidate() -> None:
    """NEGATIVE CONTROL. Sharpe 1.5 over 8 years at N=349.

    Old formula: hurdle 3.04 (a standard-normal quantile), so 1.5 - 3.04 < 0,
    statistic strongly negative, p_value ~1.0 -- a confident REJECT.
    New: hurdle 1.04 (an actual Sharpe), so the candidate clears it and scores
    DSR 0.90. Still short of the 0.95 gate, but no longer buried. The verdict
    direction reverses, which is the whole of Bug 2.
    """
    returns = _returns_with_sharpe(1.5, n=2016)
    result = compute_deflated_sharpe(returns, n_trials=349)

    n_trials = 349
    old_hurdle = (
        np.sqrt(2 * np.log(n_trials))
        - (_EULER_MASCHERONI + np.log(np.log(n_trials) + np.pi / 2))
        / (2 * np.sqrt(2 * np.log(n_trials)))
    )
    assert old_hurdle == pytest.approx(3.045, abs=0.01)
    assert old_hurdle > result.observed_sharpe          # old code: reject
    assert result.expected_max_sharpe < result.observed_sharpe   # new: clears
    assert result.dsr_statistic > 0
    assert result.dsr_probability == pytest.approx(0.904, abs=0.01)


def test_a_strong_honest_sharpe_passes_the_gate() -> None:
    """NEGATIVE CONTROL. Sharpe 1.8 over 8 years at N=349 must PASS at 0.95.

    Under the old formula the hurdle (3.04) still exceeds 1.8, so it would be
    rejected with p_value ~1.0. The gate must be passable by a genuinely
    strong candidate, or it is not a gate.
    """
    returns = _returns_with_sharpe(1.8, n=2016)
    result = compute_deflated_sharpe(returns, n_trials=349)

    assert result.observed_sharpe == pytest.approx(1.8, abs=0.01)
    assert result.dsr_probability > 0.95
    assert result.passed is True


def test_deflation_actually_bites_a_marginal_candidate() -> None:
    """The fix must not simply make everything pass.

    A Sharpe of 0.55 over the same window sits below the N=349 hurdle (~1.02)
    and must FAIL. Bug 1 (N collapsing to 0) would have passed it.
    """
    returns = _returns_with_sharpe(0.55, n=2016)
    result = compute_deflated_sharpe(returns, n_trials=349)
    assert result.expected_max_sharpe > result.observed_sharpe
    assert result.dsr_probability < 0.5
    assert result.passed is False


# ---------------------------------------------------------------------------
# Agreement between the two entry points -- one formula, one answer
# ---------------------------------------------------------------------------

def test_validation_adapter_agrees_with_statistics_dsr_bit_for_bit() -> None:
    """The two entry points must not disagree by 3x -- or at all."""
    returns = _returns_with_sharpe(1.1, n=1500)
    n_trials = 200
    result = compute_deflated_sharpe(returns, n_trials=n_trials)

    sd = returns.std()
    z = (returns - returns.mean()) / sd
    sigma = 1.0 / np.sqrt(len(returns) / 252.0)
    direct = psr(
        result.observed_sharpe,
        expected_max_sharpe(_sigma_injection(sigma), n_trials),
        len(returns),
        float(np.mean(z ** 3)),
        float(np.mean(z ** 4)),
        periods_per_year=252,
    )
    assert result.dsr_probability == pytest.approx(direct, rel=1e-12)
    assert result.p_value == pytest.approx(1.0 - direct, rel=1e-12)


def test_empirical_trial_sharpes_override_the_theoretical_prior() -> None:
    trial_sharpes = [0.1, 0.4, -0.2, 0.9, 0.3, 0.55, -0.05, 0.7]
    returns = _returns_with_sharpe(1.0, n=1500)
    result = compute_deflated_sharpe(returns, n_trials=50, trial_sharpes=trial_sharpes)
    assert result.expected_max_sharpe == pytest.approx(
        expected_max_sharpe(trial_sharpes, 50), rel=1e-12)


# ---------------------------------------------------------------------------
# Degenerate N
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_trials", [0, 1])
def test_n_trials_below_two_yields_no_deflation(n_trials: int) -> None:
    """With fewer than 2 trials there is no maximum to deflate against, so
    SR_0 = 0 and DSR degenerates to PSR(0). This is correct -- but it is also
    exactly what a silent N=0 produced project-wide, which is why
    `n_trials_project_wide()` now refuses to return 0."""
    returns = _returns_with_sharpe(1.0, n=1500)
    result = compute_deflated_sharpe(returns, n_trials=n_trials)
    assert result.expected_max_sharpe == 0.0
    assert result.dsr_probability == pytest.approx(
        psr(result.observed_sharpe, 0.0, len(returns),
            result.skewness, result.kurtosis + 3.0, periods_per_year=252),
        rel=1e-12)


def test_more_trials_never_lowers_the_hurdle() -> None:
    returns = _returns_with_sharpe(1.0, n=1500)
    hurdles = [compute_deflated_sharpe(returns, n_trials=k).expected_max_sharpe
               for k in (2, 10, 50, 349, 1000)]
    assert hurdles == sorted(hurdles)
    assert hurdles[0] > 0.0


def test_short_series_fails_closed() -> None:
    result = compute_deflated_sharpe(np.array([0.01, -0.01]), n_trials=10)
    assert result.passed is False
    assert result.n_observations == 2
    assert result.dsr_probability == 0.0


def test_zero_variance_series_does_not_explode() -> None:
    result = compute_deflated_sharpe(np.zeros(500), n_trials=10)
    assert np.isfinite(result.dsr_probability)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Structural guard: the second formula must stay gone
# ---------------------------------------------------------------------------

def test_validation_module_contains_no_second_dsr_formula() -> None:
    """Make the wrong implementation impossible to reintroduce by accident.

    `deflated_sharpe.py` must delegate. If someone re-adds a local
    Euler-Mascheroni closed form, this fails.
    """
    src = Path("src/backtesting/validation/deflated_sharpe.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    # Strip the module docstring, which quotes the old formula deliberately.
    body = code.split('"""')[-1]

    assert "0.5772" not in body, "local Euler-Mascheroni constant reintroduced"
    assert "sqrt(2 * log" not in body.replace("np.", "")
    assert "expected_max_sharpe" in body, "must delegate to statistics.dsr"
    assert "from src.backtesting.statistics.dsr import" in src
