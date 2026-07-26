"""PSR/DSR unit-consistency (bug found by the Tier B gate, 2026-07-25).

The z-score scales by sqrt(n-1) where n counts RETURN OBSERVATIONS, so the
Sharpe must be PER-OBSERVATION. Every walk-forward runner was passing an
ANNUALIZED Sharpe against a DAILY n, inflating z by ~sqrt(252) and making PSR
wildly optimistic -- a corrected 0.57 was reported as 0.998.
"""
import numpy as np

from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr


def test_default_is_per_period_and_unchanged():
    """periods_per_year defaults to 1.0 so per-period callers are untouched."""
    assert psr(0.05, 0.0, 1000, 0.0, 3.0) == psr(0.05, 0.0, 1000, 0.0, 3.0, periods_per_year=1.0)


def test_annualized_input_without_the_flag_is_wildly_optimistic():
    """Pins the bug's magnitude: the Tier B TOT-OIL case."""
    buggy = psr(0.0505, 0.0, 3180, 0.0, 3.0)
    fixed = psr(0.0505, 0.0, 3180, 0.0, 3.0, periods_per_year=252)
    assert buggy > 0.99          # looks like a decisive pass
    assert 0.55 < fixed < 0.60   # actually a coin flip
    assert buggy - fixed > 0.4


def test_deannualizing_matches_passing_a_daily_sharpe_directly():
    sr_annual, n = 0.80, 3000
    direct = psr(sr_annual / np.sqrt(252), 0.0, n, 0.0, 3.0)
    viaflag = psr(sr_annual, 0.0, n, 0.0, 3.0, periods_per_year=252)
    assert abs(direct - viaflag) < 1e-12


def test_dsr_passes_the_flag_through_consistently():
    """sr_zero inherits the units of trial_sharpes, so both must be de-annualized
    together -- otherwise the benchmark and the candidate are in different units."""
    trials = [0.2, -0.1, 0.5, 0.3, -0.4]
    a = dsr(0.8, trials, 3000, 0.0, 3.0, n_trials_project=100, periods_per_year=252)
    b = psr(0.8, __import__("src.backtesting.statistics.dsr", fromlist=["x"])
            .expected_max_sharpe(trials, 100), 3000, 0.0, 3.0, periods_per_year=252)
    assert abs(a - b) < 1e-12


def test_fix_is_conservative_for_a_positive_sharpe():
    """Direction check: correcting the units can only make a positive-Sharpe PSR
    HARDER to pass, never easier. The bug could manufacture a false PSR pass."""
    for sr in (0.1, 0.5, 1.0):
        assert psr(sr, 0.0, 3000, 0.0, 3.0, periods_per_year=252) < psr(sr, 0.0, 3000, 0.0, 3.0)
