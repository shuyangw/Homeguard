"""Tests for the V5 usable-value re-score (Phase-1 Gap 3).

The registered V5 gate text is restated verbatim in the module under test and is
NOT changed here. What changes is the METRIC the 90% threshold is applied to:
non-null rate -> usable-value rate (finite value AND the registered sanity
bounds), per CORRECTION ADDENDUM C1 of the V-battery report.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.data.vbattery.rescore_v5_usable import (
    V5_THRESHOLD,
    census_usability_by_root_year,
    rescore_v5,
)


def _row(**kw):
    base = dict(
        root="X",
        year=2020,
        n_rows=1000,
        nonnull_implied_vol=1.0,
        nonnull_delta=1.0,
        nonnull_theta=1.0,
        nonnull_vega=1.0,
        plausible_iv_frac=1.0,
        abs_delta_le1_frac=1.0,
        delta_sign_ok_frac=1.0,
        gate="PASS",
    )
    base.update(kw)
    return base


def test_threshold_is_the_registered_ninety_percent():
    assert V5_THRESHOLD == 0.90


def test_all_nan_root_year_scores_zero_usable_and_fails():
    df = pd.DataFrame([_row(
        nonnull_implied_vol=0.0, nonnull_delta=0.0, nonnull_theta=0.0,
        nonnull_vega=0.0, plausible_iv_frac=0.0, abs_delta_le1_frac=0.0,
        delta_sign_ok_frac=0.0, gate="UNTRUSTED")])
    out = rescore_v5(df)
    assert out.loc[0, "usable_rate"] == 0.0
    assert out.loc[0, "gate_usable"] == "UNTRUSTED"


def test_fully_usable_root_year_passes():
    out = rescore_v5(pd.DataFrame([_row()]))
    assert out.loc[0, "usable_rate"] == 1.0
    assert out.loc[0, "gate_usable"] == "PASS"


def test_usable_rate_is_the_min_across_the_four_greek_columns():
    df = pd.DataFrame([_row(nonnull_vega=0.42)])
    out = rescore_v5(df)
    assert out.loc[0, "usable_rate"] == pytest.approx(0.42)


def test_sanity_screen_binds_even_when_finite_rate_is_one():
    """A root-year 100% finite but only 80% inside IV in (0.01, 5.0) FAILS.

    This is the substantive correction: the shipped gate bound on the finite
    rate alone and ignored the registered sanity bounds.
    """
    df = pd.DataFrame([_row(plausible_iv_frac=0.80, gate="PASS")])
    out = rescore_v5(df)
    assert out.loc[0, "usable_rate"] == pytest.approx(0.80)
    assert out.loc[0, "gate_usable"] == "UNTRUSTED"
    assert out.loc[0, "changed"]


def test_delta_sign_violation_binds():
    df = pd.DataFrame([_row(delta_sign_ok_frac=0.5)])
    out = rescore_v5(df)
    assert out.loc[0, "usable_rate"] == pytest.approx(0.5)
    assert out.loc[0, "gate_usable"] == "UNTRUSTED"


def test_boundary_is_inclusive_at_ninety_percent():
    out = rescore_v5(pd.DataFrame([_row(plausible_iv_frac=0.90)]))
    assert out.loc[0, "gate_usable"] == "PASS"
    out = rescore_v5(pd.DataFrame([_row(plausible_iv_frac=0.8999)]))
    assert out.loc[0, "gate_usable"] == "UNTRUSTED"


def test_changed_flags_only_differences_from_the_shipped_gate():
    df = pd.DataFrame([
        _row(root="A", gate="PASS"),
        _row(root="B", plausible_iv_frac=0.10, gate="PASS"),
    ])
    out = rescore_v5(df).set_index("root")
    assert not out.loc["A", "changed"]
    assert out.loc["B", "changed"]


def test_census_usability_counts_partitions_by_class():
    census = pd.DataFrame([
        {"root": "R", "year": 2016, "month": 1, "greek_status": "ALL_NAN"},
        {"root": "R", "year": 2016, "month": 2, "greek_status": "NO_COLUMN"},
        {"root": "R", "year": 2017, "month": 1, "greek_status": "OK"},
        {"root": "R", "year": 2017, "month": 2, "greek_status": "PARTIAL_NAN"},
    ])
    out = census_usability_by_root_year(census).set_index(["root", "year"])
    assert out.loc[("R", 2016), "n_partitions"] == 2
    assert out.loc[("R", 2016), "n_ok"] == 0
    assert out.loc[("R", 2016), "census_ok_frac"] == 0.0
    assert out.loc[("R", 2017), "n_ok"] == 1
    assert out.loc[("R", 2017), "n_partial_nan"] == 1
    assert out.loc[("R", 2017), "census_ok_frac"] == pytest.approx(0.5)
