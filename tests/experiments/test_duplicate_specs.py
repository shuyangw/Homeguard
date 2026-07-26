"""Duplicate-spec detection.

Exact-duplicate rows have reached the registry before (RORO, PCA,
Carry-Seatbelt). They inflate the trial count N, which is the SAFE direction, so
the guard SURFACES them rather than dropping them: silently skipping the insert
would shrink N, and N never shrinks.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.experiments import append_run
from src.experiments.registry import duplicate_spec_run_ids


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "experiments.duckdb"


def _append(db_path, **overrides):
    kwargs = dict(
        strategy_name="FxRoro", agent_name="test",
        metrics={"sharpe": 0.1}, params={"lookback": 20},
        asset_class="fx", data_frequency="daily",
        window_start=date(2011, 1, 1), window_end=date(2026, 1, 1),
        db_path=db_path)
    kwargs.update(overrides)
    return append_run(**kwargs)


def test_duplicate_spec_is_still_inserted(db_path):
    """N never shrinks: the row goes in, it is merely flagged."""
    first = _append(db_path)
    second = _append(db_path)
    assert first != second
    assert set(duplicate_spec_run_ids(db_path)[0]["run_ids"]) == {first, second}


def test_no_duplicates_reported_for_a_single_run(db_path):
    _append(db_path)
    assert duplicate_spec_run_ids(db_path) == []


def test_different_params_are_not_duplicates(db_path):
    _append(db_path)
    _append(db_path, params={"lookback": 40})
    assert duplicate_spec_run_ids(db_path) == []


def test_different_window_is_not_a_duplicate(db_path):
    _append(db_path)
    _append(db_path, window_end=date(2025, 1, 1))
    assert duplicate_spec_run_ids(db_path) == []


def test_different_strategy_is_not_a_duplicate(db_path):
    _append(db_path)
    _append(db_path, strategy_name="FxPca")
    assert duplicate_spec_run_ids(db_path) == []


def test_duplicate_is_logged_as_an_error(db_path, monkeypatch):
    """Homeguard's logger is a Rich wrapper, not stdlib logging, so caplog
    cannot see it; assert on the call instead."""
    from src.experiments import registry
    errors = []
    monkeypatch.setattr(registry.logger, "error", errors.append)
    _append(db_path)
    assert errors == []
    _append(db_path)
    assert len(errors) == 1 and "duplicate" in errors[0].lower()


def test_three_identical_specs_group_together(db_path):
    ids = {_append(db_path) for _ in range(3)}
    dups = duplicate_spec_run_ids(db_path)
    assert len(dups) == 1
    assert set(dups[0]["run_ids"]) == ids


def test_null_params_are_never_flagged(db_path):
    """Without params there is no recoverable spec identity. Grouping on the null
    would merge genuinely different runs -- RAMP-V31 carries 47 such rows with 45
    distinct metric sets."""
    _append(db_path, params=None, metrics={"sharpe": 0.1})
    _append(db_path, params=None, metrics={"sharpe": 0.9})
    assert duplicate_spec_run_ids(db_path) == []


def test_null_params_do_not_trigger_the_warning(db_path, monkeypatch):
    from src.experiments import registry
    errors = []
    monkeypatch.setattr(registry.logger, "error", errors.append)
    _append(db_path, params=None)
    _append(db_path, params=None)
    assert errors == []
