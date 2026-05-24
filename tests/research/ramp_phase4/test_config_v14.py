"""Tests for V14 config fields + validation predicate."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.research.ramp_phase4.config import (
    HarnessConfig, load_v14_tau_constants,
)


def _base_cfg(**overrides) -> HarnessConfig:
    defaults = dict(
        start_date=datetime(2017, 1, 3),
        end_date=datetime(2026, 5, 22),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        soft_bear_tau_in=0.3,
        soft_bear_tau_out=0.2,
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


def test_v14_fields_defaults():
    cfg = _base_cfg()
    assert cfg.soft_bear_tau_in == 0.3
    assert cfg.soft_bear_tau_out == 0.2
    assert cfg.soft_bear_dampen_factor == 0.5


def test_tau_predicate_tau_out_zero_rejected():
    with pytest.raises(ValueError, match='tau_out'):
        _base_cfg(soft_bear_tau_in=0.3, soft_bear_tau_out=0.0)


def test_tau_predicate_tau_in_one_rejected():
    with pytest.raises(ValueError, match='tau_in'):
        _base_cfg(soft_bear_tau_in=1.0, soft_bear_tau_out=0.5)


def test_tau_predicate_inverted_rejected():
    with pytest.raises(ValueError, match='tau_out.*tau_in'):
        _base_cfg(soft_bear_tau_in=0.2, soft_bear_tau_out=0.3)


def test_tau_predicate_equal_rejected():
    with pytest.raises(ValueError, match='tau_out.*tau_in'):
        _base_cfg(soft_bear_tau_in=0.3, soft_bear_tau_out=0.3)


def test_dampen_factor_out_of_range_rejected():
    with pytest.raises(ValueError, match='dampen_factor'):
        _base_cfg(soft_bear_dampen_factor=1.5)


def test_dampen_factor_negative_rejected():
    with pytest.raises(ValueError, match='dampen_factor'):
        _base_cfg(soft_bear_dampen_factor=-0.1)


def test_load_v14_tau_constants_from_json(tmp_path):
    p = tmp_path / 'tau.json'
    p.write_text(json.dumps({
        'tau_in': 0.35, 'tau_out': 0.25,
        'tau_band': 0.1, 'g1_labeler_commit': 'abc',
    }))
    tau_in, tau_out = load_v14_tau_constants(p)
    assert tau_in == 0.35
    assert tau_out == 0.25


def test_load_v14_tau_constants_default_path():
    """Default path exists from Task 0."""
    tau_in, tau_out = load_v14_tau_constants()
    assert 0.0 < tau_out < tau_in < 1.0
