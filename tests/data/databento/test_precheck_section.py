"""Tests for precheck_section() in scripts/data/databento_batch_submit.py.

The module is `scripts.data.databento_batch_submit` after sys.path insertion;
tests do that insertion at import time mirroring how the production scripts run.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from databento_batch_submit import precheck_section  # noqa: E402


def _mk_client(
    symbology_result_count: int = 5,
    cost_usd: float = 1.0,
    record_count: int = 1_000_000,
    raise_symbology: bool = False,
):
    """Build a MagicMock for db.Historical with the 3 metadata probes set up."""
    client = MagicMock()
    if raise_symbology:
        client.symbology.resolve.side_effect = Exception("BentoClientError: 422")
    else:
        client.symbology.resolve.return_value = {
            "result": {f"SYM{i}": ["instr"] for i in range(symbology_result_count)},
            "partial": [],
            "not_found": [],
        }
    client.metadata.get_cost.return_value = cost_usd
    client.metadata.get_record_count.return_value = record_count
    return client


def test_precheck_pass_case():
    """Valid symbology + low cost + low volume -> passed=True."""
    client = _mk_client(
        symbology_result_count=2,
        cost_usd=0.0,
        record_count=1_200_000,
    )
    passed, summary = precheck_section(
        client,
        section="Test_OK",
        schema="ohlcv-1d",
        symbols=["ES.FUT", "NQ.FUT"],
        stype_in="parent",
        start="2024-01-01",
        end="2024-12-31",
        dataset="GLBX.MDP3",
    )
    assert passed is True, f"expected pass, got: {summary}"
    assert "OK:" in summary
    assert "$0.00" in summary
    assert "1,200,000" in summary


def test_precheck_fail_on_bad_symbol():
    """Symbology probe raises -> passed=False, summary mentions symbology."""
    client = _mk_client(raise_symbology=True)
    passed, summary = precheck_section(
        client,
        section="Test_BadSym",
        schema="ohlcv-1d",
        symbols=["ED.FUT"],
        stype_in="parent",
        start="2010-01-01",
        end="2023-12-31",
        dataset="GLBX.MDP3",
    )
    assert passed is False
    assert "symbology" in summary.lower()


def test_precheck_fail_on_high_cost():
    """Cost above threshold -> passed=False, summary mentions cost."""
    client = _mk_client(cost_usd=1040.68)
    passed, summary = precheck_section(
        client,
        section="Test_Expensive",
        schema="trades",
        symbols=["ES.v.0", "MES.v.0"],
        stype_in="continuous",
        start="2021-01-01",
        end="2026-02-22",
        dataset="GLBX.MDP3",
        budget_threshold_usd=50.0,
    )
    assert passed is False
    assert "cost" in summary.lower()
    assert "$1040.68" in summary


def test_precheck_fail_on_high_volume():
    """Record count above threshold -> passed=False, summary mentions volume."""
    client = _mk_client(record_count=6_084_258_583)
    passed, summary = precheck_section(
        client,
        section="Test_HugeF",
        schema="mbp-1",
        symbols=["ES.v.0", "MES.v.0", "NQ.v.0", "MNQ.v.0"],
        stype_in="continuous",
        start="2025-08-22",
        end="2026-02-22",
        dataset="GLBX.MDP3",
        volume_threshold_records=100_000_000,
    )
    assert passed is False
    assert "volume" in summary.lower() or "record" in summary.lower()
    assert "6,084,258,583" in summary


def test_precheck_short_circuits_on_first_failure():
    """If symbology fails, cost and record_count should NOT be called."""
    client = _mk_client(raise_symbology=True)
    precheck_section(
        client,
        section="Test_Short",
        schema="ohlcv-1d",
        symbols=["BAD.FUT"],
        stype_in="parent",
        start="2024-01-01",
        end="2024-12-31",
        dataset="GLBX.MDP3",
    )
    client.symbology.resolve.assert_called_once()
    client.metadata.get_cost.assert_not_called()
    client.metadata.get_record_count.assert_not_called()
