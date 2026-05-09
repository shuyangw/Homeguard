"""Tests for Layer 1 structural checks."""
from pathlib import Path

import polars as pl
import pytest

from src.data.validation.core.result import Severity
from src.data.validation.futures.checks.structural import (
    SectionPresenceCheck,
    PartitionKeysValidCheck,
    FileIntegrityCheck,
)
from tests.data.validation.fixtures.build_fixtures import build_minimal_futures_store


@pytest.fixture
def valid_store(tmp_path: Path, monkeypatch):
    build_minimal_futures_store(tmp_path)
    monkeypatch.setattr(
        "src.data.validation.futures.checks.structural._storage_root",
        lambda: tmp_path,
    )
    return tmp_path


def test_section_presence_pass(valid_store):
    r = SectionPresenceCheck().run()
    assert r.passed is True


def test_section_presence_critical_when_section_missing(tmp_path: Path, monkeypatch):
    build_minimal_futures_store(tmp_path)
    # remove a section
    import shutil
    shutil.rmtree(tmp_path / "futures_definitions")
    monkeypatch.setattr(
        "src.data.validation.futures.checks.structural._storage_root",
        lambda: tmp_path,
    )
    r = SectionPresenceCheck().run()
    assert r.passed is False
    assert r.severity == Severity.CRITICAL
    assert "futures_definitions" in str(r.details)


def test_partition_keys_valid_pass(valid_store):
    r = PartitionKeysValidCheck().run()
    assert r.passed is True


def test_partition_keys_invalid_year(tmp_path: Path, monkeypatch):
    build_minimal_futures_store(tmp_path)
    bad = tmp_path / "futures_1min" / "symbol=ES" / "year=1900" / "month=1"
    bad.mkdir(parents=True)
    (bad / "data.parquet").write_text("")
    monkeypatch.setattr(
        "src.data.validation.futures.checks.structural._storage_root",
        lambda: tmp_path,
    )
    r = PartitionKeysValidCheck().run()
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


def test_file_integrity_pass(valid_store):
    r = FileIntegrityCheck(mode="initial").run()
    assert r.passed is True


def test_file_integrity_fail_on_corrupt(tmp_path: Path, monkeypatch):
    build_minimal_futures_store(tmp_path)
    # corrupt a parquet by overwriting with garbage
    target = tmp_path / "futures_1min" / "symbol=ES" / "year=2024" / "month=6" / "data.parquet"
    target.write_bytes(b"not a parquet file")
    monkeypatch.setattr(
        "src.data.validation.futures.checks.structural._storage_root",
        lambda: tmp_path,
    )
    r = FileIntegrityCheck(mode="initial").run()
    assert r.passed is False
    assert r.severity == Severity.CRITICAL
