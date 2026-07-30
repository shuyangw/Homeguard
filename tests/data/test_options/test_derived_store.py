"""Tests for the options derived-daily store (Phase-1 Gap 2).

The derived dailies move out of the Wave-0 scratch path (`output/wave0/derived/`,
which is gitignored) into the conventional materialized location alongside
`spread_census`, each with a provenance sidecar matching the shape
`regime_state_daily` / `vix_spot` already use.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.options import derived_store as ds


@pytest.fixture
def frame():
    return pd.DataFrame({
        "root": ["SPY", "SPY", "IWM"],
        "session_date": pd.to_datetime(["2017-01-03", "2018-05-04", "2020-02-03"]).date,
        "atm_iv": [0.11, 0.12, 0.35],
    })


def test_path_follows_the_spread_census_convention(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    p = ds.derived_table_path("atm_iv_daily")
    assert p == tmp_path / "options" / "derived" / "atm_iv_daily" / "atm_iv_daily.parquet"
    assert ds.derived_meta_path("atm_iv_daily") == p.with_suffix(".meta.json")


def test_write_then_load_roundtrips(tmp_path, monkeypatch, frame):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    ds.write_derived_table("atm_iv_daily", frame, source="options_chain_eod")
    out = ds.load_derived_table("atm_iv_daily")
    assert len(out) == 3
    assert set(out["root"]) == {"SPY", "IWM"}


def test_sidecar_carries_the_required_provenance_fields(tmp_path, monkeypatch, frame):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    ds.write_derived_table("atm_iv_daily", frame, source="options_chain_eod")
    meta = json.loads(ds.derived_meta_path("atm_iv_daily").read_text(encoding="utf-8"))
    for field in ("dataset", "source", "snapshot_timestamp", "git_sha", "rows",
                  "date_min", "date_max", "roots", "rows_by_root"):
        assert field in meta, f"missing provenance field {field}"
    assert meta["dataset"] == "atm_iv_daily"
    assert meta["rows"] == 3
    assert meta["date_min"] == "2017-01-03"
    assert meta["date_max"] == "2020-02-03"
    assert meta["roots"] == ["IWM", "SPY"]
    assert meta["rows_by_root"] == {"IWM": 1, "SPY": 2}


def test_sidecar_records_per_root_coverage_span(tmp_path, monkeypatch, frame):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    ds.write_derived_table("atm_iv_daily", frame, source="options_chain_eod")
    meta = json.loads(ds.derived_meta_path("atm_iv_daily").read_text(encoding="utf-8"))
    assert meta["coverage_by_root"]["SPY"] == {
        "date_min": "2017-01-03", "date_max": "2018-05-04", "rows": 2}


def test_extra_meta_is_merged(tmp_path, monkeypatch, frame):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    ds.write_derived_table("atm_iv_daily", frame, source="options_chain_eod",
                           extra={"build_census": {"SPY": 1}})
    meta = json.loads(ds.derived_meta_path("atm_iv_daily").read_text(encoding="utf-8"))
    assert meta["build_census"] == {"SPY": 1}


def test_git_sha_is_a_real_forty_char_sha(tmp_path, monkeypatch, frame):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    ds.write_derived_table("atm_iv_daily", frame, source="options_chain_eod")
    meta = json.loads(ds.derived_meta_path("atm_iv_daily").read_text(encoding="utf-8"))
    assert len(meta["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in meta["git_sha"])


def test_load_missing_table_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        ds.load_derived_table("does_not_exist")


def test_table_without_root_column_still_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "get_local_storage_dir", lambda: tmp_path)
    df = pd.DataFrame({"session_date": pd.to_datetime(["2020-01-02"]).date, "x": [1]})
    ds.write_derived_table("t", df, source="s")
    meta = json.loads(ds.derived_meta_path("t").read_text(encoding="utf-8"))
    assert meta["roots"] == []
    assert meta["rows"] == 1
