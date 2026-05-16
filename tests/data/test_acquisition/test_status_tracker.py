"""Tests for the per-pass status tracker."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.acquisition.manifest import DownloadManifest


class TestComputeStatus:
    def test_complete_with_rows_is_good(self):
        from src.data.acquisition.status_tracker import compute_status

        s = compute_status(
            manifest_entry={"status": "complete", "rows": 100},
            on_disk_rows=100,
            validation_error=None,
        )
        assert s == "good"

    def test_complete_with_zero_rows_is_incomplete(self):
        from src.data.acquisition.status_tracker import compute_status

        s = compute_status(
            manifest_entry={"status": "complete", "rows": 0},
            on_disk_rows=0,
            validation_error=None,
        )
        assert s == "incomplete"

    def test_validation_error_overrides_to_broken(self):
        from src.data.acquisition.status_tracker import compute_status

        s = compute_status(
            manifest_entry={"status": "complete", "rows": 100},
            on_disk_rows=100,
            validation_error="non-monotonic timestamps",
        )
        assert s == "broken"

    def test_failed_status_passthrough(self):
        from src.data.acquisition.status_tracker import compute_status

        s = compute_status(
            manifest_entry={"status": "failed", "error": "API timeout"},
            on_disk_rows=0,
            validation_error=None,
        )
        assert s == "failed"

    def test_missing_entry_is_pending(self):
        from src.data.acquisition.status_tracker import compute_status

        s = compute_status(
            manifest_entry=None,
            on_disk_rows=0,
            validation_error=None,
        )
        assert s == "pending"


class TestWriteTrackerCsv:
    def test_write_atomic(self):
        from src.data.acquisition.status_tracker import write_tracker_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "tracker.csv"
            rows = [
                {
                    "symbol": "AAPL", "status": "good",
                    "rows": 100, "first_date": "2024-01-02",
                    "last_date": "2024-12-31", "trading_days": 252,
                    "partitions": 12, "download_error": None,
                    "validation_error": None, "last_updated": "2026-05-16T00:00:00Z",
                },
            ]
            write_tracker_csv(out_path, rows)
            assert out_path.exists()
            df = pd.read_csv(out_path)
            assert df.iloc[0]["symbol"] == "AAPL"
            assert df.iloc[0]["status"] == "good"

    def test_atomic_write_via_tmp(self, monkeypatch):
        from src.data.acquisition.status_tracker import write_tracker_csv

        seen_paths = []
        original_to_csv = pd.DataFrame.to_csv

        def capture(self, path, *args, **kwargs):
            seen_paths.append(str(path))
            return original_to_csv(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", capture)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "tracker.csv"
            write_tracker_csv(out_path, [{"symbol": "AAPL", "status": "good"}])

            assert any(p.endswith(".tmp") for p in seen_paths)
            assert out_path.exists()
            assert not Path(str(out_path) + ".tmp").exists()


class TestRebuildTracker:
    def test_rebuild_joins_manifest_and_disk(self):
        from src.data.acquisition.status_tracker import rebuild_tracker

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manifest = DownloadManifest(base, "equities_1min_sip_raw")
            manifest.set_entry("AAPL", status="complete", rows=100)
            manifest.set_entry("FAILED1", status="failed", error="api err")
            manifest.save()

            # Simulate AAPL parquet on disk
            partition_dir = (
                base / "equities_1min_sip_raw" / "symbol=AAPL"
                / "year=2024" / "month=1"
            )
            partition_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
                        utc=True,
                    ),
                    "open": [100.0, 100.1], "high": [101.0, 101.1],
                    "low": [99.0, 99.1], "close": [100.5, 100.6],
                    "volume": [1000.0, 1100.0], "trade_count": [50, 55],
                    "vwap": [100.2, 100.3],
                }
            ).to_parquet(partition_dir / "data.parquet", index=False)

            rows = rebuild_tracker(
                base_dir=base,
                subdir="equities_1min_sip_raw",
                universe=["AAPL", "FAILED1", "PENDING1"],
            )

            by_symbol = {r["symbol"]: r for r in rows}
            assert by_symbol["AAPL"]["status"] == "good"
            assert by_symbol["AAPL"]["rows"] == 2
            assert by_symbol["FAILED1"]["status"] == "failed"
            assert by_symbol["PENDING1"]["status"] == "pending"
