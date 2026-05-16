"""Tests for the unified manifest tracker."""

import json
import tempfile
import threading
from pathlib import Path

import pytest

from src.data.acquisition.manifest import DownloadManifest


class TestManifestBasic:
    def test_create_new_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            assert manifest.source == "futures"
            assert manifest.get_all_entries() == {}

    def test_set_and_get_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete", rows=28450)
            entry = manifest.get_entry("ES|2024-01")
            assert entry["status"] == "complete"
            assert entry["rows"] == 28450

    def test_update_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="in_progress")
            manifest.set_entry("ES|2024-01", status="complete", rows=1000)
            entry = manifest.get_entry("ES|2024-01")
            assert entry["status"] == "complete"
            assert entry["rows"] == 1000

    def test_get_nonexistent_entry_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            assert manifest.get_entry("NONEXISTENT") is None


class TestManifestPersistence:
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete", rows=100)
            manifest.save()

            manifest2 = DownloadManifest(Path(tmpdir), "futures")
            entry = manifest2.get_entry("ES|2024-01")
            assert entry["status"] == "complete"

    def test_manifest_file_location(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="pending")
            manifest.save()
            expected_path = Path(tmpdir) / "_manifests" / "futures.json"
            assert expected_path.exists()

    def test_manifest_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete", rows=100)
            manifest.save()
            path = Path(tmpdir) / "_manifests" / "futures.json"
            data = json.loads(path.read_text())
            assert data["source"] == "futures"
            assert "ES|2024-01" in data["entries"]


class TestManifestFiltering:
    def test_get_pending_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete")
            manifest.set_entry("ES|2024-02", status="pending")
            manifest.set_entry("NQ|2024-01", status="failed", error="timeout")
            pending = manifest.get_entries_by_status("pending")
            assert list(pending.keys()) == ["ES|2024-02"]

    def test_get_failed_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete")
            manifest.set_entry("NQ|2024-01", status="failed", error="timeout")
            failed = manifest.get_entries_by_status("failed")
            assert "NQ|2024-01" in failed
            assert failed["NQ|2024-01"]["error"] == "timeout"

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            manifest.set_entry("ES|2024-01", status="complete")
            manifest.set_entry("ES|2024-02", status="complete")
            manifest.set_entry("NQ|2024-01", status="failed", error="x")
            manifest.set_entry("CL|2024-01", status="pending")
            summary = manifest.summary()
            assert summary == {
                "total": 4,
                "complete": 2,
                "failed": 1,
                "pending": 1,
                "in_progress": 0,
            }


class TestManifestThreadSafety:
    def test_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "futures")
            errors = []

            def write_entry(symbol_idx):
                try:
                    for month in range(1, 13):
                        key = f"SYM{symbol_idx}|2024-{month:02d}"
                        manifest.set_entry(key, status="complete", rows=100)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=write_entry, args=(i,))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(manifest.get_all_entries()) == 60  # 5 symbols x 12 months


class TestReapInProgress:
    def test_reap_in_progress_downgrades_to_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "equities_1min_sip_raw")
            manifest.set_entry("AAPL", status="complete", rows=100)
            manifest.set_entry("MSFT", status="in_progress")
            manifest.set_entry("GOOG", status="in_progress")
            manifest.set_entry("TSLA", status="failed", error="x")

            reaped = manifest.reap_in_progress()

            assert set(reaped) == {"MSFT", "GOOG"}
            assert manifest.get_entry("MSFT")["status"] == "pending"
            assert manifest.get_entry("GOOG")["status"] == "pending"
            assert "interrupted_at" in manifest.get_entry("MSFT")
            assert manifest.get_entry("AAPL")["status"] == "complete"
            assert manifest.get_entry("TSLA")["status"] == "failed"

    def test_reap_in_progress_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = DownloadManifest(Path(tmpdir), "equities_1min_sip_raw")
            reaped = manifest.reap_in_progress()
            assert reaped == []
