"""Tests for BaseDownloader shared infrastructure."""

import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.acquisition.base import BaseDownloader, DownloadResult
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


class StubPlugin(BaseDownloader):
    """Concrete test implementation of BaseDownloader."""

    def __init__(self, fetch_fn=None, **kwargs):
        self._fetch_fn = fetch_fn
        super().__init__(**kwargs)

    def _create_client(self) -> Any:
        return MagicMock()

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        if self._fetch_fn:
            return self._fetch_fn(symbol, start, end)
        # Return valid OHLCV data by default
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-02 09:30:00"], utc=True
                ),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "trade_count": [50.0],
                "vwap": [100.2],
            }
        )

    def _get_schema(self) -> list[str]:
        return CANONICAL_OHLCV_SCHEMA

    def _get_storage_subdir(self) -> str:
        return "test_data"

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol


class TestDownloadResult:
    def test_success_rate(self):
        r = DownloadResult(
            total_symbols=10,
            succeeded=7,
            failed=3,
            total_rows=700,
            elapsed_seconds=10.0,
        )
        assert r.success_rate == 70.0

    def test_success_rate_zero(self):
        r = DownloadResult(
            total_symbols=0,
            succeeded=0,
            failed=0,
            total_rows=0,
            elapsed_seconds=0.0,
        )
        assert r.success_rate == 0.0


class TestHivePartitioning:
    def test_hive_partition_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(output_dir=Path(tmpdir))
            result = plugin.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.succeeded == 1
            parquet = (
                Path(tmpdir)
                / "test_data"
                / "symbol=AAPL"
                / "year=2024"
                / "month=1"
                / "data.parquet"
            )
            assert parquet.exists()

    def test_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing data
            existing_dir = (
                Path(tmpdir)
                / "test_data"
                / "symbol=AAPL"
                / "year=2024"
                / "month=1"
            )
            existing_dir.mkdir(parents=True)
            (existing_dir / "data.parquet").write_text("fake")

            call_count = 0

            def counting_fetch(sym, start, end):
                nonlocal call_count
                call_count += 1
                return pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            ["2024-01-02 09:30:00"], utc=True
                        ),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1000.0],
                        "trade_count": [50.0],
                        "vwap": [100.2],
                    }
                )

            plugin = StubPlugin(
                output_dir=Path(tmpdir), fetch_fn=counting_fetch
            )
            result = plugin.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                skip_existing=True,
            )
            assert call_count == 0  # Should not have fetched
            assert result.succeeded == 0
            assert result.failed == 0


class TestRetryLogic:
    def test_retry_on_failure(self):
        attempts = []

        def failing_then_success(sym, start, end):
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("API timeout")
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-02 09:30:00"], utc=True
                    ),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                    "trade_count": [50.0],
                    "vwap": [100.2],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(
                output_dir=Path(tmpdir),
                fetch_fn=failing_then_success,
                num_threads=1,
                retry_delay=0.01,
            )
            result = plugin.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.succeeded == 1
            assert len(attempts) == 3  # Failed twice, succeeded on third

    def test_max_retries_exceeded(self):
        def always_fail(sym, start, end):
            raise ConnectionError("permanent failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(
                output_dir=Path(tmpdir),
                fetch_fn=always_fail,
                num_threads=1,
                retry_delay=0.01,
            )
            result = plugin.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.succeeded == 0
            assert result.failed == 1


class TestSchemaValidation:
    def test_schema_validation_rejects_bad_data(self):
        def bad_schema_fetch(sym, start, end):
            return pd.DataFrame({"wrong_col": [1], "another": [2]})

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(
                output_dir=Path(tmpdir),
                fetch_fn=bad_schema_fetch,
                num_threads=1,
                retry_delay=0.01,
            )
            result = plugin.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            assert result.failed == 1


class TestManifestIntegration:
    def test_manifest_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(output_dir=Path(tmpdir))
            plugin.download(
                symbols=["AAPL", "MSFT"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            manifest = plugin.manifest
            summary = manifest.summary()
            assert summary["complete"] == 2

    def test_manifest_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: download AAPL
            plugin1 = StubPlugin(output_dir=Path(tmpdir))
            plugin1.download(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )
            plugin1.manifest.save()

            # Second run: should see AAPL as complete
            plugin2 = StubPlugin(output_dir=Path(tmpdir))
            entry = plugin2.manifest.get_entry("AAPL")
            assert entry is not None
            assert entry["status"] == "complete"


class TestAtomicParquetWrite:
    def test_save_partitioned_writes_via_tmp_then_rename(self, monkeypatch):
        """Verify _save_partitioned writes to .tmp file first, then atomically renames."""
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(output_dir=Path(tmpdir))
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-02 09:30:00"], utc=True
                    ),
                    "open": [100.0], "high": [101.0], "low": [99.0],
                    "close": [100.5], "volume": [1000.0],
                    "trade_count": [50.0], "vwap": [100.2],
                }
            )

            tmp_paths_seen = []
            real_to_parquet = pd.DataFrame.to_parquet

            def capture_tmp(self, path, *args, **kwargs):
                tmp_paths_seen.append(str(path))
                return real_to_parquet(self, path, *args, **kwargs)

            monkeypatch.setattr(pd.DataFrame, "to_parquet", capture_tmp)

            plugin._save_partitioned(df, "AAPL")

            assert tmp_paths_seen, "to_parquet was never called"
            assert all(p.endswith(".tmp") for p in tmp_paths_seen), (
                f"Expected all writes to .tmp paths, got: {tmp_paths_seen}"
            )
            final_path = (
                Path(tmpdir) / "test_data" / "symbol=AAPL"
                / "year=2024" / "month=1" / "data.parquet"
            )
            assert final_path.exists()
            assert not Path(str(final_path) + ".tmp").exists()

    def test_save_partitioned_no_partial_file_on_failure(self, monkeypatch):
        """Simulated kill in to_parquet leaves no data.parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = StubPlugin(output_dir=Path(tmpdir))
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-02 09:30:00"], utc=True
                    ),
                    "open": [100.0], "high": [101.0], "low": [99.0],
                    "close": [100.5], "volume": [1000.0],
                    "trade_count": [50.0], "vwap": [100.2],
                }
            )

            def boom(self, path, *args, **kwargs):
                raise OSError("simulated kill")

            monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)

            with pytest.raises(OSError):
                plugin._save_partitioned(df, "AAPL")

            final_path = (
                Path(tmpdir) / "test_data" / "symbol=AAPL"
                / "year=2024" / "month=1" / "data.parquet"
            )
            assert not final_path.exists()
