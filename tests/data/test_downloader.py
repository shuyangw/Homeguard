"""
Unit tests for src/data/downloader.py

Tests the AlpacaDownloader class and CLI utilities with mocked Alpaca API.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.downloader import (
    AlpacaDownloader,
    DownloadResult,
    Timeframe,
    _format_time,
    _get_thread_id,
)


class TestTimeframe:
    """Tests for Timeframe enum."""

    def test_timeframe_values(self):
        assert Timeframe.MINUTE.value == "minute"
        assert Timeframe.HOUR.value == "hour"
        assert Timeframe.DAY.value == "day"


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_success_rate_all_success(self):
        result = DownloadResult(
            total_symbols=10,
            succeeded=10,
            failed=0,
            total_bars=1000,
            elapsed_seconds=60.0,
        )
        assert result.success_rate == 100.0

    def test_success_rate_partial(self):
        result = DownloadResult(
            total_symbols=10,
            succeeded=7,
            failed=3,
            total_bars=700,
            elapsed_seconds=60.0,
        )
        assert result.success_rate == 70.0

    def test_success_rate_zero_symbols(self):
        result = DownloadResult(
            total_symbols=0,
            succeeded=0,
            failed=0,
            total_bars=0,
            elapsed_seconds=0.0,
        )
        assert result.success_rate == 0.0


class TestFormatTime:
    """Tests for _format_time utility function."""

    def test_seconds_only(self):
        assert _format_time(45) == "45s"

    def test_minutes_and_seconds(self):
        assert _format_time(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert _format_time(3725) == "1h 2m 5s"

    def test_zero(self):
        assert _format_time(0) == "0s"


class TestAlpacaDownloaderInit:
    """Tests for AlpacaDownloader initialization."""

    def test_default_values(self):
        downloader = AlpacaDownloader()
        assert downloader.start_date == "2017-01-01"
        assert downloader.end_date == datetime.now().strftime('%Y-%m-%d')
        assert downloader.num_threads == 6

    def test_custom_values(self):
        downloader = AlpacaDownloader(
            start_date="2020-01-01",
            end_date="2024-12-31",
            num_threads=4,
        )
        assert downloader.start_date == "2020-01-01"
        assert downloader.end_date == "2024-12-31"
        assert downloader.num_threads == 4


class TestAlpacaDownloaderOutputDir:
    """Tests for output directory handling."""

    def test_get_output_dir_minute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.MINUTE)
            assert output_dir == Path(tmpdir) / "equities_1min"

    def test_get_output_dir_hour(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.HOUR)
            assert output_dir == Path(tmpdir) / "equities_1hour"

    def test_get_output_dir_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.DAY)
            assert output_dir == Path(tmpdir) / "equities_1day"


class TestGetExistingSymbols:
    """Tests for get_existing_symbols method."""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            assert existing == set()

    def test_finds_existing_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake existing data
            base = Path(tmpdir) / "equities_1min"
            for symbol in ["AAPL", "MSFT"]:
                data_dir = base / f"symbol={symbol}" / "year=2024" / "month=1"
                data_dir.mkdir(parents=True)
                (data_dir / "data.parquet").write_text("fake")

            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            assert existing == {"AAPL", "MSFT"}

    def test_ignores_empty_symbol_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create symbol dir without parquet files
            base = Path(tmpdir) / "equities_1min"
            (base / "symbol=AAPL").mkdir(parents=True)

            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            assert existing == set()


class TestDownloadSymbolsSkipExisting:
    """Tests for skip_existing behavior."""

    def test_skip_all_existing_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake existing data for all symbols
            base = Path(tmpdir) / "equities_1min"
            for symbol in ["AAPL", "MSFT"]:
                data_dir = base / f"symbol={symbol}" / "year=2024" / "month=1"
                data_dir.mkdir(parents=True)
                (data_dir / "data.parquet").write_text("fake")

            downloader = AlpacaDownloader(output_dir=Path(tmpdir))
            result = downloader.download_symbols(
                symbols=["AAPL", "MSFT"],
                timeframe=Timeframe.MINUTE,
                skip_existing=True,
            )

            # All skipped, so 0 downloaded
            assert result.total_symbols == 0
            assert result.succeeded == 0
            assert result.failed == 0


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_parse_symbols_arg(self):
        from scripts.download_symbols import parse_symbols_arg

        result = parse_symbols_arg("AAPL, msft, GOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_symbols_arg_strips_whitespace(self):
        from scripts.download_symbols import parse_symbols_arg

        result = parse_symbols_arg("  AAPL  ,  MSFT  ")
        assert result == ["AAPL", "MSFT"]

    def test_parse_symbols_arg_filters_empty(self):
        from scripts.download_symbols import parse_symbols_arg

        result = parse_symbols_arg("AAPL,,MSFT,")
        assert result == ["AAPL", "MSFT"]

    def test_load_symbols_from_csv(self):
        from scripts.download_symbols import load_symbols_from_csv

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Symbol,Name\n")
            f.write("AAPL,Apple\n")
            f.write("MSFT,Microsoft\n")
            f.flush()

            result = load_symbols_from_csv(Path(f.name))
            assert result == ["AAPL", "MSFT"]

    def test_load_symbols_from_csv_filters_nan(self):
        from scripts.download_symbols import load_symbols_from_csv

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Symbol,Name\n")
            f.write("AAPL,Apple\n")
            f.write(",Missing\n")  # Empty symbol
            f.write("MSFT,Microsoft\n")
            f.flush()

            result = load_symbols_from_csv(Path(f.name))
            assert result == ["AAPL", "MSFT"]

    def test_load_symbols_from_file(self):
        from scripts.download_symbols import load_symbols_from_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AAPL\n")
            f.write("MSFT\n")
            f.write("googl\n")  # Lowercase
            f.flush()

            result = load_symbols_from_file(Path(f.name))
            assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_load_symbols_from_file_ignores_comments(self):
        from scripts.download_symbols import load_symbols_from_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("AAPL\n")
            f.write("# Another comment\n")
            f.write("MSFT\n")
            f.flush()

            result = load_symbols_from_file(Path(f.name))
            assert result == ["AAPL", "MSFT"]

    def test_load_symbols_from_file_ignores_empty_lines(self):
        from scripts.download_symbols import load_symbols_from_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AAPL\n")
            f.write("\n")
            f.write("   \n")
            f.write("MSFT\n")
            f.flush()

            result = load_symbols_from_file(Path(f.name))
            assert result == ["AAPL", "MSFT"]

    def test_parse_timeframe_minute(self):
        from scripts.download_symbols import parse_timeframe

        assert parse_timeframe("minute") == Timeframe.MINUTE
        assert parse_timeframe("min") == Timeframe.MINUTE
        assert parse_timeframe("1min") == Timeframe.MINUTE

    def test_parse_timeframe_hour(self):
        from scripts.download_symbols import parse_timeframe

        assert parse_timeframe("hour") == Timeframe.HOUR
        assert parse_timeframe("hourly") == Timeframe.HOUR
        assert parse_timeframe("1hour") == Timeframe.HOUR

    def test_parse_timeframe_day(self):
        from scripts.download_symbols import parse_timeframe

        assert parse_timeframe("day") == Timeframe.DAY
        assert parse_timeframe("daily") == Timeframe.DAY
        assert parse_timeframe("1day") == Timeframe.DAY

    def test_parse_timeframe_case_insensitive(self):
        from scripts.download_symbols import parse_timeframe

        assert parse_timeframe("MINUTE") == Timeframe.MINUTE
        assert parse_timeframe("Hour") == Timeframe.HOUR
        assert parse_timeframe("DAY") == Timeframe.DAY

    def test_parse_timeframe_invalid(self):
        from scripts.download_symbols import parse_timeframe

        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe("invalid")


class TestCLIFileNotFound:
    """Tests for file not found errors."""

    def test_csv_not_found(self):
        from scripts.download_symbols import load_symbols_from_csv

        with pytest.raises(FileNotFoundError):
            load_symbols_from_csv(Path("/nonexistent/file.csv"))

    def test_file_not_found(self):
        from scripts.download_symbols import load_symbols_from_file

        with pytest.raises(FileNotFoundError):
            load_symbols_from_file(Path("/nonexistent/file.txt"))

    def test_csv_no_symbol_column(self):
        from scripts.download_symbols import load_symbols_from_csv

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Value\n")
            f.write("Apple,100\n")
            f.flush()

            with pytest.raises(ValueError, match="No symbol column"):
                load_symbols_from_csv(Path(f.name))
