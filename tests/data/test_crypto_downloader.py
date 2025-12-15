"""
Unit tests for src/data/crypto_downloader.py

Tests the CryptoDownloader class and symbol normalization utilities.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.crypto_downloader import (
    CryptoDownloader,
    DownloadResult,
    Timeframe,
    _format_time,
    _get_thread_id,
    normalize_symbol,
    denormalize_symbol,
)


class TestSymbolNormalization:
    """Tests for symbol normalization functions."""

    def test_normalize_symbol_btc_usd(self):
        assert normalize_symbol("BTC/USD") == "BTC_USD"

    def test_normalize_symbol_eth_btc(self):
        assert normalize_symbol("ETH/BTC") == "ETH_BTC"

    def test_denormalize_symbol_btc_usd(self):
        assert denormalize_symbol("BTC_USD") == "BTC/USD"

    def test_denormalize_symbol_eth_btc(self):
        assert denormalize_symbol("ETH_BTC") == "ETH/BTC"

    def test_normalize_roundtrip(self):
        original = "DOGE/USD"
        normalized = normalize_symbol(original)
        denormalized = denormalize_symbol(normalized)
        assert denormalized == original

    def test_normalize_all_pairs(self):
        """Test normalization for all supported pairs."""
        pairs = [
            "AAVE/USD", "AVAX/USD", "BAT/USD", "BCH/USD", "BTC/USD",
            "CRV/USD", "DOGE/USD", "DOT/USD", "ETH/USD", "GRT/USD",
            "LINK/USD", "LTC/USD", "MKR/USD", "SHIB/USD", "SUSHI/USD",
            "UNI/USD", "XTZ/USD", "YFI/USD",
        ]
        for pair in pairs:
            normalized = normalize_symbol(pair)
            assert "/" not in normalized
            assert "_" in normalized
            assert denormalize_symbol(normalized) == pair


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


class TestCryptoDownloaderInit:
    """Tests for CryptoDownloader initialization."""

    def test_default_values(self):
        downloader = CryptoDownloader()
        assert downloader.start_date == "2020-01-01"
        assert downloader.end_date == datetime.now().strftime('%Y-%m-%d')
        assert downloader.num_threads == 6

    def test_custom_values(self):
        downloader = CryptoDownloader(
            start_date="2022-01-01",
            end_date="2024-12-31",
            num_threads=4,
        )
        assert downloader.start_date == "2022-01-01"
        assert downloader.end_date == "2024-12-31"
        assert downloader.num_threads == 4


class TestCryptoDownloaderOutputDir:
    """Tests for output directory handling."""

    def test_get_output_dir_minute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.MINUTE)
            assert output_dir == Path(tmpdir) / "crypto_1min"

    def test_get_output_dir_hour(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.HOUR)
            assert output_dir == Path(tmpdir) / "crypto_1hour"

    def test_get_output_dir_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            output_dir = downloader._get_output_dir(Timeframe.DAY)
            assert output_dir == Path(tmpdir) / "crypto_1day"


class TestGetExistingSymbols:
    """Tests for get_existing_symbols method."""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            assert existing == set()

    def test_finds_existing_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake existing data with normalized symbols
            base = Path(tmpdir) / "crypto_1min"
            for fs_symbol in ["BTC_USD", "ETH_USD"]:
                data_dir = base / f"symbol={fs_symbol}" / "year=2024" / "month=1"
                data_dir.mkdir(parents=True)
                (data_dir / "data.parquet").write_text("fake")

            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            # Should return API format symbols
            assert existing == {"BTC/USD", "ETH/USD"}

    def test_ignores_empty_symbol_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create symbol dir without parquet files
            base = Path(tmpdir) / "crypto_1min"
            (base / "symbol=BTC_USD").mkdir(parents=True)

            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            existing = downloader.get_existing_symbols(Timeframe.MINUTE)
            assert existing == set()


class TestDownloadSymbolsSkipExisting:
    """Tests for skip_existing behavior."""

    def test_skip_all_existing_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake existing data for all symbols
            base = Path(tmpdir) / "crypto_1min"
            for fs_symbol in ["BTC_USD", "ETH_USD"]:
                data_dir = base / f"symbol={fs_symbol}" / "year=2024" / "month=1"
                data_dir.mkdir(parents=True)
                (data_dir / "data.parquet").write_text("fake")

            downloader = CryptoDownloader(output_dir=Path(tmpdir))
            result = downloader.download_symbols(
                symbols=["BTC/USD", "ETH/USD"],
                timeframe=Timeframe.MINUTE,
                skip_existing=True,
            )

            # All skipped, so 0 downloaded
            assert result.total_symbols == 0
            assert result.succeeded == 0
            assert result.failed == 0


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_parse_timeframe_minute(self):
        from scripts.download_crypto import parse_timeframe

        assert parse_timeframe("minute") == Timeframe.MINUTE
        assert parse_timeframe("min") == Timeframe.MINUTE
        assert parse_timeframe("1min") == Timeframe.MINUTE
        assert parse_timeframe("1m") == Timeframe.MINUTE

    def test_parse_timeframe_hour(self):
        from scripts.download_crypto import parse_timeframe

        assert parse_timeframe("hour") == Timeframe.HOUR
        assert parse_timeframe("hourly") == Timeframe.HOUR
        assert parse_timeframe("1hour") == Timeframe.HOUR
        assert parse_timeframe("1h") == Timeframe.HOUR

    def test_parse_timeframe_day(self):
        from scripts.download_crypto import parse_timeframe

        assert parse_timeframe("day") == Timeframe.DAY
        assert parse_timeframe("daily") == Timeframe.DAY
        assert parse_timeframe("1day") == Timeframe.DAY
        assert parse_timeframe("1d") == Timeframe.DAY

    def test_parse_timeframe_case_insensitive(self):
        from scripts.download_crypto import parse_timeframe

        assert parse_timeframe("MINUTE") == Timeframe.MINUTE
        assert parse_timeframe("Hour") == Timeframe.HOUR
        assert parse_timeframe("DAY") == Timeframe.DAY

    def test_parse_timeframe_invalid(self):
        from scripts.download_crypto import parse_timeframe

        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe("invalid")


class TestDefaultCryptoPairs:
    """Tests for default crypto pairs in CLI script."""

    def test_default_pairs_count(self):
        from scripts.download_crypto import DEFAULT_CRYPTO_PAIRS

        # Should have 18 pairs (no stablecoins)
        assert len(DEFAULT_CRYPTO_PAIRS) == 18

    def test_no_stablecoins(self):
        from scripts.download_crypto import DEFAULT_CRYPTO_PAIRS

        # USDC and USDT should not be in the list
        for pair in DEFAULT_CRYPTO_PAIRS:
            assert "USDC/USD" not in pair or pair != "USDC/USD"
            assert "USDT/USD" not in pair or pair != "USDT/USD"

    def test_all_usd_pairs(self):
        from scripts.download_crypto import DEFAULT_CRYPTO_PAIRS

        # All pairs should end with /USD
        for pair in DEFAULT_CRYPTO_PAIRS:
            assert pair.endswith("/USD"), f"{pair} should end with /USD"

    def test_expected_pairs_present(self):
        from scripts.download_crypto import DEFAULT_CRYPTO_PAIRS

        expected = ["BTC/USD", "ETH/USD", "DOGE/USD", "LINK/USD"]
        for pair in expected:
            assert pair in DEFAULT_CRYPTO_PAIRS, f"{pair} should be in default pairs"
