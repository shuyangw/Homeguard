"""
Unit tests for scripts/download_symbols.py CLI helper functions.

Tests the symbol loading and parsing utilities used by the download script.
"""

import tempfile
from pathlib import Path

import pytest


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
