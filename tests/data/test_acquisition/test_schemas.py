"""Tests for canonical schemas and validation."""

import pandas as pd
import pytest

from src.data.acquisition.schemas import (
    CANONICAL_OHLCV_SCHEMA,
    FUTURES_TRADES_SCHEMA,
    validate_schema,
    SchemaValidationError,
)


class TestCanonicalSchemas:
    def test_ohlcv_schema_has_eight_columns(self):
        assert len(CANONICAL_OHLCV_SCHEMA) == 8

    def test_ohlcv_schema_column_names(self):
        expected = [
            "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap",
        ]
        assert CANONICAL_OHLCV_SCHEMA == expected

    def test_trades_schema_column_names(self):
        expected = ["timestamp", "price", "size"]
        assert FUTURES_TRADES_SCHEMA == expected


class TestValidateSchema:
    def test_valid_ohlcv_passes(self):
        df = pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)
        validate_schema(df, CANONICAL_OHLCV_SCHEMA)

    def test_missing_column_raises(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high"])
        with pytest.raises(SchemaValidationError, match="Missing columns"):
            validate_schema(df, CANONICAL_OHLCV_SCHEMA)

    def test_extra_columns_are_allowed(self):
        cols = CANONICAL_OHLCV_SCHEMA + ["extra_col"]
        df = pd.DataFrame(columns=cols)
        validate_schema(df, CANONICAL_OHLCV_SCHEMA)

    def test_empty_dataframe_valid_schema_passes(self):
        df = pd.DataFrame(columns=CANONICAL_OHLCV_SCHEMA)
        validate_schema(df, CANONICAL_OHLCV_SCHEMA)
