"""Canonical data schemas and validation utilities."""


class SchemaValidationError(Exception):
    """Raised when DataFrame columns don't match expected schema."""
    pass


CANONICAL_OHLCV_SCHEMA = [
    "timestamp", "open", "high", "low", "close",
    "volume", "trade_count", "vwap",
]

FUTURES_TRADES_SCHEMA = [
    "timestamp", "price", "size",
]


def validate_schema(df, expected_columns: list[str]) -> None:
    """Validate that a DataFrame contains all expected columns.

    Args:
        df: DataFrame to validate
        expected_columns: List of required column names

    Raises:
        SchemaValidationError: If required columns are missing
    """
    actual = set(df.columns)
    expected = set(expected_columns)
    missing = expected - actual
    if missing:
        raise SchemaValidationError(
            f"Missing columns: {sorted(missing)}. "
            f"Expected: {expected_columns}, Got: {sorted(df.columns)}"
        )
