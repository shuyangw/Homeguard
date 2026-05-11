"""Build tiny synthetic parquet fixtures for unit tests."""
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def build_minimal_futures_store(root: Path) -> None:
    """Build a minimal valid futures data layout for testing."""
    # futures_1min: ES one month of data
    es_dir = root / "futures_1min" / "symbol=ES" / "year=2024" / "month=6"
    es_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 6, 17, 13, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 17, 13, 1, tzinfo=timezone.utc),
        ],
        "open": [5400.0, 5400.5],
        "high": [5401.0, 5401.0],
        "low": [5399.5, 5400.0],
        "close": [5400.5, 5400.75],
        "volume": [100, 120],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col("volume").cast(pl.UInt64),
    )
    df.write_parquet(es_dir / "data.parquet")

    # Other required sections (minimal placeholder files)
    for section in [
        "futures_1min_oi_roll",
        "futures_per_contract_1min",
        "futures_options_1min",
        "futures_definitions",
        "futures_statistics",
    ]:
        sec_dir = root / section
        if section == "futures_1min_oi_roll":
            sec_dir = sec_dir / "symbol=GC" / "year=2024" / "month=6"
        else:
            sec_dir = sec_dir / "year=2024" / "month=6"
        sec_dir.mkdir(parents=True, exist_ok=True)
        # minimal valid parquet with just a timestamp
        small = pl.DataFrame({
            "timestamp": [datetime(2024, 6, 17, tzinfo=timezone.utc)],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        small.write_parquet(sec_dir / "data.parquet")
