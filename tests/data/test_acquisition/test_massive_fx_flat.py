"""Unit tests for src/data/acquisition/plugins/massive_fx_flat.py."""
from __future__ import annotations

import gzip
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.acquisition.plugins.massive_fx_flat import (
    TargetPair,
    iter_days_by_month,
    key_for,
    parse_day,
    rows_to_parquet,
)
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


# Sample CSV exactly matching Massive's flat-file schema.
SAMPLE_CSV = """ticker,volume,open,close,high,low,window_start,transactions
C:USD-NOK,5,10.46877,10.46555,10.47877,10.46518,1717362000000000000,5
C:USD-NOK,1,10.47855,10.47855,10.47855,10.47855,1717362120000000000,1
C:EUR-USD,42,1.0852,1.0851,1.0853,1.085,1717372800000000000,42
C:USD-SEK,8,10.55,10.56,10.56,10.55,1717362000000000000,8
"""


def _make_gz(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


def test_key_for_pads_month():
    assert key_for(date(2024, 6, 3)) == "global_forex/minute_aggs_v1/2024/06/2024-06-03.csv.gz"
    assert key_for(date(2010, 1, 1)) == "global_forex/minute_aggs_v1/2010/01/2010-01-01.csv.gz"


def test_parse_day_filters_to_target_tickers():
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:USD-NOK"})
    assert "C:USD-NOK" in rows
    assert "C:EUR-USD" not in rows
    assert len(rows["C:USD-NOK"]) == 2


def test_parse_day_row_schema_and_types():
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:USD-NOK"})
    r = rows["C:USD-NOK"][0]
    # Field ordering and types match what rows_to_parquet expects
    assert isinstance(r["timestamp"], int)
    assert isinstance(r["open"], float)
    assert isinstance(r["high"], float)
    assert isinstance(r["low"], float)
    assert isinstance(r["close"], float)
    assert isinstance(r["volume"], int)
    assert isinstance(r["trade_count"], int)
    assert isinstance(r["vwap"], float)
    # vwap is set to close (documented approximation; FX has no real vwap)
    assert r["vwap"] == r["close"]
    # window_start nanoseconds preserved
    assert r["timestamp"] == 1717362000000000000


def test_parse_day_skips_malformed_lines():
    bad_csv = SAMPLE_CSV + "C:USD-NOK,BAD,oops,malformed,line\n"
    rows = parse_day(_make_gz(bad_csv), {"C:USD-NOK"})
    # Original 2 valid rows preserved, bad row dropped
    assert len(rows["C:USD-NOK"]) == 2


def test_parse_day_returns_empty_for_no_matches():
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:NONEXISTENT"})
    assert rows == {}


def test_rows_to_parquet_matches_canonical_schema(tmp_path: Path):
    """Generated parquet must have EXACTLY the canonical 8-col schema with the
    correct dtypes the existing 50 FX pairs use."""
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:USD-NOK"})
    out = tmp_path / "data.parquet"
    n = rows_to_parquet(rows["C:USD-NOK"], out)
    assert n == 2

    df = pl.read_parquet(out)
    assert df.columns == CANONICAL_OHLCV_SCHEMA
    # Exact dtype tuple match
    expected_dtypes = [
        pl.Datetime(time_unit="ns", time_zone="UTC"),
        pl.Float64, pl.Float64, pl.Float64, pl.Float64,
        pl.Int64, pl.Int64, pl.Float64,
    ]
    assert df.dtypes == expected_dtypes


def test_rows_to_parquet_sorts_by_timestamp(tmp_path: Path):
    """Rows arriving out-of-order are sorted on write."""
    rows = [
        {"timestamp": 1717362120000000000, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "volume": 1, "trade_count": 1, "vwap": 1.0},
        {"timestamp": 1717362000000000000, "open": 2.0, "high": 2.0, "low": 2.0,
         "close": 2.0, "volume": 2, "trade_count": 2, "vwap": 2.0},
    ]
    out = tmp_path / "data.parquet"
    rows_to_parquet(rows, out)
    df = pl.read_parquet(out)
    assert df.height == 2
    # Smaller timestamp must come first after sort
    assert df["open"][0] == 2.0
    assert df["open"][1] == 1.0


def test_rows_to_parquet_deduplicates_timestamps(tmp_path: Path):
    """Duplicate timestamps (shouldn't happen, but defensive) are dropped."""
    rows = [
        {"timestamp": 1717362000000000000, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "volume": 1, "trade_count": 1, "vwap": 1.0},
        {"timestamp": 1717362000000000000, "open": 2.0, "high": 2.0, "low": 2.0,
         "close": 2.0, "volume": 2, "trade_count": 2, "vwap": 2.0},
    ]
    out = tmp_path / "data.parquet"
    rows_to_parquet(rows, out)
    df = pl.read_parquet(out)
    assert df.height == 1
    # keep="last" -> second row wins
    assert df["open"][0] == 2.0


def test_rows_to_parquet_empty_input_writes_nothing(tmp_path: Path):
    out = tmp_path / "data.parquet"
    n = rows_to_parquet([], out)
    assert n == 0
    assert not out.exists()


def test_iter_days_by_month_groups_correctly():
    days = list(iter_days_by_month(date(2024, 1, 30), date(2024, 2, 2)))
    assert len(days) == 2
    y1, m1, ds1 = days[0]
    y2, m2, ds2 = days[1]
    assert (y1, m1) == (2024, 1)
    assert ds1 == [date(2024, 1, 30), date(2024, 1, 31)]
    assert (y2, m2) == (2024, 2)
    assert ds2 == [date(2024, 2, 1), date(2024, 2, 2)]


def test_iter_days_by_month_single_day():
    days = list(iter_days_by_month(date(2024, 6, 3), date(2024, 6, 3)))
    assert days == [(2024, 6, [date(2024, 6, 3)])]


def test_target_pair_dataclass_is_frozen():
    p = TargetPair(hg_symbol="USDNOK", massive_ticker="C:USD-NOK",
                   effective_start=date(2010, 1, 1))
    with pytest.raises(AttributeError):  # dataclass(frozen=True) raises FrozenInstanceError ~ AttributeError
        p.hg_symbol = "X"  # type: ignore[misc]
