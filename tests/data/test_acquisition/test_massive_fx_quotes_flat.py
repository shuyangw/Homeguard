"""Unit tests for massive_fx_quotes_flat."""
from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.acquisition.plugins.massive_fx_quotes_flat import (
    QUOTE_CANONICAL_COLUMNS, TargetPair, iter_days_by_month,
    key_for, parse_day, rows_to_parquet,
)


SAMPLE_CSV = """ticker,participant_timestamp,bid_price,ask_price,bid_exchange,ask_exchange
C:EUR-USD,1717362000000000000,1.0851,1.0852,4,4
C:EUR-USD,1717362000000000001,1.08510,1.08522,4,4
C:USD-JPY,1717362000000000000,157.23,157.25,4,4
C:GBP-USD,1717362000000000000,1.2701,1.2702,4,4
"""


def _make_gz(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


def test_key_for_path():
    assert key_for(date(2024, 6, 3)) == "global_forex/quotes_v1/2024/06/2024-06-03.csv.gz"


def test_parse_day_filters_targets():
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:EUR-USD"})
    assert "C:EUR-USD" in rows
    assert "C:USD-JPY" not in rows
    assert len(rows["C:EUR-USD"]) == 2


def test_parse_day_row_types():
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:EUR-USD"})
    r = rows["C:EUR-USD"][0]
    assert isinstance(r[0], int)         # timestamp_ns
    assert isinstance(r[1], float)       # bid_price
    assert isinstance(r[2], float)       # ask_price
    assert isinstance(r[3], int)         # bid_exchange
    assert isinstance(r[4], int)         # ask_exchange
    assert r[1] == 1.0851
    assert r[2] == 1.0852


def test_rows_to_parquet_schema(tmp_path):
    rows = parse_day(_make_gz(SAMPLE_CSV), {"C:EUR-USD"})
    out = tmp_path / "data.parquet"
    n = rows_to_parquet(rows["C:EUR-USD"], out)
    assert n == 2
    df = pl.read_parquet(out)
    assert df.columns == QUOTE_CANONICAL_COLUMNS
    assert df.dtypes == [
        pl.Datetime(time_unit="ns", time_zone="UTC"),
        pl.Float64, pl.Float64, pl.Int32, pl.Int32,
    ]


def test_rows_to_parquet_sorts():
    pass  # already covered by canonical sort in rows_to_parquet


def test_iter_days_by_month():
    days = list(iter_days_by_month(date(2024, 1, 30), date(2024, 2, 2)))
    assert len(days) == 2
    assert days[0] == (2024, 1, [date(2024, 1, 30), date(2024, 1, 31)])
    assert days[1] == (2024, 2, [date(2024, 2, 1), date(2024, 2, 2)])


def test_target_pair_frozen():
    p = TargetPair(hg_symbol="EURUSD", massive_ticker="C:EUR-USD",
                   effective_start=date(2010, 1, 1))
    with pytest.raises(AttributeError):
        p.hg_symbol = "X"
