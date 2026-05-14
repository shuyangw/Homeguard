"""Tests for CFTC COT plugin (Traders in Financial Futures report)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.data.acquisition.plugins.cftc_cot import (
    CFTCCOTPlugin,
    parse_tff_csv,
)

# Minimal valid TFF CSV (real headers; abbreviated rows)
SAMPLE_TFF_CSV = b"""Report_Date_as_YYYY-MM-DD,Market_and_Exchange_Names,CFTC_Contract_Market_Code,Dealer_Positions_Long_All,Dealer_Positions_Short_All
2024-01-02,EURO FX - CHICAGO MERCANTILE EXCHANGE,099741,12345,67890
"""


def test_parse_tff_csv_extracts_relevant_columns():
    df = parse_tff_csv(SAMPLE_TFF_CSV, contract_market_code="099741")
    assert df.height == 1
    assert df.columns == ["report_date", "dealer_long", "dealer_short"]
    assert df["report_date"][0] == "2024-01-02"
    assert df["dealer_long"][0] == 12345


def test_plugin_writes_canonical_layout(tmp_path: Path):
    plugin = CFTCCOTPlugin(storage_root=tmp_path)
    df = pl.DataFrame({
        "report_date": ["2024-01-02", "2024-01-09"],
        "dealer_long": [12345, 12500],
        "dealer_short": [67890, 67500],
    })
    plugin.write_instrument("6E", df)
    out = tmp_path / "cot" / "6E" / "weekly.parquet"
    assert out.exists()
    read = pl.read_parquet(out)
    assert read.height == 2
