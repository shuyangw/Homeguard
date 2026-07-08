import polars as pl
from src.data.acquisition.plugins.cftc_cot import parse_legacy_csv, COT_LEGACY_INSTRUMENTS

# Minimal Legacy futures-only CSV (real column names; two contracts, one week)
_FIXTURE = (
    b"Market_and_Exchange_Names,Report_Date_as_YYYY-MM-DD,CFTC_Contract_Market_Code,"
    b"Commercial_Positions-Long_All,Commercial_Positions-Short_All,"
    b"Noncommercial_Positions-Long_All,Noncommercial_Positions-Short_All\n"
    b"E-MINI S&P 500,2015-01-06,13874A,1000,1500,800,300\n"
    b"CRUDE OIL,2015-01-06,067651,2000,1000,500,900\n"
)


def test_parse_legacy_filters_by_code_and_maps_columns():
    df = parse_legacy_csv(_FIXTURE, "13874A")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["commercial_long"] == 1000 and row["commercial_short"] == 1500
    assert row["noncommercial_long"] == 800 and row["noncommercial_short"] == 300
    assert row["report_date"] == "2015-01-06"


def test_broad_universe_has_core_roots():
    for r in ("ES", "CL", "GC", "ZN", "ZC"):
        assert r in COT_LEGACY_INSTRUMENTS
