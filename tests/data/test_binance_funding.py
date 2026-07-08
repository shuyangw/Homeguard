import polars as pl
from src.data.acquisition.plugins.binance_funding import parse_funding, daily_annualized

# Binance /fapi/v1/fundingRate response shape: fundingTime ms, fundingRate str
_ROWS = [
    {"symbol": "BTCUSDT", "fundingTime": 1704067200000, "fundingRate": "0.0001"},
    {"symbol": "BTCUSDT", "fundingTime": 1704096000000, "fundingRate": "0.0001"},
    {"symbol": "BTCUSDT", "fundingTime": 1704124800000, "fundingRate": "0.0001"},  # 3 events same UTC day
]

def test_parse_funding_types():
    df = parse_funding(_ROWS)
    assert df.schema["funding_rate"] == pl.Float64
    assert df.height == 3

def test_daily_annualized_sums_three_events_times_365():
    df = daily_annualized(parse_funding(_ROWS))
    # 3 events/day * 0.0001 = 0.0003/day; annualized = 0.0003 * 365
    assert df.height == 1
    assert abs(df.row(0, named=True)["funding_annualized"] - 0.0003 * 365) < 1e-9
