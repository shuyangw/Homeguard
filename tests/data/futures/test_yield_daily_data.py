import polars as pl
from src.data.futures.paths import daily_raw_dir


def test_yield_roots_are_yield_levels():
    for root in ("2YY", "5YY", "10Y", "30Y"):
        fp = daily_raw_dir() / f"{root}.parquet"
        assert fp.exists(), f"missing daily_raw for {root}"
        closes = pl.read_parquet(fp)["close"].drop_nulls()
        # yields are single-to-low-double digit percent, never bond-price ~100
        assert closes.min() > -2.0 and closes.max() < 25.0

        if root == "5YY":
            # 5YY has sparse Databento coverage from 2023 (~440 rows, multi-month
            # gaps); it is data-degraded and 5YY-legged spreads (2s5s, 5s30s) will
            # be UNGRADEABLE by the walk-forward gate. Do not treat this floor as
            # a healthy-data assertion.
            assert closes.len() >= 400
        else:
            assert closes.len() > 500


def test_rty_present_and_pricelike():
    fp = daily_raw_dir() / "RTY.parquet"
    assert fp.exists()
    closes = pl.read_parquet(fp)["close"].drop_nulls()
    assert closes.max() > 500.0  # index points, not a yield
