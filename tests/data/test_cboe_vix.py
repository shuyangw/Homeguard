import polars as pl
from datetime import date

from src.data.acquisition.plugins.cboe_vix import build_front_second


def test_front_second_picks_two_nearest_unexpired():
    # three contracts; on 2015-01-05 the nearest unexpired is exp 2015-01-21 (VX1),
    # then 2015-02-18 (VX2)
    per_contract = pl.DataFrame({
        "date":   [date(2015, 1, 5)] * 3,
        "expiry": [date(2015, 1, 21), date(2015, 2, 18), date(2015, 3, 18)],
        "settle": [18.0, 19.0, 20.0],
    })
    out = build_front_second(per_contract)
    row = out.filter(pl.col("date") == date(2015, 1, 5)).row(0, named=True)
    assert row["vx1_settle"] == 18.0 and row["vx2_settle"] == 19.0
    assert row["vx1_dte"] == (date(2015, 1, 21) - date(2015, 1, 5)).days


def test_expired_contract_excluded_from_front():
    # on 2015-01-22 the 2015-01-21 contract has expired -> VX1 becomes 2015-02-18
    per_contract = pl.DataFrame({
        "date":   [date(2015, 1, 22)] * 3,
        "expiry": [date(2015, 1, 21), date(2015, 2, 18), date(2015, 3, 18)],
        "settle": [17.0, 19.5, 20.5],
    })
    out = build_front_second(per_contract)
    row = out.row(0, named=True)
    assert row["vx1_settle"] == 19.5 and row["vx2_settle"] == 20.5
