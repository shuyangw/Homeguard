from datetime import date
import pandas as pd
from src.data.futures.front_next import front_next_history
from src.data.futures.paths import front_next_dir


def test_front_next_history_shape_and_ordering():
    df = front_next_history("CL", date(2022, 1, 1), date(2022, 6, 30))
    assert list(df.columns) == ["date", "front_symbol", "f1", "second_symbol", "f2", "months"]
    assert len(df) > 80  # ~ trading days in H1
    assert df["date"].is_monotonic_increasing
    # front and second are different contracts, positive prices
    assert (df["front_symbol"] != df["second_symbol"]).all()
    assert (df["f1"] > 0).all() and (df["f2"] > 0).all()
    assert (df["months"] != 0).all()


def test_front_next_history_persists_cache():
    front_next_history("CL", date(2022, 1, 1), date(2022, 3, 31))
    assert (front_next_dir() / "CL.parquet").exists()


def test_front_next_golden_date_cl():
    # On 2022-02-15 CL front was CLH2 (Mar) with CLJ2 (Apr) next; F1 < F2 or backwardated
    df = front_next_history("CL", date(2022, 2, 14), date(2022, 2, 16))
    row = df[df["date"] == pd.Timestamp(2022, 2, 15)]
    assert not row.empty
    assert row.iloc[0]["front_symbol"].startswith("CL")
    assert row.iloc[0]["months"] == 1  # front to next listed month
