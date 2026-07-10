from datetime import date
import numpy as np
import pandas as pd
from src.strategies.advanced.spread_steepener_strategy import (
    SEGMENTS, steepener_spread, steepener_return_stream)


def test_segments_registered():
    assert SEGMENTS["2s10s"] == ("2YY", "10Y")
    assert set(SEGMENTS) == {"2s10s", "2s5s", "5s30s"}


def test_steepener_spread_signal_is_slope():
    s = steepener_spread("2YY", "10Y", date(2021, 1, 1), date(2023, 12, 31))
    # signal = 10Y - 2YY; during 2022 hiking the curve inverted -> goes negative
    assert s.signal.notna().sum() > 300
    assert s.signal.min() < 0.5  # inversion territory reached


def test_steepener_return_stream_nonempty():
    r = steepener_return_stream("2s10s", date(2021, 1, 1), date(2024, 12, 31))
    assert len(r) > 200
    assert np.isfinite(r.std()) and r.std() > 0
