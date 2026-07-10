from datetime import date
import numpy as np
import pandas as pd
from src.strategies.advanced.spread_intermarket_strategy import (
    PAIRS, intermarket_spread, intermarket_return_stream)


def test_pairs_registered():
    assert PAIRS["NQ_ES"] == ("NQ", "ES")
    assert PAIRS["RTY_ES"] == ("RTY", "ES")


def test_intermarket_signal_is_log_ratio():
    s = intermarket_spread("NQ", "ES", date(2015, 1, 1), date(2024, 12, 31))
    assert s.signal.notna().sum() > 1000
    # NQ outperformed ES over 2015-2024 -> log(NQ/ES) trends up net
    assert s.signal.dropna().iloc[-1] > s.signal.dropna().iloc[0]


def test_intermarket_return_stream_nonempty():
    r = intermarket_return_stream("NQ_ES", date(2015, 1, 1), date(2024, 12, 31))
    assert len(r) > 500 and r.std() > 0
