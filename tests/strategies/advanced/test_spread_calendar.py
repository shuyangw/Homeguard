from datetime import date
import numpy as np
import pandas as pd
from src.strategies.advanced.spread_calendar_strategy import STORABLES, calendar_signal


def test_storables_exclude_metals():
    assert "CL" in STORABLES and "NG" in STORABLES
    assert "GC" not in STORABLES and "SI" not in STORABLES


def test_calendar_signal_is_f2_minus_f1():
    level, unit_return = calendar_signal("CL", date(2021, 1, 1), date(2023, 12, 31))
    assert level.notna().sum() > 300
    assert unit_return.notna().sum() > 300
    # unit_return is the (scaled) first difference of the level
    reconstructed = level.diff()
    assert np.sign(unit_return.dropna().values[:50]).tolist() == \
           np.sign(reconstructed.dropna().values[:50]).tolist()
