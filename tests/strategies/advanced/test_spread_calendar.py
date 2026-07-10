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
    # unit_return is the (scaled) first difference of the level, except on
    # roll days where it is masked to 0.0 (contract-swap jump, not P&L).
    reconstructed = level.diff()
    ur = unit_return.dropna().iloc[:50]
    rec = reconstructed.reindex(ur.index)
    non_roll = ur != 0.0
    assert np.sign(ur[non_roll].values).tolist() == \
           np.sign(rec[non_roll].values).tolist()


def test_calendar_roll_day_return_is_masked():
    # 2021-01-15 is a known CL front/second contract roll date (front_symbol
    # changes from CLG1 to CLH1); the F2-F1 level jump on that date is a
    # contract-swap artifact, not realized P&L, so unit_return must be 0.0.
    _, unit_return = calendar_signal("CL", date(2021, 1, 1), date(2022, 12, 31))
    roll_date = pd.Timestamp("2021-01-15")
    assert roll_date in unit_return.index
    assert unit_return.loc[roll_date] == 0.0
