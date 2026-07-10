from datetime import date, time
import pandas as pd
import numpy as np
from src.backtesting.session.session_bars import (
    extract_from_minute_frame, drop_all_nan_dates, SESSION_TIMES)


def _minute_frame(rows):
    # rows: list of (utc_str, close)
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({"close": [r[1] for r in rows]}, index=idx).sort_index()


def test_picks_first_bar_at_or_after_et_time():
    # 2015-06-01: 16:00 ET = 20:00 UTC. Provide a bar at 20:00 and 20:01.
    mf = _minute_frame([
        ("2015-06-01 19:59", 100.0), ("2015-06-01 20:00", 101.0), ("2015-06-01 20:01", 102.0),
    ])
    out = extract_from_minute_frame(mf, {"et_1600": time(16, 0)})
    assert out.loc[date(2015, 6, 1), "et_1600"] == 101.0  # the 20:00 bar (at/after 16:00 ET)


def test_missing_time_is_nan():
    # no bar within 15 min of 16:00 ET -> NaN
    mf = _minute_frame([("2015-06-01 12:00", 100.0)])
    out = extract_from_minute_frame(mf, {"et_1600": time(16, 0)})
    assert np.isnan(out.loc[date(2015, 6, 1), "et_1600"])


def test_session_times_cover_the_five_boundaries():
    assert set(SESSION_TIMES) == {"et_0200", "et_0500", "et_0930", "et_1400", "et_1600"}


def test_drop_all_nan_dates_removes_only_all_nan_rows():
    # A normal (weekday) row with some closes and an all-NaN (Sunday) row.
    cols = list(SESSION_TIMES)
    normal = {c: 100.0 + i for i, c in enumerate(cols)}
    sunday = {c: np.nan for c in cols}
    frame = pd.DataFrame(
        [normal, sunday],
        index=pd.Index([date(2015, 6, 5), date(2015, 6, 7)], name="date"),
    )
    out = drop_all_nan_dates(frame)
    assert list(out.index) == [date(2015, 6, 5)]  # only the all-NaN Sunday dropped
    assert out.loc[date(2015, 6, 5), cols[0]] == 100.0
