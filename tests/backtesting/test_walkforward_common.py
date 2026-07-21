import numpy as np, pandas as pd
from src.backtesting.walkforward_common import _stitch_oos_dedup


def test_stitch_dedup_drops_shared_boundary_day_keep_first():
    d = pd.to_datetime
    w1 = pd.Series([0.1, 0.2, 0.3], index=d(["2021-12-30", "2021-12-31", "2022-01-03"]))
    # 2022-01-03 is shared: it's w1's last OOS day AND w2's first
    w2 = pd.Series([0.9, 0.4], index=d(["2022-01-03", "2022-01-04"]))
    out = _stitch_oos_dedup([w1, w2])
    assert len(out) == 4  # 5 rows minus the one shared day
    # keep-first: the shared day keeps w1's 0.3, not w2's 0.9
    assert list(out) == [0.1, 0.2, 0.3, 0.4]


def test_stitch_dedup_no_overlap_keeps_all_sorted():
    d = pd.to_datetime
    w1 = pd.Series([0.1, 0.2], index=d(["2021-06-01", "2021-12-31"]))
    w2 = pd.Series([0.3, 0.4], index=d(["2022-01-03", "2022-06-01"]))
    out = _stitch_oos_dedup([w1, w2])
    assert list(out) == [0.1, 0.2, 0.3, 0.4]


def test_stitch_dedup_empty():
    assert list(_stitch_oos_dedup([])) == []
