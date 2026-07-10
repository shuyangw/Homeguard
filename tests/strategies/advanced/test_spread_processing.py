from datetime import date

import numpy as np

from src.strategies.advanced.spread_processing_strategy import crack_spread, crush_spread


def test_crack_spread_per_barrel_positive_typical():
    s = crack_spread("RB", date(2018, 1, 1), date(2023, 12, 31))
    # gasoline crack is usually positive (refining margin), a few dollars to ~40
    med = float(np.nanmedian(s.signal.values))
    assert 0.0 < med < 80.0


def test_crush_spread_nonempty():
    s = crush_spread(date(2018, 1, 1), date(2023, 12, 31))
    assert s.signal.notna().sum() > 500
    assert s.unit_return.notna().sum() > 500
