from datetime import date
import numpy as np
from src.strategies.advanced.spread_ratio_strategy import ratio_spread


def test_ratio_signal_is_log_gc_si():
    s = ratio_spread(date(2015, 1, 1), date(2024, 12, 31))
    assert s.signal.notna().sum() > 1000
    # gold/silver ratio ~ 60-100 in modern era -> log ~ 4.1-4.6
    med = float(np.nanmedian(s.signal.values))
    assert 3.5 < med < 5.0
