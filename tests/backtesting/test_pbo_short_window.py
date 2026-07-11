import numpy as np
from src.backtesting.walkforward_common import _compute_pbo


def _win(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.01, n)


def test_short_trailing_window_no_longer_nans():
    # 11 long windows (~250 rows) + one short trailing window (12 rows < s=16).
    windows = [_win(250, i) for i in range(11)] + [_win(12, 99)]
    val = _compute_pbo(windows)
    assert not np.isnan(val), "a single short window must not NaN the whole PBO"
    assert 0.0 <= val <= 1.0


def test_all_long_windows_unchanged():
    # With no short window, dropping sub-s windows is a no-op -> same result as
    # the plain min-len path would give.
    windows = [_win(250, i) for i in range(11)]
    val = _compute_pbo(windows)
    assert not np.isnan(val) and 0.0 <= val <= 1.0


def test_insufficient_after_drop_is_nan():
    # Only one window >= s -> honestly NaN (need >= 2 configs).
    windows = [_win(250, 1)] + [_win(10, 2), _win(11, 3)]
    assert np.isnan(_compute_pbo(windows))


def test_window_between_s_and_2s_dropped():
    # a 30-row window (>= s=16 but < 2s=32) must NOT survive -> the remaining long
    # windows give a real PBO, not NaN.
    windows = [_win(250, i) for i in range(11)] + [_win(30, 77)]
    val = _compute_pbo(windows)
    assert not np.isnan(val) and 0.0 <= val <= 1.0
