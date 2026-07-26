"""Apparatus fixes found by the #35 Kalman diagnostic (2026-07-25).

1. `_compute_pbo` truncated every window to the SHORTEST one, so a single stub
   window discarded most of the OOS sample.
2. The spread simulator filled on the SAME bar the signal was computed from.
"""
import numpy as np
import pandas as pd

from src.backtesting.engine.fx_spread_simulator import _lag_book
from src.backtesting.walkforward_common import _compute_pbo


# ---------------------------------------------------------------- PBO stub guard

def _cols(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.01, n) for n in sizes]


def test_stub_window_no_longer_truncates_every_column():
    """12 full windows (260) + one 65-day stub: the stub must be dropped rather
    than truncating all 13 columns to 65 rows."""
    cols = _cols([260] * 12 + [65])
    val = _compute_pbo(cols)
    assert not np.isnan(val)
    # equivalent to computing on the 12 full windows alone
    assert val == _compute_pbo(_cols([260] * 12))


def test_uniform_windows_are_unchanged_by_the_guard():
    """Behaviour-neutral when there is no stub (no silent change to past runs
    whose windows were already uniform)."""
    cols = _cols([200, 200, 200, 200])
    assert _compute_pbo(cols) == _compute_pbo(cols)
    assert not np.isnan(_compute_pbo(cols))


def test_guard_falls_back_rather_than_returning_nan():
    """If dropping stubs would leave < 2 columns, keep the old behaviour so we
    still get a number instead of NaN."""
    cols = _cols([300, 40])          # 40 >= 2*s=32 so it is 'usable' but is a stub
    val = _compute_pbo(cols)
    assert not np.isnan(val)


def test_too_short_windows_still_dropped_and_nan_is_honest():
    assert np.isnan(_compute_pbo(_cols([10, 12])))     # both < 2*s
    assert np.isnan(_compute_pbo(_cols([300])))        # only one column


# ------------------------------------------------------------- execution lag

def test_lag_book_shifts_signals_forward():
    dates = list(pd.date_range("2020-01-01", periods=5).date)
    book = {dates[0]: ["A"], dates[2]: ["B"]}
    sigma = {dates[0]: {"s": 1}, dates[2]: {"s": 2}}
    nb, ns = _lag_book(book, sigma, dates, lag=1)
    assert nb == {dates[1]: ["A"], dates[3]: ["B"]}
    assert ns == {dates[1]: {"s": 1}, dates[3]: {"s": 2}}


def test_lag_book_drops_signals_that_never_become_tradeable():
    dates = list(pd.date_range("2020-01-01", periods=3).date)
    book = {dates[2]: ["last"]}          # nothing left to fill against
    nb, _ = _lag_book(book, {}, dates, lag=1)
    assert nb == {}


def test_lag_zero_is_identity():
    dates = list(pd.date_range("2020-01-01", periods=3).date)
    book = {dates[0]: ["A"]}
    nb, ns = _lag_book(book, {dates[0]: 1}, dates, lag=0)
    assert nb is book and ns == {dates[0]: 1}


def test_simulator_defaults_to_a_realistic_one_bar_lag():
    from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator
    sim = FxSpreadPortfolioSimulator(100_000.0, lambda *a: 0.0)
    assert sim.execution_lag == 1, "default must be the honest convention, not same-bar"


def _sim_inputs(n=120, seed=0):
    from src.backtesting.engine.spread_sizing import Spread
    rng = np.random.default_rng(seed)
    idx = list(pd.date_range("2020-01-01", periods=n, freq="B").date)
    close = pd.DataFrame(
        {"AUDUSD": np.exp(np.cumsum(rng.normal(0, .006, n))) * 0.7,
         "NZDUSD": np.exp(np.cumsum(rng.normal(0, .006, n))) * 0.65}, index=idx)
    q = pd.DataFrame(1.0, index=idx, columns=close.columns)
    book = {d: [Spread("AUDUSD", "NZDUSD", 0.8, 10.0)] for d in idx}
    sig = {d: {("AUDUSD", "NZDUSD"): 0.01} for d in idx}
    return close, book, sig, q


def _run(lag):
    from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator
    close, book, sig, q = _sim_inputs()
    sim = FxSpreadPortfolioSimulator(100_000.0, lambda *a: 0.0, execution_lag=lag)
    return sim.run_spreads(close, dict(book), dict(sig), q, vol_target=0.10)


def test_lagged_run_still_trades():
    """REGRESSION for the action-grid bug: shifting only the BOOK keys left the
    simulator gating on its UNSHIFTED rebalance grid, so every lagged signal
    landed on a non-rebalance day and was silently dropped -- execution_lag
    became 'trade nothing' instead of 'trade later'. A lagged run must still
    place approximately as many trades as the unlagged one."""
    r0, r1 = _run(0), _run(1)
    assert len(r1.trades) > 0, "lagged run must still trade"
    assert abs(len(r1.trades) - len(r0.trades)) <= 2


def test_lag_materially_changes_results():
    r0, r1 = _run(0), _run(1)
    assert abs(float(r0.equity_curve.iloc[-1]) - float(r1.equity_curve.iloc[-1])) > 1e-6
