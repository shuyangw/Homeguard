import numpy as np
import pandas as pd

from src.backtesting.engine.spread_sizing import Spread
from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs


def _coint_panel(n=400, div_start=350):
    # AUDUSD and NZDUSD co-move; inject a residual divergence late so |z|>2.
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.004, n))
    aud = 0.70 * np.exp(common + rng.normal(0, 0.0005, n))
    nzd = 0.65 * np.exp(common + rng.normal(0, 0.0005, n))
    aud[div_start:] *= 1.03  # AUD richens vs NZD -> residual z spikes
    return pd.DataFrame({"AUDUSD": aud, "NZDUSD": nzd}, index=pd.Index(idx))


def test_emits_spread_when_residual_z_exceeds_entry():
    close = _coint_panel()
    book, sigma = AudNzdPairs().spread_book(close)
    # some late date has an active AUDUSD/NZDUSD spread
    active = [d for d, sps in book.items() if sps]
    assert active, "expected at least one active spread after the divergence"
    sp = book[active[-1]][0]
    assert {sp.leg_a, sp.leg_b} == {"AUDUSD", "NZDUSD"}
    assert (sp.leg_a, sp.leg_b) in sigma[active[-1]]  # spread vol provided


def test_hedge_ratio_is_from_regression_not_one():
    close = _coint_panel()
    book, _ = AudNzdPairs().spread_book(close)
    active = [book[d][0] for d in book if book[d]]
    assert active
    # beta from ln-regression should differ from a naive 1.0
    assert abs(active[-1].hedge_ratio - 1.0) > 1e-6


_STOP_DIV_START = 200


def _stop_panel(n=300, div=_STOP_DIV_START):
    # AUD richens sharply, then holds: residual z enters (>entry) then blows out
    # past stop_z while still > entry_z, so the stop must fire mid-trade.
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.004, n))
    aud = 0.70 * np.exp(common + rng.normal(0, 0.0003, n))
    nzd = 0.65 * np.exp(common + rng.normal(0, 0.0003, n))
    ramp = np.array([min(0.06, 0.06 * (k - div) / 15.0) if k >= div else 0.0
                     for k in range(n)])
    aud = aud * np.exp(ramp)
    return pd.DataFrame({"AUDUSD": aud, "NZDUSD": nzd}, index=pd.Index(idx))


def test_stop_is_a_real_exit_no_same_bar_reentry():
    # Pins #35: on a rebalance bar where an open position's |z| crosses ABOVE
    # stop_z while still > entry_z, the stop must genuinely flatten the book --
    # the bar must NOT re-enter on the same bar (the old bug reset the clock and
    # re-entered, so the stop never flattened).
    strat = AudNzdPairs()
    close = _stop_panel().sort_index()
    dates = list(close.index)
    ln_a = np.log(close["AUDUSD"].astype(float).values)
    ln_b = np.log(close["NZDUSD"].astype(float).values)

    rebals = []
    prev = None
    for i, d in enumerate(dates):
        is_reb = strat._is_rebalance(d, prev)
        prev = d
        # only the engineered divergence region -- pre-divergence noise can
        # briefly cross entry_z and would confound the entry/stop pairing.
        if not is_reb or i < _STOP_DIV_START:
            continue
        reg = strat._regression_z(ln_a, ln_b, i)
        if reg is not None:
            rebals.append((i, d, reg[1]))

    entry = next((r for r in rebals if abs(r[2]) > strat.entry_z), None)
    assert entry is not None, "scenario must produce an entry (|z| > entry_z)"
    entry_i, entry_d, _ = entry
    stop = next((r for r in rebals if r[0] > entry_i
                 and abs(r[2]) > strat.stop_z
                 and (r[0] - entry_i) < strat.max_days), None)
    assert stop is not None, "scenario must produce a mid-trade stop crossing"
    stop_i, stop_d, _ = stop
    # position stays open from entry through the stop bar (no earlier target exit)
    between = [r for r in rebals if entry_i < r[0] < stop_i]
    assert all(abs(z) >= strat.target_z for _, _, z in between)

    book, _ = strat.spread_book(close)
    assert book.get(entry_d), "entry bar should hold an active spread"
    # the stop bar must be FLAT -- the stop actually exited, no same-bar re-entry
    assert not book.get(stop_d), "stop bar must not hold an active spread"
