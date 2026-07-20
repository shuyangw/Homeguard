import numpy as np
import pandas as pd
import pytest

from src.strategies.registry import get_strategy_class

MAJORS = ["EURUSD", "USDJPY", "USDCHF", "GBPUSD", "USDCAD", "AUDUSD", "NZDUSD",
          "XAUUSD", "XAGUSD", "USDNOK", "USDSEK"]
MINORS = ["AUDNZD", "AUDJPY", "NZDJPY", "EURNOK", "EURSEK", "NOKSEK", "NOKJPY", "SEKJPY"]
UNIVERSE = MAJORS + MINORS


def _build_panel(pairs, n, seed, factor_scale=0.006, idio_scale=0.0007, shocks=None):
    """Deterministic synthetic G10-style panel: common dollar factor + per-pair
    idiosyncratic noise, with optional (pair, row, delta) return shocks injected."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    factor = rng.normal(0, factor_scale, n)
    rets = {p: factor + rng.normal(0, idio_scale, n) for p in pairs}
    for pair, row, delta in (shocks or []):
        rets[pair][row] += delta
    prices = {}
    for p in pairs:
        r = np.array(rets[p])
        r[0] = 0.0
        prices[p] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(prices, index=idx)


def _strategy(pairs=UNIVERSE):
    return get_strategy_class("FxPcaDollarResidual")(pairs)


def test_flat_before_250_days_of_history():
    panel = _build_panel(UNIVERSE, n=200, seed=1)
    fc = _strategy().forecast_panel(panel)
    assert (fc == 0.0).all().all()


def test_selects_two_most_negative_and_two_most_positive_major_residuals():
    # 5-day shocks ending exactly at the rebalance date (position 250):
    # GBPUSD/NZDUSD idiosyncratically weak -> most-negative residual z -> longs
    # USDCHF/USDCAD idiosyncratically strong -> most-positive residual z -> shorts
    shocks = []
    for t in range(246, 251):
        shocks += [
            ("GBPUSD", t, -0.0015), ("NZDUSD", t, -0.0015),
            ("USDCHF", t, 0.0015), ("USDCAD", t, 0.0015),
        ]
    panel = _build_panel(UNIVERSE, n=255, seed=11, shocks=shocks)
    fc = _strategy().forecast_panel(panel)
    row = fc.iloc[250]

    assert row["GBPUSD"] == 10.0
    assert row["NZDUSD"] == 10.0
    assert row["USDCHF"] == -10.0
    assert row["USDCAD"] == -10.0
    untouched_majors = [p for p in MAJORS if p not in ("GBPUSD", "NZDUSD", "USDCHF", "USDCAD")]
    for p in untouched_majors:
        assert row[p] == 0.0


def test_minor_pair_never_receives_nonzero_forecast_even_if_most_extreme():
    # AUDNZD (minor tier) gets a LARGER shock than the major-tier picks, so it
    # would rank #1 most-negative by residual z -- but the liquidity gate must
    # exclude it from trading regardless of rank.
    shocks = [("AUDNZD", t, -0.02) for t in range(246, 251)]
    for t in range(246, 251):
        shocks += [
            ("GBPUSD", t, -0.0015), ("NZDUSD", t, -0.0015),
            ("USDCHF", t, 0.0015), ("USDCAD", t, 0.0015),
        ]
    panel = _build_panel(UNIVERSE, n=255, seed=11, shocks=shocks)
    fc = _strategy().forecast_panel(panel)
    assert fc.iloc[250]["AUDNZD"] == 0.0
    # majors still pick up the trade despite AUDNZD's more extreme residual
    assert fc.iloc[250]["GBPUSD"] == 10.0
    assert fc.iloc[250]["USDCAD"] == -10.0


def test_disaster_stop_zeroes_a_leg_after_extreme_adverse_move():
    shocks = []
    for t in range(246, 251):
        shocks += [
            ("GBPUSD", t, -0.0015), ("NZDUSD", t, -0.0015),
            ("USDCHF", t, 0.0015), ("USDCAD", t, 0.0015),
        ]
    # targeted adverse move on GBPUSD two days after entry, well beyond -3 sigma
    shocks.append(("GBPUSD", 252, -0.010))
    panel = _build_panel(UNIVERSE, n=265, seed=11, shocks=shocks)
    fc = _strategy().forecast_panel(panel)

    assert fc.iloc[250]["GBPUSD"] == 10.0   # entry
    assert fc.iloc[251]["GBPUSD"] == 10.0   # survives one day
    assert fc.iloc[252]["GBPUSD"] == 0.0    # disaster stop fires
    assert fc.iloc[253]["GBPUSD"] == 0.0    # stays flat for the rest of the cycle


def test_pc1_variance_jump_flattens_whole_basket():
    # rows 251-255 (entering the SECOND rebalance's trailing window at pos=255,
    # but not the first, at pos=250) get a huge shock shared identically across
    # ALL pairs -- a near-rank-1 regime that spikes PC1's explained variance by
    # more than 15pts week-over-week relative to the first rebalance's window.
    panel = _build_panel(UNIVERSE, n=260, seed=3, factor_scale=0.004, idio_scale=0.0025)
    returns = panel.pct_change()
    for t in range(251, 256):
        shock = 0.05 if t % 2 == 0 else -0.05
        for p in UNIVERSE:
            returns.iloc[t, returns.columns.get_loc(p)] = shock
    prices = {}
    for p in UNIVERSE:
        r = returns[p].fillna(0.0).values.copy()
        prices[p] = 100.0 * np.cumprod(1.0 + r)
    panel = pd.DataFrame(prices, index=panel.index)

    fc = _strategy().forecast_panel(panel)
    assert fc.iloc[250].abs().sum() > 0.0     # normal rebalance, some legs picked
    assert (fc.iloc[255] == 0.0).all()        # regime-break flatten on the next cycle


def test_no_lookahead_truncation_invariance():
    panel = _build_panel(UNIVERSE, n=400, seed=42, idio_scale=0.001)
    strat = _strategy()
    fc_full = strat.forecast_panel(panel)
    fc_trunc = _strategy().forecast_panel(panel.iloc[:300])
    pd.testing.assert_frame_equal(fc_full.iloc[:300], fc_trunc, check_names=False)


def test_output_columns_match_close_panel():
    panel = _build_panel(UNIVERSE, n=60, seed=5)
    fc = _strategy().forecast_panel(panel)
    assert list(fc.columns) == list(panel.columns)
    assert not fc.isna().any().any()


def test_late_starting_pair_does_not_crash_pca_window():
    # NOKJPY mirrors the real NOKJPY/SEKJPY case: no price history at all for
    # its first 300 rows (simulating a pair whose data only starts in 2020,
    # long after the other pairs' 2011 start), with real prices only from
    # row 300 onward -- so the 250-day trailing PCA window is entirely NaN
    # for NOKJPY at every rebalance in this panel, well past the row-250
    # "enough history" boundary. Before the fix, dollar_factor()'s internal
    # dropna(how="any") would wipe out the ENTIRE trailing window (every row
    # has at least one NaN column) and crash with an empty-SVD IndexError.
    panel = _build_panel(UNIVERSE, n=400, seed=7)
    panel.loc[panel.index[:300], "NOKJPY"] = np.nan

    fc = _strategy().forecast_panel(panel)

    assert not fc.isna().any().any()
    # NOKJPY never has a complete trailing window in this panel -> always
    # excluded from ranking/estimation, never receives a nonzero forecast.
    assert (fc["NOKJPY"] == 0.0).all()
    # the rest of the universe still trades normally despite NOKJPY's gap.
    assert fc[MAJORS].abs().sum().sum() > 0.0
