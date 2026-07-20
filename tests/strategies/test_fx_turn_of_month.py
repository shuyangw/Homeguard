import numpy as np
import pandas as pd

from src.strategies.registry import get_strategy_class


def _panel(n=260, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(7)
    audusd = 0.70 + np.cumsum(rng.normal(0, 0.0005, n))
    usdjpy = 110.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({"AUDUSD": audusd, "USDJPY": usdjpy}, index=idx)


def _window_positions(idx):
    """Reproduce the strategy's T-1..T+3 window (by index position) per month."""
    n = len(idx)
    dt_index = pd.DatetimeIndex(idx)
    by_month = pd.Series(range(n), index=dt_index).groupby([dt_index.year, dt_index.month])
    windows = []
    for _, positions in by_month:
        t_pos = int(positions.iloc[0])
        start_pos = max(0, t_pos - 1)
        end_pos = min(n - 1, t_pos + 3)
        windows.append((start_pos, end_pos))
    return windows


def test_flat_outside_window():
    panel = _panel()
    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc = strat.forecast_panel(panel)
    windows = _window_positions(panel.index)
    in_window = np.zeros(len(panel.index), dtype=bool)
    for start_pos, end_pos in windows:
        in_window[start_pos:end_pos + 1] = True
    outside = ~in_window
    assert (fc.loc[panel.index[outside], "AUDUSD"] == 0.0).all()
    assert (fc.loc[panel.index[outside], "USDJPY"] == 0.0).all()


def test_full_forecast_within_window_absent_stop():
    # Flat, low-volatility panel -> ATR proxy stop never trips.
    idx = pd.bdate_range("2021-01-01", periods=130)
    audusd = pd.Series(0.70, index=idx)
    usdjpy = pd.Series(110.0, index=idx)
    panel = pd.DataFrame({"AUDUSD": audusd, "USDJPY": usdjpy})
    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc = strat.forecast_panel(panel)
    windows = _window_positions(panel.index)
    # Pick a window well past the 14-day ATR burn-in so atr_entry is non-NaN.
    start_pos, end_pos = next(w for w in windows if w[0] >= 20)
    for pos in range(start_pos, end_pos + 1):
        assert fc.iloc[pos]["AUDUSD"] == 10.0
        assert fc.iloc[pos]["USDJPY"] == -10.0


def test_atr_stop_zeroes_forecast_on_large_adverse_move():
    idx = pd.bdate_range("2021-01-01", periods=130)
    audusd = pd.Series(0.70, index=idx).astype(float)
    usdjpy = pd.Series(110.0, index=idx).astype(float)
    # Small daily noise before the target window so the ATR proxy is small
    # and non-NaN, then a sharp adverse move (AUDUSD collapses, USDJPY spikes)
    # inside a window well past the 14-day burn-in.
    rng = np.random.default_rng(3)
    audusd += np.concatenate([np.zeros(20), rng.normal(0, 0.0002, len(idx) - 20)])
    usdjpy += np.concatenate([np.zeros(20), rng.normal(0, 0.01, len(idx) - 20)])
    panel = pd.DataFrame({"AUDUSD": audusd, "USDJPY": usdjpy})

    windows = _window_positions(panel.index)
    start_pos, end_pos = next(w for w in windows if w[0] >= 25)
    # Force a large adverse move on the day after entry: AUDUSD craters, USDJPY spikes.
    shock_pos = start_pos + 1
    panel.iloc[shock_pos:, panel.columns.get_loc("AUDUSD")] = 0.50
    panel.iloc[shock_pos:, panel.columns.get_loc("USDJPY")] = 130.0

    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc = strat.forecast_panel(panel)
    # Stop should have tripped by the end of the window: forecast flat.
    assert fc.iloc[end_pos]["AUDUSD"] == 0.0
    assert fc.iloc[end_pos]["USDJPY"] == 0.0
    # Entry day itself is unaffected (adverse move is 0 at entry).
    assert fc.iloc[start_pos]["AUDUSD"] == 10.0
    assert fc.iloc[start_pos]["USDJPY"] == -10.0


def test_atr_stop_fires_on_audusd_only_leg_move():
    """Regression test for the raw-price-vs-fractional-scale bug: USDJPY's
    price level (~110) is ~150x AUDUSD's (~0.70), so under the old raw-price
    -diff ATR proxy / pnl_proxy, an adverse move confined to the AUDUSD leg
    is numerically swamped by USDJPY's raw-price range and never trips the
    stop -- even though it is a large move in percentage terms. Only AUDUSD
    moves here (a 5% adverse drop); USDJPY is frozen flat for the rest of the
    window. With the fractional/pct-return fix, this must still trip the stop.

    Sanity-checked by hand: pre-shock daily diffs are AUDUSD +-0.0005 (~0.07%)
    and USDJPY +-0.5 (~0.45%), alternating, so the raw-price ATR proxy is
    dominated by USDJPY (~0.5005) while the fractional ATR proxy is small and
    comparable across legs (~0.0053). A 5% AUDUSD-only adverse move (~0.035 in
    raw price terms, 0.05 in fractional terms) clears the fractional threshold
    but would NOT have cleared the raw-price threshold -- i.e. this test would
    have failed under the pre-fix implementation.
    """
    idx = pd.bdate_range("2021-01-01", periods=130)
    n = len(idx)
    parity = np.arange(n) % 2
    audusd = 0.70 + 0.0005 * parity
    usdjpy = 110.0 + 0.5 * parity
    panel = pd.DataFrame({"AUDUSD": audusd, "USDJPY": usdjpy}, index=idx)

    windows = _window_positions(panel.index)
    start_pos, end_pos = next(w for w in windows if w[0] >= 25)
    shock_pos = start_pos + 1
    audusd_entry = panel.iloc[start_pos]["AUDUSD"]
    usdjpy_entry = panel.iloc[start_pos]["USDJPY"]
    # AUDUSD: 5% adverse drop, held for the rest of the window.
    panel.iloc[shock_pos:, panel.columns.get_loc("AUDUSD")] = audusd_entry * 0.95
    # USDJPY: frozen flat at its entry-day level -- this leg does not move.
    panel.iloc[shock_pos:, panel.columns.get_loc("USDJPY")] = usdjpy_entry

    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc = strat.forecast_panel(panel)
    # Stop should have tripped by the end of the window: forecast flat.
    assert fc.iloc[end_pos]["AUDUSD"] == 0.0
    assert fc.iloc[end_pos]["USDJPY"] == 0.0
    # Entry day itself is unaffected (adverse move is 0 at entry).
    assert fc.iloc[start_pos]["AUDUSD"] == 10.0
    assert fc.iloc[start_pos]["USDJPY"] == -10.0


def test_no_lookahead():
    panel = _panel(n=260)
    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc_full = strat.forecast_panel(panel)
    for t in (30, 90, 150, 200):
        truncated = panel.iloc[:t + 5]
        fc_trunc = strat.forecast_panel(truncated)
        pd.testing.assert_series_equal(
            fc_full.iloc[t], fc_trunc.iloc[t], check_names=False)


def test_output_columns_match_close_panel():
    panel = _panel(n=40)
    strat = get_strategy_class("FxTurnOfMonth")(["AUDUSD", "USDJPY"])
    fc = strat.forecast_panel(panel)
    assert list(fc.columns) == list(panel.columns)
    assert not fc.isna().any().any()
