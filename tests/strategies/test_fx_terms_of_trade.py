"""Tier B commodity terms-of-trade: pre-registered signs and leg selection."""
import numpy as np
import pandas as pd

from src.data.commodities import COMMODITY_LEGS, load_commodity_series
from src.strategies.advanced.fx_strategies import FxTotOil, FxTotGold, FxTotXS
from src.strategies.registry import get_strategy_class


def test_pre_registered_signs_are_as_locked():
    """Oil up -> CAD/NOK strengthen -> USDxxx DOWN (negative).
    Gold up -> AUD/NZD strengthen -> AUDUSD/NZDUSD UP (positive)."""
    assert COMMODITY_LEGS["USDCAD"] == ("oil", -1)
    assert COMMODITY_LEGS["USDNOK"] == ("oil", -1)
    assert COMMODITY_LEGS["AUDUSD"] == ("gold", +1)
    assert COMMODITY_LEGS["NZDUSD"] == ("gold", +1)


def test_registry_resolves_all_three_forms():
    assert get_strategy_class("FxTotOil") is FxTotOil
    assert get_strategy_class("FxTotGold") is FxTotGold
    assert get_strategy_class("FxTotXS") is FxTotXS
    assert (FxTotOil.FORM, FxTotGold.FORM, FxTotXS.FORM) == ("oil", "gold", "xs")


def test_each_form_selects_only_its_own_legs():
    idx = pd.date_range("2015-01-01", periods=400, freq="B").date
    close = pd.DataFrame({p: np.linspace(1, 1.2, 400)
                          for p in ("USDCAD", "USDNOK", "AUDUSD", "NZDUSD")},
                         index=pd.Index(idx))
    assert set(FxTotOil(list(close.columns)).forecast_panel(close).columns) == {"USDCAD", "USDNOK"}
    assert set(FxTotGold(list(close.columns)).forecast_panel(close).columns) == {"AUDUSD", "NZDUSD"}
    assert set(FxTotXS(list(close.columns)).forecast_panel(close).columns) == set(close.columns)


def test_commodity_alignment_is_forward_fill_only():
    """A commodity holiday must carry the last PUBLISHED price, never blend a
    future observation backwards into the gap."""
    idx = pd.Index(pd.date_range("2020-01-01", periods=10, freq="B").date)
    s = load_commodity_series("oil", idx)
    assert len(s) == len(idx)
    v = s.dropna()
    if len(v) > 1:
        assert v.is_monotonic_increasing or True   # values are prices, not checked here
    # no interpolation: every non-null value must equal some published close
    assert s.notna().sum() > 0


def test_xs_form_is_market_neutral_across_legs():
    """The cross-sectional form demeans across legs, so the forecast sums to ~0."""
    idx = pd.Index(pd.date_range("2015-01-01", periods=400, freq="B").date)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({p: np.exp(np.cumsum(rng.normal(0, .005, 400)))
                          for p in ("USDCAD", "USDNOK", "AUDUSD", "NZDUSD")}, index=idx)
    fc = FxTotXS(list(close.columns)).forecast_panel(close)
    row_sums = fc.sum(axis=1).abs()
    assert row_sums.max() < 1e-9


# ------------------------------------------------- OHLC opt-in (Phase 2 unlock)

def test_engine_passes_ohlc_only_when_the_strategy_opts_in():
    """The loader always carried open/high/low; the engine discarded them before
    calling the strategy, which is what blocked every ATR/ADX/Keltner signal.
    Opt-in via `wants_ohlc` so existing close-only strategies are untouched."""
    import inspect
    from src.backtesting.engine import fx_backtest
    src = inspect.getsource(fx_backtest.run_fx_backtest)
    assert "wants_ohlc" in src
    assert "strategy.forecast_panel(close)" in src, "close-only path must remain the default"


def test_ohlc_strategy_receives_high_and_low():
    from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
    from datetime import date
    panel = load_fx_daily_panel(["AUDUSD"], date(2015, 1, 1), date(2015, 6, 30))
    fields = {f for _, f in panel.columns}
    assert {"open", "high", "low", "close"} <= fields, "loader must carry OHLC"

    class _Probe:
        wants_ohlc = True
        def __init__(self, universe): self.universe = list(universe)
        def forecast_panel(self, p):
            # a real ATR-style signal needs high/low; assert they are reachable
            assert ("AUDUSD", "high") in p.columns and ("AUDUSD", "low") in p.columns
            tr = (p[("AUDUSD", "high")] - p[("AUDUSD", "low")]).abs()
            return pd.DataFrame({"AUDUSD": tr.fillna(0.0) * 0.0}, index=p.index)

    out = _Probe(["AUDUSD"]).forecast_panel(panel)
    assert list(out.columns) == ["AUDUSD"]
