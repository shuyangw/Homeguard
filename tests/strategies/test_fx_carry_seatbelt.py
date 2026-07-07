import numpy as np
import pandas as pd

import src.data.fx_rates as fx_rates
from src.strategies.registry import get_strategy_class


def _panel(pairs, n=400, drift=0.0005):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    return pd.DataFrame({p: 100.0 * (1.0 + drift) ** t for p in pairs}, index=idx)


def _patch_rates(monkeypatch, rate_map):
    def fake(currencies, index):
        return pd.DataFrame(
            {c: pd.Series(rate_map.get(c, 0.0), index=index) for c in currencies})
    monkeypatch.setattr(fx_rates, "load_fx_rate_panel", fake)


def test_long_only_when_carry_and_momentum_agree(monkeypatch):
    # AUD 5%, JPY 0% -> AUDJPY carry +5% > 2% gate; uptrend -> long.
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=0.0008))
    assert fc["AUDJPY"].iloc[-1] == 10.0


def test_flat_when_carry_fails(monkeypatch):
    # AUD 1% -> carry +1% < 2% gate -> flat despite uptrend.
    _patch_rates(monkeypatch, {"AUD": 0.01, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=0.0008))
    assert fc["AUDJPY"].iloc[-1] == 0.0


def test_flat_when_momentum_fails(monkeypatch):
    # Good carry but downtrend -> flat (never short for carry).
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=-0.0008))
    assert fc["AUDJPY"].iloc[-1] == 0.0


def test_veto_zeroes_longs_on_risk_off(monkeypatch):
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0, "CHF": 0.0})
    pairs = ["AUDJPY", "USDJPY", "EURJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    panel = _panel(pairs, drift=0.0008)
    # risk-off shock at the end -> veto engages -> AUDJPY long zeroed
    for p in ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "NZDJPY"]:
        panel.iloc[-3:, panel.columns.get_loc(p)] *= 0.90
    panel.iloc[-3:, panel.columns.get_loc("XAUUSD")] *= 1.08
    strat = get_strategy_class("FxCarrySeatbelt")(pairs)
    fc = strat.forecast_panel(panel)
    assert fc["AUDJPY"].iloc[-1] <= 0.0  # long flattened (and maybe shorted)


def test_forecast_is_causal_and_bounded(monkeypatch):
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0, "CHF": 0.0})
    pairs = ["AUDJPY", "USDJPY", "EURJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    panel = _panel(pairs, drift=0.0006)
    strat = get_strategy_class("FxCarrySeatbelt")(pairs)
    fc = strat.forecast_panel(panel)
    assert fc.abs().max().max() <= 10.0
    assert not fc.isna().any().any()
    fc_trunc = strat.forecast_panel(panel.iloc[:250])
    pd.testing.assert_frame_equal(fc.iloc[:250], fc_trunc, check_names=False)
