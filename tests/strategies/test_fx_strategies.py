import numpy as np
import pandas as pd

from src.strategies.registry import get_strategy_class


def _price_panel(pairs, n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    data = {p: 1.0 + np.cumsum(rng.normal(0, 0.001, n)) for p in pairs}
    return pd.DataFrame(data, index=idx)


def test_fx_trend_registered_and_forecasts():
    cls = get_strategy_class("FxTrend")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"]))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]
    assert fc.abs().max().max() <= 20.0  # forecast cap


def test_fx_value_registered_and_forecasts():
    cls = get_strategy_class("FxValue")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"], n=1400))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]


def _trending_panel(specs, n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    return pd.DataFrame({name: base + slope * t for name, base, slope in specs}, index=idx)


def test_fx_tsmom_sign_convention():
    cls = get_strategy_class("FxTSMOM")
    strat = cls(["UP", "DOWN"])
    panel = _trending_panel([("UP", 1.0, 0.001), ("DOWN", 2.0, -0.001)], n=300)
    fc = strat.forecast_panel(panel)
    assert list(fc.columns) == ["UP", "DOWN"]
    assert fc["UP"].iloc[-1] == 10.0      # both lookbacks positive -> +scale
    assert fc["DOWN"].iloc[-1] == -10.0   # both lookbacks negative -> -scale
    assert not fc.isna().any().any()      # warm-up rows filled to 0
    assert fc.abs().max().max() <= 10.0


def test_fx_carry_sign_convention(monkeypatch):
    import src.data.fx_rates as fx_rates

    def fake_rate_panel(currencies, index):
        rates = {"AUD": 0.05, "USD": 0.02, "CHF": 0.08}
        return pd.DataFrame({c: pd.Series(rates[c], index=index) for c in currencies})

    monkeypatch.setattr(fx_rates, "load_fx_rate_panel", fake_rate_panel)
    cls = get_strategy_class("FxCarry")
    strat = cls(["AUDUSD", "USDCHF"])
    panel = _price_panel(["AUDUSD", "USDCHF"], n=200)
    fc = strat.forecast_panel(panel)
    assert list(fc.columns) == ["AUDUSD", "USDCHF"]
    assert fc["AUDUSD"].iloc[-1] > 0      # base carry > quote -> long
    assert fc["USDCHF"].iloc[-1] < 0      # base carry < quote -> short
    assert fc.abs().max().max() <= strat.cap
    assert not fc.isna().any().any()


def test_fx_goldsilver_two_instrument_opposite_signs():
    cls = get_strategy_class("FxGoldSilver")
    strat = cls(("XAUUSD", "XAGUSD"), lookback=252)
    idx = pd.date_range("2018-01-01", periods=400, freq="D")
    rng = np.random.default_rng(1)
    xau = 1500 + np.cumsum(rng.normal(0, 1.0, 400))
    xau[-20:] += np.arange(20) * 15.0        # ratio spikes rich at the end
    xag = 20 + np.cumsum(rng.normal(0, 0.02, 400))
    panel = pd.DataFrame({"XAUUSD": xau, "XAGUSD": xag}, index=idx)
    fc = strat.forecast_panel(panel)
    assert list(fc.columns) == ["XAUUSD", "XAGUSD"]
    assert fc["XAUUSD"].iloc[-1] < 0         # ratio rich -> short gold
    assert fc["XAGUSD"].iloc[-1] > 0         # long silver
    assert fc.abs().max().max() <= 20.0
    assert not fc.isna().any().any()


def test_fx_xsectmom_ranks_strong_over_weak():
    cls = get_strategy_class("FxXSectMom")
    strat = cls(["STRONG", "MID", "WEAK"])
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(2)
    def series(drift):
        return 1.0 + np.cumsum(rng.normal(drift, 0.001, 200))
    panel = pd.DataFrame({"STRONG": series(0.002), "MID": series(0.0),
                          "WEAK": series(-0.002)}, index=idx)
    fc = strat.forecast_panel(panel)
    assert list(fc.columns) == ["STRONG", "MID", "WEAK"]
    assert fc["STRONG"].iloc[-1] > fc["WEAK"].iloc[-1]
    assert fc["STRONG"].iloc[-1] > 0 > fc["WEAK"].iloc[-1]
    assert fc.abs().max().max() <= 20.0
    assert not fc.isna().any().any()
