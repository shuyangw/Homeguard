import numpy as np
import pandas as pd
from src.strategies.advanced.futures_cot_strategy import FuturesCoTTiltStrategy, _publication_lag


def test_publication_lag_tuesday_to_friday():
    # a Tuesday snapshot is known at Friday (+3 days)
    assert _publication_lag(pd.Timestamp("2015-01-06")) == pd.Timestamp("2015-01-09")


def test_publication_lag_holiday_week_delays_to_next_business_day():
    # Thanksgiving week 2015: Tue 2015-11-24 snapshot -> normal Friday is
    # 2015-11-27, but Thu 2015-11-26 (Thanksgiving) falls in that publication
    # week, so the CFTC delays release to the next business day: Mon 2015-11-30.
    normal_friday = pd.Timestamp("2015-11-27")
    lagged = _publication_lag(pd.Timestamp("2015-11-24"))
    assert lagged == pd.Timestamp("2015-11-30")
    assert lagged > normal_friday

    # a normal week (no holiday in Mon..Fri) is unaffected
    assert _publication_lag(pd.Timestamp("2015-01-06")) == pd.Timestamp("2015-01-09")


def test_rising_spec_long_gives_positive_forecast(monkeypatch):
    idx = pd.date_range("2015-01-01", periods=800, freq="B")
    close = pd.DataFrame({"ES": np.linspace(2000, 2500, 800)}, index=idx)
    # weekly CoT: net spec rising over time -> positioning momentum -> long
    weeks = pd.date_range("2013-01-01", "2016-01-01", freq="W-TUE")
    cot = pd.DataFrame({
        "report_date": weeks,
        "noncommercial_long": np.linspace(1000, 5000, len(weeks)),
        "noncommercial_short": np.full(len(weeks), 1000.0),
    })
    strat = FuturesCoTTiltStrategy(["ES"])
    monkeypatch.setattr(strat, "_load_cot", lambda root: cot)
    fc = strat.forecast_panel(close)
    assert fc.iloc[-1]["ES"] > 0
    assert fc.abs().max().max() <= 20.0


def test_missing_cot_gives_zero(monkeypatch):
    idx = pd.date_range("2015-01-01", periods=60, freq="B")
    close = pd.DataFrame({"ES": np.linspace(2000, 2100, 60)}, index=idx)
    strat = FuturesCoTTiltStrategy(["ES"])
    monkeypatch.setattr(strat, "_load_cot", lambda root: None)
    fc = strat.forecast_panel(close)
    assert (fc["ES"] == 0.0).all()
