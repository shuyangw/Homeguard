from datetime import date
from src.data.macro_calendar import load_cb_decisions, blackout


def test_load_returns_dates():
    d = load_cb_decisions()
    assert "ECB" in d
    assert all(isinstance(x, date) for x in d["ECB"])


def test_blackout_window():
    # A known ECB date must trigger blackout for EUR within +/- 1 day.
    d = load_cb_decisions()
    ref = d["ECB"][0]
    assert blackout("EUR", ref, days=1) is True
