"""IBKRBroker.is_market_open must follow the NYSE calendar: holidays and early closes included.

On 2026-09-07 (Labor Day) the old weekday 09:30-16:00 check let RAMP's 15:55 rebalance send
market orders that IBKR queued for the next open.
"""

from datetime import datetime

import pytest
import pytz

from src.trading.brokers.ibkr import ibkr_broker
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
from src.trading.brokers.interfaces import BrokerConnectionError
from src.utils.timezone import tz

EASTERN = pytz.timezone("US/Eastern")


def market_open_at(monkeypatch, year, month, day, hour, minute, second=0):
    moment = EASTERN.localize(datetime(year, month, day, hour, minute, second))
    monkeypatch.setattr(tz, "now", lambda: moment)
    return IBKRBroker().is_market_open()


@pytest.mark.parametrize(
    "when",
    [(2026, 10, 7, 9, 30), (2026, 10, 7, 15, 55), (2026, 10, 7, 15, 59, 59)],
    ids=["open-bell", "rebalance-time", "last-second"],
)
def test_regular_session_is_open(monkeypatch, when):
    assert market_open_at(monkeypatch, *when) is True


@pytest.mark.parametrize(
    "when",
    [(2026, 10, 7, 9, 29, 59), (2026, 10, 7, 16, 0), (2026, 10, 7, 20, 0)],
    ids=["pre-open", "closing-bell", "evening"],
)
def test_outside_regular_session_is_closed(monkeypatch, when):
    assert market_open_at(monkeypatch, *when) is False


@pytest.mark.parametrize(
    "when",
    [(2026, 9, 7, 15, 55), (2026, 11, 26, 12, 0), (2026, 12, 25, 15, 55), (2027, 1, 1, 10, 0)],
    ids=["labor-day", "thanksgiving", "christmas", "new-years-day"],
)
def test_weekday_holidays_are_closed(monkeypatch, when):
    assert market_open_at(monkeypatch, *when) is False


@pytest.mark.parametrize("day", [(2026, 11, 27), (2026, 12, 24)], ids=["black-friday", "christmas-eve"])
def test_early_close_days_close_at_13_00(monkeypatch, day):
    assert market_open_at(monkeypatch, *day, 12, 59) is True
    assert market_open_at(monkeypatch, *day, 13, 0) is False
    assert market_open_at(monkeypatch, *day, 15, 55) is False


def test_weekend_is_closed(monkeypatch):
    assert market_open_at(monkeypatch, 2026, 10, 10, 12, 0) is False


def test_calendar_failure_raises_broker_connection_error(monkeypatch):
    def broken_calendar(day):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(ibkr_broker, "_nyse_session", broken_calendar)
    monkeypatch.setattr(tz, "now", lambda: EASTERN.localize(datetime(2026, 10, 7, 15, 55)))

    with pytest.raises(BrokerConnectionError):
        IBKRBroker().is_market_open()
