from datetime import datetime, timedelta, timezone

import pytest

from tools.console import freshness

NOW = datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("kind,live,expected", [
    (freshness.INSTANCE_OWNED, True, "live"),
    (freshness.MARKET_OWNED, True, "live"),
    (freshness.MEASURED, True, "live"),
    (freshness.INSTANCE_OWNED, False, "frozen"),
    (freshness.MARKET_OWNED, False, "drifting"),
    (freshness.MEASURED, False, "unknown"),
])
def test_classify(kind, live, expected):
    assert freshness.classify(kind, live) == expected


@pytest.mark.parametrize("delta,text", [
    (timedelta(seconds=4), "4 s ago"),
    (timedelta(minutes=5, seconds=10), "5 min ago"),
    (timedelta(hours=12, minutes=3), "12 h 3 min ago"),
])
def test_age_text(delta, text):
    assert freshness.age_text(NOW - delta, NOW) == text


def test_age_text_clamps_future_readings_to_zero():
    assert freshness.age_text(NOW + timedelta(seconds=3), NOW) == "0 s ago"
