from src.data.artifacts.spread_model import synthetic_spread


def test_cross_wider_than_major():
    anchors = {"EURUSD": 1.0}
    maj = synthetic_spread("EURUSD", 10, anchors)
    cross = synthetic_spread("EURNOK", 10, anchors)
    assert cross > maj


def test_rollover_hour_widens():
    anchors = {"EURUSD": 1.0}
    normal = synthetic_spread("EURUSD", 10, anchors)
    rollover = synthetic_spread("EURUSD", 21, anchors)  # 21:00 UTC rollover
    assert rollover > normal
