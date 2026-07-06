from src.data.artifacts.regime import classify_atr_regime


def test_high_ratio_is_trend():
    assert classify_atr_regime(1.5, 1.0) == "TREND"


def test_low_ratio_is_mr():
    assert classify_atr_regime(0.7, 1.0) == "MR"


def test_middle_is_neutral():
    assert classify_atr_regime(1.0, 1.0) == "NEUTRAL"


def test_zero_slow_is_neutral():
    assert classify_atr_regime(1.5, 0.0) == "NEUTRAL"
