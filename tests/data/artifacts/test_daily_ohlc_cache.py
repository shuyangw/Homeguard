from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache, G10_PAIRS


def test_inputs_and_targets():
    b = DailyOhlcCache()
    assert b.name == "daily_ohlc_cache"
    assert b.inputs() == ["minute"]
    assert "AUDUSD" in b.target_pairs()
    assert "EURUSD" in b.target_pairs()


def test_g10_pairs_are_fourteen():
    assert len(G10_PAIRS) == 14
    assert "NOKSEK" in G10_PAIRS
