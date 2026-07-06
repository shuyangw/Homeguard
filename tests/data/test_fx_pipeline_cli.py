from src.data.fx_pipeline import list_components


def test_list_includes_daily_ohlc_cache():
    comps = list_components()
    names = {c["name"] for c in comps}
    assert "daily_ohlc_cache" in names
    row = next(c for c in comps if c["name"] == "daily_ohlc_cache")
    assert row["requires_key"] is None
