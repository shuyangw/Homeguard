import datetime as dt

import pandas as pd
import polars as pl
import pytest

from src.backtesting.data import fx_backtest_loader
from src.backtesting.data.fx_backtest_loader import build_quote_usd_panel


def _close_panel(data: dict) -> pd.DataFrame:
    idx = pd.Index([dt.date(2024, 1, 2), dt.date(2024, 1, 3)], name="fx_date")
    frames = {}
    for pair, closes in data.items():
        frames[(pair, "close")] = pd.Series(closes, index=idx)
        frames[(pair, "ret")] = pd.Series(closes, index=idx).pct_change()
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_usd_quote_pair_is_identity():
    panel = _close_panel({"EURUSD": [1.10, 1.11]})
    q = build_quote_usd_panel(panel, ["EURUSD"])
    # quote is USD -> USD->USD = 1.0
    assert (q["EURUSD"] == 1.0).all()


def test_usd_base_pair_inverts():
    panel = _close_panel({"USDJPY": [150.0, 160.0]})
    q = build_quote_usd_panel(panel, ["USDJPY"])
    # quote is JPY -> JPY->USD = 1/USDJPY
    assert q["USDJPY"].iloc[0] == pytest.approx(1 / 150.0)
    assert q["USDJPY"].iloc[1] == pytest.approx(1 / 160.0)


def test_true_cross_uses_usd_leg():
    # EURGBP: quote GBP -> USD via GBPUSD
    panel = _close_panel({"EURGBP": [0.85, 0.86], "GBPUSD": [1.25, 1.26]})
    q = build_quote_usd_panel(panel, ["EURGBP", "GBPUSD"])
    assert q["EURGBP"].iloc[0] == pytest.approx(1.25)
    assert q["EURGBP"].iloc[1] == pytest.approx(1.26)


def test_missing_usd_leg_raises():
    panel = _close_panel({"EURGBP": [0.85, 0.86]})  # no GBPUSD, no USDGBP
    with pytest.raises(ValueError):
        build_quote_usd_panel(panel, ["EURGBP"])


def _write_fixture(root, pair, dates, closes, opens=None, highs=None, lows=None):
    d = root / "fx_daily" / f"symbol={pair}" / "year=2024" / "month=1"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "fx_date": dates,
        "open": opens if opens is not None else closes,
        "high": highs if highs is not None else closes,
        "low": lows if lows is not None else closes,
        "close": closes,
    }).write_parquet(d / "data.parquet")


def test_load_fx_daily_panel_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_backtest_loader, "get_local_storage_dir", lambda: str(tmp_path))
    ds = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    _write_fixture(
        tmp_path, "EURUSD", ds,
        closes=[1.10, 1.11], opens=[1.09, 1.10], highs=[1.12, 1.13], lows=[1.08, 1.095])
    panel = fx_backtest_loader.load_fx_daily_panel(
        ["EURUSD"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert ("EURUSD", "close") in panel.columns
    assert ("EURUSD", "ret") in panel.columns
    assert ("EURUSD", "open") in panel.columns
    assert ("EURUSD", "high") in panel.columns
    assert ("EURUSD", "low") in panel.columns
    assert panel[("EURUSD", "close")].tolist() == [1.10, 1.11]
    assert panel[("EURUSD", "open")].tolist() == [1.09, 1.10]
    assert panel[("EURUSD", "high")].tolist() == [1.12, 1.13]
    assert panel[("EURUSD", "low")].tolist() == [1.08, 1.095]


def test_load_fx_daily_panel_excludes_missing_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_backtest_loader, "get_local_storage_dir", lambda: str(tmp_path))
    ds = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    _write_fixture(tmp_path, "EURUSD", ds, [1.10, 1.11])
    # GBPUSD has no data dir -> excluded, not fatal
    panel = fx_backtest_loader.load_fx_daily_panel(
        ["EURUSD", "GBPUSD"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert {c[0] for c in panel.columns} == {"EURUSD"}


def test_load_fx_daily_panel_raises_when_all_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fx_backtest_loader, "get_local_storage_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        fx_backtest_loader.load_fx_daily_panel(
            ["EURUSD"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
