import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.engine.fill_sink import FillSink


def test_init_creates_run_dir_and_meta(tmp_path):
    sink = FillSink(
        strategy="FxDemo",
        run_id="20260720T000000Z_abc123",
        meta={"kind": "walkforward", "n_windows": 3},
        root=tmp_path,
    )
    assert sink.run_dir == tmp_path / "FxDemo" / "runs" / "20260720T000000Z_abc123"
    assert sink.run_dir.is_dir()
    meta = json.loads((sink.run_dir / "meta.json").read_text())
    assert meta["kind"] == "walkforward"
    assert meta["n_windows"] == 3
    assert meta["strategy"] == "FxDemo"


def test_make_run_id_is_deterministic_given_now():
    now = datetime(2026, 7, 20, 1, 45, 30, tzinfo=timezone.utc)
    assert FillSink.make_run_id("a1b2c3", now) == "20260720T014530Z_a1b2c3"


def test_write_window_gzips_and_names(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2011-01-03"], "pair": ["EURUSD"], "units": [100.0]})
    path = sink.write_window(df, window=1, cfg_hash="a1b2c3")
    assert path.name == "w01_a1b2c3_trades.csv.gz"
    back = pd.read_csv(path)
    assert list(back.columns) == ["date", "pair", "units"]
    assert len(back) == 1


def test_write_window_without_cfg_hash(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2011-01-03"], "units": [1.0]})
    path = sink.write_window(df, window=2)
    assert path.name == "w02_trades.csv.gz"


def test_write_window_extras_sidecars(tmp_path):
    sink = FillSink("FutDemo", "rid", {}, root=tmp_path)
    trades = pd.DataFrame({"units": [1.0]})
    margin = pd.DataFrame({"margin": [0.3]})
    sink.write_window(trades, window=1, extras={"margin_utilization": margin})
    assert (sink.run_dir / "w01_margin_utilization.csv.gz").exists()


def test_write_portfolio_delegates_to_tradelogger(tmp_path):
    sink = FillSink("EqDemo", "rid", {}, root=tmp_path)

    class FakePortfolio:
        # custom-Portfolio shape TradeLogger understands: trades is a list of dicts
        trades = [
            {"type": "entry", "timestamp": "2020-01-02", "price": 10.0, "shares": 5},
            {"type": "exit", "timestamp": "2020-01-05", "price": 11.0, "shares": 5,
             "pnl": 5.0, "pnl_pct": 0.1, "exit_reason": "target"},
        ]

    path = sink.write_portfolio(FakePortfolio(), window=1, cfg_hash="cfg9", symbol="AAPL")
    assert path.name == "w01_cfg9_trades.csv.gz"
    back = pd.read_csv(path)
    assert len(back) == 2  # one buy row + one sell row
    assert (sink.run_dir / "w01_cfg9_trades.csv.gz").exists()


def test_write_portfolio_surfaces_export_failure(tmp_path):
    sink = FillSink("EqDemo", "rid", {}, root=tmp_path)

    class BrokenPortfolio:
        @property
        def trades(self):
            raise RuntimeError("boom while reading trades")

    sink.write_portfolio(BrokenPortfolio(), window=1, cfg_hash="x")
    row = next(r for r in sink._manifest_rows if r["file"] == "w01_x_trades.csv.gz")
    assert row["kind"] == "trades_error"
    assert row["row_count"] == 0


def test_finalize_writes_manifest_and_oos_concat(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    sink.write_window(pd.DataFrame({"date": ["2011-01-03"], "units": [1.0]}), window=1)
    sink.write_window(pd.DataFrame({"date": ["2012-01-03"], "units": [2.0]}), window=2)
    manifest_path = sink.finalize(oos_windows=[1, 2])

    manifest = pd.read_csv(manifest_path)
    assert set(manifest["file"]) >= {"w01_trades.csv.gz", "w02_trades.csv.gz"}

    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 2
    assert list(oos["units"]) == [1.0, 2.0]


def test_zero_trade_window_writes_header_only_and_counts_zero(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    empty = pd.DataFrame(columns=["date", "pair", "units"])
    path = sink.write_window(empty, window=1)
    back = pd.read_csv(path)
    assert len(back) == 0
    assert list(back.columns) == ["date", "pair", "units"]
    manifest = pd.read_csv(sink.finalize())
    row = manifest[manifest["file"] == "w01_trades.csv.gz"].iloc[0]
    assert row["row_count"] == 0


def test_finalize_without_oos_windows_skips_concat(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    sink.write_window(pd.DataFrame({"units": [1.0]}), window=1)
    sink.finalize()
    assert (sink.run_dir / "manifest.csv").exists()
    assert not (sink.run_dir / "trades_oos.csv.gz").exists()
