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
