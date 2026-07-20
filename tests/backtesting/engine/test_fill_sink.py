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
