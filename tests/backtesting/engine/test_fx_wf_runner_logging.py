import pandas as pd
from datetime import datetime, timezone
from src.backtesting.engine.fill_sink import FillSink


def test_wf_loop_produces_oos_concat(tmp_path, monkeypatch):
    # Simulate the runner loop contract: N windows -> N gz + trades_oos.csv.gz
    sink = FillSink("FxWFDemo", FillSink.make_run_id(
        "cfg", datetime(2026, 7, 20, tzinfo=timezone.utc)), {"kind": "walkforward"},
        root=tmp_path)
    for w in (1, 2, 3):
        sink.write_window(pd.DataFrame({"date": [f"201{w}-01-03"], "units": [float(w)]}), window=w)
    sink.finalize(oos_windows=[1, 2, 3])
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 3
    assert (sink.run_dir / "manifest.csv").exists()
