import pandas as pd
from datetime import datetime, timezone
from src.backtesting.engine.fill_sink import FillSink


def test_wf_loop_logs_every_cost_leg(tmp_path):
    # Multi-leg contract: each window logs BOTH cost legs (c1x, c15x) to its own
    # tagged file; the OOS concat stitches ONLY the base (c1x) leg.
    sink = FillSink("FxWFDemo", FillSink.make_run_id(
        "cfg", datetime(2026, 7, 20, tzinfo=timezone.utc)), {"kind": "walkforward"},
        root=tmp_path)
    legs = {"c1x": 1, "c15x": 2}  # rows written per window, per leg
    for w in (1, 2, 3):
        for tag, n_rows in legs.items():
            df = pd.DataFrame({"date": [f"201{w}-01-0{r + 1}" for r in range(n_rows)],
                               "units": [float(w)] * n_rows})
            sink.write_window(df, window=w, cfg_hash=tag)
    sink.finalize(oos_windows=[1, 2, 3], oos_cfg_hash="c1x")

    for w in (1, 2, 3):
        assert (sink.run_dir / f"w{w:02d}_c1x_trades.csv.gz").exists()
        assert (sink.run_dir / f"w{w:02d}_c15x_trades.csv.gz").exists()

    # OOS concat uses ONLY the c1x leg: 3 windows x 1 c1x row each == 3 rows.
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 3

    manifest = pd.read_csv(sink.run_dir / "manifest.csv")
    files = set(manifest["file"])
    for w in (1, 2, 3):
        assert f"w{w:02d}_c1x_trades.csv.gz" in files
        assert f"w{w:02d}_c15x_trades.csv.gz" in files
