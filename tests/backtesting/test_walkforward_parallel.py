import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.futures.paths import continuous_1min_dir, carry_dir

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists() and (carry_dir() / "GC.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures/carry store not present")


def _run(tmp_path, jobs):
    out = tmp_path / f"metrics_{jobs}.json"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    cfg = REPO / "config/backtesting/carver_tsmom.yaml"  # tiny 3-root config
    subprocess.run(
        [PY, "scripts/backtest_scripts/run_carver_walkforward.py",
         "--config", str(cfg), "--report", str(tmp_path / f"r{jobs}.md"),
         "--jobs", str(jobs), "--json", str(out),
         "--train-months", "12", "--test-months", "6", "--step-months", "6"],
        cwd=str(REPO), env=env, check=True, capture_output=True, text=True, timeout=1200)
    return json.loads(out.read_text())


def test_parallel_equals_serial(tmp_path):
    serial = _run(tmp_path, 1)
    par = _run(tmp_path, 2)
    for k in ("oos_sharpe", "psr", "dsr", "pbo", "oos_sharpe_1_5x_cost", "n_windows"):
        assert serial[k] == par[k], f"{k}: serial={serial[k]} parallel={par[k]}"
