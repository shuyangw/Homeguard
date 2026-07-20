"""Run-scoped fill logging sink.

Every simulated backtest run persists its fills here: per-window and
per-config, gzipped, under output/backtests/<strategy>/runs/<run_id>/.
See docs/superpowers/specs/2026-07-20-fill-logging-everywhere-design.md.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils import logger


class FillSink:
    def __init__(self, strategy: str, run_id: str, meta: dict,
                 root: Path = Path("output/backtests")):
        self.strategy = strategy
        self.run_id = run_id
        self.run_dir = Path(root) / strategy / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_rows: list[dict[str, Any]] = []
        full_meta = {"strategy": strategy, "run_id": run_id, **meta}
        (self.run_dir / "meta.json").write_text(json.dumps(full_meta, indent=2, default=str))

    @staticmethod
    def make_run_id(cfg_hash: str, now: datetime) -> str:
        return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash}"
