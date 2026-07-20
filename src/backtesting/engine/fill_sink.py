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

    def _stem(self, window: int, cfg_hash: Optional[str]) -> str:
        return f"w{window:02d}" + (f"_{cfg_hash}" if cfg_hash else "")

    def write_window(self, trades_df: pd.DataFrame, window: int,
                     cfg_hash: Optional[str] = None,
                     extras: Optional[dict[str, pd.DataFrame]] = None) -> Path:
        stem = self._stem(window, cfg_hash)
        path = self.run_dir / f"{stem}_trades.csv.gz"
        trades_df.to_csv(path, index=False, compression="gzip")
        self._manifest_rows.append({
            "file": path.name, "kind": "trades", "window": window,
            "cfg_hash": cfg_hash or "", "row_count": len(trades_df),
        })
        for name, extra_df in (extras or {}).items():
            epath = self.run_dir / f"{stem}_{name}.csv.gz"
            extra_df.to_csv(epath, index=False, compression="gzip")
            self._manifest_rows.append({
                "file": epath.name, "kind": name, "window": window,
                "cfg_hash": cfg_hash or "", "row_count": len(extra_df),
            })
        return path
