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
        self._manifest_path = self.run_dir / "manifest_rows.jsonl"
        self._oos_ranges: dict[int, tuple] = {}
        full_meta = {"strategy": strategy, "run_id": run_id, **meta}
        (self.run_dir / "meta.json").write_text(json.dumps(full_meta, indent=2, default=str))

    @staticmethod
    def make_run_id(cfg_hash: str, now: datetime) -> str:
        return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash}"

    def _record(self, row):
        self._manifest_rows.append(row)
        with open(self._manifest_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _stem(self, window: int, cfg_hash: Optional[str]) -> str:
        return f"w{window:02d}" + (f"_{cfg_hash}" if cfg_hash else "")

    def set_oos_range(self, window, start, end):
        self._oos_ranges[window] = (pd.Timestamp(start), pd.Timestamp(end))

    def write_window(self, trades_df: pd.DataFrame, window: int,
                     cfg_hash: Optional[str] = None,
                     extras: Optional[dict[str, pd.DataFrame]] = None) -> Path:
        stem = self._stem(window, cfg_hash)
        path = self.run_dir / f"{stem}_trades.csv.gz"
        trades_df.to_csv(path, index=False, compression="gzip")
        self._record({
            "file": path.name, "kind": "trades", "window": window,
            "cfg_hash": cfg_hash or "", "row_count": len(trades_df),
        })
        for name, extra_df in (extras or {}).items():
            epath = self.run_dir / f"{stem}_{name}.csv.gz"
            extra_df.to_csv(epath, index=False, compression="gzip")
            self._record({
                "file": epath.name, "kind": name, "window": window,
                "cfg_hash": cfg_hash or "", "row_count": len(extra_df),
            })
        return path

    def write_portfolio(self, portfolio: Any, window: int,
                        cfg_hash: Optional[str] = None, symbol: str = "") -> Path:
        from src.backtesting.engine.trade_logger import TradeLogger
        stem = self._stem(window, cfg_hash)
        path = self.run_dir / f"{stem}_trades.csv.gz"
        TradeLogger.export_trades_csv(portfolio, path, symbol=symbol)
        kind = "trades"
        row_count = 0
        if path.exists():
            try:
                df = pd.read_csv(path)
                if list(df.columns) == ["Error"]:
                    kind = "trades_error"
                    row_count = 0
                    logger.warning(
                        f"TradeLogger export failed for strategy={self.strategy} "
                        f"window={window} cfg_hash={cfg_hash or ''}; "
                        f"recording manifest kind=trades_error, row_count=0"
                    )
                else:
                    row_count = len(df)
            except Exception:
                row_count = 0
        self._record({
            "file": path.name, "kind": kind, "window": window,
            "cfg_hash": cfg_hash or "", "row_count": row_count,
        })
        return path

    def finalize(self, oos_windows=None, oos_cfg_hash=None):
        if oos_windows:
            suffix = f"_{oos_cfg_hash}" if oos_cfg_hash else ""
            global_max_end = max((e for (_, e) in self._oos_ranges.values()), default=None)
            frames = []
            for w in sorted(oos_windows):
                wpath = self.run_dir / f"w{w:02d}{suffix}_trades.csv.gz"
                if not wpath.exists():
                    continue
                df = pd.read_csv(wpath)
                rng = self._oos_ranges.get(w)
                if rng is not None and "date" in df.columns:
                    lo, hi = rng
                    d = pd.to_datetime(df["date"])
                    if global_max_end is not None and hi == global_max_end:
                        df = df[(d >= lo) & (d <= hi)]
                    else:
                        df = df[(d >= lo) & (d < hi)]
                frames.append(df)
            if frames:
                oos = pd.concat(frames, ignore_index=True)
                oos.to_csv(self.run_dir / "trades_oos.csv.gz", index=False,
                           compression="gzip")
                self._record({"file": "trades_oos.csv.gz", "kind": "oos_concat",
                              "window": -1, "cfg_hash": "", "row_count": len(oos)})
        rows = []
        if self._manifest_path.exists():
            for line in self._manifest_path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            rows = list(self._manifest_rows)
        # dedup by file, keep last (idempotent re-runs / duplicate appends)
        seen = {}
        for r in rows:
            seen[r["file"]] = r
        rows = list(seen.values())
        manifest_path = self.run_dir / "manifest.csv"
        pd.DataFrame(rows, columns=["file", "kind", "window", "cfg_hash", "row_count"]
                     ).to_csv(manifest_path, index=False)
        logger.info(f"[fill_sink] finalized run {self.run_id}: {len(rows)} artifacts in {self.run_dir}")
        return manifest_path
