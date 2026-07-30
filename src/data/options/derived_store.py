"""Materialized store for the options DERIVED DAILY tables.

These tables (`atm_iv_daily`, `skew_daily`, `term_slope_daily`, `iv_rank_daily`,
`rv_daily`, ...) were originally written to `output/wave0/derived/` -- a Wave-0
scratch path inside the gitignored `output/` tree. Phase 2 must not depend on
scratch, so they live here instead, following the convention `spread_census`
already uses:

    <storage>/options/derived/<table>/<table>.parquet
    <storage>/options/derived/<table>/<table>.meta.json

The sidecar mirrors the provenance shape `regime_state_daily` and `vix_spot`
already emit: source, snapshot timestamp, build git SHA, row counts, coverage
span (overall and per root).

DATA-PROPERTY MEASUREMENT ONLY. No positions, no P&L.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

DERIVED_DIR_NAME = "derived"


def derived_root() -> Path:
    return get_local_storage_dir() / "options" / DERIVED_DIR_NAME


def derived_table_path(name: str) -> Path:
    return derived_root() / name / f"{name}.parquet"


def derived_meta_path(name: str) -> Path:
    return derived_table_path(name).with_suffix(".meta.json")


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(Path(__file__).resolve().parents[3]),
        text=True,
    ).strip()


def _span(dates: pd.Series) -> tuple:
    d = pd.to_datetime(dates)
    return str(d.min().date()), str(d.max().date())


def _coverage(df: pd.DataFrame) -> Dict[str, Any]:
    date_min, date_max = _span(df["session_date"])
    cov: Dict[str, Any] = {
        "rows": int(len(df)),
        "date_min": date_min,
        "date_max": date_max,
        "roots": [],
        "rows_by_root": {},
        "coverage_by_root": {},
    }
    if "root" not in df.columns:
        return cov
    for root, g in df.groupby("root"):
        lo, hi = _span(g["session_date"])
        cov["roots"].append(str(root))
        cov["rows_by_root"][str(root)] = int(len(g))
        cov["coverage_by_root"][str(root)] = {
            "date_min": lo, "date_max": hi, "rows": int(len(g))}
    cov["roots"] = sorted(cov["roots"])
    return cov


def write_derived_table(name: str, df: pd.DataFrame, source: str,
                        extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write one derived daily table plus its provenance sidecar."""
    path = derived_table_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    meta: Dict[str, Any] = {
        "dataset": name,
        "source": source,
        "snapshot_timestamp": tz.now().isoformat(),
        "git_sha": _git_sha(),
        "columns": [str(c) for c in df.columns],
    }
    meta.update(_coverage(df))
    if extra:
        meta.update(extra)
    derived_meta_path(name).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        f"[+] {name}: {len(df):,} rows, roots={meta['roots']}, "
        f"{meta['date_min']} .. {meta['date_max']} -> {path}"
    )
    return path


def load_derived_table(name: str) -> pd.DataFrame:
    path = derived_table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"[-] no derived table at {path}")
    return pd.read_parquet(path)


def load_derived_meta(name: str) -> Dict[str, Any]:
    path = derived_meta_path(name)
    if not path.exists():
        raise FileNotFoundError(f"[-] no provenance sidecar at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
