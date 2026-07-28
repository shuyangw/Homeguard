"""V12 -- Legacy store disposition (options_1min vs options_combined).

Registered gate: "`options/options_1min/` (17 roots, 2024-11 -> 2025-12) vs
`options_combined` overlap equivalence. Gate: Report equivalence.
DELETE NOTHING."

This script DELETES NOTHING. It only lists and measures.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/v12")


def _tree_stats(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    n_files = n_dirs = 0
    total = 0
    for f in p.rglob("*"):
        if f.is_dir():
            n_dirs += 1
        else:
            n_files += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return {"path": str(p), "exists": True, "n_files": n_files,
            "n_subdirs": n_dirs, "total_bytes": total}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opt = Path(get_local_storage_dir()) / "options"

    entries = []
    for child in sorted(opt.iterdir()):
        entries.append({"name": child.name, "is_dir": child.is_dir(),
                        **_tree_stats(child)})
        logger.info(f"[*] {child.name}: {entries[-1]}")

    legacy_candidates = {
        "options_1min": opt / "options_1min",
        "options_eod": opt / "options_eod",
        "chains": opt / "chains",
        "gex_daily": opt / "gex_daily",
        "options_combined": opt / "options_combined",
    }
    # also check the storage root and any thetadata/ nesting mentioned in settings
    extra = {
        "storage_root_options_1min": Path(get_local_storage_dir()) / "options_1min",
        "thetadata_options_1min": opt / "thetadata" / "options_1min",
    }

    checks = {k: _tree_stats(v) for k, v in {**legacy_candidates, **extra}.items()}

    result = {
        "options_dir": str(opt),
        "top_level_listing": entries,
        "checks": checks,
        "options_1min_exists": checks["options_1min"]["exists"]
                               or checks["storage_root_options_1min"]["exists"]
                               or checks["thetadata_options_1min"]["exists"],
    }
    result["verdict"] = "DIVERGENCE_LEGACY_STORE_PRESENT" if result["options_1min_exists"] else "MOOT_CLOSED"

    (OUT_DIR / "v12_listing.json").write_text(json.dumps(result, indent=2, default=str), encoding="ascii")
    pd.DataFrame(entries).to_csv(OUT_DIR / "v12_top_level_listing.csv", index=False)
    logger.info(f"[+] V12 verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
