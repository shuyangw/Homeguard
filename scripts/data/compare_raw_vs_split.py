"""Cross-feed consistency check: equities_1min_sip_raw vs equities_1min_sip_split.

For each symbol in both passes, load matching (year, month) partitions and verify:
  - Identical timestamp series
  - Identical trade_count series
(Price columns and volume legitimately differ across adjustments.)

Output: output/data_validation/raw_vs_split_diff.csv

Usage:
    python scripts/data/compare_raw_vs_split.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.settings import get_local_storage_dir, get_output_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compare_symbol(base_dir: Path, symbol: str) -> list[dict]:
    """Compare one symbol's raw vs split partitions. Returns diff rows."""
    raw_dir = base_dir / "equities_1min_sip_raw" / f"symbol={symbol}"
    split_dir = base_dir / "equities_1min_sip_split" / f"symbol={symbol}"

    raw_parts = {
        p.relative_to(raw_dir): p
        for p in raw_dir.glob("year=*/month=*/data.parquet")
    }
    split_parts = {
        p.relative_to(split_dir): p
        for p in split_dir.glob("year=*/month=*/data.parquet")
    }

    diffs: list[dict] = []
    only_raw = set(raw_parts) - set(split_parts)
    only_split = set(split_parts) - set(raw_parts)
    for part in sorted(only_raw):
        diffs.append({
            "symbol": symbol, "partition": str(part),
            "issue": "missing_in_split",
        })
    for part in sorted(only_split):
        diffs.append({
            "symbol": symbol, "partition": str(part),
            "issue": "missing_in_raw",
        })

    for part in sorted(set(raw_parts) & set(split_parts)):
        raw_df = pd.read_parquet(
            raw_parts[part], columns=["timestamp", "trade_count"]
        )
        split_df = pd.read_parquet(
            split_parts[part], columns=["timestamp", "trade_count"]
        )
        if len(raw_df) != len(split_df):
            diffs.append({
                "symbol": symbol, "partition": str(part),
                "issue": f"row_count_mismatch:{len(raw_df)}_vs_{len(split_df)}",
            })
            continue
        if not raw_df["timestamp"].equals(split_df["timestamp"]):
            diffs.append({
                "symbol": symbol, "partition": str(part),
                "issue": "timestamp_mismatch",
            })
            continue
        if not raw_df["trade_count"].equals(split_df["trade_count"]):
            diffs.append({
                "symbol": symbol, "partition": str(part),
                "issue": "trade_count_mismatch",
            })

    return diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revalidate", action="store_true",
        help="Re-check symbols even if already in diff CSV",
    )
    args = parser.parse_args(argv)

    base_dir = get_local_storage_dir()
    raw_dir = base_dir / "equities_1min_sip_raw"
    split_dir = base_dir / "equities_1min_sip_split"
    if not (raw_dir.exists() and split_dir.exists()):
        logger.error("Both raw and split pass folders must exist")
        return 1

    raw_syms = {p.name.replace("symbol=", "") for p in raw_dir.glob("symbol=*")}
    split_syms = {p.name.replace("symbol=", "") for p in split_dir.glob("symbol=*")}
    common = sorted(raw_syms & split_syms)
    logger.info(
        f"Comparing {len(common)} common symbols "
        f"(raw_only={len(raw_syms - split_syms)}, "
        f"split_only={len(split_syms - raw_syms)})"
    )

    out_path = get_output_dir() / "data_validation" / "raw_vs_split_diff.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_checked: set = set()
    if out_path.exists() and not args.revalidate:
        already_checked = set(pd.read_csv(out_path)["symbol"].astype(str))

    all_diffs: list[dict] = []
    for i, symbol in enumerate(common, 1):
        if symbol in already_checked:
            continue
        diffs = compare_symbol(base_dir, symbol)
        all_diffs.extend(diffs)
        if i % 500 == 0:
            logger.info(f"  ...compared {i}/{len(common)}")

    if all_diffs:
        df = pd.DataFrame(all_diffs)
        df.to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False,
        )
        logger.warning(f"{len(all_diffs)} diff rows -> {out_path}")
    else:
        logger.info("No diffs detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
