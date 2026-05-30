"""Migrate H:\\Stock_Data from flat directory layout to asset-class layout.

OLD (flat):
    equities_1min/
    equities_1min_sip_raw/
    equities_1min_sip_split/
    crypto_1min/
    futures_1min/
    fx_1min/
    news/
    ...
    _manifests/
        equities_1min.json
        crypto_1min.json
        ...

NEW (asset-class):
    equities/iex/1min/
    equities/sip_raw/1min/
    equities/sip_split/1min/
    crypto/alpaca/1min/
    futures/databento/1min/
    fx/massive/1min/
    news/alpaca/
    ...
    _manifests/
        equities_iex_1min.json
        crypto_alpaca_1min.json
        ...

Operations:
  - NTFS directory rename (O(1) within same volume)
  - Manifest filename flatten: subdir slashes -> underscores
  - Both progress.jsonl + status.csv companion files moved along with the .json

Idempotent: re-runs detect already-migrated state and skip.

Usage:
    python scripts/data/migrate_to_asset_class_layout.py --dry-run
    python scripts/data/migrate_to_asset_class_layout.py --execute
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.settings import LEGACY_TO_CANONICAL, get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plan_data_moves(base: Path) -> list[tuple[Path, Path]]:
    """Return list of (src, dst) directory moves needed."""
    moves: list[tuple[Path, Path]] = []
    for legacy, canonical in LEGACY_TO_CANONICAL.items():
        src = base / legacy
        dst = base / canonical
        if not src.exists():
            continue
        if dst.exists():
            logger.warning(
                f"SKIP {legacy}: destination {canonical} already exists "
                f"(already migrated or conflict)"
            )
            continue
        moves.append((src, dst))
    return moves


def plan_manifest_renames(base: Path) -> list[tuple[Path, Path]]:
    """Return list of (src, dst) manifest file renames needed.

    For each (legacy, canonical) pair, also handle the .progress.jsonl
    and .status.csv companion files.
    """
    renames: list[tuple[Path, Path]] = []
    manifests_dir = base / "_manifests"
    if not manifests_dir.exists():
        return renames

    for legacy, canonical in LEGACY_TO_CANONICAL.items():
        canonical_flat = canonical.replace("/", "_")
        # Match the three known companion files: .json, .progress.jsonl, .status.csv
        suffixes = [".json", ".progress.jsonl", ".status.csv"]
        for suffix in suffixes:
            src = manifests_dir / f"{legacy}{suffix}"
            dst = manifests_dir / f"{canonical_flat}{suffix}"
            if not src.exists():
                continue
            if dst.exists():
                logger.warning(
                    f"SKIP {src.name}: destination {dst.name} already exists"
                )
                continue
            renames.append((src, dst))
    return renames


def _is_descendant(child: Path, parent: Path) -> bool:
    """Check whether child path lies under parent path."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def execute_move(src: Path, dst: Path, dry_run: bool) -> bool:
    """Perform NTFS directory rename. Returns True on success or dry-run.

    Handles the special case where destination is INSIDE the source
    directory (e.g., 'news' -> 'news/alpaca') by staging through a
    sibling temp name.
    """
    logger.info(f"MOVE  {src.relative_to(get_local_storage_dir())} "
                f"-> {dst.relative_to(get_local_storage_dir())}")
    if dry_run:
        return True
    try:
        if _is_descendant(dst, src):
            # Stage: src -> sibling tmp, then mkdir new parent, then tmp -> final dst
            tmp = src.with_name(src.name + ".__migration_tmp__")
            os.rename(src, tmp)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(tmp, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
        return True
    except OSError as e:
        logger.error(f"FAILED move {src} -> {dst}: {e}")
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate H:\\Stock_Data to asset-class directory layout",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned moves without executing"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform the moves (required for real run)"
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.execute:
        parser.error("Specify --dry-run or --execute")

    base = get_local_storage_dir()
    logger.info(f"Storage base: {base}")
    if args.dry_run:
        logger.info("(DRY RUN -- no changes will be made)")
    else:
        logger.info("(EXECUTING -- files will be moved)")

    data_moves = plan_data_moves(base)
    manifest_renames = plan_manifest_renames(base)

    logger.info(f"Planned data moves: {len(data_moves)}")
    logger.info(f"Planned manifest renames: {len(manifest_renames)}")

    succeeded, failed = 0, 0

    for src, dst in data_moves:
        if execute_move(src, dst, args.dry_run):
            succeeded += 1
        else:
            failed += 1

    for src, dst in manifest_renames:
        if execute_move(src, dst, args.dry_run):
            succeeded += 1
        else:
            failed += 1

    logger.info(f"Done. {succeeded} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
