"""
Fix failed options download manifest entries.

Marks FAILED entries as NO_DATA when the failure is due to:
- Pre-IPO dates (data doesn't exist yet)
- Ticker changes (data under different symbol)
- Known data gaps

Usage:
    # Dry run (preview changes)
    python scripts/fix_failed_options_manifest.py

    # Apply changes
    python scripts/fix_failed_options_manifest.py --apply

    # Also add FB entries for META historical data
    python scripts/fix_failed_options_manifest.py --apply --add-fb
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

from src.settings import get_options_data_dir


# IPO/listing dates - data doesn't exist before these dates
SYMBOL_DATA_START_DATES = {
    "IBIT": "2024-01-01",   # iShares Bitcoin Trust ETF launched Jan 2024
    "COIN": "2021-04-01",   # Coinbase IPO'd April 14, 2021
    "PLTR": "2020-10-01",   # Palantir IPO'd Sept 30, 2020 (options Oct)
    "META": "2021-11-01",   # Ticker changed from FB to META Oct 28, 2021
}

# Ticker changes: old_ticker -> (new_ticker, change_date)
# For META: data before change date is under FB
TICKER_CHANGES = {
    "META": ("FB", "2021-10-28"),  # META data before this is under FB
}


def load_manifest(manifest_path: Path) -> Dict:
    """Load manifest from JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: Dict) -> None:
    """Save manifest to JSON file with backup."""
    # Create backup
    backup_path = manifest_path.with_suffix(".json.bak")
    shutil.copy(manifest_path, backup_path)
    print(f"[+] Created backup: {backup_path}")

    # Save updated manifest
    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.move(str(temp_path), str(manifest_path))
    print(f"[+] Saved updated manifest: {manifest_path}")


def is_pre_data_date(symbol: str, year: int, month: int) -> Tuple[bool, str]:
    """
    Check if a symbol/date combination is before data availability.

    Returns:
        (is_pre_data, reason)
    """
    if symbol not in SYMBOL_DATA_START_DATES:
        return False, ""

    data_start = datetime.strptime(SYMBOL_DATA_START_DATES[symbol], "%Y-%m-%d")
    entry_date = datetime(year, month, 1)

    if entry_date < data_start:
        if symbol in TICKER_CHANGES:
            old_ticker, change_date = TICKER_CHANGES[symbol]
            return True, f"Pre-ticker-change (was {old_ticker} until {change_date})"
        else:
            return True, f"Pre-IPO/pre-listing (data starts {SYMBOL_DATA_START_DATES[symbol]})"

    return False, ""


def get_fb_entries_needed(manifest: Dict) -> List[Dict]:
    """
    Get list of FB entries needed to cover META's pre-ticker-change period.

    Returns list of entry dicts for FB symbol.
    """
    entries = manifest.get("entries", {})
    fb_entries = []

    for key, entry in entries.items():
        if entry.get("symbol") != "META":
            continue

        year = entry.get("year")
        month = entry.get("month")

        # Check if this is before ticker change
        is_pre, _ = is_pre_data_date("META", year, month)
        if is_pre:
            fb_key = f"FB_{year}_{month:02d}"
            # Only add if FB entry doesn't already exist
            if fb_key not in entries:
                fb_entries.append({
                    "symbol": "FB",
                    "year": year,
                    "month": month,
                    "state": "pending",
                    "rows_count": 0,
                    "error_message": None,
                    "completed_at": None,
                })

    return fb_entries


def main():
    parser = argparse.ArgumentParser(description="Fix failed options manifest entries")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--add-fb", action="store_true",
                        help="Add FB entries for META's pre-ticker-change period")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest file (default: auto-detect)")
    args = parser.parse_args()

    # Find manifest
    options_dir = get_options_data_dir()
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = options_dir / "_manifest_combined.json"

    if not manifest_path.exists():
        print(f"[-] Manifest not found: {manifest_path}")
        return 1

    print(f"Manifest: {manifest_path}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    # Load manifest
    manifest = load_manifest(manifest_path)
    entries = manifest.get("entries", {})

    # Find failed entries that should be NO_DATA
    changes = []
    for key, entry in entries.items():
        if entry.get("state") != "failed":
            continue

        symbol = entry.get("symbol")
        year = entry.get("year")
        month = entry.get("month")

        is_pre, reason = is_pre_data_date(symbol, year, month)
        if is_pre:
            changes.append((key, reason))

    # Report findings
    print("=" * 60)
    print("FAILED ENTRIES TO MARK AS NO_DATA")
    print("=" * 60)

    if not changes:
        print("No entries to change.")
    else:
        # Group by symbol for cleaner output
        by_symbol = {}
        for key, reason in changes:
            symbol = key.split("_")[0]
            by_symbol.setdefault(symbol, []).append((key, reason))

        for symbol, items in sorted(by_symbol.items()):
            print(f"\n{symbol} ({len(items)} entries): {items[0][1]}")
            keys = [k for k, _ in items]
            # Show first few and last few
            if len(keys) <= 6:
                for k in keys:
                    print(f"  {k}")
            else:
                for k in keys[:3]:
                    print(f"  {k}")
                print(f"  ... ({len(keys) - 6} more)")
                for k in keys[-3:]:
                    print(f"  {k}")

    # Handle FB entries
    fb_entries = []
    if args.add_fb:
        fb_entries = get_fb_entries_needed(manifest)
        print()
        print("=" * 60)
        print("FB ENTRIES TO ADD (for META historical data)")
        print("=" * 60)

        if not fb_entries:
            print("No FB entries needed (already exist or no META pre-change data).")
        else:
            print(f"\n{len(fb_entries)} FB entries to add:")
            for entry in fb_entries[:5]:
                print(f"  FB_{entry['year']}_{entry['month']:02d}")
            if len(fb_entries) > 5:
                print(f"  ... and {len(fb_entries) - 5} more")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Entries to mark as NO_DATA: {len(changes)}")
    print(f"FB entries to add: {len(fb_entries)}")

    if not args.apply:
        print()
        print("This was a DRY RUN. Use --apply to make changes.")
        return 0

    # Apply changes
    if changes or fb_entries:
        print()
        print("Applying changes...")

        # Mark entries as NO_DATA
        for key, reason in changes:
            entries[key]["state"] = "no_data"
            entries[key]["error_message"] = reason

        # Add FB entries
        for entry in fb_entries:
            key = f"FB_{entry['year']}_{entry['month']:02d}"
            entries[key] = entry

        # Update manifest
        manifest["updated_at"] = datetime.now().isoformat()

        # Save
        save_manifest(manifest_path, manifest)

        print()
        print(f"[+] Marked {len(changes)} entries as NO_DATA")
        print(f"[+] Added {len(fb_entries)} FB entries")
        print()
        print("Done! You can now re-run the downloader to fetch FB data:")
        print("  python scripts/download_options_combined.py --symbols FB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
