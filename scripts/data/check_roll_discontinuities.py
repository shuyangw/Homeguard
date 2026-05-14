"""Scan futures continuous-contract data for roll-induced price discontinuities.

Continuous futures series (`*.c.0` from Databento GLBX.MDP3 in
`futures_1min/`) are UNADJUSTED: when the front month rolls, the new
contract's price replaces the old as if it were a price move. Computing
`close.pct_change()` across a roll boundary produces fake P&L events on
the order of 5-25%.

This script is the loud canary: it scans every futures contract on disk,
flags every close-to-close jump exceeding the threshold, and prints them
with date and magnitude. Anyone about to backtest futures sees the
discontinuities upfront rather than discovering them in corrupted metrics.

Usage:
    python scripts/data/check_roll_discontinuities.py
    python scripts/data/check_roll_discontinuities.py --threshold 0.10
    python scripts/data/check_roll_discontinuities.py --symbols ES,NQ

When a real futures backtest is queued, the fix is to build
`src/backtesting/utils/futures_roll.py` with proper return-splicing
(return per contract, never `pct_change()` across the roll boundary).
This script does NOT fix the data; it documents the problem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from src.settings import get_local_storage_dir

DEFAULT_THRESHOLD = 0.05
DEFAULT_SYMBOLS = ["ES", "NQ", "YM", "RTY", "CL", "GC", "ZN", "6E", "ZC"]


def scan_contract(storage_root: Path, symbol: str, threshold: float) -> list[dict]:
    """Return rows where |close.pct_change()| > threshold for one symbol."""
    glob = str(storage_root / "futures_1min" / f"symbol={symbol}" / "**" / "*.parquet").replace("\\", "/")
    # Daily resample first -- minute-bar jumps within a session are real
    # volatility; we only care about close-to-close at the day boundary,
    # which is where roll discontinuities sit.
    sql = f"""
        WITH daily AS (
            SELECT
                DATE_TRUNC('day', timestamp) AS dt,
                LAST(close ORDER BY timestamp) AS close
            FROM read_parquet('{glob}')
            GROUP BY dt
        ),
        diffed AS (
            SELECT
                dt,
                close,
                LAG(close) OVER (ORDER BY dt) AS prev_close
            FROM daily
        )
        SELECT
            dt,
            prev_close,
            close,
            (close - prev_close) / prev_close AS pct_change
        FROM diffed
        WHERE prev_close IS NOT NULL
          AND ABS((close - prev_close) / prev_close) > {threshold}
        ORDER BY ABS((close - prev_close) / prev_close) DESC
    """
    try:
        rows = duckdb.query(sql).fetchall()
    except duckdb.Error as e:
        print(f"  [ERR] {symbol}: {e}", file=sys.stderr)
        return []
    return [
        {"date": r[0].date(), "prev_close": float(r[1]), "close": float(r[2]), "pct": float(r[3])}
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Flag close-to-close moves above this fraction (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                    help="Comma-separated contract list")
    args = ap.parse_args()

    storage_root = Path(get_local_storage_dir())
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"Storage root:  {storage_root}")
    print(f"Threshold:     {args.threshold:.1%}")
    print(f"Contracts:     {', '.join(symbols)}")
    print()

    total = 0
    for sym in symbols:
        rows = scan_contract(storage_root, sym, args.threshold)
        print(f"=== {sym}  ({len(rows)} discontinuit{'y' if len(rows) == 1 else 'ies'}) ===")
        for r in rows:
            print(f"  {r['date']}  {r['prev_close']:>10.2f} -> {r['close']:>10.2f}  ({r['pct']:+.2%})")
        if not rows:
            print("  (no jumps above threshold)")
        print()
        total += len(rows)

    print(f"Total discontinuities across {len(symbols)} contracts: {total}")
    print()
    print("These are CONTRACT-ROLL boundaries, not real price moves. Any backtest")
    print("computing returns via close.pct_change() across these dates produces")
    print("fake P&L. Fix: per-contract return computation OR back-adjustment.")
    print("See docs/methodology/backtesting.md Section 4.2 (Futures roll costs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
