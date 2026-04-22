"""
Phase B validation: verify RAMP's S&P 500 universe resolves cleanly against
live IBKR contracts.

Previously blocked by BRK.B / BF.B being rejected by IBKR (IB uses space
form -- `BRK B`, `BF B`). Commit 0fe4fb4 added normalization at the IBKR
boundary. This script is the empirical check that the fix works against
a real Gateway.

Usage:
    python scripts/trading/validate_ibkr_sp500_contracts.py            # full sweep
    python scripts/trading/validate_ibkr_sp500_contracts.py --quick    # just multi-class + sample
    python scripts/trading/validate_ibkr_sp500_contracts.py --symbols BRK.B BF.B AAPL

Exit codes:
    0  - all symbols resolved
    1  - one or more symbols failed
    2  - connection failure / setup error
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root on sys.path whether run from root or scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker  # noqa: E402
from src.trading.brokers.ibkr.symbols import to_ibkr_symbol  # noqa: E402
from src.utils.logger import logger  # noqa: E402


QUICK_SAMPLE = [
    # Multi-class: must be translated to IBKR space form
    "BRK.B",
    "BF.B",
    # Single-letter ticker
    "V",
    # Regular mega-caps
    "AAPL",
    "MSFT",
    "SPY",
    # Dotted-looking-but-not-multi-class: none in our universe, but just
    # confirm we're not mangling regular names.
    "GOOGL",
]


def load_universe(csv_path: Path) -> List[str]:
    """Load ticker column from a CSV of (_, name, ticker) rows."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        # Skip header if present
        first = next(reader, None)
        if first and first[-1].strip().upper() == "TICKER":
            pass  # header consumed
        else:
            # Not a header -- yield the first row too
            if first:
                yield first[-1].strip()
        for row in reader:
            if not row:
                continue
            yield row[-1].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Test only the multi-class + sample symbols (fast sanity check).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Explicit list of symbols to test (overrides --quick and universe).",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=REPO_ROOT / "config" / "universes" / "sp500-2025.csv",
        help="Universe CSV to load for full sweep.",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.quick:
        symbols = QUICK_SAMPLE
    else:
        symbols = list(load_universe(args.universe))

    if not symbols:
        logger.error("[validate] No symbols to test")
        return 2

    logger.info(f"[validate] Testing {len(symbols)} symbols against IBKR")

    broker = IBKRBroker()
    try:
        broker.start()
    except Exception as e:
        logger.error(f"[validate] Failed to connect to IB Gateway: {e}")
        return 2

    resolver = broker._resolver
    results = {"ok": [], "failed": []}
    t0 = time.monotonic()

    for i, sym in enumerate(symbols, 1):
        try:
            contract = resolver.resolve_stock(sym)
            expected_ib_form = to_ibkr_symbol(sym)
            ok = contract.conId > 0 and contract.symbol == expected_ib_form
            results["ok" if ok else "failed"].append((sym, contract))
            marker = "[+]" if ok else "[-]"
            logger.info(
                f"[validate] {marker} {i}/{len(symbols)} {sym} "
                f"-> ib_symbol={contract.symbol} conId={contract.conId}"
            )
        except Exception as e:
            results["failed"].append((sym, str(e)))
            logger.warning(f"[validate] [-] {i}/{len(symbols)} {sym} FAILED: {e}")

    elapsed = time.monotonic() - t0
    broker.stop()

    n_ok = len(results["ok"])
    n_fail = len(results["failed"])
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[validate] Summary: {n_ok}/{len(symbols)} resolved in {elapsed:.1f}s")
    if results["failed"]:
        logger.info(f"[validate] Failed symbols ({n_fail}):")
        for sym, detail in results["failed"]:
            logger.info(f"  - {sym}: {detail}")
    logger.info("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
