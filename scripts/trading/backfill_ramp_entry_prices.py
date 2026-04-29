"""One-shot: backfill state-manager entry_price for RAMP positions from broker.

Why this exists
---------------
Before commit 03c7a21, RAMPLiveAdapter._execute_rebalance recorded the SIZING
QUOTE (the price returned by `broker.get_latest_quote(symbol)` at the moment of
sizing) as the position's `entry_price` in `data/trading/strategy_positions.json`,
not the actual fill price returned by the broker. For yesterday's CHTR fill,
that meant state-manager stored $172.34 (sizing quote) instead of $174.45
(actual fill). The drift is small but compounds into incorrect realized-PnL
numbers when positions close, and into wrong avg_entry display anywhere the
state file is consumed.

Commit 03c7a21 fixed all FUTURE writes. This script backfills the existing
positions from the broker's `avg_entry_price`, which IBKR populates from the
actual fill (not from a sizing quote).

Usage
-----
    # Dry-run (default) -- prints the diff, makes no changes:
    python scripts/trading/backfill_ramp_entry_prices.py

    # Apply:
    python scripts/trading/backfill_ramp_entry_prices.py --apply

The script is idempotent: on an already-correct file it will print "No drift
detected" and exit 0.

Run on EC2 (where the live state file lives) -- not locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the corrected entry_price values. Default is dry-run.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Override path to strategy_positions.json. Defaults to repo standard.",
    )
    parser.add_argument(
        "--strategy",
        default="ramp",
        help="Strategy key in the state file. Defaults to 'ramp'.",
    )
    args = parser.parse_args()

    # Connect to broker
    print(f"[backfill] Connecting to IBKR (clientId=99) to read live avg_entry_price...")
    try:
        from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker
        from src.trading.brokers.ibkr.config import IBKRConfig
    except ImportError as e:
        print(f"[backfill] FATAL: could not import IBKR broker: {e}")
        return 2

    broker = IBKRBroker(IBKRConfig(host="127.0.0.1", port=4002, client_id=99))
    try:
        broker.start()
        broker_positions = broker.get_stock_positions() or []
    except Exception as e:
        print(f"[backfill] FATAL: broker connect/positions failed: {e}")
        return 2
    finally:
        try:
            broker.stop()
        except Exception:
            pass

    broker_avg_by_symbol: Dict[str, float] = {}
    for p in broker_positions:
        sym = p.get("symbol")
        avg = p.get("avg_entry_price")
        if sym and avg is not None and avg > 0:
            broker_avg_by_symbol[sym] = float(avg)

    if not broker_avg_by_symbol:
        print("[backfill] Broker reported zero stock positions. Nothing to backfill.")
        return 0
    print(f"[backfill] Broker reports {len(broker_avg_by_symbol)} stock positions.")

    # Locate state file
    if args.state_file:
        state_path = Path(args.state_file)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        state_path = repo_root / "data" / "trading" / "strategy_positions.json"

    if not state_path.exists():
        print(f"[backfill] FATAL: state file not found at {state_path}")
        return 2

    with open(state_path) as f:
        state = json.load(f)

    strat = state.get("strategies", {}).get(args.strategy)
    if not strat or not isinstance(strat.get("positions"), dict):
        print(f"[backfill] No positions for strategy '{args.strategy}' in {state_path}")
        return 0

    positions = strat["positions"]
    if not positions:
        print(f"[backfill] Strategy '{args.strategy}' has no recorded positions.")
        return 0

    # Compute diff
    drift_rows = []
    for sym, pos in positions.items():
        state_price = float(pos.get("entry_price") or 0)
        broker_price = broker_avg_by_symbol.get(sym)
        if broker_price is None:
            drift_rows.append((sym, state_price, None, "BROKER MISSING"))
            continue
        delta = broker_price - state_price
        if abs(delta) < 0.001:  # within rounding noise
            drift_rows.append((sym, state_price, broker_price, "ok"))
        else:
            drift_rows.append((sym, state_price, broker_price, f"drift {delta:+.4f}"))

    print()
    print(f"{'Symbol':<8} {'State entry':>14} {'Broker avg':>14} {'Status':>20}")
    print("-" * 60)
    needs_update = False
    for sym, state_p, broker_p, status in drift_rows:
        bp_s = f"${broker_p:>12.4f}" if broker_p is not None else f"{'MISSING':>14}"
        sp_s = f"${state_p:>12.4f}"
        print(f"{sym:<8} {sp_s:>14} {bp_s:>14} {status:>20}")
        if "drift" in status:
            needs_update = True
    print()

    if not needs_update:
        print("[backfill] No drift detected. Nothing to do.")
        return 0

    if not args.apply:
        print("[backfill] DRY-RUN. Pass --apply to write the corrected entry_price values.")
        return 0

    # Apply -- backup first, then write
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = state_path.with_name(f"{state_path.stem}.pre-backfill-{timestamp}.bak.json")
    shutil.copy(state_path, backup)
    print(f"[backfill] Backup written -> {backup}")

    for sym, pos in positions.items():
        broker_price = broker_avg_by_symbol.get(sym)
        if broker_price is None:
            continue
        state_price = float(pos.get("entry_price") or 0)
        if abs(broker_price - state_price) >= 0.001:
            pos["entry_price"] = round(broker_price, 4)
            pos["_backfilled_at"] = datetime.now().isoformat()
            pos["_backfilled_from"] = "broker.avg_entry_price"

    state["last_updated"] = datetime.now().isoformat()

    # Atomic write via tmp + replace
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)
    print(f"[backfill] Wrote {state_path}")
    print(f"[backfill] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
