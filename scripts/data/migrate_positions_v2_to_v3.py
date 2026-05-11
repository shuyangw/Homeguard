"""One-shot migration: strategy_positions.json v2 -> v3.

v2 schema (pre-futures): {
  "strategies": {
    "<name>": {
      "positions": {"<symbol>": {qty, entry_price, entry_time, order_id}},
      "last_execution": "<isoformat>",
      ...
    }
  }
}

v3 schema (adds futures support): same top-level shape with:
  - top-level "version": 3
  - each position dict gains nullable fields:
      contract_month: str | None     # "YYYYMM" for futures; null for stocks/options
      raw_symbol: str | None         # IBKR/CME format for futures; null otherwise
      multiplier: float | None
      tick_size: float | None
      tick_value: float | None
      expiration_date: str | None    # ISO date for futures; null otherwise

Existing v2 positions (stocks/options) get null values for all new fields.
Loader code branches on contract_month: null means stock/options path,
populated means the FuturesPosition path.

Usage:
    python scripts/data/migrate_positions_v2_to_v3.py <input.json> <output.json>

The script is idempotent: running on an already-v3 file returns it
unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

V3_VERSION = 3

# Fields added in v3, default null for non-futures positions
_NEW_FIELDS = (
    "contract_month", "raw_symbol",
    "multiplier", "tick_size", "tick_value",
    "expiration_date",
)


def migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Pure-function migration of an in-memory state dict.

    Idempotent: if state already has version >= 3, returned unchanged.
    """
    if state.get("version") == V3_VERSION:
        return state

    out: dict[str, Any] = {"version": V3_VERSION}
    # Preserve everything except positions, which get the field additions
    for k, v in state.items():
        if k == "strategies":
            out["strategies"] = {}
            for strategy_name, strategy_state in v.items():
                migrated_positions = {}
                positions = strategy_state.get("positions", {})
                for symbol, pos in positions.items():
                    new_pos = dict(pos)
                    for f in _NEW_FIELDS:
                        new_pos.setdefault(f, None)
                    migrated_positions[symbol] = new_pos
                out["strategies"][strategy_name] = {
                    **strategy_state,
                    "positions": migrated_positions,
                }
        else:
            out[k] = v
    return out


def migrate_file(src: Path, dest: Path) -> None:
    """Read v2 JSON from src, write v3 JSON to dest. Atomic write."""
    state = json.loads(src.read_text())
    migrated = migrate_state(state)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(migrated, indent=2, default=str))
    tmp.replace(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path, help="v2 strategy_positions.json")
    p.add_argument("output", type=Path, help="v3 output path")
    args = p.parse_args()
    if not args.input.exists():
        print(f"ERROR: input file does not exist: {args.input}")
        return 1
    migrate_file(args.input, args.output)
    print(f"Migrated: {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
