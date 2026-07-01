"""Build per-root futures roll calendars from OI + definitions.

For each root and each trading day in the requested range:
  1. read per-contract OI (statistics stat_type=9)
  2. detect OI-crossover rolls (hysteresis) -> front contract per day
  3. clamp physical-root rolls to before FND
  4. resolve next-by-cycle and next-by-OI contracts + front expiration
  5. write futures/roll_calendar/{root}.parquet

Usage:
    python scripts/data/build_roll_calendar.py --roots GC CL ES --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import polars as pl

from src.data.derivations.futures.open_interest import per_contract_open_interest
from src.data.futures.contract_specs import SPECS, get_spec
from src.data.futures.paths import roll_calendar_dir
from src.data.futures.roll_calendar import _front_by_oi, apply_fnd_clamp, detect_rolls
from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MONTH_CODES = "FGHJKMNQUVXZ"


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _cycle_order_key(sym: str, root: str) -> tuple[int, int]:
    """Sort key (year, month) for a raw contract symbol, for cycle ordering.

    Robust to single- vs two-digit year encodings (e.g. GCG4 and GCG24 both
    decode to 2024).
    """
    suffix = sym[len(root):]
    month = _MONTH_CODES.index(suffix[0])
    yd = suffix[1:]
    if len(yd) == 1:
        base = (date.today().year // 10) * 10
        year = base + int(yd)
        if year < date.today().year - 5:
            year += 10
    else:
        base = (date.today().year // 100) * 100
        year = base + int(yd)
    return (year, month)


def _next_by_cycle(day_oi: dict[str, int], current_front: str, root: str, next_oi_fallback: str) -> str:
    """Next contract after current_front in the root's LIQUID cycle order.

    Off-cycle outrights (month code not in spec.cycle_months) are excluded.
    Falls back to next_oi_fallback if current_front is not in the in-cycle set.
    """
    spec = get_spec(root)
    in_cycle = [s for s in day_oi if len(s) > len(root) and s[len(root)] in spec.cycle_months]
    by_cycle = sorted(in_cycle, key=lambda s: _cycle_order_key(s, root))
    try:
        fi = by_cycle.index(current_front)
    except ValueError:
        return next_oi_fallback
    return by_cycle[fi + 1] if fi + 1 < len(by_cycle) else next_oi_fallback


def build_root(root: str, start: date, end: date) -> pl.DataFrame:
    defs = FuturesDefinitionsLoader()
    oi_by_day: dict[date, dict[str, int]] = {}
    for d in _daterange(start, end):
        if d.weekday() >= 5:
            continue
        try:
            oi = per_contract_open_interest(root, d)
        except FileNotFoundError:
            continue
        if oi:
            oi_by_day[d] = oi

    if not oi_by_day:
        logger.warning(f"[!] no OI data for {root} in range -- skipping")
        return pl.DataFrame()

    rolls = detect_rolls(root, oi_by_day)

    # from_symbol expirations, resolved at each roll's OWN date (the outgoing
    # contract is definitely active there) -- for the FND clamp.
    expirations: dict[str, date] = {}
    for ev in rolls:
        if ev.from_symbol not in expirations:
            try:
                expirations[ev.from_symbol] = defs.get_expiration(ev.from_symbol, root, ev.roll_date)
            except (LookupError, FileNotFoundError, ValueError):
                pass
    rolls = apply_fnd_clamp(root, rolls, expirations)

    # front contract per day by walking rolls
    roll_map = {ev.roll_date: ev for ev in rolls}
    rows = []
    current_front = None
    for d in sorted(oi_by_day):
        if current_front is None:
            current_front = _front_by_oi(oi_by_day[d])
        if d in roll_map:
            current_front = roll_map[d].to_symbol
        day_oi = oi_by_day[d]
        # next-by-oi: 2nd highest OI outright
        ranked = sorted(day_oi, key=day_oi.get, reverse=True)
        next_oi = ranked[1] if len(ranked) > 1 else ranked[0]
        # next-by-cycle: next expiry after front in the liquid cycle, among present contracts
        next_cycle = _next_by_cycle(day_oi, current_front, root, next_oi)
        exp = expirations.get(current_front)
        if exp is None:
            try:
                exp = defs.get_expiration(current_front, root, d)
            except (LookupError, FileNotFoundError, ValueError):
                exp = d  # degenerate fallback; dte becomes 0
        try:
            act = defs.get_definition(current_front, root, d).activation
        except (LookupError, FileNotFoundError, ValueError):
            act = d
        rows.append({
            "date": d,
            "front_symbol": current_front,
            "front_expiration": exp,
            "front_activation": act,
            "next_cycle_symbol": next_cycle,
            "next_oi_symbol": next_oi,
            "dte_front": max((exp - d).days, 0),
            "roll_trigger": roll_map[d].trigger if d in roll_map else "hold",
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=sorted(SPECS.keys()))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = roll_calendar_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for root in args.roots:
        get_spec(root)  # validate known root
        df = build_root(root, start, end)
        if df.is_empty():
            continue
        df.write_parquet(out_dir / f"{root}.parquet")
        logger.info(f"[+] built roll calendar for {root}: {df.height} days")


if __name__ == "__main__":
    main()
