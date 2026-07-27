"""Forward-paper evidence log for specs that cannot be gated historically.

The viability screen routes two kinds of spec here: one whose best-case-if-true
Sharpe cannot clear the deflated bar, and one that clears on signal but cannot
be traded at our account's order size. Neither should consume a trial, and
neither can be settled by another backtest. What they need is a fresh sample,
and the only fresh sample available is the future.

The property that makes this forward validation rather than a backtest with
extra ceremony is narrow and absolute: **an observation cannot be recorded for a
date at or before the spec's lock date.** Backfilling would manufacture evidence
from data that already existed when the spec was written, which is the thing the
whole pre-registration discipline exists to prevent. That guard, plus an
append-only log and a spec fingerprint, is the entire integrity model.

Deliberately NOT provided: any function that computes historical observations in
bulk. There is no convenience path from here back to a backtest.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_LOG = Path("config/forward_paper/observations.jsonl")
# Below this, an average is noise. Stated so a small favourable run cannot be
# reported as a result: 12 month-end events a year means years, by construction.
MIN_CONCLUSIVE_N = 30


class ForwardPaperError(RuntimeError):
    """An operation that would compromise the forward-only guarantee."""


@dataclass(frozen=True)
class ForwardSpec:
    name: str
    locked_on: dt.date
    params: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash of the locked parameters.

        Stored on every observation so that a later parameter change cannot
        quietly absorb evidence gathered under the earlier version.
        """
        blob = json.dumps({"name": self.name, "locked_on": self.locked_on.isoformat(),
                           "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _last_business_day(year: int, month: int) -> dt.date:
    day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, day)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def month_end_events(start: dt.date, end: dt.date) -> List[dt.date]:
    """Last business day of each month in [start, end]."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        d = _last_business_day(y, m)
        if start <= d <= end:
            out.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def quarter_end_events(start: dt.date, end: dt.date) -> List[dt.date]:
    return [d for d in month_end_events(start, end) if d.month in (3, 6, 9, 12)]


def load_observations(path: Path = DEFAULT_LOG) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def record_observation(spec: ForwardSpec, event_date: dt.date, *, signal: int,
                       return_bps: float, path: Path = DEFAULT_LOG,
                       notes: str = "") -> Dict[str, Any]:
    """Append one realised event. Forward-only, append-only, one row per event."""
    if event_date <= spec.locked_on:
        raise ForwardPaperError(
            f"{spec.name}: refusing to record {event_date}, which is on or before "
            f"the lock date {spec.locked_on}. Forward paper accumulates evidence "
            "from data that did not exist when the spec was written; backfilling "
            "would make this a backtest that never declared a trial.")

    existing = load_observations(path)
    if any(r["spec"] == spec.name and r["event_date"] == event_date.isoformat()
           for r in existing):
        raise ForwardPaperError(
            f"{spec.name}: {event_date} is already recorded. The log is "
            "append-only; re-recording is how an inconvenient observation "
            "disappears.")

    row = {"spec": spec.name, "spec_hash": spec.fingerprint(),
           "locked_on": spec.locked_on.isoformat(),
           "event_date": event_date.isoformat(), "signal": int(signal),
           "return_bps": float(return_bps), "notes": notes}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def summarise(path: Path = DEFAULT_LOG) -> Dict[str, Dict[str, Any]]:
    """Per-spec counts and mean. Reports `conclusive` honestly, which is False
    until there are enough observations to mean anything."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in load_observations(path):
        s = out.setdefault(row["spec"], {"n": 0, "total_bps": 0.0,
                                         "first": row["event_date"],
                                         "last": row["event_date"]})
        s["n"] += 1
        s["total_bps"] += row["return_bps"]
        s["last"] = max(s["last"], row["event_date"])
        s["first"] = min(s["first"], row["event_date"])
    for s in out.values():
        s["mean_bps"] = s["total_bps"] / s["n"] if s["n"] else float("nan")
        s["conclusive"] = s["n"] >= MIN_CONCLUSIVE_N
    return out
