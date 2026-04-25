"""Reader API: latest, by_id, for_date, iter_records, filter, summary.

Lazy across days. Filter DSL supports common predicates without a
full query language: 'regime=BEAR', 'executions:length=0',
'error.stage=logic', 'preconditions.all_passed=false'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from src.trading.decision_log import paths
from src.trading.decision_log.record import DecisionRecord
from src.utils.logger import logger


@dataclass
class StrategySummary:
    triggers: int
    clean: int        # all preconditions passed AND no error
    blocked: int      # preconditions failed
    errored: int      # error field populated
    orders_filled: int
    orders_total: int


def latest(strategy: str) -> Optional[DecisionRecord]:
    """Return most recent record for a strategy, or None."""
    path = paths.latest_path(strategy)
    if not path.exists():
        return None
    line = path.read_text().strip()
    if not line:
        return None
    try:
        return DecisionRecord.from_jsonl_line(line)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[decision_log] corrupt latest snapshot for {strategy}: {e}")
        return None


def by_id(decision_id: str) -> Optional[DecisionRecord]:
    """Find a record by decision_id. Scans recent days descending."""
    decisions = paths.decisions_dir()
    if not decisions.exists():
        return None
    # Sort newest first
    files = sorted(
        (f for f in decisions.iterdir() if f.is_file() and f.suffix == ".jsonl"),
        reverse=True,
    )
    for f in files:
        try:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = DecisionRecord.from_jsonl_line(line)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    if rec.decision_id == decision_id:
                        return rec
        except OSError:
            continue
    return None


def for_date(strategy: str, day: date) -> List[DecisionRecord]:
    """All records for a strategy on a given date, in write order."""
    return list(iter_records(strategy, since=day, until=day))


def iter_records(
    strategy: str, since: date, until: date,
) -> Iterator[DecisionRecord]:
    """Yield records lazily for a strategy across [since, until] inclusive."""
    day = since
    while day <= until:
        path = paths.jsonl_path(strategy, day)
        if path.exists():
            try:
                with path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield DecisionRecord.from_jsonl_line(line)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(
                                f"[decision_log] skipping corrupt line in {path.name}: {e}"
                            )
                            continue
            except OSError as e:
                logger.warning(f"[decision_log] cannot read {path.name}: {e}")
        day += timedelta(days=1)


def filter_records(
    records: Iterable[DecisionRecord], where: str,
) -> Iterator[DecisionRecord]:
    """Filter records by a small DSL.

    Supported syntaxes:
        'regime=BEAR'                    -> inputs.regime == 'BEAR'
        'regime=WEAK_BULL'
        'executions:length=0'            -> len(rec.executions) == 0
        'error.stage=logic'              -> rec.error and rec.error.stage == 'logic'
        'preconditions.all_passed=false' -> rec.preconditions.all_passed is False
    """
    pred = _compile_predicate(where)
    return (r for r in records if pred(r))


def _compile_predicate(where: str):
    where = where.strip()

    # 'executions:length=N'
    if where.startswith("executions:length="):
        n = int(where.split("=", 1)[1])
        return lambda r: len(r.executions) == n

    # 'error.stage=NAME'
    if where.startswith("error.stage="):
        stage = where.split("=", 1)[1]
        return lambda r: r.error is not None and r.error.stage == stage

    # 'preconditions.all_passed=true|false'
    if where.startswith("preconditions.all_passed="):
        wanted_str = where.split("=", 1)[1].lower()
        wanted = wanted_str in ("true", "1", "yes")
        return lambda r: r.preconditions.all_passed == wanted

    # 'regime=NAME'
    if where.startswith("regime="):
        regime = where.split("=", 1)[1]
        return lambda r: (r.inputs.regime or "") == regime

    raise ValueError(f"Unrecognized filter expression: {where!r}")


def summary(start: date, end: date) -> Dict[str, StrategySummary]:
    """Per-strategy aggregates across [start, end] inclusive."""
    out: Dict[str, StrategySummary] = {}
    decisions = paths.decisions_dir()
    if not decisions.exists():
        return out

    # Discover strategies seen in this range
    seen_strategies = set()
    day = start
    while day <= end:
        for f in decisions.glob(f"*_{day.strftime('%Y%m%d')}.jsonl"):
            stem = f.stem  # 'ramp_20260424'
            strategy = stem.rsplit("_", 1)[0]
            seen_strategies.add(strategy)
        day += timedelta(days=1)

    for strategy in seen_strategies:
        triggers = 0
        clean = 0
        blocked = 0
        errored = 0
        orders_filled = 0
        orders_total = 0
        for rec in iter_records(strategy, since=start, until=end):
            triggers += 1
            if rec.error is not None:
                errored += 1
            if not rec.preconditions.all_passed:
                blocked += 1
            elif rec.error is None:
                clean += 1
            for ex in rec.executions:
                orders_total += 1
                if ex.status == "filled":
                    orders_filled += 1
        out[strategy] = StrategySummary(
            triggers=triggers, clean=clean, blocked=blocked, errored=errored,
            orders_filled=orders_filled, orders_total=orders_total,
        )
    return out


def load_legacy_cscm(path: Path) -> Iterator[DecisionRecord]:
    """Translate a legacy cscm_signals_YYYYMMDD.jsonl file into DecisionRecords.

    The old format had one entry per signal-generation moment; many were
    non-rebalance hourly checks. We yield one record per 'signal' entry.
    Executions are empty (the old log didn't capture them); preconditions
    are marked all_passed=True since the strategy did emit a signal.
    """
    from src.trading.decision_log.record import (
        TriggerInfo, PreconditionResults, GateResult, StrategyInputs,
        RunMetadata,
    )
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "signal":
                continue
            ts = d.get("timestamp", "")
            yield DecisionRecord(
                schema_version=1,
                decision_id=f"legacy-{ts}",
                strategy="cscm",
                timestamp=ts,
                trigger=TriggerInfo(
                    kind="legacy_signal",
                    schedule_time=None,
                    actual_fire_time=ts,
                    delay_seconds=0.0,
                    consecutive_fire_count=1,
                ),
                preconditions=PreconditionResults(
                    all_passed=True,
                    strategy_enabled=GateResult(passed=True, details={}),
                    shutdown_requested=GateResult(passed=True, details={}),
                    execution_lock_acquired=GateResult(passed=True, details={}),
                    health_check=GateResult(passed=True, details={}),
                    data_freshness=GateResult(passed=True, details={}),
                    extra={},
                ),
                inputs=StrategyInputs(
                    regime=d.get("regime"),
                    momentum_scores=d.get("momentum_scores", {}),
                    extra={
                        "btc_price": d.get("btc_price"),
                        "btc_sma": d.get("btc_sma"),
                        "is_rebalance_day": d.get("is_rebalance_day"),
                        "reduce_exposure": d.get("reduce_exposure"),
                        "exposure_pct": d.get("exposure_pct"),
                        "top_symbols": d.get("top_symbols", []),
                    },
                ),
                logic_decisions=None,
                executions=[],
                post_state=None,
                error=None,
                metadata=RunMetadata(
                    broker_name="cscm-legacy",
                    data_provider="cscm-legacy",
                    git_sha="",
                    initial_capital_usd=0.0,
                    strategy_version=0,
                    process_pid=0,
                    hostname="legacy",
                ),
            )
