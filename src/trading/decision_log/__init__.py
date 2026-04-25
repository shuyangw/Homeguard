"""Decision Log package -- one canonical record per strategy trigger fire.

Public API:
    DecisionRecord            -- the dataclass schema (record.py)
    begin_decision()          -- helper to construct a new record
    write_decision()          -- write a record to disk (writer.append)
    stage()                   -- context manager that tags ErrorInfo.stage on exception
    latest, by_id, for_date,
    iter_records, filter_records,
    summary                   -- reader API (reader.py)
"""
from __future__ import annotations

import os
import socket
import uuid
from contextlib import contextmanager
from typing import Optional

from src.trading.decision_log.record import (
    DecisionRecord, TriggerInfo, GateResult, PreconditionResults,
    StrategyInputs, LogicDecisions, Execution, PositionSnapshot,
    PostState, ErrorInfo, RunMetadata, SCHEMA_VERSION,
)
from src.trading.decision_log.writer import append as write_decision
from src.trading.decision_log.reader import (
    latest, by_id, for_date, iter_records, filter_records, summary,
    StrategySummary,
)
from src.utils.timezone import tz


__all__ = [
    "DecisionRecord", "TriggerInfo", "GateResult", "PreconditionResults",
    "StrategyInputs", "LogicDecisions", "Execution", "PositionSnapshot",
    "PostState", "ErrorInfo", "RunMetadata", "SCHEMA_VERSION",
    "begin_decision", "write_decision", "stage",
    "latest", "by_id", "for_date", "iter_records", "filter_records",
    "summary", "StrategySummary",
]


def begin_decision(
    strategy: str,
    trigger_kind: str,
    schedule_time: Optional[str] = None,
    *,
    broker_name: str = "",
    data_provider: str = "",
    initial_capital_usd: float = 0.0,
    strategy_version: int = 1,
    git_sha: str = "",
) -> DecisionRecord:
    """Construct a fresh DecisionRecord with identity + trigger fields populated.

    Caller is expected to populate preconditions, inputs, logic_decisions,
    executions, post_state progressively, then pass to write_decision().
    """
    now_iso = tz.iso_timestamp()
    return DecisionRecord(
        schema_version=SCHEMA_VERSION,
        decision_id=str(uuid.uuid4()),
        strategy=strategy,
        timestamp=now_iso,
        trigger=TriggerInfo(
            kind=trigger_kind,
            schedule_time=schedule_time,
            actual_fire_time=now_iso,
            delay_seconds=0.0,
            consecutive_fire_count=1,
        ),
        preconditions=PreconditionResults.empty(),
        inputs=StrategyInputs.empty(),
        logic_decisions=None,
        executions=[],
        post_state=None,
        error=None,
        metadata=RunMetadata(
            broker_name=broker_name,
            data_provider=data_provider,
            git_sha=git_sha,
            initial_capital_usd=initial_capital_usd,
            strategy_version=strategy_version,
            process_pid=os.getpid(),
            hostname=socket.gethostname(),
        ),
    )


@contextmanager
def stage(rec: DecisionRecord, name: str):
    """Tag the current pipeline stage on the record.

    If an exception escapes the with-block, ErrorInfo populated by the
    caller's outer try/except can read rec._current_stage via getattr.
    """
    rec._current_stage = name  # type: ignore[attr-defined]
    try:
        yield
    finally:
        # Stage stays set; do NOT reset, so a later exception capture
        # reflects the stage where the exception fired.
        pass
