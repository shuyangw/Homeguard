"""DecisionRecord schema -- one record per strategy trigger fire.

Pure data: no I/O, no side effects. The writer, reader, and CLI all
import these dataclasses and rely on the JSON shape being stable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 2


@dataclass
class TriggerInfo:
    kind: str  # 'scheduled_rebalance' | 'scheduled_entry' | 'scheduled_exit' | 'manual' | 'shutdown'
    schedule_time: Optional[str]
    actual_fire_time: str
    delay_seconds: float
    consecutive_fire_count: int = 1


@dataclass
class GateResult:
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PreconditionResults:
    all_passed: bool
    strategy_enabled: GateResult
    shutdown_requested: GateResult
    execution_lock_acquired: GateResult
    health_check: GateResult
    data_freshness: GateResult
    extra: Dict[str, GateResult] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PreconditionResults":
        empty_gate = GateResult(passed=False, details={})
        return cls(
            all_passed=False,
            strategy_enabled=empty_gate,
            shutdown_requested=empty_gate,
            execution_lock_acquired=empty_gate,
            health_check=empty_gate,
            data_freshness=empty_gate,
            extra={},
        )


@dataclass
class StrategyInputs:
    regime: Optional[str] = None
    regime_confidence: Optional[float] = None
    regime_params: Optional[Dict[str, Any]] = None
    vix: Optional[float] = None
    spy_drawdown_pct: Optional[float] = None
    universe_size: int = 0
    data_completeness_pct: float = 0.0
    cache_source: str = ""
    momentum_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    # F5 enrichments (Phase 4 Phase A):
    regime_scores: Dict[str, float] = field(default_factory=dict)    # all 5 regime scores
    vix_percentile: Optional[float] = None                            # trailing-252d percentile
    breadth_pct_above_50d: Optional[float] = None                     # pct of universe above 50d MA
    exposure_multiplier: Optional[float] = None                       # 0.0, 0.5, or 1.0
    fallback_mode_used: Optional[str] = None                          # "no_rebalance" | "hold_only" | None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "StrategyInputs":
        return cls()


@dataclass
class LogicDecisions:
    top_n: int
    target_symbols: List[str]
    target_weights: Dict[str, float]
    target_value_usd: Dict[str, float]
    reduce_exposure: bool
    exposure_pct: float
    exit_signals: List[str]
    hold_signals: List[str]
    skip_reasons: Dict[str, str]
    # F5 enrichments (Phase 4 Phase A):
    realized_weights: Dict[str, float] = field(default_factory=dict)
    realized_turnover_usd: Optional[float] = None
    expected_cost_bps: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    cash_after_rebalance_usd: Optional[float] = None


@dataclass
class Execution:
    symbol: str
    action: str  # 'buy' | 'sell' | 'hold' | 'skip'
    target_qty: int
    target_value_usd: float
    status: str  # 'filled' | 'partial' | 'rejected' | 'skipped' | 'error' | 'placed_unfilled'
    fill_price: Optional[float] = None
    filled_qty: Optional[int] = None
    order_id: Optional[str] = None
    reason: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class PositionSnapshot:
    qty: float
    avg_price: float
    unrealized_pnl: float


@dataclass
class PostState:
    positions_after: Dict[str, PositionSnapshot]
    cash_after: float
    strategy_equity_after: float
    state_writes: List[str]


@dataclass
class ErrorInfo:
    type: str
    message: str
    traceback: str
    stage: str  # 'preconditions' | 'inputs' | 'logic' | 'execution' | 'post_state' | 'unknown'


@dataclass
class RunMetadata:
    broker_name: str
    data_provider: str
    git_sha: str
    initial_capital_usd: float
    strategy_version: int
    process_pid: int
    hostname: str


@dataclass
class DecisionRecord:
    schema_version: int
    decision_id: str
    strategy: str
    timestamp: str

    trigger: TriggerInfo
    preconditions: PreconditionResults
    inputs: StrategyInputs
    logic_decisions: Optional[LogicDecisions]
    executions: List[Execution]
    post_state: Optional[PostState]
    error: Optional[ErrorInfo]
    metadata: RunMetadata

    # Optional linkage for OMR entry -> exit pairing
    parent_decision_id: Optional[str] = None

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line ending in newline."""
        d = asdict(self)
        return json.dumps(d, separators=(",", ":")) + "\n"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DecisionRecord":
        """Parse a JSON line back into a typed record."""
        data = json.loads(line)
        return _record_from_dict(data)


def _record_from_dict(d: Dict[str, Any]) -> DecisionRecord:
    """Reconstruct nested dataclasses from a plain dict."""
    trigger = TriggerInfo(**d["trigger"])
    preconditions = PreconditionResults(
        all_passed=d["preconditions"]["all_passed"],
        strategy_enabled=GateResult(**d["preconditions"]["strategy_enabled"]),
        shutdown_requested=GateResult(**d["preconditions"]["shutdown_requested"]),
        execution_lock_acquired=GateResult(**d["preconditions"]["execution_lock_acquired"]),
        health_check=GateResult(**d["preconditions"]["health_check"]),
        data_freshness=GateResult(**d["preconditions"]["data_freshness"]),
        extra={k: GateResult(**v) for k, v in d["preconditions"].get("extra", {}).items()},
    )
    inputs = StrategyInputs(**d["inputs"])
    logic_decisions = (
        LogicDecisions(**d["logic_decisions"]) if d["logic_decisions"] else None
    )
    executions = [Execution(**e) for e in d["executions"]]
    post_state = None
    if d["post_state"] is not None:
        post_state = PostState(
            positions_after={k: PositionSnapshot(**v) for k, v in d["post_state"]["positions_after"].items()},
            cash_after=d["post_state"]["cash_after"],
            strategy_equity_after=d["post_state"]["strategy_equity_after"],
            state_writes=d["post_state"]["state_writes"],
        )
    error = ErrorInfo(**d["error"]) if d["error"] else None
    metadata = RunMetadata(**d["metadata"])
    return DecisionRecord(
        schema_version=d["schema_version"],
        decision_id=d["decision_id"],
        strategy=d["strategy"],
        timestamp=d["timestamp"],
        trigger=trigger,
        preconditions=preconditions,
        inputs=inputs,
        logic_decisions=logic_decisions,
        executions=executions,
        post_state=post_state,
        error=error,
        metadata=metadata,
        parent_decision_id=d.get("parent_decision_id"),
    )
