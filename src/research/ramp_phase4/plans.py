"""Plan sentinels for variants that produce non-allocation outputs.

The engine pattern-matches on isinstance(_SentinelPlan) and dispatches to
the corresponding no-allocation execution path. The reason field flows
into per-day attribution logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class _SentinelPlan:
    """Marker for plans that signal 'no exposure' or other non-allocation actions.

    Engine dispatch: isinstance(plan, _SentinelPlan) -> zero_target_orders().
    The `reason` field is recorded in DailyRecord.regime for traceability.
    """
    reason: str
    weights: Dict[str, float] = field(default_factory=dict)  # always empty for sentinels


PLAN_CASH_BEAR_SOFT = _SentinelPlan(reason='BEAR_SOFT_CASH')
