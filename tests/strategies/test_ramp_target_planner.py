"""F1 RampTargetPlanner tests.

Tests cover:
- RampTarget / RampPlan dataclass shapes (Task 5).
- compute_plan() behavior across the 7 F1 base scenarios (Task 6).
- 8-scenario crash-protection parity matrix at the planner level (Task 10).
- Safe mode behavior when data coverage is insufficient (Task 7).
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestRampTargetDataclass:
    def test_ramp_target_constructs_with_required_fields(self):
        from src.strategies.advanced.ramp_target_planner import RampTarget
        t = RampTarget(
            symbol="AAPL",
            target_weight=0.10,
            rank=3,
            regime="STRONG_BULL",
            reason="new_entry",
        )
        assert t.symbol == "AAPL"
        assert t.target_weight == 0.10
        assert t.rank == 3
        assert t.regime == "STRONG_BULL"
        assert t.reason == "new_entry"

    def test_ramp_target_is_frozen(self):
        from src.strategies.advanced.ramp_target_planner import RampTarget
        t = RampTarget(symbol="A", target_weight=0.1, rank=1, regime="STRONG_BULL", reason="new_entry")
        with pytest.raises(Exception):
            t.target_weight = 0.5

    def test_ramp_target_rank_optional(self):
        from src.strategies.advanced.ramp_target_planner import RampTarget
        t = RampTarget(symbol="X", target_weight=0.0, rank=None, regime="BEAR", reason="exit")
        assert t.rank is None


class TestRampPlanDataclass:
    def test_ramp_plan_constructs_with_required_fields(self):
        from src.strategies.advanced.ramp_target_planner import RampTarget, RampPlan
        t = RampTarget(symbol="AAPL", target_weight=0.1, rank=1, regime="STRONG_BULL", reason="new_entry")
        plan = RampPlan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL",
            regime_confidence=0.85,
            regime_scores={"STRONG_BULL": 0.85, "WEAK_BULL": 0.10, "SIDEWAYS": 0.03,
                          "UNPREDICTABLE": 0.01, "BEAR": 0.01},
            exposure_pct=1.0,
            top_n=20,
            targets={"AAPL": t},
            exits={},
            diagnostics={"vix": 18.0, "spy_dd": -0.02},
        )
        assert plan.regime == "STRONG_BULL"
        assert plan.targets["AAPL"].target_weight == 0.1
        assert plan.exposure_pct == 1.0
        assert plan.top_n == 20

    def test_ramp_plan_targets_can_have_exits(self):
        from src.strategies.advanced.ramp_target_planner import RampPlan
        plan = RampPlan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL", regime_confidence=0.8, regime_scores={},
            exposure_pct=1.0, top_n=10, targets={}, exits={"OLD": "dropped_from_top_n"},
            diagnostics={},
        )
        assert "OLD" in plan.exits
