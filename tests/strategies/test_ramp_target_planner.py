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


class TestComputePlanCore:
    """F1 core scenarios from the Phase 4 doc."""

    def _make_momentum_scores(self, n: int) -> pd.Series:
        """Return a series of n momentum scores indexed by SYM00..SYM(n-1).

        Highest score for SYM00, decreasing. Deterministic ranking.
        """
        return pd.Series(
            data=np.linspace(0.10, 0.01, n),
            index=[f"SYM{i:02d}" for i in range(n)],
        )

    def test_strong_bull_targets_sum_to_100_pct(self):
        """No trigger, top_n=20 -> sum of target weights = 1.0."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL",
            regime_confidence=0.9,
            regime_scores={"STRONG_BULL": 0.9, "WEAK_BULL": 0.05, "SIDEWAYS": 0.02,
                          "UNPREDICTABLE": 0.02, "BEAR": 0.01},
            top_n=20,
            momentum_scores=scores,
            current_positions={},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={},
        )
        total_target = sum(t.target_weight for t in plan.targets.values())
        assert abs(total_target - 1.0) < 1e-9
        assert len(plan.targets) == 20
        for t in plan.targets.values():
            assert t.reason in ("new_entry", "hold")

    def test_crash_protection_halves_all_target_weights(self):
        """VIX-trigger path: every target_weight halved vs no-trigger path. Sum = 0.5."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL",
            regime_confidence=0.9,
            regime_scores={"STRONG_BULL": 0.9, "WEAK_BULL": 0.05, "SIDEWAYS": 0.02,
                          "UNPREDICTABLE": 0.02, "BEAR": 0.01},
            top_n=20,
            momentum_scores=scores,
            current_positions={},
            vix=26.0,  # VIX trigger
            spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={},
        )
        total_target = sum(t.target_weight for t in plan.targets.values())
        assert abs(total_target - 0.5) < 1e-9
        for t in plan.targets.values():
            assert abs(t.target_weight - 0.025) < 1e-9  # 0.5 / 20
        assert plan.exposure_pct == 0.5

    def test_sideways_top5_targets_are_20_pct_each(self):
        """SIDEWAYS, no trigger, top_n=5 -> every target weight = 0.20."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="SIDEWAYS",
            regime_confidence=0.7,
            regime_scores={"STRONG_BULL": 0.1, "WEAK_BULL": 0.1, "SIDEWAYS": 0.7,
                          "UNPREDICTABLE": 0.05, "BEAR": 0.05},
            top_n=5,
            momentum_scores=scores,
            current_positions={},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={},
        )
        assert len(plan.targets) == 5
        for t in plan.targets.values():
            assert abs(t.target_weight - 0.20) < 1e-9

    def test_existing_non_target_position_gets_exit(self):
        """Symbol held but no longer ranked -> in exits."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL", regime_confidence=0.9,
            regime_scores={"STRONG_BULL": 0.9, "WEAK_BULL": 0.05, "SIDEWAYS": 0.02,
                          "UNPREDICTABLE": 0.02, "BEAR": 0.01},
            top_n=20,
            momentum_scores=scores,
            current_positions={"OLDSYM": 5000.0},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={},
        )
        assert "OLDSYM" in plan.exits
        assert plan.exits["OLDSYM"] in ("dropped_from_top_n", "exit")
        assert "OLDSYM" not in plan.targets

    def test_hold_position_gets_target_weight(self):
        """Symbol previously held AND still ranked -> target_weight matches new entries, reason='hold'."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="STRONG_BULL", regime_confidence=0.9,
            regime_scores={"STRONG_BULL": 0.9, "WEAK_BULL": 0.05, "SIDEWAYS": 0.02,
                          "UNPREDICTABLE": 0.02, "BEAR": 0.01},
            top_n=20,
            momentum_scores=scores,
            current_positions={"SYM00": 5000.0},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={},
        )
        assert "SYM00" in plan.targets
        sym00 = plan.targets["SYM00"]
        assert sym00.reason == "hold"
        assert abs(sym00.target_weight - 0.05) < 1e-9  # 1.0 / 20

    def test_low_data_safe_mode_blocks_new_entries(self):
        """safe_mode=True -> only existing holds in targets; no new entries; no exits."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="SAFE_MODE",
            regime_confidence=0.0,
            regime_scores={},
            top_n=10,
            momentum_scores=scores,
            current_positions={"SYM00": 5000.0, "SYM05": 5000.0},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={"safe_mode": True, "reason": "data_coverage_below_threshold"},
            safe_mode=True,
        )
        assert plan.regime == "SAFE_MODE"
        assert set(plan.targets.keys()) <= {"SYM00", "SYM05"}
        assert plan.exits == {}

    def test_bear_cash_variant_targets_zero_gross(self):
        """bear_to_cash=True AND regime==BEAR -> total target weight = 0; positions in exits."""
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = self._make_momentum_scores(50)
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="BEAR", regime_confidence=0.8,
            regime_scores={"STRONG_BULL": 0.0, "WEAK_BULL": 0.05, "SIDEWAYS": 0.10,
                          "UNPREDICTABLE": 0.05, "BEAR": 0.80},
            top_n=10,
            momentum_scores=scores,
            current_positions={"SYM00": 5000.0},
            vix=22.0, spy_drawdown=-0.08,
            max_capital_allocation=1.0,
            diagnostics={},
            bear_to_cash=True,
        )
        total_target = sum(t.target_weight for t in plan.targets.values())
        assert total_target == 0.0
        assert "SYM00" in plan.exits
