"""F4 data safe mode tests.

Verifies:
- MarketRegimeDetector raises DataInsufficientError when SPY coverage
  is below the configured threshold (instead of silently returning SIDEWAYS).
- compute_plan() handles safe_mode=True by returning a hold-only plan.
- Hard-block threshold raises with hard_block=True attribute.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestDataSafeMode:
    def test_low_coverage_raises_data_insufficient_error(self):
        """SPY coverage 90% with threshold 95% -> DataInsufficientError (soft)."""
        from src.strategies.advanced.market_regime_detector import (
            MarketRegimeDetector, DataInsufficientError,
        )
        detector = MarketRegimeDetector()
        idx = pd.date_range("2025-01-01", periods=252, freq="B")
        close = np.linspace(400, 500, 252)
        close[:25] = np.nan  # ~90% coverage (227/252)
        spy = pd.DataFrame({"close": close}, index=idx)
        vix = pd.DataFrame({"close": np.full(252, 20.0)}, index=idx)

        with pytest.raises(DataInsufficientError) as exc_info:
            detector.classify_regime(
                spy, vix, idx[-1],
                min_coverage_pct=0.95,
                hard_block_pct=0.80,
            )
        # Soft block: hard_block attribute should be False
        assert getattr(exc_info.value, "hard_block", None) is False

    def test_very_low_coverage_raises_data_insufficient_error_at_hard_block(self):
        """SPY coverage 75% with hard_block 80% -> DataInsufficientError with hard_block=True."""
        from src.strategies.advanced.market_regime_detector import (
            MarketRegimeDetector, DataInsufficientError,
        )
        detector = MarketRegimeDetector()
        idx = pd.date_range("2025-01-01", periods=252, freq="B")
        close = np.linspace(400, 500, 252)
        close[:63] = np.nan  # ~75% coverage (189/252)
        spy = pd.DataFrame({"close": close}, index=idx)
        vix = pd.DataFrame({"close": np.full(252, 20.0)}, index=idx)

        with pytest.raises(DataInsufficientError) as exc_info:
            detector.classify_regime(
                spy, vix, idx[-1],
                min_coverage_pct=0.95,
                hard_block_pct=0.80,
            )
        assert getattr(exc_info.value, "hard_block", None) is True

    def test_planner_safe_mode_holds_existing_positions(self):
        """When safe_mode=True is passed to compute_plan, only existing
        positions appear in targets; no new entries, no exits.
        """
        from src.strategies.advanced.ramp_target_planner import compute_plan
        scores = pd.Series(
            data=np.linspace(0.1, 0.01, 50),
            index=[f"SYM{i:02d}" for i in range(50)],
        )
        plan = compute_plan(
            as_of=datetime(2026, 5, 15),
            regime="SAFE_MODE", regime_confidence=0.0, regime_scores={},
            top_n=10, momentum_scores=scores,
            current_positions={"SYM00": 5000.0, "SYM05": 5000.0},
            vix=18.0, spy_drawdown=-0.02,
            max_capital_allocation=1.0,
            diagnostics={"safe_mode": True},
            safe_mode=True,
        )
        assert plan.regime == "SAFE_MODE"
        assert set(plan.targets.keys()) == {"SYM00", "SYM05"}
        assert plan.exits == {}
