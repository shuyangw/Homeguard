"""Tests for the paper validation comparator.

The comparator reads a paper-trading decision log entry, recomputes the
RampPlan for the same date, and reports divergences with severity levels.
"""
import json
from pathlib import Path
from unittest.mock import patch


def _write_decision_log_entry(path: Path, target_weights: dict, regime: str = "STRONG_BULL") -> None:
    """Write a synthetic decision log entry as JSON."""
    rec = {
        "strategy": "ramp",
        "as_of": "2026-05-20T15:55:00-04:00",
        "schema_version": 2,
        "strategy_inputs": {
            "regime": regime,
            "regime_confidence": 0.85,
            "regime_scores": {"STRONG_BULL": 0.85, "WEAK_BULL": 0.05,
                              "SIDEWAYS": 0.05, "UNPREDICTABLE": 0.03, "BEAR": 0.02},
            "vix": 18.0,
            "spy_drawdown_pct": -0.02,
            "exposure_multiplier": 1.0,
        },
        "logic_decisions": {
            "top_n": len(target_weights),
            "target_symbols": list(target_weights.keys()),
            "target_weights": target_weights,
            "target_value_usd": {sym: w * 100_000 for sym, w in target_weights.items()},
            "reduce_exposure": False,
        },
    }
    path.write_text(json.dumps(rec))


class TestPaperComparator:
    def test_comparator_passes_on_matching_session(self, tmp_path):
        from scripts.trading.compare_paper_vs_plan import compare_session
        target_weights = {f"SYM{i}": 0.05 for i in range(20)}
        log_path = tmp_path / "log.json"
        _write_decision_log_entry(log_path, target_weights)

        with patch("scripts.trading.compare_paper_vs_plan._recompute_plan") as mock_recompute:
            mock_recompute.return_value = {
                "target_weights": target_weights,
                "regime": "STRONG_BULL",
                "exposure_pct": 1.0,
            }
            result = compare_session(log_path)

        assert result["status"] == "PASS"
        assert result["divergences"] == []

    def test_comparator_returns_vacuous_when_no_decisions_and_no_inputs(self, tmp_path):
        """When both logic_decisions and strategy_inputs are empty there is
        literally nothing to compare. The comparator must distinguish this
        from a real PASS:
          - compare_session().status == 'VACUOUS'
          - main() exit code == 3
        """
        import subprocess
        import sys
        from scripts.trading.compare_paper_vs_plan import compare_session

        rec = {
            "strategy": "ramp",
            "as_of": "2026-05-20T15:55:00-04:00",
            "schema_version": 2,
            "strategy_inputs": None,
            "logic_decisions": None,
        }
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(rec))

        result = compare_session(log_path)
        assert result["status"] == "VACUOUS", result
        assert result["divergences"] == []

        # CLI exit code must be 3 so the A7 helper can distinguish a vacuous
        # PASS from a real PASS.
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.trading.compare_paper_vs_plan", str(log_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)

    def test_comparator_flags_target_weight_delta(self, tmp_path):
        from scripts.trading.compare_paper_vs_plan import compare_session
        log_weights = {f"SYM{i}": 0.05 for i in range(20)}
        log_path = tmp_path / "log.json"
        _write_decision_log_entry(log_path, log_weights)

        recomputed_weights = dict(log_weights)
        recomputed_weights["SYM0"] = 0.11  # 6% delta -> Severity 1
        with patch("scripts.trading.compare_paper_vs_plan._recompute_plan") as mock_recompute:
            mock_recompute.return_value = {
                "target_weights": recomputed_weights,
                "regime": "STRONG_BULL",
                "exposure_pct": 1.0,
            }
            result = compare_session(log_path)

        assert result["status"] == "FAIL"
        assert any(d["symbol"] == "SYM0" and d["severity"] == 1 for d in result["divergences"])


class TestRecomputeIntegration:
    """End-to-end: write a decision log entry with strategy_inputs that
    include real momentum_scores; the comparator must derive a plan that
    matches what compute_plan() would produce.
    """

    def test_recompute_plan_from_strategy_inputs_matches_log_weights(self, tmp_path):
        """If the strategy saw a regime + momentum scores and produced
        target_weights, replaying those inputs through compute_plan() should
        give the SAME weights -> comparator PASSES.

        Planner: top_n=20, vix=18 < 25 (no crash) -> exposure_pct=1.0
        -> per_position_weight = 1.0/20 = 0.05 for each of the top 20 symbols.
        """
        import json
        import pytest
        from scripts.trading.compare_paper_vs_plan import compare_session

        # STRONG_BULL, top_n=20, 25 momentum scores (descending by index)
        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.003 * i, 4) for i in range(25)}
        # Planner ranks descending and picks top 20 (SYM00..SYM19)
        top_20 = sorted(momentum_scores.items(), key=lambda x: -x[1])[:20]
        # Expected planner weight = 1.0/20 = 0.05 per symbol
        log_weights = {sym: 0.05 for sym, _ in top_20}

        rec = {
            "strategy": "ramp",
            "as_of": "2026-05-20T15:55:00-04:00",
            "schema_version": 2,
            "strategy_inputs": {
                "regime": "STRONG_BULL",
                "regime_confidence": 0.9,
                "regime_scores": {
                    "STRONG_BULL": 0.9, "WEAK_BULL": 0.05,
                    "SIDEWAYS": 0.02, "UNPREDICTABLE": 0.02, "BEAR": 0.01,
                },
                "vix": 18.0,
                "spy_drawdown_pct": -0.02,
                "momentum_scores": momentum_scores,
                "regime_params": {"top_n": 20},
            },
            "logic_decisions": {
                "top_n": 20,
                "target_symbols": [sym for sym, _ in top_20],
                "target_weights": log_weights,
                "target_value_usd": {sym: 5000.0 for sym in log_weights},
                "reduce_exposure": False,
            },
        }
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(rec))

        result = compare_session(log_path)

        assert result["status"] == "PASS", (
            f"Expected PASS, got divergences: {result['divergences']}"
        )
        assert result["log_total_gross"] == pytest.approx(1.0, abs=1e-9)
        assert result["plan_total_gross"] == pytest.approx(1.0, abs=1e-9)

    def test_recompute_plan_detects_real_divergence(self, tmp_path):
        """If the strategy's log says target_weights include a symbol the
        planner would NOT select (wrong pick), the comparator must FAIL.

        Uses top_n=10 so per_position_weight=0.1, giving a 0.1 delta for
        the wrong symbol -- well above the 0.05 Severity 1 threshold.
        """
        import json
        from scripts.trading.compare_paper_vs_plan import compare_session

        # 15 momentum scores; planner will pick top 10 (SYM00..SYM09)
        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.005 * i, 4) for i in range(15)}
        # Log claims 9 correct picks + SYM99 (a stale/wrong symbol)
        log_weights = {f"SYM{i:02d}": 0.10 for i in range(9)}
        log_weights["SYM99"] = 0.10  # wrong pick -- not in momentum_scores

        rec = {
            "strategy": "ramp",
            "as_of": "2026-05-20T15:55:00-04:00",
            "schema_version": 2,
            "strategy_inputs": {
                "regime": "STRONG_BULL",
                "regime_confidence": 0.9,
                "regime_scores": {"STRONG_BULL": 0.9},
                "vix": 18.0,
                "spy_drawdown_pct": -0.02,
                "momentum_scores": momentum_scores,
                "regime_params": {"top_n": 10},
            },
            "logic_decisions": {
                "top_n": 10,
                "target_symbols": list(log_weights.keys()),
                "target_weights": log_weights,
                "target_value_usd": {sym: 5000.0 for sym in log_weights},
                "reduce_exposure": False,
            },
        }
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(rec))

        result = compare_session(log_path)

        # Planner picks SYM00..SYM09; log has SYM00..SYM08 + SYM99.
        # SYM99: log=0.10, plan=0.00 -> delta 0.10 > 0.05 (Severity 1)
        # SYM09: log=0.00, plan=0.10 -> delta 0.10 > 0.05 (Severity 1)
        assert result["status"] == "FAIL"
        symbols_flagged = {d["symbol"] for d in result["divergences"]}
        assert "SYM99" in symbols_flagged
        assert "SYM09" in symbols_flagged


class TestV11Comparator:
    """V11 comparator extension: thread position ledger + variant flag through
    _recompute_plan so the paper validation comparator models the same
    rank_buffer + min_hold composition the live adapter applies.

    The critical correctness criterion is that _apply_v11_filters_to_plan
    produces the same target set as RAMPLiveAdapter._apply_v11_filters for the
    same inputs and state -- otherwise comparator divergence is spurious.
    """

    def _build_strategy_inputs(self, momentum_scores: dict, top_n: int) -> dict:
        return {
            "regime": "STRONG_BULL",
            "regime_confidence": 0.9,
            "regime_scores": {"STRONG_BULL": 0.9},
            "vix": 18.0,
            "spy_drawdown_pct": -0.02,
            "momentum_scores": momentum_scores,
            "regime_params": {"top_n": top_n},
        }

    def test_recompute_plan_v01_backward_compat_unchanged(self, tmp_path):
        """variant='v01' (default) and a missing position ledger produce the
        same plan as the pre-Phase-2E V01 path."""
        from scripts.trading.compare_paper_vs_plan import _recompute_plan

        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.003 * i, 4) for i in range(25)}
        inputs = self._build_strategy_inputs(momentum_scores, top_n=20)

        baseline = _recompute_plan(inputs)
        bc_explicit = _recompute_plan(
            inputs,
            position_ledger_path=tmp_path / "nonexistent.json",
            variant="v01",
        )

        assert bc_explicit["target_weights"] == baseline["target_weights"]
        # Top 20 selected, equal-weight 1/20.
        assert len(baseline["target_weights"]) == 20
        for w in baseline["target_weights"].values():
            assert abs(w - 0.05) < 1e-9

    def test_recompute_plan_v11_with_empty_ledger_equals_v01(self, tmp_path):
        """variant='v11' with no positions in the ledger (Day 1 of paper)
        reproduces the V01 plan -- rank_buffer has nothing to retain and
        min_hold has nothing to protect."""
        from scripts.trading.compare_paper_vs_plan import _recompute_plan

        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.003 * i, 4) for i in range(25)}
        inputs = self._build_strategy_inputs(momentum_scores, top_n=20)

        ledger_path = tmp_path / "ramp_position_state.json"
        ledger_path.write_text(json.dumps({
            "strategy": "ramp",
            "timestamp": "2026-05-24T16:00:00-04:00",
            "positions": {},
            "position_open_dates": {},
        }))

        v01_plan = _recompute_plan(inputs)
        v11_plan = _recompute_plan(
            inputs, position_ledger_path=ledger_path, variant="v11"
        )

        assert set(v11_plan["target_weights"].keys()) == set(v01_plan["target_weights"].keys())
        for sym, w in v11_plan["target_weights"].items():
            assert abs(w - v01_plan["target_weights"][sym]) < 1e-9

    def test_recompute_plan_v11_retains_held_names_via_rank_buffer(self, tmp_path):
        """variant='v11' with held names that fell out of top_n but rank
        within top_n + (top_n // 2) buffer are retained."""
        from scripts.trading.compare_paper_vs_plan import _recompute_plan

        # 25 symbols, top_n=10, buffer=5 -> retain ranks 1..15 if held.
        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.003 * i, 4) for i in range(25)}
        inputs = self._build_strategy_inputs(momentum_scores, top_n=10)

        # SYM12 ranks 13 (sorted desc by score, index 12): out of top 10, in buffer.
        ledger_path = tmp_path / "ramp_position_state.json"
        ledger_path.write_text(json.dumps({
            "strategy": "ramp",
            "timestamp": "2026-05-24T16:00:00-04:00",
            "positions": {"SYM12": 100.0},
            "position_open_dates": {"SYM12": "2025-01-01T16:00:00"},
        }))

        v01_plan = _recompute_plan(inputs)
        v11_plan = _recompute_plan(
            inputs, position_ledger_path=ledger_path, variant="v11"
        )

        assert "SYM12" not in v01_plan["target_weights"]
        assert "SYM12" in v11_plan["target_weights"], v11_plan["target_weights"]

    def test_recompute_plan_v11_protects_recent_positions_via_min_hold(self, tmp_path):
        """variant='v11' with position_open_dates < 5 trading days ago protects
        names even when they rank far outside the buffer."""
        from datetime import datetime, timedelta
        from scripts.trading.compare_paper_vs_plan import _recompute_plan

        # 25 symbols, top_n=10. SYM24 ranks 25 -- far outside top_n + buffer (15).
        momentum_scores = {f"SYM{i:02d}": round(0.10 - 0.003 * i, 4) for i in range(25)}
        inputs = self._build_strategy_inputs(momentum_scores, top_n=10)

        # Opened "today - 1 day", well inside the 5-trading-day floor.
        recent_open = (datetime.now() - timedelta(days=1)).isoformat()
        ledger_path = tmp_path / "ramp_position_state.json"
        ledger_path.write_text(json.dumps({
            "strategy": "ramp",
            "timestamp": "2026-05-24T16:00:00-04:00",
            "positions": {"SYM24": 100.0},
            "position_open_dates": {"SYM24": recent_open},
        }))

        v11_plan = _recompute_plan(
            inputs, position_ledger_path=ledger_path, variant="v11"
        )

        assert "SYM24" in v11_plan["target_weights"], v11_plan["target_weights"]
