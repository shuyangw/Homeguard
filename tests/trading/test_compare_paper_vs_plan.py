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
