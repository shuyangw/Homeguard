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
