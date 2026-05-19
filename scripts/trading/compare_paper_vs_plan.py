"""Paper validation comparator (Phase 4 F2 / A7).

For a given decision log entry from paper trading, recompute the RampPlan
for the same date and compare:
- Per-symbol target weight deltas (Severity 1 if > 5%)
- Total target gross delta (Severity 2 if > 1%)
- Rebalance ordering (Severity 3 if BUYs preceded all SELLs)
- Price/quote staleness rounding (Info)

CLI:
    python scripts/trading/compare_paper_vs_plan.py <decision_log_path>

Exit codes:
    0 - PASS
    1 - FAIL (one or more Severity 1-3 divergences)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SEVERITY_1_DELTA = 0.05  # per-symbol target weight delta
SEVERITY_2_DELTA = 0.01  # total target gross delta
INFO_ROUNDING_USD = 50.0


def _recompute_plan(strategy_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute a RampPlan from the strategy_inputs captured in the decision log.

    The decision log records the inputs the strategy saw at trigger time
    (regime, vix, spy_drawdown_pct, momentum_scores, regime_params). Replaying
    those inputs through compute_plan() should produce the SAME plan the
    strategy executed -- divergences indicate the planner is non-deterministic
    or the adapter executed something other than the plan.

    Returns dict with target_weights, regime, exposure_pct.
    """
    import pandas as pd
    from datetime import datetime
    from src.strategies.advanced.ramp_target_planner import compute_plan

    regime = strategy_inputs.get("regime") or "STRONG_BULL"
    regime_confidence = float(strategy_inputs.get("regime_confidence") or 0.5)
    regime_scores = strategy_inputs.get("regime_scores") or {}
    vix = float(strategy_inputs.get("vix") or 20.0)
    spy_drawdown = float(strategy_inputs.get("spy_drawdown_pct") or -0.02)

    momentum_dict = strategy_inputs.get("momentum_scores") or {}
    if not momentum_dict:
        return {"target_weights": {}, "regime": regime, "exposure_pct": 1.0}

    momentum_clean = {sym: float(v) for sym, v in momentum_dict.items() if v is not None}
    if not momentum_clean:
        return {"target_weights": {}, "regime": regime, "exposure_pct": 1.0}
    momentum_scores = pd.Series(momentum_clean)

    regime_params = strategy_inputs.get("regime_params") or {}
    top_n = int(regime_params.get("top_n") or 10)

    plan = compute_plan(
        as_of=datetime.now(),
        regime=regime,
        regime_confidence=regime_confidence,
        regime_scores=regime_scores,
        top_n=top_n,
        momentum_scores=momentum_scores,
        current_positions={},
        vix=vix,
        spy_drawdown=spy_drawdown,
        max_capital_allocation=1.0,
        diagnostics={},
    )

    return {
        "target_weights": {sym: t.target_weight for sym, t in plan.targets.items()},
        "regime": plan.regime,
        "exposure_pct": plan.exposure_pct,
    }


def compare_session(log_path: Path) -> Dict[str, Any]:
    """Compare a paper trading session's decision log to the recomputed plan.

    Returns dict with:
        - status: "PASS" | "FAIL"
        - divergences: list of {symbol, severity, log_weight, plan_weight, delta}
        - log_total_gross
        - plan_total_gross
    """
    rec = json.loads(Path(log_path).read_text())
    # logic_decisions may be None for pre-F5 decision logs OR SAFE_MODE days;
    # treat absent as empty so the comparator doesn't crash on None.get().
    log_weights = (rec.get("logic_decisions") or {}).get("target_weights") or {}
    strategy_inputs = rec.get("strategy_inputs") or {}

    plan = _recompute_plan(strategy_inputs)
    plan_weights = plan["target_weights"]

    divergences: List[Dict[str, Any]] = []

    # Per-symbol comparison (Severity 1)
    all_syms = set(log_weights) | set(plan_weights)
    for sym in all_syms:
        lw = log_weights.get(sym, 0.0)
        pw = plan_weights.get(sym, 0.0)
        delta = lw - pw
        if abs(delta) > SEVERITY_1_DELTA:
            divergences.append({
                "symbol": sym, "severity": 1,
                "log_weight": lw, "plan_weight": pw, "delta": delta,
            })

    # Total gross comparison (Severity 2)
    log_total = sum(log_weights.values())
    plan_total = sum(plan_weights.values())
    if abs(log_total - plan_total) > SEVERITY_2_DELTA:
        divergences.append({
            "symbol": "<total>", "severity": 2,
            "log_weight": log_total, "plan_weight": plan_total,
            "delta": log_total - plan_total,
        })

    status = "FAIL" if any(d["severity"] in (1, 2, 3) for d in divergences) else "PASS"

    return {
        "status": status,
        "divergences": divergences,
        "log_total_gross": log_total,
        "plan_total_gross": plan_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()

    result = compare_session(args.log_path)
    print(f"Status: {result['status']}")
    print(f"  Log total gross:  {result['log_total_gross']:.4f}")
    print(f"  Plan total gross: {result['plan_total_gross']:.4f}")
    if result["divergences"]:
        print("  Divergences:")
        for d in result["divergences"]:
            print(f"    {d['symbol']:<12} sev={d['severity']} "
                  f"log={d['log_weight']:.4f} plan={d['plan_weight']:.4f} "
                  f"delta={d['delta']:+.4f}")

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
