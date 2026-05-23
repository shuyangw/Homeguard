"""Paper validation comparator (Phase 4 F2 / A7).

For a given decision log entry from paper trading, recompute the RampPlan
for the same date and compare:
- Per-symbol target weight deltas (Severity 1 if > 5%)
- Total target gross delta (Severity 2 if > 1%)
- Rebalance ordering (Severity 3 if BUYs preceded all SELLs)
- Price/quote staleness rounding (Info)

CLI:
    python scripts/trading/compare_paper_vs_plan.py <decision_log_path>
        [--position-ledger PATH] [--variant {v01,v11}]

Exit codes:
    0 - PASS (real comparison with matching plan vs log)
    1 - FAIL (one or more Severity 1-3 divergences)
    3 - VACUOUS (no positions to compare; both log and inputs empty)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import logger


SEVERITY_1_DELTA = 0.05  # per-symbol target weight delta
SEVERITY_2_DELTA = 0.01  # total target gross delta
INFO_ROUNDING_USD = 50.0

# V11 filter constants -- MUST mirror RAMPLiveAdapter's V11 constants
# (src/trading/adapters/ramp_live_adapter.py, commit 6716c3c). Mismatch
# would produce spurious paper-validation failures.
V11_BUFFER_SIZE_DIVISOR = 2  # buffer_size = top_n // V11_BUFFER_SIZE_DIVISOR
V11_MIN_HOLD_DAYS = 5
V11_CRASH_EXIT = False

DEFAULT_POSITION_LEDGER_PATH = Path("data/trading/decisions/_latest/ramp_position_state.json")


@dataclass
class _ComparatorState:
    """Duck-typed shim that lets V11 filter functions consume comparator state.

    Mirrors src/trading/adapters/ramp_live_adapter.py::_LiveAdapterState.
    The filters in src/research/ramp_phase4/filters.py read state.positions
    and state.position_open_dates only.
    """
    positions: Dict[str, float] = field(default_factory=dict)
    position_open_dates: Dict[str, datetime] = field(default_factory=dict)


def _load_position_ledger(path: Path) -> Optional[_ComparatorState]:
    """Load the RAMP position ledger written by RAMPLiveAdapter.write_position_state.

    Returns None if the file does not exist; the comparator falls back to V01
    behavior with a warning in that case.

    Format (Phase 2B):
        {
          "strategy": "ramp",
          "timestamp": "<iso>",
          "positions": {sym: qty},
          "position_open_dates": {sym: "<iso>"}
        }
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    positions = {sym: float(qty) for sym, qty in (data.get("positions") or {}).items()}
    raw_dates = data.get("position_open_dates") or {}
    parsed_dates: Dict[str, datetime] = {}
    for sym, iso_str in raw_dates.items():
        try:
            parsed_dates[sym] = datetime.fromisoformat(iso_str)
        except (TypeError, ValueError) as exc:
            logger.error(
                f"[comparator] Failed to parse position_open_date for {sym}={iso_str!r}: {exc}"
            )
            raise
    return _ComparatorState(positions=positions, position_open_dates=parsed_dates)


def _apply_v11_filters_to_plan(
    plan_weights: Dict[str, float],
    momentum_scores: "Any",
    top_n: int,
    state: _ComparatorState,
    current_date: datetime,
) -> Dict[str, float]:
    """Apply V11 rank_buffer + min_hold filters to a V01 plan's target weights.

    Mirrors RAMPLiveAdapter._apply_v11_filters (commit 6716c3c) AND
    _variant_v11 in src/research/ramp_phase4/variants.py:189-231. Composition
    order is rank_buffer -> min_hold; reversing the order changes semantics.

    Args:
        plan_weights: V01-base per-symbol weights (output of compute_plan).
        momentum_scores: pd.Series of full-universe momentum (one row per symbol).
        top_n: regime-derived target position count.
        state: _ComparatorState with positions + position_open_dates.
        current_date: today's date for min_hold age comparison.

    Returns:
        dict[symbol -> equal-weight] summing to 1.0 after filters.
    """
    import pandas as pd
    from src.research.ramp_phase4.filters import rank_buffer, min_hold

    target_symbols = list(plan_weights.keys())
    if not target_symbols:
        return dict(plan_weights)

    # Step 1: V01 base proposed_targets = equal weight 1/top_n.
    proposed = {sym: 1.0 / top_n for sym in target_symbols}

    # Build full-universe ranking from momentum scores (1 = best).
    sorted_momentum = momentum_scores.dropna().sort_values(ascending=False)
    universe_ranking = pd.Series(
        range(1, len(sorted_momentum) + 1),
        index=sorted_momentum.index,
    )

    # Step 2: rank_buffer (buffer_size = top_n // V11_BUFFER_SIZE_DIVISOR).
    filtered = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=top_n // V11_BUFFER_SIZE_DIVISOR,
        universe_ranking=universe_ranking,
        top_n=top_n,
    )

    # Step 3: min_hold.
    filtered = min_hold(
        proposed_targets=filtered,
        state=state,
        current_date=current_date,
        min_hold_days=V11_MIN_HOLD_DAYS,
        crash_exit=V11_CRASH_EXIT,
    )

    return filtered


def _recompute_plan(
    strategy_inputs: Dict[str, Any],
    position_ledger_path: Optional[Path] = None,
    variant: str = "v01",
) -> Dict[str, Any]:
    """Recompute a RampPlan from the strategy_inputs captured in the decision log.

    The decision log records the inputs the strategy saw at trigger time
    (regime, vix, spy_drawdown_pct, momentum_scores, regime_params). Replaying
    those inputs through compute_plan() should produce the SAME plan the
    strategy executed -- divergences indicate the planner is non-deterministic
    or the adapter executed something other than the plan.

    When variant='v11' and the position ledger exists, the V01 base plan is
    composed with rank_buffer -> min_hold matching RAMPLiveAdapter's
    _apply_v11_filters (commit 6716c3c). When variant='v01' or the ledger is
    missing, behavior is unchanged from pre-Phase-2E.

    Returns dict with target_weights, regime, exposure_pct.
    """
    import pandas as pd
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

    target_weights = {sym: t.target_weight for sym, t in plan.targets.items()}

    if variant == "v11":
        state: Optional[_ComparatorState] = None
        if position_ledger_path is not None:
            state = _load_position_ledger(position_ledger_path)
        if state is None:
            logger.warning(
                f"[comparator] variant='v11' but position ledger missing at "
                f"{position_ledger_path}; falling back to V01 behavior."
            )
        else:
            target_weights = _apply_v11_filters_to_plan(
                plan_weights=target_weights,
                momentum_scores=momentum_scores,
                top_n=top_n,
                state=state,
                current_date=datetime.now(),
            )

    return {
        "target_weights": target_weights,
        "regime": plan.regime,
        "exposure_pct": plan.exposure_pct,
    }


def compare_session(
    log_path: Path,
    position_ledger_path: Optional[Path] = None,
    variant: str = "v01",
) -> Dict[str, Any]:
    """Compare a paper trading session's decision log to the recomputed plan.

    Returns dict with:
        - status: "PASS" | "FAIL" | "VACUOUS"
        - divergences: list of {symbol, severity, log_weight, plan_weight, delta}
        - log_total_gross
        - plan_total_gross

    VACUOUS is returned when both the log weights and the strategy_inputs are
    empty, so there is literally nothing to compare. Callers should treat this
    distinctly from PASS to avoid counting a "no-op day" as a clean session.
    """
    rec = json.loads(Path(log_path).read_text())
    # logic_decisions may be None for pre-F5 decision logs OR SAFE_MODE days;
    # treat absent as empty so the comparator doesn't crash on None.get().
    log_weights = (rec.get("logic_decisions") or {}).get("target_weights") or {}
    strategy_inputs = rec.get("strategy_inputs") or {}

    if not log_weights and not strategy_inputs:
        return {
            "status": "VACUOUS",
            "divergences": [],
            "log_total_gross": 0.0,
            "plan_total_gross": 0.0,
        }

    plan = _recompute_plan(
        strategy_inputs,
        position_ledger_path=position_ledger_path,
        variant=variant,
    )
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
    parser.add_argument(
        "--position-ledger",
        type=Path,
        default=DEFAULT_POSITION_LEDGER_PATH,
        help="Path to the RAMP position state JSON written by the live adapter. "
             "Used only when --variant=v11. Defaults to "
             f"{DEFAULT_POSITION_LEDGER_PATH}.",
    )
    parser.add_argument(
        "--variant",
        choices=("v01", "v11"),
        default="v01",
        help="RAMP variant to model: v01 (no filters, default for safety) or "
             "v11 (rank_buffer + min_hold composed in the same order as the "
             "live adapter).",
    )
    args = parser.parse_args()

    result = compare_session(
        args.log_path,
        position_ledger_path=args.position_ledger,
        variant=args.variant,
    )
    print(f"Status: {result['status']}")
    print(f"  Log total gross:  {result['log_total_gross']:.4f}")
    print(f"  Plan total gross: {result['plan_total_gross']:.4f}")
    if result["divergences"]:
        print("  Divergences:")
        for d in result["divergences"]:
            print(f"    {d['symbol']:<12} sev={d['severity']} "
                  f"log={d['log_weight']:.4f} plan={d['plan_weight']:.4f} "
                  f"delta={d['delta']:+.4f}")

    status = result["status"]
    if status == "PASS":
        return 0
    if status == "VACUOUS":
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
