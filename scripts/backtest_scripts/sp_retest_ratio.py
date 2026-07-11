"""Gate 0.3 committed driver for #34 gold/silver ratio RV (weak-anchor member).

Wraps `src.strategies.advanced.spread_ratio_strategy.run_ratio` (already
gate_convergence-deflated, persists returns.csv + trades.csv + gate.json
under output_dir) in RunStatus, appends to output/experiments.duckdb, and
writes the readiness report. REJECT pre-Gate-0 (0.269, PBO 0.674; kurtosis
109 -- genuine GC/SI tails, not a data artifact). Section 11 exit
diagnostics are reported HONESTLY via
`sp_retest_common.convergence_exit_summary`: `SpreadTrade` does not persist
an `exit_reason` field, so converge/structural-stop/time-stop counts cannot
be reconstructed from trades.csv alone (a code-review finding, tracked
separately -- not fabricated here). Consult
docs/methodology/backtesting.md Sections 1, 2, 3, 4, 9, 11, 12.
"""
from __future__ import annotations

import argparse
from datetime import date

from src.strategies.advanced.spread_ratio_strategy import run_ratio
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import (
    append_registry_row, convergence_exit_summary, write_readiness_report)

_STRATEGY_NAME = "FuturesGoldSilverRatio"


def main() -> None:
    parser = argparse.ArgumentParser(description="#34 gold/silver ratio retest driver")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-dir", default="output/backtests/futures/ratio_gc_si")
    parser.add_argument("--report", default="docs/reports/futures/RATIO_GC_SI_READINESS.md")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    with RunStatus("sp_retest_ratio", meta={"sleeve": "ratio_gc_si"}) as st:
        result = run_ratio(start_d, end_d, args.output_dir)
        st.heartbeat(note=f"gate computed: oos_sharpe={result.get('oos_sharpe')}")

    run_id = append_registry_row(_STRATEGY_NAME, result, asset_class="futures", data_frequency="daily")

    exit_summary = convergence_exit_summary(f"{args.output_dir}/trades.csv")
    extra = f"## Section 11 exit-logic summary (SpreadTrade exits)\n\n{exit_summary}\n"
    write_readiness_report(_STRATEGY_NAME, "Gold/Silver Ratio RV (#34)", result,
                            args.report, extra_notes=extra, run_id=run_id)
    print(f"[sp_retest_ratio] run_id={run_id} oos_sharpe={result.get('oos_sharpe')} "
          f"dsr={result.get('dsr')} pbo={result.get('pbo')} n_trades={result.get('n_trades')}")


if __name__ == "__main__":
    main()
