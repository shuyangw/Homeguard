"""Gate 0.3 committed driver for #32 crack (RB-CL, HO-CL) and #33 crush (ZM+ZL-ZS).

Wraps `src.strategies.advanced.spread_processing_strategy.run_processing`
(already gate_convergence-deflated per name, persists returns.csv +
trades.csv + gate.json under output_dir) in RunStatus, appends one registry
row per name, and writes one readiness report per name. #32 REJECT (negative)
pre-Gate-0; #33 crush was MARGINAL (PBO 0.109 clean, Sharpe 0.136 trivial) --
the one Tier-2 candidate that might reach Phase 6.5 improvement design if it
still looks marginal-but-real after honest re-deflation (that decision is
made by strategy-lead reading this driver's output, not by this script).
Section 11 exit diagnostics are reported HONESTLY via
`sp_retest_common.convergence_exit_summary`: `SpreadTrade` does not persist
an `exit_reason` field, so converge/structural-stop/time-stop counts cannot
be reconstructed from trades.csv alone (a code-review finding, tracked
separately -- not fabricated here). Consult
docs/methodology/backtesting.md Sections 1, 2, 3, 4, 9, 11, 12.
"""
from __future__ import annotations

import argparse
from datetime import date

from src.strategies.advanced.spread_processing_strategy import _SPECS, run_processing
from src.utils.run_status import RunStatus
from scripts.backtest_scripts.sp_retest_common import (
    append_registry_row, convergence_exit_summary, write_readiness_report)

_STRATEGY_NAME = "FuturesProcessingSpread"


def main() -> None:
    parser = argparse.ArgumentParser(description="#32/#33 processing-spread retest driver")
    parser.add_argument("--names", nargs="+", default=list(_SPECS.keys()))
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--output-root", default="output/backtests/futures/processing")
    parser.add_argument("--report-root", default="docs/reports/futures")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    for name in args.names:
        out_dir = f"{args.output_root}/{name}"
        with RunStatus(f"sp_retest_processing_{name}", meta={"sleeve": "processing", "name": name}) as st:
            result = run_processing(name, start_d, end_d, out_dir)
            st.heartbeat(note=f"gate computed: oos_sharpe={result.get('oos_sharpe')}")

        catalog_num = "#33 crush" if name == "crush" else "#32 crack"
        run_id = append_registry_row(f"{_STRATEGY_NAME}_{name}", result,
                                      asset_class="futures", data_frequency="daily",
                                      universe_name=name)

        exit_summary = convergence_exit_summary(f"{out_dir}/trades.csv")
        extra = f"## Section 11 exit-logic summary (SpreadTrade exits)\n\n{exit_summary}\n"
        write_readiness_report(f"{_STRATEGY_NAME}_{name}", f"Processing Spread {name} ({catalog_num})",
                                result, f"{args.report_root}/PROCESSING_{name}_READINESS.md",
                                extra_notes=extra, run_id=run_id)
        print(f"[sp_retest_processing:{name}] run_id={run_id} oos_sharpe={result.get('oos_sharpe')} "
              f"dsr={result.get('dsr')} pbo={result.get('pbo')} n_trades={result.get('n_trades')}")


if __name__ == "__main__":
    main()
